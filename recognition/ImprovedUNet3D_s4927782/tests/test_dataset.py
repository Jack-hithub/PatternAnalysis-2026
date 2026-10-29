"""Synthetic NIfTI geometry, patient separation and deferred-TEST checks."""
from __future__ import annotations
import copy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

import dataset


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.raw = self.root / "raw"
        (self.raw / "semantic_MRs").mkdir(parents=True)
        (self.raw / "semantic_labels_only").mkdir()
        shape = (16, 18, 12)
        x, y, z = np.indices(shape)
        self.image = (x + 2 * y + 3 * z + 1).astype(np.float32)
        self.image[0] = 0
        self.mask = ((x // 3 + y // 4 + z // 3) % 6).astype(np.uint8)
        self.affine = np.diag([-1.2, -1.4, 2.0, 1.0])
        self.affine[:3, 3] = [30, 40, 10]
        for patient in ["B006", "J026", "M015"]:
            nib.save(nib.Nifti1Image(self.image, self.affine), self.raw / f"semantic_MRs/{patient}_Week0_LFOV.nii.gz")
            nib.save(nib.Nifti1Image(self.mask, self.affine), self.raw / f"semantic_labels_only/{patient}_Week0_SEMANTIC.nii.gz")
        self.splits = {"patient_splits": {"train": ["B006"], "validation": ["J026"], "test": ["M015"]},
                       "scan_counts": {"train": 1, "validation": 1, "test": 1},
                       "patient_counts": {"train": 1, "validation": 1, "test": 1}}
        self.split_path = self.root / "split.json"
        self.split_path.write_text(json.dumps(self.splits))
        self.config = {"target_shape_dhw": [8, 12, 12], "evaluation_shape_dhw": [4, 8, 8],
                       "patch_shape_dhw": [8, 8, 8], "seed": 3711, "epochs": 30,
                       "train_slices_per_scan": 4, "train_patches_per_scan": 2,
                       "foreground_probability": .5}
        self.cache = self.root / "cache"

    def tearDown(self):
        self.temporary.cleanup()

    def prepare(self):
        return dataset.prepare_cache(self.raw, self.split_path, self.config, self.cache, workers=1)

    def selection(self, manifest):
        models = {}
        for name in ["unet2d", "improved3d"]:
            p = self.root / f"{name}.synthetic-checkpoint"
            p.write_bytes(name.encode())
            models[name] = {"completed_epochs": 30, "checkpoint_path": str(p),
                            "checkpoint_sha256": sha256(p.read_bytes()).hexdigest(), "selected_epoch": 30}
        value = {"status": "frozen", "manifest_sha256": sha256(manifest.read_bytes()).hexdigest(),
                 "config_sha256": dataset.config_digest(self.config), "models": models}
        path = self.root / "selection.json"
        path.write_text(json.dumps(value))
        return path

    def rewrite_synthetic_case(self, manifest, record, mutate):
        """Rebind temporary fixture hashes to exercise checks inside the NPZ."""
        path = self.cache / record["cache_path"]
        with np.load(path, allow_pickle=False) as stored:
            payload = {name: stored[name].copy() for name in stored.files}
        value = json.loads(manifest.read_text())
        selected = next(r for r in value["records"] if r["patient"] == record["patient"])
        mutate(payload, selected)
        with path.open("wb") as stream:
            np.savez_compressed(stream, **payload)
        selected["cache_sha256"] = sha256(path.read_bytes()).hexdigest()
        manifest.write_text(json.dumps(value))
        receipt_path = self.cache / "trainval_receipt.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["records"] = [selected if r["patient"] == selected["patient"] else r
                              for r in receipt["records"]]
        receipt["manifest_sha256"] = sha256(manifest.read_bytes()).hexdigest()
        receipt_path.write_text(json.dumps(receipt))
        return selected

    def test_patient_split_rejects_overlap_and_missing_patients(self):
        bad = copy.deepcopy(self.splits["patient_splits"])
        bad["test"] = ["B006"]
        with self.assertRaises(ValueError):
            dataset.validate_patient_splits(bad)
        with self.assertRaises(ValueError):
            dataset.validate_patient_splits(self.splits["patient_splits"], ["B006", "J026"])

    def test_filename_pairing_and_duplicate_rejection(self):
        records = dataset.discover_pairs(self.raw, self.splits)
        self.assertEqual([(x["patient"], x["split"]) for x in records],
                         [("B006", "train"), ("J026", "validation"), ("M015", "test")])
        extra = self.raw / "semantic_MRs/duplicate"
        extra.mkdir()
        nib.save(nib.Nifti1Image(self.image, self.affine), extra / "B006_Week0_LFOV.nii.gz")
        with self.assertRaises(ValueError):
            dataset.discover_pairs(self.raw, self.splits)

    def test_missing_pair_rejected_before_output_created(self):
        (self.raw / "semantic_labels_only/M015_Week0_SEMANTIC.nii.gz").unlink()
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse(self.cache.exists())

    def test_image_only_ras_geometry_preserves_physical_centre(self):
        p = self.raw / "semantic_MRs/B006_Week0_LFOV.nii.gz"
        image, meta = dataset.preprocess_image(p, self.config["target_shape_dhw"])
        self.assertEqual(image.shape, (8, 12, 12))
        self.assertEqual(image.dtype, np.float32)
        self.assertTrue(np.isfinite(image).all())
        self.assertEqual(nib.aff2axcodes(np.asarray(meta["resampled_affine_xyz"])), ("R", "A", "S"))
        original = nib.load(p)
        centre_old = original.affine @ np.r_[(np.asarray(original.shape) - 1) / 2, 1]
        centre_new = np.asarray(meta["resampled_affine_xyz"]) @ np.r_[(np.asarray(image.shape[::-1]) - 1) / 2, 1]
        np.testing.assert_allclose(centre_old, centre_new, rtol=0, atol=1e-6)
        self.assertEqual(meta["normalization"]["nonzero_count"], int(np.count_nonzero(self.image)))

    def test_support_preserves_valid_raw_mean_intensity_and_excludes_raw_zero(self):
        values = np.broadcast_to(np.array([0, 2, 4, 6], np.float32)[:, None, None], (4, 2, 2)).copy()
        path = self.root / "mean_intensity.nii.gz"
        nib.save(nib.Nifti1Image(values, np.eye(4)), path)
        image, support, _ = dataset.preprocess_image(path, (2, 2, 4), return_support=True)
        self.assertEqual(support.dtype, np.bool_)
        self.assertTrue(np.all(image[:, :, 2] == 0))
        self.assertTrue(np.all(support[:, :, 2]))
        self.assertTrue(np.all(image[:, :, 0] == 0))
        self.assertFalse(support[:, :, 0].any())
        np.testing.assert_array_equal(support, (values != 0).transpose(2, 1, 0))

    def test_support_uses_raw_nearest_neighbour_geometry_in_ras_dhw(self):
        raw = self.image.copy()
        raw[3:7, 4:9, 2:5] = 0
        raw[8:11, 10:15, 7:10] = 0
        path = self.root / "support_geometry.nii.gz"
        nib.save(nib.Nifti1Image(raw, self.affine), path)
        target = (7, 11, 9)
        _, support, meta = dataset.preprocess_image(path, target, return_support=True)
        # Centre-aware nearest sampling selects floor((j + 0.5) * old/new).
        ix, iy, iz = [np.floor((np.arange(new) + .5) * old / new).astype(int)
                      for old, new in zip(raw.shape, target[::-1])]
        expected = (raw != 0)[np.ix_(raw.shape[0] - 1 - ix,
                                    raw.shape[1] - 1 - iy, iz)].transpose(2, 1, 0)
        np.testing.assert_array_equal(support, expected)
        self.assertTrue(support.flags.c_contiguous)
        self.assertEqual(meta["support"], {"policy": dataset.SUPPORT_POLICY,
                                         "sha256": sha256(support.tobytes()).hexdigest(),
                                         "voxel_count": int(support.sum())})

    def test_support_api_keeps_legacy_normalized_image_bytes_unchanged(self):
        path = self.raw / "semantic_MRs/B006_Week0_LFOV.nii.gz"
        legacy = dataset.preprocess_image(path, self.config["target_shape_dhw"])
        image, support, metadata = dataset.preprocess_image(path, self.config["target_shape_dhw"], return_support=True)
        self.assertEqual(len(legacy), 2)
        np.testing.assert_array_equal(image, legacy[0])
        self.assertEqual(metadata, legacy[1])
        # Captured from the original preprocessing before raw-support export.
        self.assertEqual(sha256(image.tobytes()).hexdigest(),
                         "d1e7f76774ae71ebb1e022dbdaa3d2cabd3540dac8ce74394277eb0e7a7ad69c")
        self.assertEqual(support.shape, image.shape)

    def test_physical_label_identity_and_translation(self):
        labels = (np.arange(8 ** 3).reshape(8, 8, 8) % 6).astype(np.uint8)
        result = dataset.resample_label(labels, np.eye(4), np.eye(4), labels.shape)
        np.testing.assert_array_equal(result, labels)
        moved = np.eye(4)
        moved[0, 3] = 1
        result = dataset.resample_label(labels, np.eye(4), moved, labels.shape)
        np.testing.assert_array_equal(result[:, :, :-1], labels[:, :, 1:])
        self.assertTrue(np.all(result[:, :, -1] == 0))

    def test_native_restore_preserves_edge_voxels_and_true_outside_is_zero(self):
        coarse_affine, _ = dataset._grid(np.eye(4), (8, 8, 8), (4, 4, 4))
        coarse = np.full((4, 4, 4), 5, dtype=np.uint8)
        restored = dataset.resample_label(coarse, coarse_affine, np.eye(4), (8, 8, 8))
        np.testing.assert_array_equal(restored, np.full((8, 8, 8), 5, dtype=np.uint8))
        outside = np.eye(4)
        outside[0, 3] = 20
        self.assertFalse(dataset.resample_label(coarse, coarse_affine, outside, (8, 8, 8)).any())

    def test_trainval_does_not_open_any_test_image_or_label(self):
        original_load = dataset.nib.load
        def guarded(path, *args, **kwargs):
            if "M015" in str(path):
                raise AssertionError("TRAIN/VAL preparation opened TEST data")
            return original_load(path, *args, **kwargs)
        with patch.object(dataset.nib, "load", side_effect=guarded):
            manifest = self.prepare()
        value = json.loads(manifest.read_text())
        test = dataset.records_for_split(manifest, "test")[0]
        self.assertNotIn("source_sha256", test)
        self.assertNotIn("class_voxel_counts", test)
        self.assertFalse((self.cache / test["cache_path"]).exists())
        self.assertEqual(value["records"][0]["split"], "train")
        with self.assertRaises((ValueError, FileNotFoundError)):
            dataset.load_case(manifest, test)

    def test_cache_pair_and_image_only_preprocessing_agree(self):
        manifest = self.prepare()
        record = dataset.records_for_split(manifest, "train")[0]
        case = dataset.load_case(manifest, record)
        image, meta = dataset.preprocess_image(self.raw / record["source_image"], self.config["target_shape_dhw"])
        np.testing.assert_array_equal(case["image"], image)
        np.testing.assert_allclose(case["affine_xyz"], meta["resampled_affine_xyz"])
        self.assertEqual(case["mask_eval"].shape, (4, 8, 8))
        self.assertEqual(set(np.unique(case["mask"])), set(range(6)))
        self.assertEqual(sum(record["class_voxel_counts"]), 8 * 12 * 12)

    def test_new_cache_contains_explicit_boolean_support_and_integrity_metadata(self):
        manifest = self.prepare()
        record = dataset.records_for_split(manifest, "train")[0]
        case = dataset.load_case(manifest, record)
        image, support, metadata = dataset.preprocess_image(
            self.raw / record["source_image"], self.config["target_shape_dhw"], return_support=True)
        np.testing.assert_array_equal(case["image"], image)
        np.testing.assert_array_equal(case["support"], support)
        self.assertEqual(case["support"].dtype, np.bool_)
        self.assertEqual(case["metadata"]["support"], metadata["support"])
        self.assertEqual(record["support_policy"], dataset.SUPPORT_POLICY)
        self.assertEqual(record["support_sha256"], sha256(support.tobytes()).hexdigest())
        self.assertEqual(record["support_voxel_count"], int(support.sum()))
        with np.load(self.cache / record["cache_path"], allow_pickle=False) as stored:
            self.assertEqual(stored["support"].dtype, np.bool_)

    def test_legacy_cache_without_support_stays_readable_with_support_none(self):
        manifest = self.prepare()
        record = dataset.records_for_split(manifest, "train")[0]
        expected_image = dataset.load_case(manifest, record)["image"].copy()
        def make_legacy(payload, selected):
            payload.pop("support")
            metadata = json.loads(str(payload["metadata_json"].item()))
            metadata.pop("support")
            payload["metadata_json"] = np.asarray(json.dumps(metadata))
            for key in ("support_policy", "support_sha256", "support_voxel_count"):
                selected.pop(key)
        record = self.rewrite_synthetic_case(manifest, record, make_legacy)
        before = {p: p.read_bytes() for p in [manifest, self.cache / record["cache_path"],
                                             self.cache / "trainval_receipt.json"]}
        case = dataset.load_case(manifest, record)
        self.assertIsNone(case["support"])
        np.testing.assert_array_equal(case["image"], expected_image)
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_new_cache_support_shape_dtype_and_hash_tampering_are_rejected(self):
        manifest = self.prepare()
        record = dataset.records_for_split(manifest, "train")[0]
        original = dataset.load_case(manifest, record)["support"].copy()
        variants = {"shape": original[:-1], "dtype": original.astype(np.uint8),
                    "hash": np.logical_not(original)}
        for name, bad_support in variants.items():
            with self.subTest(name=name):
                def mutate(payload, selected):
                    payload["support"] = bad_support
                record = self.rewrite_synthetic_case(manifest, record, mutate)
                with self.assertRaisesRegex(ValueError, "support"):
                    dataset.load_case(manifest, record)

    def test_new_cache_support_metadata_tampering_or_missing_map_are_rejected(self):
        manifest = self.prepare()
        record = dataset.records_for_split(manifest, "train")[0]
        def alter_count(payload, selected):
            metadata = json.loads(str(payload["metadata_json"].item()))
            metadata["support"]["voxel_count"] += 1
            payload["metadata_json"] = np.asarray(json.dumps(metadata))
        record = self.rewrite_synthetic_case(manifest, record, alter_count)
        with self.assertRaisesRegex(ValueError, "support"):
            dataset.load_case(manifest, record)
        def remove_map(payload, selected):
            payload.pop("support")
        record = self.rewrite_synthetic_case(manifest, record, remove_map)
        with self.assertRaisesRegex(ValueError, "support"):
            dataset.load_case(manifest, record)

    def test_existing_cache_never_overwritten(self):
        manifest = self.prepare()
        before = manifest.read_bytes()
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(before, manifest.read_bytes())

    def test_test_requires_actual_two_checkpoint_identities(self):
        manifest = self.prepare()
        with self.assertRaises(ValueError):
            dataset.prepare_cache(self.raw, self.split_path, self.config, self.cache, stage="test", workers=1)
        freeze = self.selection(manifest)
        data = json.loads(freeze.read_text())
        data["models"].pop("unet2d")
        freeze.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            dataset.prepare_cache(self.raw, self.split_path, self.config, self.cache,
                                  stage="test", workers=1, selection_freeze=freeze)
        self.assertFalse((self.cache / "test_completion_receipt.json").exists())

    def test_test_completion_keeps_manifest_bytes_immutable(self):
        manifest = self.prepare()
        before = manifest.read_bytes()
        freeze = self.selection(manifest)
        dataset.prepare_cache(self.raw, self.split_path, self.config, self.cache,
                              stage="test", workers=1, selection_freeze=freeze)
        self.assertEqual(manifest.read_bytes(), before)
        record = dataset.records_for_split(manifest, "test")[0]
        self.assertEqual(dataset.load_case(manifest, record)["mask"].shape, (8, 12, 12))
        with self.assertRaises(FileExistsError):
            dataset.prepare_cache(self.raw, self.split_path, self.config, self.cache,
                                  stage="test", workers=1, selection_freeze=freeze)

    def test_corrupt_cache_and_changed_record_rejected(self):
        manifest = self.prepare()
        record = dataset.records_for_split(manifest, "validation")[0]
        modified = dict(record, split="train")
        with self.assertRaises(ValueError):
            dataset.load_case(manifest, modified)
        with (self.cache / record["cache_path"]).open("ab") as stream:
            stream.write(b"not-the-saved-cache")
        with self.assertRaises(ValueError):
            dataset.load_case(manifest, record)

    def test_sampling_determinism_and_train_only_membership(self):
        manifest = self.prepare()
        for model in ["unet2d", "improved3d"]:
            samples = dataset.SampledTrainingDataset(manifest, model, self.config)
            a, b = samples[0], samples[0]
            self.assertTrue(torch.equal(a["image"], b["image"]))
            self.assertTrue(torch.equal(a["mask"], b["mask"]))
            self.assertEqual(samples.records[0]["patient"], "B006")
            self.assertEqual(samples.records[0]["split"], "train")
            shape = (1, 12, 12) if model == "unet2d" else (1, 8, 8, 8)
            self.assertEqual(tuple(a["image"].shape), shape)
            self.assertEqual(a["mask"].dtype, torch.int64)
            samples.set_epoch(2)
            c = samples[0]
            self.assertFalse(torch.equal(a["center_dhw"], c["center_dhw"]))

    def test_foreground_class_absence_falls_back_without_replacing_class(self):
        class FixedRandom:
            def __init__(self): self.calls = 0
            def random(self): return 0
            def integers(self, *args):
                self.calls += 1
                return 5 if self.calls == 1 else 0
        centre, requested, chosen, fallback = dataset.sample_center(np.zeros((8, 8, 8), np.uint8), FixedRandom(), .5)
        self.assertEqual((centre, requested, chosen, fallback), ((0, 0, 0), True, 5, True))

    def test_weights_use_only_full_train_counts(self):
        manifest = self.prepare()
        weights, counts = dataset.training_class_weights(manifest)
        expected_counts = json.loads(manifest.read_text())["records"][0]["class_voxel_counts"]
        self.assertEqual(counts, expected_counts)
        w = 1 / np.sqrt(np.asarray(counts) / sum(counts))
        w /= w.mean(); w = np.clip(w, .05, 10); w /= w.mean()
        np.testing.assert_allclose(weights.numpy(), w, rtol=1e-6)

    def test_fractional_or_misaligned_mask_fails_preparation(self):
        path = self.raw / "semantic_labels_only/B006_Week0_SEMANTIC.nii.gz"
        invalid = self.mask.astype(np.float32)
        invalid[0, 0, 0] = .5
        nib.save(nib.Nifti1Image(invalid, self.affine), path)
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertEqual(json.loads((self.cache / "trainval_receipt.json").read_text())["status"], "failed")
        self.assertFalse((self.cache / "manifest.json").exists())

    def test_header_spacing_disagreement_is_rejected(self):
        path = self.raw / "semantic_labels_only/B006_Week0_SEMANTIC.nii.gz"
        volume = nib.Nifti1Image(self.mask, self.affine)
        volume.header.set_zooms((2.4, 1.4, 2.0))
        nib.save(volume, path)
        with self.assertRaises(ValueError):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
