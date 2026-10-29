"""Synthetic CPU integration fixtures; no real patient data or formal training."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dataset as data
import predict
import train
from utils import load_json, save_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]


class Pointwise(torch.nn.Module):
    def forward(self, x):
        logits = torch.zeros((len(x), 6, *x.shape[2:]), device=x.device)
        logits[:, 1] = 2 - x[:, 0]
        logits[:, 5] = x[:, 0]
        return {"logits": logits, "aux_logits": []}


class WindowDisagreement(torch.nn.Module):
    """Windows disagree: probability means and logit means choose differently."""
    def forward(self, x):
        first = x[0, 0, 0, 0, 0] < 1
        logits = torch.full((1, 6, *x.shape[2:]), -100., device=x.device)
        if first:
            logits[:, 1], logits[:, 5] = 100, 99.9
        else:
            logits[:, 1], logits[:, 5] = 0, 2
        return {"logits": logits, "aux_logits": []}


class InvalidOutput(torch.nn.Module):
    def forward(self, x):
        return torch.full((len(x), 6, *x.shape[2:]), float("nan"))


class TinyTrainingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 6, 1)

    def forward(self, x):
        return {"logits": self.conv(x), "aux_logits": []}


class TinySamples(torch.utils.data.Dataset):
    def __init__(self, *args):
        self.epochs = []

    def __len__(self):
        return 2

    def set_epoch(self, epoch):
        self.epochs.append(epoch)

    def __getitem__(self, index):
        return {"image": torch.ones(1, 4, 4), "mask": torch.ones(4, 4, dtype=torch.long),
                "foreground_requested": index == 0, "selected_class": 1 if index == 0 else -1,
                "foreground_missing_fallback": False}


def small_config():
    config = load_json(ROOT / "config.json")
    config.update(patch_shape_dhw=[2, 2, 2], eval_slice_batch_size=2,
                  mask_outside_nonzero_support=False, amp=False)
    return config


def manifest_identity_fixture(directory, config):
    manifest = directory / "manifest.json"
    save_json(manifest, {"synthetic_fixture": True})
    save_json(directory / "trainval_receipt.json", {
        "status": "completed", "manifest_sha256": sha256_file(manifest),
        "config_sha256": data.config_digest(config)})
    return manifest


def make_run(directory, name, manifest, config):
    directory.mkdir()
    values = {1: .1, 5: .5, 10: .5, 15: .3, 20: .4, 25: .2, 30: .45}
    history = [{"epoch": e, "validation_prostate_dice": values.get(e)} for e in range(1, 31)]
    identity = train.checkpoint_identity(manifest, config)
    payload = {"schema_version": 1, "model": name, "config": config, "identity": identity,
               "state_dict": {"synthetic": torch.tensor([1.])}, "epoch": 5,
               "best_epoch": 5, "best_score": .5, "history": history[:5]}
    train.save_checkpoint(directory / "best.pt", payload)
    save_json(directory / "history.json", history)
    save_json(directory / "run.json", {"status": "completed", "completed_epochs": 30,
              "best_checkpoint_sha256": sha256_file(directory / "best.pt")})
    return directory


class InferenceTests(unittest.TestCase):
    def test_nondivisible_window_edges_are_covered(self):
        self.assertEqual(train.window_starts(7, 4, .5), [0, 2, 3])
        config = small_config()
        config["patch_shape_dhw"] = [2, 3, 3]
        image = np.arange(4*5*7, dtype=np.float32).reshape(4, 5, 7) / 10
        expected = np.where(image > 1, 5, 1).astype(np.uint8)
        model = Pointwise()
        actual = train.infer_volume(model, image, "improved3d", config, torch.device("cpu"))
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(model.training)

    def test_overlap_averages_softmax_probabilities(self):
        config = small_config()
        image = np.broadcast_to(np.arange(3, dtype=np.float32)[:, None, None], (3, 2, 2)).copy()
        actual = train.infer_volume(WindowDisagreement(), image, "improved3d", config, torch.device("cpu"))
        self.assertTrue((actual[0] == 1).all())
        self.assertTrue((actual[1:] == 5).all())

    def test_2d_assembly_restores_eval_state_and_explicit_support(self):
        config = small_config()
        config["mask_outside_nonzero_support"] = True
        image = np.array([0, .5, 1, 1.5, 2], np.float32)[:, None, None] * np.ones((5, 3, 4), np.float32)
        support = np.ones(image.shape, dtype=bool)
        support[0] = False
        model = Pointwise().eval()
        actual = train.infer_volume(model, image, "unet2d", config, torch.device("cpu"), support_dhw=support)
        self.assertFalse(model.training)
        self.assertEqual(actual[:, 0, 0].tolist(), [0, 1, 1, 5, 5])

    def test_raw_support_preserves_normalized_zero_foreground_in_2d_and_3d(self):
        config = small_config()
        config["mask_outside_nonzero_support"] = True
        # Constant-intensity raw foreground and raw background can both become
        # normalized zero. Their explicit pre-normalization support differs.
        image = np.zeros((3, 3, 4), dtype=np.float32)
        support = np.ones(image.shape, dtype=bool)
        support[0] = False
        expected = support.astype(np.uint8)
        for model_name in train.MODELS:
            with self.subTest(model=model_name):
                actual = train.infer_volume(Pointwise(), image, model_name, config, "cpu", support_dhw=support)
                np.testing.assert_array_equal(actual, expected)

    def test_support_is_required_when_masking_is_enabled(self):
        config = small_config()
        config["mask_outside_nonzero_support"] = True
        for model_name in train.MODELS:
            with self.subTest(model=model_name):
                model = Pointwise()
                with self.assertRaisesRegex(ValueError, "Raw-image support.*rebuild legacy caches"):
                    train.infer_volume(model, np.ones((2, 2, 2), np.float32), model_name, config, "cpu")
                self.assertTrue(model.training)

    def test_supplied_support_must_be_boolean_and_match_even_without_masking(self):
        image = np.ones((2, 2, 2), dtype=np.float32)
        invalid = [np.ones(image.shape, dtype=np.uint8), np.ones(image.shape, dtype=np.float32),
                   np.ones((2, 2), dtype=bool), np.ones((2, 2, 3), dtype=bool)]
        for masking in (False, True):
            config = small_config()
            config["mask_outside_nonzero_support"] = masking
            for support in invalid:
                with self.subTest(masking=masking, dtype=support.dtype, shape=support.shape):
                    with self.assertRaisesRegex(ValueError, "boolean support.*matching"):
                        train.infer_volume(Pointwise(), image, "unet2d", config, "cpu", support_dhw=support)

    def test_evaluation_passes_cached_support_and_rejects_legacy_without_support(self):
        config = small_config()
        config["mask_outside_nonzero_support"] = True
        support = np.ones((2, 2, 2), dtype=bool)
        support[0] = False
        case = {"image": np.zeros(support.shape, np.float32), "support": support,
                "mask": support.astype(np.uint8), "spacing_dhw": np.ones(3)}
        record = {"patient": "synthetic", "visit": "v1"}
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "manifest.json"
            save_json(manifest, {"synthetic_fixture": True})
            with patch.object(data, "records_for_split", return_value=[record]), \
                 patch.object(data, "load_case", return_value=case), \
                 patch.object(train, "infer_volume", wraps=train.infer_volume) as infer:
                result = train.evaluate(Pointwise(), manifest, "validation", "unet2d", config, "cpu")
                self.assertEqual(infer.call_count, 2)
                self.assertTrue(all(call.kwargs["support_dhw"] is support for call in infer.call_args_list))
                self.assertEqual(result["support_policy"], data.SUPPORT_POLICY)
                self.assertTrue(result["support_masking_applied"])
                case["support"] = None
                with self.assertRaisesRegex(ValueError, "rebuild legacy caches"):
                    train.evaluate(Pointwise(), manifest, "validation", "unet2d", config, "cpu")

    def test_nonfinite_images_and_logits_fail_with_state_restored(self):
        config = small_config()
        model = InvalidOutput()
        with self.assertRaisesRegex(ValueError, "Invalid 3D"):
            train.infer_volume(model, np.ones((2, 2, 2)), "improved3d", config, torch.device("cpu"))
        self.assertTrue(model.training)
        with self.assertRaisesRegex(ValueError, "finite image"):
            train.infer_volume(model, np.full((2, 2, 2), np.nan), "improved3d", config, torch.device("cpu"))

    def test_bad_window_geometry_rejected(self):
        for args in [(2, 3, .5), (7, 4, 1), (7, 4, -.1)]:
            with self.assertRaises(ValueError):
                train.window_starts(*args)

    def test_selection_strict_max_earliest_tie(self):
        history = [{"epoch": 1, "validation_prostate_dice": .2},
                   {"epoch": 2, "validation_prostate_dice": None},
                   {"epoch": 5, "validation_prostate_dice": .7},
                   {"epoch": 10, "validation_prostate_dice": .7},
                   {"epoch": 15, "validation_prostate_dice": .6}]
        self.assertEqual(train.select_best(history), (5, .7))
        with self.assertRaises(ValueError):
            train.select_best([{"epoch": 1, "validation_prostate_dice": float("nan")}])

    def test_existing_outputs_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run"
            train.fresh_dir(path)
            with self.assertRaises(ValueError):
                train.fresh_dir(path)

    def test_native_export_preserves_original_oblique_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = folder / "input.nii.gz"
            affine = np.array([[0., -2., 0., 11.], [1.2, 0., .1, -4.], [0., 0., 3., 20.], [0., 0., 0., 1.]])
            image = nib.Nifti1Image(np.arange(5*6*7, dtype=np.float32).reshape(5, 6, 7)+1, affine)
            image.set_sform(affine, code=2)
            image.set_qform(affine, code=0)
            nib.save(image, source)
            grid, metadata = data.preprocess_image(source, (8, 8, 8))
            prediction = np.full(grid.shape, 5, dtype=np.uint8)
            output = folder / "prediction.nii.gz"
            receipt = predict.save_native_prediction(prediction, metadata, source, output)
            loaded = nib.load(output)
            self.assertEqual(loaded.shape, (5, 6, 7))
            np.testing.assert_allclose(loaded.affine, nib.load(source).affine, atol=1e-5)
            self.assertEqual(receipt["sform_code"], 2)
            self.assertEqual(receipt["qform_code"], 0)
            self.assertTrue((np.asarray(loaded.dataobj) == 5).all())

    def test_image_only_predict_pipeline_writes_native_labels_and_hash_receipt(self):
        config = small_config()
        config["target_shape_dhw"] = [7, 6, 5]
        config["mask_outside_nonzero_support"] = True
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            image_path = folder / "image.nii.gz"
            raw = np.full((5, 6, 7), 2, dtype=np.float32)
            raw[0], raw[1], raw[3] = 0, 1, 3
            nib.save(nib.Nifti1Image(raw, np.diag([1.2, 2., 3., 1.])), image_path)
            checkpoint_path = folder / "best.pt"
            train.save_checkpoint(checkpoint_path, {
                "schema_version": 1, "model": "unet2d", "config": config,
                "identity": {"config_sha256": data.config_digest(config),
                             "source_sha256": train.source_identity()},
                "state_dict": {}, "epoch": 1, "best_epoch": 1, "best_score": .2,
                "history": [{"epoch": 1, "validation_prostate_dice": .2}]})
            output = folder / "prediction"
            with patch.object(predict, "create_model", return_value=Pointwise()):
                receipt = predict.predict_image(image_path, checkpoint_path, config, output, torch.device("cpu"))
            self.assertEqual(receipt["status"], "completed")
            self.assertEqual(receipt["native_export"]["shape_xyz"], [5, 6, 7])
            self.assertEqual(receipt["image_sha256"], sha256_file(image_path))
            self.assertEqual(receipt["checkpoint_sha256"], sha256_file(checkpoint_path))
            self.assertEqual(receipt["support_policy"], data.SUPPORT_POLICY)
            self.assertTrue(receipt["support_masking_applied"])
            self.assertEqual(receipt["runtime_compatibility"]["kind"], "exact_source_match")
            normalized, support, _ = data.preprocess_image(image_path, config["target_shape_dhw"], return_support=True)
            with np.load(output / "prediction_model_grid.npz") as exported:
                labels = exported["prediction"]
                self.assertTrue(np.any(support & (normalized == 0)))
                self.assertTrue((labels[support & (normalized == 0)] > 0).all())
                self.assertTrue((labels[~support] == 0).all())
            self.assertEqual(set(receipt["outputs"]), {
                "prediction_native.nii.gz", "prediction_model_grid.npz", "prediction_overlay.png"})
            self.assertEqual(set(output.iterdir()), {output / name for name in receipt["outputs"]}
                             | {output / "prediction_receipt.json"})
            for name, digest in receipt["outputs"].items():
                self.assertEqual(digest, sha256_file(output / name))

    def test_native_export_retains_edge_voxels_when_native_grid_is_larger(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = folder / "input.nii.gz"
            nib.save(nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.float32), np.eye(4)), source)
            grid, metadata = data.preprocess_image(source, (4, 4, 4))
            output = folder / "prediction.nii.gz"
            predict.save_native_prediction(np.full(grid.shape, 5, dtype=np.uint8), metadata, source, output)
            # Same physical field of view: resizing back must not invent a
            # background border at native voxel centres inside edge voxels.
            np.testing.assert_array_equal(np.asarray(nib.load(output).dataobj), np.full((8, 8, 8), 5))


class CheckpointSourceTests(unittest.TestCase):
    def test_source_mismatch_delegates_to_explicit_validator_without_mutation(self):
        config = small_config()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_path = Path(temporary) / "best.pt"
            identity = {"config_sha256": data.config_digest(config), "source_sha256": {"legacy": "pinned"}}
            payload = {"schema_version": 1, "model": "unet2d", "config": config,
                       "identity": identity, "state_dict": {}, "epoch": 1, "best_epoch": 1,
                       "best_score": .2, "history": [{"epoch": 1, "validation_prostate_dice": .2}]}
            train.save_checkpoint(checkpoint_path, payload)
            original_sha = sha256_file(checkpoint_path)
            with patch.object(train.checkpoint_compatibility, "validate_compatibility", return_value={"kind": "verified"}) as validate:
                checkpoint = train.load_selected_checkpoint(checkpoint_path, config)
            validate.assert_called_once_with(checkpoint_path, checkpoint, config, train.source_identity())
            self.assertEqual(checkpoint, payload)
            self.assertEqual(checkpoint["identity"], identity)
            self.assertEqual(sha256_file(checkpoint_path), original_sha)
            with patch.object(train.checkpoint_compatibility, "validate_compatibility", side_effect=ValueError("unpinned runtime")):
                with self.assertRaisesRegex(ValueError, "unpinned runtime"):
                    train.load_selected_checkpoint(checkpoint_path, config)

    def test_compatibility_does_not_relax_manifest_source_identity(self):
        config = small_config()
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            manifest = manifest_identity_fixture(folder, config)
            run = make_run(folder / "run", "unet2d", manifest, config)
            path = run / "best.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            checkpoint["identity"]["source_sha256"] = {"legacy": "pinned"}
            train.save_checkpoint(path, checkpoint)
            with patch.object(train.checkpoint_compatibility, "validate_compatibility", return_value={"kind": "verified"}):
                with self.assertRaisesRegex(ValueError, "data/source identity differs"):
                    train.load_selected_checkpoint(path, config, manifest)


class FreezeAndTrainingTests(unittest.TestCase):
    def test_pair_freeze_binds_actual_selected_bytes(self):
        config = small_config()
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            manifest = manifest_identity_fixture(folder, config)
            runs = {name: make_run(folder / name, name, manifest, config) for name in train.MODELS}
            freeze = folder / "frozen.json"
            result = train.freeze_selection(runs, manifest, config, freeze)
            self.assertEqual(result["models"]["improved3d"]["selected_epoch"], 5)
            train.validate_freeze(freeze, manifest, config, "unet2d", runs["unet2d"] / "best.pt")
            with (runs["improved3d"] / "best.pt").open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                train.validate_freeze(freeze, manifest, config)

    def test_incomplete_history_cannot_freeze(self):
        config = small_config()
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            manifest = manifest_identity_fixture(folder, config)
            runs = {name: make_run(folder / name, name, manifest, config) for name in train.MODELS}
            history = load_json(runs["unet2d"] / "history.json")
            save_json(runs["unet2d"] / "history.json", history[:-1])
            with self.assertRaisesRegex(ValueError, "Missing/repeated"):
                train.freeze_selection(runs, manifest, config, folder / "frozen.json")

    def test_manifest_tamper_prevents_checkpoint_load(self):
        config = small_config()
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            manifest = manifest_identity_fixture(folder, config)
            run = make_run(folder / "run", "unet2d", manifest, config)
            save_json(manifest, {"changed": True})
            with self.assertRaisesRegex(ValueError, "manifest changed"):
                train.load_selected_checkpoint(run / "best.pt", config, manifest)

    def test_synthetic_thirty_epoch_loop_counts_updates_and_reloads_earliest_best(self):
        # Real backward/AdamW/checkpoint IO, tiny images/model. Evaluation is an
        # explicit controlled fixture, not a segmentation experiment result.
        config = small_config()
        config["lr"] = .02
        scores = iter([.1, .3, .4, .4, .2, .25, .35, .4])
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            manifest = manifest_identity_fixture(folder, config)
            with patch.object(data, "SampledTrainingDataset", TinySamples), \
                 patch.object(data, "training_class_weights", return_value=(torch.ones(6), [1]*6)), \
                 patch.object(train, "create_model", side_effect=lambda *args: TinyTrainingModel()), \
                 patch.object(train, "evaluate", side_effect=lambda *args, **kwargs: {"patient_balanced_prostate_dice": next(scores)}), \
                 patch.object(train, "save_training_plot"):
                run = train.train_model("unet2d", manifest, config, folder / "run", torch.device("cpu"))
            self.assertEqual(run["status"], "completed")
            self.assertEqual(run["completed_epochs"], 30)
            self.assertEqual(run["actual_optimizer_steps"], 30)
            self.assertEqual(run["actual_training_examples"], 60)
            self.assertEqual(run["best_epoch"], 10)
            self.assertEqual(run["validation_reload_delta"], 0.)
            history = load_json(folder / "run/history.json")
            self.assertEqual(sum(row["amp_overflow_skipped_optimizer_steps"] for row in history), 0)
            self.assertTrue(all(row["sampling"]["foreground_requested"] == 1 for row in history))
            self.assertLess(history[-1]["train_loss"], history[0]["train_loss"])
            selected = train.load_selected_checkpoint(folder / "run/best.pt", config, manifest)
            self.assertEqual(selected["epoch"], 10)
            self.assertNotEqual(run["initial_state_sha256"], train.state_digest(selected["state_dict"]))


if __name__ == "__main__":
    unittest.main()
