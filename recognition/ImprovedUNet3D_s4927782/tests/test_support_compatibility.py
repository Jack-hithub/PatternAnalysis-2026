"""Pinned legacy inference compatibility and private packaging checks on CPU.

Profiles are written only into temporary test directories. Original public and
frozen checkpoint files are only read; no final runtime profile is produced.
"""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import checkpoint_compatibility as compatibility
import train
from tools.build_support_compatibility import build_profile, construct_profile
from tools.package_support_fix import anonymous_evidence, build_package


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def audit_fixture(sources):
    return {
        "schema_version": 1, "kind": "raw_support_equivalence_audit", "status": "passed",
        "cases_total": 211, "split_counts": {"train": 143, "validation": 38, "test": 30},
        "source_inventory_verified": True, "source_cache_images_checked": 30,
        "cache_manifest_sha256": "a" * 64,
        "by_split": {split: {"cases": count, "valid_zero_voxels": 0,
                              "outside_support_voxels": count, "changed_mask_voxels": 0}
                     for split, count in (("train", 143), ("validation", 38), ("test", 30))},
        "valid_zero_voxels_total": 0, "changed_mask_voxels_total": 0,
        "selected_weights_unchanged": True, "clinical_inference_performed": False,
        "runtime_source_sha256": sources, "support_policy": compatibility.SUPPORT_POLICY,
        "metrics_provenance": "retained_after_zero_changed_masks",
    }


class SupportCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.public = ROOT / "artifacts/releases/current_study_v2/payload"
        if not all((cls.public / "weights" / f"{model}_best.pt").is_file()
                   and (ROOT / "artifacts/fixed_run" / model / "best.pt").is_file()
                   for model in compatibility.WEIGHTS):
            raise unittest.SkipTest("Pinned completed-study checkpoint fixtures are not installed")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.profile = construct_profile()
        self.sources = dict(self.profile["runtime_source_sha256"])
        self.profile_path = self.folder / "profile.json"
        write_json(self.profile_path, self.profile)
        self.weight_path = self.public / "weights/unet2d_best.pt"
        self.checkpoint = torch.load(self.weight_path, map_location="cpu", weights_only=True)
        self.config = deepcopy(self.checkpoint["config"])

    def validate(self, *, path=None, checkpoint=None, config=None, sources=None, profile_path=None):
        return compatibility.validate_compatibility(
            self.weight_path if path is None else path,
            self.checkpoint if checkpoint is None else checkpoint,
            self.config if config is None else config,
            self.sources if sources is None else sources,
            self.profile_path if profile_path is None else profile_path)

    def replace_profile(self, profile, *, rehash=False):
        if rehash:
            profile["profile_content_sha256"] = compatibility.json_digest(compatibility.profile_content(profile))
        write_json(self.profile_path, profile)

    def test_both_exact_legacy_files_authorized_without_tensor_or_provenance_changes(self):
        total = 0
        for model, pin in compatibility.WEIGHTS.items():
            with self.subTest(model=model):
                path = self.public / "weights" / f"{model}_best.pt"
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                previous = deepcopy(checkpoint)
                receipt = self.validate(path=path, checkpoint=checkpoint)
                compatibility.compare_exact(previous, checkpoint)
                self.assertEqual(compatibility.sha256_file(path), pin["public_sha256"])
                frozen = ROOT / "artifacts/fixed_run" / model / "best.pt"
                self.assertEqual(compatibility.sha256_file(frozen), pin["frozen_sha256"])
                original = torch.load(frozen, map_location="cpu", weights_only=True)
                compatibility.compare_exact(original["state_dict"], checkpoint["state_dict"])
                compatibility.compare_exact(original["identity"], receipt["original_training_identity"])
                compatibility.compare_exact(original["runtime"], checkpoint["runtime"])
                self.assertEqual(receipt["legacy_identity"], checkpoint["identity"])
                self.assertEqual(receipt["kind"], "inference_only_raw_support_fix")
                self.assertEqual(receipt["training_provenance"], "unchanged")
                self.assertEqual(receipt["checkpoint_bytes"], "unchanged")
                self.assertNotEqual(receipt["runtime_source_sha256"], checkpoint["identity"]["source_sha256"])
                total += len(original["state_dict"])
        self.assertEqual(total, 94)

    def test_missing_profile_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "profile missing"):
            self.validate(profile_path=self.folder / "missing.json")

    def test_altered_checkpoint_file_is_rejected(self):
        copied = self.folder / "altered.pt"
        shutil.copyfile(self.weight_path, copied)
        with copied.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "file bytes differ"):
            self.validate(path=copied)

    def test_in_memory_tensor_and_training_identity_changes_are_rejected(self):
        for field in ("state_dict", "identity", "execution_identity", "runtime", "history"):
            changed = deepcopy(self.checkpoint)
            if field == "state_dict":
                changed[field][next(iter(changed[field]))].flatten()[0] += 1
            elif field in ("identity", "execution_identity"):
                changed[field]["manifest_sha256"] = "0" * 64
            elif field == "runtime":
                changed[field]["source_commit"] = "retroactive_source_claim"
            else:
                changed[field][0]["train_loss"] += 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(checkpoint=changed)

    def test_altered_config_and_missing_source_keys_are_rejected(self):
        changed = deepcopy(self.config)
        changed["lr"] *= 2
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            self.validate(config=changed)
        changed_checkpoint = deepcopy(self.checkpoint)
        changed_checkpoint["config"] = changed
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            self.validate(checkpoint=changed_checkpoint, config=changed)
        sources = dict(self.sources)
        sources.pop("train.py")
        with self.assertRaisesRegex(ValueError, "five-source"):
            self.validate(sources=sources)
        sources = dict(self.sources)
        sources["dataset.py"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source hashes differ"):
            self.validate(sources=sources)

    def test_changed_runtime_file_is_rejected_even_when_caller_uses_actual_new_sha(self):
        runtime = self.folder / "runtime"
        runtime.mkdir()
        for name in (*compatibility.SOURCE_FILES, "config.json"):
            shutil.copyfile(ROOT / name, runtime / name)
        with (runtime / "dataset.py").open("a") as stream:
            stream.write("\n# changed runtime\n")
        sources = compatibility.current_source_identity(runtime)
        with patch.object(compatibility, "ROOT", runtime), self.assertRaisesRegex(ValueError, "source hashes differ"):
            self.validate(sources=sources)

    def test_profile_digest_policy_legacy_config_weight_and_helper_tampering_rejected(self):
        for field in ("digest", "kind", "policy", "config", "checkpoint", "helper", "legacy", "extra"):
            profile = deepcopy(self.profile)
            if field == "digest":
                profile["profile_content_sha256"] = "0" * 64
            elif field == "kind":
                profile["kind"] = "documentation_only"
            elif field == "policy":
                profile["policy"]["training_repeated"] = True
            elif field == "config":
                profile["config_sha256"] = "0" * 64
            elif field == "checkpoint":
                profile["checkpoints"]["unet2d"]["public_sha256"] = "0" * 64
            elif field == "helper":
                profile["helper_sha256"] = "0" * 64
            elif field == "legacy":
                profile["checkpoints"]["unet2d"]["legacy_identity"]["source_sha256"]["train.py"] = "0" * 64
            else:
                profile["allow_any_checkpoint"] = True
            self.replace_profile(profile, rehash=field != "digest")
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate()

    def test_immutable_architecture_and_metric_sources_cannot_be_reprofiled(self):
        for name in ("modules.py", "utils.py"):
            with self.subTest(name=name):
                runtime = self.folder / name.removesuffix(".py")
                runtime.mkdir()
                for file in (*compatibility.SOURCE_FILES, "config.json"):
                    shutil.copyfile(ROOT / file, runtime / file)
                with (runtime / name).open("a") as stream:
                    stream.write("\n# mutation\n")
                sources = compatibility.current_source_identity(runtime)
                profile = deepcopy(self.profile)
                profile["runtime_source_sha256"] = sources
                self.replace_profile(profile, rehash=True)
                with patch.object(compatibility, "ROOT", runtime), self.assertRaisesRegex(ValueError, "AED baseline"):
                    self.validate(sources=sources)

    def test_profile_builder_never_overwrites_and_never_writes_weights(self):
        output = self.folder / "fresh_profile.json"
        before = {model: compatibility.sha256_file(self.public / "weights" / f"{model}_best.pt")
                  for model in compatibility.WEIGHTS}
        build_profile(output)
        with self.assertRaisesRegex(ValueError, "never overwrite"):
            build_profile(output)
        self.assertEqual(before, {model: compatibility.sha256_file(self.public / "weights" / f"{model}_best.pt")
                                  for model in compatibility.WEIGHTS})

    def test_profile_builder_rejects_modified_actual_frozen_original(self):
        run = self.folder / "frozen"
        (run / "unet2d").mkdir(parents=True)
        shutil.copyfile(ROOT / "artifacts/fixed_run/selection_freeze.json", run / "selection_freeze.json")
        copied = run / "unet2d/best.pt"
        shutil.copyfile(ROOT / "artifacts/fixed_run/unet2d/best.pt", copied)
        with copied.open("ab") as stream:
            stream.write(b"modified actual original")
        with self.assertRaisesRegex(ValueError, "actual frozen unet2d checkpoint bytes differ"):
            construct_profile(run_root=run)

    def test_exact_current_source_checkpoints_keep_normal_strict_guard(self):
        checkpoint = deepcopy(self.checkpoint)
        checkpoint["identity"]["source_sha256"] = train.source_identity()
        path = self.folder / "exact_current.pt"
        torch.save(checkpoint, path)
        with patch.object(compatibility, "validate_compatibility", side_effect=AssertionError("should not be called")):
            loaded = train.load_selected_checkpoint(path, self.config)
        self.assertEqual(loaded["identity"], checkpoint["identity"])

    def test_private_package_rejects_existing_output_before_any_writes(self):
        output = self.folder / "already_exists"
        output.mkdir()
        with self.assertRaisesRegex(ValueError, "fresh private"):
            build_package(output, self.folder / "missing_audit.json", profile_path=self.profile_path)


class AnonymousEvidenceTests(unittest.TestCase):
    def test_complete_anonymous_aggregate_and_zero_impact_metric_provenance(self):
        sources = {name: "a" * 64 for name in compatibility.SOURCE_FILES}
        evidence = audit_fixture(sources)
        self.assertEqual(anonymous_evidence(evidence, sources), evidence)

    def test_affected_masks_require_complete_fixed_weight_reevaluation_for_both_models(self):
        sources = {name: "a" * 64 for name in compatibility.SOURCE_FILES}
        evidence = audit_fixture(sources)
        evidence["changed_mask_voxels_total"] = 2
        evidence["by_split"]["test"]["changed_mask_voxels"] = 2
        evidence["metrics_provenance"] = "affected_cases_reevaluated_with_frozen_weights"
        evidence["affected_case_reevaluation"] = {
            "status": "passed", "affected_cases": 1, "completed_cases": 1,
            "runtime_source_sha256": sources, "evaluation_evidence_sha256": "b" * 64,
            "models": {model: {"checkpoint_sha256": pin["public_sha256"], "prediction_changed_voxels": 0}
                       for model, pin in compatibility.WEIGHTS.items()},
        }
        self.assertEqual(anonymous_evidence(evidence, sources), evidence)
        evidence["affected_case_reevaluation"]["models"].pop("improved3d")
        with self.assertRaisesRegex(ValueError, "both fixed models"):
            anonymous_evidence(evidence, sources)

    def test_arrays_paths_unproven_metric_reuse_and_nonfinite_counters_rejected(self):
        sources = {name: "a" * 64 for name in compatibility.SOURCE_FILES}
        for change in ("patient", "arrays", "counter", "changed", "status", "runtime"):
            evidence = audit_fixture(sources)
            if change == "patient":
                evidence["patient_path"] = "/private/example.nii.gz"
            elif change == "arrays":
                evidence["by_split"]["test"]["voxel_array"] = [0, 1, 0]
            elif change == "counter":
                evidence["by_split"]["test"]["valid_zero_voxels"] = float("nan")
            elif change == "changed":
                evidence["changed_mask_voxels_total"] = 1
                evidence["by_split"]["test"]["changed_mask_voxels"] = 1
            elif change == "status":
                evidence["status"] = "pending"
            else:
                evidence["runtime_source_sha256"] = {}
            with self.subTest(change=change), self.assertRaises(ValueError):
                anonymous_evidence(evidence, sources)


if __name__ == "__main__":
    unittest.main()
