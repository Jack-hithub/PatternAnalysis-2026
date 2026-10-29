"""Build a fresh PRIVATE executable support-fix ZIP without changing weights.

Only whitelisted inference sources, the explicit compatibility profile, config,
schema, split, dependencies, license, anonymous aggregate evidence and exact
previously distributed checkpoint bytes enter the archive. No upload occurs.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import checkpoint_compatibility as compatibility
from tools.build_support_compatibility import construct_profile

BASENAME = "demo3_hipmri_seed3711_support_fix.zip"
COPY_NAMES = (*compatibility.SOURCE_FILES, "checkpoint_compatibility.py",
              "inference_compatibility.json", "config.json", "label_schema.json",
              "patient_split.json", "requirements.txt", "requirements-gpu.txt",
              "tools/predict_and_report.py", "LICENSE.txt")
WEIGHT_NAMES = {model: f"weights/{model}_best.pt" for model in compatibility.WEIGHTS}
GENERATED_NAMES = ("run_summary.json", "support_fix_evidence.json",
                   "UNCHANGED_CHECKPOINTS.json", "INFERENCE_INSTRUCTIONS.md")
MANIFEST_NAME = "PACKAGE_MANIFEST.json"
SHA_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
EVIDENCE_FIELDS = {
    "schema_version", "kind", "status", "cases_total", "split_counts",
    "source_inventory_verified", "source_cache_images_checked", "cache_manifest_sha256",
    "by_split", "valid_zero_voxels_total", "changed_mask_voxels_total",
    "selected_weights_unchanged", "clinical_inference_performed", "runtime_source_sha256",
    "legacy_source_sha256", "support_policy", "metrics_provenance", "affected_case_reevaluation",
}
COUNTER_FIELDS = {"cases", "valid_zero_voxels", "outside_support_voxels", "changed_mask_voxels",
                  "raw_zero_voxels", "total_voxels", "cached_images_checked"}


def nonnegative_integer(value):
    return type(value) is int and value >= 0


def save_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def anonymous_evidence(evidence, sources):
    """Reject arrays, identifiers, paths, unknown fields and incomplete audits."""
    require = compatibility.require
    require(isinstance(evidence, dict) and set(evidence) <= EVIDENCE_FIELDS,
            "aggregate audit contains unapproved fields")
    required = EVIDENCE_FIELDS - {"legacy_source_sha256", "affected_case_reevaluation"}
    require(required <= set(evidence), "aggregate support audit is incomplete")
    require(type(evidence["schema_version"]) is int and evidence["schema_version"] == 1
            and evidence["kind"] == "raw_support_equivalence_audit"
            and evidence["status"] == "passed", "support audit has not passed")
    require(evidence["support_policy"] == compatibility.SUPPORT_POLICY,
            "audit support policy differs")
    require(evidence["runtime_source_sha256"] == sources, "audit runtime sources differ")
    if "legacy_source_sha256" in evidence:
        require(evidence["legacy_source_sha256"] == compatibility.BASELINE_SOURCES,
                "audit legacy sources differ")
    for key in ("cases_total", "source_cache_images_checked", "valid_zero_voxels_total",
                "changed_mask_voxels_total"):
        require(nonnegative_integer(evidence[key]), f"invalid aggregate audit counter: {key}")
    require(evidence["cases_total"] == 211
            and evidence["split_counts"] == {"train": 143, "validation": 38, "test": 30},
            "audit must cover every fixed-study image")
    require(evidence["source_inventory_verified"] is True
            and evidence["selected_weights_unchanged"] is True
            and evidence["clinical_inference_performed"] is False,
            "audit inventory/weight/no-clinical-inference proof missing")
    require(isinstance(evidence["cache_manifest_sha256"], str)
            and SHA_PATTERN.fullmatch(evidence["cache_manifest_sha256"]),
            "invalid cache manifest digest")
    require(evidence["source_cache_images_checked"] == 30,
            "all original TEST cache images must be checked")
    require(isinstance(evidence["by_split"], dict)
            and set(evidence["by_split"]) == set(evidence["split_counts"]),
            "audit split counters differ")
    for split, counters in evidence["by_split"].items():
        require(isinstance(counters, dict) and set(counters) <= COUNTER_FIELDS
                and {"cases", "valid_zero_voxels", "outside_support_voxels", "changed_mask_voxels"}
                <= set(counters), "unapproved/incomplete split audit counters")
        require(all(nonnegative_integer(value) for value in counters.values())
                and counters["cases"] == evidence["split_counts"][split],
                "invalid split aggregate counters")
    require(sum(row["valid_zero_voxels"] for row in evidence["by_split"].values())
            == evidence["valid_zero_voxels_total"]
            and sum(row["changed_mask_voxels"] for row in evidence["by_split"].values())
            == evidence["changed_mask_voxels_total"], "aggregate mask totals differ")
    if evidence["changed_mask_voxels_total"] == 0:
        require(evidence["metrics_provenance"] == "retained_after_zero_changed_masks"
                and "affected_case_reevaluation" not in evidence,
                "unchanged-mask metric provenance differs")
    else:
        require(evidence["metrics_provenance"] == "affected_cases_reevaluated_with_frozen_weights",
                "affected cases require fixed-weight re-evaluation")
        proof = evidence.get("affected_case_reevaluation")
        require(isinstance(proof, dict) and set(proof) == {
            "status", "affected_cases", "completed_cases", "runtime_source_sha256",
            "evaluation_evidence_sha256", "models"}, "affected-case proof fields differ")
        require(proof["status"] == "passed" and nonnegative_integer(proof["affected_cases"])
                and proof["affected_cases"] > 0 and nonnegative_integer(proof["completed_cases"])
                and proof["completed_cases"] == proof["affected_cases"]
                and proof["runtime_source_sha256"] == sources
                and isinstance(proof["evaluation_evidence_sha256"], str)
                and SHA_PATTERN.fullmatch(proof["evaluation_evidence_sha256"]),
                "affected-case fixed-weight re-evaluation is incomplete")
        require(isinstance(proof["models"], dict) and set(proof["models"]) == set(compatibility.WEIGHTS),
                "both fixed models require affected-case re-evaluation")
        for model, row in proof["models"].items():
            require(isinstance(row, dict) and set(row) == {"checkpoint_sha256", "prediction_changed_voxels"}
                    and row["checkpoint_sha256"] == compatibility.WEIGHTS[model]["public_sha256"]
                    and nonnegative_integer(row["prediction_changed_voxels"]),
                    "affected-case checkpoint/prediction proof differs")
    return deepcopy(evidence)


def instructions():
    return """# Private HipMRI raw support correction

This package contains an executable inference correction. Support is derived
from nonzero voxels of the raw MRI, follows canonical orientation, and is
resampled independently with nearest-neighbour interpolation. Intensity
normalisation cannot redefine this mask. No labels enter image-only inference.

Both previously distributed checkpoint files are included byte-for-byte. Their
training, selected epochs, scores, histories, original execution identities and
pre-existing legacy metadata remain unchanged. The corrected runtime did not
train these models. Configuration, architecture and metric source files retain
their pinned AED baseline bytes. This change is not a documentation-only proof.

Unzip and run from the package root. Install requirements.txt in an appropriate
Python environment; requirements-gpu.txt records the original observed GPU
dependencies. The included Apache 2.0 license covers upstream code; dataset
attribution is in label_schema.json. No MRI, label, prediction array or cache is
included. Use your own raw 3D MRI with a fresh output directory:

```sh
python tools/predict_and_report.py --image /path/to/mri.nii.gz --checkpoint weights/improved3d_best.pt --config config.json --output-dir /path/to/new_output --device cpu
```

Use weights/unet2d_best.pt for the selected 2D baseline. Keep all runtime files
and inference_compatibility.json together and unchanged. Exact current source
identity still takes the normal strict path. Legacy checkpoints require exact
checkpoint bytes, config, legacy identity, five runtime source hashes and helper
hashes explicitly declared in the profile. A changed or missing declaration
fails. Output receipts separate legacy training identity from corrected runtime
identity. Training/cache manifest identity remains strict.

run_summary.json labels original training information separately from the new
runtime. support_fix_evidence.json is an anonymous full-cohort aggregate audit;
metric provenance is retained only after zero changed masks, or supported by
fixed-weight affected-case re-evaluation. UNCHANGED_CHECKPOINTS.json records the
comparison of all 94 tensors to actual original SHA-pinned frozen checkpoints.
PACKAGE_MANIFEST.json hashes every payload except itself; the separate private
packaging receipt hashes the full ZIP. This tool performs no publication,
training, tuning or model inference.
"""


def build_package(output_dir, evidence_path, *, profile_path=None, public_payload=None):
    require = compatibility.require
    output = Path(output_dir).absolute()
    require(not output.exists() and not output.is_symlink(), "use a fresh private package output directory")
    profile_path = ROOT / "inference_compatibility.json" if profile_path is None else Path(profile_path)
    public = ROOT / "artifacts/releases/current_study_v2/payload" if public_payload is None else Path(public_payload)
    sources = compatibility.current_source_identity()
    profile = compatibility.read_json(profile_path)
    profile_file_sha = compatibility.sha256_file(profile_path)
    compatibility.validate_profile(profile, sources)
    # Repeat the actual frozen tensor/byte proof; do not inherit an old AST claim.
    actual_proof = construct_profile(public_payload=public)
    compatibility.compare_exact(actual_proof, profile, "explicit profile versus actual frozen proof")
    evidence = anonymous_evidence(compatibility.read_json(evidence_path), sources)
    checkpoints = {}
    for model, relative in WEIGHT_NAMES.items():
        original = public / relative
        checkpoint = torch.load(original, map_location="cpu", weights_only=True)
        compatibility.validate_compatibility(original, checkpoint, checkpoint["config"], sources, profile_path)
        checkpoints[model] = checkpoint
    output.mkdir(parents=True, exist_ok=False)
    payload = output / "payload"
    payload.mkdir()
    for name in COPY_NAMES:
        source = (ROOT.parents[1] / "LICENSE.txt" if name == "LICENSE.txt"
                  else profile_path if name == "inference_compatibility.json" else ROOT / name)
        require(source.is_file() and not source.is_symlink(), f"missing/linked package source: {name}")
        destination = payload / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    require(compatibility.current_source_identity(payload) == sources
            and compatibility.sha256_file(payload / "checkpoint_compatibility.py") == profile["helper_sha256"]
            and compatibility.sha256_file(payload / "config.json") == compatibility.CONFIG_FILE_SHA256
            and compatibility.sha256_file(payload / "inference_compatibility.json") == profile_file_sha,
            "copied runtime/profile/config differs from the explicit profile")
    for model, name in WEIGHT_NAMES.items():
        destination = payload / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(public / name, destination)
        require(compatibility.sha256_file(destination) == compatibility.WEIGHTS[model]["public_sha256"],
                "copied public checkpoint bytes differ")
        copied = torch.load(destination, map_location="cpu", weights_only=True)
        compatibility.compare_exact(checkpoints[model], copied, "exact copied legacy checkpoint")
    tensor_proof = {
        "schema_version": 1, "kind": compatibility.KIND,
        "proof_method": "Exact file SHA256 plus tensor key/dtype/shape/value comparison against actual frozen files",
        "total_state_tensors": sum(row["tensor_count"] for row in profile["checkpoints"].values()),
        "all_94_tensors_equal_to_actual_original_frozen": True,
        "all_public_checkpoint_bytes_unchanged": True,
        "all_public_checkpoint_metadata_unchanged": True,
        "training_provenance": "unchanged_original_execution",
        "selection_freeze_sha256": compatibility.FREEZE_SHA256,
        "models": deepcopy(profile["checkpoints"]),
    }
    summary = {
        "schema_version": 1, "kind": compatibility.KIND,
        "packaging_scope": "Private executable raw support correction; anonymous aggregates; no medical arrays",
        "training_provenance": "Unchanged original frozen training; corrected runtime was not used for training",
        "legacy_source_sha256": dict(compatibility.BASELINE_SOURCES),
        "original_training_source_sha256": dict(compatibility.EXECUTION_SOURCES),
        "runtime_source_sha256": sources, "support_policy": compatibility.SUPPORT_POLICY,
        "configuration_changed": False, "models": {},
        "metrics_provenance": evidence["metrics_provenance"],
    }
    for model, checkpoint in checkpoints.items():
        summary["models"][model] = {
            "original_completed_epochs": len(checkpoint["history"]),
            "original_selected_epoch": checkpoint["best_epoch"],
            "original_selected_validation_prostate_dice": checkpoint["best_score"],
            "original_training_runtime": deepcopy(checkpoint["runtime"]),
            "public_checkpoint_sha256": compatibility.WEIGHTS[model]["public_sha256"],
            "original_frozen_checkpoint_sha256": compatibility.WEIGHTS[model]["frozen_sha256"],
        }
    save_json(payload / "UNCHANGED_CHECKPOINTS.json", tensor_proof)
    save_json(payload / "run_summary.json", summary)
    save_json(payload / "support_fix_evidence.json", evidence)
    (payload / "INFERENCE_INSTRUCTIONS.md").write_text(instructions(), encoding="utf-8")
    whitelist = set(COPY_NAMES) | set(WEIGHT_NAMES.values()) | set(GENERATED_NAMES)
    actual_names = {path.relative_to(payload).as_posix() for path in payload.rglob("*") if path.is_file()}
    require(actual_names == whitelist, "private payload whitelist differs")
    manifest = {
        "schema_version": 1, "kind": compatibility.KIND, "private_only": True,
        "archive_basename": BASENAME,
        "entries": {name: {"sha256": compatibility.sha256_file(payload / name),
                           "bytes": (payload / name).stat().st_size} for name in sorted(whitelist)},
        "entry_scope": "Every whitelisted payload except PACKAGE_MANIFEST.json; external receipt hashes complete ZIP",
        "runtime_source_sha256": sources,
        "compatibility_profile_sha256": compatibility.sha256_file(payload / "inference_compatibility.json"),
        "all_public_checkpoint_bytes_unchanged": True,
        "all_94_tensors_equal_to_actual_original_frozen": True,
        "training_provenance": "unchanged_original_execution",
        "executable_inference_changed": True,
    }
    save_json(payload / MANIFEST_NAME, manifest)
    archive_path = output / BASENAME
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(whitelist | {MANIFEST_NAME}):
            archive.write(payload / name, name)
    with zipfile.ZipFile(archive_path) as archive:
        require(set(archive.namelist()) == whitelist | {MANIFEST_NAME}
                and len(archive.namelist()) == len(whitelist) + 1
                and archive.testzip() is None, "ZIP whitelist/integrity verification failed")
        import hashlib
        for name, info in manifest["entries"].items():
            copied_bytes = archive.read(name)
            require(len(copied_bytes) == info["bytes"]
                    and hashlib.sha256(copied_bytes).hexdigest() == info["sha256"],
                    f"ZIP bytes differ: {name}")
    # Recheck the source/profile after archive construction to catch concurrent edits.
    require(compatibility.current_source_identity() == sources
            and compatibility.sha256_file(compatibility.__file__) == profile["helper_sha256"],
            "runtime changed during packaging")
    receipt = {
        "schema_version": 1, "kind": compatibility.KIND, "status": "verified_private",
        "archive_basename": BASENAME, "archive_sha256": compatibility.sha256_file(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "package_manifest_sha256": compatibility.sha256_file(payload / MANIFEST_NAME),
        "whitelist_entries": len(whitelist) + 1, "private_only": True,
        "all_public_checkpoint_bytes_unchanged": True,
        "all_94_tensors_equal_to_actual_original_frozen": True,
        "training_provenance": "unchanged_original_execution", "no_publication_performed": True,
    }
    save_json(output / "packaging_receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--public-payload", type=Path)
    args = parser.parse_args()
    result = build_package(args.output_dir, args.evidence, profile_path=args.profile,
                           public_payload=args.public_payload)
    print(f"Created verified private ZIP in {args.output_dir}; SHA256 {result['archive_sha256']}")


if __name__ == "__main__":
    main()
