"""Prove the exact console-only predict.main revision from the public runtime.

The SHA-pinned prior ZIP and historical audit/check records remain unchanged.
Only the existing prediction receipt is assigned, formatted and printed. No
training, image loading, model inference or scientific result is rerun here.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import checkpoint_compatibility as compatibility
from tools.build_support_compatibility import construct_profile

KIND = "predict_cli_stdout_summary_bridge"
BASELINE_BASENAME = "demo3_hipmri_seed3711_support_fix.zip"
BASELINE_BYTES = 8171797
BASELINE_SHA256 = "8129826fd8533179a6b728338c32414d2d60101c4f0748b9f4dc4b09341652be"
BASELINE_PROFILE_SHA256 = "79e8501c909d0300d768181f9914df2cf22a723fb087eb7262dd02714dc4bd2b"
AUDIT_SHA256 = "5ddc66c1fa027e7e3f765b551754f62e7847b8e4be32aeff04b1e8bfc20af577"
CHECKS_SHA256 = "729b4aedf62dddea7ca60eff9f737bd6e446a84e02cb93c47bbca6d05d612723"
UNCHANGED_FILES = ("modules.py", "dataset.py", "utils.py", "train.py",
                   "checkpoint_compatibility.py", "config.json", "label_schema.json",
                   "patient_split.json", "requirements.txt", "requirements-gpu.txt",
                   "tools/predict_and_report.py")
BASELINE_ARCHIVE = ROOT / "artifacts/releases/support_fix_v1" / BASELINE_BASENAME


def bytes_sha(value):
    return hashlib.sha256(value).hexdigest()


def ast_dump(node):
    return ast.dump(node, include_attributes=False)


def stdout_only_proof(old_bytes, new_bytes):
    """Reject every delta except the exact, explicit receipt-summary main edit."""
    require = compatibility.require
    old, new = ast.parse(old_bytes.decode()), ast.parse(new_bytes.decode())
    old_main = [node for node in old.body if isinstance(node, ast.FunctionDef) and node.name == "main"]
    new_main = [node for node in new.body if isinstance(node, ast.FunctionDef) and node.name == "main"]
    require(len(old_main) == len(new_main) == 1, "exactly one ordinary main function required")
    old_main, new_main = old_main[0], new_main[0]
    other_old = ast.Module(body=[node for node in old.body if node is not old_main], type_ignores=old.type_ignores)
    other_new = ast.Module(body=[node for node in new.body if node is not new_main], type_ignores=new.type_ignores)
    require(ast_dump(other_old) == ast_dump(other_new), "non-main predict AST changed")
    old_lines, new_lines = old_bytes.splitlines(keepends=True), new_bytes.splitlines(keepends=True)
    require(old_lines[:old_main.lineno-1] == new_lines[:new_main.lineno-1]
            and old_lines[old_main.end_lineno:] == new_lines[new_main.end_lineno:],
            "predict bytes outside main changed")
    expected = deepcopy(old_main)
    calls = [index for index, node in enumerate(expected.body)
             if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
             and isinstance(node.value.func, ast.Name) and node.value.func.id == "predict_image"]
    require(len(calls) == 1, "exact original predict_image call required")
    index = calls[0]
    assignment = ast.Assign(targets=[ast.Name(id="receipt", ctx=ast.Store())],
                            value=deepcopy(expected.body[index].value))
    summary = ast.parse('from tools.predict_and_report import format_summary\n'
                        'print(format_summary(args.output_dir / "prediction_receipt.json", receipt), flush=True)\n').body
    expected.body[index:index+1] = [assignment, *summary]
    require(ast_dump(expected) == ast_dump(new_main),
            "main differs from the exact existing-call capture/import/print transformation")
    image = next(node for node in other_old.body if isinstance(node, ast.FunctionDef) and node.name == "predict_image")
    return {"main_before_ast_sha256": bytes_sha(ast_dump(old_main).encode()),
            "main_after_ast_sha256": bytes_sha(ast_dump(new_main).encode()),
            "all_non_main_ast_sha256": bytes_sha(ast_dump(other_old).encode()),
            "predict_image_ast_sha256": bytes_sha(ast_dump(image).encode()),
            "strict_main_transformation_verified": True,
            "all_non_main_predict_bytes_unchanged": True,
            "predict_image_ast_unchanged": True}


def construct_bridge(*, root=ROOT, baseline_archive=None, audit_path=None,
                     checks_path=None, profile_path=None):
    root = Path(root)
    archive = BASELINE_ARCHIVE if baseline_archive is None else Path(baseline_archive)
    audit_path = root / "evidence/support_fix_audit.json" if audit_path is None else Path(audit_path)
    checks_path = root / "evidence/support_fix_checks.json" if checks_path is None else Path(checks_path)
    profile_path = root / "inference_compatibility.json" if profile_path is None else Path(profile_path)
    require = compatibility.require
    require(archive.stat().st_size == BASELINE_BYTES
            and compatibility.sha256_file(archive) == BASELINE_SHA256,
            "prior public support-fix ZIP bytes differ")
    profile = compatibility.read_json(profile_path)
    sources = compatibility.current_source_identity(root)
    compatibility.validate_profile(profile, sources, root)
    compatibility.compare_exact(construct_profile(root=root), profile,
                                "current profile versus actual original frozen tensor proof")
    with zipfile.ZipFile(archive) as old:
        manifest_bytes = old.read("PACKAGE_MANIFEST.json")
        manifest = json.loads(manifest_bytes)
        names = old.namelist()
        require(len(names) == len(set(names)) and set(names) == set(manifest["entries"]) | {"PACKAGE_MANIFEST.json"}
                and old.testzip() is None, "prior ZIP whitelist/integrity differs")
        for name, item in manifest["entries"].items():
            value = old.read(name)
            require(len(value) == item["bytes"] and bytes_sha(value) == item["sha256"],
                    f"prior ZIP payload hash differs: {name}")
        old_sources = {name: bytes_sha(old.read(name)) for name in compatibility.SOURCE_FILES}
        require(manifest["runtime_source_sha256"] == old_sources,
                "prior ZIP runtime declaration differs")
        for name in UNCHANGED_FILES:
            require(old.read(name) == (root / name).read_bytes(), f"non-CLI runtime file changed: {name}")
        require([name for name in compatibility.SOURCE_FILES if old_sources[name] != sources[name]] == ["predict.py"],
                "only predict.py may differ from the audited runtime")
        main_proof = stdout_only_proof(old.read("predict.py"), (root / "predict.py").read_bytes())
        historical_profile_bytes = old.read("inference_compatibility.json")
        require(bytes_sha(historical_profile_bytes) == BASELINE_PROFILE_SHA256,
                "historical support profile differs")
        historical_profile = json.loads(historical_profile_bytes)
        expected_old = deepcopy(profile)
        expected_old["runtime_source_sha256"] = old_sources
        expected_old["profile_content_sha256"] = compatibility.json_digest(compatibility.profile_content(expected_old))
        compatibility.compare_exact(expected_old, historical_profile, "historical versus current profile anchors")
        require(audit_path.read_bytes() == old.read("support_fix_evidence.json")
                and compatibility.sha256_file(audit_path) == AUDIT_SHA256,
                "original aggregate audit bytes must remain unchanged")
        audit = compatibility.read_json(audit_path)
        checks = compatibility.read_json(checks_path)
        require(audit["runtime_source_sha256"] == old_sources and audit["cases_total"] == 211
                and audit["changed_mask_voxels_total"] == 0 and audit["source_cache_images_checked"] == 30,
                "original support audit provenance differs")
        require(compatibility.sha256_file(checks_path) == CHECKS_SHA256
                and checks["status"] == "passed" and checks["total_tests"] == 96
                and checks["runtime_source_sha256"] == old_sources
                and checks["compatibility_profile_sha256"] == BASELINE_PROFILE_SHA256,
                "historical 96-check record bytes/provenance differ")
        for model, pin in compatibility.WEIGHTS.items():
            require(bytes_sha(old.read(f"weights/{model}_best.pt")) == pin["public_sha256"],
                    "historical selected checkpoint bytes differ")
    return {
        "schema_version": 1, "kind": KIND, "status": "passed",
        "bridge_builder_sha256": compatibility.sha256_file(Path(__file__)),
        "baseline_archive_basename": BASELINE_BASENAME,
        "baseline_archive_bytes": BASELINE_BYTES, "baseline_archive_sha256": BASELINE_SHA256,
        "baseline_package_manifest_sha256": bytes_sha(manifest_bytes),
        "baseline_profile_sha256": BASELINE_PROFILE_SHA256,
        "runtime_profile_sha256": compatibility.sha256_file(profile_path),
        "historical_audit_sha256": AUDIT_SHA256, "historical_checks_sha256": CHECKS_SHA256,
        "historical_checks_count": 96, "historical_audit_runtime_source_sha256": old_sources,
        "runtime_source_sha256": sources, "helper_sha256": profile["helper_sha256"],
        "changed_core_sources": ["predict.py"],
        "allowed_change": "main_only_existing_prediction_receipt_summary",
        "other_runtime_files_byte_identical": list(UNCHANGED_FILES), **main_proof,
        "all_94_tensors_equal_to_actual_original_frozen": True,
        "all_public_checkpoint_bytes_unchanged": True,
        "training_provenance": "unchanged_original_execution",
        "historical_audit_and_checks_relabelled": False,
        "new_training_or_model_inference_performed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-archive", type=Path)
    args = parser.parse_args()
    compatibility.require(not args.output.exists() and not args.output.is_symlink(), "use a fresh CLI bridge evidence path")
    proof = construct_bridge(baseline_archive=args.baseline_archive)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(proof, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Created strict stdout-only bridge: {args.output}")


if __name__ == "__main__":
    main()
