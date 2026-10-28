"""Build a fresh, private inference package from the completed fixed study.

Optimizer/scaler training state is removed. Optional documentary migration
requires unchanged executable source, numerical settings and model tensors;
every explicitly changed metadata path and the execution identities are kept.
This utility neither runs inference nor publishes an archive.
"""
from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

import dataset as data
from train import (MODELS, SOURCE_FILES, create_model, load_selected_checkpoint,
                   require, source_identity, state_digest)
from utils import load_json, save_json, sha256_file

BASENAME = "demo3_hipmri_seed3711.zip"
REQUIRED_FIELDS = {"schema_version", "model", "config", "identity", "runtime", "state_dict",
                   "epoch", "best_epoch", "best_score", "history"}
SOURCE_WHITELIST = (*SOURCE_FILES, "config.json", "label_schema.json", "patient_split.json",
                    "requirements.txt", "requirements-gpu.txt", "LICENSE.txt",
                    "tools/predict_and_report.py")
WEIGHT_NAMES = {name: f"weights/{name}_best.pt" for name in MODELS}
GENERATED_NAMES = (*WEIGHT_NAMES.values(), "run_summary.json", "INFERENCE_INSTRUCTIONS.md")
MANIFEST_NAME = "PACKAGE_MANIFEST.json"
DOCUMENTARY_SOURCE_FILES = {"modules.py", "dataset.py", "train.py"}


def compare_exact(original, copied, location="checkpoint"):
    """Require all tensor values/dtypes/shapes and metadata to survive pruning."""
    if isinstance(original, torch.Tensor):
        require(isinstance(copied, torch.Tensor) and original.dtype == copied.dtype
                and original.shape == copied.shape and torch.equal(original, copied),
                f"Tensor differs after reload: {location}")
    elif isinstance(original, Mapping):
        require(isinstance(copied, Mapping) and set(original) == set(copied),
                f"Metadata keys differ: {location}")
        for key in original:
            compare_exact(original[key], copied[key], f"{location}/{key}")
    elif isinstance(original, (list, tuple)):
        require(type(original) is type(copied) and len(original) == len(copied),
                f"Metadata sequence differs: {location}")
        for index, value in enumerate(original):
            compare_exact(value, copied[index], f"{location}/{index}")
    else:
        require(type(original) is type(copied) and original == copied,
                f"Metadata value differs: {location}")


def json_digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def source_without_module_docstring(encoded):
    """Remove only the actual module docstring, preserving all other bytes."""
    tree = ast.parse(encoded.decode("utf-8"), type_comments=True)
    if (tree.body and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)):
        node = tree.body.pop(0)
        lines = encoded.splitlines(keepends=True)
        start = sum(map(len, lines[:node.lineno-1])) + node.col_offset
        stop = sum(map(len, lines[:node.end_lineno-1])) + node.end_col_offset
        encoded = encoded[:start] + encoded[stop:]
    return encoded, hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def documentation_proof(execution_source_dir, deployment_source_dir):
    """Prove the narrow source/config migration before touching a checkpoint."""
    old_root, new_root = Path(execution_source_dir), Path(deployment_source_dir)
    old_config, new_config = load_json(old_root/"config.json"), load_json(new_root/"config.json")
    old_numerical, new_numerical = deepcopy(old_config), deepcopy(new_config)
    old_context, new_context = old_numerical.pop("evaluation_context"), new_numerical.pop("evaluation_context")
    require(isinstance(old_context, str) and isinstance(new_context, str) and bool(new_context.strip()),
            "Evaluation context must remain documentary text.")
    compare_exact(old_numerical, new_numerical, "config excluding evaluation_context")
    sources = {}
    for name in SOURCE_FILES:
        old_bytes, new_bytes = (old_root/name).read_bytes(), (new_root/name).read_bytes()
        old_body, old_ast = source_without_module_docstring(old_bytes)
        new_body, new_ast = source_without_module_docstring(new_bytes)
        require(old_ast == new_ast, f"Executable AST changed: {name}")
        if name in DOCUMENTARY_SOURCE_FILES:
            require(old_body == new_body, f"Non-module-docstring source bytes changed: {name}")
        else:
            require(old_bytes == new_bytes, f"Byte-immutable source changed: {name}")
        sources[name] = {"execution_sha256": hashlib.sha256(old_bytes).hexdigest(),
                         "deployment_sha256": hashlib.sha256(new_bytes).hexdigest(),
                         "module_docstring_changed": old_bytes != new_bytes,
                         "ast_without_module_docstring_sha256": old_ast,
                         "non_docstring_bytes_sha256": hashlib.sha256(old_body).hexdigest()}
    proof = {"kind": "documentation_only", "source_files": sources,
             "config_changed_keys": ["evaluation_context"] if old_context != new_context else [],
             "execution_config_sha256": data.config_digest(old_config),
             "deployment_config_sha256": data.config_digest(new_config),
             "executable_source_unchanged": True, "numerical_config_unchanged": True}
    snapshot_path = old_root/"SNAPSHOT_RECEIPT.json"
    if snapshot_path.is_file():
        snapshot = load_json(snapshot_path)
        require(snapshot.get("status") == "verified"
                and snapshot.get("core_source_sha256") == {n: r["execution_sha256"] for n, r in sources.items()},
                "Execution snapshot receipt/source identity differs.")
        proof["execution_archive_sha256"] = snapshot["archive_sha256"]
    return old_config, proof


def migrate_checkpoint(original, deployment_config, deployment_sources, proof, original_sha256):
    """Rebind documentary identity only; preserve and independently prove the rest."""
    require(set(original) == REQUIRED_FIELDS | {"optimizer", "scaler"},
            "Unexpected original checkpoint fields.")
    require(original["identity"]["source_sha256"] == {
                name: row["execution_sha256"] for name, row in proof["source_files"].items()}
            and original["identity"]["config_sha256"] == proof["execution_config_sha256"]
            and data.config_digest(original["config"]) == proof["execution_config_sha256"],
            "Original checkpoint is not bound to execution snapshot.")
    compare_exact(deployment_sources, {name: row["deployment_sha256"] for name, row in proof["source_files"].items()},
                  "deployment source proof")
    compare_exact(data.config_digest(deployment_config), proof["deployment_config_sha256"], "deployment config proof")
    expected_old, expected_new = deepcopy(original["config"]), deepcopy(deployment_config)
    old_context, new_context = expected_old.pop("evaluation_context"), expected_new.pop("evaluation_context")
    compare_exact(expected_old, expected_new, "checkpoint numerical configuration")
    cleaned = {key: deepcopy(value) if key != "state_dict" else value for key, value in original.items()
               if key not in ("optimizer", "scaler")}
    cleaned["config"] = deepcopy(deployment_config)
    cleaned["identity"]["source_sha256"] = deepcopy(deployment_sources)
    cleaned["identity"]["config_sha256"] = data.config_digest(deployment_config)
    history_paths = []
    for index, row in enumerate(original["history"]):
        validation = row.get("validation")
        if validation is not None and "configuration" in validation:
            context = validation["configuration"].get("evaluation_context")
            require(context == old_context, "Unexpected embedded execution evaluation context.")
            cleaned["history"][index]["validation"]["configuration"]["evaluation_context"] = new_context
            if old_context != new_context:
                history_paths.append(f"history/{index}/validation/configuration/evaluation_context")
    # Restore only enumerated documentary changes in a comparison copy. This
    # comparison rejects every tensor, score, numerical setting or other delta.
    restored = deepcopy(cleaned)
    restored["config"]["evaluation_context"] = old_context
    restored["identity"]["source_sha256"] = deepcopy(original["identity"]["source_sha256"])
    restored["identity"]["config_sha256"] = original["identity"]["config_sha256"]
    for path in history_paths:
        index = int(path.split("/")[1])
        restored["history"][index]["validation"]["configuration"]["evaluation_context"] = old_context
    expected = {key: value for key, value in original.items() if key not in ("optimizer", "scaler")}
    compare_exact(expected, restored, "all non-documentary checkpoint metadata/tensors")
    changed_sources = [f"identity/source_sha256/{name}" for name, row in proof["source_files"].items()
                       if row["execution_sha256"] != row["deployment_sha256"]]
    changed_paths = (["config/evaluation_context", "identity/config_sha256"] if old_context != new_context else [])
    changed_paths += changed_sources + history_paths
    cleaned["execution_identity"] = deepcopy(original["identity"])
    cleaned["documentation_migration"] = {
        "schema_version": 1, "proof": deepcopy(proof), "original_selected_checkpoint_sha256": original_sha256,
        "original_full_history_sha256": json_digest(original["history"]),
        "deployed_full_history_sha256": json_digest(cleaned["history"]),
        "changed_checkpoint_paths": changed_paths,
        "added_provenance_fields": ["execution_identity", "documentation_migration"],
        "all_numerical_history_and_non_documentary_metadata_exact": True,
        "all_state_tensors_exact": True}
    verify_documentary_checkpoint(original, cleaned, proof, original_sha256)
    return cleaned


def verify_documentary_checkpoint(original, deployed, proof, original_sha256):
    """Reject any undeclared delta, including a single changed tensor or score."""
    require(set(deployed) == REQUIRED_FIELDS | {"execution_identity", "documentation_migration"},
            "Unexpected deployed checkpoint fields.")
    compare_exact(original["identity"], deployed["execution_identity"], "preserved execution identity")
    migration = deployed["documentation_migration"]
    compare_exact(proof, migration["proof"], "documentary proof")
    require(migration["original_selected_checkpoint_sha256"] == original_sha256,
            "Original selected SHA differs.")
    require(data.config_digest(deployed["config"]) == proof["deployment_config_sha256"],
            "Deployment config does not match documentary proof.")
    compare_exact(deployed["identity"]["source_sha256"], {
        name: row["deployment_sha256"] for name, row in proof["source_files"].items()}, "deployed source hashes")
    require(deployed["identity"]["config_sha256"] == proof["deployment_config_sha256"],
            "Deployment config identity differs.")
    old_context, new_context = original["config"]["evaluation_context"], deployed["config"]["evaluation_context"]
    history_paths = []
    restored = {key: deepcopy(value) for key, value in deployed.items() if key in REQUIRED_FIELDS}
    restored["config"]["evaluation_context"] = old_context
    restored["identity"]["source_sha256"] = deepcopy(original["identity"]["source_sha256"])
    restored["identity"]["config_sha256"] = original["identity"]["config_sha256"]
    require(len(deployed["history"]) == len(original["history"]), "History length changed.")
    for index, row in enumerate(original["history"]):
        validation = row.get("validation")
        if validation is not None and "configuration" in validation:
            require(validation["configuration"]["evaluation_context"] == old_context,
                    "Unexpected execution history context.")
            require(deployed["history"][index]["validation"]["configuration"]["evaluation_context"] == new_context,
                    "History context does not match deployment context.")
            restored["history"][index]["validation"]["configuration"]["evaluation_context"] = old_context
            if old_context != new_context:
                history_paths.append(f"history/{index}/validation/configuration/evaluation_context")
    expected = {key: value for key, value in original.items() if key not in ("optimizer", "scaler")}
    compare_exact(expected, restored, "numerical history/non-documentary metadata/tensors")
    changed = (["config/evaluation_context", "identity/config_sha256"] if old_context != new_context else [])
    changed += [f"identity/source_sha256/{name}" for name, row in proof["source_files"].items()
                if row["execution_sha256"] != row["deployment_sha256"]]
    compare_exact(changed + history_paths, migration["changed_checkpoint_paths"], "allowed metadata path list")
    require(migration["original_full_history_sha256"] == json_digest(original["history"])
            and migration["deployed_full_history_sha256"] == json_digest(deployed["history"]),
            "History provenance digest differs.")


def instructions():
    return """# HipMRI inference package

Both selected models completed 30 epochs with seed3711. Selection used only the
seven scheduled VAL assessments; each selected epoch is recorded in the manifest.
TRAIN, VAL and TEST are patient-disjoint. Both VAL-selected checkpoints were
frozen before this run's final TEST evaluation. This package does not contain
MRI files, labels, prediction arrays, caches or example images.

Unzip the archive and run commands from its root. Install `requirements.txt`
in your own Python environment. Exact versions used for training (Python3.10.21,
PyTorch2.14.0+cu130 on NVIDIA L40S) and the other library versions are recorded
in `run_summary.json`; choose a PyTorch build suitable for your CPU/GPU.
`requirements-gpu.txt` contains the observed package-version pins; the runtime
summary also records the CUDA build/runtime. The upstream repository's Apache2.0
license is included as `LICENSE.txt`. Dataset source and attribution are recorded
separately in `label_schema.json`.

Example on CPU, using your own raw3D MRI NIfTI and a new output directory:

```sh
python tools/predict_and_report.py --image /path/to/new_mri.nii.gz --checkpoint weights/improved3d_best.pt --config config.json --output-dir /path/to/new_output --device cpu
```

Use `weights/unet2d_best.pt` for the2D baseline. Use `--device cuda` only with a
working CUDA installation. The program prints results and writes a native-grid
NIfTI, a model-grid prediction, a prediction-only overlay and a hash receipt.
The raw image alone is used; no ground-truth label is needed. Native coordinate
restoration does not recover information lost by model-grid resampling.

To verify and print an existing output receipt without running inference:

```sh
python tools/predict_and_report.py --receipt /path/to/new_output/prediction_receipt.json
```

The original `predict.py` remains available as the implementation entry point.
Its source/config/selected-checkpoint checks remain enabled. Keep all five core
Python source files together and unmodified. Class IDs are0Background,1Body,
2Bones,3Bladder,4Rectum,5Prostate; see `label_schema.json` for the official source
and HipMRI/CSIRO dataset attribution. All model inputs cover the complete image;
3D uses overlapping patches and probability averaging.

`PACKAGE_MANIFEST.json` hashes every payload file except itself (self-hashing
would be circular). The external packaging receipt hashes the completeZIP.
Checkpoint maps connect original selected SHA256s to these inference files;
optimizer/scaler state was removed. All94model state tensors, numerical history,
scores and non-documentary metadata were verified exactly after a weights-only
CPU reload. Module docstrings and evaluation-context text were synchronized for
delivery; each changed metadata path is declared in the manifest/checkpoints.
The original execution source/config identity and full-history SHA256 remain
recorded separately. Source bodies are unchanged byte-for-byte outside the
allowed module docstrings, and numerical configuration is unchanged. Current
deployment source/config guards remain enabled. No inference or new experiment
was used to make this package. These files support image-only inference, not
resuming the original optimizer or reusing execution caches with rebound metadata.
"""


def build_package(run_root, selection_freeze, output_dir, execution_source_dir=None):
    run_root = Path(run_root).resolve(strict=True)
    freeze_path = Path(selection_freeze).resolve(strict=True)
    output = Path(output_dir).absolute()
    require(not output.exists() and not output.is_symlink(), "Use a fresh package output directory.")
    config = load_json(ROOT / "config.json")
    frozen = load_json(freeze_path)
    source_sha = source_identity()
    execution_config, proof = config, None
    execution_sources = source_sha
    if execution_source_dir is not None:
        execution_config, proof = documentation_proof(execution_source_dir, ROOT)
        execution_sources = {name: row["execution_sha256"] for name, row in proof["source_files"].items()}
    require(frozen.get("status") == "frozen" and set(frozen.get("models", {})) == set(MODELS),
            "A completed two-model selection freeze is required.")
    require(frozen.get("config_sha256") == data.config_digest(execution_config)
            and frozen.get("source_sha256") == execution_sources, "Frozen execution source/config identity differs.")
    inputs = {"selection_freeze": sha256_file(freeze_path)}
    for name in MODELS:
        original = run_root / name / "best.pt"
        run_path = run_root / name / "run.json"
        inputs[name + "/best.pt"] = sha256_file(original)
        inputs[name + "/run.json"] = sha256_file(run_path)
        require(inputs[name + "/best.pt"] == frozen["models"][name]["checkpoint_sha256"],
                f"Original selected weight does not match freeze: {name}")
    output.mkdir(parents=True, exist_ok=False)
    payload = output / "payload"
    payload.mkdir()
    receipt_path = output / "packaging_receipt.json"
    save_json(receipt_path, {"status": "running", "input_sha256": inputs})
    try:
        for relative in SOURCE_WHITELIST:
            destination = payload / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = ROOT.parents[1] / "LICENSE.txt" if relative == "LICENSE.txt" else ROOT / relative
            shutil.copyfile(source, destination)
        checks, summary = {}, {}
        for name in MODELS:
            original_path = run_root / name / "best.pt"
            run = load_json(run_root / name / "run.json")
            original = load_selected_checkpoint(original_path, execution_config, check_sources=proof is None)
            require(original["identity"]["source_sha256"] == execution_sources,
                    "Original checkpoint source does not match execution snapshot.")
            choice = frozen["models"][name]
            require(run.get("status") == "completed" and run.get("completed_epochs") == config["epochs"]
                    and run.get("model") == name and original["model"] == name
                    and run.get("config") == execution_config and run.get("identity") == original["identity"],
                    f"Completed run identity differs: {name}")
            require(choice.get("completed_epochs") == config["epochs"]
                    and choice.get("selected_epoch") == original["epoch"] == run.get("best_epoch")
                    and run.get("best_checkpoint_sha256") == inputs[name + "/best.pt"]
                    and original["identity"]["manifest_sha256"] == frozen["manifest_sha256"]
                    and original["best_score"] == choice["validation_prostate_dice"],
                    f"Selected epoch/weight/data/score differs: {name}")
            require(set(original) == REQUIRED_FIELDS | {"optimizer", "scaler"},
                    "Unexpected checkpoint fields; refuse to package unreviewed contents.")
            cleaned = (migrate_checkpoint(original, config, source_sha, proof, inputs[name + "/best.pt"])
                       if proof is not None else {key: value for key, value in original.items()
                                                 if key not in ("optimizer", "scaler")})
            # Only state_dict may contain tensors. Remaining metadata must be
            # ordinary finite JSON scalars/containers, not executable objects.
            json.dumps({key: value for key, value in cleaned.items() if key != "state_dict"}, allow_nan=False)
            destination = payload / WEIGHT_NAMES[name]
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".pt.tmp")
            torch.save(cleaned, temporary)
            temporary.replace(destination)
            reloaded = torch.load(destination, map_location="cpu", weights_only=True)
            compare_exact(cleaned, reloaded)
            if proof is not None:
                verify_documentary_checkpoint(original, reloaded, proof, inputs[name + "/best.pt"])
            accepted = load_selected_checkpoint(destination, config)
            model = create_model(name, config)
            model.load_state_dict(accepted["state_dict"], strict=True)
            require(state_digest(model.state_dict()) == state_digest(original["state_dict"]),
                    f"Instantiated model tensor identity differs: {name}")
            tensor_count = len(original["state_dict"])
            tensor_elements = sum(value.numel() for value in original["state_dict"].values())
            checks[name] = {"original_selected_sha256": inputs[name + "/best.pt"],
                           "inference_path": WEIGHT_NAMES[name], "inference_sha256": sha256_file(destination),
                           "removed_fields": ["optimizer", "scaler"],
                           "base_checkpoint_fields_retained": sorted(REQUIRED_FIELDS),
                           "all_remaining_metadata_exact": proof is None,
                           "all_numerical_history_and_non_documentary_metadata_exact": True,
                           "all_state_tensors_exact": True,
                           "state_tensor_count": tensor_count, "state_tensor_elements": tensor_elements,
                           "state_sha256": state_digest(original["state_dict"]),
                           "selected_epoch": original["epoch"], "strict_model_load_passed": True,
                           "selected_checkpoint_source_guard_passed": True}
            if proof is not None:
                checks[name]["documentation_migration"] = deepcopy(cleaned["documentation_migration"])
            scalar_fields = ("parameters", "completed_epochs", "best_epoch", "best_validation_prostate_dice",
                             "actual_optimizer_steps", "actual_training_examples", "training_loop_seconds",
                             "scheduled_validation_seconds", "total_wall_seconds", "validation_reload_delta",
                             "initial_state_sha256")
            summary[name] = {key: run[key] for key in scalar_fields}
            summary[name].update(runtime=run["runtime"], original_run_sha256=inputs[name + "/run.json"],
                                 original_selected_checkpoint_sha256=inputs[name + "/best.pt"])
            del model, original, cleaned, reloaded, accepted
        save_json(payload / "run_summary.json", {
            "evaluation_context": config["evaluation_context"], "selection": config["selection"],
            "seed": config["seed"], "models": summary,
            "source_sha256": source_sha, "execution_source_sha256": execution_sources,
            "source_execution_commit": summary["unet2d"]["runtime"]["source_commit"],
            "documentation_migration": proof,
            "selection_freeze_sha256": inputs["selection_freeze"],
            "packaging_scope": "Inference weights plus source and anonymous scalar run summaries; no medical image/label/prediction arrays."})
        (payload / "INFERENCE_INSTRUCTIONS.md").write_text(instructions(), encoding="utf-8")
        expected_payload = set(SOURCE_WHITELIST) | set(GENERATED_NAMES)
        actual_payload = {path.relative_to(payload).as_posix() for path in payload.rglob("*") if path.is_file()}
        require(actual_payload == expected_payload, "Payload differs from explicit file whitelist.")
        entries = {relative: {"sha256": sha256_file(payload / relative), "bytes": (payload / relative).stat().st_size}
                   for relative in sorted(expected_payload)}
        manifest = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                    "archive_basename": BASENAME, "entries": entries,
                    "entry_scope": "Every archive payload except PACKAGE_MANIFEST.json; external receipt hashes completeZIP.",
                    "checkpoint_checks": checks, "deployment_source_sha256": source_sha,
                    "execution_source_sha256": execution_sources, "documentation_migration": proof,
                    "config_sha256": data.config_digest(config), "evaluation_context": config["evaluation_context"],
                    "no_inference_performed": True, "packager_source_sha256": sha256_file(__file__)}
        save_json(payload / MANIFEST_NAME, manifest)
        archive = output / BASENAME
        temporary = output / (BASENAME + ".tmp")
        with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
            for relative in sorted(expected_payload | {MANIFEST_NAME}):
                zipped.write(payload / relative, arcname=relative)
        with zipfile.ZipFile(temporary) as zipped:
            require(zipped.testzip() is None, "ZIP CRC verification failed.")
            require(len(zipped.namelist()) == len(set(zipped.namelist()))
                    and set(zipped.namelist()) == expected_payload | {MANIFEST_NAME}, "ZIP entries differ from whitelist.")
            for relative in expected_payload | {MANIFEST_NAME}:
                actual_bytes = zipped.read(relative)
                require(hashlib.sha256(actual_bytes).hexdigest() == sha256_file(payload / relative),
                        f"Archive readback differs: {relative}")
        require(source_identity() == source_sha and sha256_file(freeze_path) == inputs["selection_freeze"],
                "Frozen sources or selection receipt changed during packaging.")
        if execution_source_dir is not None:
            _, final_proof = documentation_proof(execution_source_dir, ROOT)
            compare_exact(proof, final_proof, "unchanged execution/deployment documentation proof")
        for name in MODELS:
            require(sha256_file(run_root / name / "best.pt") == inputs[name + "/best.pt"]
                    and sha256_file(run_root / name / "run.json") == inputs[name + "/run.json"],
                    "Original run inputs changed during packaging.")
        temporary.replace(archive)
        receipt = {"status": "completed", "completed_at": datetime.now(timezone.utc).isoformat(),
                   "archive": archive.name, "archive_bytes": archive.stat().st_size,
                   "archive_sha256": sha256_file(archive), "whitelist": sorted(expected_payload | {MANIFEST_NAME}),
                   "zip_crc_and_entry_hashes_passed": True, "checkpoint_checks": checks,
                   "input_sha256": inputs, "source_sha256": source_sha,
                   "execution_source_sha256": execution_sources,
                   "documentation_migration": proof,
                   "manifest_sha256": sha256_file(payload / MANIFEST_NAME), "inference_performed": False}
        save_json(receipt_path, receipt)
        return receipt
    except BaseException as error:
        save_json(receipt_path, {"status": "failed", "error": f"{type(error).__name__}: {error}", "input_sha256": inputs})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--selection-freeze", type=Path, required=True)
    parser.add_argument("--execution-source-dir", type=Path,
                        help="Original verified execution sources/config; enables strict documentary-only migration.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/releases", help="Must not exist.")
    args = parser.parse_args(argv)
    receipt = build_package(args.run_root, args.selection_freeze, args.output_dir, args.execution_source_dir)
    print(f"Private package verified: {receipt['archive']} ({receipt['archive_bytes']} bytes)")
    print(f"SHA256: {receipt['archive_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
