"""Build a fresh, private inference package from the completed fixed study.

Only optimizer/scaler training state is removed. Selected weights and every
remaining checkpoint field are compared exactly after a safe CPU reload. This
utility neither runs inference nor publishes an archive.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
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

BASENAME = "demo3_hipmri_rebuild_seed3711.zip"
REQUIRED_FIELDS = {"schema_version", "model", "config", "identity", "runtime", "state_dict",
                   "epoch", "best_epoch", "best_score", "history"}
SOURCE_WHITELIST = (*SOURCE_FILES, "config.json", "label_schema.json", "patient_split.json",
                    "requirements.txt", "requirements-gpu.txt", "LICENSE.txt",
                    "tools/predict_and_report.py")
WEIGHT_NAMES = {name: f"weights/{name}_best.pt" for name in MODELS}
GENERATED_NAMES = (*WEIGHT_NAMES.values(), "run_summary.json", "INFERENCE_INSTRUCTIONS.md")
MANIFEST_NAME = "PACKAGE_MANIFEST.json"


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


def instructions():
    return """# HipMRI fixed-run inference package

Both selected models completed 30 epochs with seed3711. Selection used only the
seven scheduled VAL assessments; each selected epoch is recorded in the manifest.
This cohort was explored during earlier development, so reported TEST results
must not be described as a new independent blind test. This package does not
contain MRI files, labels, prediction arrays, caches or example images.

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
only optimizer/scaler state was removed. Every remaining field and every model
tensor was verified exactly after a weights-only CPU reload. No inference or
new experiment was used to make this package. The pruned files support inference,
not resuming optimizer training. Training's original five-source hash record and
runtime/commit metadata are retained inside each checkpoint.
"""


def build_package(run_root, selection_freeze, output_dir):
    run_root = Path(run_root).resolve(strict=True)
    freeze_path = Path(selection_freeze).resolve(strict=True)
    output = Path(output_dir).absolute()
    require(not output.exists() and not output.is_symlink(), "Use a fresh package output directory.")
    config = load_json(ROOT / "config.json")
    frozen = load_json(freeze_path)
    source_sha = source_identity()
    require(frozen.get("status") == "frozen" and set(frozen.get("models", {})) == set(MODELS),
            "A completed two-model selection freeze is required.")
    require(frozen.get("config_sha256") == data.config_digest(config)
            and frozen.get("source_sha256") == source_sha, "Frozen source/config identity differs.")
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
            original = load_selected_checkpoint(original_path, config)
            choice = frozen["models"][name]
            require(run.get("status") == "completed" and run.get("completed_epochs") == config["epochs"]
                    and run.get("model") == name and original["model"] == name
                    and run.get("config") == config and run.get("identity") == original["identity"],
                    f"Completed run identity differs: {name}")
            require(choice.get("completed_epochs") == config["epochs"]
                    and choice.get("selected_epoch") == original["epoch"] == run.get("best_epoch")
                    and run.get("best_checkpoint_sha256") == inputs[name + "/best.pt"]
                    and original["identity"]["manifest_sha256"] == frozen["manifest_sha256"]
                    and original["best_score"] == choice["validation_prostate_dice"],
                    f"Selected epoch/weight/data/score differs: {name}")
            require(set(original) == REQUIRED_FIELDS | {"optimizer", "scaler"},
                    "Unexpected checkpoint fields; refuse to package unreviewed contents.")
            cleaned = {key: value for key, value in original.items() if key not in ("optimizer", "scaler")}
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
            accepted = load_selected_checkpoint(destination, config)
            model = create_model(name, config)
            model.load_state_dict(accepted["state_dict"], strict=True)
            require(state_digest(model.state_dict()) == state_digest(original["state_dict"]),
                    f"Instantiated model tensor identity differs: {name}")
            tensor_count = len(original["state_dict"])
            tensor_elements = sum(value.numel() for value in original["state_dict"].values())
            checks[name] = {"original_selected_sha256": inputs[name + "/best.pt"],
                           "inference_path": WEIGHT_NAMES[name], "inference_sha256": sha256_file(destination),
                           "removed_fields": ["optimizer", "scaler"], "preserved_fields": sorted(REQUIRED_FIELDS),
                           "all_remaining_metadata_exact": True, "all_state_tensors_exact": True,
                           "state_tensor_count": tensor_count, "state_tensor_elements": tensor_elements,
                           "state_sha256": state_digest(original["state_dict"]),
                           "selected_epoch": original["epoch"], "strict_model_load_passed": True,
                           "selected_checkpoint_source_guard_passed": True}
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
            "source_sha256": source_sha, "source_execution_commit": summary["unet2d"]["runtime"]["source_commit"],
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
                    "checkpoint_checks": checks, "original_source_sha256": source_sha,
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
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/releases", help="Must not exist.")
    args = parser.parse_args(argv)
    receipt = build_package(args.run_root, args.selection_freeze, args.output_dir)
    print(f"Private package verified: {receipt['archive']} ({receipt['archive_bytes']} bytes)")
    print(f"SHA256: {receipt['archive_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
