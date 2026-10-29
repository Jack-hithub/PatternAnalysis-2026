"""Build a new inference-only compatibility profile after runtime source review.

Both previously public checkpoint files are read unchanged, then all 94 state
tensors are compared with the actual, independently SHA-pinned frozen training
checkpoints. This executable support correction never rebinds training identity.
No profile is overwritten and no checkpoint is written.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import checkpoint_compatibility as compatibility


def construct_profile(*, root=ROOT, public_payload=None, run_root=None, freeze_path=None):
    root = Path(root).resolve(strict=True)
    public = (root / "artifacts/releases/current_study_v2/payload"
              if public_payload is None else Path(public_payload).resolve(strict=True))
    run_root = root / "artifacts/fixed_run" if run_root is None else Path(run_root).resolve(strict=True)
    freeze_path = run_root / "selection_freeze.json" if freeze_path is None else Path(freeze_path)
    require = compatibility.require
    sources = compatibility.current_source_identity(root)
    helper_sha = compatibility.sha256_file(compatibility.__file__)
    require(compatibility.sha256_file(freeze_path) == compatibility.FREEZE_SHA256,
            "selection freeze evidence bytes differ")
    freeze = compatibility.read_json(freeze_path)
    require(freeze.get("status") == "frozen" and set(freeze.get("models", {})) == set(compatibility.WEIGHTS),
            "original model selection is not frozen")
    manifest_path = public / "PACKAGE_MANIFEST.json"
    require(compatibility.sha256_file(manifest_path) == compatibility.PUBLIC_MANIFEST_SHA256,
            "original public package manifest bytes differ")
    manifest = compatibility.read_json(manifest_path)
    entries = {}
    for model, pin in compatibility.WEIGHTS.items():
        public_path = public / "weights" / f"{model}_best.pt"
        require(public_path.stat().st_size == pin["public_bytes"]
                and compatibility.sha256_file(public_path) == pin["public_sha256"],
                f"public {model} checkpoint bytes differ")
        require(manifest["entries"][f"weights/{model}_best.pt"]
                == {"sha256": pin["public_sha256"], "bytes": pin["public_bytes"]},
                f"public {model} manifest anchor differs")
        frozen_path = run_root / model / "best.pt"
        require(compatibility.sha256_file(frozen_path) == pin["frozen_sha256"]
                and freeze["models"][model]["checkpoint_sha256"] == pin["frozen_sha256"],
                f"actual frozen {model} checkpoint bytes differ")
        public_checkpoint = torch.load(public_path, map_location="cpu", weights_only=True)
        original = torch.load(frozen_path, map_location="cpu", weights_only=True)
        entries[model] = compatibility.checkpoint_entry(public_checkpoint)
        require(original["model"] == model and original["schema_version"] == 1,
                "actual frozen model/schema differs")
        compatibility.compare_exact(original["identity"], public_checkpoint["execution_identity"],
                                    "preserved original training identity")
        compatibility.compare_exact(original["runtime"], public_checkpoint["runtime"],
                                    "preserved original training runtime")
        compatibility.compare_exact(original["state_dict"], public_checkpoint["state_dict"],
                                    "every public tensor versus actual frozen original")
        require(compatibility.state_digest(original["state_dict"]) == pin["state_sha256"],
                "actual frozen state digest differs")
        require(compatibility.json_digest(original["history"]) == entries[model]["original_history_sha256"],
                "actual original training history provenance differs")
        require(original["epoch"] == original["best_epoch"] == public_checkpoint["epoch"]
                == public_checkpoint["best_epoch"] == freeze["models"][model]["selected_epoch"]
                and original["best_score"] == public_checkpoint["best_score"]
                == freeze["models"][model]["validation_prostate_dice"],
                "original selected epoch/score differs")
        require(compatibility.sha256_file(public_path) == pin["public_sha256"]
                and compatibility.sha256_file(frozen_path) == pin["frozen_sha256"],
                "checkpoint changed during proof")
    profile = {
        "schema_version": 1, "kind": compatibility.KIND,
        "baseline_commit": compatibility.BASELINE_COMMIT,
        "policy": dict(compatibility.POLICY), "runtime_source_sha256": sources,
        "helper_sha256": helper_sha, "config_sha256": compatibility.CONFIG_SHA256,
        "config_file_sha256": compatibility.CONFIG_FILE_SHA256,
        "selection_freeze_sha256": compatibility.FREEZE_SHA256,
        "public_package_manifest_sha256": compatibility.PUBLIC_MANIFEST_SHA256,
        "checkpoints": entries,
    }
    profile["profile_content_sha256"] = compatibility.json_digest(profile)
    compatibility.validate_profile(profile, sources, root)
    require(compatibility.current_source_identity(root) == sources
            and compatibility.sha256_file(compatibility.__file__) == helper_sha,
            "runtime changed during profile construction; rebuild after sources are stable")
    return profile


def build_profile(output_path, **kwargs):
    output = Path(output_path)
    compatibility.require(not output.exists() and not output.is_symlink(),
                          "use a fresh profile path; never overwrite a profile")
    profile = construct_profile(**kwargs)
    import json
    with output.open("x", encoding="utf-8") as stream:
        json.dump(profile, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "inference_compatibility.json")
    parser.add_argument("--public-payload", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--selection-freeze", type=Path)
    args = parser.parse_args()
    profile = build_profile(args.output, public_payload=args.public_payload,
                            run_root=args.run_root, freeze_path=args.selection_freeze)
    print(f"Created {args.output}: {sum(row['tensor_count'] for row in profile['checkpoints'].values())} "
          "unchanged tensors; original checkpoint bytes and training provenance retained.")


if __name__ == "__main__":
    main()
