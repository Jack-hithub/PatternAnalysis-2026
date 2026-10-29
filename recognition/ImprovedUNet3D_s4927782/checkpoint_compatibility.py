"""Narrow, explicit inference compatibility for the raw MRI support correction.

The published checkpoint files retain their original bytes and legacy identity.
This module authorizes only those two files with an exact recorded runtime. It
does not claim that corrected inference code trained the frozen models.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_FILES = ("modules.py", "dataset.py", "utils.py", "train.py", "predict.py")
KIND = "inference_only_raw_support_fix"
SUPPORT_POLICY = "raw_nonzero_canonical_ras_nearest_grid_v1"
BASELINE_COMMIT = "aed3804"
BASELINE_SOURCES = {
    "modules.py": "a0b86adb6fd5810e5b65924f8cddac95ac4daa80cdf6a6f6492116a4450b0de9",
    "dataset.py": "faeac1458e9d7d1bd2e39443c9e1b13bca46cfd43840d5c21b8f877dea7ea9c9",
    "utils.py": "7fd2166db4ee173e3662e74306a607c80996cef20fd7b8245530505e3895cdf3",
    "train.py": "ca19f4aa6980f5fbbed149431a3d73af7658a85de87b4ef2ce51966b87ac0d1c",
    "predict.py": "c0c19d227b13c923e79d37d96f86353616a77ccdbedf1c2847b5ae2d9e8d2b12",
}
EXECUTION_SOURCES = {
    "modules.py": "61848f6aea1416c4a16e47c9de9a1686e2a56673d8fc43a0e3799579e0a6f878",
    "dataset.py": "c994dc31e525573ff13c4a11d91e8dd4dbebbcc0abcd9b1b0c0e208ea6d8ff09",
    "utils.py": BASELINE_SOURCES["utils.py"],
    "train.py": "2f1994f0857e622d44e8776efa7b9649c3b1a312625b2cb58cd39894c3f9b038",
    "predict.py": BASELINE_SOURCES["predict.py"],
}
CONFIG_FILE_SHA256 = "6006e0bc8e4b84e4e06e2405823fe3a3a76aea74627c476e981a076016a71d39"
CONFIG_SHA256 = "6b19d3627214e6301af834c2b7429b50fcb7672d68b2887f4aa2a5bcfb7e0d77"
EXECUTION_CONFIG_SHA256 = "461dd91c3ef3722857d0077e5e19b1bd68272daafaa10cf702224573f4281bc1"
FREEZE_SHA256 = "3f3b3b6b40b98cb25b900e1af9174861e247abf75afac19c5ae09b50fdc87c15"
PUBLIC_MANIFEST_SHA256 = "1da67e526ce916c6b1af863c1aff6ec809f644b54a618490d1da51fd8bc28d31"
WEIGHTS = {
    "unet2d": {
        "public_sha256": "d69a02a085a4748301962037bcc3cdd096a4055f873833c42b12a8771d89ad43",
        "public_bytes": 2263645,
        "frozen_sha256": "88b2ddbcaf69df71d61af7736c7e551d5d485c9673c339db47d2538e33f8009c",
        "state_sha256": "5345f695317afda4d1831bdfe5312065af01d1113a9047c6d7279bf4332fc128",
        "tensor_count": 36, "tensor_elements": 481830,
    },
    "improved3d": {
        "public_sha256": "ff65f8ca2daedd5c2e8100c1499b978cb309c9ce762a5f9342511f8c81c09c1e",
        "public_bytes": 6855349,
        "frozen_sha256": "92b4cf0a6480f6eee8ecfac029df4c9a04f5bac622923dba441e497b56608447",
        "state_sha256": "39f43b5e591a595f98eebb9624594c3cebc0979662748042903f08c71048af00",
        "tensor_count": 58, "tensor_elements": 1627634,
    },
}
POLICY = {
    "executable_inference_changed": True,
    "training_repeated": False,
    "legacy_checkpoint_bytes_changed": False,
    "legacy_checkpoint_metadata_changed": False,
    "scientific_config_changed": False,
    "architecture_and_metric_sources_changed": False,
    "training_provenance": "unchanged_original_execution",
    "support_policy": SUPPORT_POLICY,
}


def require(condition, message):
    if not condition:
        raise ValueError("Checkpoint compatibility: " + message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_json(path):
    # Duplicate keys must not create an ambiguous compatibility declaration.
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=unique)


def current_source_identity(root=ROOT):
    return {name: sha256_file(Path(root) / name) for name in SOURCE_FILES}


def legacy_identity(*, execution=False):
    return {
        "manifest_sha256": "c6481dc712ea1ceac52086b7a469365d2ae5302e61e41505641b3c0032953546",
        "config_sha256": EXECUTION_CONFIG_SHA256 if execution else CONFIG_SHA256,
        "trainval_receipt_sha256": "c1b42eaff02b051bbefdbf69c1c3d785bc4e9bdb3c250c42dd90cb82609e83e4",
        "source_sha256": dict(EXECUTION_SOURCES if execution else BASELINE_SOURCES),
    }


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().contiguous().cpu()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def compare_exact(original, copied, location="checkpoint"):
    """Compare tensors and all metadata; never alter either checkpoint object."""
    import torch
    if isinstance(original, torch.Tensor):
        require(isinstance(copied, torch.Tensor) and original.dtype == copied.dtype
                and original.shape == copied.shape and torch.equal(original, copied),
                f"tensor differs: {location}")
    elif isinstance(original, Mapping):
        require(isinstance(copied, Mapping) and set(original) == set(copied),
                f"metadata keys differ: {location}")
        for key, value in original.items():
            compare_exact(value, copied[key], f"{location}/{key}")
    elif isinstance(original, (list, tuple)):
        require(type(original) is type(copied) and len(original) == len(copied),
                f"metadata sequence differs: {location}")
        for index, value in enumerate(original):
            compare_exact(value, copied[index], f"{location}/{index}")
    else:
        require(type(original) is type(copied) and original == copied,
                f"metadata differs: {location}")


def checkpoint_entry(checkpoint):
    """Expected declaration derived from an independently SHA-pinned public file."""
    model = checkpoint.get("model")
    require(model in WEIGHTS and checkpoint.get("schema_version") == 1,
            "unknown legacy model/schema")
    pin = WEIGHTS[model]
    require(checkpoint["identity"] == legacy_identity(), "legacy identity differs")
    require(checkpoint.get("execution_identity") == legacy_identity(execution=True),
            "original training identity differs")
    require(json_digest(checkpoint["config"]) == CONFIG_SHA256, "legacy config differs")
    migration = checkpoint.get("documentation_migration", {})
    require(migration.get("original_selected_checkpoint_sha256") == pin["frozen_sha256"],
            "original frozen checkpoint provenance differs")
    state = checkpoint["state_dict"]
    require(len(state) == pin["tensor_count"]
            and sum(value.numel() for value in state.values()) == pin["tensor_elements"]
            and state_digest(state) == pin["state_sha256"], "pinned model tensors differ")
    return {
        "model": model, **pin,
        "legacy_identity": deepcopy(checkpoint["identity"]),
        "original_training_identity": deepcopy(checkpoint["execution_identity"]),
        "legacy_metadata_sha256": json_digest({k: v for k, v in checkpoint.items()
                                               if k != "state_dict"}),
        "original_history_sha256": migration["original_full_history_sha256"],
        "legacy_history_sha256": json_digest(checkpoint["history"]),
        "original_training_runtime_sha256": json_digest(checkpoint["runtime"]),
        "selected_epoch": checkpoint["best_epoch"],
        "selected_validation_prostate_dice": checkpoint["best_score"],
        "all_tensors_equal_to_original_frozen": True,
        "public_checkpoint_bytes_unchanged": True,
    }


def profile_content(profile):
    return {key: value for key, value in profile.items() if key != "profile_content_sha256"}


def validate_profile(profile, current_sources, root=ROOT):
    require(isinstance(profile, dict), "profile must be an object")
    require(set(profile) == {"schema_version", "kind", "baseline_commit", "policy",
                            "runtime_source_sha256", "helper_sha256", "config_sha256",
                            "config_file_sha256", "selection_freeze_sha256",
                            "public_package_manifest_sha256", "checkpoints",
                            "profile_content_sha256"}, "profile fields differ")
    require(type(profile["schema_version"]) is int and profile["schema_version"] == 1 and profile["kind"] == KIND
            and profile["baseline_commit"] == BASELINE_COMMIT, "profile kind/baseline differs")
    require(profile["profile_content_sha256"] == json_digest(profile_content(profile)),
            "profile content digest differs")
    require(json_digest(profile["policy"]) == json_digest(POLICY), "profile policy differs")
    require(profile["config_sha256"] == CONFIG_SHA256
            and profile["config_file_sha256"] == CONFIG_FILE_SHA256
            and sha256_file(Path(root) / "config.json") == CONFIG_FILE_SHA256,
            "runtime configuration bytes differ")
    require(profile["selection_freeze_sha256"] == FREEZE_SHA256
            and profile["public_package_manifest_sha256"] == PUBLIC_MANIFEST_SHA256,
            "profile original evidence anchors differ")
    require(isinstance(current_sources, dict) and set(current_sources) == set(SOURCE_FILES),
            "exact five-source identity required")
    require(current_sources == current_source_identity(root)
            and profile["runtime_source_sha256"] == current_sources,
            "runtime source hashes differ from explicit profile")
    for name in ("modules.py", "utils.py"):
        require(current_sources[name] == BASELINE_SOURCES[name],
                f"architecture/metric source differs from AED baseline: {name}")
    require(profile["helper_sha256"] == sha256_file(Path(__file__).resolve()),
            "compatibility helper differs from explicit profile")
    require(isinstance(profile["checkpoints"], dict)
            and set(profile["checkpoints"]) == set(WEIGHTS), "profile model set differs")


def validate_compatibility(path, checkpoint, config, current_sources, profile_path=None):
    """Authorize one exact legacy file for the explicitly profiled fixed runtime.

    Call only after the ordinary exact source check fails. This path never
    rewrites identity, bypasses configuration checks, or authorizes retraining.
    A missing/altered profile, source, helper, file, tensor, or identity fails.
    """
    import torch
    selected_profile = ROOT / "inference_compatibility.json" if profile_path is None else Path(profile_path)
    require(selected_profile.is_file(), "explicit inference compatibility profile missing")
    profile = read_json(selected_profile)
    validate_profile(profile, current_sources, ROOT)
    require(isinstance(checkpoint, dict) and checkpoint.get("model") in WEIGHTS,
            "unknown checkpoint model")
    model = checkpoint["model"]
    pin = WEIGHTS[model]
    path = Path(path)
    require(path.is_file() and path.stat().st_size == pin["public_bytes"]
            and sha256_file(path) == pin["public_sha256"], "legacy checkpoint file bytes differ")
    require(config == checkpoint.get("config") and json_digest(config) == CONFIG_SHA256,
            "scientific configuration differs")
    disk_checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    compare_exact(disk_checkpoint, checkpoint)
    require(json_digest(profile["checkpoints"][model]) == json_digest(checkpoint_entry(disk_checkpoint)),
            "profile legacy checkpoint declaration differs")
    # The other model is pinned structurally even when only this model is loaded.
    for name, other in profile["checkpoints"].items():
        require(isinstance(other, dict) and all(other.get(key) == value
                for key, value in WEIGHTS[name].items()), "profile weight anchors differ")
        require(other.get("legacy_identity") == legacy_identity()
                and other.get("original_training_identity") == legacy_identity(execution=True),
                "profile original training/legacy identity differs")
    return {
        "kind": KIND, "profile_sha256": sha256_file(selected_profile),
        "checkpoint_sha256": pin["public_sha256"], "model": model,
        "legacy_identity": deepcopy(checkpoint["identity"]),
        "original_training_identity": deepcopy(checkpoint["execution_identity"]),
        "original_frozen_checkpoint_sha256": pin["frozen_sha256"],
        "runtime_source_sha256": dict(current_sources),
        "helper_sha256": profile["helper_sha256"], "support_policy": SUPPORT_POLICY,
        "training_provenance": "unchanged", "configuration": "unchanged",
        "checkpoint_tensors": "unchanged", "checkpoint_bytes": "unchanged",
    }
