"""Patient-disjoint HipMRI preprocessing, immutable caches and TRAIN sampling.

The geometry, image-only normalisation and centre-aware resampling adapt the
author's previously audited prototype. This standalone implementation has no
dependency on its checkpoints or historical source files. Images are canonical
RAS, arrays use D,H,W, and affines always map X,Y,Z voxel centres to world space.
Official labels: CSIRO v2, https://doi.org/10.25919/45t8-p065.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import nibabel as nib
from scipy.ndimage import affine_transform, zoom
try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch, Dataset = None, object


SPLITS = ("train", "validation", "test")
CLASS_NAMES = ["Background", "Body", "Bones", "Bladder", "Rectum", "Prostate"]
SCAN_NAME = re.compile(r"^(?P<patient>[A-Z]\d{3})_(?P<visit>Week\d+)_(?P<kind>LFOV|SEMANTIC)\.nii(?:\.gz)?$")
SCHEMA = "hipmri_standalone_rebuild_cache_v1"


def config_digest(config: dict) -> str:
    """Canonical scientific configuration identity, independent of indentation."""
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict, *, replace=False) -> None:
    if path.exists() and not replace:
        raise FileExistsError(f"Refusing to overwrite {path}")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def _shape(values: Sequence[int]) -> tuple[int, int, int]:
    if len(values) != 3 or any(type(v) is not int or v <= 0 for v in values):
        raise ValueError("A grid must contain three positive integer D,H,W sizes.")
    return tuple(values)


def _safe(root: Path, relative: str) -> Path:
    name = Path(relative)
    if name.is_absolute() or ".." in name.parts or not name.parts:
        raise ValueError("Expected a relative path within its declared root.")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Path escapes its declared root.")
    return path


def validate_patient_splits(splits: dict, patients=None) -> None:
    if not isinstance(splits, dict) or set(splits) != set(SPLITS):
        raise ValueError("Expected TRAIN/VAL/TEST patient partitions.")
    seen = set()
    for name in SPLITS:
        values = splits[name]
        if (not isinstance(values, list) or not values
                or any(not isinstance(p, str) or re.fullmatch(r"[A-Z]\d{3}", p) is None for p in values)
                or len(values) != len(set(values)) or seen.intersection(values)):
            raise ValueError("Patient partitions contain invalid, duplicate or overlapping IDs.")
        seen.update(values)
    if patients is not None and seen != set(patients):
        raise ValueError("Patient partitions do not cover exactly the discovered patients.")


def discover_pairs(raw_root: str | Path, split_data: dict) -> list[dict]:
    """Inventory names only; this does not open TEST headers or voxel arrays."""
    raw_root = Path(raw_root).expanduser().resolve(strict=True)
    groups = {}
    for folder, kind, role in [("semantic_MRs", "LFOV", "source_image"),
                               ("semantic_labels_only", "SEMANTIC", "source_mask")]:
        directory = raw_root / folder
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or not path.name.lower().endswith((".nii", ".nii.gz")):
                continue
            match = SCAN_NAME.fullmatch(path.name)
            if match is None or match["kind"] != kind:
                raise ValueError(f"Unexpected HipMRI filename in {folder}: {path.name}")
            if not path.resolve().is_relative_to(raw_root):
                raise ValueError("Raw input escapes the data root.")
            key = match["patient"], match["visit"]
            item = groups.setdefault(key, {})
            if role in item:
                raise ValueError(f"Duplicate {role} for {key}")
            item[role] = path.relative_to(raw_root).as_posix()
    if not groups:
        raise ValueError("No HipMRI pairs discovered.")
    splits = split_data["patient_splits"]
    validate_patient_splits(splits, [p for p, _ in groups])
    memberships = {p: name for name in SPLITS for p in splits[name]}
    records = []
    for (patient, visit), source in sorted(groups.items()):
        if set(source) != {"source_image", "source_mask"}:
            raise ValueError(f"Missing image or mask for {patient}/{visit}")
        partition = memberships[patient]
        records.append({"patient": patient, "visit": visit, "split": partition,
                        "cache_path": f"volumes/{patient}_{visit}.npz",
                        "pending_test": partition == "test", **source})
    actual_counts = {name: sum(r["split"] == name for r in records) for name in SPLITS}
    if split_data.get("scan_counts", actual_counts) != actual_counts:
        raise ValueError(f"Discovered scan counts differ from the fixed split: {actual_counts}")
    actual_patients = {name: len(splits[name]) for name in SPLITS}
    if split_data.get("patient_counts", actual_patients) != actual_patients:
        raise ValueError("Patient count metadata differs from its fixed assignments.")
    return records


def _grid(affine_xyz, old_shape_xyz, target_shape_dhw):
    target_xyz = np.asarray(_shape(target_shape_dhw)[::-1], dtype=np.float64)
    old_shape = np.asarray(old_shape_xyz, dtype=np.float64)
    affine = np.asarray(affine_xyz, dtype=np.float64)
    if (old_shape.shape != (3,) or np.any(old_shape < 1) or affine.shape != (4, 4)
            or not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-12):
        raise ValueError("Invalid source image geometry.")
    scale = old_shape / target_xyz
    transform = np.eye(4)
    transform[:3, :3] = np.diag(scale)
    transform[:3, 3] = .5 * (scale - 1)
    return affine @ transform, target_xyz / old_shape


def _check_header_spacing(volume):
    spacing = np.asarray(volume.header.get_zooms()[:3], dtype=np.float64)
    affine = np.asarray(volume.affine, dtype=np.float64)
    if (spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0)
            or affine.shape != (4, 4) or not np.isfinite(affine).all()
            or not np.allclose(spacing, np.linalg.norm(affine[:3, :3], axis=0), rtol=1e-5, atol=1e-5)):
        raise ValueError("Image or mask header spacing disagrees with its physical affine.")
    return spacing


def preprocess_image(path: str | Path, target_shape_dhw: Sequence[int]):
    """Read MRI only: per-case nonzero z-score, clip and linear resampling.

    Background support is transformed by the matching nearest-neighbour map.
    No label, class location or patient-level pooled statistic is consulted.
    """
    target = _shape(target_shape_dhw)
    original = nib.load(str(path))
    if len(original.shape) != 3:
        raise ValueError("MRI must be three-dimensional.")
    _check_header_spacing(original)
    canonical = nib.as_closest_canonical(original)
    affine = np.asarray(canonical.affine, dtype=np.float64)
    if nib.aff2axcodes(affine) != ("R", "A", "S"):
        raise ValueError("Canonical MRI must have RAS orientation.")
    header_spacing = np.asarray(canonical.header.get_zooms()[:3], dtype=np.float64)
    if (header_spacing.shape != (3,) or not np.isfinite(header_spacing).all()
            or np.any(header_spacing <= 0)
            or not np.allclose(header_spacing, np.linalg.norm(affine[:3, :3], axis=0), rtol=1e-5, atol=1e-5)):
        raise ValueError("MRI header spacing must be positive and agree with its affine.")
    new_affine, factors = _grid(affine, canonical.shape, target)
    image = canonical.get_fdata(dtype=np.float32, caching="unchanged")
    if not np.isfinite(image).all():
        raise ValueError("MRI contains nonfinite voxel values.")
    support = image != 0
    count = int(np.count_nonzero(support))
    mean = float(image[support].mean(dtype=np.float64)) if count else 0.0
    std = float(image[support].std(dtype=np.float64)) if count else 0.0
    normal = np.zeros(image.shape, dtype=np.float32)
    if count:
        normal[support] = np.clip((image[support].astype(np.float64) - mean)
                                  / max(std, 1e-8), -5, 5).astype(np.float32)
    opts = dict(grid_mode=True, mode="nearest", prefilter=False)
    resized = zoom(normal, factors, order=1, **opts)
    resized_support = zoom(support.astype(np.uint8), factors, order=0, **opts).astype(bool)
    resized[~resized_support] = 0
    if resized.shape != target[::-1]:
        raise ValueError("Resampled MRI has the wrong shape.")
    spacing_xyz = np.linalg.norm(new_affine[:3, :3], axis=0)
    metadata = {"original_shape_xyz": list(original.shape),
                "original_affine_xyz": np.asarray(original.affine).tolist(),
                "canonical_shape_xyz": list(canonical.shape),
                "canonical_affine_xyz": affine.tolist(),
                "canonical_spacing_xyz": header_spacing.tolist(),
                "resampled_affine_xyz": new_affine.tolist(),
                "target_shape_dhw": list(target),
                "spacing_dhw": spacing_xyz[::-1].tolist(),
                "normalization": {"method": "per_image_nonzero_zscore_then_clip",
                                  "nonzero_count": count, "mean": mean, "std": std,
                                  "minimum_scale": 1e-8, "clip": [-5, 5]},
                "resampling": {"image_order": 1, "label_order": 0, **opts},
                "native_resolution_accuracy_claimed": False}
    return np.ascontiguousarray(resized.transpose(2, 1, 0), dtype=np.float32), metadata


def _label_values(volume):
    values = np.asanyarray(volume.dataobj)
    if (values.ndim != 3 or not np.isfinite(values).all()
            or np.any(values != np.rint(values)) or values.min() < 0 or values.max() > 5):
        raise ValueError("Mask must contain finite integer class IDs 0-5.")
    return values.astype(np.uint8)


def resample_label(label_dhw, from_affine_xyz, to_affine_xyz, to_shape_dhw):
    """Map categorical labels by physical nearest-neighbour sampling.

    This is also used to restore native geometry; it cannot restore detail lost
    by downsampling, and it never changes a decision using a true target mask.
    Grid-constant boundaries retain each edge voxel's half-voxel support while
    returning background for locations outside the source field of view.
    """
    values = np.asarray(label_dhw)
    if (values.ndim != 3 or not np.isfinite(values).all()
            or np.any(values != np.rint(values)) or values.min() < 0 or values.max() > 255):
        raise ValueError("Expected a finite categorical D,H,W array.")
    shape = _shape(to_shape_dhw)
    first, second = np.asarray(from_affine_xyz, dtype=float), np.asarray(to_affine_xyz, dtype=float)
    if (first.shape != (4, 4) or second.shape != (4, 4)
            or not np.isfinite(first).all() or not np.isfinite(second).all()):
        raise ValueError("Affines must be finite 4x4 arrays.")
    transform = np.linalg.solve(first, second)
    result = affine_transform(values.transpose(2, 1, 0), transform[:3, :3],
                              offset=transform[:3, 3], output_shape=shape[::-1],
                              order=0, mode="grid-constant", cval=0, prefilter=False)
    return np.ascontiguousarray(result.transpose(2, 1, 0), dtype=np.uint8)


def _prepare_one(raw_root, out_dir, record, config):
    root, destination = Path(raw_root), _safe(Path(out_dir), record["cache_path"])
    if destination.exists():
        raise FileExistsError(destination)
    image_path, mask_path = _safe(root, record["source_image"]), _safe(root, record["source_mask"])
    source_sha = {"image": _sha(image_path), "mask": _sha(mask_path)}
    image, metadata = preprocess_image(image_path, config["target_shape_dhw"])
    original_mask = nib.load(str(mask_path))
    _check_header_spacing(original_mask)
    canonical_mask = nib.as_closest_canonical(original_mask)
    if (list(canonical_mask.shape) != metadata["canonical_shape_xyz"]
            or not np.allclose(canonical_mask.affine, metadata["canonical_affine_xyz"], rtol=1e-5, atol=1e-5)):
        raise ValueError("Canonical image and mask grids differ.")
    if not np.allclose(canonical_mask.header.get_zooms()[:3], metadata["canonical_spacing_xyz"], rtol=1e-5, atol=1e-5):
        raise ValueError("Image and mask header spacings differ.")
    labels = _label_values(canonical_mask)
    aff96, factors96 = _grid(canonical_mask.affine, labels.shape, config["target_shape_dhw"])
    aff64, factors64 = _grid(canonical_mask.affine, labels.shape, config["evaluation_shape_dhw"])
    options = dict(grid_mode=True, mode="nearest", prefilter=False)
    mask = np.ascontiguousarray(zoom(labels, factors96, order=0, **options).transpose(2, 1, 0))
    mask_eval = np.ascontiguousarray(zoom(labels, factors64, order=0, **options).transpose(2, 1, 0))
    spacing = np.linalg.norm(aff96[:3, :3], axis=0)[::-1].astype(np.float32)
    eval_spacing = np.linalg.norm(aff64[:3, :3], axis=0)[::-1].astype(np.float32)
    if source_sha != {"image": _sha(image_path), "mask": _sha(mask_path)}:
        raise RuntimeError("Raw image or mask changed while preparing its cache.")
    metadata.update(source_sha256=source_sha, original_mask_shape_xyz=list(original_mask.shape),
                    original_mask_affine_xyz=np.asarray(original_mask.affine).tolist())
    payload = dict(image=image, mask=mask, mask_eval=mask_eval, affine_xyz=aff96,
                   affine_eval_xyz=aff64, spacing_dhw=spacing, eval_spacing_dhw=eval_spacing,
                   metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + ".tmp")
    with temp.open("xb") as stream:
        np.savez_compressed(stream, **payload)
    temp.replace(destination)
    return {**record, "cache_sha256": _sha(destination), "source_sha256": source_sha,
            "class_voxel_counts": np.bincount(mask.ravel(), minlength=6).tolist(),
            "evaluation_class_voxel_counts": np.bincount(mask_eval.ravel(), minlength=6).tolist(),
            "cache_size_bytes": destination.stat().st_size}


def _prepare_records(raw_root, output, records, config, workers):
    results = []
    if workers == 1:
        iterator = (_prepare_one(str(raw_root), str(output), r, config) for r in records)
        for result in iterator:
            results.append(result)
            print(f"Prepared {len(results)}/{len(records)} {result['patient']}/{result['visit']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_prepare_one, str(raw_root), str(output), r, config) for r in records]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"Prepared {len(results)}/{len(records)} {result['patient']}/{result['visit']}", flush=True)
    return sorted(results, key=lambda r: (r["patient"], r["visit"]))


def _validate_freeze(path, manifest_path, config):
    if path is None:
        raise ValueError("TEST preparation requires a real completed pair selection freeze.")
    path = Path(path).expanduser().resolve(strict=True)
    value = _json(path)
    if (value.get("status") != "frozen" or value.get("manifest_sha256") != _sha(manifest_path)
            or value.get("config_sha256") != config_digest(config)
            or set(value.get("models", {})) != {"unet2d", "improved3d"}):
        raise ValueError("Selection freeze is not bound to this cache/config and both models.")
    for choice in value["models"].values():
        relative = choice.get("checkpoint_path")
        if (choice.get("completed_epochs") != config["epochs"] or not isinstance(relative, str)
                or not re.fullmatch(r"[0-9a-f]{64}", choice.get("checkpoint_sha256", ""))):
            raise ValueError("Incomplete selected model metadata.")
        checkpoint = Path(relative).expanduser()
        checkpoint = checkpoint if checkpoint.is_absolute() else path.parent / checkpoint
        if not checkpoint.is_file() or _sha(checkpoint) != choice["checkpoint_sha256"]:
            raise ValueError("Selected checkpoint bytes changed or are unavailable.")
    return _sha(path)


def prepare_cache(raw_root, split_path, config, out_dir, stage="trainval", workers=4,
                  selection_freeze=None):
    """Prepare a fresh TRAIN/VAL cache, then later fill TEST without editing it.

    The first immutable manifest includes TEST filename identities but no TEST
    array, label count, geometry or source-file hash. Completion receipts bind
    all actual prepared bytes; TEST can be loaded only after its receipt commits.
    """
    if stage not in {"trainval", "test"} or type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError("Use trainval/test and 1-16 preparation workers.")
    _shape(config["target_shape_dhw"]); _shape(config["evaluation_shape_dhw"])
    root = Path(raw_root).expanduser().resolve(strict=True)
    split_path = Path(split_path).expanduser().resolve(strict=True)
    output = Path(out_dir).expanduser().absolute()
    manifest_path = output / "manifest.json"
    split_data = _json(split_path)
    if stage == "trainval":
        records = discover_pairs(root, split_data)
        if output.exists() or output.is_symlink():
            raise FileExistsError("Use a fresh cache directory; existing evidence is not overwritten.")
        output.mkdir(parents=True, exist_ok=False)
        receipt_path = output / "trainval_receipt.json"
        receipt = {"status": "preparing", "stage": stage, "started_at": datetime.now(timezone.utc).isoformat(),
                   "config_sha256": config_digest(config), "split_sha256": _sha(split_path),
                   "test_arrays_read": 0}
        _write_json(receipt_path, receipt)
        try:
            prepared = _prepare_records(root, output, [r for r in records if r["split"] != "test"], config, workers)
            by_key = {(r["patient"], r["visit"]): r for r in prepared}
            records = [by_key.get((r["patient"], r["visit"]), r) for r in records]
            manifest = {"schema": SCHEMA, "schema_version": 1, "num_classes": 6,
                        "label_ids": list(range(6)), "class_names": CLASS_NAMES,
                        "patient_splits": split_data["patient_splits"], "records": records,
                        "config": config, "config_sha256": config_digest(config),
                        "split_sha256": _sha(split_path), "target_shape_dhw": config["target_shape_dhw"],
                        "evaluation_shape_dhw": config["evaluation_shape_dhw"],
                        "source_scope": "TRAIN/VAL images and labels; TEST filenames only before selection freeze",
                        "evaluation_context": config.get("evaluation_context", ""),
                        "preprocessing": "image-only per-case nonzero z-score; RAS; centre-aware linear/nearest grids"}
            _write_json(manifest_path, manifest)
            receipt.update(status="completed", manifest_sha256=_sha(manifest_path), records=prepared,
                           prepared_records=len(prepared),
                           train_class_voxel_counts=np.sum([r["class_voxel_counts"] for r in prepared if r["split"] == "train"], axis=0).astype(np.int64).tolist())
            _write_json(receipt_path, receipt, replace=True)
        except Exception as error:
            receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
            _write_json(receipt_path, receipt, replace=True)
            raise
    else:
        manifest = load_manifest(manifest_path)
        if manifest["config_sha256"] != config_digest(config) or manifest["split_sha256"] != _sha(split_path):
            raise ValueError("Config or patient split differs from the prepared cache.")
        freeze_digest = _validate_freeze(selection_freeze, manifest_path, config)
        receipt_path = output / "test_completion_receipt.json"
        receipt = {"status": "preparing", "stage": stage, "manifest_sha256": _sha(manifest_path),
                   "config_sha256": config_digest(config), "selection_freeze_sha256": freeze_digest,
                   "started_at": datetime.now(timezone.utc).isoformat()}
        _write_json(receipt_path, receipt)
        before = manifest_path.read_bytes()
        try:
            prepared = _prepare_records(root, output, records_for_split(manifest_path, "test"), config, workers)
            if manifest_path.read_bytes() != before:
                raise RuntimeError("Immutable cache manifest changed during TEST completion.")
            receipt.update(status="completed", records=prepared, prepared_records=len(prepared))
            _write_json(receipt_path, receipt, replace=True)
        except Exception as error:
            receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
            _write_json(receipt_path, receipt, replace=True)
            raise
    return manifest_path


def load_manifest(path):
    path = Path(path).expanduser().resolve(strict=True)
    value = _json(path)
    if (value.get("schema") != SCHEMA or value.get("label_ids") != list(range(6))
            or value.get("num_classes") != 6 or value.get("config_sha256") != config_digest(value.get("config", {}))):
        raise ValueError("Invalid standalone cache manifest or configuration identity.")
    validate_patient_splits(value["patient_splits"])
    _shape(value["target_shape_dhw"]); _shape(value["evaluation_shape_dhw"])
    seen = set()
    for r in value["records"]:
        key = r["patient"], r["visit"]
        if (key in seen or r["split"] not in SPLITS
                or r["patient"] not in value["patient_splits"][r["split"]]
                or not isinstance(r.get("pending_test"), bool)
                or r["pending_test"] != (r["split"] == "test")):
            raise ValueError("Invalid/duplicate cache membership.")
        seen.add(key)
        if _safe(path.parent, r["cache_path"]).suffix != ".npz":
            raise ValueError("Cache entries must be relative NPZ files.")
    validate_patient_splits(value["patient_splits"], [r["patient"] for r in value["records"]])
    return value


def records_for_split(manifest_path, split):
    if split not in SPLITS:
        raise ValueError("Unknown split.")
    return [r for r in load_manifest(manifest_path)["records"] if r["split"] == split]


def load_case(manifest_path, record):
    """Verify actual cache bytes, geometry, finite values and recorded counts."""
    manifest_path = Path(manifest_path).expanduser().resolve(strict=True)
    manifest = load_manifest(manifest_path)
    matches = [r for r in manifest["records"] if (r["patient"], r["visit"]) == (record["patient"], record["visit"])]
    if len(matches) != 1 or matches[0] != record:
        raise ValueError("Case record must come unchanged from this manifest.")
    selected = record
    receipt_path = manifest_path.parent / ("test_completion_receipt.json" if record["split"] == "test" else "trainval_receipt.json")
    receipt = _json(receipt_path)
    if receipt.get("status") != "completed" or receipt.get("manifest_sha256") != _sha(manifest_path):
        raise ValueError("Cache preparation receipt is incomplete or bound to a different manifest.")
    if record["split"] == "test":
        matches = [r for r in receipt.get("records", []) if (r["patient"], r["visit"]) == (record["patient"], record["visit"])]
        if len(matches) != 1:
            raise ValueError("TEST case has not completed under the selection freeze.")
        selected = matches[0]
    path = _safe(manifest_path.parent, selected["cache_path"])
    if _sha(path) != selected.get("cache_sha256"):
        raise ValueError("Cache bytes differ from the completed preparation identity.")
    with np.load(path, allow_pickle=False) as stored:
        expected = {"image", "mask", "mask_eval", "affine_xyz", "affine_eval_xyz", "spacing_dhw", "eval_spacing_dhw", "metadata_json"}
        if set(stored.files) != expected:
            raise ValueError("Unexpected cache fields.")
        value = {k: stored[k].copy() for k in expected if k != "metadata_json"}
        value["metadata"] = json.loads(str(stored["metadata_json"].item()))
    if (value["image"].dtype != np.float32 or value["image"].shape != tuple(manifest["target_shape_dhw"])
            or not np.isfinite(value["image"]).all()):
        raise ValueError("Invalid cached MRI.")
    for name, shape in [("mask", manifest["target_shape_dhw"]), ("mask_eval", manifest["evaluation_shape_dhw"])]:
        mask = value[name]
        if mask.dtype != np.uint8 or mask.shape != tuple(shape) or mask.min() < 0 or mask.max() > 5:
            raise ValueError("Invalid cached six-class mask.")
        counts_key = "class_voxel_counts" if name == "mask" else "evaluation_class_voxel_counts"
        if np.bincount(mask.ravel(), minlength=6).tolist() != selected[counts_key]:
            raise ValueError("Cached class counts differ from prepared counts.")
    for affine_name, spacing_name in [("affine_xyz", "spacing_dhw"), ("affine_eval_xyz", "eval_spacing_dhw")]:
        affine, spacing = value[affine_name], value[spacing_name]
        if (affine.shape != (4, 4) or not np.isfinite(affine).all() or spacing.shape != (3,)
                or not np.isfinite(spacing).all() or np.any(spacing <= 0)
                or not np.allclose(np.linalg.norm(affine[:3, :3], axis=0)[::-1], spacing, rtol=1e-5, atol=1e-5)):
            raise ValueError("Invalid cached affine or physical spacing.")
    return value


def sample_center(mask, rng, probability):
    """Uniform space, or uniform foreground class then uniform class voxel."""
    requested = bool(rng.random() < probability)
    chosen = int(rng.integers(1, 6)) if requested else -1
    fallback = False
    if requested:
        locations = np.flatnonzero(np.asarray(mask).reshape(-1) == chosen)
        if locations.size:
            flat = int(locations[int(rng.integers(len(locations)))])
            return tuple(int(x) for x in np.unravel_index(flat, mask.shape)), requested, chosen, fallback
        fallback = True
    return tuple(int(rng.integers(size)) for size in mask.shape), requested, chosen, fallback


def patch_slices(center, image_shape, patch_shape):
    patch = np.asarray(_shape(patch_shape), dtype=int)
    shape = np.asarray(image_shape, dtype=int)
    if np.any(patch > shape):
        raise ValueError("Patch must fit the training volume.")
    start = np.clip(np.asarray(center) - patch // 2, 0, shape - patch)
    return tuple(slice(int(s), int(s + n)) for s, n in zip(start, patch))


class SampledTrainingDataset(Dataset):
    """TRAIN-only, deterministic per-epoch sampling; validation never uses it."""
    def __init__(self, manifest_path, model_name, config):
        if torch is None:
            raise ImportError("TRAIN datasets require PyTorch.")
        if model_name not in {"unet2d", "improved3d"}:
            raise ValueError("Unknown model.")
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = load_manifest(self.manifest_path)
        if config_digest(config) != self.manifest["config_sha256"]:
            raise ValueError("Training configuration differs from prepared cache.")
        self.records = records_for_split(manifest_path, "train")
        self.model_name, self.config = model_name, config
        self.samples_per_scan = config["train_slices_per_scan"] if model_name == "unet2d" else config["train_patches_per_scan"]
        self.cache = OrderedDict()
        self.set_epoch(1)

    def set_epoch(self, epoch):
        if type(epoch) is not int or epoch < 1:
            raise ValueError("Epoch must be a positive integer.")
        self.epoch = epoch
        self.order = np.random.default_rng(np.random.SeedSequence([self.config["seed"], epoch])).permutation(len(self.records))

    def __len__(self):
        return len(self.records) * self.samples_per_scan

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        record_index = int(self.order[index // self.samples_per_scan])
        if record_index not in self.cache:
            self.cache[record_index] = load_case(self.manifest_path, self.records[record_index])
        value = self.cache.pop(record_index)
        self.cache[record_index] = value
        while len(self.cache) > 2:
            self.cache.popitem(last=False)
        image, mask = value["image"], value["mask"]
        rng = np.random.default_rng(np.random.SeedSequence([self.config["seed"], self.epoch, index]))
        center, requested, chosen, fallback = sample_center(mask, rng, self.config["foreground_probability"])
        region = (center[0], slice(None), slice(None)) if self.model_name == "unet2d" else patch_slices(center, mask.shape, self.config["patch_shape_dhw"])
        x, y = image[region], mask[region]
        for axis in (-2, -1):
            if rng.random() < .5:
                x, y = np.flip(x, axis), np.flip(y, axis)
        return {"image": torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0),
                "mask": torch.from_numpy(np.ascontiguousarray(y)).long(),
                "foreground_requested": requested, "selected_class": chosen,
                "foreground_missing_fallback": fallback, "record_index": record_index,
                "center_dhw": torch.tensor(center)}


def training_class_weights(manifest_path):
    """Inverse-root full-TRAIN-grid frequencies, never sampled/VAL/TEST counts."""
    if torch is None:
        raise ImportError("Class weights require PyTorch.")
    records = records_for_split(manifest_path, "train")
    counts = np.sum([r["class_voxel_counts"] for r in records], axis=0).astype(np.int64)
    if counts.shape != (6,) or np.any(counts <= 0):
        raise ValueError("Every declared class must be present in the TRAIN voxel counts.")
    weights = 1 / np.sqrt(counts / counts.sum())
    weights /= weights.mean()
    weights = np.clip(weights, .05, 10)
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32), counts.tolist()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("trainval", "test"), default="trainval")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--selection-freeze", type=Path)
    args = parser.parse_args(argv)
    result = prepare_cache(args.raw_root, args.split, _json(args.config), args.output_dir,
                           args.stage, args.workers, args.selection_freeze)
    print(f"Manifest: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
