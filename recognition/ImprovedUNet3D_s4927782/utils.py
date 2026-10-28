"""Hard-label segmentation metrics, patient-balanced summaries and run utilities.

Classes are mutually exclusive; class zero is background. Reported overlap
scores have no smoothing. A class absent from both masks is undefined rather
than a perfect score. Organ-level geometry uses six-neighbour surfaces in 3D;
HD95 is undefined for an empty surface and is not replaced by zero.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np


def load_json(path):
    """Read a saved UTF-8 JSON object without executable deserialization."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, data):
    """Atomically replace one JSON file; this is not a multifile transaction."""
    destination = Path(path)
    payload = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(destination)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed):
    """Seed available libraries; CUDA warn-only cannot promise bitwise equality."""
    if type(seed) is not int or seed < 0:
        raise ValueError("The seed must be a nonnegative integer.")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return {"seed": seed, "torch_seeded": False}
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    return {"seed": seed, "torch_seeded": True, "determinism": "warn_only"}


def _defined_mean(values):
    values = [float(value) for value in values if value is not None]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Aggregate values must be finite or None.")
    return sum(values) / len(values) if values else None


def _label_array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if not np.issubdtype(value.dtype, np.integer):
        raise ValueError("Prediction and target must contain integer class IDs.")
    return value


def _surface_distance95(predicted, actual, spacing):
    """Combined bidirectional nearest-surface distance, 95th percentile in mm."""
    from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure

    structure = generate_binary_structure(predicted.ndim, 1)
    pred_surface = predicted & ~binary_erosion(predicted, structure=structure, border_value=0)
    true_surface = actual & ~binary_erosion(actual, structure=structure, border_value=0)
    to_true = distance_transform_edt(~true_surface, sampling=spacing)[pred_surface]
    to_pred = distance_transform_edt(~pred_surface, sampling=spacing)[true_surface]
    return float(np.percentile(np.concatenate((to_true, to_pred)), 95))


def class_metrics(pred, target, num_classes=6, spacing=None, include_hd95=False):
    """Count an entire label volume, with optional grid-aware surface distances.

    ``spacing`` is in the same axis order as the arrays (D,H,W for this project).
    It must be supplied when requesting millimetre distances. Both-empty overlap
    is None; a one-sided nonempty class scores zero and retains its FP/FN counts.
    Empty-surface distances are None, with explicit status for matched-support
    comparisons. Background is kept separately from the foreground mean.
    """
    predicted, actual = _label_array(pred), _label_array(target)
    if predicted.shape != actual.shape or not actual.size or actual.ndim == 0:
        raise ValueError("Prediction and target must have the same nonempty array shape.")
    if type(num_classes) is not int or num_classes < 2:
        raise ValueError("num_classes must be an integer of at least two.")
    if (min(int(actual.min()), int(predicted.min())) < 0
            or max(int(actual.max()), int(predicted.max())) >= num_classes):
        raise ValueError("Label IDs must be in [0, num_classes-1].")
    if include_hd95:
        if spacing is None:
            raise ValueError("Explicit array-axis spacing is required for HD95 in millimetres.")
        spacing = tuple(float(value) for value in spacing)
        if len(spacing) != actual.ndim or not all(math.isfinite(v) and v > 0 for v in spacing):
            raise ValueError("Spacing must have one positive finite value per array axis.")

    encoded = (num_classes * actual.astype(np.int64, copy=False).ravel()
               + predicted.astype(np.int64, copy=False).ravel())
    matrix = np.bincount(encoded, minlength=num_classes**2).reshape(num_classes, num_classes)
    rows = []
    for index in range(num_classes):
        tp = int(matrix[index, index])
        g, p = int(matrix[index].sum()), int(matrix[:, index].sum())
        fp, fn, union = p - tp, g - tp, g + p - tp
        status = ("both_present" if g and p else "both_empty" if not (g or p)
                  else "prediction_empty" if not p else "target_empty")
        row = {"class_index": index, "dice": 2 * tp / (g + p) if g + p else None,
               "iou": tp / union if union else None, "tp": tp, "fp": fp, "fn": fn,
               "intersection": tp, "target_voxels": g, "predicted_voxels": p,
               "union": union, "surface_status": status}
        if include_hd95:
            row["hd95_mm"] = (_surface_distance95(predicted == index, actual == index, spacing)
                              if status == "both_present" else None)
        rows.append(row)
    return {"per_class": rows, "confusion_matrix": matrix.tolist(),
            "foreground_mean_dice": _defined_mean(row["dice"] for row in rows[1:]),
            "foreground_mean_iou": _defined_mean(row["iou"] for row in rows[1:])}


def aggregate_patients(cases, num_classes=6):
    """Average visits within each patient, then give each patient equal weight.

    Each per-class overlap has its own defined-case support. The foreground
    headline first averages the defined classes 1-5 in each case, then visits,
    then patients. It is not pooled-voxel Dice. Distances use defined nonempty
    surfaces only; explicit support counts allow a caller to restrict paired
    model comparisons to their common eligible cases. Empty predictions retain
    their zero overlap even when surface distance is undefined.
    """
    cases = list(cases)
    if type(num_classes) is not int or num_classes < 2:
        raise ValueError("num_classes must be an integer of at least two.")
    if not cases:
        return {"num_cases": 0, "num_patients": 0, "patients": [], "per_class": [],
                "foreground_mean_dice": None, "foreground_mean_iou": None}
    groups, identities = defaultdict(list), set()
    has_distance = any("hd95_mm" in row for case in cases for row in case["per_class"])
    metrics = ("dice", "iou", "hd95_mm") if has_distance else ("dice", "iou")
    for case in cases:
        patient, visit = case.get("patient"), case.get("visit")
        if not isinstance(patient, str) or not patient or not isinstance(visit, str) or not visit:
            raise ValueError("Every case requires nonempty patient and visit strings.")
        if (patient, visit) in identities:
            raise ValueError("Duplicate patient/visit would change the evaluation weighting.")
        identities.add((patient, visit))
        if [row["class_index"] for row in case["per_class"]] != list(range(num_classes)):
            raise ValueError("All cases must have the same ordered class indices.")
        for row in case["per_class"]:
            for name in ("dice", "iou"):
                value = row[name]
                if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                    raise ValueError("Overlap metrics must be None or finite values in [0,1].")
            distance = row.get("hd95_mm")
            if distance is not None and (not math.isfinite(distance) or distance < 0):
                raise ValueError("HD95 must be None or a finite nonnegative distance.")
        groups[patient].append(case)

    patients = []
    for patient, records in sorted(groups.items()):
        per_class = []
        for index in range(num_classes):
            row = {"class_index": index}
            for name in metrics:
                values = [case["per_class"][index].get(name) for case in records]
                row[name] = _defined_mean(values)
                row[name + "_defined_cases"] = sum(v is not None for v in values)
            per_class.append(row)
        patients.append({"patient": patient, "num_cases": len(records), "per_class": per_class,
                         **{f"foreground_mean_{metric}": _defined_mean(
                             _defined_mean(row[metric] for row in case["per_class"][1:])
                             for case in records) for metric in ("dice", "iou")}})
    per_class = []
    for index in range(num_classes):
        row = {"class_index": index}
        for name in metrics:
            values = [patient["per_class"][index][name] for patient in patients]
            row[name] = _defined_mean(values)
            row[name + "_defined_patients"] = sum(v is not None for v in values)
            row[name + "_defined_cases"] = sum(
                patient["per_class"][index][name + "_defined_cases"] for patient in patients)
        per_class.append(row)
    return {"num_cases": len(cases), "num_patients": len(patients), "patients": patients,
            "per_class": per_class,
            "foreground_mean_dice": _defined_mean(p["foreground_mean_dice"] for p in patients),
            "foreground_mean_iou": _defined_mean(p["foreground_mean_iou"] for p in patients),
            "aggregation": "defined classes per case, equal visits per patient, then equal patients; per-class separately"}


def save_training_plot(history, path):
    """Plot recorded TRAIN loss and scheduled VAL prostate Dice, without smoothing."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    losses = [row["train_loss"] for row in history]
    selected = [row for row in history if row.get("validation_prostate_dice") is not None]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, losses, marker=".")
    axes[0].set(xlabel="Epoch", ylabel="Mean TRAIN loss", title="Recorded training")
    axes[1].plot([r["epoch"] for r in selected],
                 [r["validation_prostate_dice"] for r in selected], marker="o")
    axes[1].set(xlabel="Scheduled epoch", ylabel="Patient-balanced prostate Dice",
                title="VAL checkpoint selection", ylim=(0, 1))
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)
