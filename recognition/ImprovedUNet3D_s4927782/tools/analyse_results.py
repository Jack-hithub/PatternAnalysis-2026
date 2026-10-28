"""Audit saved paired predictions and produce descriptive report material.

No model is loaded and inference is never repeated. Overlap scores are recounted
with Boolean masks, independently of the trainer's confusion-matrix method.
Figures and anatomical thirds are post-hoc analyses, never prediction inputs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import affine_transform

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dataset as data
from utils import class_metrics, load_json, save_json, sha256_file

MODELS = ("unet2d", "improved3d")
CLASS_NAMES = ("Background", "Body", "Bones", "Bladder", "Rectum", "Prostate")
COUNT_FIELDS = ("tp", "fp", "fn", "intersection", "target_voxels", "predicted_voxels", "union")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def mean_defined(values):
    values = [float(v) for v in values if v is not None]
    require(all(math.isfinite(v) for v in values), "Nonfinite analysis value.")
    return sum(values) / len(values) if values else None


def close(actual, expected, label):
    if actual is None or expected is None:
        require(actual is None and expected is None, f"Undefined-value mismatch: {label}")
    else:
        require(math.isfinite(float(actual)) and math.isfinite(float(expected))
                and math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-9),
                f"Saved metric differs from recount: {label}")


def manual_overlap(prediction, target, num_classes=6):
    """Independent unsmoothed Boolean counts; no confusion-matrix helper."""
    prediction, target = np.asarray(prediction), np.asarray(target)
    require(prediction.shape == target.shape and target.size > 0, "Prediction/GT shapes differ or are empty.")
    require(np.issubdtype(prediction.dtype, np.integer) and np.issubdtype(target.dtype, np.integer),
            "Masks must be integer class IDs.")
    require(int(min(prediction.min(), target.min())) >= 0
            and int(max(prediction.max(), target.max())) < num_classes, "Invalid class ID.")
    rows = []
    for index in range(num_classes):
        pred, true = prediction == index, target == index
        tp = int(np.count_nonzero(pred & true))
        fp = int(np.count_nonzero(pred & ~true))
        fn = int(np.count_nonzero(~pred & true))
        denominator, union = 2 * tp + fp + fn, tp + fp + fn
        rows.append({"class_index": index, "tp": tp, "fp": fp, "fn": fn,
                     "intersection": tp, "target_voxels": tp + fn, "predicted_voxels": tp + fp,
                     "union": union, "dice": 2 * tp / denominator if denominator else None,
                     "iou": tp / union if union else None})
    return rows


def independent_summary(cases):
    """Mean visits within patient, then mean patients, independently of utils."""
    groups = defaultdict(list)
    for case in cases:
        groups[case["patient"]].append(case)
    patients = []
    for patient in sorted(groups):
        rows, visits = [], groups[patient]
        for index in range(6):
            row = {"class_index": index}
            for metric in ("dice", "iou", "hd95_mm"):
                values = [visit["per_class"][index].get(metric) for visit in visits]
                row[metric] = mean_defined(values)
                row[metric + "_defined_cases"] = sum(v is not None for v in values)
            rows.append(row)
        patients.append({"patient": patient, "num_cases": len(visits), "per_class": rows,
                         **{f"foreground_mean_{metric}": mean_defined(
                             mean_defined(row[metric] for row in visit["per_class"][1:])
                             for visit in visits) for metric in ("dice", "iou")}})
    classes = []
    for index in range(6):
        row = {"class_index": index}
        for metric in ("dice", "iou", "hd95_mm"):
            values = [patient["per_class"][index][metric] for patient in patients]
            row[metric] = mean_defined(values)
            row[metric + "_defined_patients"] = sum(v is not None for v in values)
            row[metric + "_defined_cases"] = sum(p["per_class"][index][metric + "_defined_cases"] for p in patients)
        classes.append(row)
    return {"num_cases": len(cases), "num_patients": len(patients), "patients": patients,
            "per_class": classes,
            **{f"foreground_mean_{metric}": mean_defined(p[f"foreground_mean_{metric}"] for p in patients)
               for metric in ("dice", "iou")}}


def verify_summary(saved, recounted, label):
    for count in ("num_cases", "num_patients"):
        require(saved.get(count) == recounted[count], f"Aggregate {count} differs: {label}")
    for metric in ("foreground_mean_dice", "foreground_mean_iou"):
        close(saved.get(metric), recounted[metric], f"{label}/{metric}")
    require([p["patient"] for p in saved["patients"]] == [p["patient"] for p in recounted["patients"]],
            f"Aggregate patient order/set differs: {label}")
    for original, checked in [(saved, recounted), *zip(saved["patients"], recounted["patients"])]:
        if "patient" in checked:
            require(original.get("num_cases") == checked["num_cases"], f"Patient visit count differs: {label}")
            for metric in ("foreground_mean_dice", "foreground_mean_iou"):
                close(original.get(metric), checked[metric], f"{label}/{checked['patient']}/{metric}")
        require([r["class_index"] for r in original["per_class"]] == list(range(6)), "Class order differs.")
        for index in range(6):
            for metric in ("dice", "iou", "hd95_mm"):
                close(original["per_class"][index].get(metric), checked["per_class"][index][metric],
                      f"{label}/class{index}/{metric}")
            for key, value in checked["per_class"][index].items():
                if "_defined_" in key:
                    require(original["per_class"][index].get(key) == value, f"Metric support differs: {label}/{key}")


def prostate_thirds(prediction, target):
    """Full H/W slices in thirds of the true prostate's inclusive D extent.

    RAS D increases superiorly. This GT-defined descriptive subdivision is not
    an inference crop; false positives outside this extent remain in global
    prostate scores but are outside this specific regional analysis.
    """
    positions = np.flatnonzero(np.any(target == 5, axis=(1, 2)))
    ranges = np.array_split(np.arange(positions[0], positions[-1] + 1), 3) if len(positions) else [[], [], []]
    result = {}
    for name, indices in zip(("inferior", "middle", "superior"), ranges):
        if len(indices):
            region = manual_overlap(prediction[indices], target[indices])[5]
            result[name] = {**region, "d_start": int(indices[0]), "d_stop_exclusive": int(indices[-1] + 1)}
        else:
            result[name] = {"class_index": 5, "dice": None, "iou": None, "tp": 0, "fp": 0, "fn": 0,
                            "target_voxels": 0, "predicted_voxels": 0, "d_start": None, "d_stop_exclusive": None}
    return result


def common_hd95(cases_by_model):
    """Patient-balanced distances using the exact same eligible visits in each model."""
    indexed = {name: {(c["patient"], c["visit"]): c for c in cases_by_model[name]} for name in MODELS}
    keys = sorted(indexed["unet2d"])
    require(keys == sorted(indexed["improved3d"]), "Paired HD95 case identities differ.")
    rows = []
    for index in range(6):
        eligible = [key for key in keys if all(indexed[name][key]["per_class"][index].get("hd95_mm") is not None for name in MODELS)]
        counts = {name: sum(indexed[name][key]["per_class"][index].get("hd95_mm") is not None for key in keys) for name in MODELS}
        values = {}
        for name in MODELS:
            groups = defaultdict(list)
            for key in eligible:
                groups[key[0]].append(indexed[name][key]["per_class"][index]["hd95_mm"])
            values[name] = mean_defined(mean_defined(v) for v in groups.values())
        rows.append({"class_index": index, "common_cases": len(eligible),
                     "common_patients": len({key[0] for key in eligible}),
                     "common_case_ids": [f"{p}/{v}" for p, v in eligible],
                     "defined_cases_per_model": counts, "patient_balanced_hd95_mm": values})
    return {"definition": "Combined bidirectional six-neighbour surface distances, 95th percentile in mm; empty surfaces undefined.",
            "comparison": "Common eligible visits, equal visits within patient, then equal patients. Zero-overlap empty predictions remain in Dice/IoU.",
            "per_class": rows}


def choose_failures(cases, limit=5):
    """Worst patient means, then worst visit per selected distinct patient."""
    groups = defaultdict(list)
    for case in cases:
        groups[case["patient"]].append(case)
    score = lambda case: case["per_class"][5]["dice"]
    ranked = sorted(groups, key=lambda p: (mean_defined(score(c) for c in groups[p])
                    if mean_defined(score(c) for c in groups[p]) is not None else math.inf, p))
    return [min(groups[p], key=lambda c: (score(c) if score(c) is not None else math.inf, c["visit"]))
            for p in ranked[:limit]]


def resolve_prediction(directory, case):
    relative = Path(case["prediction_file"])
    require(not relative.is_absolute() and ".." not in relative.parts, "Prediction path escapes evaluation directory.")
    path = (directory / relative).resolve(strict=True)
    require(path.is_relative_to(directory.resolve()), "Prediction symlink escapes evaluation directory.")
    return path


def read_prediction(path, cached, expected_shape):
    with np.load(path, allow_pickle=False) as saved:
        require({"prediction", "prediction96", "spacing_dhw", "affine_xyz"} == set(saved.files), "Unexpected prediction fields.")
        pred = saved["prediction"].copy()
        require(pred.shape == tuple(expected_shape), "Prediction is not on the fixed report grid.")
        require(np.array_equal(saved["spacing_dhw"], cached["eval_spacing_dhw"]), "Prediction spacing differs from GT.")
        require(np.allclose(saved["affine_xyz"], cached["affine_eval_xyz"], rtol=0, atol=1e-6), "Prediction affine differs from GT.")
        require(saved["prediction96"].shape == cached["image"].shape, "Stored model-grid prediction shape differs.")
    return pred


def image_on_report_grid(cached):
    transform = np.linalg.solve(cached["affine_xyz"], cached["affine_eval_xyz"])
    image_xyz = affine_transform(cached["image"].transpose(2, 1, 0), transform[:3, :3],
        offset=transform[:3, 3], output_shape=cached["mask_eval"].shape[::-1],
        order=1, mode="grid-constant", cval=0, prefilter=False)
    return np.ascontiguousarray(image_xyz.transpose(2, 1, 0))


def write_csv(path, rows):
    rows = list(rows)
    require(bool(rows), "Refusing an empty report table.")
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(summary, destination):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric in zip(axes, ("dice", "iou")):
        for offset, name, label, colour in [(-.18, "unet2d", "2D U-Net", "#4c78a8"), (.18, "improved3d", "3D Improved U-Net", "#f58518")]:
            values = [summary[name]["per_class"][index][metric] for index in range(1, 6)]
            ax.bar(np.arange(5) + offset, [np.nan if v is None else v for v in values], .36, label=label, color=colour)
        ax.set(xticks=np.arange(5), xticklabels=CLASS_NAMES[1:], ylim=(0, 1), ylabel=metric.title(),
               title=f"Patient-balanced {metric.title()} on the report grid")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def plot_failure(cached, predictions, case_rows, destination):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    image, target = image_on_report_grid(cached), cached["mask_eval"]
    true = target == 5
    errors = (predictions["improved3d"] == 5) != true
    z = int(np.argmax(errors.sum(axis=(1, 2)))) if errors.any() else int(np.argmax(true.sum(axis=(1, 2))))
    palette = ["black", "#e7bd48", "#589ce1", "#4fbf83", "#ca70ad", "#ee5353"]
    cmap = ListedColormap(palette)
    error_cmap = ListedColormap(["black", "#54c678", "#ef594c", "#5495ef"])
    # Portrait layout preserves readable labels when placed on an A4 report page.
    fig, grid = plt.subplots(3, 2, figsize=(9.5, 11.5))
    axes = grid.ravel()
    spacing = cached["eval_spacing_dhw"]
    extent = (0, image.shape[2]*spacing[2], 0, image.shape[1]*spacing[1])
    for ax in axes:
        ax.imshow(image[z], cmap="gray", vmin=-3, vmax=3, origin="lower", extent=extent)
        ax.set_axis_off()
    axes[0].set_title("MRI", fontsize=13)
    for ax, labels, title in [(axes[1], target, "Ground truth"), (axes[2], predictions["unet2d"], "2D U-Net"),
                               (axes[3], predictions["improved3d"], "3D Improved U-Net")]:
        ax.imshow(np.ma.masked_equal(labels[z], 0), cmap=cmap, vmin=-.5, vmax=5.5,
                  alpha=.5, interpolation="nearest", origin="lower", extent=extent)
        ax.set_title(title, fontsize=13)
    for ax, name, title in [(axes[4], "unet2d", "2D prostate errors"), (axes[5], "improved3d", "3D prostate errors")]:
        pred = predictions[name] == 5
        error = np.zeros(target.shape, np.uint8)
        error[pred & true], error[pred & ~true], error[~pred & true] = 1, 2, 3
        ax.imshow(np.ma.masked_equal(error[z], 0), cmap=error_cmap, vmin=-.5, vmax=3.5,
                  alpha=.7, interpolation="nearest", origin="lower", extent=extent)
        ax.set_title(title, fontsize=13)
    row = case_rows["improved3d"]
    title = f"{row['patient']} / {row['visit']} · axial D={z}\nReport-grid prostate Dice: "
    display = lambda value: f"{value:.4f}" if value is not None else "undefined"
    title += f"2D {display(case_rows['unet2d']['per_class'][5]['dice'])}, 3D {display(row['per_class'][5]['dice'])}"
    fig.suptitle(title, fontsize=14, y=.98)
    handles = [Patch(color=palette[i], label=CLASS_NAMES[i]) for i in range(1, 6)]
    handles += [Patch(color=colour, label=label) for colour, label in [("#54c678", "Prostate TP"), ("#ef594c", "Prostate FP"), ("#5495ef", "Prostate FN")]]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=12, bbox_to_anchor=(.5, .065))
    fig.text(.5, .018, "Post-hoc slice: largest 3D prostate FP+FN area.\nCanonical RAS axial view; full image; matching physical extent.",
             ha="center", fontsize=12)
    fig.subplots_adjust(left=.025, right=.975, top=.90, bottom=.17, wspace=.06, hspace=.18)
    fig.savefig(destination, dpi=170)
    plt.close(fig)
    return {"patient": row["patient"], "visit": row["visit"], "d_index": z,
            "slice_selection": "largest 3D prostate FP+FN area; GT maximum area only if no errors",
            "figure": destination.name,
            "prostate_per_model": {name: case_rows[name]["per_class"][5] for name in MODELS}}


def analyse(manifest_path, evaluation_dirs, output_dir):
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = data.load_manifest(manifest_path)
    evaluation_dirs = {name: Path(evaluation_dirs[name]).resolve(strict=True) for name in MODELS}
    saved = {name: load_json(evaluation_dirs[name] / "evaluation.json") for name in MODELS}
    split = saved["unet2d"].get("split")
    require(split in ("validation", "test"), "Only paired VAL or TEST reporting is supported.")
    records = data.records_for_split(manifest_path, split)
    expected = {(r["patient"], r["visit"]): r for r in records}
    require(len(records) == (30 if split == "test" else 38) and len({r["patient"] for r in records}) == 6,
            "Fixed study requires every TEST30/VAL38 scan from six patients.")
    indexed = {}
    hashes = {"manifest": sha256_file(manifest_path)}
    for name in MODELS:
        original = saved[name]
        require(original.get("status") == "completed" and original.get("model") == name
                and original.get("split") == split and original.get("grid") == "report64", "Evaluation identity/status/grid differs.")
        require(original.get("configuration") == manifest["config"]
                and original.get("manifest_sha256") == hashes["manifest"], "Evaluation data/config identity differs.")
        require(len(original["cases"]) == len(expected), "Incomplete or duplicate saved evaluation cases.")
        indexed[name] = {(c["patient"], c["visit"]): c for c in original["cases"]}
        require(set(indexed[name]) == set(expected), "Evaluation case identities differ from the manifest.")
        hashes[name + "/evaluation.json"] = sha256_file(evaluation_dirs[name] / "evaluation.json")
    require(saved["unet2d"].get("selection_freeze_sha256") == saved["improved3d"].get("selection_freeze_sha256")
            and isinstance(saved["unet2d"].get("selection_freeze_sha256"), str), "Paired selection freezes differ or are missing.")
    output = Path(output_dir)
    require(not output.exists() and not output.is_symlink(), "Use a fresh analysis output directory.")
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "analysis_receipt.json", {"status": "running", "split": split})
    cases, regions, prediction_paths = {name: [] for name in MODELS}, {name: [] for name in MODELS}, {}
    try:
        for key, record in sorted(expected.items()):
            cached = data.load_case(manifest_path, record)
            axes = np.asarray(cached["affine_eval_xyz"], dtype=float)[:3, :3]
            gram = axes.T @ axes
            require(np.allclose(gram, np.diag(np.diag(gram)), rtol=0, atol=1e-4),
                    "HD95 spacing metric requires orthogonal voxel axes; sheared grid needs a different physical-distance method.")
            for name in MODELS:
                path = resolve_prediction(evaluation_dirs[name], indexed[name][key])
                prediction_paths[name, key] = path
                hashes[name + "/" + indexed[name][key]["prediction_file"]] = sha256_file(path)
                prediction = read_prediction(path, cached, manifest["evaluation_shape_dhw"])
                counts = manual_overlap(prediction, cached["mask_eval"])
                distances = class_metrics(prediction, cached["mask_eval"], spacing=cached["eval_spacing_dhw"], include_hd95=True)
                original = indexed[name][key]
                require([r["class_index"] for r in original["per_class"]] == list(range(6)), "Saved case class order differs.")
                for index, row in enumerate(counts):
                    old = original["per_class"][index]
                    for field in COUNT_FIELDS:
                        require(old.get(field) == row[field], f"Count mismatch {name}/{key}/class{index}/{field}")
                    for metric in ("dice", "iou"):
                        close(old.get(metric), row[metric], f"{name}/{key}/class{index}/{metric}")
                    row["hd95_mm"] = distances["per_class"][index]["hd95_mm"]
                    require("hd95_mm" in old, "Final evaluation omitted physical boundary distances.")
                    close(old["hd95_mm"], row["hd95_mm"], f"{name}/{key}/class{index}/hd95_mm")
                checked = {"patient": key[0], "visit": key[1], "per_class": counts}
                for metric in ("dice", "iou"):
                    close(original.get("foreground_mean_" + metric), mean_defined(r[metric] for r in counts[1:]),
                          f"{name}/{key}/foreground_{metric}")
                cases[name].append(checked)
                regions[name].append({"patient": key[0], "visit": key[1], "thirds": prostate_thirds(prediction, cached["mask_eval"])})
            print(f"Recounted {key[0]}/{key[1]}", flush=True)
        summary = {name: independent_summary(cases[name]) for name in MODELS}
        for name in MODELS:
            verify_summary(saved[name]["aggregate"], summary[name], name)
            close(saved[name].get("patient_balanced_prostate_dice"), summary[name]["per_class"][5]["dice"], name + "/headline")
        common = common_hd95(cases)
        selected = choose_failures(cases["improved3d"])
        failures = []
        for selected_case in selected:
            key = selected_case["patient"], selected_case["visit"]
            cached = data.load_case(manifest_path, expected[key])
            predictions = {name: read_prediction(prediction_paths[name, key], cached, manifest["evaluation_shape_dhw"]) for name in MODELS}
            rows = {name: next(c for c in cases[name] if (c["patient"], c["visit"]) == key) for name in MODELS}
            failures.append(plot_failure(cached, predictions, rows, output / f"failure_{key[0]}_{key[1]}.png"))
        region_summary = {}
        for name in MODELS:
            region_summary[name] = {}
            for third in ("inferior", "middle", "superior"):
                per_patient = defaultdict(list)
                for case in regions[name]:
                    per_patient[case["patient"]].append(case["thirds"][third])
                region_summary[name][third] = {metric: mean_defined(mean_defined(r[metric] for r in records)
                    for records in per_patient.values()) for metric in ("dice", "iou")}
                region_summary[name][third].update(defined_cases=sum(c["thirds"][third]["dice"] is not None for c in regions[name]),
                    defined_patients=sum(any(r["dice"] is not None for r in records) for records in per_patient.values()))
        save_json(output / "recounted_cases.json", cases)
        save_json(output / "patient_balanced_summary.json", summary)
        save_json(output / "boundary_common_support.json", common)
        save_json(output / "prostate_thirds.json", {"definition": prostate_thirds.__doc__, "summary": region_summary, "cases": regions})
        save_json(output / "failure_cases.json", {"selection": "Worst five distinct 3D patient-balanced prostate Dice patients, then their worst visit; descriptive post-hoc selection.",
                  "causality": "These figures locate disagreement; they do not establish a unique root cause.", "cases": failures})
        write_csv(output / "case_metrics.csv", ({"model": name, "patient": case["patient"], "visit": case["visit"], **row}
                  for name in MODELS for case in cases[name] for row in case["per_class"]))
        write_csv(output / "patient_metrics.csv", ({"model": name, "patient": patient["patient"], "num_visits": patient["num_cases"], **row}
                  for name in MODELS for patient in summary[name]["patients"] for row in patient["per_class"]))
        write_csv(output / "class_comparison.csv", ({"class_index": index, "class_name": CLASS_NAMES[index],
                  **{name + "_" + metric: summary[name]["per_class"][index][metric] for name in MODELS for metric in ("dice", "iou")},
                  "common_hd95_cases": common["per_class"][index]["common_cases"],
                  **{name + "_common_hd95_mm": common["per_class"][index]["patient_balanced_hd95_mm"][name] for name in MODELS}}
                  for index in range(6)))
        plot_comparison(summary, output / "class_comparison.png")
        receipt = {"status": "completed", "completed_at": datetime.now(timezone.utc).isoformat(), "split": split,
                   "cases_per_model": len(expected), "patients": len({p for p, _ in expected}),
                   "manual_overlap_recount_passed": True, "patient_aggregation_recount_passed": True,
                   "boundary_distances_recomputed": True, "prediction_inference_repeated": False,
                   "input_sha256": hashes, "analysis_source_sha256": sha256_file(__file__),
                   "outputs_sha256": {p.name: sha256_file(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "analysis_receipt.json"},
                   "evaluation_context": manifest["config"].get("evaluation_context"),
                   "limitations": "One fixed patient-disjoint run on a previously explored cohort. Six patients do not support a broad clinical generalisation claim."}
        save_json(output / "analysis_receipt.json", receipt)
        return receipt
    except BaseException as exc:
        save_json(output / "analysis_receipt.json", {"status": "failed", "split": split, "error": f"{type(exc).__name__}: {exc}"})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--unet2d-evaluation", type=Path, required=True)
    parser.add_argument("--improved3d-evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    analyse(args.manifest, {"unet2d": args.unet2d_evaluation, "improved3d": args.improved3d_evaluation}, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
