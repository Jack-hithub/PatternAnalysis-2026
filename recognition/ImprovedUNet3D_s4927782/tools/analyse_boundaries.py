"""Describe organ-interface errors and continuity in saved paired predictions.

Run only after analyse_results.py has completed. This report-only operation
preserves its exact five cases and never loads a model, performs inference,
changes a prediction, chooses weights or establishes clinical safety.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import generate_binary_structure, label

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyse_results as analysis
import dataset as data
from utils import load_json, save_json, sha256_file

MODELS = analysis.MODELS
OTHER_CLASSES = ((0, "background"), (1, "body"), (2, "bones"), (3, "bladder"), (4, "rectum"))
RATIOS = {
    "fp_in_bladder_fraction_of_all_fp": ("fp_in_bladder_voxels", "prostate_fp_voxels"),
    "fp_in_rectum_fraction_of_all_fp": ("fp_in_rectum_voxels", "prostate_fp_voxels"),
    "fp_in_other_fraction_of_all_fp": ("fp_in_other_voxels", "prostate_fp_voxels"),
    "prostate_false_negative_rate": ("prostate_fn_voxels", "gt_prostate_voxels"),
    "prostate_false_positive_fraction": ("prostate_fp_voxels", "pred_prostate_voxels"),
    **{f"fn_as_{name}_fraction_of_gt_prostate": (f"fn_as_{name}_voxels", "gt_prostate_voxels")
       for _, name in OTHER_CLASSES},
}


def ratios(counts):
    return {key: counts[numerator] / counts[denominator] if counts[denominator] else None
            for key, (numerator, denominator) in RATIOS.items()}


def interface_counts(prediction, target):
    """P5 FP destination and FN assignment, using exclusive categorical labels.

    Intersection with a whole-organ GT label is not proof about its thin wall
    or mucosa. All quantities describe the declared downsampled report grid.
    """
    overlap = analysis.manual_overlap(prediction, target)[5]
    fp = (prediction == 5) & (target != 5)
    fn = (target == 5) & (prediction != 5)
    result = {"gt_prostate_voxels": overlap["target_voxels"],
              "pred_prostate_voxels": overlap["predicted_voxels"],
              "prostate_tp_voxels": overlap["tp"], "prostate_fp_voxels": overlap["fp"],
              "prostate_fn_voxels": overlap["fn"],
              "fp_in_bladder_voxels": int(np.count_nonzero(fp & (target == 3))),
              "fp_in_rectum_voxels": int(np.count_nonzero(fp & (target == 4))),
              "fp_in_other_voxels": int(np.count_nonzero(fp & ((target == 0) | (target == 1) | (target == 2)))),
              **{f"fn_as_{name}_voxels": int(np.count_nonzero(fn & (prediction == index)))
                 for index, name in OTHER_CLASSES}}
    analysis.require(sum(result[f"fp_in_{name}_voxels"] for name in ("bladder", "rectum", "other"))
                     == result["prostate_fp_voxels"], "FP partition does not sum to all prostate FP.")
    analysis.require(sum(result[f"fn_as_{name}_voxels"] for _, name in OTHER_CLASSES)
                     == result["prostate_fn_voxels"], "FN partition does not sum to all prostate FN.")
    return {**result, **ratios(result)}


def component_metrics(mask):
    """Six-neighbour 3D components, with explicit empty-prediction status.

    Zero components or zero area variation for an empty mask is not a measure
    of good smoothness. Neither these counts nor a largest-component fraction
    is a clinical topology or correctness score.
    """
    mask = np.asarray(mask, dtype=bool)
    analysis.require(mask.ndim == 3 and mask.size > 0, "Expected a nonempty D,H,W mask.")
    components, count = label(mask, structure=generate_binary_structure(3, 1))
    sizes = np.bincount(components.ravel())[1:]
    voxels = int(mask.sum())
    largest = int(sizes.max()) if len(sizes) else 0
    area = mask.sum(axis=(1, 2))
    nonempty = np.flatnonzero(area)
    gaps = int(np.count_nonzero(area[nonempty[0]:nonempty[-1]+1] == 0)) if len(nonempty) else None
    return {"empty": voxels == 0, "voxels": voxels, "component_count_6": int(count),
            "largest_component_voxels": largest,
            "largest_component_fraction": largest / voxels if voxels else None,
            "positive_d_slices": int(len(nonempty)),
            "empty_d_slices_inside_predicted_extent": gaps}


def inplane_area_mm2(affine_xyz):
    """Axial voxel area from X/Y affine columns, including oblique geometry."""
    affine = np.asarray(affine_xyz, dtype=float)
    analysis.require(affine.shape == (4, 4) and np.isfinite(affine).all(), "Invalid report-grid affine.")
    area = float(np.linalg.norm(np.cross(affine[:3, 0], affine[:3, 1])))
    analysis.require(area > 0, "Report-grid in-plane area must be positive.")
    return area


def area_curves(target, predictions, affine_xyz):
    area = inplane_area_mm2(affine_xyz)
    masks = {"ground_truth": target, **predictions}
    analysis.require(all(value.shape == target.shape for value in masks.values()), "Area-curve grids differ.")
    return {"inplane_voxel_area_mm2": area,
            "d_indices": list(range(target.shape[0])),
            "prostate_area_mm2": {name: ((mask == 5).sum(axis=(1, 2)) * area).tolist()
                                   for name, mask in masks.items()}}


def summarize_interfaces(cases_by_model):
    result = {}
    for name in MODELS:
        cases = cases_by_model[name]
        groups = defaultdict(list)
        for case in cases:
            groups[case["patient"]].append(case)
        counts = {key: sum(row[key] for row in cases) for key in cases[0] if key.endswith("_voxels")}
        patient_balanced = {}
        for metric in RATIOS:
            patient_values = [analysis.mean_defined(row[metric] for row in visits) for visits in groups.values()]
            patient_balanced[metric] = {"value": analysis.mean_defined(patient_values),
                "defined_cases": sum(row[metric] is not None for row in cases),
                "defined_patients": sum(value is not None for value in patient_values)}
        result[name] = {"num_cases": len(cases), "num_patients": len(groups),
                        "patient_balanced_rates": patient_balanced,
                        "pooled_counts_secondary": counts, "pooled_rates_secondary": ratios(counts)}
    return {"definition": "Means of defined case rates within each patient, then equal patient means; support counts shown.",
            "pooled_warning": "Pooled voxel counts/rates are secondary descriptive summaries; visits are correlated and voxel volume varies across scans.",
            "interface_limit": "P5 FP inside GT whole bladder/rectum describes categorical overlap, not millimetre-level wall/mucosal safety.",
            "models": result}


def selected_planes(target):
    support = target == 5
    if support.any():
        return {"coronal_h": int(np.argmax(support.sum(axis=(0, 2)))),
                "sagittal_w": int(np.argmax(support.sum(axis=(0, 1)))),
                "selection": "post-hoc largest GT prostate area independently in each orientation"}
    return {"coronal_h": target.shape[1] // 2, "sagittal_w": target.shape[2] // 2,
            "selection": "post-hoc image-centre fallback because GT prostate is empty"}


def plot_case(cached, predictions, identity, destination):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    image, target = analysis.image_on_report_grid(cached), cached["mask_eval"]
    planes = selected_planes(target)
    curves = area_curves(target, predictions, cached["affine_eval_xyz"])
    palette = ["black", "#e7bd48", "#589ce1", "#4fbf83", "#ca70ad", "#ee5353"]
    cmap = ListedColormap(palette)
    spacing = np.linalg.norm(np.asarray(cached["affine_eval_xyz"])[:3, :3], axis=0)[::-1]
    fig = plt.figure(figsize=(9.5, 11.5))
    grid = fig.add_gridspec(3, 3, height_ratios=(1, 1, .75), hspace=.28, wspace=.06)
    masks = [("Ground truth", target), ("2D U-Net", predictions["unet2d"]),
             ("3D Improved U-Net", predictions["improved3d"])]
    for row in range(2):
        for column, (name, mask) in enumerate(masks):
            ax = fig.add_subplot(grid[row, column])
            if row == 0:
                index = planes["coronal_h"]
                raw, labels = image[:, index, :], mask[:, index, :]
                extent = (0, image.shape[2]*spacing[2], 0, image.shape[0]*spacing[0])
                title = f"Coronal H={index}\n{name}"
            else:
                index = planes["sagittal_w"]
                raw, labels = image[:, :, index], mask[:, :, index]
                extent = (0, image.shape[1]*spacing[1], 0, image.shape[0]*spacing[0])
                title = f"Sagittal W={index}\n{name}"
            ax.imshow(raw, cmap="gray", vmin=-3, vmax=3, origin="lower", extent=extent)
            ax.imshow(np.ma.masked_equal(labels, 0), cmap=cmap, vmin=-.5, vmax=5.5,
                      alpha=.55, interpolation="nearest", origin="lower", extent=extent)
            ax.set_title(title, fontsize=12)
            ax.set_axis_off()
    ax = fig.add_subplot(grid[2, :])
    for name, label_name, colour in [("ground_truth", "GT", "black"), ("unet2d", "2D U-Net", "#4c78a8"),
                                     ("improved3d", "3D Improved U-Net", "#f58518")]:
        ax.plot(curves["d_indices"], curves["prostate_area_mm2"][name], label=label_name, color=colour, linewidth=2)
    ax.set_xlabel("Report-grid D index", fontsize=12)
    ax.set_ylabel("Prostate area (mm²)", fontsize=12)
    ax.set_title("Full D extent: GT and predicted prostate slice area", fontsize=12)
    ax.tick_params(labelsize=12)
    ax.set_xlim(0, max(1, target.shape[0]-1))
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=12, loc="best")
    ax.grid(alpha=.25)
    fig.suptitle(f"{identity['patient']} / {identity['visit']}\nSaved-prediction spatial continuity", fontsize=14, y=.98)
    fig.legend(handles=[Patch(color=palette[i], label=analysis.CLASS_NAMES[i]) for i in range(1, 6)],
               loc="lower center", ncol=5, frameon=False, fontsize=12, bbox_to_anchor=(.5, .065))
    fig.text(.5, .018, "Planes chosen post hoc from GT; full report-grid images, no inference crop.\nD increases superiorly; these are not precise anatomical base/apex labels.",
             fontsize=12, ha="center")
    fig.subplots_adjust(left=.105, right=.97, top=.90, bottom=.18)
    fig.savefig(destination, dpi=170)
    plt.close(fig)
    return {**identity, **planes, "figure": destination.name,
            "inplane_voxel_area_mm2": curves["inplane_voxel_area_mm2"]}


def read_context(manifest_path, evaluation_dirs, main_analysis):
    receipt_path = main_analysis / "analysis_receipt.json"
    receipt = load_json(receipt_path)
    analysis.require(receipt.get("status") == "completed", "Main analysis has not completed.")
    hashes = receipt["input_sha256"]
    analysis.require(hashes.get("manifest") == sha256_file(manifest_path), "Main analysis used a different manifest.")
    selection_path = main_analysis / "failure_cases.json"
    analysis.require(receipt["outputs_sha256"].get(selection_path.name) == sha256_file(selection_path), "Main failure selection changed.")
    selection = load_json(selection_path)["cases"]
    analysis.require(len(selection) == 5 and len({c["patient"] for c in selection}) == 5,
                     "Expected the same five distinct-patient cases from main analysis.")
    evaluations, indexed = {}, {}
    for name in MODELS:
        path = evaluation_dirs[name] / "evaluation.json"
        analysis.require(hashes.get(name + "/evaluation.json") == sha256_file(path), "Evaluation metadata differs from audited input.")
        evaluations[name] = load_json(path)
        analysis.require(evaluations[name].get("split") == receipt["split"]
                         and evaluations[name].get("model") == name
                         and evaluations[name].get("grid") == "report64", "Evaluation identity differs.")
        indexed[name] = {(c["patient"], c["visit"]): c for c in evaluations[name]["cases"]}
    selected = [(c["patient"], c["visit"]) for c in selection]
    analysis.require(all(key in indexed[name] for key in selected for name in MODELS), "Selected case absent from paired evaluation.")
    return receipt, indexed, selected


def analyse(manifest_path, evaluation_dirs, main_analysis, output_dir):
    manifest_path = Path(manifest_path).resolve(strict=True)
    evaluation_dirs = {name: Path(evaluation_dirs[name]).resolve(strict=True) for name in MODELS}
    main_analysis = Path(main_analysis).resolve(strict=True)
    manifest = data.load_manifest(manifest_path)
    main, indexed, selected = read_context(manifest_path, evaluation_dirs, main_analysis)
    records = data.records_for_split(manifest_path, main["split"])
    expected = {(r["patient"], r["visit"]): r for r in records}
    analysis.require(len(expected) == main["cases_per_model"] and len(expected) == len(records)
                     and all(set(indexed[name]) == set(expected) for name in MODELS), "Paired case coverage differs from completed main analysis.")
    output = Path(output_dir)
    analysis.require(not output.exists() and not output.is_symlink(), "Use a fresh boundary-analysis output directory.")
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "boundary_receipt.json", {"status": "running"})
    errors, connectivity, curves, figures = {name: [] for name in MODELS}, [], [], []
    hashes = {"manifest": sha256_file(manifest_path),
              "main_analysis_receipt": sha256_file(main_analysis / "analysis_receipt.json"),
              "main_failure_cases": sha256_file(main_analysis / "failure_cases.json")}
    try:
        for key, record in sorted(expected.items()):
            cached = data.load_case(manifest_path, record)
            target = cached["mask_eval"]
            predictions = {}
            for name in MODELS:
                original = indexed[name][key]
                path = analysis.resolve_prediction(evaluation_dirs[name], original)
                input_key = name + "/" + original["prediction_file"]
                hashes[input_key] = sha256_file(path)
                analysis.require(main["input_sha256"].get(input_key) == hashes[input_key], "Prediction differs from completed main analysis.")
                prediction = analysis.read_prediction(path, cached, manifest["evaluation_shape_dhw"])
                predictions[name] = prediction
                breakdown = interface_counts(prediction, target)
                old = original["per_class"][5]
                analysis.require((breakdown["prostate_tp_voxels"], breakdown["prostate_fp_voxels"], breakdown["prostate_fn_voxels"])
                                 == (old["tp"], old["fp"], old["fn"]), "Interface counts differ from audited prostate counts.")
                errors[name].append({"patient": key[0], "visit": key[1], **breakdown})
            area = area_curves(target, predictions, cached["affine_eval_xyz"])
            gt_slices = (target == 5).any(axis=(1, 2))
            for name, mask in {"ground_truth": target, **predictions}.items():
                stats = component_metrics(mask == 5)
                stats["gt_positive_slices_missing_prediction"] = int(np.count_nonzero(gt_slices & ~(mask == 5).any(axis=(1, 2))))
                connectivity.append({"patient": key[0], "visit": key[1], "source": name, **stats})
                curves.extend({"patient": key[0], "visit": key[1], "source": name, "d_index": d,
                               "area_mm2": value, "inplane_voxel_area_mm2": area["inplane_voxel_area_mm2"]}
                              for d, value in enumerate(area["prostate_area_mm2"][name]))
            if key in selected:
                figures.append(plot_case(cached, predictions, {"patient": key[0], "visit": key[1]},
                                         output / f"continuity_{key[0]}_{key[1]}.png"))
            print(f"Described saved boundaries {key[0]}/{key[1]}", flush=True)
        # Preserve the main analysis's case order even though cache iteration is sorted.
        figures.sort(key=lambda row: selected.index((row["patient"], row["visit"])))
        summary = summarize_interfaces(errors)
        component_summary = {}
        for name in ("ground_truth", *MODELS):
            rows = [row for row in connectivity if row["source"] == name]
            groups = defaultdict(list)
            for row in rows:
                groups[row["patient"]].append(row)
            component_summary[name] = {"empty_cases": sum(row["empty"] for row in rows),
                "num_cases": len(rows), "num_patients": len(groups),
                "patient_balanced_component_count": analysis.mean_defined(analysis.mean_defined(r["component_count_6"] for r in visits) for visits in groups.values()),
                "patient_balanced_largest_fraction_nonempty": analysis.mean_defined(analysis.mean_defined(r["largest_component_fraction"] for r in visits) for visits in groups.values()),
                "largest_fraction_defined_cases": sum(not row["empty"] for row in rows),
                "largest_fraction_defined_patients": sum(any(not r["empty"] for r in visits) for visits in groups.values())}
        save_json(output / "interface_cases.json", errors)
        save_json(output / "interface_summary.json", summary)
        save_json(output / "connectivity_summary.json", {"definition": component_metrics.__doc__, "models": component_summary})
        save_json(output / "boundary_figures.json", {"selection": "Exact five patient/visit identities and order from completed main analysis; no reselection.",
                  "cases": figures})
        analysis.write_csv(output / "interface_cases.csv", ({"model": name, **row} for name in MODELS for row in errors[name]))
        analysis.write_csv(output / "connectivity_cases.csv", connectivity)
        analysis.write_csv(output / "slice_areas.csv", curves)
        result = {"status": "completed", "completed_at": datetime.now(timezone.utc).isoformat(), "split": main["split"],
                  "cases_per_model": len(expected), "patients": len({key[0] for key in expected}),
                  "selected_case_ids": [f"{patient}/{visit}" for patient, visit in selected],
                  "main_case_selection_preserved": True, "inference_performed": False,
                  "prediction_modified": False, "input_sha256": hashes,
                  "source_sha256": sha256_file(__file__),
                  "outputs_sha256": {p.name: sha256_file(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "boundary_receipt.json"},
                  "interpretation": "Descriptive report-grid overlap, component and area analysis; no clinical boundary-safety or causal claim. An empty prediction is a failure to draft, not perfect smoothness."}
        save_json(output / "boundary_receipt.json", result)
        return result
    except BaseException as exc:
        save_json(output / "boundary_receipt.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--unet2d-evaluation", type=Path, required=True)
    parser.add_argument("--improved3d-evaluation", type=Path, required=True)
    parser.add_argument("--main-analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    analyse(args.manifest, {"unet2d": args.unet2d_evaluation, "improved3d": args.improved3d_evaluation}, args.main_analysis, args.output_dir)
    return 0


if __name__ == "__main__":
    main()
