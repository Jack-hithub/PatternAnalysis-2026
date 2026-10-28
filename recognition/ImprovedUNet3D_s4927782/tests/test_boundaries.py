"""Synthetic report-only interface/continuity checks, without real MRI or models."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("boundaries_tool", ROOT / "tools/analyse_boundaries.py")
boundaries = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boundaries)
from utils import load_json, save_json, sha256_file


def make_inputs(root):
    manifest = root / "manifest.json"
    save_json(manifest, {"synthetic": True})
    records = [{"patient": f"{chr(65+p)}{p+1:03}", "visit": f"Week{v}"} for p in range(6) for v in range(5)]
    target = np.zeros((6, 4, 4), np.uint8)
    target[1:5, 1:3, 1:3] = 5
    target[0, 1, 1], target[0, 1, 2] = 3, 4
    affine = np.diag([2., 2., 2., 1.])
    affine[:3, 3] = .5
    cached = {"image": np.ones((12, 8, 8), np.float32), "mask_eval": target,
              "eval_spacing_dhw": np.full(3, 2., np.float32), "affine_eval_xyz": affine,
              "affine_xyz": np.eye(4)}
    directories, hashes = {}, {"manifest": sha256_file(manifest)}
    for name in boundaries.MODELS:
        directory = root / name
        (directory / "predictions").mkdir(parents=True)
        directories[name] = directory
        cases = []
        for record in records:
            pred = target.copy()
            pred[1, 1, 1] = 3
            if name == "improved3d":
                pred[0, 1, 2] = 5
            else:
                pred[0, 1, 1] = 5
            filename = f"predictions/{record['patient']}_{record['visit']}.npz"
            np.savez_compressed(directory / filename, prediction=pred,
                prediction96=np.zeros_like(cached["image"], dtype=np.uint8),
                spacing_dhw=cached["eval_spacing_dhw"], affine_xyz=affine)
            metrics = boundaries.analysis.manual_overlap(pred, target)
            cases.append({**record, "per_class": metrics, "prediction_file": filename})
            hashes[name + "/" + filename] = sha256_file(directory / filename)
        save_json(directory / "evaluation.json", {"status": "completed", "model": name,
                  "split": "test", "grid": "report64", "cases": cases})
        hashes[name + "/evaluation.json"] = sha256_file(directory / "evaluation.json")
    main_analysis = root / "main_analysis"
    main_analysis.mkdir()
    selected = list(reversed(records[::5][:5]))
    save_json(main_analysis / "failure_cases.json", {"cases": selected})
    save_json(main_analysis / "analysis_receipt.json", {"status": "completed", "split": "test",
        "cases_per_model": 30, "input_sha256": hashes,
        "outputs_sha256": {"failure_cases.json": sha256_file(main_analysis / "failure_cases.json")}})
    return manifest, records, cached, directories, main_analysis, selected


class BoundaryAnalysisTests(unittest.TestCase):
    def test_exclusive_interface_partition_and_fn_assignments(self):
        target = np.array([5, 5, 5, 5, 3, 4, 1, 0], np.uint8).reshape(2, 2, 2)
        pred = np.array([0, 3, 4, 5, 5, 5, 5, 0], np.uint8).reshape(2, 2, 2)
        row = boundaries.interface_counts(pred, target)
        self.assertEqual((row["prostate_tp_voxels"], row["prostate_fp_voxels"], row["prostate_fn_voxels"]), (1, 3, 3))
        self.assertEqual([row[f"fp_in_{name}_voxels"] for name in ("bladder", "rectum", "other")], [1, 1, 1])
        self.assertEqual([row[f"fn_as_{name}_voxels"] for name in ("background", "bladder", "rectum")], [1, 1, 1])
        self.assertEqual(row["prostate_false_negative_rate"], .75)
        self.assertAlmostEqual(row["fp_in_bladder_fraction_of_all_fp"], 1/3)
        self.assertEqual(row["fn_as_bladder_fraction_of_gt_prostate"], .25)

    def test_empty_denominators_are_undefined_not_perfect_scores(self):
        empty = np.zeros((3, 3, 3), np.uint8)
        row = boundaries.interface_counts(empty, empty)
        self.assertIsNone(row["fp_in_bladder_fraction_of_all_fp"])
        self.assertIsNone(row["prostate_false_negative_rate"])
        metrics = boundaries.component_metrics(empty == 5)
        self.assertTrue(metrics["empty"])
        self.assertEqual(metrics["component_count_6"], 0)
        self.assertIsNone(metrics["largest_component_fraction"])
        self.assertIsNone(metrics["empty_d_slices_inside_predicted_extent"])

    def test_components_use_six_neighbours_and_detect_d_gaps(self):
        mask = np.zeros((3, 3, 3), bool)
        mask[0, 0, 0], mask[1, 1, 0], mask[1, 1, 1] = True, True, True
        row = boundaries.component_metrics(mask)
        self.assertEqual(row["component_count_6"], 2)
        self.assertAlmostEqual(row["largest_component_fraction"], 2/3)
        mask[:] = False
        mask[0, 1, 1], mask[2, 1, 1] = True, True
        self.assertEqual(boundaries.component_metrics(mask)["empty_d_slices_inside_predicted_extent"], 1)

    def test_area_is_cross_product_not_spacing_product(self):
        affine = np.eye(4)
        affine[:3, 0], affine[:3, 1] = [2, 0, 0], [1, 3, 0]
        self.assertEqual(boundaries.inplane_area_mm2(affine), 6)
        target = np.zeros((3, 2, 2), np.uint8)
        target[0, 0, 0] = 5
        target[1, 0, :] = 5
        result = boundaries.area_curves(target, {"unet2d": target, "improved3d": target}, affine)
        self.assertEqual(result["prostate_area_mm2"]["ground_truth"], [6, 12, 0])
        self.assertEqual(result["d_indices"], [0, 1, 2])

    def test_rate_summary_gives_equal_patients_and_shows_defined_support(self):
        prostate = np.array([[[5]]], np.uint8)
        cases = []
        for patient, visit, tissue in [("A", "W0", 3), ("A", "W1", 4), ("B", "W0", 4)]:
            row = boundaries.interface_counts(prostate, np.array([[[tissue]]], np.uint8))
            cases.append({"patient": patient, "visit": visit, **row})
        summary = boundaries.summarize_interfaces({name: deepcopy(cases) for name in boundaries.MODELS})["models"]["unet2d"]
        row = summary["patient_balanced_rates"]["fp_in_bladder_fraction_of_all_fp"]
        self.assertEqual(row, {"value": .25, "defined_cases": 3, "defined_patients": 2})
        self.assertAlmostEqual(summary["pooled_rates_secondary"]["fp_in_bladder_fraction_of_all_fp"], 1/3)

    def test_planes_are_posthoc_gt_choices_and_empty_fallback_is_explicit(self):
        target = np.zeros((3, 4, 5), np.uint8)
        target[1, 2, 3] = 5
        planes = boundaries.selected_planes(target)
        self.assertEqual((planes["coronal_h"], planes["sagittal_w"]), (2, 3))
        target[:] = 0
        planes = boundaries.selected_planes(target)
        self.assertEqual((planes["coronal_h"], planes["sagittal_w"]), (2, 2))
        self.assertIn("empty", planes["selection"])

    def test_main_selection_hash_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, _, _, directories, main, _ = make_inputs(Path(temporary))
            save_json(main / "failure_cases.json", {"cases": []})
            with self.assertRaisesRegex(ValueError, "selection changed"):
                boundaries.read_context(manifest, directories, main)

    def test_full_posthoc_output_keeps_five_case_order_and_rejects_changed_prediction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, records, cached, directories, main, selected = make_inputs(root)
            with patch.object(boundaries.data, "load_manifest", return_value={"evaluation_shape_dhw": [6, 4, 4]}), \
                 patch.object(boundaries.data, "records_for_split", return_value=records), \
                 patch.object(boundaries.data, "load_case", side_effect=lambda *args: cached):
                output = root / "boundaries"
                result = boundaries.analyse(manifest, directories, main, output)
                self.assertEqual(result["status"], "completed")
                self.assertFalse(result["inference_performed"])
                self.assertTrue(result["main_case_selection_preserved"])
                wanted = [f"{c['patient']}/{c['visit']}" for c in selected]
                self.assertEqual(result["selected_case_ids"], wanted)
                figures = load_json(output / "boundary_figures.json")["cases"]
                self.assertEqual([f"{c['patient']}/{c['visit']}" for c in figures], wanted)
                self.assertEqual(len(list(output.glob("continuity_*.png"))), 5)
                self.assertEqual(len((output / "interface_cases.csv").read_text().splitlines()), 61)
                self.assertEqual(len((output / "connectivity_cases.csv").read_text().splitlines()), 91)
                self.assertEqual(len((output / "slice_areas.csv").read_text().splitlines()), 541)
                with self.assertRaisesRegex(ValueError, "fresh"):
                    boundaries.analyse(manifest, directories, main, output)
                pred = next((directories["unet2d"] / "predictions").iterdir())
                with pred.open("ab") as stream:
                    stream.write(b"tamper")
                with self.assertRaisesRegex(ValueError, "Prediction differs"):
                    boundaries.analyse(manifest, directories, main, root / "tamper")
                self.assertEqual(load_json(root / "tamper/boundary_receipt.json")["status"], "failed")


if __name__ == "__main__":
    unittest.main()
