"""Synthetic saved-prediction audits; no real MRI, models or inference."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("analysis_tool", ROOT / "tools/analyse_results.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
from utils import aggregate_patients, class_metrics, load_json, save_json, sha256_file


def synthetic_case(patient, visit, dice, distance=2):
    rows = [{"class_index": index, "dice": dice, "iou": dice / (2-dice), "hd95_mm": distance}
            for index in range(6)]
    return {"patient": patient, "visit": visit, "per_class": rows}


class ResultAnalysisTests(unittest.TestCase):
    def test_boolean_recount_has_unsmoothed_overlap_and_real_empty_policy(self):
        true = np.array([5, 5, 0, 0], np.uint8)
        pred = np.array([5, 0, 5, 0], np.uint8)
        rows = analysis.manual_overlap(pred, true)
        self.assertEqual((rows[5]["tp"], rows[5]["fp"], rows[5]["fn"]), (1, 1, 1))
        self.assertEqual(rows[5]["dice"], .5)
        self.assertAlmostEqual(rows[5]["iou"], 1/3)
        self.assertIsNone(rows[1]["dice"])
        self.assertEqual(analysis.manual_overlap(np.zeros(4, np.uint8), true)[5]["dice"], 0)
        with self.assertRaises(ValueError):
            analysis.manual_overlap(pred.astype(float), true)
        with self.assertRaises(ValueError):
            analysis.manual_overlap(pred, true[:2])

    def test_independent_means_equal_patients_not_visits(self):
        cases = [synthetic_case("A", "Week0", 0), synthetic_case("A", "Week1", 1),
                 synthetic_case("B", "Week0", 1)]
        result = analysis.independent_summary(cases)
        self.assertEqual(result["per_class"][5]["dice"], .75)
        self.assertEqual(result["per_class"][5]["dice_defined_cases"], 3)
        self.assertEqual(result["per_class"][5]["dice_defined_patients"], 2)
        self.assertEqual(result["foreground_mean_dice"], .75)
        analysis.verify_summary(aggregate_patients(cases), result, "fixture")
        changed = deepcopy(result)
        changed["patients"][0]["num_cases"] = 3
        with self.assertRaisesRegex(ValueError, "visit count"):
            analysis.verify_summary(changed, result, "fixture")

    def test_common_hd95_uses_exact_same_visits_and_keeps_missing_overlap(self):
        two = [synthetic_case("A", "W0", .5, 2), synthetic_case("A", "W1", .5, 100),
               synthetic_case("B", "W0", .5, 4)]
        three = [synthetic_case("A", "W0", .6, 1), synthetic_case("A", "W1", 0, None),
                 synthetic_case("B", "W0", .6, 3)]
        result = analysis.common_hd95({"unet2d": two, "improved3d": three})["per_class"][5]
        self.assertEqual(result["common_cases"], 2)
        self.assertEqual(result["common_patients"], 2)
        self.assertEqual(result["defined_cases_per_model"], {"unet2d": 3, "improved3d": 2})
        self.assertEqual(result["patient_balanced_hd95_mm"], {"unet2d": 3, "improved3d": 2})
        self.assertEqual(analysis.independent_summary(three)["patients"][0]["per_class"][5]["dice"], .3)

    def test_gt_extent_thirds_are_descriptive_and_global_false_positives_remain(self):
        target = np.zeros((10, 3, 3), np.uint8)
        target[2:8, 1, 1] = 5
        prediction = target.copy()
        prediction[2:4, 1, 1] = 0
        prediction[0, 0, 0] = 5
        thirds = analysis.prostate_thirds(prediction, target)
        self.assertEqual((thirds["inferior"]["d_start"], thirds["inferior"]["d_stop_exclusive"]), (2, 4))
        self.assertEqual(thirds["inferior"]["dice"], 0)
        self.assertEqual(thirds["middle"]["dice"], 1)
        self.assertEqual(thirds["superior"]["dice"], 1)
        self.assertEqual(analysis.manual_overlap(prediction, target)[5]["fp"], 1)
        self.assertEqual(sum(row["fp"] for row in thirds.values()), 0)
        absent = analysis.prostate_thirds(np.zeros_like(target), np.zeros_like(target))
        self.assertTrue(all(row["dice"] is None for row in absent.values()))

    def test_failure_selection_is_distinct_patients_then_their_worst_visit(self):
        cases = [synthetic_case("A", "W0", 0), synthetic_case("A", "W1", 1),
                 synthetic_case("B", "W0", .4), synthetic_case("C", "W0", .9)]
        result = analysis.choose_failures(cases, limit=2)
        self.assertEqual([(c["patient"], c["visit"]) for c in result], [("B", "W0"), ("A", "W0")])

    def test_report_image_follows_physical_centres(self):
        image = np.indices((8, 8, 8))[0].astype(np.float32)
        second = np.diag([2., 2., 2., 1.])
        second[:3, 3] = .5
        cached = {"image": image, "mask_eval": np.zeros((4, 4, 4), np.uint8),
                  "affine_xyz": np.eye(4), "affine_eval_xyz": second}
        output = analysis.image_on_report_grid(cached)
        np.testing.assert_allclose(output[:, 0, 0], [.5, 2.5, 4.5, 6.5])

    def test_prediction_path_and_geometry_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "escapes"):
                analysis.resolve_prediction(root, {"prediction_file": "../wrong.npz"})
            pred = np.zeros((2, 2, 2), np.uint8)
            path = root / "prediction.npz"
            np.savez_compressed(path, prediction=pred, prediction96=pred, spacing_dhw=np.ones(3), affine_xyz=np.eye(4))
            cached = {"image": pred, "eval_spacing_dhw": np.ones(3), "affine_eval_xyz": np.eye(4)}
            np.testing.assert_array_equal(analysis.read_prediction(path, cached, pred.shape), pred)
            cached["eval_spacing_dhw"] = np.array([2, 1, 1])
            with self.assertRaisesRegex(ValueError, "spacing"):
                analysis.read_prediction(path, cached, pred.shape)

    def test_full_saved_pair_recount_outputs_and_rejects_tampered_score(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            save_json(manifest, {"synthetic": True})
            config = {"evaluation_shape_dhw": [6, 4, 4], "evaluation_context": "Synthetic fixture only."}
            records = [{"patient": f"{chr(65+p)}{p+1:03}", "visit": f"Week{v}"} for p in range(6) for v in range(5)]
            target = np.zeros((6, 4, 4), np.uint8)
            target[1:5, 1:3, 1:3] = 5
            for index in range(1, 5):
                target[0, 0, index-1] = index
            affine = np.diag([2., 2., 2., 1.])
            affine[:3, 3] = .5
            cached = {"image": np.ones((12, 8, 8), np.float32), "mask_eval": target,
                      "eval_spacing_dhw": np.full(3, 2., np.float32), "affine_eval_xyz": affine,
                      "affine_xyz": np.eye(4)}
            directories = {}
            for name in analysis.MODELS:
                directory = root / name
                (directory / "predictions").mkdir(parents=True)
                directories[name] = directory
                cases = []
                for record in records:
                    pred = target.copy()
                    if name == "unet2d":
                        pred[1, 1, 1] = 0
                    else:
                        pred[1, 1:3, 1:3] = 0
                    filename = f"predictions/{record['patient']}_{record['visit']}.npz"
                    np.savez_compressed(directory / filename, prediction=pred,
                        prediction96=np.zeros_like(cached["image"], dtype=np.uint8),
                        spacing_dhw=cached["eval_spacing_dhw"], affine_xyz=affine)
                    metrics = class_metrics(pred, target, spacing=cached["eval_spacing_dhw"], include_hd95=True)
                    cases.append({**record, **metrics, "prediction_file": filename})
                aggregate = aggregate_patients(cases)
                save_json(directory / "evaluation.json", {"status": "completed", "model": name,
                    "split": "test", "grid": "report64", "configuration": config, "cases": cases,
                    "aggregate": aggregate, "patient_balanced_prostate_dice": aggregate["per_class"][5]["dice"],
                    "manifest_sha256": sha256_file(manifest), "selection_freeze_sha256": "a"*64})
            with patch.object(analysis.data, "load_manifest", return_value={"config": config, "evaluation_shape_dhw": [6, 4, 4]}), \
                 patch.object(analysis.data, "records_for_split", return_value=records), \
                 patch.object(analysis.data, "load_case", side_effect=lambda *args: cached):
                output = root / "analysis"
                result = analysis.analyse(manifest, directories, output)
                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["cases_per_model"], 30)
                self.assertFalse(result["prediction_inference_repeated"])
                self.assertEqual(len(list(output.glob("failure_*.png"))), 5)
                self.assertGreater((output / "class_comparison.png").stat().st_size, 1000)
                self.assertEqual(len(load_json(output / "failure_cases.json")["cases"]), 5)
                self.assertEqual(len((output / "case_metrics.csv").read_text().splitlines()), 361)
                with self.assertRaisesRegex(ValueError, "fresh"):
                    analysis.analyse(manifest, directories, output)
                original = load_json(directories["improved3d"] / "evaluation.json")
                original["cases"][0]["per_class"][5]["dice"] = .99
                save_json(directories["improved3d"] / "evaluation.json", original)
                with self.assertRaisesRegex(ValueError, "differs from recount"):
                    analysis.analyse(manifest, directories, root / "tamper")
                self.assertEqual(load_json(root / "tamper/analysis_receipt.json")["status"], "failed")


if __name__ == "__main__":
    unittest.main()
