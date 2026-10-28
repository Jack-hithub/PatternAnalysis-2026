"""Independent small examples for overlap, weighting, distances and model loss."""
from __future__ import annotations

import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import aggregate_patients, class_metrics, load_json, save_json

try:
    import torch
    from modules import ImprovedUNet3D, UNet2D, segmentation_loss
except ImportError:
    torch = None

try:
    import scipy
except ImportError:
    scipy = None


class MetricTests(unittest.TestCase):
    def test_known_intersections_and_confusion(self):
        target = np.array([[0, 1], [1, 2]])
        pred = np.array([[0, 1], [2, 2]])
        result = class_metrics(pred, target, num_classes=4)
        self.assertEqual(result["confusion_matrix"],
                         [[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
        for c in (1, 2):
            self.assertAlmostEqual(result["per_class"][c]["dice"], 2 / 3)
            self.assertAlmostEqual(result["per_class"][c]["iou"], 1 / 2)
        self.assertEqual(result["per_class"][1]["fn"], 1)
        self.assertEqual(result["per_class"][2]["fp"], 1)
        self.assertAlmostEqual(result["foreground_mean_dice"], 2 / 3)
        # Valid unsigned NIfTI-style IDs must count identically to signed IDs.
        unsigned = class_metrics(pred.astype(np.uint64), target.astype(np.uint64), 4)
        self.assertEqual(unsigned, result)

    def test_empty_and_one_sided_classes_are_distinguished(self):
        r = class_metrics(np.array([3, 0]), np.array([0, 0]), 4)
        self.assertIsNone(r["per_class"][1]["dice"])
        self.assertEqual(r["per_class"][1]["surface_status"], "both_empty")
        self.assertEqual(r["per_class"][3]["dice"], 0)
        self.assertEqual(r["per_class"][3]["fp"], 1)
        self.assertEqual(r["per_class"][3]["surface_status"], "target_empty")

    def test_reject_nonlabel_shapes_and_ranges(self):
        for pred, target in [(np.array([0.0]), np.array([0])),
                             (np.array([6]), np.array([0])),
                             (np.array([0]), np.array([-1])),
                             (np.zeros((1, 2), int), np.zeros(2, int))]:
            with self.subTest(pred=pred, target=target), self.assertRaises(ValueError):
                class_metrics(pred, target)

    def test_hd95_requires_real_spacing(self):
        with self.assertRaises(ValueError):
            class_metrics(np.zeros((3, 3, 3), int), np.zeros((3, 3, 3), int),
                          include_hd95=True)

    def test_patient_weighting_is_not_visit_or_voxel_pooling(self):
        cases = []
        for patient, visit, correct in [("A", "0", True), ("A", "1", False), ("B", "0", True)]:
            m = class_metrics(np.array([1 if correct else 0]), np.array([1]), 2)
            cases.append({"patient": patient, "visit": visit, **m})
        result = aggregate_patients(cases, 2)
        self.assertAlmostEqual(result["per_class"][1]["dice"], .75)
        self.assertAlmostEqual(result["foreground_mean_dice"], .75)
        self.assertNotAlmostEqual(result["per_class"][1]["dice"], 2 / 3)

    def test_duplicate_or_misordered_cases_are_rejected(self):
        case = {"patient": "A", "visit": "0", **class_metrics(np.array([1]), np.array([1]), 2)}
        with self.assertRaises(ValueError):
            aggregate_patients([case, case], 2)
        case["per_class"] = list(reversed(case["per_class"]))
        with self.assertRaises(ValueError):
            aggregate_patients([case], 2)

    def test_nonfinite_json_does_not_replace_existing_record(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "record.json"
            save_json(p, {"valid": 1})
            with self.assertRaises(ValueError):
                save_json(p, {"invalid": float("nan")})
            self.assertEqual(load_json(p), {"valid": 1})

    @unittest.skipIf(scipy is None, "SciPy is required for physical surface distance")
    def test_surface_distance_accounts_for_anisotropic_spacing(self):
        true = np.zeros((4, 4, 4), np.uint8)
        pred = np.zeros_like(true)
        true[1, 1, 1] = pred[2, 1, 1] = 1
        r = class_metrics(pred, true, 2, spacing=(2, 3, 4), include_hd95=True)
        self.assertEqual(r["per_class"][1]["dice"], 0)
        self.assertAlmostEqual(r["per_class"][1]["hd95_mm"], 2)

    @unittest.skipIf(scipy is None, "SciPy is required for physical surface distance")
    def test_empty_prediction_keeps_zero_dice_and_undefined_distance(self):
        true = np.zeros((3, 3, 3), np.uint8)
        true[1, 1, 1] = 1
        r = class_metrics(np.zeros_like(true), true, 3, spacing=(1, 1, 1), include_hd95=True)
        self.assertEqual(r["per_class"][1]["dice"], 0)
        self.assertIsNone(r["per_class"][1]["hd95_mm"])
        self.assertEqual(r["per_class"][1]["surface_status"], "prediction_empty")
        aggregate = aggregate_patients([{"patient": "A", "visit": "0", **r}], 3)
        self.assertEqual(aggregate["per_class"][1]["dice"], 0)
        self.assertEqual(aggregate["per_class"][1]["hd95_mm_defined_cases"], 0)


@unittest.skipIf(torch is None, "PyTorch is required for actual model/gradient checks")
class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_2d_odd_shape_and_real_backward(self):
        model = UNet2D(6, 4)
        result = model(torch.randn(2, 1, 17, 19))
        self.assertEqual(tuple(result["logits"].shape), (2, 6, 17, 19))
        self.assertEqual(result["aux_logits"], [])
        loss = segmentation_loss(result, torch.randint(0, 6, (2, 17, 19)))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(model.head.weight.grad.abs().sum().item(), 0)

    def test_3d_odd_shape_reaches_both_deep_supervision_heads(self):
        model = ImprovedUNet3D(6, 4)
        result = model(torch.randn(1, 1, 9, 11, 13))
        self.assertEqual(tuple(result["logits"].shape), (1, 6, 9, 11, 13))
        self.assertEqual(len(result["aux_logits"]), 2)
        loss = segmentation_loss(result, torch.randint(0, 6, (1, 9, 11, 13)),
                                 class_weights=torch.tensor([.1, .1, .4, 1, 2, 2.4]))
        loss.backward()
        for head in model.aux_heads:
            self.assertTrue(torch.isfinite(head.weight.grad).all())
            self.assertGreater(head.weight.grad.abs().sum().item(), 0)

    def test_weighted_ce_and_dice_match_hand_calculation(self):
        logits = torch.log(torch.tensor([[[[.8, .4]], [[.2, .6]]]]))
        target = torch.tensor([[[0, 1]]])
        actual = segmentation_loss({"logits": logits, "aux_logits": []}, target,
                                   class_weights=torch.tensor([1., 3.]), smooth=1e-5)
        ce = (-math.log(.8) - 3 * math.log(.6)) / 4
        dice = ((1.6 + 1e-5) / (2.2 + 1e-5) + (1.2 + 1e-5) / (1.8 + 1e-5)) / 2
        self.assertAlmostEqual(actual.item(), ce + 1 - dice, places=6)

    def test_loss_rejects_one_hot_target_and_zero_weights(self):
        output = {"logits": torch.zeros(1, 3, 8, 8), "aux_logits": []}
        with self.assertRaises(ValueError):
            segmentation_loss(output, torch.zeros(1, 8, 8))
        with self.assertRaises(ValueError):
            segmentation_loss(output, torch.zeros(1, 8, 8, dtype=torch.long),
                              class_weights=torch.zeros(3))


if __name__ == "__main__":
    unittest.main()
