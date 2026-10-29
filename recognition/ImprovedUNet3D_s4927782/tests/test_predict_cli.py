"""Mocked command-line output checks; no images or models are loaded."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
import unittest
from unittest.mock import patch, sentinel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import predict


class PredictCliTests(unittest.TestCase):
    def test_direct_cli_prints_one_prediction_summary(self):
        image = ROOT / "artifacts" / "mock_scan.nii.gz"
        checkpoint = ROOT / "artifacts" / "mock_best.pt"
        config_path = ROOT / "config.json"
        output = ROOT / "artifacts" / "mock_cli_output"
        config = {"mocked_configuration": True}
        names = ("prediction_native.nii.gz", "prediction_model_grid.npz", "prediction_overlay.png")
        receipt = {"model": "improved3d", "selected_epoch": 30,
                   "native_export": {"shape_xyz": [256, 256, 128]},
                   "timing_seconds": {"inference": 1.23456},
                   "outputs": {name: "mocked_hash" for name in names}}
        captured = StringIO()
        with patch.object(predict, "read_config", return_value=config) as read_config, \
             patch.object(predict, "choose_device", return_value=sentinel.device) as choose_device, \
             patch.object(predict, "predict_image", return_value=receipt) as predict_image, \
             redirect_stdout(captured):
            result = predict.main(["--image", str(image), "--checkpoint", str(checkpoint),
                                   "--config", str(config_path), "--output-dir", str(output),
                                   "--device", "cpu"])
        self.assertEqual(result, 0)
        read_config.assert_called_once_with(config_path)
        choose_device.assert_called_once_with("cpu")
        predict_image.assert_called_once_with(image, checkpoint, config, output, sentinel.device)
        expected = ["Prediction completed: improved3d", "Selected epoch: 30",
                    "Native shape (X, Y, Z): 256 x 256 x 128", "Inference time: 1.235 seconds",
                    "Verified outputs:"]
        expected.extend(f"  {output / name}" for name in sorted(names))
        expected.append(f"Receipt: {output / 'prediction_receipt.json'}")
        self.assertEqual(captured.getvalue(), "\n".join(expected) + "\n")

    def test_wrapper_calls_prediction_function_and_prints_summary_once(self):
        from tools import predict_and_report as report
        receipt = {"model": "unet2d", "selected_epoch": 30,
                   "native_export": {"shape_xyz": [8, 10, 12]},
                   "timing_seconds": {"inference": .125},
                   "outputs": {"prediction_native.nii.gz": "mocked_hash"}}
        captured = StringIO()
        with patch.object(predict, "read_config", return_value={}), \
             patch.object(predict, "choose_device", return_value=sentinel.device), \
             patch.object(predict, "predict_image", return_value=receipt) as predict_image, \
             patch.object(predict, "main", side_effect=AssertionError("Wrapper called the direct CLI")), \
             patch.object(report, "load_verified_receipt", return_value=receipt), \
             redirect_stdout(captured):
            result = report.main(["--image", "mock_scan.nii.gz", "--checkpoint", "mock_best.pt",
                                  "--output-dir", "mock_cli_output", "--device", "cpu"])
        self.assertEqual(result, 0)
        predict_image.assert_called_once()
        self.assertEqual(captured.getvalue().count("Prediction completed:"), 1)
        self.assertIn("Prediction completed: unet2d", captured.getvalue())


if __name__ == "__main__":
    unittest.main()
