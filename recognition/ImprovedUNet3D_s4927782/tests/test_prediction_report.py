"""Synthetic receipt tests; no MRI, checkpoint or inference is used."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import predict_and_report as report
from utils import save_json, sha256_file


def write_fixture(folder, status="completed"):
    output = folder / "prediction_native.nii.gz"
    output.write_bytes(b"synthetic output bytes, not an MRI")
    receipt = {"status": status, "model": "improved3d", "selected_epoch": 25,
               "native_export": {"shape_xyz": [8, 10, 12]},
               "timing_seconds": {"inference": .125},
               "outputs": {output.name: sha256_file(output)}}
    path = folder / "prediction_receipt.json"
    save_json(path, receipt)
    return path.resolve()


class PredictionReportTests(unittest.TestCase):
    def test_existing_receipt_prints_results_without_calling_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write_fixture(Path(temporary))
            stdout = io.StringIO()
            with patch.object(report, "run_image_prediction", side_effect=AssertionError("Inference called")) as infer:
                with redirect_stdout(stdout):
                    self.assertEqual(report.main(["--receipt", str(path)]), 0)
            infer.assert_not_called()
            text = stdout.getvalue()
            for expected in ("improved3d", "Selected epoch: 25", "8 x 10 x 12", "0.125 seconds",
                             str(path.parent / "prediction_native.nii.gz"), f"Receipt: {path}"):
                self.assertIn(expected, text)

    def test_incomplete_receipt_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write_fixture(Path(temporary), status="running")
            with self.assertRaisesRegex(ValueError, "not completed"):
                report.load_verified_receipt(path)

    def test_modified_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            path = write_fixture(folder)
            (folder / "prediction_native.nii.gz").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                report.load_verified_receipt(path)


if __name__ == "__main__":
    unittest.main()
