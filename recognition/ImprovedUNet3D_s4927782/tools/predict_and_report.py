"""Image-only prediction entry point with a readable console result.

The underlying predict.py implementation and its checkpoint source checks remain
unchanged. --receipt verifies and prints an existing result without inference.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import load_json, sha256_file


def load_verified_receipt(path):
    """Read completed metadata and hash output bytes; never load image arrays."""
    path = Path(path).resolve(strict=True)
    receipt = load_json(path)
    if receipt.get("status") != "completed":
        raise ValueError("Prediction receipt is not completed.")
    if receipt.get("model") not in ("unet2d", "improved3d"):
        raise ValueError("Unknown prediction model.")
    epoch = receipt.get("selected_epoch")
    if type(epoch) is not int or epoch < 1:
        raise ValueError("Selected epoch must be a positive integer.")
    shape = receipt.get("native_export", {}).get("shape_xyz")
    if not isinstance(shape, list) or len(shape) != 3 or any(type(n) is not int or n < 1 for n in shape):
        raise ValueError("Native shape must contain three positive integer dimensions.")
    seconds = receipt.get("timing_seconds", {}).get("inference")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Inference timing is missing or invalid.")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError("Receipt declares no prediction outputs.")
    for name, digest in outputs.items():
        if not isinstance(name, str) or Path(name).name != name or name in (".", ".."):
            raise ValueError("Output names must be plain filenames beside the receipt.")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("Invalid declared output SHA256.")
        output = (path.parent / name).resolve(strict=True)
        if output.parent != path.parent or not output.is_file():
            raise ValueError("Prediction output escapes the receipt directory.")
        if sha256_file(output) != digest:
            raise ValueError(f"Prediction output hash differs: {name}")
    return receipt


def format_summary(receipt_path, receipt):
    """Format only verified receipt fields; no model or image access."""
    receipt_path = Path(receipt_path).resolve()
    shape = " x ".join(str(n) for n in receipt["native_export"]["shape_xyz"])
    lines = [f"Prediction completed: {receipt['model']}",
             f"Selected epoch: {receipt['selected_epoch']}",
             f"Native shape (X, Y, Z): {shape}",
             f"Inference time: {receipt['timing_seconds']['inference']:.3f} seconds",
             "Verified outputs:"]
    lines.extend(f"  {receipt_path.parent / name}" for name in sorted(receipt["outputs"]))
    lines.append(f"Receipt: {receipt_path}")
    return "\n".join(lines)


def run_image_prediction(args):
    # Lazy import keeps --receipt entirely separate from model/image loading.
    from predict import predict_image
    from train import choose_device, read_config
    predict_image(args.image, args.checkpoint, read_config(args.config),
                  args.output_dir, choose_device(args.device))
    return args.output_dir / "prediction_receipt.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image", type=Path, help="Raw MRI NIfTI for image-only prediction.")
    mode.add_argument("--receipt", type=Path, help="Verify and print an existing completed result; no inference.")
    parser.add_argument("--checkpoint", type=Path, help="Selected-best checkpoint; required with --image.")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output-dir", type=Path, help="Fresh output directory; required with --image.")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    args = parser.parse_args(argv)
    if args.receipt is not None:
        if args.checkpoint is not None or args.output_dir is not None:
            parser.error("--receipt cannot be combined with --checkpoint or --output-dir.")
        receipt_path = args.receipt
    else:
        if args.checkpoint is None or args.output_dir is None:
            parser.error("--image requires --checkpoint and --output-dir.")
        receipt_path = run_image_prediction(args)
    receipt = load_verified_receipt(receipt_path)
    print(format_summary(receipt_path, receipt), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
