"""Predict a raw MRI with selected frozen weights; no labels are accepted.

Outputs are restored to the source NIfTI grid. Restoring coordinates does not
recover detail lost when resizing the image for the model.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch

import dataset as data
from train import (checkpoint_runtime_compatibility, choose_device, create_model, fresh_dir, infer_volume,
                   load_selected_checkpoint, read_config, require,
                   runtime_identity, source_identity, sync, utc_now)
from utils import save_json, seed_everything, sha256_file


def save_native_prediction(prediction_dhw, metadata, image_path, destination):
    """Nearest-neighbour label restoration to the original shape and affine.

    This function receives prediction/image geometry only. Original qform and
    sform header codes are preserved; reloaded file geometry is verified.
    """
    original = nib.load(str(image_path))
    shape = tuple(metadata["original_shape_xyz"])
    affine = np.asarray(metadata["original_affine_xyz"], dtype=float)
    require(original.shape == shape and np.allclose(original.affine, affine, rtol=0, atol=1e-6),
            "Original image geometry changed after preprocessing.")
    restored_dhw = data.resample_label(
        prediction_dhw, metadata["resampled_affine_xyz"], affine, shape[::-1])
    restored_xyz = np.ascontiguousarray(restored_dhw.transpose(2, 1, 0), dtype=np.uint8)
    header = original.header.copy()
    header.set_data_dtype(np.uint8)
    output = original.__class__(restored_xyz, original.affine, header)
    output.header.set_slope_inter(1, 0)
    qform, qcode = original.get_qform(coded=True)
    sform, scode = original.get_sform(coded=True)
    output.set_qform(qform, int(qcode))
    output.set_sform(sform, int(scode))
    destination = Path(destination)
    temporary = destination.with_name(destination.stem + ".tmp.nii.gz")
    nib.save(output, str(temporary))
    temporary.replace(destination)
    check = nib.load(str(destination))
    require(check.shape == original.shape and np.allclose(check.affine, original.affine, rtol=0, atol=1e-5),
            "Exported prediction does not match original geometry.")
    require(check.get_data_dtype() == np.dtype(np.uint8), "Exported prediction must be integer labels.")
    return {"shape_xyz": list(check.shape), "affine_xyz": check.affine.tolist(),
            "dtype": str(check.get_data_dtype()), "qform_code": int(check.header["qform_code"]),
            "sform_code": int(check.header["sform_code"]),
            "geometry_note": "Original voxel coordinates restored with nearest-neighbour labels; model-grid information limits remain."}


def save_prediction_overlay(image, prediction, destination):
    """Three orthogonal prediction-only panels, located without a target mask."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.colors import ListedColormap

    support = prediction == 5
    if not support.any():
        support = prediction > 0
    centre = [int(np.argmax(support.sum(axis=tuple(j for j in range(3) if j != axis))))
              if support.any() else image.shape[axis] // 2 for axis in range(3)]
    z, y, x = centre
    planes = [(image[z], prediction[z], f"Axial D={z}"),
              (image[:, y, :], prediction[:, y, :], f"Coronal H={y}"),
              (image[:, :, x], prediction[:, :, x], f"Sagittal W={x}")]
    cmap = ListedColormap(["black", "#f4cf67", "#76a9ed", "#66bb88", "#e88dba", "#e65353"])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (raw, labels, title) in zip(axes, planes):
        ax.imshow(raw, cmap="gray", interpolation="nearest", origin="lower")
        ax.imshow(np.ma.masked_equal(labels, 0), cmap=cmap, vmin=-.5, vmax=5.5,
                  interpolation="nearest", alpha=.5, origin="lower")
        ax.set_title(title)
        ax.set_axis_off()
    fig.suptitle("Model-grid prediction overlay; planes chosen from prediction only")
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return {"plane_indices_dhw": centre, "selection": "max predicted prostate area, else foreground, else image centre"}


def predict_image(image_path, checkpoint_path, config, output_dir, device):
    image_path, checkpoint_path = Path(image_path), Path(checkpoint_path)
    image_sha, checkpoint_sha = sha256_file(image_path), sha256_file(checkpoint_path)
    source_sha = source_identity()
    checkpoint = load_selected_checkpoint(checkpoint_path, config)
    compatibility = checkpoint_runtime_compatibility(checkpoint_path, checkpoint, config, source_sha)
    seed_everything(config["seed"])
    output = fresh_dir(output_dir)
    model = create_model(checkpoint["model"], config).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    start = time.perf_counter()
    image, support, metadata = data.preprocess_image(image_path, config["target_shape_dhw"], return_support=True)
    preprocess_seconds = time.perf_counter() - start
    sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    prediction = infer_volume(model, image, checkpoint["model"], config, device, support_dhw=support)
    sync(device)
    inference_seconds = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    start = time.perf_counter()
    native = save_native_prediction(prediction, metadata, image_path, output / "prediction_native.nii.gz")
    np.savez_compressed(output / "prediction_model_grid.npz", prediction=prediction,
                        affine_xyz=np.asarray(metadata["resampled_affine_xyz"]),
                        spacing_dhw=np.asarray(metadata["spacing_dhw"]))
    overlay = save_prediction_overlay(image, prediction, output / "prediction_overlay.png")
    require(sha256_file(image_path) == image_sha and sha256_file(checkpoint_path) == checkpoint_sha
            and source_identity() == source_sha, "Input, checkpoint or sources changed during prediction.")
    receipt = {"status": "completed", "created_at": utc_now(), "model": checkpoint["model"],
               "selected_epoch": checkpoint["epoch"], "image_sha256": image_sha,
               "checkpoint_sha256": checkpoint_sha, "config_sha256": data.config_digest(config),
               "source_sha256": source_sha, "runtime": runtime_identity(device),
               "runtime_compatibility": compatibility, "support_policy": data.SUPPORT_POLICY,
               "support_masking_applied": bool(config.get("mask_outside_nonzero_support", False)),
               "preprocessing": metadata, "native_export": native, "overlay": overlay,
               "timing_seconds": {"preprocessing": preprocess_seconds, "inference": inference_seconds,
                                  "export_and_verification": time.perf_counter()-start},
               "peak_allocated_cuda_bytes": peak,
               "inference_timing_note": "Single cold prediction; excludes model load, preprocessing and export; synchronized on CUDA.",
               "label_use": "No ground-truth image or labels supplied to prediction.",
               "outputs": {path.name: sha256_file(path) for path in sorted(output.iterdir()) if path.is_file()}}
    save_json(output / "prediction_receipt.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Raw 3D MRI NIfTI; no mask argument.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Selected-best checkpoint produced by train.py.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing output is not overwritten.")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    args = parser.parse_args(argv)
    predict_image(args.image, args.checkpoint, read_config(args.config), args.output_dir, choose_device(args.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
