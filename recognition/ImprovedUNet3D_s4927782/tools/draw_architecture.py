"""Draw the frozen compact 3D architecture and image-only inference pipeline.

The figure is a code-native schematic, not a learned-image output. Architecture,
parameter counts, patch geometry and inference coverage are checked against the
saved implementation/configuration before rendering. No MRI or weights are read.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import inspect
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from modules import ImprovedUNet3D, UNet2D, segmentation_loss
from train import window_starts

PURPLE = "#51247A"
INK = "#202834"
MUTED = "#506071"
ENCODER = "#E9E1F2"
DECODER = "#DCEBF4"
HEAD = "#E2F0E8"
AUXILIARY = "#FBEDCF"
BORDER = "#758493"


def box(ax, x, y, width, height, text, fill, *, weight="normal", fontsize=13):
    patch = FancyBboxPatch((x-width/2, y-height/2), width, height,
                          boxstyle="round,pad=0.035,rounding_size=0.08",
                          facecolor=fill, edgecolor=BORDER, linewidth=1.1,
                          zorder=3)
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", color=INK,
            fontsize=fontsize, fontweight=weight, linespacing=1.3, zorder=4)
    return patch


def arrow(ax, start, end, *, color=MUTED, dashed=False, width=1.5):
    patch = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                            linewidth=width, color=color,
                            linestyle=(0, (4, 3)) if dashed else "solid",
                            shrinkA=0, shrinkB=0, zorder=2)
    ax.add_patch(patch)


def validate_sources(config):
    model = ImprovedUNet3D(config["num_classes"], config["base_channels"])
    baseline = UNet2D(config["num_classes"], config["base_channels"])
    widths = [layer.layers[0].out_channels for layer in model.encoders]
    aux_weights = inspect.signature(segmentation_loss).parameters["aux_weights"].default
    windows = int(np.prod([len(window_starts(n, k, config["inference_overlap"]))
                           for n, k in zip(config["target_shape_dhw"], config["patch_shape_dhw"])]))
    details = {"encoder_channels": widths,
               "parameters_3d": sum(x.numel() for x in model.parameters()),
               "parameters_2d": sum(x.numel() for x in baseline.parameters()),
               "aux_weights": list(aux_weights), "inference_windows": windows,
               "input_shape_dhw": config["target_shape_dhw"],
               "patch_shape_dhw": config["patch_shape_dhw"],
               "source_sha256": {name: sha256((PROJECT/name).read_bytes()).hexdigest()
                                 for name in ("modules.py", "train.py", "config.json")}}
    assert widths == [16, 32, 64, 128]
    assert len(model.aux_heads) == 2 and model.head.out_channels == 6
    assert tuple(aux_weights) == (.5, .25) and windows == 8
    assert details["parameters_3d"] == 1627634 and details["parameters_2d"] == 481830
    assert config["target_shape_dhw"] == [96, 192, 192]
    assert config["patch_shape_dhw"] == [64, 128, 128]
    assert config["train_patches_per_scan"] == 2
    assert config["inference_blending"] == "uniform_softmax_mean"
    assert config["mask_outside_nonzero_support"] is True
    return details


def draw(output_dir: Path):
    config = json.loads((PROJECT/"config.json").read_text())
    details = validate_sources(config)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
                         "svg.fonttype": "none", "svg.hashsalt": "hipmri-architecture-v1"})
    fig = plt.figure(figsize=(9.5, 8.4), facecolor="white")
    ax = fig.add_axes([.025, .325, .95, .61])
    ax.set(xlim=(-.2, 11.7), ylim=(-.25, 8.5))
    ax.axis("off")
    fig.text(.04, .975, "Residual 3D U-Net with deep supervision", fontsize=17,
             fontweight="bold", color=PURPLE, va="top")
    fig.text(.04, .943, "Fixed compact architecture | channels and D × H × W shown", fontsize=13,
             color=MUTED, va="top")

    rows = [6.50, 4.70, 2.90, 1.10]
    dims = ["64×128×128", "32×64×64", "16×32×32", "8×16×16"]
    left, right, head_x = 1.65, 6.35, 9.95
    enc_w, dec_w, head_w, height = 2.7, 3.15, 2.65, 1.0
    ax.text(left, 7.70, "Input: 1 channel\n64×128×128 patch", ha="center", va="center",
            fontsize=13, color=INK, linespacing=1.15)
    arrow(ax, (left, 7.24), (left, 7.05))
    for index, (channels, y, size) in enumerate(zip(details["encoder_channels"], rows, dims)):
        box(ax, left, y, enc_w, height, f"Residual | {channels} ch\n{size}", ENCODER)
        if index < 3:
            arrow(ax, (left, y-.54), (left, rows[index+1]+.54), color=PURPLE)
            ax.text(.05, (y+rows[index+1])/2, "Pool ×2", fontsize=13,
                    ha="left", va="center", color=PURPLE)
    for index in (2, 1, 0):
        channels, y = details["encoder_channels"][index], rows[index]
        box(ax, right, y, dec_w, height,
            f"Concat + residual\n{channels} ch | {dims[index]}", DECODER)
        arrow(ax, (left+enc_w/2+.07, y), (right-dec_w/2-.08, y), dashed=True)
        ax.text(3.85, y+.2, "Skip", fontsize=13, color=MUTED, ha="center")
        if index == 2:
            # The first up-step is routed from the bottleneck through open space.
            arrow(ax, (left+enc_w/2+.07, rows[3]), (right, rows[3]), color="#286184")
            arrow(ax, (right, rows[3]), (right, y-.57), color="#286184")
        else:
            arrow(ax, (right, rows[index+1]+.54), (right, y-.57), color="#286184")
        ax.text(right+.14, y-.9, "Up-step", fontsize=13, color="#286184", va="center")
        arrow(ax, (right+dec_w/2+.08, y), (head_x-head_w/2-.08, y),
              color="#477956" if index == 0 else "#AD8532")
        main = index == 0
        label = "Primary" if main else "Auxiliary"
        loss_weight = "1" if main else "0.5" if index == 1 else "0.25"
        box(ax, head_x, y, head_w, 1.18,
            f"{label}: 6 classes\n{dims[index]}\nLoss weight {loss_weight}",
            HEAD if main else AUXILIARY)
    ax.text(.05, .12, "Up-step: trilinear resize → 3×3×3 conv → skip concat → residual unit.",
            fontsize=13, color=INK, va="center")
    ax.text(.05, -.34, "Residual unit: two 3×3×3 convs, GroupNorm, LeakyReLU + identity/1×1×1 shortcut.",
            fontsize=13, color=INK, va="center")

    strip = fig.add_axes([.025, .065, .95, .23])
    strip.set(xlim=(-.2, 11.7), ylim=(-.4, 3.55))
    strip.axis("off")
    strip.text(.05, 3.38, "TRAIN: two patches/scan/epoch; label-guided centres only during training.",
               fontsize=13, color=INK, va="top")
    strip.text(.05, 2.82, "LOSS: (Lmain + 0.5 Lhalf + 0.25 Lquarter) / 1.75; each L = weighted CE + Dice.",
               fontsize=13, color=INK, va="top")
    xs, box_w = [1.13, 4.05, 6.97, 9.89], 2.6
    texts = ["Raw MRI → RAS\nPer-image z-score\nclip; 96×192×192",
             "8 overlapping\n64×128×128\nwindows",
             "Mean softmax\n6-class argmax\nzero-input mask",
             "Nearest labels\nrestore original\ncoordinates"]
    for index, (x, text) in enumerate(zip(xs, texts)):
        box(strip, x, 1.47, box_w, 1.42, text, "#F0F3F6")
        if index < 3:
            arrow(strip, (x+box_w/2+.03, 1.47), (xs[index+1]-box_w/2-.03, 1.47))
    strip.text(.05, .27, "INFERENCE: primary head only; complete image, with no target-derived ROI.",
               fontsize=13, color=INK, va="center")
    fig.text(.04, .035,
             "2D U-Net: 481,830 parameters; 3D: 1,627,634. Exact compact adaptation, not nnU-Net.",
             fontsize=13, color=MUTED, va="center")
    # Ensure source-derived labels remain inside the delivered canvas.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in [*fig.texts, *ax.texts, *strip.texts]:
        bounds = text.get_window_extent(renderer)
        assert bounds.x0 >= 0 and bounds.x1 <= fig.bbox.width, text.get_text()
        assert bounds.y0 >= 0 and bounds.y1 <= fig.bbox.height, text.get_text()
    output_dir.mkdir(parents=True, exist_ok=True)
    description = json.dumps(details, sort_keys=True)
    fig.savefig(output_dir/"architecture.png", dpi=300, facecolor="white",
                metadata={"Title": "Compact HipMRI architecture", "Description": description})
    fig.savefig(output_dir/"architecture.svg", facecolor="white",
                metadata={"Title": "Compact HipMRI architecture", "Description": description, "Date": None})
    svg = output_dir / "architecture.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    return details


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT/"figures")
    details = draw(parser.parse_args(argv).output_dir)
    print(json.dumps(details, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
