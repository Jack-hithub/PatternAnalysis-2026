"""Render an A4-readable class chart from existing anonymous metric rows.

This changes presentation only. It does not load predictions, calculate scores,
perform inference or alter the archived analysis outputs and their receipts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = args.table.read_bytes()
    with args.table.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if [int(row["class_index"]) for row in rows] != list(range(6)):
        raise ValueError("Expected the existing ordered six-class metric table.")
    foreground = rows[1:]
    for row in foreground:
        for name in ("unet2d", "improved3d"):
            for metric in ("dice", "iou"):
                value = float(row[f"{name}_{metric}"])
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("Existing overlap values must be finite and in [0, 1].")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 14,
                         "axes.titlesize": 16, "axes.labelsize": 14,
                         "xtick.labelsize": 14, "ytick.labelsize": 14,
                         "legend.fontsize": 14}):
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.8))
        positions = np.arange(len(foreground))
        for ax, metric in zip(axes, ("dice", "iou")):
            for shift, name, label, colour in [(-.18, "unet2d", "2D U-Net", "#4c78a8"),
                                               (.18, "improved3d", "3D Improved U-Net", "#f58518")]:
                values = [float(row[f"{name}_{metric}"]) for row in foreground]
                ax.bar(positions + shift, values, width=.36, label=label, color=colour, zorder=3)
            ax.set_xticks(positions, [row["class_name"] for row in foreground], rotation=25, ha="right")
            ax.set_ylim(0, 1.02)
            ax.set_yticks(np.arange(0, 1.01, .2))
            ax.set_ylabel("Dice" if metric == "dice" else "IoU")
            ax.set_title("Patient-balanced Dice" if metric == "dice" else "Patient-balanced IoU", pad=14)
            ax.grid(axis="y", alpha=.23, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
                   bbox_to_anchor=(.5, .075))
        fig.suptitle("TEST overlap on the common reporting grid", fontsize=16, y=.98)
        fig.text(.5, .025, "Foreground classes 1–5; background is reported separately.",
                 ha="center", fontsize=14)
        fig.subplots_adjust(left=.085, right=.98, bottom=.29, top=.82, wspace=.34)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.stem + ".reporting.tmp.png")
        fig.savefig(temporary, dpi=200, metadata={"Description": "Existing patient-balanced TEST scores; presentation-only redraw."})
        plt.close(fig)
        temporary.replace(args.output)
    print(json.dumps({"source_table_sha256": hashlib.sha256(payload).hexdigest(),
                      "output": str(args.output), "pixels": [1900, 1160],
                      "source_inches": [9.5, 5.8], "dpi": 200, "minimum_font_pt": 14,
                      "minimum_print_width_cm_for_9pt": 9 * 9.5 * 2.54 / 14,
                      "scores_recomputed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
