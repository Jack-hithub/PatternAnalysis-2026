"""Plot completed, recorded TRAIN/VAL traces without smoothing or model reruns.

Only history.json and run.json are read. The seven scheduled VAL observations
select checkpoints; TEST results and trained arrays are never loaded here.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
SCHEDULE = [1, 5, 10, 15, 20, 25, 30]
STYLES = {"unet2d": {"label": "2D U-Net", "color": "#176D9C", "marker": "o"},
          "improved3d": {"label": "3D Improved U-Net", "color": "#703894", "marker": "s"}}


def read_completed_run(run_dir: Path, name: str):
    paths = {key: run_dir/name/(key+".json") for key in ("history", "run")}
    encoded = {key: path.read_bytes() for key, path in paths.items()}
    hashes = {key+"_sha256": sha256(value).hexdigest() for key, value in encoded.items()}
    history, run = json.loads(encoded["history"]), json.loads(encoded["run"])
    if run.get("status") != "completed" or run.get("completed_epochs") != 30:
        raise ValueError(f"{name}: plot only a completed fixed thirty-epoch run.")
    if [r.get("epoch") for r in history] != list(range(1, 31)):
        raise ValueError(f"{name}: missing, repeated or unordered recorded epochs.")
    if any(not isinstance(r.get("train_loss"), (int, float)) or not math.isfinite(r["train_loss"])
           or r["train_loss"] < 0 for r in history):
        raise ValueError(f"{name}: invalid recorded TRAIN loss.")
    validation = [r for r in history if r.get("validation_prostate_dice") is not None]
    if [r["epoch"] for r in validation] != SCHEDULE:
        raise ValueError(f"{name}: VAL observations differ from the fixed schedule.")
    if any(not math.isfinite(r["validation_prostate_dice"]) or not 0 <= r["validation_prostate_dice"] <= 1
           for r in validation):
        raise ValueError(f"{name}: invalid VAL Dice.")
    selected = max(validation, key=lambda r: (r["validation_prostate_dice"], -r["epoch"]))
    if (run.get("best_epoch") != selected["epoch"] or
            run.get("best_validation_prostate_dice") != selected["validation_prostate_dice"]):
        raise ValueError(f"{name}: selected checkpoint differs from the earliest scheduled maximum.")
    if run.get("config", {}).get("validation_epochs") != SCHEDULE:
        raise ValueError(f"{name}: recorded configuration has a different VAL schedule.")
    attempted, steps, skipped = (sum(r[k] for r in history) for k in (
        "training_batches", "optimizer_steps", "amp_overflow_skipped_optimizer_steps"))
    if attempted != steps+skipped or steps != run.get("actual_optimizer_steps"):
        raise ValueError(f"{name}: actual update/AMP skip accounting is inconsistent.")
    if any(path.read_bytes() != encoded[key] for key, path in paths.items()):
        raise ValueError(f"{name}: source records changed while loading.")
    detail = {**hashes, "completed_epochs": 30, "validation_epochs": SCHEDULE,
              "selected_epoch": selected["epoch"],
              "selected_validation_patient_prostate_dice": selected["validation_prostate_dice"],
              "attempted_updates": attempted, "optimizer_steps": steps, "amp_skips": skipped}
    return history, validation, selected, detail


def plot(run_dir: Path, output_dir: Path):
    loaded = {name: read_completed_run(run_dir, name) for name in STYLES}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
                         "axes.titlesize": 14, "axes.labelsize": 13,
                         "xtick.labelsize": 13, "ytick.labelsize": 13,
                         "legend.fontsize": 13})
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.8), facecolor="white")
    fig.subplots_adjust(left=.08, right=.98, bottom=.32, top=.82, wspace=.27)
    fig.suptitle("Recorded training and validation traces", x=.04, y=.98,
                 ha="left", fontsize=17, fontweight="bold", color="#51247A")
    for name, style in STYLES.items():
        history, validation, selected, _ = loaded[name]
        axes[0].plot([r["epoch"] for r in history], [r["train_loss"] for r in history],
                     color=style["color"], marker=style["marker"], markersize=3.3,
                     linewidth=1.7, label=style["label"])
        axes[1].plot([r["epoch"] for r in validation],
                     [r["validation_prostate_dice"] for r in validation],
                     color=style["color"], marker=style["marker"], markersize=5.5,
                     linewidth=1.7, label=style["label"])
        axes[1].scatter([selected["epoch"]], [selected["validation_prostate_dice"]],
                        marker="D", s=105, facecolor="white", edgecolor=style["color"],
                        linewidth=2.1, zorder=6)
    axes[0].set(title="TRAIN: recorded mean loss", xlabel="Epoch", ylabel="Weighted CE + Dice",
                xlim=(0, 31), xticks=[1, 10, 20, 30], ylim=(0, None))
    axes[1].set(title="VAL96: patient prostate Dice", xlabel="Scheduled epoch", ylabel="Patient-balanced Dice",
                xlim=(0, 31), xticks=SCHEDULE, ylim=(0, 1))
    axes[0].legend(loc="upper right", frameon=False, handlelength=1.5)
    axes[1].legend(loc="lower right", frameon=False, handlelength=1.5)
    for ax in axes:
        ax.grid(True, alpha=.20)
        ax.spines[["top", "right"]].set_visible(False)
    selected_epochs = {loaded[name][2]["epoch"] for name in STYLES}
    selected_text = ("both checkpoints selected at epoch "+str(next(iter(selected_epochs)))) if len(selected_epochs) == 1 else (
        "; ".join(STYLES[name]["label"]+" selected at epoch "+str(loaded[name][2]["epoch"]) for name in STYLES))
    fig.text(.04, .16, "Inputs and auxiliary heads differ; TRAIN loss magnitudes are not a quality comparison.",
             fontsize=13, color="#506071", ha="left")
    fig.text(.04, .095, "VAL: seven scheduled observations; "+selected_text+" (diamonds).",
             fontsize=13, color="#506071", ha="left")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in [*fig.texts, *(x for ax in axes for x in ax.texts),
                 *(ax.title for ax in axes), *(ax.xaxis.label for ax in axes), *(ax.yaxis.label for ax in axes)]:
        b = text.get_window_extent(renderer)
        if not (0 <= b.x0 <= b.x1 <= fig.bbox.width and 0 <= b.y0 <= b.y1 <= fig.bbox.height):
            raise ValueError("Figure text extends beyond canvas: "+text.get_text())
    metadata = {"schema": "hipmri_recorded_training_trace_v1",
                "source_directory": "artifacts/fixed_run", "models": {n: loaded[n][3] for n in STYLES},
                "smoothing": "none", "test_results_used": False,
                "interpretation": "Different inputs/auxiliary heads prevent direct TRAIN-loss magnitude comparison."}
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir/"training_progress.png"
    fig.savefig(output, dpi=300, facecolor="white",
                metadata={"Title": "Recorded TRAIN and scheduled VAL traces", "Description": json.dumps(metadata, sort_keys=True)})
    plt.close(fig)
    return {"output": str(output), "bytes": output.stat().st_size,
            "output_sha256": sha256(output.read_bytes()).hexdigest(), **metadata}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=PROJECT/"artifacts/fixed_run")
    parser.add_argument("--output-dir", type=Path, default=PROJECT/"figures")
    args = parser.parse_args(argv)
    print(json.dumps(plot(args.run_dir, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
