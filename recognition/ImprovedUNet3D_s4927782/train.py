"""Train, freeze and evaluate the standalone HipMRI segmentation study.

Models/loss live in modules.py; TRAIN-only sampling and image geometry live in
dataset.py. Validation selects weights on the 96-grid. Final paired reporting
uses the fixed 64-grid and preserves all patients/visits. Previously explored
TEST patients do not become a new blind test by rerunning this program.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import itertools
from importlib.metadata import PackageNotFoundError, version
import math
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import dataset as data
from modules import UNet2D, ImprovedUNet3D, segmentation_loss
from utils import (aggregate_patients, class_metrics, load_json, save_json,
                   save_training_plot, seed_everything, sha256_file)

MODELS = ("unet2d", "improved3d")
SOURCE_FILES = ("modules.py", "dataset.py", "utils.py", "train.py", "predict.py")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def source_identity():
    root = Path(__file__).resolve().parent
    return {name: sha256_file(root / name) for name in SOURCE_FILES}


def runtime_identity(device):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = os.environ.get("DEMO3_SOURCE_COMMIT")
    packages = {}
    for name in ("scipy", "nibabel", "matplotlib"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": np.__version__, "cuda_runtime": torch.version.cuda,
            **packages, "cpu_threads": torch.get_num_threads(),
            "cpu_interop_threads": torch.get_num_interop_threads(),
            "platform": platform.platform(), "machine": platform.machine(),
            "device": str(device), "gpu": torch.cuda.get_device_name(device)
            if device.type == "cuda" else None, "source_commit": commit,
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory
            if device.type == "cuda" else None,
            "cudnn_version": torch.backends.cudnn.version() if device.type == "cuda" else None,
            "deterministic_policy": "requested with warn_only; bitwise retraining not guaranteed"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def choose_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable.")
    return torch.device(name)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def primary_logits(output):
    if isinstance(output, dict):
        return output["logits"]
    if isinstance(output, (list, tuple)):
        return output[0]
    return output


def window_starts(size, patch, overlap):
    require(0 <= overlap < 1 and 1 <= patch <= size, "Invalid window dimensions/overlap.")
    stride = max(1, round(patch * (1 - overlap)))
    positions = list(range(0, size - patch + 1, stride))
    if positions[-1] != size - patch:
        positions.append(size - patch)
    return positions


def infer_volume(model, image_dhw, model_name, config, device):
    """Full-image prediction; accepts no GT, crop centre or label-derived ROI.

    3D windows accumulate arithmetic-mean softmax probabilities on CPU; 2D
    processes every slice. A fixed image-only zero-support policy is applied
    after argmax. Auxiliary heads train the model but do not vote at inference.
    """
    require(model_name in MODELS, "Unknown model.")
    image = np.asarray(image_dhw, dtype=np.float32)
    require(image.ndim == 3 and all(n > 0 for n in image.shape) and np.isfinite(image).all(),
            "Expected finite image(D,H,W).")
    device = torch.device(device)
    previous = model.training
    model.eval()
    use_amp = bool(config.get("amp", False) and device.type == "cuda")
    tensor = torch.from_numpy(np.ascontiguousarray(image))
    try:
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=use_amp):
            if model_name == "unet2d":
                predictions = []
                for block in tensor.unsqueeze(1).split(config["eval_slice_batch_size"]):
                    logits = primary_logits(model(block.to(device)))
                    require(tuple(logits.shape) == (len(block), config["num_classes"], *image.shape[1:])
                            and bool(torch.isfinite(logits).all()), "Invalid 2D inference output.")
                    predictions.append(logits.argmax(1).cpu())
                prediction = torch.cat(predictions).numpy().astype(np.uint8)
            else:
                patch = tuple(config["patch_shape_dhw"])
                starts = [window_starts(n, k, config["inference_overlap"])
                          for n, k in zip(image.shape, patch)]
                total = torch.zeros((config["num_classes"], *image.shape), dtype=torch.float32)
                coverage = torch.zeros(image.shape, dtype=torch.float32)
                for start in itertools.product(*starts):
                    region = tuple(slice(s, s+k) for s, k in zip(start, patch))
                    block = tensor[region].unsqueeze(0).unsqueeze(0).to(device)
                    logits = primary_logits(model(block))
                    require(tuple(logits.shape) == (1, config["num_classes"], *patch)
                            and bool(torch.isfinite(logits).all()), "Invalid 3D inference output.")
                    total[(slice(None), *region)] += torch.softmax(logits.float(), dim=1)[0].cpu()
                    coverage[region] += 1
                require(bool((coverage > 0).all()), "Sliding windows left uncovered voxels.")
                prediction = (total / coverage.unsqueeze(0)).argmax(0).numpy().astype(np.uint8)
        if config.get("mask_outside_nonzero_support", False):
            prediction[image == 0] = 0
        return prediction
    finally:
        model.train(previous)


def select_best(history):
    """Strict improvement only: the earliest scheduled maximum wins ties."""
    best_epoch, best_score = None, None
    for row in history:
        score = row.get("validation_prostate_dice")
        if score is None:
            continue
        require(isinstance(score, (int, float)) and math.isfinite(score), "Invalid VAL score.")
        if best_score is None or score > best_score:
            best_epoch, best_score = int(row["epoch"]), float(score)
    require(best_epoch is not None, "No completed scheduled validation.")
    return best_epoch, best_score


def create_model(model_name, config):
    require(model_name in MODELS, "Unknown model.")
    factory = UNet2D if model_name == "unet2d" else ImprovedUNet3D
    return factory(config["num_classes"], config["base_channels"])


def read_config(path):
    config = load_json(path)
    require(config["num_classes"] == 6 and config["epochs"] == 30,
            "This fixed study uses six classes and 30 epochs.")
    require(config["validation_epochs"] == [1, 5, 10, 15, 20, 25, 30], "Fixed validation schedule changed.")
    return config


def fresh_dir(path):
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), "Use a fresh output directory.")
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_checkpoint(path, payload):
    """Atomic single-file replacement; best/latest are not a paired transaction."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def checkpoint_identity(manifest_path, config):
    path = Path(manifest_path)
    receipt = path.parent / "trainval_receipt.json"
    require(receipt.is_file(), "Completed TRAIN/VAL preparation receipt missing.")
    prepared = load_json(receipt)
    require(prepared.get("status") == "completed", "TRAIN/VAL cache preparation incomplete.")
    require(prepared.get("manifest_sha256") == sha256_file(path), "Cache manifest changed.")
    require(prepared.get("config_sha256") == data.config_digest(config), "Preparation config changed.")
    return {"manifest_sha256": sha256_file(path), "config_sha256": data.config_digest(config),
            "trainval_receipt_sha256": sha256_file(receipt), "source_sha256": source_identity()}


def load_selected_checkpoint(path, config, manifest_path=None, check_sources=True):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    require(checkpoint.get("schema_version") == 1 and checkpoint.get("model") in MODELS,
            "Unknown checkpoint schema/model.")
    require(checkpoint.get("config") == config, "Checkpoint scientific configuration differs.")
    identity = checkpoint["identity"]
    require(identity["config_sha256"] == data.config_digest(config), "Checkpoint config digest differs.")
    if check_sources:
        require(identity["source_sha256"] == source_identity(), "Checkpoint sources differ.")
    if manifest_path is not None:
        require(identity == checkpoint_identity(manifest_path, config), "Checkpoint data/source identity differs.")
    epoch, score = select_best(checkpoint["history"])
    require(checkpoint["epoch"] == checkpoint["best_epoch"] == epoch
            and checkpoint["best_score"] == score, "Use an actual selected-best checkpoint.")
    return checkpoint


def validate_freeze(path, manifest_path, config, model_name=None, checkpoint_path=None):
    receipt = load_json(path)
    require(receipt.get("status") == "frozen", "Selection is not frozen.")
    require(receipt.get("manifest_sha256") == sha256_file(manifest_path)
            and receipt.get("config_sha256") == data.config_digest(config), "Freeze identity differs.")
    require(set(receipt.get("models", {})) == set(MODELS), "Both real models must finish before TEST.")
    for name, item in receipt["models"].items():
        require(item.get("completed_epochs") == config["epochs"], "Incomplete formal training.")
        candidate = Path(item["checkpoint_path"])
        if not candidate.is_absolute():
            candidate = Path(path).resolve().parent / candidate
        require(candidate.is_file() and sha256_file(candidate) == item["checkpoint_sha256"],
                "Frozen checkpoint bytes changed.")
    if model_name is not None:
        require(receipt["models"][model_name]["checkpoint_sha256"] == sha256_file(checkpoint_path),
                "Evaluation checkpoint differs from frozen choice.")
    return receipt


def evaluate(model, manifest_path, split, model_name, config, device, *, grid="native96",
             output_dir=None, include_hd95=False):
    """Run image-only predictions, then compare to labels; labels never enter infer_volume."""
    device = torch.device(device)
    require(split in ("validation", "test"), "Only VAL/TEST evaluation is supported.")
    require(grid in ("native96", "report64"), "Unknown evaluation grid.")
    records = data.records_for_split(manifest_path, split)
    require(bool(records), "Empty evaluation split.")
    if output_dir is not None:
        output_dir = Path(output_dir)
        (output_dir / "predictions").mkdir(parents=True, exist_ok=False)
    cases = []
    for index, record in enumerate(records):
        case = data.load_case(manifest_path, record)
        if index == 0:
            infer_volume(model, case["image"], model_name, config, device)
        sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        prediction = infer_volume(model, case["image"], model_name, config, device)
        sync(device)
        milliseconds = (time.perf_counter() - start) * 1000
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        if grid == "report64":
            scored = data.resample_label(prediction, case["affine_xyz"], case["affine_eval_xyz"],
                                         tuple(config["evaluation_shape_dhw"]))
            target, spacing = case["mask_eval"], case["eval_spacing_dhw"]
        else:
            scored, target, spacing = prediction, case["mask"], case["spacing_dhw"]
        metrics = class_metrics(scored, target, config["num_classes"], spacing=spacing,
                                include_hd95=include_hd95)
        row = {"patient": record["patient"], "visit": record["visit"], **metrics,
               "shape_dhw": list(scored.shape), "spacing_dhw": np.asarray(spacing).tolist(),
               "inference_ms": milliseconds, "peak_allocated_cuda_bytes": peak}
        if output_dir is not None:
            filename = f"{record['patient']}_{record['visit']}.npz"
            destination = output_dir / "predictions" / filename
            np.savez_compressed(destination, prediction96=prediction,
                prediction=scored.astype(np.uint8), spacing_dhw=spacing,
                affine_xyz=case["affine_eval_xyz"] if grid == "report64" else case["affine_xyz"])
            row["prediction_file"] = str(destination.relative_to(output_dir))
        cases.append(row)
    aggregate = aggregate_patients(cases, config["num_classes"])
    result = {"status": "completed", "model": model_name, "split": split, "grid": grid,
              "configuration": config, "cases": cases, "aggregate": aggregate,
              "patient_balanced_prostate_dice": aggregate["per_class"][5]["dice"],
              "mean_inference_ms": float(np.mean([x["inference_ms"] for x in cases])),
              "peak_allocated_cuda_bytes": max((x["peak_allocated_cuda_bytes"] or 0 for x in cases), default=0) or None,
              "test_interpretation": config["evaluation_context"] if split == "test" else "Validation used for checkpoint selection.",
              "latency_protocol": "One whole-case warmup; synchronized transfer/network/window-probability-average/argmax/output; excludes disk/preprocessing/metrics.",
              "manifest_sha256": sha256_file(manifest_path)}
    if output_dir is not None:
        save_json(output_dir / "evaluation.json", result)
    return result


def train_model(model_name, manifest_path, config, output_dir, device):
    wall_start = time.perf_counter()
    device = torch.device(device)
    identity = checkpoint_identity(manifest_path, config)
    seed_everything(config["seed"])
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    output = fresh_dir(output_dir)
    dataset = data.SampledTrainingDataset(manifest_path, model_name, config)
    model = create_model(model_name, config).to(device)
    weights, counts = data.training_class_weights(manifest_path)
    weights = weights.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    amp = bool(config["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    loader = DataLoader(dataset, batch_size=config["batch_size_2d"] if model_name == "unet2d" else config["batch_size_3d"],
                        shuffle=False, num_workers=config["workers"], pin_memory=device.type == "cuda")
    runtime = runtime_identity(device)
    history, best_score, best_epoch = [], None, None
    run = {"status": "running", "model": model_name, "started_at": utc_now(), "config": config,
           "identity": identity, "runtime": runtime, "parameters": sum(x.numel() for x in model.parameters()),
           "training_class_counts": counts, "class_weights": weights.detach().cpu().tolist(),
           "initial_state_sha256": state_digest(model.state_dict())}
    save_json(output / "run.json", run)
    try:
        for epoch in range(1, config["epochs"] + 1):
            dataset.set_epoch(epoch)
            model.train()
            sync(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            loss_sum, examples, batches, skipped = 0., 0, 0, 0
            sampling = {"foreground_requested": 0, "foreground_missing_fallback": 0,
                        "selected_class_counts": [0] * config["num_classes"]}
            for batch in loader:
                images, targets = batch["image"].to(device), batch["mask"].to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp):
                    outputs = model(images)
                    loss = segmentation_loss(outputs, targets, class_weights=weights)
                require(bool(torch.isfinite(loss)), "Nonfinite loss; run cannot advance.")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad"])
                if not amp:
                    require(bool(torch.isfinite(norm)), "Nonfinite FP32 gradient.")
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                overflow = scaler.get_scale() < old_scale
                require(bool(torch.isfinite(norm)) or overflow, "Nonfinite gradient was not skipped by AMP.")
                skipped += int(overflow)
                count = len(images)
                examples += count
                batches += 1
                loss_sum += float(loss.detach()) * count
                sampling["foreground_requested"] += int(batch["foreground_requested"].sum())
                sampling["foreground_missing_fallback"] += int(batch["foreground_missing_fallback"].sum())
                for cls in batch["selected_class"].tolist():
                    if cls >= 0:
                        sampling["selected_class_counts"][cls] += 1
            sync(device)
            seconds = time.perf_counter() - start
            peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            require(examples == len(dataset) and batches == len(loader), "Incomplete epoch sampling.")
            optimizer.zero_grad(set_to_none=True)
            del outputs, loss, images, targets
            rng = torch.random.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
            validation_start = time.perf_counter()
            validation = evaluate(model, manifest_path, "validation", model_name, config, device) if epoch in config["validation_epochs"] else None
            validation_seconds = time.perf_counter() - validation_start if validation else 0.
            score = validation["patient_balanced_prostate_dice"] if validation else None
            require(validation is None or score is not None, "Undefined scheduled VAL prostate score.")
            if score is not None and (best_score is None or score > best_score):
                best_score, best_epoch = score, epoch
            row = {"epoch": epoch, "train_loss": loss_sum/examples, "training_examples": examples,
                   "training_batches": batches, "optimizer_steps": batches-skipped,
                   "amp_overflow_skipped_optimizer_steps": skipped, "train_seconds": seconds,
                   "peak_allocated_cuda_bytes": peak, "sampling": sampling,
                   "validation_seconds": validation_seconds,
                   "validation_prostate_dice": score, "validation": validation}
            history.append(row)
            payload = {"schema_version": 1, "model": model_name, "config": deepcopy(config),
                       "identity": identity, "runtime": runtime, "state_dict": model.state_dict(),
                       "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                       "epoch": epoch, "best_epoch": best_epoch, "best_score": best_score,
                       "history": history}
            save_checkpoint(output / "latest.pt", payload)
            if best_epoch == epoch:
                save_checkpoint(output / "best.pt", payload)
            if epoch == config["epochs"]:
                save_checkpoint(output / "final.pt", payload)
            save_json(output / "history.json", history)
            save_training_plot(history, output / "curves.png")
            torch.random.set_rng_state(rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            run.update(completed_epochs=epoch, best_epoch=best_epoch,
                       best_validation_prostate_dice=best_score,
                       actual_optimizer_steps=sum(r["optimizer_steps"] for r in history),
                       actual_training_examples=sum(r["training_examples"] for r in history))
            save_json(output / "run.json", run)
            print(f"{model_name} epoch{epoch}/{config['epochs']} loss{row['train_loss']:.6f} selected{best_epoch}", flush=True)
        selected = load_selected_checkpoint(output / "best.pt", config, manifest_path)
        model.load_state_dict(selected["state_dict"], strict=True)
        reloaded = evaluate(model, manifest_path, "validation", model_name, config, device)
        save_json(output / "best_validation_reloaded.json", reloaded)
        run.update(status="completed", finished_at=utc_now(),
                   training_loop_seconds=sum(r["train_seconds"] for r in history),
                   scheduled_validation_seconds=sum(r["validation_seconds"] for r in history),
                   total_wall_seconds=time.perf_counter()-wall_start,
                   best_checkpoint_sha256=sha256_file(output / "best.pt"),
                   final_checkpoint_sha256=sha256_file(output / "final.pt"),
                   validation_reload_delta=reloaded["patient_balanced_prostate_dice"]-best_score)
        save_json(output / "run.json", run)
    except BaseException as exc:
        run.update(status="failed", failed_at=utc_now(), error=f"{type(exc).__name__}: {exc}")
        save_json(output / "run.json", run)
        raise
    return run


def state_digest(state):
    import hashlib
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        a = value.detach().contiguous().cpu()
        h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(str(tuple(a.shape)).encode())
        h.update(a.numpy().tobytes())
    return h.hexdigest()


def freeze_selection(run_dirs, manifest_path, config, destination):
    destination = Path(destination)
    require(not destination.exists(), "Selection freeze already exists.")
    choices, identities = {}, []
    for name in MODELS:
        run_dir = Path(run_dirs[name]).resolve()
        run, history = load_json(run_dir / "run.json"), load_json(run_dir / "history.json")
        require(run.get("status") == "completed" and run.get("completed_epochs") == config["epochs"],
                "Both models must complete their fixed budget.")
        require([r["epoch"] for r in history] == list(range(1, config["epochs"]+1)), "Missing/repeated training epochs.")
        require([r["epoch"] for r in history if r.get("validation_prostate_dice") is not None]
                == config["validation_epochs"], "Scheduled validation history differs.")
        epoch, score = select_best(history)
        path = run_dir / "best.pt"
        checkpoint = load_selected_checkpoint(path, config, manifest_path)
        require(checkpoint["model"] == name and checkpoint["epoch"] == epoch
                and checkpoint["best_score"] == score, "Selected weight differs from full VAL history.")
        require(run["best_checkpoint_sha256"] == sha256_file(path), "Saved best weight digest differs.")
        identities.append(checkpoint["identity"])
        try:
            portable_path = os.path.relpath(path, destination.resolve().parent)
        except ValueError:
            portable_path = str(path)
        choices[name] = {"completed_epochs": config["epochs"], "selected_epoch": epoch,
                         "validation_prostate_dice": score, "checkpoint_path": portable_path,
                         "checkpoint_sha256": sha256_file(path),
                         "history_sha256": sha256_file(run_dir / "history.json")}
    require(identities[0] == identities[1], "Paired model data/config/source identities differ.")
    receipt = {"status": "frozen", "created_at": utc_now(),
               "manifest_sha256": sha256_file(manifest_path), "config_sha256": data.config_digest(config),
               "source_sha256": source_identity(), "models": choices,
               "evaluation_context": config["evaluation_context"]}
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_json(destination, receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "freeze", "evaluate"), default="train")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Fresh directory for train/evaluate; fresh JSON receipt FILE for freeze.")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--selection-freeze", type=Path)
    parser.add_argument("--run-unet2d", type=Path)
    parser.add_argument("--run-improved3d", type=Path)
    args = parser.parse_args(argv)
    config = read_config(args.config)
    if args.mode == "train":
        require(args.model is not None, "Training requires --model.")
        train_model(args.model, args.manifest, config, args.output_dir, choose_device(args.device))
    elif args.mode == "freeze":
        require(args.run_unet2d is not None and args.run_improved3d is not None, "Both formal run directories are required.")
        freeze_selection({"unet2d": args.run_unet2d, "improved3d": args.run_improved3d},
                         args.manifest, config, args.output_dir)
    else:
        require(args.model is not None and args.checkpoint is not None, "Evaluation requires model/checkpoint.")
        require(args.selection_freeze is not None, "Final reporting requires frozen paired weights.")
        validate_freeze(args.selection_freeze, args.manifest, config, args.model, args.checkpoint)
        checkpoint = load_selected_checkpoint(args.checkpoint, config, args.manifest)
        require(checkpoint["model"] == args.model, "Checkpoint model differs.")
        device = choose_device(args.device)
        seed_everything(config["seed"])
        model = create_model(args.model, config).to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        output = fresh_dir(args.output_dir)
        result = evaluate(model, args.manifest, args.split, args.model, config, device,
                          grid="report64", output_dir=output, include_hd95=True)
        result.update(checkpoint_sha256=sha256_file(args.checkpoint),
                      selection_freeze_sha256=sha256_file(args.selection_freeze), runtime=runtime_identity(device))
        save_json(output / "evaluation.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
