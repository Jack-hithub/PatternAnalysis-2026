"""Measure the fixed models on the actual GPU before formal training.

Synthetic labels test numerical/dataflow correctness only. No accuracy threshold,
model choice or scientific setting is inferred from this check.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from modules import segmentation_loss
from train import create_model, infer_volume, runtime_identity, source_identity, sync
from utils import load_json, save_json, seed_everything, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Runtime check records are not overwritten.')
    config = load_json(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError('The formal study requires one CUDA GPU.')
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device('cuda')
    seed_everything(config['seed'])
    rows = []
    for name in ('unet2d', 'improved3d'):
        model = create_model(name, config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'],
                                      weight_decay=config['weight_decay'])
        batch = config['batch_size_2d'] if name == 'unet2d' else config['batch_size_3d']
        shape = config['target_shape_dhw'][1:] if name == 'unet2d' else config['patch_shape_dhw']
        image = torch.randn(batch, 1, *shape, device=device)
        target = torch.randint(config['num_classes'], (batch, *shape), device=device)
        scaler = torch.amp.GradScaler('cuda', enabled=config['amp'])
        steps = []
        for index in range(12):
            optimizer.zero_grad(set_to_none=True)
            sync(device)
            torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            with torch.autocast('cuda', enabled=config['amp']):
                output = model(image)
                loss = segmentation_loss(output, target, class_weights=torch.ones(6, device=device))
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('Nonfinite synthetic loss.')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad'])
            before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            overflow = scaler.get_scale() < before
            if not bool(torch.isfinite(norm)) and not overflow:
                raise RuntimeError('Nonfinite gradient was not rejected by AMP.')
            if not overflow:
                if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                    raise RuntimeError('A model parameter did not receive a finite gradient.')
            sync(device)
            steps.append({'index': index, 'seconds': time.perf_counter()-start,
                          'loss': float(loss.detach()), 'amp_skipped': overflow,
                          'peak_allocated_cuda_bytes': torch.cuda.max_memory_allocated(device)})
            if index >= 3 and sum(not row['amp_skipped'] for row in steps) >= 2:
                break
        if all(row['amp_skipped'] for row in steps):
            raise RuntimeError('All synthetic updates skipped: runtime not ready.')
        optimizer.zero_grad(set_to_none=True)
        del output, loss, image, target, optimizer
        whole = np.random.default_rng(config['seed']).normal(size=config['target_shape_dhw']).astype(np.float32)
        sync(device)
        start = time.perf_counter()
        prediction = infer_volume(model, whole, name, config, device,
                                  support_dhw=np.ones(whole.shape, dtype=bool))
        sync(device)
        if prediction.shape != tuple(config['target_shape_dhw']) or not np.isin(prediction, range(6)).all():
            raise RuntimeError('Full-image inference geometry/class coverage failed.')
        rows.append({'model': name, 'parameters': sum(p.numel() for p in model.parameters()),
                     'steps': steps, 'mean_last_two_step_seconds': float(np.mean([r['seconds'] for r in steps[-2:]])),
                     'full_image_inference_seconds': time.perf_counter()-start,
                     'training_shape': [batch, 1, *shape]})
        del model
        torch.cuda.empty_cache()
    result = {'status': 'passed', 'scope': 'synthetic numerical/geometry readiness, not task accuracy',
              'runtime': runtime_identity(device),
              'packages': {name: importlib.metadata.version(name) for name in ('torch','numpy','nibabel','scipy','matplotlib')},
              'config_file_sha256': sha256_file(args.config), 'source_sha256': source_identity(),
              'models': rows}
    save_json(args.output, result)
    print('Synthetic GPU readiness passed; formal task scores remain unmeasured.', flush=True)


if __name__ == '__main__':
    main()
