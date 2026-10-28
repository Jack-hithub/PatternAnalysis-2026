"""Execute the fixed single-GPU study once in a new output directory.

Preflight is synthetic and separate. This runner never searches settings,
repartitions patients, resumes another run or starts an additional candidate.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train import source_identity
from utils import load_json, save_json, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True, help='Must not exist.')
    parser.add_argument('--gpu-readiness', type=Path, required=True)
    args = parser.parse_args()
    raw, work = args.raw_root.resolve(strict=True), args.work_dir.absolute()
    if not (raw/'semantic_MRs/B006_Week0_LFOV.nii.gz').is_file():
        raise FileNotFoundError('The declared image-only TRAIN demonstration source is missing.')
    config_path, split = ROOT / 'config.json', ROOT / 'patient_split.json'
    readiness = load_json(args.gpu_readiness)
    if (readiness.get('status') != 'passed'
            or readiness.get('source_sha256') != source_identity()
            or readiness.get('config_file_sha256') != sha256_file(config_path)):
        raise ValueError('A passing GPU preflight bound to these sources/config is required.')
    if work.exists() or work.is_symlink():
        raise FileExistsError('Choose a fresh work directory; never overwrite a run.')
    work.mkdir(parents=True)
    lock = (work / 'runner.lock').open('x')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
             'pid': os.getpid(), 'config_sha256': sha256_file(config_path),
             'source_sha256': source_identity(), 'gpu_readiness_sha256': sha256_file(args.gpu_readiness),
             'phases': [], 'phase': 'starting',
             'policy': 'one fresh 30-epoch pair; fixed scheduled VAL selection; both frozen before this run TEST; no further candidates'}
    manifest, frozen = work / 'cache/manifest.json', work / 'selection_freeze.json'
    common = ['--manifest', str(manifest), '--config', str(config_path)]
    phases = [
        ('prepare_trainval', [str(ROOT/'dataset.py'), '--raw-root', str(raw), '--split', str(split),
                             '--config', str(config_path), '--output-dir', str(work/'cache'),
                             '--stage', 'trainval', '--workers', '4']),
    ]
    for name in ('unet2d', 'improved3d'):
        phases.append(('train_'+name, [str(ROOT/'train.py'), '--mode', 'train', '--model', name,
                                       *common, '--output-dir', str(work/name), '--device', 'cuda']))
    phases += [
        ('freeze_both_models', [str(ROOT/'train.py'), '--mode', 'freeze', *common,
                               '--run-unet2d', str(work/'unet2d'), '--run-improved3d', str(work/'improved3d'),
                               '--output-dir', str(frozen)]),
        ('prepare_test', [str(ROOT/'dataset.py'), '--raw-root', str(raw), '--split', str(split),
                          '--config', str(config_path), '--output-dir', str(work/'cache'), '--stage', 'test',
                          '--workers', '4', '--selection-freeze', str(frozen)]),
    ]
    for partition in ('validation', 'test'):
        for name in ('unet2d', 'improved3d'):
            phases.append(('report_'+partition+'_'+name,
                [str(ROOT/'train.py'), '--mode', 'evaluate', '--model', name, *common,
                 '--split', partition, '--checkpoint', str(work/name/'best.pt'),
                 '--selection-freeze', str(frozen), '--output-dir', str(work/'evaluation'/partition/name),
                 '--device', 'cuda']))
        phases.append(('audit_'+partition, [str(ROOT/'tools/analyse_results.py'),
                      '--manifest', str(manifest),
                      '--unet2d-evaluation', str(work/'evaluation'/partition/'unet2d'),
                      '--improved3d-evaluation', str(work/'evaluation'/partition/'improved3d'),
                      '--output-dir', str(work/'analysis'/partition)]))
    phases.append(('image_only_native_export', [str(ROOT/'predict.py'), '--image',
                   str(raw/'semantic_MRs/B006_Week0_LFOV.nii.gz'), '--checkpoint', str(work/'improved3d/best.pt'),
                   '--config', str(config_path), '--output-dir', str(work/'native_demo'), '--device', 'cuda']))
    started = time.perf_counter()
    try:
        for phase, command in phases:
            state['phase'] = phase
            save_json(work/'study_state.json', state)
            stamp = datetime.now(timezone.utc).isoformat()
            print(f'{stamp} START {phase}', flush=True)
            phase_start = time.perf_counter()
            subprocess.run([sys.executable, '-u', *command], cwd=ROOT, check=True)
            state['phases'].append({'phase': phase, 'status': 'completed', 'started_at': stamp,
                                    'seconds': time.perf_counter()-phase_start})
            print(f'{datetime.now(timezone.utc).isoformat()} FINISH {phase}', flush=True)
        state.update(status='completed', phase='complete', finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}',
                     failed_at=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        state['wall_seconds'] = time.perf_counter()-started
        save_json(work/'study_state.json', state)
        lock.close()


if __name__ == '__main__':
    main()
