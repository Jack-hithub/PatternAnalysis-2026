# HipMRI segmentation: fixed higher-resolution 3D Improved U-Net study

COMP3710, Baoxu Wang (s4927782). Clean implementation and execution restart.

An imaging specialist needs draft pelvic-organ contours for correction. This study compares a residual, deeply supervised 3D U-Net with a standard 2D U-Net, asking whether its anatomical detail and boundary performance justify additional computation. Both models use the same patient assignments and input/evaluation grids.

## Fixed plan

- Patient-disjoint TRAIN/VAL/TEST: 26/6/6 patients and 143/38/30 scans; every visit remains with its patient.
- Input 96 x 192 x 192; 3D uses two 64 x 128 x 128 TRAIN patches per scan/epoch; 2D uses 32 TRAIN slices per scan/epoch. Half of sampled centres are uniform in space; half choose one foreground organ uniformly and then one of its voxels. Labels guide sampling only in TRAIN.
- Architecture, loss and the 30-epoch budget are fixed in config.json and modules.py. Scheduled VAL prostate Dice selects checkpoints; all final weights are fixed before this run's TEST evaluation.
- Final evaluation of both models on the same 64 x 128 x 128 grid reports every class Dice/IoU, patient-balanced values, boundary distances, failure examples and resource cost. The declared prostate Dice target is 0.70; it is not a grade guarantee.
- This cohort has been explored during earlier development. The fresh run keeps patient-level training separation, but does not claim that these TEST patients have never informed prior development.

## Current status

Configuration and patient assignments are fixed. The five standalone modules and image-only prediction workflow passed 44 synthetic CPU integration checks. Actual full-shape CUDA numerical, memory and complete-volume inference checks passed on a GPU Lab NVIDIA L40S. Package versions are recorded in [requirements-gpu.txt](requirements-gpu.txt) and the [GPU readiness receipt](evidence/gpu_readiness.json). These are software checks, not task-accuracy results. Formal training has not started, and no previous run's score is reported as this run's result.

The fixed question, data, resource estimates and limitations are summarized in the [feasibility plan](FEASIBILITY.md).

Eight additional synthetic saved-prediction analysis checks passed after the core implementation, bringing local integration coverage to 52 checks. They verify independent overlap recounting, patient weighting, paired boundary support and figure generation. Synthetic QA figures are never presented as patient examples.

## Reproduction

Use a CUDA-enabled PyTorch environment for the formal study. The observed Linux/Python 3.10 GPU package versions are pinned in requirements-gpu.txt; requirements.txt lists the cross-platform dependencies. A CPU can run the synthetic checks and image-only predictor. Set up an isolated environment and run:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python tools/check_runtime.py --config config.json --output artifacts/gpu_readiness.json
```

The course raw directory must contain semantic_MRs/*_LFOV.nii.gz and semantic_labels_only/*_SEMANTIC.nii.gz. Preserve the course patient identifiers; the inventory must exactly match patient_split.json. With a passing preflight and a new WORK_DIR, execute the fixed sequence:

```bash
python tools/run_study.py \
  --raw-root /path/to/HipMRI_Study_open \
  --work-dir artifacts/fixed_run \
  --gpu-readiness artifacts/gpu_readiness.json
```

The runner prepares TRAIN/VAL, trains each model from scratch, freezes both VAL-selected checkpoints, prepares TEST, performs paired reporting and independent saved-prediction analysis, then demonstrates raw-image-only native-coordinate output. Logs and study_state.json identify actual completed phases. Existing output directories are rejected. This is a fixed run, with no search over seeds, settings or additional candidates. Failed partial runs must be retained and investigated rather than described as completed.

For an independent image, predict.py needs an MRI and a selected checkpoint; it accepts no target mask:

```bash
python predict.py --image /path/to/scan.nii.gz \
  --checkpoint artifacts/fixed_run/improved3d/best.pt \
  --config config.json --output-dir artifacts/new_prediction --device cuda
```

It saves an integer native-grid NIfTI, a model-grid prediction, a prediction-only overlay and a geometry/hash receipt. Coordinate restoration cannot recover detail lost during resizing. Formal evaluation uses the declared common reporting grid, rather than claiming native-resolution accuracy.

## References

- Ronneberger et al., U-Net (2015): https://arxiv.org/abs/1505.04597.
- Isensee et al., residual units/deep supervision architectural reference: https://arxiv.org/abs/1802.10508. The implementation is an adaptation, not a claim of exact nnU-Net reproduction.
- HipMRI, Dowling and Greer, CSIRO v2: https://doi.org/10.25919/45t8-p065. Official labels 0 background, 1 body, 2 bones, 3 bladder, 4 rectum and 5 prostate are preserved in label_schema.json.
