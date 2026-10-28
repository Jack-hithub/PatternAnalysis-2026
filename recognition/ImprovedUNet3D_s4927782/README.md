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

Configuration and patient assignments are fixed. Standalone modules.py, dataset.py, train.py, predict.py and supporting tests are undergoing integration checks. No result from a previous run is used as this run's result. Formal training has not started. Runtime versions and real execution receipts will be recorded before results are reported.

## References

- Ronneberger et al., U-Net (2015): https://arxiv.org/abs/1505.04597.
- Isensee et al., residual units/deep supervision architectural reference: https://arxiv.org/abs/1802.10508. The implementation is an adaptation, not a claim of exact nnU-Net reproduction.
- HipMRI, Dowling and Greer, CSIRO v2: https://doi.org/10.25919/45t8-p065. Official labels 0 background, 1 body, 2 bones, 3 bladder, 4 rectum and 5 prostate are preserved in label_schema.json.
