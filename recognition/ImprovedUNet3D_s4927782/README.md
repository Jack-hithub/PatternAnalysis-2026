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

Configuration and patient assignments are fixed. The five standalone modules and image-only prediction workflow passed 44 synthetic CPU integration checks. Actual full-shape CUDA numerical, memory and complete-volume inference checks passed on a GPU Lab NVIDIA L40S. Package versions are recorded in [requirements-gpu.txt](requirements-gpu.txt) and the [GPU readiness receipt](evidence/gpu_readiness.json). These are software checks, not task-accuracy results. The fixed execution started on 28 October 2026: fresh TRAIN/VAL preparation and all 30 epochs of 2D training have completed; 3D training is underway. Final scores will be added only after the fixed pair finishes; no previous run's score is reported as this run's result.

The fixed question, data, resource estimates and limitations are summarized in the [feasibility plan](FEASIBILITY.md).

Eight additional synthetic saved-prediction analysis checks passed after the core implementation, bringing local integration coverage to 52 checks. They verify independent overlap recounting, patient weighting, paired boundary support and figure generation. Synthetic QA figures are never presented as patient examples.

Eleven further checks cover interface/continuity analysis and verified receipt-only result printing, bringing the current local total to **63 passing synthetic checks** ([record](evidence/report_helpers_checks.json)). These complement the data/model checks and do not establish clinical accuracy.

## Working principles and data pipeline

The baseline U-Net encodes each 2D slice through three pooling stages and decodes it using transposed convolutions and matching encoder skip connections. The 3D model processes volumes with residual convolution blocks, GroupNorm and LeakyReLU, using channel widths 16, 32, 64 and 128. Its decoder resizes features with trilinear interpolation, applies a convolution, concatenates the matching skip and applies a residual block. Two auxiliary six-class heads supervise half- and quarter-resolution decoder features. All heads use weighted categorical cross-entropy plus soft Dice; head weights 1, 0.5 and 0.25 are normalized by their sum. Only the primary head participates in prediction. This compact adaptation differs from the cited architecture in its normalization, pooling and supervision details.

![Residual 3D U-Net, training supervision and image-only prediction workflow](figures/architecture.png)

The diagram shows a training patch. At inference the eight windows cover the complete 96 x 192 x 192 image. The 64 x 128 x 128 reporting grid spans that complete physical field of view; it is a different physical grid from a training patch despite having the same array dimensions.

The six labels are mutually exclusive. Stored masks contain integer IDs; the Dice loss creates a one-hot representation internally, while cross-entropy uses categorical targets. Softmax probabilities produce one final label per voxel. Complete 3D prediction averages the probabilities from eight overlapping windows, including every voxel, before argmax; it does not average class IDs. Complete 2D prediction assembles every slice. An image-only zero-support policy is fixed for both models. No target-defined ROI or organ location enters prediction.

Each original image is reoriented to canonical RAS, standardized using its own nonzero intensities, clipped to [-5, 5] and linearly resampled to the input grid. Labels use nearest neighbours. Voxel-centre-aware affines retain the physical field of view; native-coordinate output uses the original image geometry. Paired H/W flips are restricted to training. The class weights use inverse-square-root frequencies from complete TRAIN grids, followed by fixed clipping and normalization; no sampled, VAL or TEST counts determine them.

The original course copy was transferred without substituting a different archive: all **422 files** matched the source byte length and SHA256 ([migration receipt](evidence/raw_source_migration.json)). The fresh TRAIN/VAL audit verified **143/38 scans from 26/6 disjoint patients**, all six training classes, 362 source-file identities and complete preparation checks. Before fixed selection, TEST has only filename/split entries and no decoded arrays in the cache ([audit](evidence/fresh_trainval_audit.json)). Prostate occupies approximately **0.108% of TRAIN grid voxels**, which motivates class weighting and foreground sampling. Sampling is applied across all five foreground classes rather than to a particular held-out patient.

## Evaluation definitions

For each class, Dice is `2TP / (2TP + FP + FN)` and IoU is `TP / (TP + FP + FN)`, without smoothing. A class absent from both masks is undefined; one-sided absence scores zero and retains the missed or extra voxel counts. The primary score averages visits within each patient and then weights patients equally. It does not pool all voxels or give patients with more visits greater weight. Defined-case and patient support accompany summaries.

Both frozen models are scored on the same reporting grid. Boundary analysis uses six-neighbour surfaces and the 95th percentile of the combined bidirectional nearest-surface distances, in physical millimetres. Empty surfaces have undefined distance rather than zero; comparisons use the same eligible visits in both models and report their support. Inferior/middle/superior regions are descriptive thirds of the GT prostate's D-axis extent, not expert anatomical apex/base annotations. Post-hoc image panels and region definitions are analysis tools and are never inference inputs.

Additional saved-array analysis counts prostate false positives inside GT bladder and rectum, records the predicted class of prostate misses, and describes six-neighbour components and missed GT-positive slices. The same five worst-patient case choices are shown in coronal/sagittal views with complete D-axis area curves. These are descriptive error analyses with explicit denominators, rather than a new model selection or a validated clinical routing rule. An empty prediction retains its zero overlap; it cannot earn a favourable continuity conclusion merely by being smooth.

This is a comparison of two complete configurations. Equal epochs and attempted update counts do not make their computation equal: dimensionality, parameter count, sampled voxels, supervision and inference differ. Actual optimizer updates and AMP overflow skips are recorded separately. A single-seed comparison cannot isolate the causal effect of 3D context or establish clinical safety. Final recommendations must consider all patients and boundary errors alongside measured runtime and memory.

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

For an independent image, the prediction entry below delegates to predict.py and prints the result paths and timing. It needs an MRI and a selected checkpoint; it accepts no target mask:

```bash
python tools/predict_and_report.py --image /path/to/scan.nii.gz \
  --checkpoint artifacts/fixed_run/improved3d/best.pt \
  --config config.json --output-dir artifacts/new_prediction --device cuda
```

It saves an integer native-grid NIfTI, a model-grid prediction, a prediction-only overlay and a geometry/hash receipt. Coordinate restoration cannot recover detail lost during resizing. Formal evaluation uses the declared common reporting grid, rather than claiming native-resolution accuracy.

To inspect a completed prediction without rerunning inference, use `python tools/predict_and_report.py --receipt artifacts/new_prediction/prediction_receipt.json`. The mandatory predict.py module also retains its direct file-producing CLI.

## Feasibility review

Baoxu Wang (s4927782), COMP3710 — 28 October 2026. Written before the formal execution described here.

**Problem and user.** A pelvic MRI reviewer needs initial contours for body, bones, bladder, rectum and prostate. The model produces draft categorical segmentations for manual correction. This is a coursework prototype; clinical suitability is not established.

**Question and acceptance criteria.** Compare a compact residual 3D U-Net with deep supervision against a standard 2D U-Net. Five testable criteria are: (1) 211 aligned pairs with patient-disjoint splits; (2) prostate Dice at least 0.70, averaging visits within patient then patients equally; (3) raw-image-only prediction with valid class IDs and restored original coordinates; (4) one GPU, 30 epochs per model and no additional candidates, recording time and memory; (5) paired class metrics and five distinct-patient boundary error examples. A favourable average must not hide a failed patient. No minimum VAL improvement over the baseline is imposed.

**Data and separation.** The course HipMRI copy has 211 paired scans from 38 patients. Fixed TRAIN/VAL/TEST assignments contain 26/6/6 patients and 143/38/30 scans. Keep every patient's visits together. The official CSIRO label key identifies prostate as class 5. Use the course files without substituting a different archive. The cohort was explored during earlier development, so this execution is patient-disjoint but does not establish a new blind test.

**Fixed method.** Canonical RAS orientation, per-image nonzero z-score and clipping, then a 96 x 192 x 192 grid. Interpolate images linearly and labels with nearest neighbours. Train two 64 x 128 x 128 patches per scan for 3D, or 32 slices per scan for 2D. Half of TRAIN sample centres are spatially uniform; half select a foreground class and voxel. Neither inference nor validation uses target-derived cropping. The 3D model has 1,627,634 parameters; 2D has 481,830. Both use weighted categorical cross-entropy plus Dice, 30 epochs, seed 3711 and the fixed optimizer in config.json. Scheduled VAL prostate Dice selects each checkpoint, taking the earliest tie. Freeze both choices before final TEST evaluation. No further candidate is planned.

**Resources and checks.** A single GPU Lab NVIDIA L40S is available. Synthetic full-shape CUDA checks passed before real-data training: about 2.23 GiB peak allocated memory for a 3D training patch and 0.34 GiB for the 2D batch. These are readiness measurements, not formal task costs. CPU integration checks cover geometry, patient separation, metrics, freezing, full-image inference and raw-image-only output. Allow roughly 25 minutes for preparation and 100 minutes for the formal pair as planning estimates; record actual time and memory. The pipelines share an attempted-update budget; actual AMP skips are recorded. Computation is not equal.

**Evaluation, risks and fallback.** Score both models on the same 64 x 128 x 128 grid. Save all cases, independently recount overlap, and compare boundary distances on common eligible support. Show MRI, target, both predictions and FP/FN; distinguish observed errors from unverified causes. Small patient counts, previously explored TEST data, downsampling, one seed and pipeline differences limit generalization and causal conclusions. If 3D cannot complete within available resources, retain the measured 2D prototype and exact failure record rather than claiming a successful 3D result. Unsafe contours require manual correction; neither model is approved for unattended clinical use. Publish source and anonymous report material; keep raw arrays and weights outside Git.

References: [U-Net](https://arxiv.org/abs/1505.04597); [residual/deep-supervision reference](https://arxiv.org/abs/1802.10508), implemented as an adaptation; [CSIRO HipMRI](https://doi.org/10.25919/45t8-p065); [official label key](https://data.csiro.au/dap/ws/v2/collections/52549/support/5034).

## References

- Ronneberger et al., U-Net (2015): https://arxiv.org/abs/1505.04597.
- Isensee et al., residual units/deep supervision architectural reference: https://arxiv.org/abs/1802.10508. The implementation is an adaptation, not a claim of exact nnU-Net reproduction.
- HipMRI, Dowling and Greer, CSIRO v2: https://doi.org/10.25919/45t8-p065. Official labels 0 background, 1 body, 2 bones, 3 bladder, 4 rectum and 5 prostate are preserved in label_schema.json.
