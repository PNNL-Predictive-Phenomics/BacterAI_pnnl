# Transfer Learning (RF) Instructions

This document describes how to run BacterAI in RF transfer-learning mode using timed-hpc logic.

The current transfer mode is:

- Enabled only when `--transfer-learning` is provided.
- RF-only (no MLP/VAE branches).
- Trained end-to-end from bundled timed-hpc source/target datasets (users do not provide model/data paths).

## 1. Build the Docker image

Run this from repository root:

```bash
docker build -f Dockerfile.transfer-rf -t bacterai-transfer-rf .
```

## 2. Start an interactive container

```bash
docker run --rm -it \
  -v "$PWD":/workspaces/BacterAI_pnnl \
  -w /workspaces/BacterAI_pnnl \
  bacterai-transfer-rf \
  bash
```

Inside container, repo path is:

- `/workspaces/BacterAI_pnnl`

## 3. Verify transfer-learning CLI option

```bash
bacterai run -h
```

You should see:

- `--transfer-learning`

## 4. Run transfer-learning mode

Template:

```bash
bacterai run <experiment_path> --transfer-learning
```

Example:

```bash
bacterai run tests/test_rf_experiment --transfer-learning
```
Important: If using the above path, ensure round 1 folder is removed

## 5. Round 1 requirement

Transfer-learning mode currently supports Round 1 only.

If the experiment already contains `Round*` folders, BacterAI may detect a later round and fail with:

- `Transfer-learning mode currently supports Round 1 only.`

To force a clean Round 1 test run:

```bash
rm -rf tests/test_rf_experiment/Round*
bacterai run tests/test_rf_experiment --transfer-learning
```

## 6. Feature overlap requirement

Transfer-learning uses pputida RF feature names from timed-hpc and intersects them with your ingredients.

- At least 2 overlapping features are required.
- If overlap is too small, the run fails with a feature-overlap error.

## 7. Expected outputs

Round folder outputs:

- `batch_meta_<timestamp>.csv`: main output; use this file.
- `run_metrics.json`: source/model selection and recommendation metadata.
- `batch_dp_<...>.csv`: legacy DP export format.

Important: the DP CSV can appear empty/non-informative for transfer-learning results because DP export writes ingredient names only when value equals 0.

## 8. Troubleshooting

### Error: No module named rpy2

Run inside Docker image above. Local env may not include R/rpy2 stack.

### Error: cannot open file .../src/omicstl/r/requirements.R

Use latest code and rebuild image so timed-hpc R bootstrap changes are included:

```bash
docker build --no-cache -f Dockerfile.transfer-rf -t bacterai-transfer-rf .
```

### Error: Transfer-learning mode currently supports Round 1 only

Remove existing `Round*` folders in the test experiment path and rerun.

### Warning spam from pandas FutureWarning during model_utils concat

This is non-fatal and does not invalidate output artifacts.
