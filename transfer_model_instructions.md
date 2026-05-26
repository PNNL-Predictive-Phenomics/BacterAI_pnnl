# Transfer RF Model Instructions

This document describes how to run BacterAI with timed-hpc transfer random forest (RF) models in Docker.

## 1. Build the Docker image

Run this from the repository root:

```bash
docker build -f Dockerfile.transfer-rf -t bacterai-transfer-rf .
```

## 2. Start a container with this repo mounted

```bash
docker run --rm -it \
  -v "$PWD":/workspaces/BacterAI_pnnl \
  -w /workspaces/BacterAI_pnnl \
  bacterai-transfer-rf
```

Inside the container, this repository is available at:

- `/workspaces/BacterAI_pnnl`

## 3. Verify the run command includes transfer RF support

```bash
bacterai run -h
```

You should see:

- `--transfer-rf-pkl TRANSFER_RF_PKL`

## 4. Run BacterAI using a transfer RF model

Template command:

```bash
bacterai run <experiment_path> \
  --transfer-rf-pkl <path_to_transfer_rf_model.pkl>
```

Example command:

```bash
bacterai run /workspaces/BacterAI_pnnl/tests/test_rf \
  --transfer-rf-pkl /workspaces/BacterAI_pnnl/timed-hpc/viral_use_case/model_outputs/rf_model.pkl
```

## 5. Important schema requirement

The transfer RF model must be trained with the same feature columns that your experiment provides.

- If the model was trained on proteomics columns (for example, many `*_HUMAN` features) and your experiment has media ingredient columns, prediction will fail.
- This is expected behavior and indicates a feature schema mismatch, not a Docker or environment issue.

## 6. Recommended pre-run check

Before using a model in production runs, verify:

1. The `.pkl` loads in the container.
2. The model accepts a test row with your experiment feature columns.
3. Expected feature count is aligned with your ingredients schema.

## 7. Troubleshooting

### Error: timed-hpc dependencies unavailable

Use the Docker workflow above. The image includes R, rpy2, and timed-hpc runtime dependencies.

### Error: variables in the training data missing in newdata

This indicates model-feature mismatch.

Fix by using a transfer RF `.pkl` trained on the same feature schema as your BacterAI experiment.

### Error: --transfer-rf-pkl not found

Ensure you are running the current repository code inside the container and not an older image.

Rebuild with:

```bash
docker build --no-cache -f Dockerfile.transfer-rf -t bacterai-transfer-rf .
```
