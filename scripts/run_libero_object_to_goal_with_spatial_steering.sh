#!/bin/bash
set -euo pipefail

DATA_ROOT_DIR="${1:-/project/6003584/tsungen/datasets/modified_libero_rlds}"
MAX_SAMPLES="${2:-512}"
BATCH_SIZE="${3:-1}"

REPO_DIR="/project/6003584/tsungen/openvla-oft"
VENV_DIR="${VENV_DIR:-$HOME/venvs/openvla-oft}"
VECTOR_DATASET_NAME="${VECTOR_DATASET_NAME:-libero_object_no_noops}"
EVAL_DATASET_NAME="${EVAL_DATASET_NAME:-libero_goal_no_noops}"
UNNORM_KEY="${UNNORM_KEY:-libero_spatial_no_noops}"
MODEL_NAME="${MODEL_NAME:-moojink/openvla-7b-oft-finetuned-libero-spatial}"
LAYERS="${LAYERS:-all}"
COEFF="${COEFF:-0.1}"
RUN_NAME="${RUN_NAME:-${VECTOR_DATASET_NAME}_to_${EVAL_DATASET_NAME}_with_spatial_steering_${LAYERS//,/_}_c${COEFF//./p}_${MAX_SAMPLES}s}"

if [[ ! -d "$DATA_ROOT_DIR" ]]; then
  echo "Dataset root not found: $DATA_ROOT_DIR"
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Virtualenv not found: $VENV_DIR"
  exit 1
fi

cd "$REPO_DIR"
source "$VENV_DIR/bin/activate"

if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  export HF_HOME="${HF_HOME:-$SLURM_TMPDIR/hf_cache}"
  export OUTPUT_DIR="${OUTPUT_DIR:-$SLURM_TMPDIR/$RUN_NAME}"
else
  export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
  export OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/outputs/$RUN_NAME}"
fi

mkdir -p "$HF_HOME"
mkdir -p "$(dirname "$OUTPUT_DIR")"

python experiments/analysis/libero_steering_eval.py \
  --data-root-dir "$DATA_ROOT_DIR" \
  --vector-dataset-name "$VECTOR_DATASET_NAME" \
  --eval-dataset-name "$EVAL_DATASET_NAME" \
  --pretrained-checkpoint "$MODEL_NAME" \
  --unnorm-key "$UNNORM_KEY" \
  --batch-size "$BATCH_SIZE" \
  --max-samples "$MAX_SAMPLES" \
  --layers "$LAYERS" \
  --coeff "$COEFF" \
  --output-dir "$OUTPUT_DIR"

if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  FINAL_OUTPUT_DIR="${FINAL_OUTPUT_DIR:-$REPO_DIR/outputs/$RUN_NAME}"
  mkdir -p "$REPO_DIR/outputs"
  rm -rf "$FINAL_OUTPUT_DIR"
  cp -r "$OUTPUT_DIR" "$FINAL_OUTPUT_DIR"
  echo "Copied outputs to: $FINAL_OUTPUT_DIR"
else
  echo "Outputs written to: $OUTPUT_DIR"
fi
