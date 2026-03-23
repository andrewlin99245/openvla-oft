#!/bin/bash
set -euo pipefail

DATA_ROOT_DIR="${1:-/project/6003584/tsungen/datasets/modified_libero_rlds}"
MAX_SAMPLES="${2:-512}"
BATCH_SIZE="${3:-1}"

REPO_DIR="/project/6003584/tsungen/openvla-oft"
VENV_DIR="${VENV_DIR:-$HOME/venvs/openvla-oft}"
MODEL_NAME="${MODEL_NAME:-moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10}"
UNNORM_KEY="${UNNORM_KEY:-libero_spatial_no_noops}"
DATASETS="${DATASETS:-libero_goal_no_noops,libero_spatial_no_noops,libero_object_no_noops}"
LAYERS="${LAYERS:-all}"
RUN_NAME="${RUN_NAME:-libero_vector_similarity_${MAX_SAMPLES}s}"

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

python experiments/analysis/libero_vector_similarity.py \
  --data-root-dir "$DATA_ROOT_DIR" \
  --pretrained-checkpoint "$MODEL_NAME" \
  --unnorm-key "$UNNORM_KEY" \
  --datasets "$DATASETS" \
  --batch-size "$BATCH_SIZE" \
  --max-samples "$MAX_SAMPLES" \
  --layers "$LAYERS" \
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
