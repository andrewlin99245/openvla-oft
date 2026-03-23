#!/bin/bash
set -euo pipefail

DATA_ROOT_DIR="${1:-/project/6003584/tsungen/datasets/modified_libero_rlds}"
MAX_SAMPLES="${2:-512}"
BATCH_SIZE="${3:-1}"

REPO_DIR="/project/6003584/tsungen/openvla-oft"
VENV_DIR="${VENV_DIR:-$HOME/venvs/openvla-oft-h100}"
MODEL_NAME="${MODEL_NAME:-moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10}"
COEFF="${COEFF:-0.1}"
PHASE="${PHASE:-all}"
PRESERVE_NORM="${PRESERVE_NORM:-}"
RUN_NAME="${RUN_NAME:-full_steering_eval_c${COEFF//./p}_${MAX_SAMPLES}s}"

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

# Use SLURM_TMPDIR only for model cache; outputs go to persistent storage
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  export HF_HOME="${HF_HOME:-$SLURM_TMPDIR/hf_cache}"
else
  export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
fi
export OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/outputs/$RUN_NAME}"

mkdir -p "$HF_HOME"
mkdir -p "$OUTPUT_DIR"

EXTRA_ARGS=""
if [[ -n "$PRESERVE_NORM" ]]; then
  EXTRA_ARGS="--preserve-norm"
fi

python experiments/analysis/libero_full_steering_eval.py \
  --data-root-dir "$DATA_ROOT_DIR" \
  --pretrained-checkpoint "$MODEL_NAME" \
  --batch-size "$BATCH_SIZE" \
  --max-samples "$MAX_SAMPLES" \
  --coeff "$COEFF" \
  --phase "$PHASE" \
  --output-dir "$OUTPUT_DIR" \
  $EXTRA_ARGS

echo "Outputs written to: $OUTPUT_DIR"
