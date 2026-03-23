#!/bin/bash
set -euo pipefail

export DATASET_NAME="${DATASET_NAME:-libero_object_no_noops}"
export UNNORM_KEY="${UNNORM_KEY:-libero_object_no_noops}"
export MODEL_NAME="${MODEL_NAME:-moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10}"
export RUN_NAME="${RUN_NAME:-${DATASET_NAME}_sira_${2:-1024}s}"

exec bash /project/6003584/tsungen/openvla-oft/scripts/run_libero_spatial_sira.sh "$@"
