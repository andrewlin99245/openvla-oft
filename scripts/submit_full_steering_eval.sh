#!/bin/bash
#SBATCH --job-name=full_steering_eval
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=outputs/slurm/%x_%j.out
#SBATCH --error=outputs/slurm/%x_%j.err

set -euo pipefail

mkdir -p outputs/slurm

DATA_ROOT_DIR="${1:-/project/6003584/tsungen/datasets/modified_libero_rlds}"
MAX_SAMPLES="${2:-512}"

# Override these env vars before sbatch to customize:
#   COEFF=0.05 PHASE=steer PRESERVE_NORM=1 sbatch scripts/submit_full_steering_eval.sh

bash scripts/run_libero_full_steering_eval.sh "$DATA_ROOT_DIR" "$MAX_SAMPLES"
