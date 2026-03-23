#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/setup_h100_env.sh /path/to/venv"
  exit 1
fi

ENV_DIR="$1"
PY_MODULE="${PY_MODULE:-python/3.10}"
STDENV_MODULE="${STDENV_MODULE:-StdEnv/2023}"

module load "$STDENV_MODULE" "$PY_MODULE"

virtualenv --no-download "$ENV_DIR"
source "$ENV_DIR/bin/activate"

python -m pip install --upgrade pip

# Install the matching Compute Canada 2.5.1 stack. This torch build exposes
# `sm_90`, which is sufficient for H100 nodes, and avoids mixed-wheel issues.
python -m pip install --no-index torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

python -m pip install --no-index "numpy<2" "tensorflow>=2.15,<2.18" pandas
python -m pip install --no-deps "dlimp @ git+https://github.com/moojink/dlimp_openvla"

python -m pip install -e /project/6003584/tsungen/openvla-oft

echo
echo "Environment ready at: $ENV_DIR"
echo "Activate with: source $ENV_DIR/bin/activate"
