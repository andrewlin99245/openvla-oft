#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/setup_cc_env.sh /path/to/venv"
  exit 1
fi

ENV_DIR="$1"
PY_MODULE="${PY_MODULE:-python/3.10}"
STDENV_MODULE="${STDENV_MODULE:-StdEnv/2023}"

module load "$STDENV_MODULE" "$PY_MODULE"

virtualenv --no-download "$ENV_DIR"
source "$ENV_DIR/bin/activate"

python -m pip install --upgrade pip

# Alliance wheels are typically available locally for PyTorch.
python -m pip install --no-index torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0
python -m pip install --no-index "tensorflow>=2.15,<2.18"
python -m pip install --no-index pandas
python -m pip install --no-deps "dlimp @ git+https://github.com/moojink/dlimp_openvla"

# Install OpenVLA-OFT and its git-based dependencies on a login node.
python -m pip install -e .

echo
echo "Environment ready at: $ENV_DIR"
echo "Activate with: source $ENV_DIR/bin/activate"
