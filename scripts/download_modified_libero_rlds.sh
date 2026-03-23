#!/bin/bash
set -euo pipefail

DEST_DIR="${1:-/project/6003584/tsungen/datasets/modified_libero_rlds}"

if [[ -e "$DEST_DIR" ]]; then
  echo "Destination already exists: $DEST_DIR"
  exit 1
fi

git clone https://huggingface.co/datasets/openvla/modified_libero_rlds "$DEST_DIR"

echo "Downloaded dataset to: $DEST_DIR"
