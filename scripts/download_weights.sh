#!/usr/bin/env bash
# Download deployed VOOM weights from HuggingFace.
# Usage: ./scripts/download_weights.sh
set -eu

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$REPO_ROOT/release"
mkdir -p "$DEST"

hf download abielalmonte/voom voom.safetensors --local-dir "$DEST"

echo "weights at $DEST/voom.safetensors"
