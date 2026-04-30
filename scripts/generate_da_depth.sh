#!/usr/bin/env bash
# Set up Depth-Anything-V2 and generate pseudo-depth for SemanticKITTI.
# Usage: ./scripts/generate_da_depth.sh <semantic_kitti_dataset_root>

set -eu

SKITTI_ROOT="${1:?usage: $0 <skitti_dataset_root>}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DA="$REPO_ROOT/third_party/Depth-Anything-V2"
WEIGHTS_DIR="$DA/metric_depth/checkpoints"
WEIGHTS="$WEIGHTS_DIR/depth_anything_v2_metric_vkitti_vits.pth"

if [ ! -d "$DA" ]; then
    git clone https://github.com/DepthAnything/Depth-Anything-V2 "$DA"
fi

if [ ! -f "$WEIGHTS" ]; then
    mkdir -p "$WEIGHTS_DIR"
    hf download depth-anything/Depth-Anything-V2-Metric-VKITTI-Small \
        depth_anything_v2_metric_vkitti_vits.pth \
        --local-dir "$WEIGHTS_DIR"
fi

python "$REPO_ROOT/scripts/generate_da_depth.py" --skitti_root "$SKITTI_ROOT"
