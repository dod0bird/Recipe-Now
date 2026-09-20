#!/usr/bin/env bash
# Fetch the one thing the pipeline needs that isn't checked into this repo
# and isn't yours to provide: the open-source CLIP weights. Your dish photos
# and recipes.csv go in data/custom_dishes/ (see its README).
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p models

if [ ! -f models/clip_vit_b32_laion400m.pt ]; then
  curl -sSL -o models/clip_vit_b32_laion400m.pt \
    "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e32-46683a32.pt"
fi

echo "CLIP weights ready."
