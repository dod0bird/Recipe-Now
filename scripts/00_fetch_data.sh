#!/usr/bin/env bash
# Fetch everything the pipeline needs that isn't checked into this repo:
# the two source datasets and the open-source CLIP weights.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p data/raw models

if [ ! -d data/raw/recipe-dataset ]; then
  git clone --depth 1 https://github.com/josephrmartinez/recipe-dataset.git data/raw/recipe-dataset
fi

if [ ! -d data/raw/fruit-images ]; then
  git clone --depth 1 https://github.com/Horea94/Fruit-Images-Dataset.git data/raw/fruit-images
fi

if [ ! -f models/clip_vit_b32_laion400m.pt ]; then
  curl -sSL -o models/clip_vit_b32_laion400m.pt \
    "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e32-46683a32.pt"
fi

echo "Data + weights ready."
