#!/usr/bin/env bash
# Fetch the Epicurious "Food Ingredients and Recipes Dataset with Images"
# from Kaggle: ~13,500 real dishes, each with its real matching photo.
#
# One-time setup before running this -- create a free Kaggle account, then
# go to https://www.kaggle.com/settings/account -> API -> "Create New Token".
# Kaggle currently issues a bearer token (KGAT_...) -- save it at
# ~/.kaggle/access_token, then: chmod 600 ~/.kaggle/access_token
# (a legacy kaggle.json with a username+key also works, if that's what you have)
#
# This uses curl directly against Kaggle's REST endpoint rather than the
# `kaggle` pip package: that package's KGAT_ token support requires Python
# 3.11+, and downgrading to a version that installs on older Pythons means
# losing KGAT_ support entirely -- curl has no such constraint.
#
# This script requires network access to kaggle.com, which is NOT available
# in Claude's sandboxed environment -- run this on your own machine.
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET="pes12017000148/food-ingredients-and-recipe-dataset-with-images"
URL="https://www.kaggle.com/api/v1/datasets/download/${DATASET}"
ZIP_PATH="data/raw/kaggle_dataset.zip"
OUT_DIR="data/raw/kaggle_epicurious"

mkdir -p data/raw "$OUT_DIR"

if [ -f ~/.kaggle/access_token ] || [ -n "${KAGGLE_API_TOKEN:-}" ]; then
  TOKEN="${KAGGLE_API_TOKEN:-$(cat ~/.kaggle/access_token)}"
  curl -L -H "Authorization: Bearer ${TOKEN}" -o "$ZIP_PATH" "$URL"
elif [ -f ~/.kaggle/kaggle.json ]; then
  USERNAME=$(python3 -c "import json;print(json.load(open('$HOME/.kaggle/kaggle.json'))['username'])")
  KEY=$(python3 -c "import json;print(json.load(open('$HOME/.kaggle/kaggle.json'))['key'])")
  curl -L -u "${USERNAME}:${KEY}" -o "$ZIP_PATH" "$URL"
else
  echo "No Kaggle credentials found (~/.kaggle/access_token, ~/.kaggle/kaggle.json," >&2
  echo "or KAGGLE_API_TOKEN env var) -- see the comment at the top of this script." >&2
  exit 1
fi

if ! file "$ZIP_PATH" | grep -qi "zip archive"; then
  echo "Download didn't come back as a zip file -- likely an auth error." >&2
  echo "First 300 bytes of the response:" >&2
  head -c 300 "$ZIP_PATH" >&2
  exit 1
fi

unzip -o -q "$ZIP_PATH" -d "$OUT_DIR"
rm "$ZIP_PATH"

echo "Kaggle dataset ready in $OUT_DIR/"
