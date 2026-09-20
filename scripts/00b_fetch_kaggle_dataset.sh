#!/usr/bin/env bash
# Fetch the Epicurious "Food Ingredients and Recipes Dataset with Images"
# from Kaggle: ~13,500 real dishes, each with its real matching photo.
#
# One-time setup before running this:
#   1. Create a free Kaggle account: https://www.kaggle.com
#   2. Go to https://www.kaggle.com/settings/account -> "Create New Token".
#      This downloads kaggle.json.
#   3. Put it at ~/.kaggle/kaggle.json (mkdir -p ~/.kaggle first), then:
#        chmod 600 ~/.kaggle/kaggle.json
#
# This script requires network access to kaggle.com, which is NOT available
# in Claude's sandboxed environment -- run this on your own machine.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f ~/.kaggle/kaggle.json ]; then
  echo "Missing ~/.kaggle/kaggle.json -- see the comment at the top of this" >&2
  echo "script for how to get your Kaggle API token." >&2
  exit 1
fi

pip show kaggle > /dev/null 2>&1 || pip install kaggle

mkdir -p data/raw/kaggle_epicurious
kaggle datasets download \
  -d pes12017000148/food-ingredients-and-recipe-dataset-with-images \
  -p data/raw/kaggle_epicurious --unzip

echo "Kaggle dataset ready in data/raw/kaggle_epicurious/"
