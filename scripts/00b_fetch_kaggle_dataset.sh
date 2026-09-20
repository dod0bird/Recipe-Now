#!/usr/bin/env bash
# Fetch the Epicurious "Food Ingredients and Recipes Dataset with Images"
# from Kaggle: ~13,500 real dishes, each with its real matching photo.
#
# One-time setup before running this -- create a free Kaggle account, then
# go to https://www.kaggle.com/settings/account -> API -> "Create New Token".
# Kaggle currently issues one of two token formats; either works here:
#   - a kaggle.json file -> save it at ~/.kaggle/kaggle.json, then:
#       chmod 600 ~/.kaggle/kaggle.json
#   - a KGAT_... token string -> save it at ~/.kaggle/access_token, then:
#       chmod 600 ~/.kaggle/access_token
#
# This script requires network access to kaggle.com, which is NOT available
# in Claude's sandboxed environment -- run this on your own machine.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f ~/.kaggle/kaggle.json ] && [ ! -f ~/.kaggle/access_token ] && [ -z "${KAGGLE_API_TOKEN:-}" ]; then
  echo "No Kaggle credentials found (~/.kaggle/kaggle.json, ~/.kaggle/access_token," >&2
  echo "or KAGGLE_API_TOKEN env var) -- see the comment at the top of this script." >&2
  exit 1
fi

pip show kaggle > /dev/null 2>&1 || pip install kaggle

mkdir -p data/raw/kaggle_epicurious
kaggle datasets download \
  -d pes12017000148/food-ingredients-and-recipe-dataset-with-images \
  -p data/raw/kaggle_epicurious --unzip

echo "Kaggle dataset ready in data/raw/kaggle_epicurious/"
