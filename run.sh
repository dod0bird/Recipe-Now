#!/usr/bin/env bash
# Start the app. This only loads the already-built index from disk --
# it does NOT re-run the embedding pipeline (see scripts/01-03 for that).
set -euo pipefail
cd "$(dirname "$0")"

source .venv/bin/activate
uvicorn backend.main:app --reload
