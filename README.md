# Recipe Now — find a recipe from a photo of a dish

Take a picture of a finished dish, get back the closest matching recipes. A
text search box also works, using the same embedding space.

## How it works

1. **Recipe corpus** — real dishes, each with a real photo of the finished
   dish and its actual recipe (title, ingredients, instructions). Two
   sources are supported (see "Building the corpus" below): the full Kaggle
   Epicurious dataset (~13,500 dishes, sampled down to a few thousand), or a
   small hand-curated set for quick iteration.
2. **Embeddings** — every dish photo and every recipe's text are embedded
   with **CLIP** (`ViT-B-32-quickgelu`, LAION-400M weights, via
   [`open_clip`](https://github.com/mlfoundations/open_clip)), an open-source
   CLIP variant. Images and text land in the *same* vector space, which is
   what makes a single index searchable by either a photo or a sentence.
3. **Vector index** — embeddings are indexed with **FAISS** for fast
   approximate nearest-neighbor search.
4. **API** — a **FastAPI** backend exposes `/search/image` (upload a photo)
   and `/search/text` (type a query), both returning ranked recipe results.
5. **Clustering / visualization** — recipe embeddings are clustered with
   **K-Means** (unsupervised learning) and projected to 2D with **UMAP** for
   a visual map of "what recipes are near what," served at `/cluster-map`.

## Project layout

```
scripts/               data pipeline: fetch, build corpus, embed, index, cluster
backend/                FastAPI app serving search + static images
frontend/               single-page UI (vanilla JS)
data/custom_dishes/     small hand-curated dish photos + recipes.csv (see its README)
data/raw/kaggle_epicurious/   full Kaggle dataset once fetched (git-ignored)
data/processed/         cleaned recipes.parquet + recipe_images/ + cluster_map.json (generated)
data/index/             FAISS indices + embedding matrices (generated)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

bash scripts/00_fetch_data.sh   # downloads the open-source CLIP weights
```

## Building the corpus

`scripts/01_build_corpus.py` uses whichever of these it finds, in order:

**Option A — the full dataset (~13,500 real dishes, real photos), recommended.**
Requires a free Kaggle account (Kaggle isn't reachable from a sandboxed
environment, so this step has to run on your own machine):

1. Create an API token at https://www.kaggle.com/settings/account →
   API → "Create New Token" (Kaggle currently issues a `KGAT_...` bearer
   token string; a legacy `kaggle.json` username+key file also works if
   that's what you get).
2. `mkdir -p ~/.kaggle && echo YOUR_TOKEN_HERE > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token`
3. `bash scripts/00b_fetch_kaggle_dataset.sh`

This uses `curl` directly against Kaggle's REST API rather than the
`kaggle` pip package -- that package's support for the newer bearer-token
format requires Python 3.11+, which would force a Python upgrade for no
good reason.

The corpus builder samples this down to `MAX_RECIPES` (6,000 by default —
edit the constant at the top of `scripts/01_build_corpus.py` to change it)
to keep CLIP embedding time reasonable on a laptop CPU, while still clearing
a 5,000+ recipe target. Since this dataset has no dish-category column, one
is auto-derived per recipe from a keyword heuristic on its title (pizza,
soup, dessert, etc.) purely to label groups on the cluster map — it has no
effect on search or ranking, which is entirely embedding-based.

**Option B — a small hand-curated set**, for a fast first pass before
pulling in the full dataset. See `data/custom_dishes/README.md` for the
format (a `recipes.csv` + `images/` folder you fill in yourself).

Then, regardless of which source you used:

```bash
python scripts/01_build_corpus.py       # build recipes.parquet from whichever source is present
python scripts/02_embed_and_index.py    # CLIP-embed everything, build FAISS index (~1 min per ~150 photos on CPU)
python scripts/03_cluster.py            # K-Means + UMAP cluster map

uvicorn backend.main:app --reload
# open http://localhost:8000  (search UI + /cluster.html)
```

## Notes on the data-processing bonus

At this dataset's scale (thousands of rows, a few hundred MB of images),
pandas is the appropriate tool for the corpus-cleaning step in
`scripts/01_build_corpus.py` — reaching for PySpark here would be
distributed-computing overhead the data doesn't need. If you want a PySpark
version of the same ETL step for the resume line, ask and it can be added
as a supplementary script alongside the pandas one.
