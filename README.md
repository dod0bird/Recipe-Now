# Recipe Now — find a recipe from a photo of a dish

Take a picture of a finished dish, get back the closest matching recipes. A
text search box also works, using the same embedding space.

## How it works

1. **Recipe corpus** — a hand-curated set of real dishes, each with a real
   photo of the finished dish and its actual recipe (title, ingredients,
   instructions). See `data/custom_dishes/README.md` for the exact format
   and how to add your own dishes.
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
scripts/               data pipeline: build corpus, embed, index, cluster
backend/                FastAPI app serving search + static images
frontend/               single-page UI (vanilla JS)
data/custom_dishes/     your hand-curated dish photos + recipes.csv (see its README)
data/processed/         cleaned recipes.parquet + recipe_images/ + cluster_map.json (generated)
data/index/             FAISS indices + embedding matrices (generated)
```

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

bash scripts/00_fetch_data.sh   # downloads the open-source CLIP weights
```

Then fill in `data/custom_dishes/recipes.csv` and `data/custom_dishes/images/`
per `data/custom_dishes/README.md` (start with ~10 dishes), and run:

```bash
python scripts/01_build_corpus.py       # read your dishes into recipes.parquet
python scripts/02_embed_and_index.py    # CLIP-embed everything, build FAISS index
python scripts/03_cluster.py            # K-Means + UMAP cluster map

uvicorn backend.main:app --reload
# open http://localhost:8000  (search UI + /cluster.html)
```

## Scaling up

Once the small set works end-to-end, growing the corpus is just adding more
rows to `recipes.csv` (and more photos to `images/`) and re-running steps
01-03. The pipeline and app code don't change — `MAX_CLUSTERS` and UMAP's
neighbor count in `scripts/03_cluster.py` already scale automatically with
corpus size.
