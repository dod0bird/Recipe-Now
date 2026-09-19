# Recipe Now — find a recipe from a photo of a dish

Take a picture of food, get back the closest matching recipes. A text search box
also works, using the same embedding space.

## How it works

1. **Recipe corpus** — ~13,500 real recipes (title, ingredients, instructions),
   sourced from the [Epicurious recipe dataset](https://github.com/josephrmartinez/recipe-dataset).
2. **Image corpus** — real food photos from the
   [Fruits-360 dataset](https://github.com/Horea94/Fruit-Images-Dataset) (fruits
   and vegetables, plain-background photographs, GitHub-native so it is
   redistributable). Each recipe is linked to a representative photo of its
   main ingredient(s), extracted from its ingredient list — the original
   Epicurious photos are not redistributable, so this project pairs each
   recipe with a *real* labeled photo of its key ingredient rather than a
   single scraped hero shot. This is the same "photo → ingredient → recipe"
   flow used by real cooking apps.
3. **Embeddings** — every recipe image and every recipe's text are embedded
   with **CLIP** (`ViT-B-32-quickgelu`, LAION-400M weights, via
   [`open_clip`](https://github.com/mlfoundations/open_clip)), an open-source
   CLIP variant. Images and text land in the *same* vector space, which is
   what makes a single index searchable by either a photo or a sentence.
4. **Vector index** — embeddings are indexed with **FAISS** for fast
   approximate nearest-neighbor search.
5. **API** — a **FastAPI** backend exposes `/search/image` (upload a photo)
   and `/search/text` (type a query), both returning ranked recipe results.
6. **Clustering / visualization** — recipe embeddings are clustered with
   **K-Means** (unsupervised learning) and projected to 2D with **UMAP** for
   a visual map of "what recipes are near what," served at `/cluster-map`.

## Project layout

```
scripts/            data pipeline: build corpus, embed, index, cluster
backend/             FastAPI app serving search + static images
frontend/            single-page UI (vanilla JS)
data/raw/            cloned source datasets (git-ignored, regenerate via scripts)
data/processed/      cleaned recipes.parquet + recipe_images/ + cluster_map.json
data/index/          FAISS indices + embedding matrices
```

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

bash scripts/00_fetch_data.sh           # clone the 2 source datasets + CLIP weights
python scripts/01_build_corpus.py       # clean recipes, link to ingredient photos
python scripts/02_embed_and_index.py    # CLIP-embed everything, build FAISS index
python scripts/03_cluster.py            # K-Means + UMAP cluster map

uvicorn backend.main:app --reload
# open http://localhost:8000  (search UI + /cluster.html)
```

## Current corpus stats

- **10,350 recipes** matched to a real ingredient photo (76.7% of the 13,493
  source recipes — the rest didn't mention any of the 63 supported
  ingredient categories and were dropped)
- **63 ingredient categories**, e.g. Onion, Lemon, Tomato, Potato, Bell
  Pepper, Coconut, Ginger, Strawberry
- CLIP embeddings: 512-dim, `ViT-B-32-quickgelu` / LAION-400M
- Photo search example: a photo of an orange bell pepper returns
  "Slow-Roasted Cod with Bell Peppers and Capers" at 99% cosine similarity
- 12 K-Means clusters over the embedding space, visualized via UMAP at
  `/cluster.html`

## Honest limitations (worth knowing before you demo it)

- Images are photos of **ingredients**, not of finished plated dishes — the
  original Epicurious recipe photos aren't redistributable, so this project
  substitutes a real photo of the recipe's key ingredient. A photo of a
  finished lasagna won't match well; a photo of a tomato, lemon, or onion
  will.
- Text search reuses the image index (both live in CLIP's shared space), so
  it works best for ingredient-forward queries ("lemon dessert") rather than
  abstract ones ("something comforting for a rainy day").
