"""
FastAPI backend for Recipe Now.

Loads the CLIP model once at startup, embeds an uploaded photo or a text
query into the same vector space the recipe corpus was indexed in, and
returns the nearest recipes from the FAISS index.
"""
from __future__ import annotations

import os

# torch and faiss each ship their own OpenMP runtime; loading both in one
# process aborts on macOS with "OMP Error #15" unless this is set before
# either library is imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import io
import json
from contextlib import asynccontextmanager
from pathlib import Path

import faiss
import numpy as np
import open_clip
import pandas as pd
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
INDEX_DIR = ROOT / "data" / "index"
FRONTEND_DIR = ROOT / "frontend"
MODEL_WEIGHTS = ROOT / "models" / "clip_vit_b32_laion400m.pt"
MODEL_NAME = "ViT-B-32-quickgelu"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K_DEFAULT = 12


class SearchState:
    def __init__(self):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=str(MODEL_WEIGHTS)
        )
        self.tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        self.model.eval().to(DEVICE)

        self.recipes = pd.read_parquet(PROCESSED_DIR / "recipes.parquet")
        self.image_index = faiss.read_index(str(INDEX_DIR / "image.index"))
        self.text_index = faiss.read_index(str(INDEX_DIR / "text.index"))

        cluster_path = PROCESSED_DIR / "cluster_map.json"
        self.cluster_map = json.loads(cluster_path.read_text()) if cluster_path.exists() else None

    def embed_image(self, image: Image.Image) -> np.ndarray:
        tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feats = self.model.encode_image(tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype("float32")

    def embed_text(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text], context_length=77).to(DEVICE)
        with torch.no_grad():
            feats = self.model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype("float32")

    def results_from(self, index: faiss.Index, query_vec: np.ndarray, top_k: int):
        scores, indices = index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            row = self.recipes.iloc[int(idx)]
            results.append({
                "id": int(row["id"]),
                "title": row["title"],
                "category": row["category"],
                "ingredients": list(row["ingredients"]),
                "instructions": row["instructions"],
                "image_url": f"/images/{Path(row['image_path']).name}",
                "score": float(score),
            })
        return results


state: SearchState | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global state
    state = SearchState()
    yield


app = FastAPI(title="Recipe Now", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class TextQuery(BaseModel):
    query: str
    top_k: int = TOP_K_DEFAULT


@app.post("/search/image")
async def search_by_image(file: UploadFile = File(...), top_k: int = TOP_K_DEFAULT):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    query_vec = state.embed_image(image)
    return {"results": state.results_from(state.image_index, query_vec, top_k)}


@app.post("/search/text")
async def search_by_text(payload: TextQuery):
    query_vec = state.embed_text(payload.query)
    # text queries are matched against the recipe *image* embeddings, since
    # CLIP puts both in the same space -- this is what lets "spicy noodles"
    # retrieve photos that look like spicy noodles.
    return {"results": state.results_from(state.image_index, query_vec, payload.top_k)}


@app.get("/cluster-map")
def cluster_map():
    return state.cluster_map


@app.get("/health")
def health():
    return {"status": "ok", "recipes": len(state.recipes) if state else 0}


app.mount("/images", StaticFiles(directory=str(PROCESSED_DIR / "recipe_images")), name="images")
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
