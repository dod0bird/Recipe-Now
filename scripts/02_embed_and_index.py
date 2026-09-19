"""
Embed every recipe's photo and text with CLIP (open_clip ViT-B-32, LAION-400M
weights) into a single shared vector space, then build FAISS indices over
each so the backend can do image->recipe and text->recipe search with the
same model.
"""
from __future__ import annotations

import os

# torch and faiss each ship their own OpenMP runtime; loading both in one
# process aborts on macOS with "OMP Error #15" unless this is set before
# either library is imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import faiss
import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
INDEX_DIR = ROOT / "data" / "index"
MODEL_WEIGHTS = ROOT / "models" / "clip_vit_b32_laion400m.pt"

MODEL_NAME = "ViT-B-32-quickgelu"
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_clip():
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=str(MODEL_WEIGHTS)
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model.eval().to(DEVICE)
    return model, preprocess, tokenizer


def embed_images(df: pd.DataFrame, model, preprocess) -> np.ndarray:
    vectors = []
    for start in tqdm(range(0, len(df), BATCH_SIZE), desc="embedding images"):
        batch = df.iloc[start:start + BATCH_SIZE]
        tensors = []
        for rel_path in batch["image_path"]:
            img = Image.open(PROCESSED_DIR / rel_path).convert("RGB")
            tensors.append(preprocess(img))
        batch_tensor = torch.stack(tensors).to(DEVICE)
        with torch.no_grad():
            feats = model.encode_image(batch_tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        vectors.append(feats.cpu().numpy())
    return np.concatenate(vectors, axis=0).astype("float32")


def embed_texts(texts: list[str], model, tokenizer) -> np.ndarray:
    vectors = []
    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="embedding text"):
        batch = texts[start:start + BATCH_SIZE]
        tokens = tokenizer(batch, context_length=77).to(DEVICE)
        with torch.no_grad():
            feats = model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        vectors.append(feats.cpu().numpy())
    return np.concatenate(vectors, axis=0).astype("float32")


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    # vectors are already L2-normalized -> inner product == cosine similarity
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def main():
    df = pd.read_parquet(PROCESSED_DIR / "recipes.parquet")
    print(f"Embedding {len(df)} recipes with {MODEL_NAME} on {DEVICE}")

    model, preprocess, tokenizer = load_clip()

    image_vectors = embed_images(df, model, preprocess)
    # CLIP's shared space lets us search recipes by title + key ingredients too
    text_for_embedding = (df["title"] + ". Key ingredient: " + df["category"]).tolist()
    text_vectors = embed_texts(text_for_embedding, model, tokenizer)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_DIR / "image_vectors.npy", image_vectors)
    np.save(INDEX_DIR / "text_vectors.npy", text_vectors)
    df[["id"]].to_parquet(INDEX_DIR / "ids.parquet", index=False)

    faiss.write_index(build_faiss_index(image_vectors), str(INDEX_DIR / "image.index"))
    faiss.write_index(build_faiss_index(text_vectors), str(INDEX_DIR / "text.index"))

    print(f"Saved embeddings + FAISS indices to {INDEX_DIR}")


if __name__ == "__main__":
    main()
