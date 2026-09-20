"""
Unsupervised exploration of the recipe embedding space: K-Means clusters the
image embeddings into groups, UMAP projects them to 2D for plotting. Output
is a small JSON the frontend scatter-plot page reads directly.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import umap
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
INDEX_DIR = ROOT / "data" / "index"

MAX_CLUSTERS = 12


def main():
    df = pd.read_parquet(PROCESSED_DIR / "recipes.parquet")
    vectors = np.load(INDEX_DIR / "image_vectors.npy")
    assert len(df) == len(vectors)

    # both K-Means and UMAP need their hyperparameters scaled down for a
    # small corpus (e.g. a first pass with 10 hand-picked dishes) --
    # n_clusters can't exceed the number of points, and UMAP's n_neighbors
    # needs to be smaller than the number of points too.
    n_clusters = max(1, min(MAX_CLUSTERS, len(vectors) // 2, len(vectors)))
    n_neighbors = max(1, min(15, len(vectors) - 1))

    print(f"Clustering {len(vectors)} recipes into {n_clusters} groups (K-Means)")
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(vectors)

    print("Projecting to 2D with UMAP")
    # init="random" instead of the default "spectral": spectral init does an
    # eigen-decomposition that outright fails on very small corpora (a first
    # pass with ~10 dishes) and was already silently falling back on the
    # full 10k+ corpus too -- random init works at every scale.
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=0.1, metric="cosine",
        init="random", random_state=42,
    )
    coords = reducer.fit_transform(vectors)

    points = []
    for i, row in enumerate(df.itertuples()):
        points.append({
            "id": int(row.id),
            "title": row.title,
            "category": row.category,
            "cluster": int(cluster_labels[i]),
            "image_path": row.image_path,
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
        })

    # a readable label per cluster: its most common ingredient category
    cluster_labels_readable = {}
    for c in range(n_clusters):
        cats = df["category"].iloc[np.where(cluster_labels == c)[0]]
        cluster_labels_readable[c] = cats.value_counts().idxmax() if len(cats) else f"Cluster {c}"

    out = {
        "n_clusters": n_clusters,
        "cluster_names": cluster_labels_readable,
        "points": points,
    }
    out_path = PROCESSED_DIR / "cluster_map.json"
    out_path.write_text(json.dumps(out))
    print(f"Saved cluster map ({len(points)} points) to {out_path}")


if __name__ == "__main__":
    main()
