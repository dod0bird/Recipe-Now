"""
Build the recipe corpus from a small, hand-curated set of dish photos and
their real recipes -- a direct photo-of-the-finished-dish -> recipe mapping,
rather than a stand-in ingredient photo.

Expected input, see data/custom_dishes/README.md for the full format:
  data/custom_dishes/recipes.csv
  data/custom_dishes/images/<file>

recipes.csv required columns: title, image_filename, ingredients,
instructions. Optional: category (defaults to the title).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CUSTOM_DIR = ROOT / "data" / "custom_dishes"
CSV_PATH = CUSTOM_DIR / "recipes.csv"
IMAGES_DIR = CUSTOM_DIR / "images"
PROCESSED_DIR = ROOT / "data" / "processed"
IMAGES_OUT = PROCESSED_DIR / "recipe_images"

REQUIRED_COLUMNS = {"title", "image_filename", "ingredients", "instructions"}


def main():
    if not CSV_PATH.exists():
        raise SystemExit(
            f"Expected {CSV_PATH} but it doesn't exist.\n"
            f"See data/custom_dishes/README.md for the format, then add your "
            f"recipes.csv there and put the dish photos in {IMAGES_DIR}"
        )

    df = pd.read_csv(CSV_PATH)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise SystemExit(f"recipes.csv is missing required column(s): {sorted(missing)}")

    IMAGES_OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for i, row in df.iterrows():
        image_filename = str(row["image_filename"]).strip()
        src_image = IMAGES_DIR / image_filename
        if not src_image.exists():
            raise SystemExit(
                f"Row {i} ('{row['title']}') points to {src_image}, "
                f"which doesn't exist. Check the image_filename column."
            )

        ingredients = [s.strip() for s in str(row["ingredients"]).split(";") if s.strip()]
        if "category" in df.columns and pd.notna(row.get("category")) and str(row["category"]).strip():
            category = str(row["category"]).strip()
        else:
            category = str(row["title"]).strip()

        dest_name = f"{i}{src_image.suffix.lower()}"
        shutil.copyfile(src_image, IMAGES_OUT / dest_name)

        records.append({
            "id": i,
            "title": str(row["title"]).strip(),
            "ingredients": ingredients,
            "ingredients_text": "; ".join(ingredients),
            "instructions": str(row["instructions"]).strip(),
            "category": category,
            "image_path": f"recipe_images/{dest_name}",
        })

    out_df = pd.DataFrame.from_records(records)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(PROCESSED_DIR / "recipes.parquet", index=False)

    print(f"Built corpus from {len(out_df)} hand-curated dishes:")
    for _, row in out_df.iterrows():
        print(f"  - {row['title']}  ({row['image_path']})")
    print(f"Saved to {PROCESSED_DIR / 'recipes.parquet'}")


if __name__ == "__main__":
    main()
