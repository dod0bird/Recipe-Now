"""
Build the recipe corpus: a real dish photo mapped to its real recipe.

Two possible sources, tried in this order:

1. data/raw/kaggle_epicurious/ -- the full Kaggle "Food Ingredients and
   Recipes Dataset with Images" (~13,500 real Epicurious dishes, each with
   its real photo). Fetch it with scripts/00b_fetch_kaggle_dataset.sh (run
   on your own machine -- Kaggle isn't reachable from Claude's sandbox).
   Since this dataset ships no dish-category column, one is derived here
   with a simple keyword heuristic, used only to label groups on the
   cluster map -- it has no effect on search/ranking, which is entirely
   embedding-based.

2. data/custom_dishes/ -- a small, hand-curated recipes.csv + images/ (see
   data/custom_dishes/README.md). Good for a first smoke test before
   pulling in the full dataset above.

Either way the output is the same: data/processed/recipes.parquet +
data/processed/recipe_images/, which scripts 02 and 03 consume unchanged.
"""
from __future__ import annotations

import ast
import random
import re
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
KAGGLE_DIR = ROOT / "data" / "raw" / "kaggle_epicurious"
CUSTOM_DIR = ROOT / "data" / "custom_dishes"
CUSTOM_CSV_PATH = CUSTOM_DIR / "recipes.csv"
CUSTOM_IMAGES_DIR = CUSTOM_DIR / "images"
PROCESSED_DIR = ROOT / "data" / "processed"
IMAGES_OUT = PROCESSED_DIR / "recipe_images"

# cap the corpus for a reasonable embedding runtime on a laptop CPU -- still
# comfortably clears the "at least 5,000 recipes" target. Set to None (or a
# bigger number) to use the full dataset instead.
MAX_RECIPES = 6000
RANDOM_SEED = 42

CUSTOM_REQUIRED_COLUMNS = {"title", "image_filename", "ingredients", "instructions"}
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

# simple keyword -> dish-type bucket, checked in order, first match wins.
# This is only used to label groups on the /cluster.html map -- it has no
# effect on search or ranking. Word-boundary matching avoids the obvious
# substring traps (e.g. "egg" inside "eggplant", "ham" inside "hamburger",
# "tart" inside "steak tartare").
#
# Order matters: protein/seafood identity is checked first since it's the
# most specific, reliable signal in a title (e.g. "Crab Cakes" should read
# as Seafood, not Cake & Cupcakes just because of the word "cake"; "Miso-
# Butter Roast Chicken" should read as Chicken, not Japanese just because
# it mentions miso). Distinct dish-forms and cuisines come next, and the
# most generic dessert/baked-good buckets are checked last.
CATEGORY_KEYWORDS = [
    ("Seafood", ["fish", "shrimp", "salmon", "tuna", "crab", "lobster", "scallop",
                 "seafood", "cod", "halibut", "oyster", "clam", "mussel", "ceviche"]),
    ("Chicken & Poultry", ["chicken", "turkey", "duck", "poultry"]),
    ("Beef", ["beef", "steak", "brisket", "short rib", "meatloaf", "meatball"]),
    ("Pork", ["pork", "bacon", "ham", "sausage", "prosciutto"]),
    ("Lamb", ["lamb"]),
    ("Egg Dishes", ["egg", "eggs", "omelet", "frittata", "quiche"]),
    ("Pizza", ["pizza"]),
    ("Curry", ["curry"]),
    ("Mexican", ["taco", "burrito", "quesadilla", "enchilada", "guacamole", "nachos", "fajita"]),
    ("Mediterranean", ["hummus", "falafel", "kebab", "tzatziki", "tabbouleh", "shawarma", "pita"]),
    ("Sushi & Japanese", ["sushi", "ramen", "udon", "teriyaki", "tempura", "miso"]),
    ("Asian Stir-Fry & Dumplings", ["stir-fry", "stir fry", "dumpling", "pad thai",
                                     "kimchi", "hoisin", "spring roll", "potsticker"]),
    ("Pasta & Noodles", ["pasta", "spaghetti", "noodle", "lasagna", "ravioli", "macaroni",
                          "fettuccine", "gnocchi", "orzo", "linguine", "ziti"]),
    ("Soup & Stew", ["soup", "stew", "chowder", "bisque", "gumbo", "chili"]),
    ("Salad", ["salad"]),
    ("Sandwich & Burgers", ["sandwich", "burger", "wrap", "panini", "sub"]),
    ("Vegetable Dishes", ["eggplant", "broccoli", "spinach", "kale", "cauliflower",
                           "zucchini", "mushroom", "vegetable"]),
    ("Rice & Grains", ["risotto", "quinoa", "rice", "barley", "couscous"]),
    ("Cake & Cupcakes", ["cake", "cupcake"]),
    ("Cookie", ["cookie", "biscotti"]),
    ("Brownies & Bars", ["brownie", "blondie", "lemon bar", "date bar"]),
    ("Pie & Tart", ["pie", "tart", "tarte"]),
    ("Cobbler & Crisp", ["cobbler", "crisp", "crumble"]),
    ("Bread & Baked Goods", ["bread", "biscuit", "muffin", "scone", "bagel", "cinnamon roll", "dinner roll"]),
    ("Breakfast", ["pancake", "waffle", "granola", "oatmeal", "crepe", "french toast"]),
    ("Drinks & Cocktails", ["cocktail", "smoothie", "margarita", "punch", "sangria", "mocktail"]),
    ("Dip & Appetizer", ["dip", "bruschetta", "crostini", "spread"]),
    ("Dessert", ["chocolate", "ice cream", "pudding", "custard", "mousse",
                 "sorbet", "cheesecake", "tiramisu", "fudge"]),
]

# keywords that commonly appear fused into a larger word ("burger" inside
# "hamburger"/"cheeseburger") need the leading boundary relaxed, or they'd
# never match at all.
_SUFFIX_ONLY_KEYWORDS = {"burger"}


def _keyword_pattern(kw: str) -> str:
    escaped = re.escape(kw)
    # trailing "s?" catches simple plurals ("cakes", "rolls", "tacos");
    # irregular plurals (e.g. "sandwiches") aren't covered by this heuristic.
    if kw in _SUFFIX_ONLY_KEYWORDS:
        return escaped + r"s?\b"
    return r"\b" + escaped + r"s?\b"


_CATEGORY_PATTERNS = [
    (name, re.compile("|".join(_keyword_pattern(kw) for kw in keywords), re.IGNORECASE))
    for name, keywords in CATEGORY_KEYWORDS
]


def categorize(title: str) -> str:
    for category, pattern in _CATEGORY_PATTERNS:
        if pattern.search(title):
            return category
    return "Other"


def parse_ingredients(raw: str) -> list[str]:
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed]
    except (ValueError, SyntaxError):
        pass
    return [p.strip() for p in re.split(r",\s*", str(raw)) if p.strip()]


def find_kaggle_csv() -> Path | None:
    if not KAGGLE_DIR.is_dir():
        return None
    for csv_path in KAGGLE_DIR.rglob("*.csv"):
        try:
            header = pd.read_csv(csv_path, nrows=0).columns
        except Exception:
            continue
        if {"Title", "Ingredients", "Instructions", "Image_Name"}.issubset(set(header)):
            return csv_path
    return None


def find_kaggle_images_dir(csv_path: Path) -> Path | None:
    best_dir, best_count = None, 0
    for candidate in csv_path.parent.rglob("*"):
        if not candidate.is_dir():
            continue
        count = sum(1 for _ in candidate.glob("*.jpg")) + sum(1 for _ in candidate.glob("*.png"))
        if count > best_count:
            best_dir, best_count = candidate, count
    return best_dir


def resolve_kaggle_image(images_dir: Path, image_name: str) -> Path | None:
    name = str(image_name).strip()
    if not name or name.lower() in {"nan", "#name?"}:
        return None
    candidate = images_dir / name
    if candidate.suffix.lower() in IMAGE_EXTENSIONS and candidate.exists():
        return candidate
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{name}{ext}"
        if candidate.exists():
            return candidate
    return None


def build_from_kaggle(csv_path: Path) -> pd.DataFrame:
    images_dir = find_kaggle_images_dir(csv_path)
    if images_dir is None:
        raise SystemExit(
            f"Found {csv_path} but couldn't find an images folder near it. "
            f"Check that scripts/00b_fetch_kaggle_dataset.sh finished downloading."
        )
    print(f"Reading {csv_path}")
    print(f"Using images from {images_dir}")

    df = pd.read_csv(csv_path, index_col=0)
    df = df.dropna(subset=["Title", "Ingredients", "Instructions"])
    print(f"Loaded {len(df)} raw recipes")

    if MAX_RECIPES is not None and len(df) > MAX_RECIPES:
        df = df.sample(n=MAX_RECIPES, random_state=RANDOM_SEED)
        print(f"Sampled down to {MAX_RECIPES} recipes for a manageable embedding runtime")

    IMAGES_OUT.mkdir(parents=True, exist_ok=True)
    records = []
    skipped_no_image = 0
    for recipe_id, row in df.iterrows():
        src_image = resolve_kaggle_image(images_dir, row.get("Image_Name", ""))
        if src_image is None:
            skipped_no_image += 1
            continue

        ingredients = parse_ingredients(row["Ingredients"])
        title = str(row["Title"]).strip()
        dest_name = f"{recipe_id}{src_image.suffix.lower()}"
        shutil.copyfile(src_image, IMAGES_OUT / dest_name)

        records.append({
            "id": int(recipe_id),
            "title": title,
            "ingredients": ingredients,
            "ingredients_text": "; ".join(ingredients),
            "instructions": str(row["Instructions"]).strip(),
            "category": categorize(title),
            "image_path": f"recipe_images/{dest_name}",
        })

    print(f"Skipped {skipped_no_image} recipes with a missing/invalid image")
    return pd.DataFrame.from_records(records)


def build_from_custom() -> pd.DataFrame:
    if not CUSTOM_CSV_PATH.exists():
        raise SystemExit(
            f"No dataset found. Either:\n"
            f"  - run scripts/00b_fetch_kaggle_dataset.sh to get the full "
            f"Kaggle dataset (expected at {KAGGLE_DIR}), or\n"
            f"  - create {CUSTOM_CSV_PATH} for a small hand-curated set "
            f"(see data/custom_dishes/README.md)"
        )

    df = pd.read_csv(CUSTOM_CSV_PATH)
    missing = CUSTOM_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise SystemExit(f"recipes.csv is missing required column(s): {sorted(missing)}")

    IMAGES_OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for i, row in df.iterrows():
        image_filename = str(row["image_filename"]).strip()
        src_image = CUSTOM_IMAGES_DIR / image_filename
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

    return pd.DataFrame.from_records(records)


def main():
    random.seed(RANDOM_SEED)
    kaggle_csv = find_kaggle_csv()

    if kaggle_csv is not None:
        print("Using the full Kaggle dataset")
        out_df = build_from_kaggle(kaggle_csv)
    else:
        print(f"No Kaggle dataset found at {KAGGLE_DIR}, using data/custom_dishes/ instead")
        out_df = build_from_custom()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(PROCESSED_DIR / "recipes.parquet", index=False)

    print(f"\nBuilt corpus of {len(out_df)} dishes")
    print(out_df["category"].value_counts())
    print(f"Saved to {PROCESSED_DIR / 'recipes.parquet'}")


if __name__ == "__main__":
    main()
