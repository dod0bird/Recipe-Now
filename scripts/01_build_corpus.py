"""
Build the recipe corpus: clean the Epicurious text recipes and link each one
to a real photo of its primary ingredient.

Why "link to an ingredient photo" instead of a 1:1 recipe photo: the original
Epicurious photos in the source Kaggle dataset are not redistributable and
were dropped from the GitHub mirror we pull text from. Instead we pair each
recipe with a real, openly-licensed photo of the ingredient that drives the
dish (from the Fruits-360 dataset, which ships its images directly in the
git repo). This mirrors the "photo -> ingredient -> recipe" flow many
cooking apps use, and keeps every image in this project a genuine photograph
rather than a placeholder.
"""
import ast
import itertools
import random
import re
import shutil
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
IMAGES_OUT = PROCESSED_DIR / "recipe_images"
RECIPES_CSV = RAW_DIR / "recipe-dataset" / "13k-recipes.csv"
FRUIT_TRAINING_DIR = RAW_DIR / "fruit-images" / "Training"

random.seed(42)

# canonical ingredient name -> Fruits-360 folders whose photos represent it
CATEGORY_FOLDERS = {
    "Apple": ["Apple Braeburn", "Apple Crimson Snow", "Apple Golden 1", "Apple Golden 2",
              "Apple Golden 3", "Apple Granny Smith", "Apple Pink Lady", "Apple Red 1",
              "Apple Red 2", "Apple Red 3", "Apple Red Delicious", "Apple Red Yellow 1",
              "Apple Red Yellow 2"],
    "Apricot": ["Apricot"],
    "Avocado": ["Avocado", "Avocado ripe"],
    "Banana": ["Banana", "Banana Lady Finger", "Banana Red"],
    "Beet": ["Beetroot"],
    "Blueberry": ["Blueberry"],
    "Cactus Fruit": ["Cactus fruit"],
    "Cantaloupe": ["Cantaloupe 1", "Cantaloupe 2"],
    "Starfruit": ["Carambula"],
    "Cauliflower": ["Cauliflower"],
    "Cherry": ["Cherry 1", "Cherry 2", "Cherry Rainier", "Cherry Wax Black",
               "Cherry Wax Red", "Cherry Wax Yellow"],
    "Chestnut": ["Chestnut"],
    "Clementine": ["Clementine"],
    "Coconut": ["Cocos"],
    "Corn": ["Corn", "Corn Husk"],
    "Cucumber": ["Cucumber Ripe", "Cucumber Ripe 2"],
    "Date": ["Dates"],
    "Eggplant": ["Eggplant"],
    "Fig": ["Fig"],
    "Ginger": ["Ginger Root"],
    "Grape": ["Grape Blue", "Grape Pink", "Grape White", "Grape White 2",
              "Grape White 3", "Grape White 4"],
    "Grapefruit": ["Grapefruit Pink", "Grapefruit White"],
    "Guava": ["Guava"],
    "Hazelnut": ["Hazelnut", "Nut Forest"],
    "Huckleberry": ["Huckleberry"],
    "Kiwi": ["Kiwi"],
    "Kohlrabi": ["Kohlrabi"],
    "Kumquat": ["Kumquats"],
    "Lemon": ["Lemon", "Lemon Meyer"],
    "Lime": ["Limes"],
    "Lychee": ["Lychee"],
    "Mandarin": ["Mandarine"],
    "Mango": ["Mango", "Mango Red"],
    "Mangosteen": ["Mangostan"],
    "Melon": ["Melon Piel de Sapo"],
    "Mulberry": ["Mulberry"],
    "Nectarine": ["Nectarine", "Nectarine Flat"],
    "Onion": ["Onion Red", "Onion Red Peeled", "Onion White"],
    "Papaya": ["Papaya"],
    "Passion Fruit": ["Passion Fruit", "Maracuja", "Granadilla"],
    "Peach": ["Peach", "Peach 2", "Peach Flat"],
    "Pear": ["Pear", "Pear 2", "Pear Abate", "Pear Forelle", "Pear Kaiser",
             "Pear Monster", "Pear Red", "Pear Stone", "Pear Williams"],
    "Pecan": ["Nut Pecan"],
    "Bell Pepper": ["Pepper Green", "Pepper Orange", "Pepper Red", "Pepper Yellow"],
    "Persimmon": ["Kaki"],
    "Cape Gooseberry": ["Physalis", "Physalis with Husk"],
    "Pineapple": ["Pineapple", "Pineapple Mini"],
    "Dragon Fruit": ["Pitahaya Red"],
    "Plum": ["Plum", "Plum 2", "Plum 3"],
    "Pomegranate": ["Pomegranate"],
    "Pomelo": ["Pomelo Sweetie"],
    "Potato": ["Potato Red", "Potato Red Washed", "Potato Sweet", "Potato White"],
    "Quince": ["Quince"],
    "Rambutan": ["Rambutan"],
    "Raspberry": ["Raspberry"],
    "Redcurrant": ["Redcurrant"],
    "Salak": ["Salak"],
    "Strawberry": ["Strawberry", "Strawberry Wedge"],
    "Tamarillo": ["Tamarillo"],
    "Tangelo": ["Tangelo"],
    "Tomato": ["Tomato 1", "Tomato 2", "Tomato 3", "Tomato 4", "Tomato Cherry Red",
               "Tomato Heart", "Tomato Maroon", "Tomato Yellow", "Tomato not Ripened"],
    "Walnut": ["Walnut"],
    "Watermelon": ["Watermelon"],
}

# canonical ingredient name -> regex patterns that count as a match
CATEGORY_KEYWORDS = {
    "Apple": [r"\bapples?\b"], "Apricot": [r"\bapricots?\b"],
    "Avocado": [r"\bavocado(?:s|es)?\b"], "Banana": [r"\bbananas?\b"],
    "Beet": [r"\bbeets?\b", r"\bbeetroots?\b"], "Blueberry": [r"\bblueberr(?:y|ies)\b"],
    "Cactus Fruit": [r"\bcactus fruit\b", r"\bprickly pears?\b"],
    "Cantaloupe": [r"\bcantaloupes?\b"], "Starfruit": [r"\bstar ?fruit\b"],
    "Cauliflower": [r"\bcauliflower\b"],
    "Cherry": [r"\bcherr(?:y|ies)\b"], "Chestnut": [r"\bchestnuts?\b"],
    "Clementine": [r"\bclementines?\b"], "Coconut": [r"\bcoconuts?\b"],
    "Corn": [r"\bcorn\b"], "Cucumber": [r"\bcucumbers?\b"],
    "Date": [r"\bdates?\b"], "Eggplant": [r"\beggplants?\b", r"\baubergines?\b"],
    "Fig": [r"\bfigs?\b"], "Ginger": [r"\bginger\b"],
    "Grape": [r"\bgrapes?\b"], "Grapefruit": [r"\bgrapefruits?\b"],
    "Guava": [r"\bguavas?\b"], "Hazelnut": [r"\bhazelnuts?\b", r"\bfilberts?\b"],
    "Huckleberry": [r"\bhuckleberr(?:y|ies)\b"], "Kiwi": [r"\bkiwis?\b", r"\bkiwifruit\b"],
    "Kohlrabi": [r"\bkohlrabi\b"], "Kumquat": [r"\bkumquats?\b"],
    "Lemon": [r"\blemons?\b"], "Lime": [r"\blimes?\b"],
    "Lychee": [r"\blychees?\b", r"\blitchis?\b"], "Mandarin": [r"\bmandarins?\b", r"\btangerines?\b"],
    "Mango": [r"\bmangoe?s?\b"], "Mangosteen": [r"\bmangosteens?\b"],
    "Melon": [r"\bhoneydew\b", r"\bmelons?\b"], "Mulberry": [r"\bmulberr(?:y|ies)\b"],
    "Nectarine": [r"\bnectarines?\b"], "Onion": [r"\bonions?\b"],
    "Papaya": [r"\bpapayas?\b"], "Passion Fruit": [r"\bpassion ?fruit\b", r"\bmaracuja\b", r"\bgranadilla\b"],
    "Peach": [r"\bpeaches?\b"], "Pear": [r"\bpears?\b"], "Pecan": [r"\bpecans?\b"],
    "Bell Pepper": [r"\bbell peppers?\b"],
    "Persimmon": [r"\bpersimmons?\b"],
    "Cape Gooseberry": [r"\bcape gooseberr(?:y|ies)\b", r"\bgroundcherr(?:y|ies)\b", r"\bphysalis\b"],
    "Pineapple": [r"\bpineapples?\b"], "Dragon Fruit": [r"\bdragon ?fruit\b", r"\bpitaya\b", r"\bpitahaya\b"],
    "Plum": [r"\bplums?\b"], "Pomegranate": [r"\bpomegranates?\b"],
    "Pomelo": [r"\bpomelos?\b", r"\bpummelo\b"], "Potato": [r"\bpotato(?:es)?\b"],
    "Quince": [r"\bquinces?\b"], "Rambutan": [r"\brambutans?\b"],
    "Raspberry": [r"\braspberr(?:y|ies)\b"], "Redcurrant": [r"\bred ?currants?\b"],
    "Salak": [r"\bsalak\b", r"\bsnake fruit\b"],
    "Strawberry": [r"\bstrawberr(?:y|ies)\b"], "Tamarillo": [r"\btamarillos?\b"],
    "Tangelo": [r"\btangelos?\b"], "Tomato": [r"\btomato(?:es)?\b"],
    "Walnut": [r"\bwalnuts?\b"], "Watermelon": [r"\bwatermelons?\b"],
}

# substrings that disqualify an otherwise-matching ingredient (avoids e.g.
# "black pepper" -> Bell Pepper, "cherry tomato" -> Cherry)
CATEGORY_EXCLUDES = {
    "Corn": ["corn starch", "cornstarch", "corn syrup", "corn tortilla", "corn flake", "corn meal", "cornmeal"],
    "Cherry": ["cherry tomato"],
    "Plum": ["plum tomato"],
    "Grape": ["grape tomato"],
    "Date": ["to date", "date night", "expiration date"],
}

COMPILED_KEYWORDS = {
    cat: [re.compile(p, re.IGNORECASE) for p in patterns]
    for cat, patterns in CATEGORY_KEYWORDS.items()
}


def parse_ingredients(raw: str) -> list[str]:
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except (ValueError, SyntaxError):
        pass
    return [p.strip() for p in re.split(r",\s*", str(raw)) if p.strip()]


def match_category(ingredients: list[str]) -> str | None:
    for ingredient in ingredients:
        low = ingredient.lower()
        for cat, patterns in COMPILED_KEYWORDS.items():
            excludes = CATEGORY_EXCLUDES.get(cat, [])
            if any(ex in low for ex in excludes):
                continue
            if any(p.search(low) for p in patterns):
                return cat
    return None


def build_image_pools() -> dict[str, itertools.cycle]:
    pools = {}
    for category, folders in CATEGORY_FOLDERS.items():
        files: list[Path] = []
        for folder in folders:
            folder_path = FRUIT_TRAINING_DIR / folder
            if folder_path.is_dir():
                files.extend(sorted(folder_path.glob("*.jpg")))
        if not files:
            continue
        random.shuffle(files)
        pools[category] = itertools.cycle(files)
    return pools


def main():
    print(f"Loading recipes from {RECIPES_CSV}")
    df = pd.read_csv(RECIPES_CSV, index_col=0)
    df = df.dropna(subset=["Title", "Ingredients", "Instructions"])
    print(f"Loaded {len(df)} raw recipes")

    pools = build_image_pools()
    print(f"Built image pools for {len(pools)} ingredient categories "
          f"from {FRUIT_TRAINING_DIR}")

    IMAGES_OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for recipe_id, row in df.iterrows():
        ingredients = parse_ingredients(row["Ingredients"])
        category = match_category(ingredients)
        if category is None or category not in pools:
            continue
        src_image = next(pools[category])
        dest_name = f"{recipe_id}.jpg"
        shutil.copyfile(src_image, IMAGES_OUT / dest_name)
        records.append({
            "id": int(recipe_id),
            "title": str(row["Title"]).strip(),
            "ingredients": ingredients,
            "ingredients_text": "; ".join(ingredients),
            "instructions": str(row["Instructions"]).strip(),
            "category": category,
            "image_path": f"recipe_images/{dest_name}",
        })

    out_df = pd.DataFrame.from_records(records)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(PROCESSED_DIR / "recipes.parquet", index=False)

    print(f"Matched {len(out_df)} / {len(df)} recipes to an ingredient photo "
          f"({len(out_df) / len(df):.1%})")
    print(out_df["category"].value_counts().head(20))
    print(f"Saved corpus to {PROCESSED_DIR / 'recipes.parquet'}")


if __name__ == "__main__":
    main()
