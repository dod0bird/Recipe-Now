# Custom dish dataset

This is the real "photo of a finished dish -> recipe" dataset for Recipe
Now, hand-curated instead of scraped. Start with ~10 dishes to prove the
idea works, then add more the same way.

## What to put here

```
data/custom_dishes/
  recipes.csv       <- one row per dish (this file)
  images/
    your_dish.jpg   <- a real photo of the finished, plated dish
```

## `recipes.csv` columns

| column           | required | description                                                              |
|------------------|----------|---------------------------------------------------------------------------|
| `title`          | yes      | dish name, e.g. `Margherita Pizza`                                        |
| `image_filename` | yes      | exact filename inside `images/`, e.g. `margherita_pizza.jpg`              |
| `ingredients`     | yes      | semicolon-separated, e.g. `pizza dough; tomato sauce; fresh mozzarella`   |
| `instructions`   | yes      | plain-text steps                                                          |
| `category`       | no       | short tag used to label groups on the cluster map; defaults to the title |

See `recipes.csv` in this folder for a filled-in example — replace its rows
with your own dishes.

## Photo guidelines

- A real photo of the finished, plated dish (yours, or one you have the
  rights to use) — not a stock ingredient photo.
- JPG or PNG, any resolution.
- One photo per dish to start. You can add more photos of the same dish
  later as extra rows (same `title`/`ingredients`/`instructions`, different
  `image_filename`) — more real photos per dish generally makes matching
  more robust.

## After adding your dishes

From the project root:

```bash
python scripts/01_build_corpus.py
python scripts/02_embed_and_index.py
python scripts/03_cluster.py
uvicorn backend.main:app --reload
```
