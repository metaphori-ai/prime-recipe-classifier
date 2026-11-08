#!/usr/bin/env python3
"""Survey the Food.com recipes parquet (irkaal/foodcom-recipes-and-reviews, CC0-1.0):
columns, a sample row, RecipeCategory distribution, cuisine-like Keywords, and trap-ingredient hits.

Usage:
    uv run --with pandas --with pyarrow python -I scripts/foodcom_inspect.py data/_staging/foodcom/recipes.parquet
"""
import collections
import sys

import pandas as pd

TRAPS = {
    "soy sauce": "gluten", "fish sauce": "fish", "dashi": "fish", "oyster sauce": "fish",
    "worcestershire": "fish/gluten", "anchov": "fish", "gelatin": "gelatin", "ghee": "dairy",
    "paneer": "dairy", "mirin": "alcohol", "shaoxing": "alcohol", "miso": "soy", "tempeh": "soy",
    "tamari": "soy", "pesto": "nuts", "mole": "nuts", "romesco": "nuts", "satay": "nuts/peanut",
    "marzipan": "nuts", "panko": "gluten", "seitan": "gluten", "couscous": "gluten",
    "mayonnaise": "egg", "hollandaise": "egg", "gochujang": "fusion/soy", "kimchi": "fish?",
}

CUISINE_WORDS = [
    "indian", "pakistani", "punjabi", "bengali", "sri lankan", "nepal", "afghan",
    "chinese", "cantonese", "szechuan", "sichuan", "hunan", "japanese", "korean",
    "thai", "vietnamese", "indonesian", "malaysian", "filipino",
    "lebanese", "turkish", "persian", "iranian", "israeli", "moroccan", "egyptian",
    "ethiopian", "nigerian", "african", "south african",
    "italian", "french", "spanish", "basque", "portuguese", "greek", "german", "polish", "hungarian", "russian",
    "scandinavian", "swedish", "norwegian", "danish", "finnish",
    "mexican", "tex mex", "tex-mex", "southwestern", "cajun", "creole", "southern", "soul food", "hawaiian",
    "caribbean", "jamaican", "cuban", "puerto rican", "peruvian", "brazilian", "argentin", "colombian", "chilean",
    "canadian", "australian", "new zealand", "polynesian",
]


def main(path):
    if path.endswith(".zip"):
        import os
        import zipfile
        outdir = os.path.dirname(path)
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.endswith(".parquet")]
            z.extractall(outdir, members=names)
            path = os.path.join(outdir, names[0])
        print("extracted:", path)
    df = pd.read_parquet(path)
    print("rows:", len(df))
    print("columns:", list(df.columns))
    print()
    r = df.iloc[0]
    for c in df.columns:
        v = r[c]
        s = str(v)
        print(f"  {c:28s} {s[:160]}")
    print()

    cat = df["RecipeCategory"].fillna("").value_counts()
    print("RecipeCategory top 60:")
    for k, v in cat.head(60).items():
        print(f"  {v:7d}  {k}")
    print()

    kw = collections.Counter()
    for arr in df["Keywords"].dropna():
        for k in arr:
            kw[str(k).lower()] += 1
    print("Keywords matching cuisine words:")
    for w in CUISINE_WORDS:
        hits = {k: v for k, v in kw.items() if w in k}
        if hits:
            total = sum(hits.values())
            print(f"  {total:7d}  {w:16s} {dict(sorted(hits.items(), key=lambda x: -x[1])[:5])}")
    print()

    # trap hits over ingredient parts + instructions (lowercased)
    def joined(row):
        parts = row["RecipeIngredientParts"]
        parts = " | ".join(str(p) for p in parts) if parts is not None else ""
        return parts.lower()

    ing_text = df.apply(joined, axis=1)
    print("Trap-word hits in RecipeIngredientParts:")
    for w, cls in TRAPS.items():
        n = int(ing_text.str.contains(w, regex=False).sum())
        print(f"  {n:7d}  {w:16s} ({cls})")
    print()

    # how are quantities/parts aligned?
    n_mismatch = 0
    n_checked = 0
    for _, row in df.head(2000).iterrows():
        q = row["RecipeIngredientQuantities"]
        p = row["RecipeIngredientParts"]
        if q is None or p is None:
            continue
        n_checked += 1
        if len(q) != len(p):
            n_mismatch += 1
    print(f"quantities/parts length mismatch in first 2000: {n_mismatch}/{n_checked}")
    print()
    print("sample reconstructed ingredient lines:")
    row = df.iloc[5]
    for q, p in zip(row["RecipeIngredientQuantities"] or [], row["RecipeIngredientParts"] or []):
        print(f"   [{q}] {p}")


if __name__ == "__main__":
    main(sys.argv[1])
