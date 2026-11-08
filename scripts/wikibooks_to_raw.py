#!/usr/bin/env python3
"""Convert the Wikibooks Cookbook dump (gossminn/wikibooks-cookbook, CC-BY-SA 4.0)
into the data/raw JSONL schema defined in CLAUDE.md.

Usage:
    python3 -I scripts/wikibooks_to_raw.py data/_staging/wikibooks/recipes_parsed.json data/raw/wikibooks.jsonl

Ingredients are kept as the original list-item strings. No quantity parsing.
"""
import collections
import json
import re
import sys

SCRAPE_DATE = "2024-07-31T00:00:00Z"  # dataset README: scraped 2024-07-31
ING_SECTION = re.compile(r"ingredient", re.I)
PROC_SECTION = re.compile(r"procedure|method|direction|preparation|instruction|steps", re.I)
SKIP_SECTIONS = re.compile(r"notes|tips|variations|warnings|see also|references|external links|gallery", re.I)


def category_from_href(href):
    if not href:
        return None
    m = re.search(r"Category:([^&?#]+)", href)
    if not m:
        return None
    return m.group(1).replace("_", " ").strip()


def slug_from_url(url, title):
    m = re.search(r"/wiki/Cookbook:(.+)$", url or "")
    s = m.group(1) if m else title
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower()
    return s[:120]


def convert(src_path, out_path):
    with open(src_path, encoding="utf-8") as f:
        data = json.load(f)

    section_names = collections.Counter()
    skipped = collections.Counter()
    seen_ids = set()
    out = []

    for rec in data:
        rd = rec.get("recipe_data") or {}
        title = (rd.get("title") or "").strip()
        url = rd.get("url") or ""
        lines = rd.get("text_lines") or []
        for ln in lines:
            section_names[ln.get("section")] += 1

        ingredients = []
        steps = []
        for ln in lines:
            sec = ln.get("section") or ""
            txt = (ln.get("text") or "").strip()
            if not txt:
                continue
            if ING_SECTION.search(sec) and ln.get("line_type") in ("ul", "ol"):
                ingredients.append(txt)
            elif PROC_SECTION.search(sec) and not SKIP_SECTIONS.search(sec):
                steps.append(txt)

        if not title:
            skipped["no_title"] += 1
            continue
        if not ingredients:
            skipped["no_ingredients"] += 1
            continue
        if not steps:
            skipped["no_instructions"] += 1
            continue

        rid = "wikibooks-" + slug_from_url(url, title)
        if rid in seen_ids:
            skipped["duplicate_id"] += 1
            continue
        seen_ids.add(rid)

        info = rd.get("infobox") or {}
        tags = []
        cat = category_from_href(info.get("category"))
        if cat:
            tags.append(cat)
        if info.get("difficulty") is not None:
            tags.append(f"difficulty:{info['difficulty']}")

        out.append({
            "id": rid,
            "source": url,
            "title": title,
            "ingredients": ingredients,
            "instructions": "\n".join(steps),
            "existing_tags": tags,
            "lang": "en",
            "fetched_at": SCRAPE_DATE,
        })

    with open(out_path, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"input records:  {len(data)}")
    print(f"written:        {len(out)} -> {out_path}")
    print(f"skipped:        {dict(skipped)}")
    print("top sections:")
    for name, n in section_names.most_common(25):
        print(f"  {n:6d}  {name}")
    cats = collections.Counter(t for r in out for t in r["existing_tags"] if not t.startswith("difficulty:"))
    print(f"distinct categories: {len(cats)}")
    for name, n in cats.most_common(40):
        print(f"  {n:5d}  {name}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    convert(sys.argv[1], sys.argv[2])
