#!/usr/bin/env python3
"""Survey data/raw/*.jsonl and write data/corpus_report.md (+ data/dedupe_manifest.json).

Usage:
    uv run --with pyyaml python -I scripts/corpus_report.py taxonomy.yaml data/raw data/corpus_report.md data/dedupe_manifest.json

- Dedupe key: normalized title + first 5 ingredients (lowercased, punctuation stripped). Raw files are
  NOT modified; the manifest lists which ids are kept and which are dropped as duplicates (first seen wins,
  in source priority order: saffronsifu_postgres, saffronsifu_paprika, wikibooks, foodcom).
- Origin guess is a SURVEY ONLY: taxonomy node names + countries + a few source-tag synonyms matched against
  title and existing_tags. It is not a label and must not be copied into gold or silver.
"""
import collections
import json
import os
import re
import statistics
import sys

import yaml

SOURCE_PRIORITY = ["saffronsifu_postgres", "saffronsifu_paprika", "wikibooks", "foodcom"]

# source-tag vocab that is not a taxonomy node name
SYNONYMS = {
    "szechuan": "Sichuan", "szechwan": "Sichuan", "tex mex": "Tex-Mex", "tex-mex": "Tex-Mex",
    "southwestern u.s.": "Southwestern", "southwestern": "Southwestern", "soul food": "Soul Food",
    "scandinavian": "Nordic", "middle eastern": "West Asian & Caucasus", "middle east": "West Asian & Caucasus",
    "latin american": "Latin American & Caribbean", "south american": "Latin American & Caribbean",
    "central american": "Central American", "african": "Sub-Saharan African", "european": "European",
    "asian": None, "native american": "American", "american": "American", "usa": "American",
    "nepalese": "Nepali", "lebanese": "Lebanese", "persian": "Persian", "iranian": "Persian",
    "jewish": "Ashkenazi Jewish", "ashkenazi": "Ashkenazi Jewish", "sephardic": "Sephardic Jewish",
    "cantonese": "Cantonese", "hunan": "Hunan", "quebec": "Québécois", "québécois": "Québécois",
    "hawaiian": "Hawaiian", "cajun": "Cajun", "creole": "Creole", "english": "English", "british": "British",
    "mexican": "Mexican", "indian": "Indian", "chinese": "Chinese", "italian": "Italian",
}


def load_taxonomy(path):
    with open(path, encoding="utf-8") as f:
        tax = yaml.safe_load(f)
    node_to_path = {}
    node_countries = {}

    def walk(node, path, inherited_c):
        if node is None:
            return
        if isinstance(node, dict):
            c = node.get("_c", inherited_c)
            for k, v in node.items():
                if k == "_c":
                    continue
                child_path = path + [k]
                if isinstance(v, dict) and set(v.keys()) == {"c"}:
                    node_to_path[k] = child_path
                    node_countries[k] = v["c"]
                else:
                    node_to_path[k] = child_path
                    node_countries[k] = (v or {}).get("_c", c) if isinstance(v, dict) else c
                    walk(v, child_path, node_countries[k])

    walk(tax["origin"], [], [])
    traps = tax.get("ingredient_traps", {})
    return node_to_path, node_countries, traps


def norm_text(s):
    return re.sub(r"[^a-z0-9 ]+", " ", str(s).lower())


def dedupe_key(rec):
    t = " ".join(norm_text(rec.get("title", "")).split())
    ings = [" ".join(norm_text(i).split()) for i in rec.get("ingredients", [])[:5]]
    return t + "||" + "|".join(ings)


def main(tax_path, raw_dir, out_md, out_manifest):
    node_to_path, node_countries, traps = load_taxonomy(tax_path)

    # match vocab: node names, country names, synonyms -> target node (or macro)
    vocab = {}
    for name in node_to_path:
        vocab[name.lower()] = name
    for name, cs in node_countries.items():
        for c in cs or []:
            vocab.setdefault(c.lower(), name)   # country -> first node carrying it (coarse; macro-level use only)
    for k, v in SYNONYMS.items():
        if v is not None:
            vocab[k] = v
    # longest-first so "south indian" wins over "indian"
    vocab_items = sorted(vocab.items(), key=lambda kv: -len(kv[0]))
    vocab_re = re.compile(r"\b(" + "|".join(re.escape(k) for k, _ in vocab_items) + r")\b")

    def macro_of(node):
        p = node_to_path.get(node)
        return p[0] if p else node   # synonyms may map straight to a macro name

    files = sorted(f for f in os.listdir(raw_dir) if f.endswith(".jsonl"))
    files.sort(key=lambda f: SOURCE_PRIORITY.index(f[:-6]) if f[:-6] in SOURCE_PRIORITY else 99)

    per_source = collections.OrderedDict()
    seen = {}
    kept, dropped = [], []
    lang = collections.Counter()
    ing_counts = []
    ing_counts_by_src = collections.defaultdict(list)
    trap_hits = collections.defaultdict(collections.Counter)   # class -> word -> n recipes
    trap_any = collections.Counter()
    macro_counts = collections.Counter()
    tradition_counts = collections.Counter()
    regional_counts = collections.Counter()
    unmatched = 0
    total = 0

    for fn in files:
        src = fn[:-6]
        n = 0
        with open(os.path.join(raw_dir, fn), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                n += 1
                total += 1
                key = dedupe_key(rec)
                if key in seen:
                    dropped.append({"id": rec["id"], "duplicate_of": seen[key], "source_file": fn})
                    continue
                seen[key] = rec["id"]
                kept.append(rec["id"])

                lang[(rec.get("lang") or "").split("-")[0].lower() or "unknown"] += 1
                k = len(rec.get("ingredients", []))
                ing_counts.append(k)
                ing_counts_by_src[src].append(k)

                ing_text = " | ".join(rec.get("ingredients", [])).lower()
                hit_any = False
                for cls, words in traps.items():
                    for w in words:
                        w_clean = re.sub(r"\s*\(.*?\)", "", w).strip().lower()
                        if w_clean.startswith("some ") or w_clean.startswith("many "):
                            continue
                        if w_clean in ing_text:
                            trap_hits[cls][w_clean] += 1
                            hit_any = True
                if hit_any:
                    trap_any[src] += 1

                text = norm_text(rec.get("title", "")) + " " + " ".join(norm_text(t) for t in rec.get("existing_tags", []))
                found = set(m.group(1) for m in vocab_re.finditer(text))
                nodes = set(vocab[f] for f in found)
                if not nodes:
                    unmatched += 1
                macros = set()
                for node in nodes:
                    p = node_to_path.get(node)
                    if p is None:        # synonym mapped to a macro name
                        macros.add(node)
                        continue
                    macros.add(p[0])
                    if len(p) >= 2:
                        tradition_counts[" > ".join(p[:2])] += 1
                    if len(p) >= 3:
                        regional_counts[" > ".join(p)] += 1
                for m in macros:
                    macro_counts[m] += 1
        per_source[src] = n

    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump({"kept": len(kept), "dropped": dropped}, f, indent=0)

    def med(xs):
        return statistics.median(xs) if xs else 0

    L = []
    L.append("# Corpus report — data/raw (survey, no labels)\n")
    L.append(f"Generated by `scripts/corpus_report.py`. Raw files are untouched; duplicates are listed in `data/dedupe_manifest.json`.\n")
    L.append("## Totals\n")
    L.append("| | count |\n|---|---|")
    L.append(f"| raw records | {total} |")
    L.append(f"| after dedupe (title + first 5 ingredients) | {len(kept)} |")
    L.append(f"| duplicates dropped | {len(dropped)} |\n")
    L.append("## Per source\n")
    L.append("| source | raw rows | median ingredients | recipes with ≥1 trap word |\n|---|---|---|---|")
    for src, n in per_source.items():
        L.append(f"| {src} | {n} | {med(ing_counts_by_src[src])} | {trap_any[src]} |")
    L.append(f"| **all** | {total} | {med(ing_counts)} | {sum(trap_any.values())} |\n")
    L.append("## Duplicates by source file\n")
    dsrc = collections.Counter(d["source_file"] for d in dropped)
    L.append("| file | dropped |\n|---|---|")
    for k, v in dsrc.most_common():
        L.append(f"| {k} | {v} |")
    L.append("")
    L.append("## Language (primary subtag)\n")
    L.append("| lang | count |\n|---|---|")
    for k, v in lang.most_common():
        L.append(f"| {k} | {v} |")
    L.append("")
    L.append("## Ingredient-trap keyword hits (recipes whose ingredient lines contain the word)\n")
    L.append("Source: `taxonomy.yaml → ingredient_traps`. Parenthetical qualifiers and 'some …' entries skipped.\n")
    L.append("| class | word | recipes |\n|---|---|---|")
    for cls in traps:
        for w, n in trap_hits[cls].most_common():
            L.append(f"| {cls} | {w} | {n} |")
    L.append("")
    L.append("## Rough origin distribution (title + existing_tags matched against taxonomy node/country names)\n")
    L.append("Survey only. A recipe can count toward several macros; 'unmatched' had no recognizable origin word.\n")
    L.append("| macro group | recipes |\n|---|---|")
    for k, v in macro_counts.most_common():
        L.append(f"| {k} | {v} |")
    L.append(f"| *(unmatched)* | {unmatched} |\n")
    L.append("### Traditions (second tier), top 50\n")
    L.append("| tradition | recipes |\n|---|---|")
    for k, v in tradition_counts.most_common(50):
        L.append(f"| {k} | {v} |")
    L.append("")
    L.append("### Regional nodes (third tier and deeper) with any hits\n")
    L.append("These are the nodes the corpus can actually support. Nodes absent here have no title/tag evidence yet.\n")
    L.append("| node path | recipes |\n|---|---|")
    for k, v in regional_counts.most_common():
        L.append(f"| {k} | {v} |")
    L.append("")
    L.append("## Notes on sources\n")
    L.append("- `saffronsifu_postgres`, `saffronsifu_paprika`: owner's own captures (SaffronSifu cache / Paprika library). Raw ingredient lines, quantities intact.")
    L.append("- `wikibooks`: Wikibooks Cookbook, CC-BY-SA 4.0 (HF `gossminn/wikibooks-cookbook`, scraped 2024-07-31). Raw ingredient lines. Skews Anglo + strong Nigerian section.")
    L.append("- `foodcom`: Food.com via Kaggle `irkaal/foodcom-recipes-and-reviews`, CC0-1.0 (snapshot 2020-12-28). **Ingredients are normalized parts without quantities or units** (the Kaggle dump's quantity array is misaligned in ~75% of rows, so lines were not reconstructed). Trap wording survives; amounts do not. Hard-case skewed selection; reasons in `data/_staging/foodcom/selection_reasons.json`.")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {out_md} and {out_manifest}")
    print(f"total {total}, kept {len(kept)}, dropped {len(dropped)}")
    print("macros:", dict(macro_counts.most_common()))


if __name__ == "__main__":
    main(*sys.argv[1:5])
