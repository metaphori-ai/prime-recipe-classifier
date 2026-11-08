"""Build the environment dataset from raw + gold + silver.

  uv run --with pyyaml python -m recipe_tax.build_dataset            # writes data/dataset/{eval,train}.jsonl + manifest

Rules (see CLAUDE.md / TAXONOMY.md):
  - eval  = gold.jsonl (human-verified). NEVER trained on.
  - train = silver: contains from Jev (contains_jev_full.jsonl), origin from origin_silver.jsonl when present.
            Records with a gold id, or a content-duplicate of a gold record, are EXCLUDED (leakage check).
  - prompt shows ingredient NAMES ONLY (quantities stripped) for every source -> no format shortcut
    (Food.com is names-only and hard-skewed; Wikibooks has quantities and is ordinary).
  - existing_tags are never in the prompt (label leakage).
  - publishable flag by source: Food.com (CC0) + Wikibooks (CC-BY-SA) yes; SaffronSifu scraped captures no.
"""
from __future__ import annotations
import json, re, hashlib, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"; GOLD = ROOT / "data" / "gold" / "gold.jsonl"
JEV_FULL = ROOT / "data" / "silver" / "contains_jev_full.jsonl"
JEV_CAND = ROOT / "data" / "silver" / "contains_jev_candidates.jsonl"
ORIGIN_SILVER = ROOT / "data" / "silver" / "origin_silver.jsonl"             # single-teacher file (used if present)
ORIGIN_SILVER_GLOB = "origin_silver_*.jsonl"                                  # multi-teacher files -> agreement filter
OUT = ROOT / "data" / "dataset"

PUBLISHABLE = {"foodcom": True, "wikibooks": True, "ss": False}

_QTY = re.compile(r"""^\s*(?:(?:about|approx\.?|approximately|heaping|scant|generous)\s+)?
    (?:\d+[\d\s/\.,\-–½⅓⅔¼¾⅛⅜⅝⅞]*|[½⅓⅔¼¾⅛⅜⅝⅞]+)\s*
    (?:(?:x|×)\s*)?(?:\(.*?\)\s*)?
    (?:(?:fl\.?\s*)?(?:cups?|c\.|tbsps?|tbs|tb|tablespoons?|tsps?|teaspoons?|oz\.?|ounces?|lbs?\.?|pounds?|g|grams?|kg|kilograms?|ml|millilit(?:er|re)s?|l|lit(?:er|re)s?|
        cans?|jars?|packages?|pkgs?|packets?|sticks?|cloves?|slices?|pieces?|pinch(?:es)?|dash(?:es)?|bunch(?:es)?|sprigs?|stalks?|heads?|ears?|sheets?|drops?|quarts?|qts?|pints?|pts?|gallons?|handfuls?|
        large|medium|small|whole|fresh|dried|cubes?|inch(?:es)?|cm|mm)\.?\s+(?:of\s+)?)*""", re.X | re.I)

def ingredient_name(line: str) -> str:
    s = line.strip()
    if not s or s.endswith(":"): return ""
    s = re.sub(r"\([^)]*\)", " ", s)                 # drop parentheticals
    s = re.sub(r"\*+|\bNote \d+\b", " ", s)
    s = _QTY.sub("", s, count=1)
    s = re.sub(r"^(?:plus|and|or)\s+", "", s, flags=re.I)
    s = _QTY.sub("", s, count=1)
    s = re.split(r",|;| – | - |—", s)[0]              # drop prep notes after comma/dash
    s = re.sub(r"\s+", " ", s).strip(" .-–")
    return s.lower()

def ingredient_names(ings: list[str]) -> list[str]:
    out = []
    for l in ings:
        n = ingredient_name(l)
        if n and n not in out: out.append(n)
    return out

def content_key(rec: dict) -> str:
    t = re.sub(r"[^a-z0-9]+", " ", (rec.get("title") or "").lower()).strip()
    first = "|".join(ingredient_names(rec.get("ingredients", []))[:5])
    return hashlib.sha1(f"{t}||{first}".encode()).hexdigest()[:16]

def source_key(rec_id: str) -> str:
    return "ss" if rec_id.startswith("ss-") else rec_id.split("-")[0]

def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []

def _primary(origin):
    return next((tuple(x["path"]) for x in origin if x.get("role", "primary") == "primary"), None)

def load_origin_silver(teacher: str | None = None) -> tuple[dict, dict]:
    """One teacher file -> use it. Several `origin_silver_*.jsonl` -> keep a recipe only when ALL teachers agree on the primary
    path (measured on gold: agreement subset 82.9% exact vs ~68% for any single teacher); the first file's labels are used.
    `teacher="dsv4"` -> use origin_silver_dsv4.jsonl alone, NO agreement gate. Agreement raises exact-match on the kept rows but
    deletes most influence tags (each teacher finds ~37% of gold influences; they rarely agree on the same ones): consensus
    silver carries 0.006 influences/recipe vs 0.073 in gold; single-teacher ≈0.03."""
    if teacher:
        f = ROOT / "data" / "silver" / f"origin_silver_{teacher}.jsonl"
        if not f.exists(): raise SystemExit(f"no such teacher file: {f}")
        rows = {r["id"]: r for r in read_jsonl(f) if "labels" in r}
        return rows, {"teachers": 1, "files": [f.name], "agreement_gate": False}
    files = sorted((ROOT / "data" / "silver").glob(ORIGIN_SILVER_GLOB))
    if not files and ORIGIN_SILVER.exists(): files = [ORIGIN_SILVER]
    if not files: return {}, {"teachers": 0}
    per = [{r["id"]: r for r in read_jsonl(f) if "labels" in r} for f in files]
    if len(per) == 1: return per[0], {"teachers": 1, "files": [f.name for f in files]}
    common = set.intersection(*(set(p) for p in per))
    agreed = {i: per[0][i] for i in common if len({_primary(p[i]["labels"]["origin"]) for p in per}) == 1}
    for i in agreed: agreed[i] = {**agreed[i], "labeled_by": "teachers-agree:" + "+".join(p[i].get("model", "?").split("/")[-1] for p in per)}
    return agreed, {"teachers": len(per), "files": [f.name for f in files], "common": len(common), "agreed": len(agreed),
                    "agreement_rate": round(len(agreed) / max(len(common), 1), 3)}

def build_prompt(rec: dict) -> str:
    names = ingredient_names(rec.get("ingredients", []))
    return f"Title: {rec.get('title','')}\nIngredients: " + "; ".join(names)

def cap_by_primary(rows: list[dict], cap: int, seed: int = 7) -> tuple[list[dict], dict]:
    """Keep at most `cap` train records per primary path (deterministic shuffle). Prevents a majority label
    (American ≈ 28 % of silver) from giving the policy a free partial-credit floor."""
    import random
    rnd = random.Random(seed); rows = rows[:]; rnd.shuffle(rows)
    seen = collections.Counter(); kept = []; dropped = collections.Counter()
    for r in rows:
        k = "/".join(_primary(r["labels"]["origin"]) or ())
        if seen[k] < cap: kept.append(r); seen[k] += 1
        else: dropped[k] += 1
    return kept, dict(dropped.most_common(5))

def main(argv=None) -> int:
    global OUT
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--cap-primary", type=int, default=0, help="max train records per primary path (0 = no cap)")
    ap.add_argument("--teacher", default=None, help="use origin_silver_<teacher>.jsonl alone, no agreement gate (e.g. dsv4)")
    ap.add_argument("--out-dir", default=None, help="write dataset to this dir instead of data/dataset (keeps the frozen set intact)")
    a = ap.parse_args(argv)
    if a.out_dir: OUT = Path(a.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    raw = {}
    for f in sorted(RAW.glob("*.jsonl")):
        for r in read_jsonl(f): raw[r["id"]] = r
    gold = {g["id"]: g for g in read_jsonl(GOLD)}
    jev = {j["id"]: j for j in read_jsonl(JEV_FULL)} or {j["id"]: j for j in read_jsonl(JEV_CAND)}
    osilver, teacher_note = load_origin_silver(a.teacher)
    gold_keys = {content_key(raw[i]) for i in gold if i in raw}

    def record(rid: str, labels: dict, split: str, labeled_by: str) -> dict:
        r = raw[rid]
        return {"id": rid, "split": split, "source": r.get("source"), "publishable": PUBLISHABLE.get(source_key(rid), False),
                "prompt": build_prompt(r), "ingredients": [l for l in r.get("ingredients", []) if l and not l.rstrip().endswith(":")],
                "labels": {"origin": labels["origin"], "contains": labels["contains"], "contains_uncertain": labels.get("contains_uncertain", [])},
                "labeled_by": labeled_by}

    ev = [record(i, g["labels"], "eval", "human") for i, g in gold.items() if i in raw]
    tr = []; skipped = collections.Counter()
    for rid, r in raw.items():
        if rid in gold: skipped["gold id"] += 1; continue
        if content_key(r) in gold_keys: skipped["content-duplicate of gold"] += 1; continue
        j = jev.get(rid)
        if not j: skipped["no jev"] += 1; continue
        if "incomplete" in j["labels"]["contains"]: skipped["jev: incomplete"] += 1; continue
        o = osilver.get(rid)
        if not o: skipped["no origin silver"] += 1; continue
        labels = {"origin": o["labels"]["origin"], "contains": j["labels"]["contains"], "contains_uncertain": j["labels"].get("contains_uncertain", [])}
        tr.append(record(rid, labels, "train", f"{j.get('labeled_by','jev')} + {o.get('labeled_by','origin-silver')}"))

    capped = {}
    if a.cap_primary: tr, capped = cap_by_primary(tr, a.cap_primary)
    for name, rows in (("eval", ev), ("train", tr)):
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    man = {"eval": len(ev), "train": len(tr), "skipped": dict(skipped), "origin_silver": teacher_note, "cap_primary": a.cap_primary, "capped_dropped": capped, "publishable_train": sum(r["publishable"] for r in tr),
           "publishable_eval": sum(r["publishable"] for r in ev), "sources": dict(collections.Counter(source_key(r["id"]) for r in ev + tr))}
    json.dump(man, open(OUT / "manifest.json", "w"), indent=1)
    print(json.dumps(man, indent=1))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
