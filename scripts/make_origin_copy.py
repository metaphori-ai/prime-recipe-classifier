#!/usr/bin/env python3
"""Origin-only data copy: data/dataset_v04 -> data/dataset_v04_origin, with ONLY the origin labels.

Nothing in dataset_v04 (or any earlier dataset, gold or silver file) is modified: this script opens them read-only and
checks their SHA-256 before and after. Same recipes, same ids, same prompts, same origin labels, same order; the
`contains` / `contains_uncertain` labels are simply not written to the copy, so Phase 5 cannot train on them by accident.

  python3 scripts/make_origin_copy.py
  cd env && python3 -m recipe_tax.sft_prepare --mode origin --dataset-dir ../data/dataset_v04_origin --out-dir ../data/sft_v04_origin
"""
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC, DST = ROOT / "data" / "dataset_v04", ROOT / "data" / "dataset_v04_origin"
FILES = ["train.jsonl", "eval.jsonl", "eval_gold_hard.jsonl"]

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    files = [f for f in FILES if (SRC / f).exists()]      # the public repo has no eval_gold_hard.jsonl (private)
    before = {f: sha(SRC / f) for f in files}
    DST.mkdir(exist_ok=True); counts = {}
    for f in files:
        n = 0
        with open(DST / f, "w", encoding="utf-8") as out:
            for l in open(SRC / f, encoding="utf-8"):
                if not l.strip(): continue
                r = json.loads(l)
                r["labels"] = {"origin": r["labels"]["origin"]}
                out.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
        counts[f] = n
    assert before == {f: sha(SRC / f) for f in files}, "source changed"
    json.dump({"copied_from": "data/dataset_v04", "mode": "origin", "labels_kept": ["origin"], "labels_dropped": ["contains", "contains_uncertain"],
               "counts": counts, "source_sha256": before}, open(DST / "manifest.json", "w"), indent=1)
    print(json.dumps(counts), "->", DST)

if __name__ == "__main__": main()
