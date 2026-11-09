#!/usr/bin/env python3
"""Assemble the public repo (public-repo/) from the working project. Idempotent; re-run after every milestone.

Rules:
  - Code, configs, docs, diagrams, tests, scripts: copied as-is (no .venv, no outputs, no *.env, no runs/).
  - Data: ONLY records whose `publishable` flag is true (Food.com = CC0; Wikibooks = CC BY-SA 3.0, attributed via `source` URL).
    SaffronSifu captures (ids ss-*) are never exported. Silver/teacher files are filtered to publishable ids.
  - Secrets: refuses to run if any staged text file matches common key patterns.
"""
from __future__ import annotations
import json, re, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public-repo"

COPY_FILES = ["taxonomy.yaml", "LOG.md", "TRAINING.md", "TAXONOMY.md", "OVERVIEW.md", "PLAN.md", "journey.svg", "journey-2.svg", "process-flow.svg"]
COPY_TREES = {"env/recipe_tax": "env/recipe_tax", "env/tests": "env/tests",
              "env/environments/recipe_classify": "env/environments/recipe_classify", "scripts": "scripts"}
EXCLUDE_DIRS = {".venv", "outputs", "__pycache__", "*.egg-info", ".pytest_cache"}
EXCLUDE_FILES = {"ollama.env", "mlx.env", ".env"}
DATA_FILES = {  # src -> dst, filtered to publishable ids
    "data/dataset/eval.jsonl": "data/dataset/eval.jsonl",
    "data/dataset/train.jsonl": "data/dataset/train.jsonl",
    "data/dataset/manifest.json": "data/dataset/manifest.json",
    "data/dataset_dsv4/train.jsonl": "data/dataset_dsv4/train.jsonl",
    "data/dataset_sonnet/train.jsonl": "data/dataset_sonnet/train.jsonl",
    "data/gold/gold.jsonl": "data/gold/gold.jsonl",
    "data/silver/origin_eval_dsv4.jsonl": "data/silver/origin_eval_dsv4.jsonl",
    "data/silver/origin_eval_qwen36.jsonl": "data/silver/origin_eval_qwen36.jsonl",
    "data/silver/origin_eval_sonnet55.jsonl": "data/silver/origin_eval_sonnet55.jsonl",
    "data/silver/origin_eval_qwen36_27b_nothink.jsonl": "data/silver/origin_eval_qwen36_27b_nothink.jsonl",
}
RAW_PUBLIC = {"data/raw/foodcom.jsonl": "data/raw/foodcom.jsonl", "data/raw/wikibooks.jsonl": "data/raw/wikibooks.jsonl"}
SECRET_RE = re.compile(r"(sk-ant-[A-Za-z0-9_-]{10,}|hf_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|api[_-]?key\s*[:=]\s*['\"][A-Za-z0-9_-]{16,}|Bearer [A-Za-z0-9._-]{20,})", re.I)

def publishable_ids() -> set[str]:
    ids = set()
    for f in ("data/dataset/eval.jsonl", "data/dataset/train.jsonl", "data/dataset_dsv4/train.jsonl", "data/dataset_sonnet/train.jsonl"):
        p = ROOT / f
        if p.exists():
            for l in open(p):
                r = json.loads(l)
                if r.get("publishable"): ids.add(r["id"])
    # gold rows that are not in any dataset file: publishable iff source is foodcom/wikibooks
    return ids

def is_public_id(i: str, ids: set[str]) -> bool:
    return i in ids or i.startswith(("foodcom-", "wikibooks-"))

def copy_tree(src: Path, dst: Path):
    # overwrite in place (no rmtree: some mounts forbid deletes). Removed source files would linger — none expected.
    def ignore(d, names):
        out = set()
        for n in names:
            if n in EXCLUDE_DIRS or n in EXCLUDE_FILES or n.endswith(".egg-info") or n.endswith(".pyc"): out.add(n)
        return out
    shutil.copytree(src, dst, ignore=ignore, dirs_exist_ok=True)

def main() -> int:
    OUT.mkdir(exist_ok=True)
    for f in COPY_FILES:
        s = ROOT / f
        if s.exists(): shutil.copy2(s, OUT / Path(f).name)
    for s, d in COPY_TREES.items(): copy_tree(ROOT / s, OUT / d)
    ids = publishable_ids(); stats = {}
    for s, d in {**DATA_FILES, **RAW_PUBLIC}.items():
        sp = ROOT / s
        if not sp.exists(): continue
        dp = OUT / d; dp.parent.mkdir(parents=True, exist_ok=True)
        if sp.suffix == ".json": shutil.copy2(sp, dp); continue
        kept = total = 0
        with open(dp, "w") as out:
            for l in open(sp):
                if not l.strip(): continue
                r = json.loads(l); total += 1
                if is_public_id(r["id"], ids) and not r["id"].startswith("ss-"):
                    out.write(json.dumps(r, ensure_ascii=False) + "\n"); kept += 1
        stats[d] = (kept, total)
    # secret scan over every text file staged
    hits = []
    for p in OUT.rglob("*"):
        if p.is_file() and p.suffix in {".py", ".md", ".toml", ".yaml", ".yml", ".sh", ".json", ".jsonl", ".svg", ".txt", ".env"} and ".git" not in p.parts:
            try: txt = p.read_text(errors="ignore")
            except Exception: continue
            if SECRET_RE.search(txt): hits.append(str(p.relative_to(OUT)))
    if hits:
        print("REFUSING: possible secrets in", hits, file=sys.stderr); return 2
    for d, (k, t) in stats.items(): print(f"  {d}: {k}/{t} records (publishable only)")
    print("exported to", OUT)
    return 0

if __name__ == "__main__": raise SystemExit(main())
