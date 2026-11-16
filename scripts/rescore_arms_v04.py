#!/usr/bin/env python3
"""Re-score saved pod eval traces (taxonomy <= 0.3.3, reward v2) against the CURRENT gold with the CURRENT scoring (0.4.0, v3).
No model is re-run: the saved assistant answers are parsed and scored again. "Unplaced" resolves to "Global" via scoring.PATH_ALIASES.
Usage (from the project root, system python with pyyaml + zstandard):
    PYTHONPATH=env:env/environments/recipe_classify python3 scripts/rescore_arms_v04.py"""
import io, json, sys, glob, statistics
from pathlib import Path
import zstandard
ROOT = Path(__file__).resolve().parents[1]
from recipe_tax.taxonomy import Taxonomy
from recipe_tax import scoring
import re
src = (ROOT / "env/environments/recipe_classify/recipe_classify/taskset.py").read_text()
ns = {}; exec(re.sub(r"import verifiers\.v1 as vf.*?\n", "", src.split("class RecipeData")[0]), ns)   # reuse parse_answer without importing verifiers
parse_answer = ns["parse_answer"]

ARMS = {"B'' SFT (Sonnet silver)": "recipe-classify-gold--sft-2b-sonnet--ce7cfdc0", "D GRPO from B''": "recipe-classify-gold--grpo-sft-sonnet-step_80--8425e14f"}

def rows(run):
    for f in sorted(glob.glob(str(ROOT / "pod-archive/outputs" / run / "monitors/file/traces/stream/*.jsonl.zst"))):
        with open(f, "rb") as fh:
            for line in io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8"):
                if line.strip(): yield json.loads(line)

def find(obj, pred, out):
    if isinstance(obj, dict):
        if pred(obj): out.append(obj)
        for v in obj.values(): find(v, pred, out)
    elif isinstance(obj, list):
        for v in obj: find(v, pred, out)

def main():
    tax = Taxonomy.load()
    gold = {json.loads(l)["prompt"]: json.loads(l) for l in open(ROOT / "data/dataset_v04/eval.jsonl")}
    for name, run in ARMS.items():
        n = hit = 0; rew = []; org = []; con = []; old = []; fus_pred = 0; infl_pred = 0
        for r in rows(run):
            users = []; find(r, lambda d: d.get("role") == "user" and isinstance(d.get("content"), str), users)
            asst = []; find(r, lambda d: d.get("role") == "assistant" and "content" in d, asst)
            g = next((gold[u["content"]] for u in users if u["content"] in gold), None)
            if g is None or not asst: continue
            n += 1
            txt = asst[-1]["content"]; txt = txt if isinstance(txt, str) else json.dumps(txt)
            pred = parse_answer(txt)
            s = scoring.score(tax, g["labels"], pred, ingredients=g.get("ingredients"), title=g["prompt"].split("\n", 1)[0].removeprefix("Title:").strip()) if pred is not None else {"reward": 0.0, "origin": 0.0, "contains": 0.0}
            rew.append(s["reward"]); org.append(s["origin"]); con.append(s["contains"])
            rr = []; find(r, lambda d: "reward" in d and isinstance(d["reward"], (int, float)), rr)
            if rr: old.append(rr[0]["reward"])
            if pred:
                o = pred.get("origin") or []
                fus_pred += any(isinstance(x, dict) and x.get("path") == ["Global", "Fusion"] for x in o)
                infl_pred += any(isinstance(x, dict) and x.get("role") == "influence" for x in o)
        m = statistics.mean
        print(f"{name}: matched {n}/439 | v2 as recorded {m(old):.3f} (n={len(old)}) | v3 on 0.4.0 gold: reward {m(rew):.3f} origin {m(org):.3f} contains {m(con):.3f} | answers with an influence {infl_pred}")
if __name__ == "__main__": main()
