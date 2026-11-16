#!/usr/bin/env python3
"""What does the model read? Build copies of the two gold sets where only the recipe text shown to the model changes:

  title        Title only
  ingredients  Ingredients only
  method       Title + ingredients + method (method capped at METHOD_CAP characters)
  (the existing files are "both": title + ingredients)

Labels, ids and order are unchanged; nothing in data/dataset_v04 is modified. Output: data/ablation/<set>--<variant>.jsonl
(private: contains hard-set and app recipes) and one vf-run config per set/variant/backend in configs/ablation/.

  python3 scripts/input_ablation.py
"""
import json, glob
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
METHOD_CAP = 2000
SETS = {"gold": "eval.jsonl", "hard": "eval_gold_hard.jsonl"}
CFG = ROOT / "env/environments/recipe_classify/configs"

def main():
    raw = {}
    for f in glob.glob(str(ROOT / "data/raw/*.jsonl")):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l); ins = r.get("instructions") or ""
            raw[r["id"]] = " ".join(map(str, ins)) if isinstance(ins, list) else str(ins)
    out = ROOT / "data/ablation"; out.mkdir(exist_ok=True); (CFG / "ablation").mkdir(exist_ok=True)
    ollama = (CFG / "ollama.toml").read_text().split("\n", 3)[3]
    mlx = (CFG / "mlx-hard-v04.toml").read_text().split("[env.taskset]")[0].split("\n", 2)[2]
    for s, fn in SETS.items():
        rows = [json.loads(l) for l in open(ROOT / "data/dataset_v04" / fn, encoding="utf-8")]
        stats = {}
        for v in ("title", "ingredients", "method"):
            n_m = 0; cut = 0
            with open(out / f"{s}--{v}.jsonl", "w", encoding="utf-8") as o:
                for r in rows:
                    title, ingr = r["prompt"].split("\n", 1)
                    assert title.startswith("Title:") and ingr.startswith("Ingredients:"), r["id"]
                    if v == "title": p = title
                    elif v == "ingredients": p = ingr
                    else:
                        m = " ".join(raw.get(r["id"], "").split()); n_m += bool(m); cut += len(m) > METHOD_CAP
                        p = r["prompt"] + ("\nMethod: " + m[:METHOD_CAP] if m else "")
                    o.write(json.dumps({**r, "prompt": p}, ensure_ascii=False) + "\n")
            stats[v] = (len(rows), n_m, cut)
            path = f"/Users/jasdeep/svml/prime-intellect/data/ablation/{s}--{v}.jsonl"
            (CFG / "ablation" / f"ollama-{s}-{v}.toml").write_text(
                f"# INPUT ABLATION ({v}), {s} set, ORIGIN-ONLY instructions, local Ollama:\n#   bash scripts/vf_run_ollama.sh <tag> {len(rows)} 2 ablation/ollama-{s}-{v}.toml\n"
                + ollama + f'\n[env.taskset]\ndataset_path = "{path}"\nmode = "origin"\n')
            (CFG / "ablation" / f"mlx-{s}-{v}.toml").write_text(
                f"# INPUT ABLATION ({v}), {s} set, FULL instructions (for the Phase 3 weights), local mlx_lm.server:\n#   bash scripts/vf_run_mlx.sh <model-as-served> {len(rows)} 4 ablation/mlx-{s}-{v}.toml\n"
                + mlx + f'[env.taskset]\ndataset_path = "{path}"\n')
        print(s, {v: f"{n} rows" + (f", method on {m}, capped on {c}" if v == "method" else "") for v, (n, m, c) in stats.items()})

if __name__ == "__main__": main()
