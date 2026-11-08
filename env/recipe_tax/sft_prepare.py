"""Turn data/dataset/train.jsonl into chat-format SFT data (messages: system / user / assistant), split train/valid.

  uv run --with pyyaml python -m recipe_tax.sft_prepare            # -> data/sft/{train,valid}.jsonl (+ test.jsonl = gold, for reference only)

Target = exactly what the reward wants: <answer>{"origin": [...], "contains": [...], "evidence": {...}}</answer>.
Evidence is synthesized by matching each class to an ingredient line via taxonomy.ingredient_traps + a small lexicon, so the
student learns to point at a line (the reward checks evidence by substring). Classes with no matchable line get no evidence entry.
System prompt is the environment's (identical to eval/RL) so SFT, RL and eval share one frame.
"""
from __future__ import annotations
import json, random, re, sys
from pathlib import Path
from .taxonomy import Taxonomy

ROOT = Path(__file__).resolve().parents[2]
from recipe_tax.lexicon import LEX

def evidence_for(cls: str, ingredients: list[str], traps: dict) -> str | None:
    words = list(LEX.get(cls, [])) + [w.lower() for w in traps.get({"tree_nuts": "nuts"}.get(cls, cls), [])]
    low = [(l, l.lower()) for l in ingredients]
    for w in words:
        for orig, l in low:
            if w in l:
                # return the shortest substring that still matches: the matched word in its original casing context
                i = l.find(w); return orig[i:i + len(w)]
    return None

def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default=None, help="dir with train.jsonl (default data/dataset)")
    ap.add_argument("--out-dir", default=None, help="where to write {train,valid}.jsonl (default data/sft)")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT / "env" / "environments" / "recipe_classify"))
    src = (ROOT / "env" / "environments" / "recipe_classify" / "recipe_classify" / "taskset.py").read_text()
    ns = {}; exec(re.sub(r"import verifiers\.v1 as vf.*?\n", "", src.split("class RecipeData")[0]), ns)
    tax = Taxonomy.load(); system = ns["system_prompt"](tax)
    ds_dir = Path(a.dataset_dir) if a.dataset_dir else ROOT / "data" / "dataset"
    rows = [json.loads(l) for l in open(ds_dir / "train.jsonl") if l.strip()]
    random.Random(7).shuffle(rows)
    out_dir = (Path(a.out_dir).resolve() if a.out_dir else ROOT / "data" / "sft"); out_dir.mkdir(parents=True, exist_ok=True)
    def to_chat(r):
        g = r["labels"]
        origin = [{"path": o["path"], "role": o.get("role", "primary")} for o in g["origin"]]
        ev = {c: e for c in g["contains"] if (e := evidence_for(c, r["ingredients"], tax.traps))}
        ans = {"origin": origin, "contains": g["contains"], "evidence": ev}
        return {"messages": [{"role": "system", "content": system}, {"role": "user", "content": r["prompt"]},
                             {"role": "assistant", "content": "<answer>" + json.dumps(ans, ensure_ascii=False) + "</answer>"}]}
    n_valid = max(100, len(rows) // 20)
    splits = {"valid": rows[:n_valid], "train": rows[n_valid:]}
    for name, rs in splits.items():
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rs: f.write(json.dumps(to_chat(r), ensure_ascii=False) + "\n")
    tot = cov = 0
    for r in splits["train"]:
        for c in r["labels"]["contains"]: tot += 1; cov += evidence_for(c, r["ingredients"], tax.traps) is not None
    print(json.dumps({"train": len(splits["train"]), "valid": len(splits["valid"]), "evidence_coverage": round(cov / max(tot, 1), 3),
                      "system_prompt_tokens≈": len(system) // 4, "out": str(out_dir.resolve())}, indent=1))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
