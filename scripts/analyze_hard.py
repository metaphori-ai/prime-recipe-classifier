#!/usr/bin/env python3
"""Slice breakdown of saved answers on a gold set (0.4.0 gold, reward v3). Reads vf-run traces.jsonl or prime-rl *.jsonl.zst.
Usage (project root): PYTHONPATH=env python3 scripts/analyze_hard.py <eval file> name=path [name=path ...]"""
import io, json, sys, re, glob, statistics, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
from recipe_tax.taxonomy import Taxonomy
from recipe_tax import scoring
src = (ROOT / "env/environments/recipe_classify/recipe_classify/taskset.py").read_text()
ns = {}; exec(re.sub(r"import verifiers\.v1 as vf.*?\n", "", src.split("class RecipeData")[0]), ns); parse = ns["parse_answer"]
def find(o, p, out):
    if isinstance(o, dict):
        if p(o): out.append(o)
        for v in o.values(): find(v, p, out)
    elif isinstance(o, list):
        for v in o: find(v, p, out)
def lines(path):
    p = Path(path)
    if p.is_dir():
        fs = sorted(glob.glob(str(p / "monitors/file/traces/stream/*.jsonl.zst"))) or [str(p / "traces.jsonl")]
    else: fs = [str(p)]
    for f in fs:
        if f.endswith(".zst"):
            import zstandard
            with open(f, "rb") as fh:
                yield from io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
        else: yield from open(f, encoding="utf-8")
def main():
    tax = Taxonomy.load(); F = ["Global", "Fusion"]
    gold = {json.loads(l)["prompt"]: json.loads(l) for l in open(sys.argv[1])}
    reason = {}
    for s in ("gold_hard", "gold"):
        f = ROOT / "data" / s / "candidates.jsonl"
        if f.exists():
            for l in open(f): c = json.loads(l); reason[c["id"]] = c.get("selection_reason", "")
    def sl(i):
        r = reason.get(i, "").split(" ")[0]
        return "fusion" if r == "R8" else "regional" if r == "R9" else "ownership" if r.startswith("R") else "all"
    m = statistics.mean; out = {}
    for arg in sys.argv[2:]:
        name, path = arg.split("=", 1); R = []
        for line in lines(path):
            if not line.strip(): continue
            r = json.loads(line); u = []; find(r, lambda d: d.get("role") == "user" and isinstance(d.get("content"), str), u)
            pr = r.get("task", {}).get("data", {}).get("prompt") if isinstance(r.get("task"), dict) else None
            g = gold.get(pr) or next((gold[x["content"]] for x in u if x["content"] in gold), None)
            if not g: continue
            a = []; find(r, lambda d: d.get("role") == "assistant" and "content" in d, a)
            t = a[-1]["content"] if a else ""; p = parse(t if isinstance(t, str) else json.dumps(t))
            s = scoring.score(tax, g["labels"], p, ingredients=g.get("ingredients"), title=g["prompt"].split("\n", 1)[0].removeprefix("Title:").strip()) if p else {"reward": 0, "origin": 0, "contains": 0}
            go = g["labels"]["origin"]; po = [o for o in ((p or {}).get("origin") or []) if isinstance(o, dict)]
            gp = [o for o in go if o["role"] == "primary"][0]["path"]
            pp = [("Global" if x == "Unplaced" else x) for x in (next((o.get("path") for o in po if o.get("role", "primary") == "primary"), None) or [])]
            gi = {tuple(o["path"]) for o in go if o["role"] == "influence"}; pi = {tuple(o.get("path") or []) for o in po if o.get("role") == "influence"}
            R.append(dict(id=g["id"], sl=sl(g["id"]), rew=s["reward"], org=s["origin"], con=s["contains"], exact=gp == pp, gd=len(gp), pd=len(pp),
                          short=len(pp) < len(gp) and pp == gp[:len(pp)], samedepth_wrong=pp != gp and len(pp) == len(gp), ginf=bool(gi), pinf=bool(pi), infhit=bool(gi & pi), gf=gp == F, pf=pp == F,
                          title=g["prompt"].split("\n")[0][7:60], gp=gp, pp=pp, pi=sorted(pi)))
        out[name] = R
        print(f"\n== {name}: n={len(R)} reward {m(x['rew'] for x in R):.3f} origin {m(x['org'] for x in R):.3f} contains {m(x['con'] for x in R):.3f} exact {m(x['exact'] for x in R):.0%} depth pred {m(x['pd'] for x in R):.2f} gold {m(x['gd'] for x in R):.2f}")
        for k in ("ownership", "fusion", "regional", "all"):
            S = [x for x in R if x["sl"] == k]
            if S: print(f"   {k:9s} n={len(S):3d} origin {m(x['org'] for x in S):.3f} contains {m(x['con'] for x in S):.3f} exact {sum(x['exact'] for x in S):3d} short {sum(x['short'] for x in S):2d} same-depth-wrong {sum(x['samedepth_wrong'] for x in S):2d}")
        print(f"   influences: gold rows {sum(x['ginf'] for x in R)}, offered {sum(x['pinf'] for x in R)}, offered where gold has one {sum(x['pinf'] and x['ginf'] for x in R)}, exact-path hit {sum(x['infhit'] for x in R)}, offered where gold has none {sum(x['pinf'] and not x['ginf'] for x in R)}")
        print(f"   Fusion: gold {sum(x['gf'] for x in R)}, predicted {sum(x['pf'] for x in R)}, both {sum(x['gf'] and x['pf'] for x in R)}, false Fusion {sum(x['pf'] and not x['gf'] for x in R)}; origin on gold-Fusion rows {m([x['org'] for x in R if x['gf']] or [0]):.3f}")
    json.dump(out, open("/tmp/analyze_hard_last.json", "w"))
if __name__ == "__main__": main()
