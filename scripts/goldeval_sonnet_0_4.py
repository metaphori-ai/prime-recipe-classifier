#!/usr/bin/env python3
"""Measure the Sonnet teacher against gold with the CURRENT scoring.py (reward v3): old prompt vs new prompt on the 439,
new prompt on the 224 hard set. Read-only; prints to stdout.

    uv run --with pyyaml python scripts/goldeval_sonnet_0_4.py

Every teacher answer and every gold row goes through scoring.normalize, so "Unplaced" in the 2026-11-08 file resolves to
"Global" (PATH_ALIASES) and old and new are scored by identical code on identical gold.
"""
import collections, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "env"))
from recipe_tax.taxonomy import Taxonomy                      # noqa: E402
from recipe_tax import scoring                                # noqa: E402
from recipe_tax.scoring import normalize, score_origin        # noqa: E402

FUSION = ("Global", "Fusion")
tax = Taxonomy.load()


def load(p):
    return {json.loads(l)["id"]: json.loads(l) for l in open(p, encoding="utf-8") if l.strip()}


def norm_origin(origin):
    n, _ = normalize(tax, {"origin": origin, "contains": []})
    return n["origin"]


def prim(o):
    return next((tuple(x["path"]) for x in o if x.get("role", "primary") == "primary"), None)


def infl(o):
    return [tuple(x["path"]) for x in o if x.get("role") == "influence"]


def J(p):
    return " > ".join(p) if p else "—"


def show(o):
    return J(prim(o)) + ("  + " + ", ".join(J(p) for p in infl(o)) if infl(o) else "")


def related(a, b):
    """Same tradition at two levels, or one path is a prefix of the other (a bare macro like 'South Asian' vs 'South Asian > Indian')."""
    if a is None or b is None: return False
    n = min(len(a), len(b), 2)
    return a[:n] == b[:n]


def evaluate(name, teacher, gold):
    rows = {}
    errs = [i for i, r in teacher.items() if "error" in r]
    missing = [i for i in gold if i not in teacher]
    for i, g in gold.items():
        go = norm_origin(g["labels"]["origin"])
        t = teacher.get(i)
        if t is None or "error" in t:
            rows[i] = {"gold": go, "pred": None, "score": 0.0}; continue
        po = norm_origin(t["labels"]["origin"])
        rows[i] = {"gold": go, "pred": po, "score": score_origin(tax, go, po) if prim(po) else 0.0}
    n = len(rows); ans = [r for r in rows.values() if r["pred"]]
    exact = sum(prim(r["gold"]) == prim(r["pred"]) for r in ans)
    print(f"\n=== {name} ===")
    print(f"gold rows {n} · answered {len(ans)} · error rows {len(errs)} · missing {len(missing)}")
    print(f"exact primary match: {exact}/{n} = {100*exact/n:.1f}%")
    print(f"mean origin score (v3, unanswered = 0): {sum(r['score'] for r in rows.values())/n:.3f}")

    gi = [r for r in ans if infl(r["gold"]) and prim(r["gold"]) != FUSION]
    if gi:
        same = sum(prim(r["pred"])[:2] == prim(r["gold"])[:2] for r in gi)
        kept_exact = sum(bool(set(infl(r["pred"])) & set(infl(r["gold"]))) for r in gi)
        kept_trad = sum(any(related(a, b) for a in infl(r["pred"]) for b in infl(r["gold"])) for r in gi)
        promoted = sum(any(related(prim(r["pred"]), b) for b in infl(r["gold"])) and prim(r["pred"])[:2] != prim(r["gold"])[:2] for r in gi)
        dropped_all = sum(not infl(r["pred"]) for r in gi)
        tofus = sum(prim(r["pred"]) == FUSION for r in gi)
        m = len(gi)
        print(f"gold rows WITH an influence, not Fusion: {m}")
        print(f"   teacher primary same tradition (first two levels): {same}/{m} = {100*same/m:.1f}%")
        print(f"   teacher kept >=1 gold influence, exact path:       {kept_exact}/{m} = {100*kept_exact/m:.1f}%")
        print(f"   teacher kept >=1 gold influence, same tradition:   {kept_trad}/{m} = {100*kept_trad/m:.1f}%")
        print(f"   teacher made a gold influence its PRIMARY:         {promoted}/{m} = {100*promoted/m:.1f}%")
        print(f"   (teacher gave no influence at all: {dropped_all} · teacher called it Fusion: {tofus})")
    gn = [r for r in ans if not infl(r["gold"])]
    added = sum(bool(infl(r["pred"])) for r in gn)
    print(f"gold rows with NO influence: {len(gn)} · teacher added one to {added} ({100*added/max(1,len(gn)):.1f}%)")
    gf = [r for r in rows.values() if prim(r["gold"]) == FUSION]; tf = [r for r in ans if prim(r["pred"]) == FUSION]
    both = [r for r in gf if r["pred"] and prim(r["pred"]) == FUSION]
    same_set = sum(set(infl(r["gold"])) == set(infl(r["pred"])) for r in both)
    print(f"Fusion: gold {len(gf)} · teacher {len(tf)} · agree {len(both)} (identical influence sets on {same_set}) · "
          f"teacher-only {len(tf)-len(both)} · gold-only {len(gf)-len(both)}")
    return rows


def main():
    print(f"scoring.REWARD_VERSION = {scoring.REWARD_VERSION} · taxonomy {tax.version}")
    S = ROOT / "data" / "silver"
    gold = load(ROOT / "data" / "gold" / "gold.jsonl"); hard = load(ROOT / "data" / "gold_hard" / "gold.jsonl")
    titles = {i: g.get("title", "") for i, g in {**gold, **hard}.items()}
    old = evaluate("OLD prompt (2026-11-08 origin_eval_sonnet55.jsonl) on the 439, current gold", load(S / "origin_eval_sonnet55.jsonl"), gold)
    new = evaluate("NEW prompt (_goldeval_sonnet_0_4_gold.jsonl) on the 439", load(S / "_goldeval_sonnet_0_4_gold.jsonl"), gold)
    evaluate("NEW prompt (_goldeval_sonnet_0_4_hard.jsonl) on the 224 hard set", load(S / "_goldeval_sonnet_0_4_hard.jsonl"), hard)

    d = sorted(((new[i]["score"] - old[i]["score"], i) for i in gold), key=lambda x: x[0])
    up = sum(1 for x, _ in d if x > 1e-9); down = sum(1 for x, _ in d if x < -1e-9)
    print(f"\n=== per-recipe change old → new on the 439: improved {up} · worse {down} · unchanged {len(d)-up-down} · "
          f"sum of gains {sum(x for x,_ in d if x>0):.2f} · sum of losses {sum(x for x,_ in d if x<0):.2f}")
    print("20 largest drops (title | gold | old answer | new answer):")
    for x, i in d[:20]:
        print(f"  {x:+.3f}  {titles[i][:64]}   [{i}]")
        print(f"        gold: {show(new[i]['gold'])}")
        print(f"        old : {show(old[i]['pred']) if old[i]['pred'] else 'no answer'}   ({old[i]['score']:.3f})")
        print(f"        new : {show(new[i]['pred']) if new[i]['pred'] else 'no answer'}   ({new[i]['score']:.3f})")
    print("\n10 largest gains:")
    for x, i in d[-10:][::-1]:
        print(f"  {x:+.3f}  {titles[i][:64]} | gold: {show(new[i]['gold'])} | old: {show(old[i]['pred']) if old[i]['pred'] else '—'} | new: {show(new[i]['pred']) if new[i]['pred'] else '—'}")
    kinds = collections.Counter()
    for x, i in d:
        if x >= -1e-9: continue
        g, o, n_ = new[i]["gold"], old[i]["pred"], new[i]["pred"]
        if not n_: kinds["new: no answer"] += 1
        elif infl(g) and not infl(n_) and o and infl(o): kinds["new dropped the influence old had"] += 1
        elif prim(n_) != prim(o or []) and prim(g) == prim(o or []): kinds["new changed a primary old had right"] += 1
        elif (prim(n_) == FUSION) != (prim(g) == FUSION): kinds["Fusion boundary"] += 1
        elif not infl(g) and infl(n_): kinds["new added a spurious influence"] += 1
        else: kinds["other"] += 1
    print("\nwhy the worse rows got worse:", dict(kinds.most_common()))


if __name__ == "__main__":
    main()
