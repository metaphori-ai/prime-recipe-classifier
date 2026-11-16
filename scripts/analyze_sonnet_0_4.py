#!/usr/bin/env python3
"""Report on the Sonnet 0.4.0 relabel outputs. Read-only; prints to stdout.

    uv run --with pyyaml python scripts/analyze_sonnet_0_4.py

2a: data/silver/origin_silver_sonnet55_refusion_0_4.jsonl vs data/silver/origin_silver_sonnet55.jsonl (old label, same ids)
2b: data/silver/origin_silver_sonnet55_sourced.jsonl
Validation: every path must exist in taxonomy.yaml, exactly one primary, at most `MAX_INFL` influences, no duplicate paths.
"""
import collections, json, random, sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "data" / "silver"
MAX_INFL = 3   # taxonomy 0.4.0: influence cap 2 -> 3
FUSION = ("Global", "Fusion")


def load(p):
    return {json.loads(l)["id"]: json.loads(l) for l in open(p, encoding="utf-8") if l.strip()}


def tax_paths():
    tax = yaml.safe_load(open(ROOT / "taxonomy.yaml", encoding="utf-8"))
    paths = set()

    def walk(node, path):
        paths.add(tuple(path))
        if isinstance(node, dict):
            for k, v in node.items():
                if str(k).startswith("_") or k == "c":
                    continue
                walk(v, path + [k])

    for k, v in tax["origin"].items():
        walk(v, [k])
    return paths, tax.get("version")


def prim(r):
    return next((tuple(x["path"]) for x in r["labels"]["origin"] if x.get("role", "primary") == "primary"), None)


def infl(r):
    return sorted(tuple(x["path"]) for x in r["labels"]["origin"] if x.get("role") == "influence")


def J(p):
    return " > ".join(p) if p else "—"


def validate(rows, paths):
    bad = []
    for i, r in rows.items():
        if "error" in r:
            continue
        o = r["labels"]["origin"]
        prims = [x for x in o if x.get("role", "primary") == "primary"]
        infs = [x for x in o if x.get("role") == "influence"]
        why = []
        if len(prims) != 1: why.append(f"{len(prims)} primaries")
        if len(infs) > MAX_INFL: why.append(f"{len(infs)} influences")
        for x in o:
            if tuple(x["path"]) not in paths: why.append("path not in taxonomy: " + J(x["path"]))
        allp = [tuple(x["path"]) for x in o]
        if len(allp) != len(set(allp)): why.append("duplicate path")
        if prims and tuple(prims[0]["path"]) == FUSION and len(infs) < 2: why.append(f"Fusion with {len(infs)} influence(s)")
        if why: bad.append((i, why))
    return bad


def errors(rows):
    errs = [(i, r["error"], (r.get("reply") or "")[:80]) for i, r in rows.items() if "error" in r]
    return errs


def main():
    paths, ver = tax_paths()
    print(f"taxonomy version {ver} · {len(paths)} valid paths\n")
    rng = random.Random(7)

    # ---------- 2a ----------
    fa = S / "origin_silver_sonnet55_refusion_0_4.jsonl"
    if fa.exists():
        new = load(fa); old = load(S / "origin_silver_sonnet55.jsonl"); raw = load(S / "_input_refusion_0_4.jsonl")
        ok = {i: r for i, r in new.items() if "error" not in r}
        print(f"== 2a refusion: {len(new)} rows / {len(raw)} inputs · missing {sorted(set(raw) - set(new))[:5]} · errors {len(new) - len(ok)}")
        changed = [i for i in ok if prim(ok[i]) != prim(old[i])]
        fus = [i for i in ok if prim(ok[i]) == FUSION]
        print(f"   primary changed: {len(changed)} / {len(ok)}")
        print(f"   now Global > Fusion: {len(fus)}")
        print(f"   changed to Fusion: {sum(prim(ok[i]) == FUSION for i in changed)} · changed to something else: {sum(prim(ok[i]) != FUSION for i in changed)}")
        trans = collections.Counter((J(prim(old[i])[:2]), J(prim(ok[i])[:2])) for i in changed)
        print("   top transitions (old → new, first two levels):")
        for (a, b), n in trans.most_common(12): print(f"     {n:4d}  {a}  →  {b}")
        ic = collections.Counter(len(infl(ok[i])) for i in ok); io = collections.Counter(len(infl(old[i])) for i in ok)
        print(f"   influences per row, new {dict(sorted(ic.items()))} · old {dict(sorted(io.items()))}")
        print(f"   lost all influences: {sum(len(infl(ok[i])) == 0 for i in ok)} · influence set changed (primary same): "
              f"{sum(prim(ok[i]) == prim(old[i]) and infl(ok[i]) != infl(old[i]) for i in ok)}")
        print("   15 examples of changes (title | old | new):")
        tofus = [i for i in changed if prim(ok[i]) == FUSION]; other = [i for i in changed if prim(ok[i]) != FUSION]
        rng.shuffle(tofus); rng.shuffle(other)
        for i in tofus[:10] + other[:5]:
            print(f"     · {raw[i]['title'][:60]}")
            print(f"         old: {J(prim(old[i]))}  + {[J(p) for p in infl(old[i])]}")
            print(f"         new: {J(prim(ok[i]))}  + {[J(p) for p in infl(ok[i])]}")
        bad = validate(new, paths); print(f"   validation failures: {len(bad)}"); [print("     ", b) for b in bad[:10]]
        er = errors(new); print(f"   error rows: {len(er)}"); [print("     ", e) for e in er[:10]]
        print(f"   name-resolved rows: {sum(any(x.get('resolved_by_name') for x in r['labels']['origin']) for r in ok.values())}\n")
    else:
        print("2a output not found\n")

    # ---------- 2b ----------
    fb = S / "origin_silver_sonnet55_sourced.jsonl"
    if fb.exists():
        new = load(fb); raw = load(S / "_input_sourced_rest.jsonl")
        ok = {i: r for i, r in new.items() if "error" not in r}
        print(f"== 2b sourced: {len(new)} rows / {len(raw)} inputs · missing {sorted(set(raw) - set(new))[:5]} · errors {len(new) - len(ok)}")
        top = collections.Counter(prim(r)[0] for r in ok.values())
        print("   per top-level region:")
        for k, n in top.most_common(): print(f"     {n:4d}  {k}")
        print(f"   with any influence: {sum(bool(infl(r)) for r in ok.values())} · influences per row {dict(sorted(collections.Counter(len(infl(r)) for r in ok.values()).items()))}")
        print(f"   Global > Fusion: {sum(prim(r) == FUSION for r in ok.values())} · Global breakdown {dict(collections.Counter(J(prim(r)) for r in ok.values() if prim(r)[0] == 'Global'))}")
        print(f"   primary depth {dict(sorted(collections.Counter(len(prim(r)) for r in ok.values()).items()))}")
        bad = validate(new, paths); print(f"   validation failures: {len(bad)}"); [print("     ", b) for b in bad[:10]]
        er = errors(new); print(f"   error rows: {len(er)}"); [print("     ", e) for e in er[:10]]
        print(f"   name-resolved rows: {sum(any(x.get('resolved_by_name') for x in r['labels']['origin']) for r in ok.values())}")
    else:
        print("2b output not found")


if __name__ == "__main__":
    main()
