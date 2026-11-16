#!/usr/bin/env python3
"""How much of a GRPO run's training signal came from each reward term?

GRPO ranks the attempts inside a group (same recipe, 8 attempts) by reward; only differences INSIDE a group teach anything.
For a saved prime-rl run this reads every training attempt's origin and contains scores and reports, per block of steps:
  - share of the within-group reward variance that came from origin, from contains, and from their overlap;
  - groups where every attempt scored the same (no signal), under the real reward and under an origin-only reward;
  - how often an attempt's advantage would change sign if the reward were origin alone.
Counts every attempt the sampler produced, including ones later dropped as stale.

  python3 scripts/reward_signal_split.py pod-archive/models/rl-v04-run
"""
import io, json, sys, glob, collections, statistics
import zstandard
W_O, W_C = 0.55, 0.45

def main(run):
    groups = collections.defaultdict(list)
    for f in sorted(glob.glob(f"{run}/monitors/file/traces/stream/*.jsonl.zst")):
        with open(f, "rb") as fh:
            for line in io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8"):
                r = json.loads(line)
                if r["env"]["name"].endswith("-train") is False or not r.get("traces"): continue
                t = r["traces"][0]; m = t.get("metrics") or {}
                if "origin" not in m or "contains" not in m: continue
                step = (r["run"].get("work") or {}).get("step")
                groups[r["group"]["id"]].append((step, float(m["origin"]), float(m["contains"]), float(t["reward"])))
    blocks = collections.defaultdict(list)
    for g in groups.values():
        if len(g) < 2: continue
        s = [x[0] for x in g if x[0] is not None]
        blocks[min(3, (max(s) - 1) // 20) if s else 0].append(g)
    print(f"{'steps':>8s} {'groups':>7s} {'origin':>8s} {'contains':>9s} {'overlap':>8s} {'no signal':>10s} {'no signal if':>13s} {'sign flips if':>14s} {'contains-only':>14s}")
    print(f"{'':>8s} {'':>7s} {'share of within-group variance':>27s} {'(real)':>10s} {'origin only':>13s} {'origin only':>14s} {'groups':>14s}")
    for b in sorted(blocks):
        G = blocks[b]; vo = vc = cov = 0.0; nos = nos_o = flips = tot = conly = 0
        for g in G:
            o = [W_O * x[1] for x in g]; c = [W_C * x[2] for x in g]; n = len(g)
            mo, mc = sum(o) / n, sum(c) / n
            do = [x - mo for x in o]; dc = [x - mc for x in c]
            vo += sum(x * x for x in do); vc += sum(x * x for x in dc); cov += 2 * sum(x * y for x, y in zip(do, dc))
            flat_o = max(o) - min(o) < 1e-9; flat_r = max(a + b_ for a, b_ in zip(o, c)) - min(a + b_ for a, b_ in zip(o, c)) < 1e-9
            nos += flat_r; nos_o += flat_o; conly += flat_o and not flat_r
            for x, y in zip(do, dc):
                a = x + y
                if abs(a) > 1e-9:
                    tot += 1; flips += (a > 0) != (x > 0) or abs(x) < 1e-9
        T = vo + vc + cov
        print(f"{b*20+1:>4d}-{b*20+20:<3d} {len(G):7d} {vo/T:8.0%} {vc/T:9.0%} {cov/T:8.0%} {nos/len(G):10.0%} {nos_o/len(G):13.0%} {flips/max(tot,1):14.0%} {conly/len(G):14.0%}")

if __name__ == "__main__": main(sys.argv[1])
