#!/usr/bin/env python3
"""Where is the model unsure, and was the right answer among the alternatives?

Two commands.

  run     (on the Mac, needs mlx-lm; ONE model on the GPU at a time). For every recipe in a gold file:
            1. greedy answer (temperature 0), keeping each token's log-probability and its top alternatives;
            2. "forced" scoring: feed the GOLD origin and the model's OWN origin back in, token by token, and read the exact
               probability the model gives each token. No sampling, no top-k limit.
          Writes one JSON line per recipe.
  report  (anywhere, plain Python). Reads those lines and prints: near misses vs blind misses, the decision where the model left
          the gold answer, whether confidence predicts correctness, the influence decision, and a back-off what-if.

  # Phase 3 weights were trained with the full instructions -> --mode full. Phase 5 weights -> --mode origin.
  cd env && uv run --with mlx-lm --with pyyaml python ../scripts/logprob_probe.py run \
      --model ../pod-archive/models/grpo-v04-step_80 --data ../data/dataset_v04/eval_gold_hard.jsonl \
      --mode full --out ../data/logprobs/hard--grpo-v04-step_80.jsonl
  PYTHONPATH=env python3 scripts/logprob_probe.py report data/logprobs/hard--grpo-v04-step_80.jsonl

Reading the numbers. P(answer) is the probability of the whole origin part of the answer, exactly as the training targets
format it. A wrong answer with P(gold) = 0.30 is a near miss (the model had it and chose otherwise); P(gold) = 0.0005 is a blind
miss (it never considered it). After GRPO the model is very confident (entropy fell to 0.02 in arm E), so always read the
calibration table before trusting any single probability.
"""
from __future__ import annotations
import argparse, copy, json, math, re, sys, statistics, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "env"))
from recipe_tax.taxonomy import Taxonomy          # noqa: E402
from recipe_tax import scoring                    # noqa: E402

_src = (ROOT / "env/environments/recipe_classify/recipe_classify/taskset.py").read_text()
_ns: dict = {}
exec(re.sub(r"import verifiers\.v1 as vf.*?\n", "", _src.split("class RecipeData")[0]), _ns)
system_prompt, parse_answer = _ns["system_prompt"], _ns["parse_answer"]

TOPK = 10
PRIORITY = {"more": 0, "depth": 1, "node": 2, "role": 3, "open": 4, "head": 5, "tail": 6}
HEAD = '<answer>{"origin": '
TAIL = {"full": ', "contains": [', "origin": "}</answer>"}   # what follows the origin list in each mode's training targets
CUT = {"full": len(', "con'), "origin": len("}</ans")}        # score tokens that START before this point of the tail: the
                                                               # boundary token is included whole, the rest is the same for
                                                               # every candidate answer and is left out


# ----------------------------------------------------------------------------------------------------------------------
# Canonical origin string + a label for every character (which decision it belongs to)
# ----------------------------------------------------------------------------------------------------------------------
def canonical(origin: list[dict]) -> tuple[str, list[tuple]]:
    """Return json.dumps(origin) and, per character, a label:
       ("node", e, j)  inside the j-th name of entry e (j from 1)
       ("depth", e, j) the characters after name j: either '", "' (go deeper) or '"]' (stop)   <- the depth decision
       ("role", e)     ', "role": "primary"'
       ("more", e)     after entry e: either '}, {"path": ["' (another entry) or '}]' (finish)  <- the influence decision
       ("open",)       the opening '[{"path": ["'
    """
    s, lab = "", []
    def add(txt, label):
        nonlocal s
        s += txt; lab.extend([label] * len(txt))
    add('[{"path": ["', ("open",))
    for e, o in enumerate(origin):
        path = [str(x) for x in o["path"]]
        for j, name in enumerate(path, 1):
            add(json.dumps(name, ensure_ascii=False)[1:-1], ("node", e, j))
            add('", "' if j < len(path) else '"]', ("depth", e, j))
        add(', "role": ' + json.dumps(o.get("role", "primary")), ("role", e))
        add('}, {"path": ["' if e < len(origin) - 1 else "}]", ("more", e))
    clean = [{"path": [str(x) for x in o["path"]], "role": o.get("role", "primary")} for o in origin]
    assert s == json.dumps(clean, ensure_ascii=False), (s, json.dumps(clean, ensure_ascii=False))
    return s, lab


def label_tokens(origin: list[dict], toks: list[list], mode: str) -> list[dict]:
    """toks = [[text, logprob], ...] for HEAD + canonical(origin) + TAIL[mode]. A token can straddle two parts ('"],' or
    '"}],' or '["South'), so it gets the most decision-bearing label among the characters it covers: the influence decision,
    then the depth decision, then the node. Tokens past the cut are dropped."""
    body, lab = canonical(origin)
    full_lab = [("head",)] * len(HEAD) + lab + [("tail",)] * len(TAIL[mode])
    cut = len(HEAD) + len(body) + CUT[mode]
    out, pos = [], 0
    for text, lp in toks:
        start, end = pos, pos + len(text); pos = end
        if start >= cut: break
        covered = full_lab[start:end] or [("tail",)]
        out.append({"text": text, "lp": lp, "label": min(covered, key=lambda l: PRIORITY.get(l[0], 9)), "start": start})
    return out


def answer_logp(labelled: list[dict]) -> float:
    return sum(t["lp"] for t in labelled)


# ----------------------------------------------------------------------------------------------------------------------
# run (mlx)
# ----------------------------------------------------------------------------------------------------------------------
def cmd_run(a) -> int:
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    tax = Taxonomy.load(); system = system_prompt(tax, a.mode)
    model, tok = load(a.model)
    hf = getattr(tok, "_tokenizer", tok)
    eos = set(getattr(tok, "eos_token_ids", None) or [hf.eos_token_id])

    def dec(i: int) -> str: return hf.decode([i])
    def enc(s: str) -> list[int]: return list(hf.encode(s, add_special_tokens=False))
    def logsoftmax(x):
        x = x.astype(mx.float32)
        return x - mx.logsumexp(x, axis=-1, keepdims=True)

    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    if a.limit: rows = rows[: a.limit]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in open(a.out)} if (a.resume and Path(a.out).exists()) else set()
    out = open(a.out, "a" if a.resume else "w", encoding="utf-8")

    def forced(cache, first_lp, s: str) -> list[list]:
        """Exact per-token log-probabilities of continuing the prompt with string s."""
        ids = enc(s)
        c = clone(cache)
        lps = [float(first_lp[ids[0]].item())]
        if len(ids) > 1:
            lg = logsoftmax(model(mx.array(ids[:-1])[None], cache=c)[0])
            mx.eval(lg)
            lps += [float(lg[k, ids[k + 1]].item()) for k in range(len(ids) - 1)]
        return [[dec(i), lp] for i, lp in zip(ids, lps)]

    state = {"ids": None}
    def clone(cache):
        """A private copy of the prompt cache. If this model's cache cannot be deep-copied, re-read the prompt instead."""
        try:
            return copy.deepcopy(cache)
        except Exception:
            c = make_prompt_cache(model); mx.eval(model(mx.array(state["ids"])[None], cache=c)); return c

    for n, r in enumerate(rows):
        if r["id"] in done: continue
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": r["prompt"]}]
        ids = hf.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, enable_thinking=False)
        if not isinstance(ids, list): ids = list(ids["input_ids"])
        if ids and isinstance(ids[0], list): ids = ids[0]
        state["ids"] = ids
        cache = make_prompt_cache(model)
        first = logsoftmax(model(mx.array(ids)[None], cache=cache)[0, -1]); mx.eval(first)
        base = clone(cache)                              # the prompt, before any answer token

        # 1. greedy answer with alternatives
        lp, text, gtoks = first, "", []
        for _ in range(a.max_tokens):
            top = mx.argsort(-lp)[:TOPK]; mx.eval(top)
            tids = [int(x) for x in top.tolist()]
            tl = [float(lp[i].item()) for i in tids]
            t = tids[0]
            if t in eos: break
            piece = dec(t); text += piece
            gtoks.append({"t": piece, "lp": tl[0], "top": [[dec(i), round(x, 4)] for i, x in zip(tids, tl)]})
            if "</answer>" in text: break
            lp = logsoftmax(model(mx.array([[t]]), cache=cache)[0, -1]); mx.eval(lp)

        rec = {"id": r["id"], "title": r["prompt"].split("\n", 1)[0].removeprefix("Title:").strip(), "mode": a.mode,
               "model": a.model, "gold_origin": r["labels"]["origin"], "greedy_text": text, "greedy_tokens": gtoks}
        if n == 0: rec["prompt_tail"] = hf.decode(ids[-25:])      # shows whether thinking is off in the rendered prompt

        # 2. forced scoring of the gold origin and of the model's own origin, in the training-target format
        gold_s = HEAD + canonical(r["labels"]["origin"])[0] + TAIL[a.mode]
        rec["forced_gold"] = forced(base, first, gold_s)
        pred = parse_answer(text)
        try:
            po = [{"path": [str(x) for x in o["path"]], "role": o.get("role", "primary")} for o in (pred or {}).get("origin") or []
                  if isinstance(o, dict) and isinstance(o.get("path"), list) and o["path"]]
        except Exception:
            po = []
        if po:
            rec["pred_origin_raw"] = po
            rec["forced_pred"] = forced(base, first, HEAD + canonical(po)[0] + TAIL[a.mode])
        out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
        if (n + 1) % 20 == 0: print(f"  {n + 1}/{len(rows)}", file=sys.stderr)
    print(f"done -> {a.out}", file=sys.stderr)
    return 0


# ----------------------------------------------------------------------------------------------------------------------
# report (plain python)
# ----------------------------------------------------------------------------------------------------------------------
def _prim(o):
    return next((tuple(x["path"]) for x in o if x.get("role", "primary") == "primary"), None)

def _infl(o):
    return {tuple(x["path"]) for x in o if x.get("role") == "influence"}

def _auroc(scores: list[float], labels: list[bool]) -> float | None:
    pos = [s for s, l in zip(scores, labels) if l]; neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg: return None
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))

def describe(label: tuple, gold_origin: list[dict]) -> str:
    k = label[0]
    if k == "node":
        e, j = label[1], label[2]
        return f"which node, level {j}" if e == 0 else "which influence"
    if k == "depth":
        e, j = label[1], label[2]
        if e > 0: return "which influence"
        return f"go deeper after level {j}" if j < len(gold_origin[0]["path"]) else f"stop at level {j}"
    if k == "more":
        return "add an influence" if label[1] < len(gold_origin) - 1 else "stop adding influences"
    return k

def analyse(rec: dict, tax: Taxonomy) -> dict:
    mode = rec["mode"]; gold = rec["gold_origin"]
    g_lab = label_tokens(gold, rec["forced_gold"], mode)
    d = {"id": rec["id"], "title": rec["title"], "gold": gold, "lp_gold": answer_logp(g_lab), "answered": "forced_pred" in rec}
    # first token where the model would probably NOT have followed the gold answer
    d["gold_break"] = next((t for t in g_lab if t["lp"] < math.log(0.5)), None)
    if not d["answered"]:
        d.update(exact=False, origin=0.0, lp_pred=None, pred=None); return d
    norm, _ = scoring.normalize(tax, {"origin": rec["pred_origin_raw"], "contains": []})
    d["pred"] = [{"path": o["path"], "role": o["role"]} for o in norm["origin"]]
    has_prim = any(o.get("role", "primary") == "primary" for o in norm["origin"])
    d["origin"] = scoring.score_origin(tax, gold, norm["origin"]) if has_prim else 0.0
    d["exact"] = has_prim and _prim(norm["origin"]) == _prim(gold)
    d["same_answer"] = d["exact"] and _infl(norm["origin"]) == _infl(gold)
    p_lab = label_tokens(rec["pred_origin_raw"], rec["forced_pred"], mode)
    d["lp_pred"] = answer_logp(p_lab)
    # the model's own decisions along its primary path
    conf = collections.defaultdict(float)             # level j -> log P(reach level j and choose this node)
    lp_stop = lp_more = 0.0
    for t in p_lab:
        lab = t["label"]
        if lab[0] == "node" and lab[1] == 0: conf[lab[2]] += t["lp"]
        elif lab[0] == "depth" and lab[1] == 0 and lab[2] < len(rec["pred_origin_raw"][0]["path"]): conf[lab[2] + 1] += t["lp"]
        elif lab[0] == "depth" and lab[1] == 0: lp_stop += t["lp"]; d["p_stop"] = math.exp(lp_stop)
        elif lab[0] == "more" and lab[1] == 0:       # the separator can span several tokens; the first one decides, the rest are ~1
            lp_more += t["lp"]; d["p_more_or_end"] = math.exp(lp_more); d["pred_has_more"] = len(rec["pred_origin_raw"]) > 1
    d["level_conf"] = [math.exp(conf[j]) for j in sorted(conf)]
    d["raw_norm_pairs"] = (rec["pred_origin_raw"], norm["origin"])
    d["same_string"] = canonical(rec["pred_origin_raw"])[0] == canonical(gold)[0]   # same text, not just the same label (influence order, name-resolved paths)
    return d

def cmd_report(a) -> int:
    tax = Taxonomy.load()
    reason = {}
    for s in ("gold_hard", "gold"):
        f = ROOT / "data" / s / "candidates.jsonl"
        if f.exists():
            for l in open(f, encoding="utf-8"):
                c = json.loads(l); reason[c["id"]] = c.get("selection_reason", "")
    def sl(i):
        r = reason.get(i, "").split(" ")[0]
        return "fusion" if r == "R8" else "regional" if r == "R9" else "ownership" if r.startswith("R") else "all"
    m = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")
    for path in a.files:
        recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        D = [analyse(r, tax) for r in recs]
        n = len(D)
        print(f"\n{'=' * 110}\n{path}\nmodel {recs[0]['model']} · instructions: {recs[0]['mode']} · recipes {n} · greedy (temperature 0)")
        if recs[0].get("prompt_tail"): print(f"prompt ends with: {recs[0]['prompt_tail']!r}")
        ans = [d for d in D if d["answered"]]
        print(f"answered {len(ans)}/{n} · exact primary {sum(d['exact'] for d in D)}/{n} = {m([d['exact'] for d in D]):.0%} · "
              f"origin score {m([d['origin'] for d in D]):.3f} · same answer as gold incl. influences {sum(d.get('same_answer', False) for d in D)}")
        chk = [abs(d["lp_pred"] - d["lp_gold"]) for d in D if d.get("same_answer") and d.get("same_string")]
        if chk: print(f"self-check: where the answer is the same text as gold, the two forced probabilities differ by at most {max(chk):.4f} nats (should be ~0)")

        # 1. near vs blind misses
        wrong = [d for d in ans if not d.get("same_answer")]
        print(f"\n1. WAS THE GOLD ANSWER AVAILABLE?   ({len(wrong)} recipes where the answer differs from gold in primary or influences)")
        def bucket(p): return "near (P(gold) >= 10%)" if p >= 0.10 else "possible (1-10%)" if p >= 0.01 else "blind (< 1%)"
        order = ["near (P(gold) >= 10%)", "possible (1-10%)", "blind (< 1%)"]
        slices = sorted({sl(d["id"]) for d in D})
        print(f"   {'':24s}" + "".join(f"{s:>11s}" for s in ["all"] + (slices if slices != ["all"] else [])))
        for b in order:
            row = [sum(bucket(math.exp(d["lp_gold"])) == b for d in wrong)]
            if slices != ["all"]: row += [sum(bucket(math.exp(d["lp_gold"])) == b for d in wrong if sl(d["id"]) == s) for s in slices]
            print(f"   {b:24s}" + "".join(f"{x:11d}" for x in row))
        if wrong:
            print(f"   median P(gold) {statistics.median(math.exp(d['lp_gold']) for d in wrong):.3f} · median P(model's answer) {statistics.median(math.exp(d['lp_pred']) for d in wrong):.3f}")

        # 2. where the model leaves the gold answer
        print("\n2. THE DECISION WHERE THE MODEL LEFT THE GOLD ANSWER   (first gold token the model gave under 50%)")
        kinds = collections.defaultdict(list)
        for d in wrong:
            b = d["gold_break"]
            if b: kinds[describe(tuple(b["label"]), d["gold"])].append(math.exp(b["lp"]))
            else: kinds["none under 50% (greedy differs in formatting or a later tie)"].append(float("nan"))
        for k, ps in sorted(kinds.items(), key=lambda kv: -len(kv[1])):
            good = [p for p in ps if p == p]
            print(f"   {len(ps):4d}  {k:34s} P(gold choice there): median {statistics.median(good):.3f} · >=10%: {sum(p >= 0.1 for p in good)} · <1%: {sum(p < 0.01 for p in good)}" if good else f"   {len(ps):4d}  {k}")

        # 3. calibration
        print("\n3. DOES CONFIDENCE PREDICT CORRECTNESS?   (confidence = P of the model's whole origin answer)")
        edges = [(0, .5), (.5, .8), (.8, .95), (.95, .99), (.99, 1.0001)]
        print(f"   {'confidence':>14s} {'recipes':>8s} {'exact primary':>14s} {'origin score':>13s}")
        for lo, hi in edges:
            S = [d for d in ans if lo <= math.exp(d["lp_pred"]) < hi]
            if S: print(f"   {f'{lo:.2f}-{min(hi, 1):.2f}':>14s} {len(S):8d} {m([d['exact'] for d in S]):14.0%} {m([d['origin'] for d in S]):13.3f}")
        au = _auroc([d["lp_pred"] for d in ans], [bool(d["exact"]) for d in ans])
        if au is not None: print(f"   AUROC for 'primary is exact' = {au:.3f}   (0.5 = confidence tells nothing, 1.0 = it separates right from wrong perfectly)")

        # 4. influence decision
        print("\n4. THE INFLUENCE DECISION   (the token after the primary entry: finish, or add another entry)")
        for name, cond in (("gold HAS an influence", lambda d: len(d["gold"]) > 1), ("gold has NONE", lambda d: len(d["gold"]) == 1)):
            S = [d for d in ans if cond(d) and "p_more_or_end" in d]
            if not S: continue
            p_more = [d["p_more_or_end"] if d["pred_has_more"] else 1 - d["p_more_or_end"] for d in S]   # upper bound when the model finished
            print(f"   {name:22s} n={len(S):3d} · model added one on {sum(d['pred_has_more'] for d in S):3d} · P(add) median {statistics.median(p_more):.3f} · "
                  f"finished but P(add) >= 10%: {sum((not d['pred_has_more']) and p >= 0.1 for d, p in zip(S, p_more))}")

        # 5. back-off what-if
        print("\n5. WHAT-IF: STOP AT THE PARENT WHEN A DEEPER LEVEL IS UNCERTAIN   (thresholds read off this same set: descriptive, not a result)")
        print(f"   {'threshold':>10s} {'origin score':>13s} {'answers shortened':>18s} {'exact primary':>14s}")
        for tau in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
            tot, cut_n, ex = 0.0, 0, 0
            for d in D:
                if not d["answered"]: continue
                raw, norm = d["raw_norm_pairs"]; lc = d["level_conf"]
                keep = len(lc)
                rp = raw[0]; np_ = next((o for o in norm if o.get("role", "primary") == "primary"), None)
                comparable = rp.get("role", "primary") == "primary" and np_ is not None and len(rp["path"]) == len(np_["path"]) == len(lc)
                for j in range(2, len(lc) + 1) if comparable else ():
                    if lc[j - 1] < tau: keep = j - 1; break
                new = [dict(o) for o in norm]
                if keep < len(lc):
                    for o in new:
                        if o.get("role", "primary") == "primary": o["path"] = o["path"][:keep]
                    cut_n += 1
                ok = any(o.get("role", "primary") == "primary" for o in new) and tax.validate_path(tuple(_prim(new) or ())) is None
                tot += scoring.score_origin(tax, d["gold"], new) if ok else 0.0
                ex += ok and _prim(new) == _prim(d["gold"])
            print(f"   {tau:10.2f} {tot / n:13.3f} {cut_n:18d} {ex / n:14.0%}")

        # examples
        if a.examples:
            def show(o): return " + ".join((" > ".join(x["path"]) + ("" if x.get("role", "primary") == "primary" else " (infl)")) for x in o)
            for title, S in (("NEAR MISSES (highest P(gold) among wrong answers)", sorted(wrong, key=lambda d: -d["lp_gold"])[: a.examples]),
                             ("BLIND MISSES (lowest P(gold))", sorted(wrong, key=lambda d: d["lp_gold"])[: a.examples])):
                print(f"\n{title}")
                for d in S:
                    print(f"   {d['title'][:46]:46s} P(gold) {math.exp(d['lp_gold']):.3f}  P(model) {math.exp(d['lp_pred']):.3f}\n      gold : {show(d['gold'])}\n      model: {show(d['pred'])}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True, help="local model directory (mlx-lm loads HF safetensors directly)")
    r.add_argument("--data", required=True, help="gold jsonl with prompt + labels.origin (dataset_v04 or dataset_v04_origin eval files)")
    r.add_argument("--mode", required=True, choices=("full", "origin"), help="the instructions the model was TRAINED with")
    r.add_argument("--out", required=True); r.add_argument("--limit", type=int); r.add_argument("--resume", action="store_true")
    r.add_argument("--max-tokens", type=int, default=400)
    p = sub.add_parser("report"); p.add_argument("files", nargs="+"); p.add_argument("--examples", type=int, default=8)
    a = ap.parse_args(argv)
    return cmd_run(a) if a.cmd == "run" else cmd_report(a)


if __name__ == "__main__":
    raise SystemExit(main())
