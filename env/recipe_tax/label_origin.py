"""Origin silver labeler: a teacher model answers the SAME prompt the student sees, via any OpenAI-compatible endpoint
(default: Prime Inference, key from $PRIME_API_KEY or ~/.prime/config.json after `prime login`).

  # 1. measure a candidate teacher against gold (439 eval recipes):
  uv run --with pyyaml --with httpx python -m recipe_tax.label_origin run  ../data/dataset/eval.jsonl -m qwen/qwen3.6-35b-a3b --out ../data/silver/origin_eval_qwen36.jsonl
  uv run --with pyyaml python -m recipe_tax.label_origin compare ../data/silver/origin_eval_qwen36.jsonl
  # 2. label the corpus with the winner (raw jsonl in; prompts built like the dataset builder):
  uv run --with pyyaml --with httpx python -m recipe_tax.label_origin run ../data/raw/*.jsonl -m <model> --raw --out ../data/silver/origin_silver.jsonl --resume

Writes: id, labels.origin (paths resolved to full taxonomy paths; name-resolved ones flagged), teacher_contains (for a
Jev cross-check), reply, model, labeled_by. Paths that cannot be resolved are dropped; a record with no resolvable primary is skipped.
"""
from __future__ import annotations
import argparse, json, os, sys, time, glob, collections
from datetime import datetime, timezone
from pathlib import Path
from .taxonomy import Taxonomy
from .scoring import normalize, score_origin, gate
from .build_dataset import build_prompt

PRIME_URL = "https://api.pinference.ai/api/v1/chat/completions"

def prime_config() -> dict:
    try: return json.loads((Path.home() / ".prime" / "config.json").read_text())
    except Exception: return {}

def prime_key() -> str | None:
    return os.environ.get("PRIME_API_KEY") or prime_config().get("api_key")

def prime_team() -> str | None:
    """Team to bill: $PRIME_TEAM_ID, else the `prime login` config's team_id. Without it Prime bills the personal balance."""
    return os.environ.get("PRIME_TEAM_ID") or prime_config().get("team_id")

def system_prompt() -> str:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "environments" / "recipe_classify"))
    import importlib, re
    src = (Path(__file__).resolve().parents[1] / "environments" / "recipe_classify" / "recipe_classify" / "taskset.py").read_text()
    ns = {}; exec(re.sub(r"import verifiers\.v1 as vf.*?\n", "", src.split("class RecipeData")[0]), ns)   # reuse without importing verifiers
    return ns["system_prompt"](Taxonomy.load()), ns["parse_answer"]

ANTHROPIC_NATIVE = "https://api.anthropic.com/v1/messages"

def call(client, url, key, model, sys_p, user, max_tokens, reasoning, retries=4, team: str | None = None):
    native = url.rstrip("/").endswith("/v1/messages")           # Anthropic Messages API: system block with prompt caching
    if native:
        body = {"model": model, "max_tokens": max_tokens,
                "system": [{"type": "text", "text": sys_p, "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user", "content": user}]}
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
    else:
        body = {"model": model, "messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": user}],
                "max_tokens": max_tokens}
        if "anthropic.com" not in url: body["temperature"] = 0.2      # newer Anthropic models reject the temperature field
        if reasoning: body["reasoning_effort"] = reasoning
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        if team and "pinference" in url: headers["X-Prime-Team-ID"] = team
    import httpx
    last = ""
    for a in range(max(retries, 8)):
        try:
            r = client.post(url, json=body, headers=headers, timeout=180)
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last = f"transport: {e}"; time.sleep(min(60, 3 * 2 ** a)); continue
        if r.status_code == 200:
            j = r.json(); u = j.get("usage", {})
            if native:
                text = "".join(b.get("text", "") for b in j.get("content", []) if b.get("type") == "text")
                # report cache hits so the saving is visible: input = uncached + cache_read + cache_write
                tin = u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
                call.cache_read = getattr(call, "cache_read", 0) + u.get("cache_read_input_tokens", 0)
                return text, tin, u.get("output_tokens", 0)
            return j["choices"][0]["message"].get("content") or "", u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        last = f"{r.status_code}: {r.text[:200]}"
        if r.status_code in (429, 500, 502, 503, 504, 529):
            wait = float(r.headers.get("Retry-After") or 0) or min(60, 3 * 2 ** a)
            print(f"  retry {a+1}: {last[:80]} → sleeping {wait:.0f}s", file=sys.stderr); time.sleep(wait); continue
        raise SystemExit(last)
    raise SystemExit(f"retries exhausted (last: {last}) — rerun with --resume")

def cmd_run(a):
    tax = Taxonomy.load(); sys_p, parse = system_prompt()
    key = os.environ.get(a.api_key_var) if a.api_key_var else prime_key()
    if not key and "pinference" in a.url: sys.exit("no key: `prime login` or set PRIME_API_KEY")
    team = a.team_id or prime_team()
    if "pinference" in a.url: print(f"billing: {'team ' + team if team else 'personal balance'}", file=sys.stderr)
    import httpx
    files = [f for pat in a.inputs for f in sorted(glob.glob(pat))]
    done = {json.loads(l)["id"] for l in open(a.out) if l.strip()} if (a.resume and os.path.exists(a.out)) else set()
    out = open(a.out, "a", encoding="utf-8"); n = tin = tout = skipped = 0
    with httpx.Client() as client:
        for f in files:
            for line in open(f, encoding="utf-8"):
                if not line.strip(): continue
                rec = json.loads(line)
                if rec["id"] in done: continue
                if a.limit and n >= a.limit: break
                user = build_prompt(rec) if a.raw else rec["prompt"]
                reply, i, o = call(client, a.url, key or "EMPTY", a.model, sys_p, user, a.max_tokens, a.reasoning, team=team)
                tin += i; tout += o
                pred = parse(reply)
                if pred is None or gate(tax, pred):
                    skipped += 1; out.write(json.dumps({"id": rec["id"], "error": "unparseable_or_no_primary", "reply": reply[-400:], "model": a.model}) + "\n"); out.flush(); n += 1; continue
                norm, diag = normalize(tax, pred)
                prim = [x for x in norm["origin"] if x.get("role", "primary") == "primary"]
                if not prim:
                    skipped += 1; out.write(json.dumps({"id": rec["id"], "error": "unresolvable_primary", "pred": pred, "model": a.model}) + "\n"); out.flush(); n += 1; continue
                origin = [{"path": x["path"], "role": x.get("role", "primary"), **({"resolved_by_name": True} if x.get("_factor", 1) < 1 else {})} for x in norm["origin"]]
                out.write(json.dumps({"id": rec["id"], "labels": {"origin": origin}, "teacher_contains": norm["contains"], "unknown_classes": diag["unknown_classes"],
                                      "model": a.model, "labeled_by": f"model:{a.model}", "labeled_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}, ensure_ascii=False) + "\n")
                out.flush(); n += 1
                if n % 25 == 0: print(f"  {n} · tokens in {tin} out {tout} · skipped {skipped}", file=sys.stderr)
    cr = getattr(call, "cache_read", 0)
    print(f"done: {n} · tokens in {tin} out {tout}" + (f" · of which cache-read {cr} ({100*cr/max(tin,1):.0f}%)" if cr else "") + f" · skipped {skipped} → {a.out}", file=sys.stderr)

def cmd_compare(a):
    tax = Taxonomy.load()
    gold = {json.loads(l)["id"]: json.loads(l) for l in open(Path(__file__).resolve().parents[2] / "data" / "gold" / "gold.jsonl") if l.strip()}
    rows = [json.loads(l) for l in open(a.file) if l.strip()]
    ok = [r for r in rows if "labels" in r and r["id"] in gold]
    def prim(o): return next((tuple(x["path"]) for x in o if x.get("role", "primary") == "primary"), None)
    exact = sum(prim(r["labels"]["origin"]) == prim(gold[r["id"]]["labels"]["origin"]) for r in ok)
    jacc = [score_origin(tax, gold[r["id"]]["labels"]["origin"], [{**x, "_factor": 1.0} for x in r["labels"]["origin"]]) for r in ok]
    byname = sum(any(x.get("resolved_by_name") for x in r["labels"]["origin"]) for r in ok)
    errs = collections.Counter(r.get("error") for r in rows if "error" in r)
    print(f"{a.file}\n  {len(rows)} rows · {len(ok)} scorable vs gold · errors {dict(errs)}")
    print(f"  primary exact: {exact}/{len(ok)} = {100*exact/max(len(ok),1):.1f}%   origin score (satisfaction Jaccard, infl incl.): {sum(jacc)/max(len(jacc),1):.3f}   name-resolved: {byname}")
    # influences: the number that decides whether a teacher can label fusion at all
    def infl(o): return {tuple(x["path"]) for x in o if x.get("role") == "influence"}
    g_inf = sum(len(infl(gold[r["id"]]["labels"]["origin"])) for r in ok)
    t_inf = sum(len(infl(r["labels"]["origin"])) for r in ok)
    hit = sum(len(infl(r["labels"]["origin"]) & infl(gold[r["id"]]["labels"]["origin"])) for r in ok)
    gold_fusion = [r for r in ok if infl(gold[r["id"]]["labels"]["origin"])]
    rec = sum(bool(infl(r["labels"]["origin"]) & infl(gold[r["id"]]["labels"]["origin"])) for r in gold_fusion)
    print(f"  influences: teacher {t_inf} vs gold {g_inf} · exact-path hits {hit} · recipes with gold influence {len(gold_fusion)}, teacher found one on {rec}")
    conf = collections.Counter()
    for r in ok:
        g, p = prim(gold[r["id"]]["labels"]["origin"]), prim(r["labels"]["origin"])
        if g != p: conf[(" > ".join(g), " > ".join(p))] += 1
    print("  top confusions (gold → teacher):")
    for (g, p), n in conf.most_common(a.top): print(f"    {n:3d}  {g}  →  {p}")

def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("inputs", nargs="+"); r.add_argument("-m", "--model", required=True); r.add_argument("--out", required=True)
    r.add_argument("--raw", action="store_true", help="inputs are raw recipe jsonl (build prompts); else dataset jsonl with `prompt`")
    r.add_argument("--url", default=PRIME_URL); r.add_argument("--api-key-var", default=None); r.add_argument("--limit", type=int); r.add_argument("--resume", action="store_true")
    r.add_argument("--team-id", default=None, help="Prime team to bill (default: $PRIME_TEAM_ID or ~/.prime/config.json team_id)"); r.add_argument("--max-tokens", type=int, default=1200); r.add_argument("--reasoning", default=None, help='e.g. "none" | "low" | "medium"')
    c = sub.add_parser("compare"); c.add_argument("file"); c.add_argument("--top", type=int, default=12)
    a = ap.parse_args(argv); return cmd_run(a) if a.cmd == "run" else cmd_compare(a)

if __name__ == "__main__":
    sys.exit(main())
