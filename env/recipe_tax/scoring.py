"""Reward functions. Mirrors TAXONOMY.md → Reward sketch. Every function returns a float in [0, 1]."""
from __future__ import annotations
from .taxonomy import Taxonomy

W_ORIGIN, W_CONTAINS = 0.55, 0.45
W_PRIMARY, W_INFL = 0.6, 0.4
FN_COST, FP_COST = 2.0, 1.0           # v2: FP 0.5→1.0 — at 0.5, listing every class beat an honest partial answer           # a missed class costs 4x a spurious one
INCOMPLETE_FLAT = 0.5  # v2: only when GOLD also says incomplete; a false "incomplete" scores 0 (was a free floor)
MAX_INFLUENCES = 2

def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 1.0

def _primary(origin: list[dict]) -> tuple | None:
    for o in origin:
        if o.get("role", "primary") == "primary":
            return tuple(o["path"])
    return None

def _influences(origin: list[dict]) -> list[tuple]:
    return [tuple(o["path"]) for o in origin if o.get("role") == "influence"][:MAX_INFLUENCES]

RESOLVE_DISCOUNT = 0.8
MAX_CONTAINS = 10                      # v2: more than this many classes is spam → contains = 0   # a path given by unique leaf name ("Indian") instead of its full path is accepted at this factor

def resolve_path(tax: Taxonomy, path) -> tuple[tuple | None, float]:
    """Exact path -> (path, 1.0). Otherwise, if the given sequence is the unique suffix of one node's path, resolve to it at
    RESOLVE_DISCOUNT. Unresolvable -> (None, 0.0). Soft gate: vocabulary slips cost, they don't zero the answer."""
    t = tuple(str(x) for x in (path or []))
    if not t: return None, 0.0
    if tax.validate_path(t) is None: return t, 1.0
    cands = [p for p in tax.nodes if p[-len(t):] == t] if len(t) <= 4 else []
    if len(cands) == 1: return cands[0], RESOLVE_DISCOUNT
    # last-element-only fallback (e.g. ["Indian","Hyderabadi"] mis-tiered -> try the leaf alone)
    leaf = tax.find_by_name(t[-1])
    if len(leaf) == 1: return leaf[0], RESOLVE_DISCOUNT
    return None, 0.0

def normalize(tax: Taxonomy, pred: dict) -> tuple[dict, dict]:
    """Resolve paths and drop unknown classes. Returns (normalized_pred, diagnostics)."""
    diag = {"resolved": 0, "unresolved": 0, "unknown_classes": []}
    origin = []
    for o in pred.get("origin") or []:
        if not isinstance(o, dict): continue
        full, f = resolve_path(tax, o.get("path"))
        if full is None: diag["unresolved"] += 1; continue
        if f < 1.0: diag["resolved"] += 1
        origin.append({"path": list(full), "role": o.get("role", "primary"), "_factor": f})
    contains = []
    for c in pred.get("contains") or []:
        if isinstance(c, dict): c = c.get("class") or c.get("name") or ""   # tolerate {"class": "pork", ...}
        c = str(c).strip().lower().replace(" ", "_")
        if not c: continue
        if c in tax.contains_classes: contains.append(c)
        else: diag["unknown_classes"].append(c)
    ev = pred.get("evidence")
    if not isinstance(ev, dict): ev = {}                      # a string / list here is a malformed answer, not a crash
    return {"origin": origin, "contains": contains, "evidence": ev}, diag

def gate(tax: Taxonomy, pred: dict) -> str | None:
    """None if the prediction is structurally usable, else the reason. Structural only: JSON shape and exactly one primary.
    Vocabulary (unknown classes, unresolvable paths) is handled by normalize()/scoring, not the gate."""
    if not isinstance(pred, dict): return "not an object"
    origin, contains = pred.get("origin"), pred.get("contains")
    if not isinstance(origin, list) or not isinstance(contains, list): return "origin/contains missing or wrong type"
    prim = [o for o in origin if isinstance(o, dict) and o.get("role", "primary") == "primary"]
    if len(prim) != 1: return f"expected exactly one primary origin, got {len(prim)}"
    for o in origin:
        if not isinstance(o, dict) or o.get("role", "primary") not in ("primary", "influence"): return f"bad origin entry {o!r}"[:80]
    return None

def strict_gate(tax: Taxonomy, pred: dict) -> str | None:
    """The old hard gate (exact vocabulary required). Kept for validation of gold rows and for ablations."""
    why = gate(tax, pred)
    if why: return why
    for o in pred["origin"]:
        why = tax.validate_path(o.get("path", []))
        if why: return why
    bad = [x for x in pred["contains"] if x not in tax.contains_classes]
    if bad: return f"unknown contains classes {bad}"
    return None

def score_origin(tax: Taxonomy, gold: list[dict], pred: list[dict]) -> float:
    g, p = _primary(gold), _primary(pred)
    if g is None or p is None: return 0.0
    sg = {s for s in tax.satisfaction_set(g) if not s.startswith("origin:")}
    sp = {s for s in tax.satisfaction_set(p) if not s.startswith("origin:")}
    primary = _jaccard(sg, sp)
    if len(p) == len(g) and primary > 0: primary = min(1.0, primary * 1.1)   # depth bonus
    pf = next((o.get("_factor", 1.0) for o in pred if o.get("role", "primary") == "primary"), 1.0)
    primary *= pf                                                            # resolved-by-name discount
    gi = set().union(*[tax.satisfaction_set(x) for x in _influences(gold)]) if _influences(gold) else set()
    pi = set().union(*[tax.satisfaction_set(x) for x in _influences(pred)]) if _influences(pred) else set()
    if not gi and not pi:
        return primary                      # no influences on either side: primary is the whole score.
    return W_PRIMARY * primary + W_INFL * _jaccard(gi, pi)   # spurious or missing influences cost the 0.4 term

def check_evidence(pred: dict, ingredients: list[str], gold: set[str] | None = None, traps: dict | None = None) -> set[str]:
    """v2. Return the contains classes whose claim is DROPPED for ungrounded evidence.
    A class is dropped when it is in gold (so the claim is right) but the model's quoted evidence is fabricated: the quote is
    not in the ingredient text, or it points at a line with no lexicon word for that class while a supporting line exists.
    Claims NOT in gold are never 'dropped' here — they are false positives and are charged as such in score_contains.
    Missing evidence (no quote) is not penalised (gold rows carry none). Known hole: a policy could learn to omit evidence."""
    from recipe_tax.lexicon import supporting_line
    ev = pred.get("evidence")
    if not isinstance(ev, dict) or gold is None: return set()
    text = "\n".join(ingredients).lower()
    dropped = set()
    for cls, quote in ev.items():
        if cls not in gold or not isinstance(quote, str) or not quote.strip(): continue
        q = quote.strip().lower()
        if q not in text: dropped.add(cls); continue                               # quote is not an ingredient at all
        sup = supporting_line(cls, ingredients, traps or {}, hint=q)
        if sup is None: continue                                                   # nothing in this recipe can ground it; trust gold
        if q not in sup.lower() and supporting_line(cls, [l for l in ingredients if q in l.lower()], traps or {}) is None:
            dropped.add(cls)                                                       # quote points at an unrelated line
    return dropped

def score_contains(gold: set[str], pred: list[str] | set[str], uncertain: set[str] | None = None, extra_fp: int = 0) -> float:
    """v2. `pred` may be a list (duplicates count as false positives). `uncertain` = labeler could not decide: don't-care.
    incomplete: gold&pred → INCOMPLETE_FLAT·2 capped 1.0 (=1.0); pred only → 0.0; gold only → score pred against gold's listed classes.
    > MAX_CONTAINS distinct classes → 0.0 (spam)."""
    unc = set(uncertain or ())
    pl = list(pred); ps = set(pl)
    dup_fp = len(pl) - len(ps)
    gold_inc, pred_inc = "incomplete" in gold, "incomplete" in ps
    gold, ps = (set(gold) - {"incomplete"}) - unc, (ps - {"incomplete"}) - unc
    if pred_inc and not gold_inc: return 0.0
    if pred_inc and gold_inc: return 1.0
    if len(ps) > MAX_CONTAINS: return 0.0
    tp = len(gold & ps); fn = len(gold - ps); fp = len(ps - gold) + dup_fp + extra_fp
    if tp + fn + fp == 0: return 1.0
    return tp / (tp + FN_COST * fn + FP_COST * fp)

def score(tax: Taxonomy, gold: dict, pred: dict, ingredients: list[str] | None = None, title: str | None = None) -> dict:
    empty = {"reward": 0.0, "origin": 0.0, "contains": 0.0, "unsupported": [], "resolved": 0, "unresolved": 0, "unknown_classes": []}
    why = gate(tax, pred)
    if why: return {**empty, "gate": why}
    norm, diag = normalize(tax, pred)
    if not any(o.get("role", "primary") == "primary" for o in norm["origin"]):
        o = 0.0                                               # primary path unresolvable: origin earns nothing, rest still scores
    else:
        o = score_origin(tax, gold["origin"], norm["origin"])
    lines = list(ingredients or []) + ([f"Title: {title}"] if title else [])   # the title can name the protein (policy)
    gold_c = set(gold.get("contains") or ())
    unsupported = check_evidence(norm, lines, gold=gold_c, traps=getattr(tax, "traps", {}))
    pred_contains = [c for c in norm["contains"] if c not in unsupported]          # list: duplicates stay and are charged
    c = score_contains(gold_c, pred_contains, set(gold.get("contains_uncertain") or ()),
                       extra_fp=len(diag["unknown_classes"]))
    return {"reward": W_ORIGIN * o + W_CONTAINS * c, "gate": None, "origin": o, "contains": c, "unsupported": sorted(unsupported),
            "resolved": diag["resolved"], "unresolved": diag["unresolved"], "unknown_classes": diag["unknown_classes"]}
