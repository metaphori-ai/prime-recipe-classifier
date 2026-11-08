"""recipe-classify: origin tradition path + ingredient-class `contains` for a recipe (single-turn).

The model sees a title and ingredient names and must emit JSON inside <answer> tags:
  {"origin": [{"path": [...], "role": "primary"}, {"path": [...], "role": "influence"}],
   "contains": [...], "evidence": {"class": "ingredient substring"}}

Reward = recipe_tax.scoring.score(): structural gate -> 0; origin via satisfaction-set Jaccard (hierarchical partial
credit); contains recall-weighted (a missed class costs 4x a spurious one); evidence substring-checked against the
ingredient lines; `contains_uncertain` classes are don't-care. Metrics expose each component and the gate reason.
"""
from __future__ import annotations
import json, re
from pathlib import Path
from typing import Any

import verifiers.v1 as vf
from recipe_tax import Taxonomy, score
from recipe_tax.scoring import gate, strict_gate

_ANS = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_JSON = re.compile(r"\{.*\}", re.DOTALL)


def _json_objects(body: str):
    """Yield every JSON object that parses from a '{' in body, last first."""
    dec = json.JSONDecoder()
    starts = [i for i, ch in enumerate(body) if ch == "{"]
    for i in reversed(starts):
        try:
            obj, _ = dec.raw_decode(body[i:])
            if isinstance(obj, dict): yield obj
        except json.JSONDecodeError:
            continue


def parse_answer(text: str) -> dict | None:
    """Prefer the LAST <answer> block (models draft an answer while thinking, then write the real one), and within it the
    outermost JSON object that has an 'origin' key. Falls back to any such object anywhere in the text."""
    blocks = _ANS.findall(text or "")
    for body in reversed(blocks) if blocks else [text or ""]:
        best = None
        for obj in _json_objects(body):
            if "origin" in obj: best = obj          # last-first iteration: keep the earliest start = outermost object
        if best is not None: return best
    for obj in _json_objects(text or ""):
        if "origin" in obj: return obj
    return None


def taxonomy_block(tax: Taxonomy) -> str:
    lines = []
    for p, n in tax.nodes.items():
        c = f"  [{', '.join(n.countries)}]" if n.countries and len(p) > 1 else ""
        lines.append("  " * (len(p) - 1) + p[-1] + c)
    return "\n".join(lines)


def system_prompt(tax: Taxonomy) -> str:
    classes = [c for c in tax.contains_classes]
    return f"""You classify recipes against a fixed taxonomy. Reply with ONLY a JSON object inside <answer></answer> tags.

ORIGIN. Give exactly one "primary" path: the culinary tradition that owns the dish's FORM and technique. Optionally up to two
"influence" paths for a borrowed FLAVOR SYSTEM (e.g. a burger with bulgogi marinade: primary American, influence Korean).
Named fusion traditions (Anglo-Indian, Tex-Mex, Chifa, Nikkei, Indo-Chinese, American Chinese, Italian-American, Cape Malay) are
their own nodes — use the node, not an influence pair. Stop at the deepest node the recipe SUPPORTS; do not guess a region from one
ingredient. Pan-national restaurant idiom -> the national node.
CONVENTIONS (follow these over your own instinct):
- HOST TABLE, not source culture. A dish is owned by the tradition at whose table it is eaten now. Americanized dishes are
  "North American" > "American": Bisquick, canned soup, cream cheese, Jell-O, "Italian seasoning", garlic powder, Cool Whip,
  crock pot, copycat chain recipes, generic casseroles, muffins, cookies, bars, dips, burgers, sloppy joes, BBQ sauce, coleslaw.
  "Italian seasoning"/cream pasta/chicken parm/marinara/pizza bombs -> "Italian-American", NOT "Italian". Teriyaki chicken with
  honey and cornstarch -> "American" with influence "Japanese". A roux-and-curry-powder dish -> "Anglo-Indian", not "Indian".
- "Unplaced" > "Contemporary" ONLY when no tradition owns the FORM at all: grain/protein bowls, smoothies, mug cakes, energy bars,
  keto/vegan substitutes. An ordinary American home recipe is "American", never "Contemporary". Contemporary is about the FORM,
  not the country: gluten-free/keto/vegan substitute bakes, microwave and mug meals, grain or protein bowls, smoothies and
  overnight oats are "Contemporary" even when every ingredient is American pantry. Dips, casseroles, sheet cakes, slow-cooker
  dishes are "American" however modern their ingredients. "Unplaced" > "Foundational" =
  bare building blocks (plain rice, fried eggs, stock).
- Tandoori chicken, butter chicken (murgh makhani), chole/chhole, aloo gobi, bhindi masala, roti -> "South Asian" > "Punjabi".
  Dal fry, korma, malai kofta, saag aloo -> "South Asian" > "Indian" > "North Indian". Biryani/haleem/double ka meetha ->
  "Indian" > "Deccan" > "Hyderabadi". Kedgeree, mulligatawny, tikka masala, balti, "Madras" curry -> "British" > "Anglo-Indian".
  Lomo saltado, arroz chaufa, tallarín saltado (soy sauce + wok in Peru) -> "Peruvian" > "Chifa"; tiradito, Nikkei ceviche ->
  "Peruvian" > "Nikkei"; a ceviche with tomato, cucumber, avocado or Tabasco is "Mexican", not Peruvian. Bobotie, sosaties ->
  "Southern African" > "Cape Malay"; bunny chow, Durban curry -> "Southern African" > "Durban Indian".
- Give the FULL path from the root, e.g. ["South Asian", "Indian", "South Indian", "Kerala"], never a bare leaf.
Every path must be an exact sequence of node names from this tree (indentation = depth):
{taxonomy_block(tax)}

CONTAINS. List every ingredient class present. Positive facts only; absence is never stated. Classes:
{", ".join(classes)}
Rules: optional / garnish / "A or B" alternatives COUNT as present. Explicit vegan/vegetarian/X-free qualifiers NEGATE the class
("vegan worcestershire" is not fish). Soy sauce = gluten AND soy unless tamari/gluten-free. Nutmeg is not a nut; coconut is not a nut;
flavor extracts are not alcohol; cooking wine, mirin, sake, beer ARE alcohol. An ingredient named in the TITLE counts even if the list
omits it. If the ingredient list is missing or hides contents behind placeholders, include "incomplete". For each class you list,
give one "evidence" string copied from an ingredient line that justifies it.

Output format:
<answer>{{"origin": [{{"path": ["European", "British", "Anglo-Indian"], "role": "primary"}}], "contains": ["poultry", "gluten", "dairy"],
"evidence": {{"poultry": "chicken", "gluten": "flour", "dairy": "butter"}}}}</answer>"""


class RecipeData(vf.TaskData):
    recipe_id: str
    ingredients: list[str]
    gold: dict[str, Any]
    """{"origin": [...], "contains": [...], "contains_uncertain": [...]}"""


class RecipeClassifyTask(vf.Task[RecipeData]):
    _tax: Taxonomy | None = None

    @property
    def tax(self) -> Taxonomy:
        if RecipeClassifyTask._tax is None:
            RecipeClassifyTask._tax = Taxonomy.load(self.config.taxonomy_path) if getattr(self.config, "taxonomy_path", None) else Taxonomy.load()
        return RecipeClassifyTask._tax

    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 1

    async def validate(self, runtime: vf.Runtime) -> bool:
        """Model-free check: the gold labels, emitted as an answer, must score exactly 1.0, and an empty
        answer must gate to 0. Catches gold rows the taxonomy can't validate (bad path, unknown class)."""
        g = self.data.gold
        gold_answer = {"origin": g["origin"], "contains": g["contains"]}
        r = score(self.tax, g, gold_answer, self.data.ingredients)
        if strict_gate(self.tax, gold_answer) or r["gate"] or abs(r["reward"] - 1.0) > 1e-9:
            raise RuntimeError(f"gold does not score 1.0: {r}")
        nothing = score(self.tax, g, parse_answer("no answer") or {}, self.data.ingredients)
        return nothing["reward"] == 0.0

    def _scored(self, trace: vf.Trace) -> dict:
        if "scored" not in trace.info:
            try:
                pred = parse_answer(trace.last_reply or "")     # last_reply is None on an empty completion
            except Exception:
                pred = None
            if pred is None:
                trace.info["scored"] = {"reward": 0.0, "gate": "unparseable", "origin": 0.0, "contains": 0.0, "unsupported": [], "resolved": 0, "unresolved": 0, "unknown_classes": []}
            else:
                trace.info["pred"] = pred
                try:
                    title = self.data.prompt.split("\n", 1)[0].removeprefix("Title:").strip() if self.data.prompt.startswith("Title:") else None
                    trace.info["scored"] = score(self.tax, self.data.gold, pred, self.data.ingredients, title=title)
                except Exception as e:  # malformed-but-parseable JSON (wrong shapes). Treat as a gate failure, never kill the episode.
                    trace.info["scored"] = {"reward": 0.0, "gate": f"malformed: {type(e).__name__}", "origin": 0.0, "contains": 0.0,
                                            "unsupported": [], "resolved": 0, "unresolved": 0, "unknown_classes": []}
        return trace.info["scored"]

    @vf.reward(weight=1.0)
    async def total(self, trace: vf.Trace) -> float:
        return float(self._scored(trace)["reward"])

    @vf.metric
    async def origin(self, trace: vf.Trace) -> float:
        return float(self._scored(trace)["origin"])

    @vf.metric
    async def contains(self, trace: vf.Trace) -> float:
        return float(self._scored(trace)["contains"])

    @vf.metric
    async def gate_fail(self, trace: vf.Trace) -> float:
        return 1.0 if self._scored(trace)["gate"] else 0.0

    @vf.metric
    async def unsupported_evidence(self, trace: vf.Trace) -> float:
        return float(len(self._scored(trace)["unsupported"]))

    @vf.metric
    async def resolved_paths(self, trace: vf.Trace) -> float:
        return float(self._scored(trace).get("resolved", 0))

    @vf.metric
    async def unknown_classes(self, trace: vf.Trace) -> float:
        return float(len(self._scored(trace).get("unknown_classes", [])))

    @vf.metric
    async def response_chars(self, trace: vf.Trace) -> float:
        return float(len(trace.last_reply or ""))

    @vf.metric
    async def evidence_coverage(self, trace: vf.Trace) -> float:
        """Share of contains claims that carry a non-empty evidence quote. Reward v2 does not penalise missing evidence
        (gold carries none); if this falls while contains rises, the policy has found that hole."""
        self._scored(trace)
        pred = trace.info.get("pred") or {}
        cl = pred.get("contains") if isinstance(pred.get("contains"), list) else []
        ev = pred.get("evidence") if isinstance(pred.get("evidence"), dict) else {}
        if not cl: return 1.0
        return sum(1 for c in cl if isinstance(c, str) and isinstance(ev.get(c), str) and ev.get(c).strip()) / len(cl)

    @vf.metric
    async def influence_pred(self, trace: vf.Trace) -> float:
        """Influence tags the policy emitted (gold has ~0.07/recipe; SFT-from-silver emits ~0.01, base ~0.26)."""
        self._scored(trace)
        pred = trace.info.get("pred") or {}
        origin = pred.get("origin") if isinstance(pred.get("origin"), list) else []
        return float(sum(1 for o in origin if isinstance(o, dict) and o.get("role") == "influence"))

    @vf.metric
    async def influence_gold(self, trace: vf.Trace) -> float:
        """Influence tags in gold for this recipe, so the two curves can be compared per step."""
        return float(sum(1 for o in (self.data.gold.get("origin") or []) if isinstance(o, dict) and o.get("role") == "influence"))

    @vf.metric
    async def primary_depth(self, trace: vf.Trace) -> float:
        """Depth of the predicted primary path (gold mean ≈ 2.6); tracks depth under-commitment during RL."""
        self._scored(trace)
        pred = trace.info.get("pred") or {}
        origin = pred.get("origin") if isinstance(pred.get("origin"), list) else []
        prim = [o for o in origin if isinstance(o, dict) and o.get("role", "primary") == "primary"]
        return float(len(prim[0].get("path", []))) if prim and isinstance(prim[0].get("path"), list) else 0.0


class RecipeClassifyConfig(vf.TasksetConfig):
    dataset_path: str = str(Path(__file__).resolve().parents[4] / "data" / "dataset" / "eval.jsonl")
    """JSONL from recipe_tax.build_dataset (eval.jsonl for evaluation, train.jsonl for RL)."""
    taxonomy_path: str | None = None
    """Override taxonomy.yaml location (defaults to the project root copy)."""
    publishable_only: bool = False
    """Keep only records whose source license allows redistribution (for Hub datasets)."""


class RecipeClassifyTaskset(vf.Taskset[RecipeClassifyTask, RecipeClassifyConfig]):
    def load(self) -> list[RecipeClassifyTask]:
        tax = Taxonomy.load(self.config.taxonomy_path) if self.config.taxonomy_path else Taxonomy.load()
        sys_prompt = system_prompt(tax)
        tasks = []
        with open(self.config.dataset_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                r = json.loads(line)
                if self.config.publishable_only and not r.get("publishable"): continue
                tasks.append(RecipeClassifyTask(
                    RecipeData(id=r["id"], recipe_id=r["id"], prompt=r["prompt"], system_prompt=sys_prompt,
                               ingredients=r["ingredients"], gold=r["labels"]),
                    self.config.task))
        return tasks
