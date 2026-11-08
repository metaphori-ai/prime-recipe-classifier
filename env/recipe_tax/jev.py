"""Label `contains` with TypeSafe Jev (System One, Noul questions). One request per recipe, one Noul per class.

usage:
  # key: TYPESAFE_AI_JEV_API_KEY in <project>/.env (already there) or in the environment
  uv run --with pyyaml --with httpx python -m recipe_tax.jev ../data/raw/foodcom.jsonl --out ../data/silver/contains_jev.jsonl --limit 50
  uv run --with pyyaml python -m recipe_tax.jev ../data/raw/foodcom.jsonl --dry-run   # prints the questions, no network

Output: one JSON line per recipe with id, contains (p >= YES), contains_uncertain (NO < p < YES), probs (all 21),
labeled_by, labeled_at. Origin is NOT labeled here — merge with the generative origin labels by id.
Thresholds encode the safety asymmetry: a missed class is the expensive error, so YES is low and NO is strict.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from datetime import datetime, timezone
from .taxonomy import Taxonomy

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
YES, NO = 0.70, 0.25   # 0.60 gave a false 'yes' at 0.64 (gluten from a 'serve with naan' line); true positives sit ≥0.73

# Per-class question. `true`/`false` criteria carry the trap vocabulary so the boundary is explicit.
QUESTIONS: dict[str, tuple[str, str, str]] = {
  "red_meat":  ("Does this recipe contain red meat (beef, lamb, mutton, goat, pork, veal, venison) as an ingredient?",
                "Any red meat or product made from it (bacon, sausage, ham, lard, stock made from red meat).",
                "No red meat. Poultry, fish, and plant 'meat' substitutes do not count."),
  "pork":      ("Does this recipe contain pork or a pork product?",
                "Pork, bacon, ham, pancetta, prosciutto, chorizo (pork), lard, pork gelatin, pork stock.",
                "No pork-derived ingredient. Beef or chicken bacon does not count."),
  "beef":      ("Does this recipe contain beef or a beef product?",
                "Beef, veal, beef stock, beef gelatin, beef tallow, oxtail.",
                "No beef-derived ingredient."),
  "poultry":   ("Does this recipe contain poultry (chicken, turkey, duck, goose) or a poultry product?",
                "Any poultry meat, chicken stock or broth, duck fat, schmaltz.",
                "No poultry. Eggs alone do not count as poultry."),
  "fish":      ("Does this recipe contain fish or a fish-derived ingredient?",
                "Fish, fish sauce (nam pla, nuoc mam), dashi/bonito, anchovies or anchovy paste, Worcestershire sauce, Caesar dressing, fish stock, kimchi made with fish sauce.",
                "No fish-derived ingredient. Shellfish alone does not count as fish."),
  "shellfish": ("Does this recipe contain shellfish (crustaceans or mollusks) or a shellfish-derived ingredient?",
                "Shrimp, prawns, crab, lobster, clams, mussels, oysters, scallops, squid, oyster sauce, shrimp paste, belacan, XO sauce.",
                "No shellfish-derived ingredient. Finned fish does not count."),
  "egg":       ("Does this recipe contain eggs or an egg-derived ingredient?",
                "Eggs, egg whites or yolks, mayonnaise, aioli, hollandaise, meringue, fresh egg pasta, custard.",
                "No egg-derived ingredient. 'Eggless' or vegan mayonnaise does not count."),
  "dairy":     ("Does this recipe contain dairy (milk-derived) ingredients?",
                "Milk, cream, butter, ghee, cheese, paneer, yogurt, buttermilk, whey, condensed milk, sour cream, ice cream.",
                "No dairy. Coconut milk, almond milk, oat milk, and vegan butter/cheese do not count."),
  "honey":     ("Does this recipe contain honey?", "Honey in any form.", "No honey. Maple syrup, agave, sugar do not count."),
  "gelatin":   ("Does this recipe contain gelatin or an ingredient made with animal gelatin?",
                "Gelatin, gelatine sheets or powder, marshmallows, many gummy candies, aspic, some jellies.",
                "No gelatin. Agar-agar, pectin, and carrageenan do not count."),
  "blood":     ("Does this recipe contain animal blood as an ingredient?",
                "Blood, blood sausage, black pudding, morcilla, dinuguan, blood tofu, tiết canh.",
                "No blood as an ingredient."),
  "gluten":    ("Does this recipe contain gluten-containing ingredients?",
                "Wheat flour, bread, pasta, noodles (wheat), panko or breadcrumbs, soy sauce (wheat-brewed), seitan, couscous, bulgur, farro, semolina, barley, rye, beer, malt vinegar, most Worcestershire.",
                "No gluten source. Rice, corn, buckwheat, tamari labelled gluten-free, and gluten-free flours do not count."),
  "tree_nuts": ("Does this recipe contain tree nuts or any tree-nut-derived ingredient, in any form or amount?",
                "Almonds, cashews, walnuts, pistachios, pecans, hazelnuts, pine nuts, macadamia, Brazil nuts, chestnuts; marzipan, almond paste, almond extract, almond flour/meal, almond milk, frangipane, praline, orgeat, pesto (pine nuts), romesco, mole with nuts, nut butters, Nutella, amaretti. Even trace forms count — the allergy is highly sensitive.",
                "No tree nut in any form. Peanuts, coconut, nutmeg, water chestnuts, and butternut squash do not count."),
  "peanuts":   ("Does this recipe contain peanuts or a peanut-derived ingredient?",
                "Peanuts, peanut butter, peanut oil (unrefined), satay sauce, kung pao peanuts, groundnut.",
                "No peanut-derived ingredient. Tree nuts do not count."),
  "soy":       ("Does this recipe contain soy or a soy-derived ingredient?",
                "Soy sauce, tamari, miso, tofu, tempeh, edamame, natto, gochujang, doenjang, soy milk, soybean oil, textured vegetable protein, hoisin, many teriyaki sauces.",
                "No soy-derived ingredient."),
  "sesame":    ("Does this recipe contain sesame or a sesame-derived ingredient?",
                "Sesame seeds, sesame oil, tahini, halva, hummus (tahini), gomashio, za'atar (usually), dukkah, benne.",
                "No sesame-derived ingredient."),
  "alcohol":   ("Does this recipe contain an alcoholic beverage or spirit as an ingredient (even if cooked)?",
                "Wine, beer, cider, spirits, mirin, sake, shaoxing wine, cooking wine, marsala, kirsch, rum, brandy, liqueurs, beer batter.",
                "No alcoholic beverage or spirit. Flavor extracts (vanilla, almond, lemon extract), vinegar, non-alcoholic wine, and alcohol-free substitutes do NOT count — their alcohol is negligible."),
  "mustard":   ("Does this recipe contain mustard (seed, powder, or prepared)?", "Mustard seeds, mustard powder, prepared mustard, mustard oil, mustard greens.", "No mustard-derived ingredient."),
  "celery":    ("Does this recipe contain celery, celeriac, or celery seed/salt?", "Celery stalks or leaves, celeriac, celery seed, celery salt, many stock cubes.", "No celery-derived ingredient."),
  "sulfites":  ("Does this recipe contain ingredients commonly high in sulfites?", "Wine, dried apricots or other sulfured dried fruit, some vinegars, grape juice, molasses, pickled foods with sulfite preservatives.", "No notable sulfite source."),
  "incomplete":("Is the ingredient list missing or truncated, or does it use placeholders that hide what the dish contains?",
                "No ingredients given; an obviously partial list (e.g. a burger with only salt, pepper and ginger); or placeholders like 'your favorite sauce', 'assorted toppings', 'seasoning mix' that hide the contents.",
                "The ingredients are named, even without quantities or units. A short list is fine if it is plausibly the whole dish."),
}

KEY_NAMES = ("TYPESAFE_AI_JEV_API_KEY", "TYPESAFE_API_KEY")

def load_key() -> str | None:
    """Environment first, then <project root>/.env (KEY=value lines, no dependency). Never printed."""
    for n in KEY_NAMES:
        if os.environ.get(n): return os.environ[n]
    dotenv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env")
    if os.path.exists(dotenv):
        for line in open(dotenv, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line: continue
            k, v = line.split("=", 1); k, v = k.strip(), v.strip().strip('"').strip("'")
            if k in KEY_NAMES and v: return v
    return None

def build_questions(tax: Taxonomy) -> dict:
    missing = [c for c in tax.contains_classes if c not in QUESTIONS]
    if missing: raise SystemExit(f"no Jev question defined for contains classes: {missing}")
    return {cls: {"type": "noul", "instructions": q, "criteria": {"true": t, "false": f}}
            for cls, (q, t, f) in QUESTIONS.items() if cls in tax.contains_classes}

POLICY = ("Policy. (1) An ingredient listed as optional, as a garnish or topping, or as one of several alternatives "
          "(e.g. 'oil or ghee') COUNTS AS PRESENT. (2) An ingredient explicitly qualified as vegan, vegetarian, plant-based, "
          "dairy-free, gluten-free, etc. (e.g. 'vegan worcestershire sauce', 'vegetarian oyster sauce', 'gluten-free tamari') "
          "does NOT contain the class it is qualified against. (3) Missing quantities or units do not make a list incomplete; "
          "judge completeness by whether the ingredients themselves are named. "
          "(4) Judge the ingredient list below; nothing outside it is part of the dish — EXCEPT (5): if the title names an ingredient "
          "that the list omits (e.g. 'Barbecued Beef' with no beef line, 'Baked Fish' with no fish line), count it: the list is truncated.")

def recipe_state(rec: dict) -> str:
    """Title + ingredient list ONLY. The method is deliberately excluded: it carries serving suggestions and assembly
    ('serve on buns', 'wrap in a tortilla') that Jev was reading as ingredients (measured: systematic gluten FPs)."""
    ings = [l for l in rec.get("ingredients", []) if l and not l.rstrip().endswith(":")]   # drop section headers
    return f"{POLICY}\n\nTitle: {rec.get('title','')}\n\nIngredients:\n" + "\n".join(f"- {l}" for l in ings)

def call_jev(client, state: str, questions: dict, key: str, retries: int = 4) -> dict:
    body = {"model": MODEL, "state": state, "questions": questions}
    for attempt in range(retries):
        r = client.post(ENDPOINT, json=body, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, timeout=60)
        if r.status_code == 200: return r.json()
        if r.status_code in (429, 500, 502, 503, 504): time.sleep(2 ** attempt); continue
        raise SystemExit(f"Jev error {r.status_code}: {r.text[:300]}")
    raise SystemExit("Jev: retries exhausted")

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+"); ap.add_argument("--out"); ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--resume", action="store_true", help="skip ids already in --out")
    a = ap.parse_args(argv)
    tax = Taxonomy.load(); questions = build_questions(tax)
    if a.dry_run:
        print(json.dumps({"endpoint": ENDPOINT, "model": MODEL, "n_questions": len(questions), "thresholds": {"yes": YES, "no": NO}}, indent=1))
        for k, v in list(questions.items())[:3]: print(k, "→", v["instructions"])
        return 0
    key = load_key() or sys.exit("no key: set TYPESAFE_AI_JEV_API_KEY (or TYPESAFE_API_KEY) in the environment or in <project>/.env")
    import httpx
    done = set()
    if a.resume and a.out and os.path.exists(a.out):
        done = {json.loads(l)["id"] for l in open(a.out) if l.strip()}
    out = open(a.out, "a", encoding="utf-8") if a.out else sys.stdout
    n = 0; usage_in = usage_out = 0
    with httpx.Client() as client:
        for path in a.inputs:
            for line in open(path, encoding="utf-8"):
                if not line.strip(): continue
                rec = json.loads(line)
                if rec["id"] in done: continue
                if a.limit and n >= a.limit: break
                resp = call_jev(client, recipe_state(rec), questions, key)
                probs = {k: round(float(v["noul"]), 4) for k, v in resp["answers"].items()}
                u = resp.get("usage", {}); usage_in += u.get("input_tokens", 0); usage_out += u.get("output_tokens", 0)
                contains = sorted(k for k, p in probs.items() if p >= YES)
                uncertain = sorted(k for k, p in probs.items() if NO < p < YES)
                out.write(json.dumps({"id": rec["id"], "source": rec.get("source"), "labels": {"contains": contains, "contains_uncertain": uncertain},
                                      "probs": probs, "labeled_by": f"jev:{resp.get('model', MODEL)}",
                                      "labeled_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}, ensure_ascii=False) + "\n")
                out.flush(); n += 1
                if n % 25 == 0: print(f"  {n} labeled · tokens in {usage_in} out {usage_out}", file=sys.stderr)
    print(f"done: {n} recipes · tokens in {usage_in} out {usage_out} → check console for $", file=sys.stderr)
    return 0

if __name__ == "__main__":
    sys.exit(main())
