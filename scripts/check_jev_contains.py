#!/usr/bin/env python3
"""Sanity-check a Jev contains pre-fill against its candidates file. Read-only; prints to stdout.
Flags rows where a plain keyword in the ingredient list (or title) disagrees with Jev's label, rows with no classes,
rows Jev marked incomplete, and rows whose ingredient list looks like something other than ingredients.

    uv run python scripts/check_jev_contains.py data/gold_hard/candidates.jsonl data/gold_hard/contains_jev.jsonl
"""
import json, re, sys

# keyword -> class it should imply. Deliberately conservative; qualified forms are excluded below.
KW = {
    "dairy": r"\b(butter|milk|cream|cheese|yogurt|yoghurt|ghee|paneer|buttermilk|parmesan|mozzarella|feta|ricotta|mascarpone|cheddar)\b",
    "egg": r"\beggs?\b|\byolks?\b|mayonnaise|\bmayo\b",
    "gluten": r"\b(flour|bread|breadcrumbs?|panko|pasta|noodles?|spaghetti|soy sauce|tortillas?|bulgur|couscous|semolina|barley|beer|seitan|buns?|baguette|pita)\b",
    "soy": r"\b(soy sauce|tofu|miso|tempeh|edamame|tamari|gochujang|doenjang|hoisin|soybean)\b",
    "fish": r"\b(fish sauce|anchov\w*|salmon|tuna|cod|bonito|dashi|worcestershire|sardines?|mackerel|tilapia|halibut)\b",
    "shellfish": r"\b(shrimp|prawns?|crab|lobster|clams?|mussels?|oysters?|scallops?|squid|calamari|oyster sauce|shrimp paste)\b",
    "sesame": r"\b(sesame|tahini)\b",
    "tree_nuts": r"\b(almonds?|cashews?|walnuts?|pistachios?|pecans?|hazelnuts?|pine nuts?|macadamia)\b",
    "peanuts": r"\bpeanuts?\b",
    "pork": r"\b(pork|bacon|ham|pancetta|prosciutto|lard|chorizo|sausage)\b",
    "beef": r"\b(beef|veal|oxtail|brisket|steak)\b",
    "poultry": r"\b(chicken|turkey|duck)\b",
    "alcohol": r"\b(wine|beer|sake|mirin|rum|brandy|vodka|whiskey|bourbon|sherry|marsala|vermouth)\b",
    "honey": r"\bhoney\b",
    "mustard": r"\bmustard\b",
    "celery": r"\bceler(y|iac)\b",
}
# a line carrying one of these qualifiers is not counted as evidence FOR the class
QUALIFIED = re.compile(r"vegan|vegetarian|plant.based|dairy.free|gluten.free|egg.free|non.dairy|nondairy|coconut (milk|cream|yogurt)|"
                       r"(almond|oat|soy|rice|cashew|nut) milk|peanut butter|nut butter|rice (flour|noodles?|wine vinegar|vinegar)|"
                       r"(corn|almond|coconut|chickpea|tapioca|potato|cassava|buckwheat|oat) (flour|tortillas?|starch)|corn tortillas?|"
                       r"cream of tartar|butter ?(beans?|nut|lettuce)|wine vinegar|vinegar|cocoa butter|shea butter|"
                       r"(beef|chicken|turkey|veggie|vegetable) (bacon|sausage|ham)|tamari|non.alcoholic|eggplant", re.I)
EQUIP = re.compile(r"\b(pan|pot|skillet|bowl|whisk|spatula|oven|blender|processor|knife|board|sheet|rack|thermometer|mixer|spoon|ladle|"
                   r"colander|strainer|grater|peeler|tongs|dutch oven|saucepan|stockpot|baking dish)\b", re.I)
QTY = re.compile(r"\d|½|¼|¾|⅓|⅔|\b(cup|tbsp|tsp|tablespoon|teaspoon|gram|ounce|oz|lb|pound|pinch|clove|bunch|can)\b", re.I)


def main(cand_path, jev_path):
    cand = {json.loads(l)["id"]: json.loads(l) for l in open(cand_path, encoding="utf-8") if l.strip()}
    jev = {json.loads(l)["id"]: json.loads(l) for l in open(jev_path, encoding="utf-8") if l.strip()}
    print(f"candidates {len(cand)} · jev rows {len(jev)}\n")

    def head(rid):
        c, j = cand[rid], jev[rid]
        return f"{rid}  «{c['title'][:70]}»  ings={len(c['ingredients'])}  contains={j['labels']['contains']}  unc={j['labels']['contains_uncertain']}"

    print("== A. no class at all ==")
    for rid, j in jev.items():
        if not j["labels"]["contains"]:
            print(" ", head(rid)); print("     ", " | ".join(cand[rid]["ingredients"])[:420])

    print("\n== B. Jev says incomplete (yes or uncertain) ==")
    for rid, j in jev.items():
        p = j["probs"].get("incomplete", 0)
        if p > 0.25:
            print(f"  incomplete p={p:.2f}  {head(rid)}"); print("     ", " | ".join(cand[rid]["ingredients"])[:300])

    print("\n== C. ingredient list looks like equipment / not ingredients ==")
    for rid, c in cand.items():
        ings = c["ingredients"]
        eq = sum(1 for l in ings if EQUIP.search(l) and not QTY.search(l))
        noq = sum(1 for l in ings if not QTY.search(l))
        if eq >= max(2, len(ings) // 3) or "borscht" in c["title"].lower():
            j = jev[rid]
            print(f"  equip-like lines {eq}/{len(ings)}, no-quantity lines {noq}  {head(rid)}")
            print("     ", " | ".join(ings)[:500]); print("      incomplete p=", j["probs"].get("incomplete"))

    print("\n== D. keyword in ingredients but Jev p <= 0.25 (possible MISS; the dangerous direction) ==")
    n = 0
    for rid, c in cand.items():
        j = jev[rid]
        for cls, pat in KW.items():
            hits = [l for l in c["ingredients"] if re.search(pat, l, re.I) and not QUALIFIED.search(l)]
            if hits and j["probs"].get(cls, 0) <= 0.25:
                n += 1; print(f"  {cls:10s} p={j['probs'].get(cls):.2f}  {rid} «{c['title'][:50]}»  ← {hits[:3]}")
    print(f"  ({n} flags)")

    print("\n== E. Jev says yes (p >= 0.70) but no keyword anywhere in title+ingredients (possible FALSE POSITIVE; review) ==")
    n = 0
    for rid, c in cand.items():
        j = jev[rid]; text = c["title"] + " | " + " | ".join(c["ingredients"])
        for cls, pat in KW.items():
            if j["probs"].get(cls, 0) >= 0.70 and not re.search(pat, text, re.I):
                n += 1; print(f"  {cls:10s} p={j['probs'].get(cls):.2f}  {rid} «{c['title'][:50]}»")
    print(f"  ({n} flags — many are legitimate derived cases: stock, sauces, red_meat from pork/beef)")

    print("\n== F. title names a class Jev did not mark (policy rule 5: title implies a truncated list) ==")
    for rid, c in cand.items():
        j = jev[rid]
        for cls, pat in KW.items():
            if re.search(pat, c["title"], re.I) and not QUALIFIED.search(c["title"]) and j["probs"].get(cls, 0) < 0.70:
                print(f"  {cls:10s} p={j['probs'].get(cls):.2f}  {rid} «{c['title'][:70]}»")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
