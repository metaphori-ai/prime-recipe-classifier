"""derive(): the ONE closure function. Positive facts in → every derived satisfaction out.

Used by (a) the verifiers rubric to score the model and (b) the app's write path to fill `satisfies`.
Rules mirror taxonomy.yaml → derived. Keep them in sync; the test suite checks the documented cases.
"""
from __future__ import annotations
from .taxonomy import Taxonomy, slug

ANIMAL   = {"red_meat", "pork", "beef", "poultry", "fish", "shellfish", "egg", "dairy", "honey", "gelatin", "blood"}
FLESH    = {"red_meat", "pork", "beef", "poultry", "fish", "shellfish", "gelatin", "blood"}
LAND     = {"red_meat", "pork", "beef", "poultry", "gelatin", "blood"}
MEAT     = {"red_meat", "pork", "beef", "poultry"}
SEAFOOD  = {"fish", "shellfish"}
FREE_OF  = {  # filter name -> class whose absence it asserts
    "gluten-free": "gluten", "dairy-free": "dairy", "egg-free": "egg", "nut-free": "tree_nuts",
    "peanut-free": "peanuts", "soy-free": "soy", "shellfish-free": "shellfish", "fish-free": "fish",
    "sesame-free": "sesame", "alcohol-free": "alcohol", "pork-free": "pork", "beef-free": "beef",
}

def dietary_pattern(contains: set[str]) -> str:
    if "incomplete" in contains: return "unknown"
    if not (contains & ANIMAL): return "vegan"
    if not (contains & FLESH):  return "vegetarian"
    if not (contains & LAND):   return "pescatarian"
    return "omnivore"

def derive(tax: Taxonomy, origin: list[dict], contains: list[str] | set[str]) -> set[str]:
    """origin: [{"path": [...], "role": "primary"|"influence"}, ...]; contains: ingredient classes.
    Returns the flat `satisfies` set (slugs) that search, agents and the rubric all run on."""
    c = set(contains)
    sat: set[str] = set()

    for o in origin:
        path = tuple(o["path"])
        if tax.validate_path(path) is not None:
            continue
        if o.get("role", "primary") == "primary":
            sat |= tax.satisfaction_set(path)
        else:
            sat |= {"influence:" + s for s in tax.satisfaction_set(path) if not s.startswith(("origin:", "country:"))}

    sat |= {"contains:" + x for x in c if x != "incomplete"}
    pat = dietary_pattern(c)
    sat.add(pat)
    if pat == "unknown":
        sat.add("incomplete")
        return sat                                   # no absence claims on an incomplete list

    # lattice: vegan satisfies vegetarian satisfies pescatarian satisfies omnivore
    order = ["vegan", "vegetarian", "pescatarian", "omnivore"]
    sat |= set(order[order.index(pat):])

    for name, cls in FREE_OF.items():
        if cls not in c: sat.add(name)
    if not ({"pork", "blood", "alcohol", "gelatin"} & c):
        sat.add("halal-ingredient-compatible")
    if not ({"pork", "shellfish", "blood", "gelatin", "alcohol"} & c) and not ("dairy" in c and (c & {"red_meat", "poultry"})):
        sat.add("kosher-ingredient-compatible")

    if c & MEAT:    sat.add("contains_meat")
    if c & SEAFOOD: sat.add("contains_seafood")
    if c & ANIMAL:  sat.add("contains_animal")
    else:           sat.add("plant_based")
    return sat
