from recipe_tax import Taxonomy, derive, score, score_origin, score_contains, check_evidence
from recipe_tax.scoring import gate, strict_gate, resolve_path, RESOLVE_DISCOUNT

tax = Taxonomy.load()
P = lambda *p: [{"path": list(p), "role": "primary"}]

def test_taxonomy_loads():
    assert len(tax) > 100
    assert tax.has_path(("South Asian", "Indian", "Deccan", "Hyderabadi"))
    assert tax.validate_path(("South Asian", "Indian", "Hyderabadi")) is not None   # wrong parent

def test_cross_border_countries():
    assert set(tax.countries(("South Asian", "Punjabi"))) == {"India", "Pakistan"}
    assert tax.countries(("South Asian", "Punjabi", "Lahori")) == ("Pakistan",)
    s = derive(tax, P("South Asian", "Punjabi", "Lahori"), ["poultry", "dairy"])
    assert {"punjabi", "south_asian", "country:pakistan"} <= s and "country:india" not in s

def test_hyderabadi_satisfies_indian_and_south_asian():
    s = derive(tax, P("South Asian", "Indian", "Deccan", "Hyderabadi"), ["red_meat", "dairy"])
    assert {"hyderabadi", "deccan", "indian", "south_asian", "country:india"} <= s

def test_dietary_lattice_and_free_filters():
    s = derive(tax, P("East Asian", "Japanese"), ["dairy", "egg"])
    assert "vegetarian" in s and "pescatarian" in s and "omnivore" in s and "vegan" not in s
    assert "dairy-free" not in s and "gluten-free" in s and "plant_based" not in s
    s2 = derive(tax, P("Southeast Asian", "Thai"), ["fish"])        # fish sauce trap → pescatarian, not vegetarian
    assert "pescatarian" in s2 and "vegetarian" not in s2 and "contains_seafood" in s2

def test_incomplete_makes_no_absence_claims():
    s = derive(tax, P("Global", "Contemporary"), ["dairy", "incomplete"])
    assert "unknown" in s and "incomplete" in s and not any(x.endswith("-free") for x in s)

def test_kosher_combination_rule():
    assert "kosher-ingredient-compatible" in derive(tax, P("European", "Ashkenazi Jewish"), ["poultry"])
    assert "kosher-ingredient-compatible" not in derive(tax, P("European", "Ashkenazi Jewish"), ["poultry", "dairy"])
    assert "kosher-ingredient-compatible" in derive(tax, P("European", "Ashkenazi Jewish"), ["fish", "dairy"])  # fish is pareve
    assert "halal-ingredient-compatible" not in derive(tax, P("European", "German"), ["pork"])

def test_origin_partial_credit_near_miss():
    gold = P("South Asian", "Punjabi", "Amritsari")
    exact = score_origin(tax, gold, gold)
    near = score_origin(tax, gold, P("South Asian", "Indian", "North Indian"))   # shares south_asian + country:india
    far  = score_origin(tax, gold, P("European", "French"))
    unpl = score_origin(tax, gold, P("Global", "Foundational"))
    assert exact > near > far == unpl == 0.0

def test_contains_recall_weighted():
    assert score_contains({"gluten", "tree_nuts"}, {"gluten", "tree_nuts"}) == 1.0
    missed = score_contains({"gluten", "tree_nuts"}, {"gluten"})
    extra  = score_contains({"gluten", "tree_nuts"}, {"gluten", "tree_nuts", "soy"})
    assert extra > missed                         # a miss costs more than a spurious claim
    # v2 incomplete rules: a false "incomplete" is 0 (it was a free 0.5 floor the policy learned to always claim)
    assert score_contains({"gluten"}, {"incomplete"}) == 0.0
    assert score_contains({"gluten", "incomplete"}, {"incomplete"}) == 1.0
    assert score_contains({"gluten", "incomplete"}, {"gluten"}) == 1.0           # gold incomplete, pred names the classes: fine
    # v2 spam defences
    assert score_contains({"gluten"}, ["gluten", "gluten", "gluten"]) < 1.0        # duplicates are false positives
    assert score_contains({"gluten"}, set(list(__import__("recipe_tax.scoring", fromlist=["x"]).__dict__ and ["a%d" % i for i in range(11)]))) == 0.0  # > MAX_CONTAINS

def test_evidence_rejection_and_gate():
    ing = ["2 cups basmati rice", "500 g lamb", "1/2 cup yogurt", "2 tbsp ghee"]
    gold = {"origin": P("South Asian", "Indian", "Deccan", "Hyderabadi"), "contains": ["red_meat", "dairy"]}
    # (a) a WRONG class with fabricated evidence is a false positive, not a quietly dropped claim
    pred = {"origin": P("South Asian", "Indian", "Deccan", "Hyderabadi"), "contains": ["red_meat", "dairy", "tree_nuts"],
            "evidence": {"tree_nuts": "cashews", "dairy": "yogurt"}}
    r = score(tax, gold, pred, ing)
    assert r["unsupported"] == [] and abs(r["contains"] - 2 / 3) < 1e-9
    # (b) a RIGHT class whose quote points at an unrelated line is dropped (grounding), becoming a miss
    pred2 = {"origin": P("South Asian", "Indian", "Deccan", "Hyderabadi"), "contains": ["red_meat", "dairy"],
             "evidence": {"dairy": "basmati rice", "red_meat": "lamb"}}
    r2 = score(tax, gold, pred2, ing)
    assert r2["unsupported"] == ["dairy"] and r2["contains"] < 1.0
    # (c) a right class with no evidence at all still stands (gold rows carry none)
    pred3 = {"origin": P("South Asian", "Indian", "Deccan", "Hyderabadi"), "contains": ["red_meat", "dairy"]}
    assert score(tax, gold, pred3, ing)["contains"] == 1.0
    # (d) the title can ground a claim the ingredient lines cannot
    gold_b = {"origin": P("Latin American & Caribbean", "Peruvian", "Chifa"), "contains": ["beef", "red_meat", "soy", "gluten"]}
    pred_b = {"origin": P("Latin American & Caribbean", "Peruvian", "Chifa"), "contains": ["beef", "red_meat", "soy", "gluten"],
              "evidence": {"beef": "lomo", "soy": "soy sauce", "gluten": "soy sauce"}}
    r_b = score(tax, gold_b, pred_b, ["red onion", "tomato", "soy sauce", "french fries"], title="Lomo Saltado (Beef Stir Fry)")
    assert r_b["contains"] == 1.0
    assert gate(tax, {"origin": [], "contains": []}) is not None            # no primary: structural
    assert gate(tax, {"origin": P("Mars"), "contains": []}) is None          # unknown path is NOT a gate failure any more
    assert strict_gate(tax, {"origin": P("Mars"), "contains": []}) is not None

def test_soft_gate_resolution():
    full, f = resolve_path(tax, ["Indian"]);                     assert full == ("South Asian", "Indian") and f == RESOLVE_DISCOUNT
    full, f = resolve_path(tax, ["Indian", "Hyderabadi"]);       assert full == ("South Asian", "Indian", "Deccan", "Hyderabadi")
    full, f = resolve_path(tax, ["South Asian", "Indian"]);      assert f == 1.0
    assert resolve_path(tax, ["Mars"]) == (None, 0.0)
    assert resolve_path(tax, ["Southern Thai"])[0] is not None   # unique leaf
    gold = {"origin": P("South Asian", "Indian"), "contains": ["dairy"]}
    exact = score(tax, gold, {"origin": P("South Asian", "Indian"), "contains": ["dairy"]})
    byname = score(tax, gold, {"origin": P("Indian"), "contains": ["dairy"]})
    junk = score(tax, gold, {"origin": P("Indian"), "contains": ["dairy", "spinach", "garlic cloves"]})
    assert exact["reward"] == 1.0 and byname["origin"] == RESOLVE_DISCOUNT and byname["reward"] < exact["reward"]
    assert junk["contains"] < byname["contains"] and junk["unknown_classes"] == ["spinach", "garlic_cloves"]
    assert score(tax, gold, {"origin": P("Mars"), "contains": ["dairy"]})["origin"] == 0.0   # unresolvable primary: origin 0, contains still earns


def test_uncertain_is_dont_care():
    gold, unc = {"gluten", "tree_nuts"}, {"dairy"}
    assert score_contains(gold, {"gluten", "tree_nuts", "dairy"}, unc) == 1.0     # predicting an uncertain class is free
    assert score_contains(gold, {"gluten", "tree_nuts"}, unc) == 1.0              # omitting it is free too
    assert score_contains(gold, {"gluten", "tree_nuts", "soy"}, unc) < 1.0        # a real spurious class still costs


def test_null_uncertain_and_malformed_shapes_do_not_raise():
    from recipe_tax.taxonomy import Taxonomy
    from recipe_tax.scoring import score
    tax = Taxonomy.load()
    gold = {"origin": [{"path": ["South Asian", "Punjabi"], "role": "primary"}], "contains": ["dairy"], "contains_uncertain": None}
    ok = {"origin": [{"path": ["South Asian", "Punjabi"], "role": "primary"}], "contains": ["dairy"], "evidence": {"dairy": "butter"}}
    assert score(tax, gold, ok, ["butter"])["reward"] == 1.0
    for bad in (
        {**ok, "evidence": "butter"},
        {**ok, "evidence": ["butter"]},
        {**ok, "contains": [{"class": "dairy"}]},
        {**ok, "origin": [{"path": None, "role": "primary"}]},
    ):
        r = score(tax, gold, bad, ["butter"])          # must return, never raise
        assert 0.0 <= r["reward"] <= 1.0


def test_v3_fusion_and_influence_scoring():
    tax = Taxonomy.load()
    def O(prim, *infl): return [{"path": list(prim), "role": "primary"}] + [{"path": list(i), "role": "influence"} for i in infl]
    F, KO, MX, AM, JP = ("Global", "Fusion"), ("East Asian", "Korean"), ("Latin American & Caribbean", "Mexican"), ("North American", "American"), ("East Asian", "Japanese")
    gold = O(F, KO, MX)                                    # bulgogi tacos
    full, one, bare, old = (score_origin(tax, gold, x) for x in (O(F, KO, MX), O(F, KO), O(F), O(AM, KO)))
    assert full == 1.0 and full > one > bare               # the parents are the answer; bare "Fusion" is worth little
    assert bare <= 0.35 and 0.15 < old < one               # old-convention answer keeps partial credit, below Fusion + one parent
    assert score_origin(tax, gold, O(F, MX, KO)) == 1.0    # influence order does not matter
    g2 = O(AM, JP)                                         # teriyaki burger: owner + influence
    wrong_side = score_origin(tax, g2, O(F, AM, JP))
    assert wrong_side == 0.5                               # same two parents, wrong side of the line: half credit
    assert score_origin(tax, g2, O(F, KO, MX)) < 0.2       # wrong parents get nothing from the boundary rule
    g3 = O(AM)                                             # plain American dish
    assert abs(score_origin(tax, g3, O(AM, JP)) - 0.8) < 1e-9          # spurious influence costs 0.2
    assert score_origin(tax, g3, O(F, AM, JP)) < 0.5 < score_origin(tax, g3, O(AM, JP))   # calling everything Fusion never pays
    assert score_origin(tax, g2, O(AM)) < score_origin(tax, g2, O(AM, JP)) == 1.0          # a missed influence still costs

def test_unplaced_alias_rescoring():
    from recipe_tax.scoring import resolve_path
    tax = Taxonomy.load()
    assert resolve_path(tax, ["Unplaced", "Contemporary"]) == (("Global", "Contemporary"), 1.0)


def test_origin_mode():
    """Phase 5: origin-only answers. Same origin score as full mode; contains neither required nor scored."""
    tax = Taxonomy.load()
    def O(prim, *infl): return [{"path": list(prim), "role": "primary"}] + [{"path": list(i), "role": "influence"} for i in infl]
    AM, JP = ("North American", "American"), ("East Asian", "Japanese")
    gold_full = {"origin": O(AM, JP), "contains": ["red_meat", "gluten"], "contains_uncertain": []}
    gold_copy = {"origin": O(AM, JP)}                                     # the Origin-only data copy carries origin only
    ans = {"origin": O(AM, JP)}
    assert gate(tax, ans) is not None and gate(tax, ans, "origin") is None   # full mode still demands a contains list
    assert score(tax, gold_full, ans)["reward"] == 0.0                        # ... so earlier runs are unaffected
    for g in (gold_full, gold_copy):
        r = score(tax, g, ans, ["ground beef", "soy sauce"], mode="origin")
        assert r["reward"] == r["origin"] == 1.0 and r["gate"] is None
    for pred in (O(AM), O(JP), O(("Global", "Fusion"), AM, JP)):              # reward IS score_origin, nothing else
        assert score(tax, gold_copy, {"origin": pred}, mode="origin")["reward"] == score_origin(tax, gold_copy["origin"], pred)
    with_allergens = {"origin": O(AM, JP), "contains": ["peanuts"] * 30, "evidence": {"peanuts": "x"}}
    assert score(tax, gold_copy, with_allergens, mode="origin")["reward"] == 1.0   # extra keys are ignored, never rewarded
    assert score(tax, gold_copy, {"contains": []}, mode="origin")["reward"] == 0.0
    assert score(tax, gold_copy, {"origin": O(AM) + O(JP)}, mode="origin")["gate"]   # two primaries still gate
