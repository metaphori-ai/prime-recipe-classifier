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
    s = derive(tax, P("Unplaced", "Contemporary"), ["dairy", "incomplete"])
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
    unpl = score_origin(tax, gold, P("Unplaced", "Foundational"))
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
