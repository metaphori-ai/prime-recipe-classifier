# Taxonomy design notes — v0.3 (positive space · tradition spine)

## Governing principle: positive space in, negative space out

**Store the most specific positive fact; derive coarser satisfactions by lattice.**

| | Positive space (property) | Negative space (filter) |
|---|---|---|
| truth condition | existential — point at the line | universal — requires exhaustive scan |
| who produces it | the model | the database, at query time |
| hallucination risk | can be wrong | can't be *fabricated* — only derived |
| example | `contains: [tree_nuts]` | `nut-free` ⇔ tree_nuts ∉ contains |
| example | `origin: Asia > India > Hyderabadi` | satisfies "Indian", "Asian" |
| example | `contains: [fish, dairy]` | pattern = pescatarian (and so satisfies omnivore) |

Vegetarian ⊂ pescatarian ⊂ omnivore is a subset order on *what the dish contains*; so
"pescatarian" is something a dish **satisfies**, not something it **is**. Properties get stored,
filters get computed. Same shape for origin: Hyderabadi *is*; Indian/Asian are *satisfied*.

## Facets in env #1

| Facet | Cardinality | Reward nature | Why |
|---|---|---|---|
| `origin` | multi (paths + role) | hierarchical overlap, strict on primary / lenient on influence | the judgment facet |
| `contains` | multi (ingredient classes) | rule verifier vs ingredient list; **recall-weighted** | the safety facet; model can't hallucinate absence |

`dietary_pattern` and all `*-free` restrictions are **derived**, never emitted. See `taxonomy.yaml → derived`.

## Origin spine: tradition, not geography (v0.3)

Two questions forced this: *where does "North Indian" go?* and *what is Punjabi — Indian or Pakistani?*
Both come from building the tree on political borders. Food is organized by **tradition**; borders are a
gradient laid over it. So:

- **The spine is culinary tradition.** `South Asian > Punjabi > Lahori`. Countries are **metadata** on
  nodes (`_c: [India, Pakistan]`), and "Indian", "Pakistani", "South Asian" are **derived satisfactions** —
  the same move as `pescatarian` being derived from `contains`.
- **Cross-border traditions are first-class nodes**, siblings of the national cuisines they straddle:
  Punjabi, Bengali, Basque, Levantine, Kurdish, Malay/Nyonya, Swahili Coast, Ethiopian & Eritrean.
  A Lahori dish satisfies Punjabi, Pakistani, South Asian — and *not* Indian. Honest.
- **National cuisines remain as nodes** for dishes that are pan-national idiom with no regional identity
  ("Indian restaurant food"). They are siblings of the cross-border traditions, not parents.
- **Optional macro-region tier** (North/South/Deccan Indian, Northern/Southern Italian, …) only where
  it carries real information: Indian, Chinese, Italian, Mexican, American. Depth is variable; scoring handles it.
- **One parent per node, by convention, written down.** Hyderabadi → Deccan (not South). Uyghur →
  Central Asian (tradition wins over country). Gold labelers follow the file.
- **Register** (home / street / court / temple / diaspora) is a separate facet, deferred. Not origin nodes.

### Scoring change

With country metadata and cross-border nodes, strict path-prefix overlap under-rewards near-misses
(predicting `Indian > North Indian` for a `Punjabi > Amritsari` dish shares no prefix but shares
*India* and *South Asian*). So origin is scored on **derived satisfaction sets**:

```
sat(node) = ancestors(node) ∪ countries(node) ∪ {node}
origin_primary = jaccard(sat(gold), sat(pred))       # 1.0 exact; graded for near-misses
                 × depth_bonus                         # small multiplier when pred depth == gold depth
```
This also makes the macro-region tier a genuine partial-credit stop rather than a penalty.

## Named diaspora / fusion traditions (v0.3.3)

Rule: if the fusion has a **name and a canon**, it is a node, not an influence pair. File it under the **host**
tradition — the table it is eaten at — with the source culture in country metadata.

| node | parent | c | why host = parent |
|---|---|---|---|
| Anglo-Indian | British | UK, India | kedgeree is a British breakfast; curry powder is a British invention |
| Indo-Chinese | Indian | India | invented and eaten in India (Kolkata Hakka); Indian table |
| American Chinese | American | USA, China | US restaurant tradition; General Tso's does not exist in China |
| Italian-American | American | USA, Italy | red-sauce canon is American |
| Nikkei / Chifa | Peruvian | Peru | Peruvian regional traditions |
| Tex-Mex | American | USA, Mexico | |
| Cape Malay / Durban Indian | Southern African | South Africa | |

Threshold applied: ≥ 20 corpus hits by title/tag (Anglo-Indian 39, Indo-Chinese 20, American Chinese 20,
Italian-American 16 — admitted as under-detected by keywords). Not added: Indo-Caribbean (1), Korean-American /
Korean-Mexican (12 — stays as influence-pair fusion for now; revisit if SaffronSifu's own corpus grows it).

Worked example: **Kedgeree** → `European > British > Anglo-Indian`, no influence needed. Not `Indian` (the technique is
a British roux casserole), and not `British + influence Indian` (the pairing has a name).

## Fusion

Fusion is not a tag. It is the presence of ≥1 `influence` path alongside the `primary`.

- `primary` = the dish's lineage / technique / form (what is this *thing*)
- `influence` = flavor system or ingredient vocabulary borrowed (what does it *taste like*)

Gochujang ribs → primary `Americas > USA > Southern`, influence `Asia > Korea`.
Nikkei ceviche → primary `Americas > Peru > Nikkei` (the fusion is itself a named regional tradition — no influence needed).
Korean-adjacent home cooking with no clear primary → primary `Asia > Korea`, no influence, and the
eval set should contain enough of these to find out whether the model over-reaches for a region.

### Worked example: Spicy Korean Chicken Katsu (saffronsifu.app/r/68DmjO0Spt27WOzyaOzztP)

Current app tags: `chicken katsu, gochujang, korean, breaded chicken` — right words, no structure.

```json
"origin": [
  {"path": ["East Asian", "Japanese"], "role": "primary"},    // katsu = the FORM (breaded fried cutlet)
  {"path": ["East Asian", "Korean"],   "role": "influence"}   // gochujang sauce = the FLAVOR SYSTEM
],
"contains": ["poultry", "gluten", "egg", "soy", "sesame"],     // confirm against ingredient list; + alcohol if mirin
"evidence": {"gluten": "panko", "soy": "gochujang", "egg": "egg"}
```
Why not primary Korean: nothing in the technique is Korean. Why not two primaries: the roles say what each
tradition contributes; a 50/50 pair throws that away. Why Japanese and not "European cutlet → Japanese →
Korean": we tag at the tradition that owns the dish *now*; traditions absorb, and etymology is not a tag.
Derived consequence worth noticing: this recipe correctly disappears under a `soy-free` filter even though
"soy" appears nowhere in the title — that is the `ingredient_traps` list doing its job.

Exploits to expect and tune:
- **under-commit**: always stop at country, never risk region → fix by rewarding depth when gold has depth
- **over-commit**: confabulate a region from one ingredient → fix by penalizing wrong region more than missing region
- **influence spam**: list every plausible influence → fix with per-influence cost and a cap (max 2)
- **Unplaced dumping**: tag hard cases `Unplaced > Foundational/Contemporary` → already scores 0 vs. partial credit for a near-miss; monitor rate, don't add penalty

## Output schema (what the model emits)

```json
{
  "origin": [
    {"path": ["South Asian", "Indian", "Deccan", "Hyderabadi"], "role": "primary"},
    {"path": ["European", "British"], "role": "influence"}
  ],
  "contains": ["dairy", "gluten", "tree_nuts"],
  "evidence": {"tree_nuts": "cashews", "gluten": "naan"},
  "confidence": {"origin": 0.8, "contains": 0.95}
}
```

**`evidence` — kept, and load-bearing.** The verifier checks that every evidence string is a substring of
some ingredient line (case-insensitive). A `contains` claim whose evidence does not appear in the recipe
is rejected. This closes the last way the model could fabricate a positive.

**`confidence` — kept in the schema, NOT in the reward (v0.1).** Three uses, in order of adoption:
1. App routing: ≥0.85 auto-apply, below → human review queue.
2. Calibration diagnostic on the eval set (does 0.8 mean right 80 % of the time?). Logged every run.
3. Later experiment: fold into reward via a proper scoring rule (Brier) to price overconfident region-tier
   guesses. Not before the loop is trusted — it opens a "always say 0.5" hack that needs its own run to study.

## Reward sketch (per recipe, 0–1)

```
gate      = schema_valid AND all_paths_in_taxonomy AND no_contradictions AND contains ⊆ vocab   # 0 if fails
origin    = 0.6 * jaccard(sat(primary_gold), sat(primary_pred))   # satisfaction-set overlap, see above
          + 0.4 * set_hier_f1(influences_gold, influences_pred)   # lenient, capped at 2 influences
contains  = recall-weighted:  TP / (TP + 2·FN + 0.5·FP)           # a missed class costs 4× a spurious one
            if "incomplete" emitted: contains score = 0.5 flat (honest abstention, never rewarded above effort)
reward    = gate * (0.55 * origin + 0.45 * contains)
```
Weights are a first guess. Exploits to expect on `contains`:
- **over-claim**: list every class to max recall → FP cost + the 0.5·FP term; tune until it stops
- **incomplete-dumping**: emit "incomplete" on every hard case → the 0.5 flat makes it a floor, not a strategy
- **trap-blindness**: miss soy sauce → gluten, fish sauce → fish; the `ingredient_traps` list is the verifier's
  lookup AND the eval set's hard slice, so this shows up as a recall curve on the trap subset specifically

## Dataset plan

- Source: SaffronSifu captured recipes (own data) + optionally a public set for breadth.
- **Gold eval set**: 300–500 recipes, human-verified, deliberately over-weighted toward
  fusion, diaspora, regional specificity, and ingredient traps. This is sacred; never train on it.
- **Silver train set**: 2–5k recipes labeled by a strong model against this taxonomy,
  spot-checked, noise tolerated.
- Both stored as JSONL: `{"id", "title", "ingredients": [...], "instructions": "...", "labels": {...}}`

## Open design questions

- ~~Caribbean / Scandinavia / Balkans at country tier~~ → resolved by tradition spine (multi-country metadata).
- Which 5–6 traditions get the optional macro-region tier? Current: Indian, Chinese, Italian, Mexican, American.
- ~~`Global > None`~~ → split into `Unplaced > Foundational` (building blocks) and `Unplaced > Contemporary` (modern, no parent). Both legal primaries.
- ~~`confidence`~~ → in schema for routing + calibration logging; out of reward until a dedicated run.
- ~~kosher / halal~~ → `kosher-ingredient-compatible` / `halal-ingredient-compatible`; derived only.
  "Ingredient" covers species exclusions AND the kosher meat/dairy combination rule (a rule over the
  ingredient set). What it cannot cover: ritual slaughter, equipment separation, rennet hidden in cheese,
  certification. UI copy: "no categorically excluded ingredients or combinations; sourcing not verified".
  Known gap: the `fish` class does not distinguish finned/scaled fish from eel, catfish, squid.
- ~~`evidence`~~ → kept; verifier substring-checks it against ingredient lines.
- Granularity of `contains`: is `red_meat` + `pork` + `beef` the right split, or go per-animal?

## Derived umbrellas (v0.3.1)

`contains` stays per-class. The closure additionally derives `contains_meat`, `contains_seafood`,
`contains_animal`, `plant_based`, and the dietary pattern — all land in the `satisfies` array. Fast search
and agent restriction both run on that array:

```
recipe_scout filter spec:  require: [vegetarian]   exclude: [tree_nuts, sesame]
→ SQL:  satisfies @> '{vegetarian}'  AND NOT (contains && '{tree_nuts,sesame}')
```
Same `derive()` enforces it for agents and fills the column for search.

## Labeling policies (from the Jev probe, 2026-11-07)

- **Optionals and alternatives count as present.** "Oil or ghee" → `dairy`. Optional cheese topping → `dairy`.
  Conservative on purpose: a derived `dairy-free` must not surface a recipe that *might* contain dairy. Applies to
  Jev criteria, the silver origin labeler, and human gold labeling alike.
- **Explicit vegan/vegetarian/X-free qualifiers negate the class.** "vegan worcestershire sauce" → not `fish`;
  "vegetarian oyster sauce" → not `shellfish`; "gluten-free tamari" → `soy` but not `gluten`. (Jev hedged 0.26–0.65
  on these before the rule was stated; they are prime gold hard-cases.)
- **`alcohol` = alcoholic beverages and spirits used as ingredients** (wine, beer, mirin, sake, spirits, liqueurs), cooked or not —
  cooking does not remove it (≈50 % remains after 25 min baking). **Flavor extracts do not count**: a teaspoon of vanilla extract
  leaves ~0.1–0.3 % in the finished dish, below ripe fruit. Strict-halal users may want an `extract_alcohol` class later.
- **Tree nuts count in every form and amount** — marzipan, almond extract, almond flour, frangipane, orgeat, pesto: the allergy is
  highly sensitive. **Nutmeg is not a nut.** And the claim is about *this* ingredient list, not the canonical dish: a bobotie
  without almonds listed is not `tree_nuts` even though bobotie usually has them. (6 gold rows corrected 2026-11-07.)
- **Generic vegetable oil / margarine / shortening ≠ `soy`** (policy 6, for the NEXT Jev run). FDA exempts highly refined
  soybean oil from allergen labeling (protein removed). Only *named* soybean oil, or soy sauce/tofu/miso etc., counts. The full
  corpus run has 583 soy-uncertain, mostly this.
- **Food.com truncation is systematic:** 16.5 % of its recipes are `incomplete` (headline ingredient dropped). Excluded from train.
- **Soy sauce is gluten + soy.** Brewed with wheat unless labeled tamari / gluten-free. Most common hidden-gluten trap; the
  tool now shows a 'trap:' chip on the tile when a trap word is in the ingredient list.
- **Method-only mentions are never ingredients** — buns, tortillas, *and* a flambé of pisco. Same rule, no exceptions by appeal.
- **For `contains`, a title-named ingredient counts** even when the ingredient list dropped it (Food.com truncation:
  'Bulgogi (Korean Barbecued Beef)' with no beef line → beef, red_meat yes).
- **Missing quantities ≠ incomplete.** Food.com records are names-only by construction; `incomplete` means ingredients
  absent, obviously truncated, or hidden behind placeholders.
- **Serving suggestions and assembly are not ingredients.** "Serve with naan", "serve on buns", "wrap in a tortilla" ≠ `gluten`
  unless the bun/tortilla/bread is *in the ingredient list* — then it counts. (Measured as a systematic Jev FP on burgers and
  bánh mì; fixed by sending Jev the ingredient list only, no method text.)
- **Jev thresholds: YES ≥ 0.70, NO ≤ 0.25** (was 0.60). True positives in the probe sat ≥ 0.73; the 0.60–0.70 band is where hedges live.
- **`contains_uncertain` is a don't-care set in scoring** (Jev 0.25 < p < 0.70). Predicting or omitting those
  classes neither helps nor hurts the student. Gold may also use it for cases the human can't resolve from text.
- **`sulfites` removed (v0.3.2).** It is a property of how an ingredient was *processed* (sulfured vs. unsulfured raisins),
  not of the ingredient, so recipe text cannot settle it — 73/438 landed uncertain. Belongs at purchase time, not tagging.
  Jev probabilities for it remain in the candidates file for reference.

## Proposed `contains` classes (NOT in env #1 — add later as Jev-only questions, then gold touch-up pass)

| class | who filters on it | notes |
|---|---|---|
| `sulfites` | sulfite sensitivity | removed from env #1: processing property; only taggable when text says "sulfured" |
| `lamb_goat` | preference ("no lamb"), agent exclude filters | lamb, mutton, goat; stays inside `red_meat` for the dietary lattice. Not a restriction class — no religious exclusion singles it out, unlike pork/beef |
| `legumes` | legume allergy, some paleo | chickpeas, lentils, beans, peas, soy (overlap), peanuts (overlap) |
| `nightshades` | AIP / autoimmune protocols | tomato, potato, eggplant, peppers incl. chili & paprika — very high hit rate |
| `alliums` | low-FODMAP, Jain | onion, garlic, leek, shallot, chive, asafoetida as substitute signal |
| `root_vegetables` | Jain | potato, carrot, beet, radish, ginger? (convention needed) |
| `corn` | corn allergy | corn, cornstarch, corn syrup, polenta, masa |
| `coconut` | coconut allergy (FDA tree-nut listing) | coconut milk/oil/flakes |
| `added_sugar` | diabetic, "no added sugar" | sugar, syrups, honey (overlap), jaggery |
| `caffeine` | — | coffee, tea, chocolate/cocoa, matcha |
| `mushroom` | allergy / aversion | all fungi incl. truffle oil |
| `non_kosher_seafood` | kosher ingredient check | eel, catfish, squid, shark, sturgeon — closes the known `fish` gap |

Adding a class = one Jev Noul appended to `QUESTIONS` + a per-class pass over gold with pre-fill. Does not touch origin or the student's existing targets.
