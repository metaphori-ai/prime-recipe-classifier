#!/usr/bin/env python3
"""Select a hard-case-skewed slice of Food.com (irkaal/foodcom-recipes-and-reviews, CC0-1.0)
and write it in the data/raw JSONL schema.

Usage:
    uv run --with pandas --with pyarrow python -I scripts/foodcom_select.py \
        data/_staging/foodcom/recipes.parquet data/raw/foodcom.jsonl data/_staging/foodcom/selection_reasons.json

Ingredient note: Food.com's Kaggle dump carries normalized ingredient PARTS (no quantities/units,
and the quantities array is misaligned in ~75% of rows). We keep the parts verbatim as the
ingredient strings. Trap wording ("soy sauce", "fish sauce", "gelatin") survives; amounts do not.
"""
import collections
import json
import random
import re
import sys

import pandas as pd

SEED = 7
FETCHED_AT = "2020-12-28T00:00:00Z"  # dataset snapshot date on Kaggle

# --- bucket 1: regional specificity by title (take all, cap per pattern) -------------------
REGIONAL_TITLE = {
    # South Asian
    "hyderabadi": r"\bhyderabad", "chettinad": r"\bchettinad", "goan": r"\bgoan?\b", "kerala": r"\bkerala|malabar",
    "punjabi": r"\bpunjab|amritsar|lahor", "kashmiri": r"\bkashmir", "gujarati": r"\bgujarat", "bengali": r"\bbengal|\bbangla",
    "awadhi": r"\bawadh|lucknow", "rajasthani": r"\brajasthan", "mughlai": r"\bmughl", "andhra": r"\bandhra|\bguntur",
    "tamil": r"\btamil|\bchennai|\bmadras", "maharashtrian": r"\bmaharasht|\bmumbai|\bbombay", "mangalorean": r"\bmangalor",
    "sindhi": r"\bsindhi", "sri_lankan": r"\bsri lank|\bceylon", "nepali": r"\bnepal", "afghan": r"\bafghan",
    # East Asian
    "sichuan": r"\bsichuan|\bszechuan|\bszechwan", "hunan": r"\bhunan", "cantonese": r"\bcanton|\bhong kong|\bdim sum",
    "shanghai": r"\bshanghai", "beijing": r"\bbeijing|\bpeking", "hakka": r"\bhakka", "fujian": r"\bfujian|\bhokkien",
    "yunnan": r"\byunnan", "xinjiang": r"\bxinjiang|\buyghur|\buighur", "dongbei": r"\bdongbei|\bmanchuri",
    "taiwanese": r"\btaiwan", "okinawan": r"\bokinawa", "kansai": r"\bosaka|\bkyoto|\bkansai", "hokkaido": r"\bhokkaido",
    "korean_regional": r"\bjeju|\bjeolla|\bbusan|\bgyeongsang",
    # Southeast Asian
    "isan": r"\bisan\b|\bisaan|\bnortheastern thai", "northern_thai": r"\bchiang mai|\bnorthern thai|\bkhao soi",
    "southern_thai": r"\bsouthern thai|\bphuket", "padang": r"\bpadang|\bminang", "javanese": r"\bjavanese|\bjava\b",
    "balinese": r"\bbali", "nyonya": r"\bnyonya|\bnonya|\bperanakan", "malay": r"\bmalay\b|\bmalaysian",
    "bicolano": r"\bbicol", "ilocano": r"\biloc", "kapampangan": r"\bkapampangan|\bpampanga",
    "vietnamese_regional": r"\bhue\b|\bhanoi|\bsaigon|\bhoi an",
    # West Asian / Caucasus
    "levantine": r"\blebanese|\bsyrian|\bpalestin|\blevant", "gaziantep": r"\bgaziantep|\bantep", "persian": r"\bpersian|\biranian|\bshiraz",
    "kurdish": r"\bkurd", "georgian": r"\bgeorgian\b", "armenian": r"\barmenian", "gulf": r"\bemirati|\bkuwaiti|\bsaudi|\bomani|\bqatari",
    "iraqi": r"\biraqi", "yemeni": r"\byemen",
    # Central Asian / African
    "uzbek": r"\buzbek", "moroccan_regional": r"\bfez\b|\bfassi|\bmarrakesh|\bmarrakech",
    "ethiopian": r"\bethiopia|\beritrea|\binjera|\bberbere|\bdoro wat|\bwot\b", "nigerian": r"\bnigeria|\bjollof|\byoruba|\bigbo|\bhausa|\begusi",
    "ghanaian": r"\bghana", "senegalese": r"\bsenegal|\bthieboudienne", "swahili": r"\bswahili|\bzanzibar|\bkenyan|\btanzan",
    "cape_malay": r"\bcape malay|\bbobotie", "durban": r"\bdurban|\bbunny chow",
    # European regional
    "sicilian": r"\bsicil", "tuscan": r"\btuscan", "neapolitan": r"\bneapolitan|\bnaples", "roman": r"\broman\b|\bromana\b|\balla romana",
    "venetian": r"\bvenetian|\bveneto", "ligurian": r"\bligurian|\bgenoese|\bgenovese", "emilian": r"\bbolognese|\bemilia|\bmodena|\bparma",
    "piedmontese": r"\bpiedmont|\bpiemont", "lombard": r"\bmilanese|\blombard", "umbrian": r"\bumbria", "pugliese": r"\bpuglia|\bapulia",
    "calabrian": r"\bcalabr", "sardinian": r"\bsardin",
    "provencal": r"\bproven[cç]", "lyonnais": r"\blyon", "norman": r"\bnormand", "breton": r"\bbreton|\bbrittany",
    "alsatian": r"\balsat", "burgundian": r"\bbourguign|\bburgund", "sw_french": r"\bgascon|\bcassoulet|\btoulouse|\bp[ée]rigord",
    "basque": r"\bbasque|\bpintxo|\bpiperade",
    "catalan": r"\bcatalan", "andalusian": r"\bandalu", "galician": r"\bgalician|\bgallega", "valencian": r"\bvalencia", "asturian": r"\basturia",
    "cretan": r"\bcretan|\bcrete", "bavarian": r"\bbavarian", "swabian": r"\bswabian|\bschwäb",
    "scottish": r"\bscottish|\bscotch\b", "welsh": r"\bwelsh", "alentejo": r"\balentej", "azorean": r"\bazore",
    "ashkenazi": r"\bashkenaz|\bjewish|\bpassover|\bhanukkah|\brosh hashan|\bshabbat|\bkugel|\blatke",
    "sephardic": r"\bsephard",
    # Americas regional
    "lowcountry": r"\blowcountry|\blow country|\bcharleston|\bgullah", "appalachian": r"\bappalach",
    "soul_food": r"\bsoul food", "cajun": r"\bcajun", "creole": r"\bcreole", "new_england": r"\bnew england|\bboston|\bmaine\b",
    "midwestern": r"\bmidwest|\bwisconsin|\bminnesota|\bhotdish", "pnw": r"\bpacific northwest|\bseattle|\bportland|\boregon",
    "californian": r"\bcaliforni", "hawaiian": r"\bhawaii|\bpoke\b|\bloco moco|\bspam musubi", "southwestern": r"\bsouthwest|\bnew mexic|\bhatch chile",
    "tex_mex": r"\btex.?mex", "quebecois": r"\bqu[eé]b[eé]c|\bpoutine|\btourti[eè]re",
    "oaxacan": r"\boaxaca", "yucatecan": r"\byucat", "poblano": r"\bpuebla|\bpoblano\b.*mole|\bmole poblano", "jalisco": r"\bjalisco|\bguadalajara|\bbirria",
    "baja": r"\bbaja\b|\bensenada", "veracruz": r"\bveracruz", "northern_mexican": r"\bsonora|\bmonterrey|\bnorteñ|\bnorteno|\bchihuahua",
    "chifa": r"\bchifa|\blomo saltado", "limeno": r"\blime[ñn]o|\bceviche|\bcausa\b|\baji de gallina", "andean": r"\bandean|\bcusco|\bquinoa soup",
    "bahian": r"\bbahia|\bmoqueca|\bacaraj", "mineiro": r"\bminas|\bmineiro|\bpão de queijo|\bpao de queijo",
    # fusion / diaspora
    "indo_chinese": r"\bindo.?chinese|\bmanchurian|\bhakka noodle|\bchilli chicken|\bchili chicken|\bgobi manchurian|\bschezwan",
    "gochujang_fusion": r"\bgochujang|\bkorean (bbq|barbecue|fried chicken|taco)|\bkimchi (quesadilla|taco|burger|fries|grilled cheese)",
    "nikkei_fusion": r"\bnikkei|\btiradito",
    "anglo_indian": r"\banglo.?indian|\bkedgeree|\bmulligatawny|\bcountry captain|\bcoronation chicken",
    "british_curry": r"\bbalti|\btikka masala|\bphall|\bjalfrezi|\bmadras curry",
    "japanese_fusion": r"\bsushi (burrito|pizza|bake)|\bteriyaki (burger|pizza)|\bramen burger|\bcalifornia roll",
    "diaspora_misc": r"\bbutter chicken (pizza|poutine|lasagna)|\bcurry (pizza|poutine|wurst)|\bbanh mi (burger|taco)|\bbulgogi (taco|burger)",
}
REGIONAL_CAP = 12
CUISINE_SCALE = 5     # cuisine caps below are keyword supply ceilings; divide to hit the corpus target
TRAP_SCALE = 2

# --- bucket 2: cuisine keywords (Food.com Keywords), capped per cuisine ---------------------
CUISINE_KEYWORDS = {
    # (keyword lowercased, cap)
    "indian": 220, "pakistani": 120, "nepalese": 60,
    "chinese": 160, "cantonese": 100, "szechuan": 100, "hunan": 44, "japanese": 110, "korean": 150,
    "thai": 120, "vietnamese": 130, "indonesian": 110, "malaysian": 110, "filipino": 130,
    "lebanese": 120, "turkish": 120, "egyptian": 100, "moroccan": 110, "ethiopian": 105, "nigerian": 54,
    "south african": 120, "african": 80,
    "spanish": 90, "portuguese": 110, "greek": 90, "german": 70, "polish": 90, "hungarian": 100, "russian": 90,
    "scandinavian": 60, "swedish": 60, "norwegian": 50, "danish": 50, "finnish": 50,
    "mexican": 180, "tex mex": 120, "southwestern u.s.": 90, "cajun": 90, "creole": 90, "hawaiian": 90,
    "caribbean": 110, "cuban": 110, "puerto rican": 100, "peruvian": 100, "brazilian": 100, "colombian": 80, "chilean": 70,
    "canadian": 60, "australian": 60, "new zealand": 60, "polynesian": 70,
}

# --- bucket 3: ingredient traps (gated on a misleading keyword where relevant) --------------
TRAPS = [
    # (name, ingredient regex, required keyword regex over Keywords (None = any), cap)
    ("veg_keyword_fish_sauce", r"\bfish sauce|\bnam pla|\bnuoc mam", r"vegetarian|vegan", 60),
    ("veg_keyword_oyster_sauce", r"\boyster sauce", r"vegetarian|vegan", 40),
    ("veg_keyword_worcestershire", r"\bworcestershire", r"vegetarian|vegan", 60),
    ("veg_keyword_anchovy", r"\banchov", r"vegetarian|vegan", 40),
    ("veg_keyword_gelatin", r"\bgelatin", r"vegetarian|vegan", 60),
    ("gf_keyword_soy_sauce", r"\bsoy sauce", r"gluten|free of", 80),
    ("gf_keyword_worcestershire", r"\bworcestershire", r"gluten|free of", 40),
    ("gf_keyword_panko_couscous_seitan", r"\bpanko|\bcouscous|\bseitan|\bbulgur|\bfarro|\bsemolina", r"gluten|free of", 40),
    ("df_keyword_ghee_paneer", r"\bghee|\bpaneer", r"dairy|lactose|free of|vegan", 40),
    ("nut_pesto", r"\bpesto", None, 60),
    ("nut_satay_peanut_sauce", r"\bsatay|\bpeanut sauce", None, 40),
    ("nut_marzipan_praline", r"\bmarzipan|\bpraline|\bnutella|\bfrangipane", None, 50),
    ("nut_mole_romesco_dukkah", r"\bmole\b|\bromesco|\bdukkah", None, 40),
    ("alcohol_mirin_shaoxing_wine", r"\bmirin|\bshaoxing|\brice wine|\bmarsala|\bkirsch", None, 60),
    ("soy_miso_tempeh_tamari", r"\bmiso|\btempeh|\bnatto", None, 50),
    ("egg_mayo_aioli_hollandaise", r"\baioli|\bhollandaise|\bmeringue", None, 40),
    ("hidden_gelatin_dessert", r"\bgelatin", None, 50),
    ("hidden_dashi_bonito", r"\bdashi|\bbonito|\bkatsuobushi|\bhondashi", None, 40),
]

ORDINARY_N = 200
MIN_INGREDIENTS = 3


def norm_title(t):
    return re.sub(r"[^a-z0-9 ]+", " ", str(t).lower()).split()


def arr(v):
    if v is None:
        return []
    try:
        return [str(x) for x in list(v)]
    except TypeError:
        return []


def slug(name):
    s = re.sub(r"[^A-Za-z0-9]+", "-", str(name)).strip("-").lower()
    return s[:80]


def main(parquet, out_jsonl, out_reasons):
    rng = random.Random(SEED)
    df = pd.read_parquet(parquet, columns=[
        "RecipeId", "Name", "RecipeCategory", "Keywords", "RecipeIngredientParts",
        "RecipeInstructions", "ReviewCount", "DatePublished",
    ])
    df["parts"] = df["RecipeIngredientParts"].apply(arr)
    df["steps"] = df["RecipeInstructions"].apply(arr)
    df["kws"] = df["Keywords"].apply(lambda v: [k.lower() for k in arr(v)])
    df = df[(df["parts"].apply(len) >= MIN_INGREDIENTS) & (df["steps"].apply(len) >= 1)].copy()
    df["title_l"] = df["Name"].astype(str).str.lower()
    df["ing_l"] = df["parts"].apply(lambda p: " | ".join(p).lower())
    df["kw_l"] = df["kws"].apply(lambda k: " | ".join(k))
    print("usable rows:", len(df))

    selected = {}   # RecipeId -> reason
    counts = collections.Counter()

    def take(mask, reason, cap):
        ids = [rid for rid in df.loc[mask, "RecipeId"].tolist() if rid not in selected]
        rng.shuffle(ids)
        for rid in ids[:cap]:
            selected[rid] = reason
            counts[reason] += 1

    def nc(pat):  # make groups non-capturing so pandas doesn't warn
        return re.sub(r"\((?!\?)", "(?:", pat)

    # 1. regional by title
    for name, pat in REGIONAL_TITLE.items():
        take(df["title_l"].str.contains(nc(pat), regex=True), f"regional:{name}", REGIONAL_CAP)

    # 2. cuisine keywords
    for kw, cap in CUISINE_KEYWORDS.items():
        take(df["kws"].apply(lambda ks, kw=kw: kw in ks), f"cuisine:{kw.replace(' ', '_')}", max(10, cap // CUISINE_SCALE))

    # 3. traps
    for name, ing_pat, kw_pat, cap in TRAPS:
        mask = df["ing_l"].str.contains(nc(ing_pat), regex=True)
        if kw_pat:
            mask &= df["kw_l"].str.contains(nc(kw_pat), regex=True)
        take(mask, f"trap:{name}", max(8, cap // TRAP_SCALE))

    # 4. ordinary: no cuisine keyword at all, reviewed at least 5 times
    cuisine_set = set(CUISINE_KEYWORDS)
    no_cuisine = df["kws"].apply(lambda ks: not (set(ks) & cuisine_set))
    reviewed = df["ReviewCount"].fillna(0) >= 5
    take(no_cuisine & reviewed, "ordinary:random_reviewed", ORDINARY_N)

    # dedupe within selection on normalized title + first 5 parts
    sel = df[df["RecipeId"].isin(selected)].copy()
    sel["dkey"] = sel.apply(lambda r: " ".join(norm_title(r["Name"])) + "||" + "|".join(p.lower() for p in r["parts"][:5]), axis=1)
    before = len(sel)
    sel = sel.drop_duplicates("dkey")
    print(f"selected {before}, after internal dedupe {len(sel)}")

    reasons = {}
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for _, r in sel.iterrows():
            rid = int(r["RecipeId"])
            fid = f"foodcom-{rid}"
            tags = []
            if isinstance(r["RecipeCategory"], str) and r["RecipeCategory"]:
                tags.append(r["RecipeCategory"])
            tags.extend(arr(r["Keywords"]))
            rec = {
                "id": fid,
                "source": f"https://www.food.com/recipe/{slug(r['Name'])}-{rid}",
                "title": str(r["Name"]),
                "ingredients": r["parts"],
                "instructions": "\n".join(r["steps"]),
                "existing_tags": tags,
                "lang": "en",
                "fetched_at": FETCHED_AT,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            reasons[fid] = selected[rid]

    with open(out_reasons, "w", encoding="utf-8") as f:
        json.dump(reasons, f, indent=0, sort_keys=True)

    print(f"written: {len(reasons)} -> {out_jsonl}")
    print("\nper-reason counts (post-dedupe):")
    final = collections.Counter(reasons.values())
    by_bucket = collections.Counter(k.split(":")[0] for k in reasons.values())
    for k, v in sorted(final.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {v:5d}  {k}")
    print("\nby bucket:", dict(by_bucket))


if __name__ == "__main__":
    main(*sys.argv[1:4])
