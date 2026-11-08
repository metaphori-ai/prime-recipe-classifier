#!/usr/bin/env python3
"""Select ~400 recipes for human gold labeling: ~200 hard cases, ~200 ordinary.
Copies the full raw record and adds "selection_reason". No labels are written.

Usage:
    uv run python -I scripts/gold_candidates.py data/raw data/dedupe_manifest.json \
        data/_staging/foodcom/selection_reasons.json data/gold/candidates.jsonl
"""
import collections
import json
import os
import random
import re
import sys

SEED = 7
HARD_TARGET = 200
ORDINARY_TARGET = 200

FUSION_REASONS = {
    "regional:indo_chinese", "regional:gochujang_fusion", "regional:anglo_indian", "regional:british_curry",
    "regional:japanese_fusion", "regional:diaspora_misc", "regional:nikkei_fusion", "regional:chifa",
    "regional:cape_malay", "regional:durban", "regional:nyonya", "regional:tex_mex", "regional:limeno",
}
CROSS_BORDER_TITLE = re.compile(
    r"\bpunjab|\bamritsar|\blahor|\bbengal|\bbangla|\bbasque|\blebanese|\bsyrian|\bpalestin|\blevant|\bkurd"
    r"|\bmalay\b|\bnyonya|\bperanakan|\bswahili|\bzanzibar|\bethiopia|\beritrea|\binjera",
    re.I,
)
REGIONAL_TITLE = re.compile(
    r"\bhyderabad|\bchettinad|\bgoan?\b|\bkerala|\bkashmir|\bgujarat|\bawadh|\brajasthan|\bmughl|\bandhra|\btamil|\bmaharasht|\bmangalor|\bsindhi"
    r"|\bsichuan|\bszechuan|\bhunan|\bcanton|\bshanghai|\bbeijing|\bpeking|\bhakka|\bfujian|\byunnan|\bxinjiang|\buyghur|\btaiwan"
    r"|\bsicil|\btuscan|\bneapolitan|\bromana?\b|\bvenetian|\bligurian|\bgenovese|\bbolognese|\bemilia|\bpiedmont|\bmilanese|\bpuglia|\bcalabr|\bsardin"
    r"|\boaxaca|\byucat|\bpoblano\b|\bjalisco|\bbaja\b|\bveracruz|\bsonora"
    r"|\blowcountry|\bappalach|\bsoul food|\bcajun|\bcreole|\bnew england|\bmidwest|\bhawaii|\bsouthwest|\bcaliforni"
    r"|\bjollof|\begusi|\byoruba|\bigbo|\bhausa|\bdoro wat|\bberbere|\bbobotie|\bbunny chow"
    r"|\bisan\b|\bchiang mai|\bkhao soi|\bpadang|\bjavanese|\bbali\b|\bbicol|\biloc|\bkapampangan",
    re.I,
)
TRAP_ING = re.compile(
    r"\bfish sauce|\bnam pla|\boyster sauce|\bdashi|\bbonito|\bworcestershire|\banchov|\bgelatin|\bghee|\bpaneer"
    r"|\bmirin|\bshaoxing|\bmiso|\btempeh|\btamari|\bpesto|\bsatay|\bmarzipan|\bpraline|\bmole\b|\bromesco|\bdukkah|\bpanko|\bseitan|\bcouscous",
    re.I,
)
VEG_TAG = re.compile(r"vegetarian|vegan|gluten|free of|dairy.?free", re.I)
WIKI_ORDINARY_CATS = {"Dessert recipes", "Soup recipes", "Chicken recipes", "Sauce recipes", "Meat recipes", "Rice recipes",
                      "Stew recipes", "Pasta recipes", "Salad recipes", "Cake recipes", "Cookie recipes", "Bread recipes",
                      "Pork recipes", "Potato recipes", "Sandwich recipes", "Fish recipes", "Egg recipes", "Seafood recipes",
                      "Beef recipes", "Vegetable recipes", "Breakfast recipes", "Pancake recipes", "Bean recipes"}


def load_raw(raw_dir, dropped_ids):
    recs = {}
    for fn in sorted(os.listdir(raw_dir)):
        if not fn.endswith(".jsonl"):
            continue
        src = fn[:-6]
        with open(os.path.join(raw_dir, fn), encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r["id"] in dropped_ids:
                    continue
                r["_src"] = src
                recs[r["id"]] = r
    return recs


def main(raw_dir, manifest_path, reasons_path, out_path):
    rng = random.Random(SEED)
    with open(manifest_path, encoding="utf-8") as f:
        dropped_ids = {d["id"] for d in json.load(f)["dropped"]}
    with open(reasons_path, encoding="utf-8") as f:
        fc_reasons = json.load(f)
    recs = load_raw(raw_dir, dropped_ids)
    print("deduped raw records:", len(recs))

    chosen = {}       # id -> reason
    bucket_counts = collections.Counter()

    def take(ids, reason, n):
        ids = [i for i in ids if i not in chosen]
        rng.shuffle(ids)
        for i in ids[:n]:
            chosen[i] = reason
            bucket_counts[reason.split(":")[0]] += 1
        return min(n, len(ids))

    # ---------------- HARD (~200) ----------------
    # H1 fusion / diaspora from Food.com reasons: up to 4 per reason
    by_reason = collections.defaultdict(list)
    for rid, rs in fc_reasons.items():
        if rid in recs:
            by_reason[rs].append(rid)
    for rs in sorted(FUSION_REASONS):
        take(by_reason.get(rs, []), f"hard:fusion_diaspora:{rs.split(':')[1]}", 4)

    # H2 cross-border traditions, all sources, by title
    xb = [i for i, r in recs.items() if CROSS_BORDER_TITLE.search(r["title"])]
    take(xb, "hard:cross_border:title", 20)

    # H3 owner's own captures with regional/cuisine signal — highest product value
    ss = [i for i, r in recs.items() if r["_src"].startswith("saffronsifu")]
    ss_regional = [i for i in ss if REGIONAL_TITLE.search(recs[i]["title"]) or CROSS_BORDER_TITLE.search(recs[i]["title"])]
    take(ss_regional, "hard:regional:saffronsifu_title", 15)
    ss_tagged = [i for i in ss if any(re.search(r"indian|korean|pakistani|punjabi|thai|japanese|chinese|mexican|italian|persian|filipino",
                                                 t, re.I) for t in recs[i]["existing_tags"])]
    take(ss_tagged, "hard:regional:saffronsifu_tag", 20)

    # H4 regional specificity by title across the five macro-tier traditions + Africa: Food.com + Wikibooks
    reg_fc = [rid for rs, ids in by_reason.items() if rs.startswith("regional:") and rs not in FUSION_REASONS for rid in ids]
    # stratify: at most 2 per Food.com regional reason
    per = collections.defaultdict(list)
    for rid in reg_fc:
        per[fc_reasons[rid]].append(rid)
    reg_pick = []
    for rs, ids in sorted(per.items()):
        rng.shuffle(ids)
        reg_pick.extend(ids[:2])
    take(reg_pick, "hard:regional:foodcom_title", 60)
    wiki_reg = [i for i, r in recs.items() if r["_src"] == "wikibooks" and REGIONAL_TITLE.search(r["title"])]
    take(wiki_reg, "hard:regional:wikibooks_title", 15)
    wiki_afr = [i for i, r in recs.items() if r["_src"] == "wikibooks"
                and any(t in ("Nigerian recipes", "Ethiopian recipes", "African recipes", "Swallow recipes") for t in r["existing_tags"])]
    take(wiki_afr, "hard:regional:wikibooks_african_category", 15)

    # H5 ingredient traps: a trap word present AND a misleading diet tag, any source; else plain trap word
    trap_tagged = [i for i, r in recs.items()
                   if TRAP_ING.search(" | ".join(r["ingredients"])) and any(VEG_TAG.search(t) for t in r["existing_tags"])]
    take(trap_tagged, "hard:trap:ingredient_with_diet_tag", 25)
    trap_fc = [rid for rs, ids in by_reason.items() if rs.startswith("trap:") for rid in ids]
    take(trap_fc, "hard:trap:foodcom_selected", 20)

    hard_n = sum(v for k, v in bucket_counts.items() if k == "hard")
    # top up hard to target from remaining Food.com regional/cuisine picks
    if hard_n < HARD_TARGET:
        pool = [rid for rs, ids in by_reason.items() if rs.startswith(("regional:", "cuisine:")) for rid in ids]
        take(pool, "hard:regional:foodcom_topup", HARD_TARGET - hard_n)

    # ---------------- ORDINARY (~200) ----------------
    def is_plain(r):
        if REGIONAL_TITLE.search(r["title"]) or CROSS_BORDER_TITLE.search(r["title"]):
            return False
        if TRAP_ING.search(" | ".join(r["ingredients"])):
            return False
        return len(r["ingredients"]) >= 4

    wiki_plain = [i for i, r in recs.items() if r["_src"] == "wikibooks" and is_plain(r)
                  and any(t in WIKI_ORDINARY_CATS for t in r["existing_tags"])]
    take(wiki_plain, "ordinary:wikibooks_common_category", 90)
    fc_plain = [rid for rid in by_reason.get("ordinary:random_reviewed", []) if is_plain(recs[rid])]
    take(fc_plain, "ordinary:foodcom_random_reviewed", 60)
    ss_plain = [i for i in ss if is_plain(recs[i]) and i not in chosen]
    take(ss_plain, "ordinary:saffronsifu_plain", 50)

    # ---------------- write ----------------
    out = []
    for rid, reason in chosen.items():
        r = dict(recs[rid])
        r.pop("_src", None)
        r["selection_reason"] = reason
        out.append(r)
    out.sort(key=lambda r: (r["selection_reason"], r["id"]))
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"written {len(out)} -> {out_path}")
    print("hard / ordinary:", bucket_counts["hard"], "/", bucket_counts["ordinary"])
    reasons = collections.Counter(r["selection_reason"] for r in out)
    for k, v in sorted(reasons.items()):
        print(f"  {v:4d}  {k}")
    srcs = collections.Counter(r["id"].split("-")[0] for r in out)
    print("by source prefix:", dict(srcs))


if __name__ == "__main__":
    main(*sys.argv[1:5])
