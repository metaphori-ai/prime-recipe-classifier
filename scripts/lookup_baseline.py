#!/usr/bin/env python3
"""How far does a plain lookup go? No model: find the most similar TRAINING recipe and copy its origin label.

  title   nearest training title (character 3-5-gram TF-IDF cosine, so misspellings and word order still match)
  ingr    nearest training recipe by ingredient-name overlap (IDF-weighted Jaccard), for recipes with unhelpful titles
Scored with the same origin score as every model. Gold recipes are never in the training set (held out by id and by content).

  PYTHONPATH=env python3 scripts/lookup_baseline.py
"""
import json, math, re, sys, collections, io, glob
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
from recipe_tax.taxonomy import Taxonomy
from recipe_tax import scoring

def grams(t):
    t = " " + re.sub(r"[^a-z0-9 ]+", " ", t.lower()).strip() + " "
    return collections.Counter(t[i:i + n] for n in (3, 4, 5) for i in range(len(t) - n + 1))

def title_of(r): return r["prompt"].split("\n", 1)[0].removeprefix("Title:").strip()
def ingr_of(r): return {re.sub(r"[^a-z ]+", "", x.lower()).strip() for x in r["ingredients"]} - {""}

class Index:
    def __init__(self, train):
        self.train = train
        self.tg = [grams(title_of(r)) for r in train]
        df = collections.Counter(g for c in self.tg for g in c)
        self.idf = {g: math.log(len(train) / d) + 1 for g, d in df.items()}
        self.tv = [{g: c * self.idf[g] for g, c in cnt.items()} for cnt in self.tg]
        self.tn = [math.sqrt(sum(v * v for v in d.values())) for d in self.tv]
        self.inv = collections.defaultdict(list)
        for i, d in enumerate(self.tv):
            for g, w in d.items(): self.inv[g].append((i, w))
        self.ing = [ingr_of(r) for r in train]
        idf2 = collections.Counter(x for s in self.ing for x in s)
        self.iidf = {x: math.log(len(train) / d) + 1 for x, d in idf2.items()}
        self.iinv = collections.defaultdict(list)
        for i, s in enumerate(self.ing):
            for x in s: self.iinv[x].append(i)
        self.iw = [sum(self.iidf[x] for x in s) for s in self.ing]
    def title(self, t, k=1):
        q = {g: c * self.idf.get(g, 0) for g, c in grams(t).items()}; qn = math.sqrt(sum(v * v for v in q.values())) or 1
        acc = collections.defaultdict(float)
        for g, w in q.items():
            for i, tw in self.inv.get(g, ()): acc[i] += w * tw
        return sorted(((v / (qn * self.tn[i]), i) for i, v in acc.items()), reverse=True)[:k]
    def ingredients(self, s, k=1):
        acc = collections.defaultdict(float)
        for x in s:
            for i in self.iinv.get(x, ()): acc[i] += self.iidf[x]
        qw = sum(self.iidf.get(x, max(self.iidf.values())) for x in s) or 1
        return sorted(((v / (qw + self.iw[i] - v), i) for i, v in acc.items()), reverse=True)[:k]

def main():
    tax = Taxonomy.load()
    train = [json.loads(l) for l in open(ROOT / "data/dataset_v04/train.jsonl")]
    ix = Index(train)
    prim = lambda o: next((tuple(x["path"]) for x in o if x.get("role", "primary") == "primary"), None)
    res = {}
    for name, fn in (("gold 439", "eval.jsonl"), ("hard 224", "eval_gold_hard.jsonl")):
        rows = [json.loads(l) for l in open(ROOT / "data/dataset_v04" / fn)]
        out = []
        for r in rows:
            g = r["labels"]["origin"]
            ts, ti = (ix.title(title_of(r)) or [(0.0, 0)])[0]; is_, ii = (ix.ingredients(ingr_of(r)) or [(0.0, 0)])[0]
            lt, li = train[ti]["labels"]["origin"], train[ii]["labels"]["origin"]
            out.append(dict(id=r["id"], title=title_of(r), gold=g, ts=ts, t_title=title_of(train[ti]), t_lab=lt, t_sc=scoring.score_origin(tax, g, lt), t_ex=prim(lt) == prim(g),
                            is_=is_, i_title=title_of(train[ii]), i_lab=li, i_sc=scoring.score_origin(tax, g, li), i_ex=prim(li) == prim(g)))
        res[name] = out
        m = lambda xs: sum(xs) / max(len(xs), 1)
        print(f"\n== {name}: copy the label of the nearest TRAINING recipe (no model)")
        print(f"   by title:       origin {m([o['t_sc'] for o in out]):.3f} · exact primary {m([o['t_ex'] for o in out]):.0%}")
        print(f"   by ingredients: origin {m([o['i_sc'] for o in out]):.3f} · exact primary {m([o['i_ex'] for o in out]):.0%}")
        print("   by how close the nearest title is:")
        print(f"   {'similarity':>12s} {'recipes':>8s} {'origin':>7s} {'exact':>6s}")
        for lo, hi in ((.8, 1.01), (.6, .8), (.45, .6), (.3, .45), (0, .3)):
            S = [o for o in out if lo <= o["ts"] < hi]
            if S: print(f"   {f'{lo:.2f}-{min(hi,1):.2f}':>12s} {len(S):8d} {m([o['t_sc'] for o in S]):7.3f} {m([o['t_ex'] for o in S]):6.0%}")
    json.dump(res, open("/tmp/lookup_last.json", "w"))

if __name__ == "__main__": main()
