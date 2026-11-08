"""CLI: validate JSONL label files against the taxonomy.  usage: python -m recipe_tax.validate data/silver/*.jsonl"""
from __future__ import annotations
import json, sys
from collections import Counter
from .taxonomy import Taxonomy
from .scoring import gate

def main(paths: list[str]) -> int:
    tax = Taxonomy.load()
    bad = Counter(); n = 0; errors = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line: continue
                n += 1
                try: rec = json.loads(line)
                except json.JSONDecodeError as e:
                    bad["json"] += 1; errors.append(f"{p}:{i} json: {e}"); continue
                labels = rec.get("labels")
                if not labels: bad["no labels"] += 1; errors.append(f"{p}:{i} no labels"); continue
                why = gate(tax, labels)
                if why: bad[why.split(" ")[0]] += 1; errors.append(f"{p}:{i} {why}")
    print(f"{n} records · {sum(bad.values())} invalid · taxonomy {tax.version} ({len(tax)} nodes)")
    for k, v in bad.most_common(): print(f"  {v:5d}  {k}")
    for e in errors[:40]: print("   ", e)
    if len(errors) > 40: print(f"    ... {len(errors)-40} more")
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
