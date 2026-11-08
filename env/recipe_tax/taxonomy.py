"""Load taxonomy.yaml and answer structural questions about it.

Origin spine = culinary tradition. Countries are node metadata (`_c` on branches, `{c: [...]}` on leaves),
inherited from the nearest ancestor that sets them. Paths are tuples of display names exactly as in the YAML.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
import yaml

def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

@dataclass
class Node:
    path: tuple[str, ...]
    countries: tuple[str, ...]
    children: list[str] = field(default_factory=list)

    @property
    def name(self) -> str: return self.path[-1]
    @property
    def depth(self) -> int: return len(self.path)
    @property
    def is_leaf(self) -> bool: return not self.children

class Taxonomy:
    def __init__(self, data: dict):
        self.version = str(data.get("version", "?"))
        self.contains_classes: list[str] = list(data["contains"])
        self.derived: dict = data.get("derived", {})
        self.traps: dict[str, list[str]] = data.get("ingredient_traps", {})
        self.nodes: dict[tuple[str, ...], Node] = {}
        self._walk(data["origin"], (), ())

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Taxonomy":
        p = Path(path) if path else Path(__file__).resolve().parents[2] / "taxonomy.yaml"
        with open(p, encoding="utf-8") as f:
            return cls(yaml.safe_load(f))

    def _walk(self, tree: dict, prefix: tuple[str, ...], inherited: tuple[str, ...]):
        own = tuple(tree.get("_c", inherited)) if isinstance(tree, dict) else inherited
        for name, val in (tree or {}).items():
            if name == "_c":
                continue
            path = prefix + (name,)
            if val is None:                                   # leaf, inherits
                self.nodes[path] = Node(path, own)
            elif isinstance(val, dict) and set(val) == {"c"}:  # leaf with own countries
                self.nodes[path] = Node(path, tuple(val["c"]))
            else:                                             # branch
                branch_c = tuple(val.get("_c", own))
                node = Node(path, branch_c, children=[k for k in val if k != "_c"])
                self.nodes[path] = node
                self._walk(val, path, branch_c)

    # ---- queries -------------------------------------------------------
    def has_path(self, path: Iterable[str]) -> bool:
        return tuple(path) in self.nodes

    def validate_path(self, path: Iterable[str]) -> str | None:
        """Return None if valid, else a reason."""
        t = tuple(path)
        if not t: return "empty path"
        if t not in self.nodes:
            for i in range(1, len(t) + 1):
                if t[:i] not in self.nodes:
                    return f"unknown node {t[:i]!r}"
        return None

    def ancestors(self, path: Iterable[str]) -> list[tuple[str, ...]]:
        t = tuple(path)
        return [t[:i] for i in range(1, len(t))]

    def countries(self, path: Iterable[str]) -> tuple[str, ...]:
        return self.nodes[tuple(path)].countries

    def satisfaction_set(self, path: Iterable[str]) -> set[str]:
        """Everything a dish at this node *satisfies*: itself, ancestors, countries. All slugs."""
        t = tuple(path)
        out = {slug(n) for n in t}                          # node + ancestors by name
        out.add("origin:" + "/".join(slug(n) for n in t))   # exact path token
        out |= {"country:" + slug(c) for c in self.countries(t)}
        return out

    def find_by_name(self, name: str) -> list[tuple[str, ...]]:
        return [p for p in self.nodes if p[-1] == name]

    def __len__(self): return len(self.nodes)
