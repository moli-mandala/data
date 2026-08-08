"""
edges_util.py — shared readers for the typed edge table (cldf/edges.csv).

Post-cutover, scripts that used to read forms.csv's Origin_ID/Relation/Variant_Of columns load
the graph from here instead. Keep the resolvers in this one place: align.py, the analysis
scripts, and the tests all agree on what "origin" means only because they share these functions.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict

EDGES = "cldf/edges.csv"

# Languages whose head-forms are alignable reconstructions (mirrors align.py's PROTO_LANGS).
PROTO_LANGS = {"Indo-Aryan", "PDr", "PMu", "PNur", "PA", "PIA", "OIA"}


def load_edges(path: str = EDGES):
    """All edge rows as dicts (Child_ID, Parent_ID, Kind, Rank, Pos, Source, Note)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing — run unify_cldf.py")
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rank1_map(edges) -> dict[str, tuple[str, str]]:
    """child id → (parent id, kind) for the accepted attestation edge."""
    out = {}
    for e in edges:
        if e["Rank"] == "1" and e["Kind"] in ("reflex", "borrowed", "variant"):
            out[e["Child_ID"]] = (e["Parent_ID"], e["Kind"])
    return out


def children_map(edges) -> dict[str, list[str]]:
    """parent id → child ids over rank-1 attestation edges."""
    out = defaultdict(list)
    for e in edges:
        if e["Rank"] == "1" and e["Kind"] in ("reflex", "borrowed", "variant"):
            out[e["Parent_ID"]].append(e["Child_ID"])
    return out


def relation_of(rank1: dict[str, tuple[str, str]], node: str) -> str:
    """Legacy-shim Relation string for one node ('' when the node has no attestation edge)."""
    e = rank1.get(node)
    return e[1] if e else ""


def effective_etymon(rank1: dict[str, tuple[str, str]], node: str) -> str | None:
    """The attestation-tree root above `node` (None for parentless nodes; cycle-guarded)."""
    seen = set()
    cur = node
    while cur in rank1 and cur not in seen:
        seen.add(cur)
        cur = rank1[cur][0]
    return None if cur == node else cur


def aligned_parent(
    rank1: dict[str, tuple[str, str]],
    lang_of: dict[str, str],
    node: str,
    proto_langs: frozenset | set = frozenset(PROTO_LANGS),
) -> str | None:
    """The head-form `node` should be phonetically aligned against, or None.

    reflex/borrowed edges align against their immediate target (never walked further — matching
    the legacy Origin_ID semantics for those rows). Variant chains are walked upward until the
    nearest proto-language ancestor (or the chain head). Relative to the legacy column this
    changes alignment coverage in three enumerated, reviewed ways (2026-08 corpus): 784 variant
    forms gain an alignment (their chains pass through attested non-proto parents that used to
    hide the etymon), 37 variants of proto-language siblings align against the nearer ancestor,
    and none lose one.
    """
    seen = set()
    cur = node
    while True:
        if cur in seen:
            return None
        seen.add(cur)
        e = rank1.get(cur)
        if e is None:
            return None
        parent, kind = e
        if kind != "variant":
            return parent
        if lang_of.get(parent) in proto_langs:
            return parent
        if parent not in rank1:
            return parent
        cur = parent


def attach_legacy_graph(rows, edges_path: str = EDGES) -> None:
    """Synthesize the legacy Origin_ID / Relation fields onto forms-row dicts in place.

    Post-cutover convenience for analysis scripts whose logic predates the edge table:
    Origin_ID = the rank-1 target (the immediate parent — note this is the variant TARGET for
    variant rows, which legacy Origin_ID only was for parent-encoded variants) and Relation =
    the rank-1 kind, with 'local' restored from Status for unlinked nodes.
    """
    rank1 = rank1_map(load_edges(edges_path))
    for row in rows:
        e = rank1.get(row.get("ID", ""))
        row["Origin_ID"] = e[0] if e else ""
        row["Relation"] = e[1] if e else ("local" if row.get("Status") == "unlinked" else "")
        row["Variant_Of"] = e[0] if e and e[1] == "variant" else ""
        row["Borrowed_From"] = e[0] if e and e[1] == "borrowed" else ""
