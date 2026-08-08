"""
edges_build.py — derive the single typed edge table (cldf/edges.csv) from the unified graph.

This is the serialization boundary of the edge-model migration: `unify_cldf.py` keeps building
its battle-tested internal row lists (positional columns, seven in-place mutating producers),
and this module reads the FINISHED columns plus the ordered derivation edge list and classifies
everything into one table:

    edges(Child_ID, Parent_ID, Kind, Rank, Pos, Source, Note)

      Kind ∈ {reflex, borrowed, variant, component, derived}
      Rank  1  = the accepted etymology (invariant: ≤1 rank-1 reflex/borrowed/variant edge
                 per node); 2+ = alternate hypotheses, in stored derivation order
      Pos   1… = compound member order on `component` edges (empty otherwise)
      Note      review markers for auto-classified edges (`review:auto-alternate`,
                 `review:cross-entry-variant`)

Rank-1 classification from the legacy columns (Origin_ID / Relation / Variant_Of):

    reflex                              → (ID, Origin, reflex, 1)
    borrowed, no Variant_Of             → (ID, Origin, borrowed, 1)
    borrowed, sibling Variant_Of        → (ID, Variant_Of, variant, 1)      [kind-change waiver:
        rows whose Relation was flipped to 'borrowed' in place by mark_cross_family_borrowings
        while their Variant_Of survived — they are variants of a borrowed sibling, and the
        borrowing is now inherited transitively]
    variant, Variant_Of ∈ {'', Origin}  → (ID, Origin-or-target, variant, 1)  [the two legacy
        encodings of "variant of my parent" collapse into one]
    variant, sibling Variant_Of         → (ID, Variant_Of, variant, 1)  [etymon now transitive]
    variant/borrowed, cross-entry Variant_Of → (ID, Origin, reflex-or-borrowed, 1)
        + (ID, Variant_Of, variant, 2, Note="review:cross-entry-variant")
    local                               → no edge  (node Status 'unlinked')
    ''  (etymon)                        → no rank-1 edge  (node Status 'entry')

Derivation-list classification (order preserved from cldf/derivation.csv + section_edges):

    edge duplicating the child's rank-1 target        → dropped (Khowar double-encoding)
    child has a rank-1 edge (attested form)           → (child, parent, reflex, rank 2…);
        children with exactly ONE such edge get Note="review:auto-alternate" (ambiguous between
        alternate etymology and morphological derivation — resolved later in the etymology lab)
    parentless child, parent is a √root               → (child, parent, derived, 1)
    parentless child, ≥2 non-root parents             → (child, parent, component, 1, Pos=1…n)
    parentless child, single non-root parent          → (child, parent, derived, 1)

Standalone use (reads the already-unified cldf/):  python edges_build.py [--check]
"""

from __future__ import annotations

import csv
import os
import re
import sys
from collections import defaultdict

EDGES_HEADER = ["Child_ID", "Parent_ID", "Kind", "Rank", "Pos", "Source", "Note"]

# unified forms.csv positional indices (see unify_cldf.UNIFIED)
_ID, _FORM, _ORIGIN, _REL, _RD, _VOF, _BF = 0, 2, 11, 13, 14, 15, 16

# Verbal roots were minted as `r<N>` by link_refs but now carry durable f_ ids; the only
# stable marker is the √ head-form. (This replaces the legacy /^r\d/ id sniffing, which has
# been dead since durable-ID assignment.)
_ROOT_RE = re.compile(r"^r\d+$")
NOTE_ALTERNATE = "review:auto-alternate"
NOTE_CROSS_ENTRY = "review:cross-entry-variant"


def _strip_marker(pid: str) -> str:
    return pid[1:] if pid and pid[0] in ">~" else pid


def build_edges(rows, deriv_edges):
    """Classify the finished unified rows + ordered derivation edges into typed edges.

    rows        — iterable of 17-column lists (the final state of etyma/reflex/ext rows)
    deriv_edges — ordered (child_id, parent_id) pairs: link_refs output + appended section edges

    Returns (edge_rows, status_by_id, stats). edge_rows are EDGES_HEADER-shaped lists sorted by
    (Child_ID, Kind, Rank, Pos, Parent_ID); status_by_id maps every node id to
    'entry' | 'unlinked' | '' (attested).
    """
    by_id = {}
    root_ids = set()
    for r in rows:
        by_id[r[_ID]] = r
        if r[_FORM].startswith("\u221a") or _ROOT_RE.match(r[_ID]):
            root_ids.add(r[_ID])

    edges = []  # (child, parent, kind, rank, pos, source, note)
    status_by_id = {}
    rank1 = {}  # child id → (parent id, kind)
    stats = defaultdict(int)

    def emit(child, parent, kind, rank, pos="", source="", note=""):
        edges.append([child, parent, kind, rank, pos, source, note])

    def origin_of(row):
        return _strip_marker(row[_ORIGIN])

    # ---- rank-1 edges + node status from the legacy columns --------------------------------
    for r in rows:
        rid, rel = r[_ID], r[_REL]
        origin, vof = origin_of(r), _strip_marker(r[_VOF])
        if rel == "":
            status_by_id[rid] = "entry"
            stats["status_entry"] += 1
            continue
        if rel == "local":
            status_by_id[rid] = "unlinked"
            stats["status_unlinked"] += 1
            continue
        status_by_id[rid] = ""
        if r[_RD]:
            # an attested row cannot also be a redirect stub (one known case: CDIAL 14302,
            # merged into 1693 AND flipped to borrowed by cross-family inference). The shipped
            # v1/v2 builders silently dropped the redirect for such rows; make that explicit.
            stats["redirect_dropped_on_attested"] += 1
        if rel == "reflex":
            rank1[rid] = (origin, "reflex")
            stats["rank1_reflex"] += 1
            continue
        if rel not in ("variant", "borrowed"):
            raise ValueError(f"unexpected Relation {rel!r} on {rid}")

        # variant-target analysis (shared by 'variant' and borrowed-with-Variant_Of rows)
        if vof and vof != origin:
            target = by_id.get(vof)
            target_origin = origin_of(target) if target is not None else None
            sibling = target is not None and (target_origin == origin or vof == origin)
        else:
            sibling = False

        if rel == "borrowed":
            if not vof:
                rank1[rid] = (origin, "borrowed")
                stats["rank1_borrowed"] += 1
            elif vof == origin or sibling:
                # relation was overwritten to 'borrowed' in place while Variant_Of survived:
                # the row is a variant of a borrowed sibling; borrowing is transitive
                rank1[rid] = (vof, "variant")
                stats["rank1_variant"] += 1
                stats["waiver_borrowed_to_variant"] += 1
            else:
                rank1[rid] = (origin, "borrowed")
                stats["rank1_borrowed"] += 1
                emit(rid, vof, "variant", 2, note=NOTE_CROSS_ENTRY)
                stats["waiver_cross_entry"] += 1
        else:  # variant
            if not vof:
                rank1[rid] = (origin, "variant")
                stats["rank1_variant"] += 1
                stats["variant_of_parent"] += 1
            elif vof == origin:
                rank1[rid] = (vof, "variant")
                stats["rank1_variant"] += 1
                stats["variant_of_parent"] += 1
            elif sibling:
                rank1[rid] = (vof, "variant")
                stats["rank1_variant"] += 1
                stats["variant_sibling"] += 1
            else:
                # contradictory cross-entry double parent: keep the etymon linkage as accepted,
                # demote the stray variant pointer to a reviewable rank-2 edge
                rank1[rid] = (origin, "reflex")
                stats["rank1_reflex"] += 1
                emit(rid, vof, "variant", 2, note=NOTE_CROSS_ENTRY)
                stats["waiver_cross_entry"] += 1

    for child, (parent, kind) in rank1.items():
        emit(child, parent, kind, 1)

    # ---- derivation-list classification ----------------------------------------------------
    per_child = defaultdict(list)
    seen_pairs = set()
    for child, parent in deriv_edges:
        pair = (child, parent)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        per_child[child].append(parent)

    for child, parents in per_child.items():
        r1 = rank1.get(child)
        live = []
        for parent in parents:
            if r1 is not None and parent == r1[0]:
                stats["dedup_vs_rank1"] += 1  # Khowar-style double encoding
                continue
            live.append(parent)
        if not live:
            continue
        if r1 is not None:
            # attested child → alternate etymology hypotheses, in stored order
            note = NOTE_ALTERNATE if len(live) == 1 else ""
            next_rank = 2 + sum(1 for e in edges if e[0] == child and e[3] >= 2)
            for i, parent in enumerate(live):
                emit(child, parent, "reflex", next_rank + i, note=note)
                stats["alt_edges"] += 1
            if note:
                stats["alt_reviewable"] += 1
        else:
            roots = [p for p in live if p in root_ids]
            non_roots = [p for p in live if p not in root_ids]
            for parent in roots:
                emit(child, parent, "derived", 1)
                stats["derived_root"] += 1
            if len(non_roots) >= 2:
                for pos, parent in enumerate(non_roots, start=1):
                    emit(child, parent, "component", 1, pos=pos)
                    stats["component_edges"] += 1
                stats["component_groups"] += 1
            elif len(non_roots) == 1:
                emit(child, non_roots[0], "derived", 1)
                stats["derived_single"] += 1

    edges.sort(key=lambda e: (e[0], e[2], e[3], e[4] if e[4] != "" else 0, e[1]))
    validate_edges(edges, by_id, status_by_id, rank1)
    return edges, status_by_id, dict(stats)


def validate_edges(edges, by_id, status_by_id, rank1):
    """Structural invariants; raises on violation (build-time guard, mirrored in tests)."""
    rank1_counts = defaultdict(int)
    for child, parent, kind, rank, pos, _src, _note in edges:
        if child == parent:
            raise ValueError(f"self-edge on {child}")
        if parent not in by_id:
            raise ValueError(f"edge {child} → {parent}: parent not a node")
        if child not in by_id:
            raise ValueError(f"edge {child} → {parent}: child not a node")
        if rank == 1 and kind in ("reflex", "borrowed", "variant"):
            rank1_counts[child] += 1
    for child, n in rank1_counts.items():
        if n > 1:
            raise ValueError(f"{child} has {n} rank-1 attestation edges")
    for child, n in rank1_counts.items():
        if status_by_id.get(child) in ("entry", "unlinked"):
            raise ValueError(f"{child} has Status {status_by_id[child]!r} AND a rank-1 edge")
    # variant chains terminate (cycle guard)
    for child, (parent, kind) in rank1.items():
        if kind != "variant":
            continue
        seen = {child}
        cur = parent
        while cur in rank1:
            if cur in seen:
                raise ValueError(f"variant chain cycle through {child}")
            seen.add(cur)
            nxt, k = rank1[cur]
            if k != "variant":
                break
            cur = nxt
    # component groups: contiguous Pos starting at 1, ≥2 members
    groups = defaultdict(list)
    for child, _parent, kind, _rank, pos, _src, _note in edges:
        if kind == "component":
            groups[child].append(int(pos))
    for child, poss in groups.items():
        if sorted(poss) != list(range(1, len(poss) + 1)) or len(poss) < 2:
            raise ValueError(f"component Pos not contiguous for {child}: {sorted(poss)}")


def rank1_chain_target(rank1, child, _cache={}):
    """Effective etymon: follow rank-1 edges to the attestation-tree root (cycle-guarded)."""
    seen = []
    cur = child
    while cur in rank1 and cur not in seen:
        seen.append(cur)
        cur = rank1[cur][0]
    return cur


# ---- standalone: derive edges.csv from the already-unified cldf/ ---------------------------


def load_unified(forms_path="cldf/forms.csv"):
    """The 17-column internal-format rows: pre-cutover forms.csv, or the forms-legacy.csv
    cross-check file written by `unify_cldf.py --legacy-cols` after the cutover."""
    for path in (forms_path, "cldf/forms-legacy.csv"):
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            if "Origin_ID" in header:
                return list(reader)
    raise SystemExit(
        "no legacy-format rows available — run `python unify_cldf.py --legacy-cols` (or use a "
        "pre-cutover cldf/) to produce the 17-column cross-check file"
    )


def load_derivation(path="cldf/derivation.csv"):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [tuple(r) for r in list(csv.reader(f))[1:] if len(r) == 2]


def write_edges(edges, path="cldf/edges.csv"):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(EDGES_HEADER)
        w.writerows(edges)


def main():
    rows = load_unified()
    deriv = load_derivation()
    edges, status_by_id, stats = build_edges(rows, deriv)
    write_edges(edges)
    print(
        f"cldf/edges.csv: {len(edges)} edges from {len(rows)} nodes "
        f"({len(deriv)} derivation pairs)",
        file=sys.stderr,
    )
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}", file=sys.stderr)


if __name__ == "__main__":
    main()
