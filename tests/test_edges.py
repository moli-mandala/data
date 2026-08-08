"""Integration checks for the typed edge table (cldf/edges.csv) against the legacy columns.

Runs on the built cldf/ like the other integration tests. While the legacy graph columns still
exist in forms.csv (pre-cutover), every edge is cross-checked against them; the only permitted
divergences are the enumerated waiver classes, with exact counts:

  - kind change borrowed→variant (rows whose Relation was flipped in place by
    mark_cross_family_borrowings while Variant_Of survived)
  - kind change variant→reflex on cross-entry Variant_Of rows (contradictory double parents,
    demoted to rank-2 review edges)

The effective-origin equivalence (variant-chain walk == legacy Origin_ID) holds with NO waivers.
"""

import csv
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edges_build import (  # noqa: E402
    EDGES_HEADER,
    NOTE_ALTERNATE,
    NOTE_CROSS_ENTRY,
    build_edges,
    load_derivation,
    load_unified,
)

_ID, _FORM, _ORIGIN, _REL, _RD, _VOF, _BF = 0, 2, 11, 13, 14, 15, 16


def _strip(pid):
    return pid[1:] if pid and pid[0] in ">~" else pid


@pytest.fixture(scope="module")
def graph():
    if not (os.path.exists("cldf/forms.csv") or os.path.exists("cldf/forms-legacy.csv")):
        pytest.skip("cldf not built")
    try:
        rows = load_unified()
    except SystemExit:
        pytest.skip("legacy cross-check columns unavailable (run unify_cldf.py --legacy-cols)")
    deriv = load_derivation()
    if not deriv:
        pytest.skip("derivation intermediate unavailable (run unify_cldf.py --legacy-cols)")
    edges, status_by_id, stats = build_edges(rows, deriv)
    return {
        "rows": rows,
        "by_id": {r[_ID]: r for r in rows},
        "deriv": deriv,
        "edges": edges,
        "status": status_by_id,
        "stats": stats,
        "rank1": {
            e[0]: (e[1], e[2]) for e in edges if e[3] == 1 and e[2] in ("reflex", "borrowed", "variant")
        },
    }


def test_header_and_written_file(graph):
    if os.path.exists("cldf/edges.csv"):
        with open("cldf/edges.csv", encoding="utf-8") as f:
            assert next(csv.reader(f)) == EDGES_HEADER


def test_status_matches_relation(graph):
    for r in graph["rows"]:
        status = graph["status"][r[_ID]]
        if r[_REL] == "":
            assert status == "entry"
        elif r[_REL] == "local":
            assert status == "unlinked"
        else:
            assert status == ""


def test_every_attested_row_has_exactly_one_rank1_edge(graph):
    attested = [r for r in graph["rows"] if r[_REL] in ("reflex", "variant", "borrowed")]
    assert len(attested) == len(graph["rank1"])
    for r in attested:
        assert r[_ID] in graph["rank1"]


def test_origin_reconstruction_roundtrip(graph):
    """The legacy Origin_ID is recoverable from the rank-1 edge for every attested row:
    reflex/borrowed → the target; variant → the target for parent-encodings, the target's own
    origin for sibling-encodings. Only the cross-entry rows (where the classifier deliberately
    chose the etymon over the contradictory pointer) may differ."""
    rank1 = graph["rank1"]
    by_id = graph["by_id"]
    bad = []
    for r in graph["rows"]:
        if r[_REL] not in ("reflex", "variant", "borrowed"):
            continue
        legacy_origin = _strip(r[_ORIGIN])
        target, kind = rank1[r[_ID]]
        if kind in ("reflex", "borrowed"):
            reconstructed = target
        else:  # variant: sibling-encoding iff the target shares the row's legacy origin
            target_origin = _strip(by_id[target][_ORIGIN]) if target in by_id else ""
            reconstructed = (
                target_origin if target != legacy_origin and target_origin == legacy_origin else target
            )
        if reconstructed != legacy_origin:
            bad.append((r[_ID], legacy_origin, kind, target))
    # even the cross-entry rows round-trip: their rank-1 edge IS the etymon link, and the
    # contradictory pointer survives as the rank-2 review edge
    assert not bad, f"{len(bad)} origin round-trip failures, e.g. {bad[:5]}"


def test_immediate_parent_preserved_modulo_cross_entry(graph):
    """Rank-1 target == legacy COALESCE(Variant_Of, Origin) except the cross-entry rows."""
    diverged = 0
    for r in graph["rows"]:
        if r[_REL] not in ("reflex", "variant", "borrowed"):
            continue
        legacy = (
            _strip(r[_VOF])
            if r[_REL] in ("variant", "borrowed") and r[_VOF]
            else _strip(r[_ORIGIN])
        )
        new = graph["rank1"][r[_ID]][0]
        if new != legacy:
            diverged += 1
    assert diverged == graph["stats"].get("waiver_cross_entry", 0)


def test_kind_changes_are_enumerated(graph):
    stats = graph["stats"]
    changed = 0
    cross_borrowed_kept = 0
    for r in graph["rows"]:
        if r[_REL] not in ("reflex", "variant", "borrowed"):
            continue
        kind = graph["rank1"][r[_ID]][1]
        if kind != r[_REL]:
            changed += 1
            if r[_REL] == "borrowed":
                assert kind == "variant"
            else:
                assert (r[_REL], kind) == ("variant", "reflex")
    expected = stats.get("waiver_borrowed_to_variant", 0)
    # cross-entry rows: variant rows become reflex; the borrowed cross-entry row keeps its kind
    for e in graph["edges"]:
        if e[6] == NOTE_CROSS_ENTRY and graph["by_id"][e[0]][_REL] == "borrowed":
            cross_borrowed_kept += 1
    expected += stats.get("waiver_cross_entry", 0) - cross_borrowed_kept
    assert changed == expected


def test_derivation_edges_fully_accounted(graph):
    stats = graph["stats"]
    distinct_pairs = len({tuple(p) for p in graph["deriv"]})
    accounted = (
        stats.get("dedup_vs_rank1", 0)
        + stats.get("alt_edges", 0)
        + stats.get("component_edges", 0)
        + stats.get("derived_root", 0)
        + stats.get("derived_single", 0)
    )
    assert accounted == distinct_pairs


def test_component_groups_ordered_and_reviewables_marked(graph):
    by_child = {}
    for child, parent, kind, rank, pos, _s, note in graph["edges"]:
        if kind == "component":
            by_child.setdefault(child, []).append(int(pos))
        if note == NOTE_ALTERNATE:
            assert kind == "reflex" and rank >= 2
    for child, poss in by_child.items():
        assert sorted(poss) == list(range(1, len(poss) + 1)) and len(poss) >= 2
    assert sum(1 for e in graph["edges"] if e[6] == NOTE_ALTERNATE) == graph["stats"].get(
        "alt_reviewable", 0
    )


def test_redirect_stubs_have_no_rank1_edge(graph):
    """Redirect stubs are etyma (no rank-1 edge of their own). They MAY be edge parents — the
    merges re-pointer deliberately leaves borrowed children on the addendum (23 such rows in the
    2026-08 corpus, present in the shipped DB too) — so only the outgoing side is constrained."""
    redirected = {r[_ID] for r in graph["rows"] if r[_RD] and not r[_REL]}
    attested_with_redirect = [r[_ID] for r in graph["rows"] if r[_RD] and r[_REL]]
    # attested rows with a stale Redirect lose it at serialization (v1/v2 builders already
    # dropped it silently); tracked via the redirect_dropped_on_attested stat
    assert len(attested_with_redirect) == graph["stats"].get("redirect_dropped_on_attested", 0)
    for rid in redirected:
        assert rid not in graph["rank1"], f"redirect stub {rid} has a rank-1 edge"
    into = sum(1 for e in graph["edges"] if e[1] in redirected and e[3] == 1)
    legacy_into = sum(
        1
        for r in graph["rows"]
        if r[_REL] in ("reflex", "variant", "borrowed") and _strip(r[_ORIGIN]) in redirected
    )
    assert into == legacy_into, f"stub-parent edges {into} != legacy references {legacy_into}"
