#!/usr/bin/env python3
"""
checkpoint_b.py — one-shot verification of the edge-model data cutover (run after `make all`).

    python tests/checkpoint_b.py BASELINE_CLDF_DIR

Checks, against a pre-migration cldf/ snapshot:
  1. The durable form-ID set is byte-identical (no f_-ID churn from the Status predicate swap).
  2. alignments.csv is identical modulo the enumerated aligned-parent waivers: forms may GAIN
     alignments (variant chains through attested parents) or align against a nearer proto
     ancestor — but every waived form is enumerated and counted, and no other row changed.
  3. edges.csv structural totals match the Phase-1 classifier stats.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from edges_util import PROTO_LANGS, aligned_parent, load_edges, rank1_map  # noqa: E402

baseline = Path(sys.argv[1])
new = Path("cldf")
failures = 0


def fail(msg):
    global failures
    failures += 1
    print(f"FAIL: {msg}")


def ok(msg):
    print(f"ok: {msg}")


# ---- 1. durable ID set ----------------------------------------------------------------------

with (baseline / "forms.csv").open(encoding="utf-8") as f:
    old_ids = {r["ID"] for r in csv.DictReader(f)}
with (new / "forms.csv").open(encoding="utf-8") as f:
    new_rows = list(csv.DictReader(f))
new_ids = {r["ID"] for r in new_rows}
if old_ids != new_ids:
    fail(
        f"ID set changed: {len(old_ids - new_ids)} lost (e.g. {sorted(old_ids - new_ids)[:5]}), "
        f"{len(new_ids - old_ids)} gained (e.g. {sorted(new_ids - old_ids)[:5]})"
    )
else:
    ok(f"durable ID set identical ({len(new_ids)} ids)")

# ---- 2. alignments modulo enumerated waivers ------------------------------------------------

def load_alignment_rows(path):
    by_form = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for row in csv.reader(f):
            by_form[row[0]].append(tuple(row[1:]))
    return by_form


old_align = load_alignment_rows(baseline / "alignments.csv")
new_align = load_alignment_rows(new / "alignments.csv")

# expected waivers, recomputed from the shipped edges + the legacy cross-check columns
edges = load_edges(str(new / "edges.csv"))
rank1 = rank1_map(edges)
lang_of = {r["ID"]: r["Language_ID"] for r in new_rows}
legacy_origin = {}
legacy_path = new / "forms-legacy.csv"
if legacy_path.exists():
    with legacy_path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            o = r["Origin_ID"]
            legacy_origin[r["ID"]] = o[1:] if o and o[0] in ">~" else o

expected_gain, expected_change, expected_loss = set(), set(), set()
for fid, legacy in legacy_origin.items():
    new_parent = aligned_parent(rank1, lang_of, fid, PROTO_LANGS)
    la = lang_of.get(legacy) in PROTO_LANGS
    na = new_parent is not None and lang_of.get(new_parent) in PROTO_LANGS
    if legacy == new_parent or not (la or na):
        continue
    if na and not la:
        expected_gain.add(fid)
    elif la and not na:
        expected_loss.add(fid)
    else:
        expected_change.add(fid)

waived = expected_gain | expected_change | expected_loss
diff_forms = set()
for fid in set(old_align) | set(new_align):
    if old_align.get(fid, []) != new_align.get(fid, []):
        diff_forms.add(fid)
unexpected = diff_forms - waived
if unexpected:
    fail(f"{len(unexpected)} forms' alignments changed outside the waiver set, e.g. {sorted(unexpected)[:5]}")
else:
    ok(
        f"alignments identical outside waivers "
        f"(+{len(expected_gain & diff_forms)} gained, {len(expected_change & diff_forms)} re-anchored, "
        f"-{len(expected_loss & diff_forms)} lost; {len(waived - diff_forms)} waivers unused)"
    )

# ---- 3. edge totals -------------------------------------------------------------------------

kinds = defaultdict(int)
for e in edges:
    key = (e["Kind"], "1" if e["Rank"] == "1" else "2+")
    kinds[key] += 1
print("edge totals:", dict(sorted(kinds.items())))
rank1_total = sum(n for (k, r), n in kinds.items() if r == "1" and k in ("reflex", "borrowed", "variant"))
attested = sum(1 for r in new_rows if r["Status"] == "")
if rank1_total != attested:
    fail(f"rank-1 attestation edges {rank1_total} != attested rows {attested}")
else:
    ok(f"rank-1 edges cover exactly the {attested} attested rows")

print("\nALL CHECKS PASSED" if not failures else f"\n{failures} FAILURES")
sys.exit(1 if failures else 0)
