#!/usr/bin/env python3
"""
reconcile_form_ids.py — one-off repair of durable-ID churn caused by the historic
nondeterministic field joins in make_cldf.py.

The committed data/form-identities.csv registry stores fingerprints computed from '; '-joined
multi-value fields whose order was hash-randomised per run. Now that make_cldf joins
deterministically, one final rebuild mints fresh f_ ids for every row whose historic join order
differed. This script maps each freshly-minted id back to the registry identity whose *baseline*
row is order-insensitively identical, rewrites the current cldf/ in place, and updates the
registry fingerprints to the new deterministic values — after which rebuilds are stable and this
script has nothing left to do.

Run immediately after assign_form_ids.py (before concepts.py / align.py):

    python reconcile_form_ids.py --baseline /path/to/pre-migration-cldf
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

from assign_form_ids import (
    ALIASES,
    FORMS,
    REGISTRY,
    REGISTRY_FIELDS,
    fingerprint,
    read_rows,
    write_rows,
)

ROOT = Path(__file__).resolve().parent


def oi_key(row: dict[str, str]) -> tuple:
    """Order-insensitive identity over the same fields fingerprint() uses."""

    def parts(value: str, sep: str) -> tuple:
        return tuple(sorted(p.strip() for p in (value or "").split(sep) if p.strip()))

    source = tuple(sorted(t.split("[", 1)[0].strip() for t in (row.get("Source") or "").split(";") if t))
    return (
        source,
        (row.get("Language_ID") or "").strip(),
        parts(row.get("Original") or row.get("Form") or "", ";"),
        parts(row.get("Gloss") or "", ";"),
        parts(row.get("Native") or "", ";"),
    )


def rewrite_csv(path: Path, columns: tuple[str, ...], mapping: dict[str, str]) -> int:
    if not path.exists():
        return 0
    fields, rows = read_rows(path)
    if not fields:
        return 0
    hits = 0
    for row in rows:
        for column in columns:
            if column in row and row[column] in mapping:
                row[column] = mapping[row[column]]
                hits += 1
    write_rows(path, fields, rows)
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True, help="pre-migration cldf/ snapshot")
    args = ap.parse_args()

    with (args.baseline / "forms.csv").open(encoding="utf-8") as f:
        baseline_rows = {r["ID"]: r for r in csv.DictReader(f)}
    fields, forms = read_rows(FORMS)
    new_ids = {r["ID"] for r in forms}
    lost = {i: baseline_rows[i] for i in baseline_rows.keys() - new_ids if i.startswith("f_")}
    minted = [r for r in forms if r["ID"].startswith("f_") and r["ID"] not in baseline_rows]

    lost_by_key: dict[tuple, list[str]] = defaultdict(list)
    for lid, row in sorted(lost.items()):
        lost_by_key[oi_key(row)].append(lid)

    mapping: dict[str, str] = {}  # minted id → restored registry id
    for row in sorted(minted, key=lambda r: r["ID"]):
        candidates = lost_by_key.get(oi_key(row))
        if candidates:
            mapping[row["ID"]] = candidates.pop(0)

    unmatched_minted = sum(1 for r in minted if r["ID"] not in mapping)
    unmatched_lost = sum(len(v) for v in lost_by_key.values())
    print(
        f"reconcile: {len(minted)} minted / {len(lost)} lost registry ids; "
        f"matched {len(mapping)}, leftover minted {unmatched_minted}, leftover lost {unmatched_lost}",
        file=sys.stderr,
    )
    if not mapping:
        return

    # rewrite the current cldf in place (concepts.py / align.py run after this, so their
    # outputs are born with the restored ids)
    n = 0
    n += rewrite_csv(FORMS, ("ID", "Redirect"), mapping)
    n += rewrite_csv(ROOT / "cldf/edges.csv", ("Child_ID", "Parent_ID"), mapping)
    n += rewrite_csv(ROOT / "cldf/form-source-keys.csv", ("Legacy_ID",), mapping)
    n += rewrite_csv(ROOT / "cldf/forms-legacy.csv",
                     ("ID", "Origin_ID", "Redirect", "Variant_Of", "Borrowed_From"), mapping)
    n += rewrite_csv(ROOT / "cldf/derivation.csv", ("Child_ID", "Parent_ID"), mapping)
    n += rewrite_csv(ALIASES, ("Legacy_ID", "Form_ID"), mapping)
    n += rewrite_csv(ROOT / "data/etymology-assignments.csv", ("Form_ID", "Etymon_ID"), mapping)

    # registry: drop the minted rows, restore the historic rows with refreshed fingerprints
    _, registry = read_rows(REGISTRY)
    fields_by_new = {r["ID"]: r for r in read_rows(FORMS)[1]}
    restored = {v: k for k, v in mapping.items()}  # registry id → minted id it replaces
    out = []
    for row in registry:
        fid = row.get("Form_ID", "")
        if fid in mapping:  # a freshly-minted identity being retired
            continue
        if fid in restored:
            src = fields_by_new[fid]
            row["Fingerprint"] = fingerprint(src)
            row["Source"] = src.get("Source", "")
            row["Language_ID"] = src.get("Language_ID", "")
            row["Original"] = src.get("Original", "") or src.get("Form", "")
            row["Gloss"] = src.get("Gloss", "")
            row["Status"] = "active"
        out.append(row)
    write_rows(REGISTRY, REGISTRY_FIELDS, out)

    # aliases that pointed a legacy id at a now-retired minted id were rewritten above; drop
    # self-aliases that may result (Legacy == Form after restoration)
    afields, aliases = read_rows(ALIASES)
    aliases = [a for a in aliases if a.get("Legacy_ID") != a.get("Form_ID")]
    write_rows(ALIASES, afields or ["Legacy_ID", "Form_ID"], aliases)

    print(f"reconcile: rewrote {n} references across cldf/, registry updated", file=sys.stderr)


if __name__ == "__main__":
    main()
