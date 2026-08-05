#!/usr/bin/env python3
"""Assign durable, content-independent public IDs to form nodes.

The legacy importer numbers reflexes from file order and etymon-local counters.  Those IDs change
when files are inserted, records are reordered, or a form is assigned to a different etymon.  This
post-unification pass gives every form an opaque ``f_…`` ID and rewrites the completed graph
atomically, except where the entry has a native CDIAL or DEDR identifier.  Those dictionary IDs
remain unchanged because they are stable source identifiers and useful citations.

Identity lives in ``data/form-identities.csv``.  A fingerprint is only a reconciliation aid: it is
computed from source transcription and provenance, never used again as the identity once a form
has entered the registry.  Consequently profile/normalisation changes do not change IDs.  When a
source supplies no immutable record key, a simultaneous source-text edit and row reorder may need
manual registry reconciliation; the script refuses ambiguous matches rather than silently moving
an ID to another form.

Run after ``unify_cldf.py`` and before ``concepts.py`` / ``align.py``.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FORMS = ROOT / "cldf/forms.csv"
REGISTRY = ROOT / "data/form-identities.csv"
ALIASES = ROOT / "cldf/form-id-aliases.csv"
ASSIGNMENTS = ROOT / "data/etymology-assignments.csv"
SOURCE_KEYS = ROOT / "cldf/form-source-keys.csv"
GRAPH_FILE_COLUMNS = {
    "derivation.csv": ("Child_ID", "Parent_ID"),
    "merges.csv": ("Addendum_ID", "Main_ID"),
    # These are normally regenerated later in the pipeline. Rewriting them too keeps a checkout
    # internally consistent immediately after the one-time ID migration.
    "form_concepts.csv": ("Form_ID",),
    "alignments.csv": ("Form_ID", "Origin_ID"),
}

REGISTRY_FIELDS = [
    "Form_ID", "Legacy_ID", "Source_Key", "Fingerprint", "Source", "Language_ID", "Original",
    "Gloss", "Status",
]
ASSIGNMENT_FIELDS = ["Form_ID", "Etymon_ID", "Relation", "Status", "Notes"]


def normalized(value: str) -> str:
    return " ".join((value or "").strip().split())


def source_identity(value: str) -> str:
    """Citation locators are mutable metadata, not part of a form's identity."""
    bare = re.sub(r"\[[^\]]*\]", "", value or "")
    return ";".join(sorted(filter(None, (normalized(item) for item in bare.split(";")))))


def fingerprint(row: dict[str, str], source_key: str = "") -> str:
    """Fingerprint raw-ish provenance, deliberately excluding generated transcription and graph."""
    if source_key:
        return hashlib.blake2b(
            ("jambu-source-key-v1\x1f" + source_key).encode("utf-8"), digest_size=16
        ).hexdigest()
    identity = "\x1f".join(
        (
            source_identity(row.get("Source", "")),
            normalized(row.get("Language_ID", "")),
            normalized(row.get("Original", "") or row.get("Form", "")),
            normalized(row.get("Gloss", "")),
            normalized(row.get("Native", "")),
        )
    )
    return hashlib.blake2b(identity.encode("utf-8"), digest_size=16).hexdigest()


def mint_id(fp: str, discriminator: str, used: set[str]) -> str:
    nonce = 0
    while True:
        seed = f"jambu-form-v1\x1f{fp}\x1f{discriminator}\x1f{nonce}".encode()
        # 64 bits gives a ~1 in 7.4 billion birthday-collision chance at 370k forms; the explicit
        # used-set check makes a collision harmless while keeping URLs substantially shorter.
        token = base64.b32encode(hashlib.blake2b(seed, digest_size=8).digest()).decode()
        candidate = "f_" + token.rstrip("=").lower()
        if candidate not in used:
            return candidate
        nonce += 1


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        return [], []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_rows(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def has_dictionary_entry_id(row: dict[str, str]) -> bool:
    """Keep only native CDIAL and DEDR entry identifiers as public IDs."""
    form_id = row.get("ID", "")
    language_id = row.get("Language_ID", "")
    sources = {part.strip() for part in row.get("Source", "").split(";")}
    is_cdial = language_id == "Indo-Aryan" and (
        bool(re.fullmatch(r"\d+[a-z]?", form_id))
        or (not row.get("Relation") and "CDIAL" in sources)
    )
    is_dedr = language_id == "PDr" and bool(re.fullmatch(r"d\d+", form_id))
    return is_cdial or is_dedr


def assign_ids(
    forms: list[dict[str, str]], registry: list[dict[str, str]], source_keys: dict[str, str] | None = None
) -> tuple[dict[str, str], list[dict[str, str]]]:
    source_keys = source_keys or {}
    by_form_id = {row["Form_ID"]: row for row in registry if row.get("Form_ID")}
    by_legacy = {row["Legacy_ID"]: row for row in registry if row.get("Legacy_ID")}
    by_fp: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in registry:
        if row.get("Fingerprint"):
            by_fp[row["Fingerprint"]].append(row)

    used = set(by_form_id)
    claimed: set[str] = set()
    mapping: dict[str, str] = {}
    snapshots: dict[str, dict[str, str]] = {}

    for row in forms:
        old_id = row["ID"]
        if has_dictionary_entry_id(row):
            continue
        source_key = source_keys.get(old_id, "")
        fp = fingerprint(row, source_key)
        match = by_form_id.get(old_id)
        legacy_match = by_legacy.get(old_id)
        if not match and legacy_match and legacy_match.get("Fingerprint") == fp:
            match = legacy_match
        if not match:
            candidates = [candidate for candidate in by_fp.get(fp, []) if candidate["Form_ID"] not in claimed]
            if len(candidates) == 1:
                match = candidates[0]
        if not match and legacy_match:
            # This preserves a corrected source row when its old generated position did not move.
            same_source_record = bool(source_key and legacy_match.get("Source_Key") == source_key) or (
                source_identity(legacy_match.get("Source", ""))
                == source_identity(row.get("Source", ""))
                and legacy_match.get("Language_ID") == row.get("Language_ID", "")
                and legacy_match.get("Original") == (row.get("Original", "") or row.get("Form", ""))
            )
            if same_source_record:
                match = legacy_match

        if match:
            form_id = match["Form_ID"]
            if form_id in claimed:
                match = None
        if not match:
            form_id = mint_id(fp, old_id, used)
            used.add(form_id)

        claimed.add(form_id)
        mapping[old_id] = form_id
        snapshots[form_id] = {
            "Form_ID": form_id,
            "Legacy_ID": old_id if not old_id.startswith("f_") else by_form_id.get(old_id, {}).get("Legacy_ID", ""),
            "Source_Key": source_key or (match.get("Source_Key", "") if match else ""),
            "Fingerprint": fp,
            "Source": row.get("Source", ""),
            "Language_ID": row.get("Language_ID", ""),
            "Original": row.get("Original", "") or row.get("Form", ""),
            "Gloss": row.get("Gloss", ""),
            "Status": "active",
        }

    for old in registry:
        if old.get("Form_ID") not in snapshots:
            tombstone = dict(old)
            tombstone["Status"] = "retired"
            snapshots[old["Form_ID"]] = tombstone
    return mapping, sorted(snapshots.values(), key=lambda row: row["Form_ID"])


def rewrite_graph_file(path: Path, columns: tuple[str, ...], mapping: dict[str, str]) -> None:
    fields, rows = read_rows(path)
    if not fields:
        return
    for row in rows:
        for column in columns:
            row[column] = mapping.get(row.get(column, ""), row.get(column, ""))
    write_rows(path, fields, rows)


def apply_assignments(forms: list[dict[str, str]], assignments: list[dict[str, str]]) -> int:
    by_id = {row["ID"]: row for row in forms}
    active = {row["ID"] for row in forms if row.get("Relation") != "local"}
    changed = 0
    for assignment in assignments:
        if assignment.get("Status", "accepted").strip().lower() not in {"accepted", "yes", "active"}:
            continue
        form_id = assignment.get("Form_ID", "").strip()
        etymon_id = assignment.get("Etymon_ID", "").strip()
        relation = assignment.get("Relation", "reflex").strip() or "reflex"
        if form_id not in by_id:
            raise ValueError(f"etymology assignment references missing form {form_id}")
        if etymon_id not in active:
            raise ValueError(f"etymology assignment for {form_id} references missing etymon {etymon_id}")
        if relation not in {"reflex", "borrowed"}:
            raise ValueError(f"unsupported assignment relation {relation!r} for {form_id}")
        row = by_id[form_id]
        if row.get("Origin_ID") != etymon_id or row.get("Relation") != relation:
            row["Origin_ID"] = etymon_id
            row["Relation"] = relation
            row["Borrowed_From"] = etymon_id if relation == "borrowed" else ""
            changed += 1
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forms", type=Path, default=FORMS)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--aliases", type=Path, default=ALIASES)
    parser.add_argument("--assignments", type=Path, default=ASSIGNMENTS)
    parser.add_argument("--source-keys", type=Path, default=SOURCE_KEYS)
    parser.add_argument(
        "--fresh", action="store_true",
        help="replace a just-created registry during an unshipped migration (never use after curation)",
    )
    args = parser.parse_args()

    fields, forms = read_rows(args.forms)
    if not forms or "Relation" not in fields:
        raise ValueError(f"{args.forms} is not a unified Jambu forms table")
    _, registry = read_rows(args.registry)
    if args.fresh:
        reverse = {
            row["Form_ID"]: row["Legacy_ID"]
            for row in registry
            if row.get("Form_ID") and row.get("Legacy_ID") and row.get("Status") == "active"
        }
        reference_columns = ("Origin_ID", "Redirect", "Variant_Of", "Borrowed_From")
        for row in forms:
            row["ID"] = reverse.get(row["ID"], row["ID"])
            for column in reference_columns:
                row[column] = reverse.get(row.get(column, ""), row.get(column, ""))
        for name, columns in GRAPH_FILE_COLUMNS.items():
            rewrite_graph_file(args.forms.parent / name, columns, reverse)
        registry = []
    _, source_key_rows = read_rows(args.source_keys)
    source_keys = {
        row["Legacy_ID"]: row["Source_Key"]
        for row in source_key_rows
        if row.get("Legacy_ID") and row.get("Source_Key")
    }
    mapping, next_registry = assign_ids(forms, registry, source_keys)

    _, old_aliases = read_rows(args.aliases)
    if args.fresh:
        old_aliases = []
    aliases = {row["Legacy_ID"]: row["Form_ID"] for row in old_aliases if row.get("Legacy_ID")}
    aliases.update({old: new for old, new in mapping.items() if old != new})

    reference_columns = ("Origin_ID", "Redirect", "Variant_Of", "Borrowed_From")
    for row in forms:
        row["ID"] = mapping.get(row["ID"], row["ID"])
        for column in reference_columns:
            row[column] = mapping.get(row.get(column, ""), row.get(column, ""))

    active_ids = {row["ID"] for row in forms}
    # A policy migration may restore a source-owned dictionary ID that an earlier run aliased to
    # an opaque ID. Prefer the now-active dictionary ID and discard that obsolete redirect.
    restored_ids = {
        form_id: legacy
        for legacy, form_id in aliases.items()
        if legacy in active_ids and form_id not in active_ids
    }
    aliases = {legacy: form_id for legacy, form_id in aliases.items() if legacy not in active_ids}

    if not args.assignments.exists():
        write_rows(args.assignments, ASSIGNMENT_FIELDS, [])
    _, assignments = read_rows(args.assignments)
    for assignment in assignments:
        for column in ("Form_ID", "Etymon_ID"):
            value = restored_ids.get(assignment.get(column, ""), assignment.get(column, ""))
            assignment[column] = value if value in active_ids else aliases.get(value, value)
    changed = apply_assignments(forms, assignments)

    # Do not mutate sidecar graph files until every assignment has validated. A bad assignment
    # must leave the whole pre-ID build intact rather than producing a half-rewritten graph.
    for name, columns in GRAPH_FILE_COLUMNS.items():
        rewrite_graph_file(args.forms.parent / name, columns, mapping)

    write_rows(args.forms, fields, forms)
    write_rows(args.registry, REGISTRY_FIELDS, next_registry)
    write_rows(args.assignments, ASSIGNMENT_FIELDS, assignments)
    write_rows(
        args.aliases,
        ["Legacy_ID", "Form_ID"],
        [
            {"Legacy_ID": legacy, "Form_ID": form_id}
            for legacy, form_id in sorted(aliases.items())
            if legacy != form_id
        ],
    )
    print(
        f"assigned {len(mapping):,} durable form IDs; "
        f"preserved {len(aliases):,} aliases; applied {changed:,} etymology assignments"
    )


if __name__ == "__main__":
    main()
