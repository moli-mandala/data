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
    "edges.csv": ("Child_ID", "Parent_ID"),
    # build intermediate; present only under `unify_cldf.py --legacy-cols` (else a no-op)
    "derivation.csv": ("Child_ID", "Parent_ID"),
    "forms-legacy.csv": ("ID", "Origin_ID", "Redirect", "Variant_Of", "Borrowed_From"),
    "merges.csv": ("Addendum_ID", "Main_ID"),
    # Optional structured source-prose sidecar consumed by jambu-static's DB builder.
    "entry-texts.csv": ("Form_ID",),
    # Records which evidence row supplied each DEDR display head. Keep the pointer durable even
    # when its source form receives an opaque public ID in this pass.
    "pdr-headword-audit.csv": ("Source_Form_ID",),
    # Article-level cross-family links normally retain native DEDR/CDIAL IDs, but keeping them in
    # the generic rewrite map makes the sidecar safe if either endpoint is ever canonicalized.
    "comparisons.csv": ("Entry_ID", "Compared_Entry_ID"),
    # These are normally regenerated later in the pipeline. Rewriting them too keeps a checkout
    # internally consistent immediately after the one-time ID migration.
    "form_concepts.csv": ("Form_ID",),
    "alignments.csv": ("Form_ID", "Origin_ID"),
}

REGISTRY_FIELDS = [
    "Form_ID", "Legacy_ID", "Source_Key", "Fingerprint", "Source", "Language_ID", "Original",
    "Gloss", "Status",
]
ASSIGNMENT_FIELDS = ["Form_ID", "Etymon_ID", "Kind", "Rank", "Status", "Source", "Notes"]
EDGES_FIELDS = ["Child_ID", "Parent_ID", "Kind", "Rank", "Pos", "Source", "Note"]


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
        or (row.get("Status") == "entry" and "CDIAL" in sources)
    )
    is_dedr = language_id == "PDr" and bool(re.fullmatch(r"d\d+", form_id))
    return is_cdial or is_dedr


def assign_ids(
    forms: list[dict[str, str]], registry: list[dict[str, str]], source_keys: dict[str, str] | None = None
) -> tuple[dict[str, str], list[dict[str, str]]]:
    source_keys = source_keys or {}
    by_form_id = {row["Form_ID"]: row for row in registry if row.get("Form_ID")}
    # A retired tombstone can legitimately retain a positional legacy ID that was later reused.
    # Prefer the active identity regardless of CSV sort order; otherwise a retired row can steal
    # the lookup and cause curated assignments to appear to reference a missing form.
    by_legacy: dict[str, dict[str, str]] = {}
    for row in registry:
        legacy_id = row.get("Legacy_ID", "")
        if legacy_id and (
            legacy_id not in by_legacy or row.get("Status") == "active"
        ):
            by_legacy[legacy_id] = row
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


ACCEPTED = {"accepted", "yes", "active"}
REJECTED = {"rejected", "no"}


def migrate_assignment_schema(assignments: list[dict[str, str]]) -> None:
    """One-time upgrade of legacy overlay rows (Relation column, implicit rank 1)."""
    for row in assignments:
        if "Kind" not in row or not row.get("Kind"):
            row["Kind"] = (row.get("Relation") or "reflex").strip() or "reflex"
        if not row.get("Rank"):
            row["Rank"] = "1"


def validate_assignments(forms: list[dict[str, str]], assignments: list[dict[str, str]]) -> None:
    """Hard-fail before any file is mutated (same contract as the legacy overlay)."""
    by_id = {row["ID"]: row for row in forms}
    linkable = {row["ID"] for row in forms if row.get("Status") != "unlinked"}
    for assignment in assignments:
        status = assignment.get("Status", "accepted").strip().lower()
        if status not in ACCEPTED | REJECTED:
            raise ValueError(f"unsupported assignment status {status!r}")
        form_id = assignment.get("Form_ID", "").strip()
        etymon_id = assignment.get("Etymon_ID", "").strip()
        if form_id not in by_id:
            raise ValueError(f"etymology assignment references missing form {form_id}")
        if status in REJECTED:
            continue
        if etymon_id not in linkable:
            raise ValueError(f"etymology assignment for {form_id} references missing etymon {etymon_id}")
        if assignment.get("Kind") not in {"reflex", "borrowed"}:
            raise ValueError(f"unsupported assignment kind {assignment.get('Kind')!r} for {form_id}")
        if not re.fullmatch(r"[1-9]\d*", assignment.get("Rank", "1")):
            raise ValueError(f"bad assignment rank {assignment.get('Rank')!r} for {form_id}")


def apply_assignments(
    edges_path: Path, forms: list[dict[str, str]], assignments: list[dict[str, str]]
) -> int:
    """Patch the curated overlay into cldf/edges.csv (rank-1 upserts replace the accepted
    etymology; rank≥2 upserts add hypotheses; rejected rows delete generated non-primary edges).
    A form gaining a rank-1 edge stops being `unlinked`."""
    fields, edges = read_rows(edges_path)
    if not fields:
        raise ValueError(f"{edges_path} missing — run unify_cldf.py first")
    by_form = {row["ID"]: row for row in forms}
    rank1_by_child = {}
    for edge in edges:
        if edge.get("Rank") == "1" and edge.get("Kind") in {"reflex", "borrowed", "variant"}:
            rank1_by_child[edge["Child_ID"]] = edge
    changed = 0
    for assignment in assignments:
        status = assignment.get("Status", "accepted").strip().lower()
        form_id = assignment.get("Form_ID", "").strip()
        etymon_id = assignment.get("Etymon_ID", "").strip()
        kind = assignment.get("Kind", "reflex")
        rank = assignment.get("Rank", "1")
        if status in REJECTED:
            before = len(edges)
            edges = [
                e for e in edges
                if not (e["Child_ID"] == form_id and e["Parent_ID"] == etymon_id and e["Rank"] != "1")
            ]
            changed += before - len(edges)
            continue
        if rank == "1":
            existing = rank1_by_child.get(form_id)
            if existing is not None:
                if (existing["Parent_ID"], existing["Kind"]) != (etymon_id, kind):
                    existing.update(
                        Parent_ID=etymon_id, Kind=kind,
                        Source=assignment.get("Source", ""), Note="",
                    )
                    changed += 1
            else:
                edge = dict(
                    Child_ID=form_id, Parent_ID=etymon_id, Kind=kind, Rank="1", Pos="",
                    Source=assignment.get("Source", ""), Note="",
                )
                edges.append(edge)
                rank1_by_child[form_id] = edge
                changed += 1
            row = by_form.get(form_id)
            if row is not None and row.get("Status") == "unlinked":
                row["Status"] = ""
                changed += 1
        else:
            match = [
                e for e in edges
                if e["Child_ID"] == form_id and e["Parent_ID"] == etymon_id and e["Rank"] != "1"
            ]
            if match:
                for e in match:
                    if e.get("Note", "").startswith("review:") or e.get("Kind") != kind:
                        e.update(Kind=kind, Rank=rank, Source=assignment.get("Source", ""), Note="")
                        changed += 1
            else:
                edges.append(dict(
                    Child_ID=form_id, Parent_ID=etymon_id, Kind=kind, Rank=rank, Pos="",
                    Source=assignment.get("Source", ""), Note="",
                ))
                changed += 1
    edges.sort(key=lambda e: (
        e["Child_ID"], e["Kind"], int(e["Rank"] or 1), int(e["Pos"] or 0), e["Parent_ID"]
    ))
    write_rows(edges_path, EDGES_FIELDS, edges)
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
    if not forms or "Status" not in fields or "Redirect" not in fields:
        raise ValueError(f"{args.forms} is not a unified Jambu forms table (edge-model format)")
    _, registry = read_rows(args.registry)
    if args.fresh:
        reverse = {
            row["Form_ID"]: row["Legacy_ID"]
            for row in registry
            if row.get("Form_ID") and row.get("Legacy_ID") and row.get("Status") == "active"
        }
        for row in forms:
            row["ID"] = reverse.get(row["ID"], row["ID"])
            row["Redirect"] = reverse.get(row.get("Redirect", ""), row.get("Redirect", ""))
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

    for row in forms:
        row["ID"] = mapping.get(row["ID"], row["ID"])
        row["Redirect"] = mapping.get(row.get("Redirect", ""), row.get("Redirect", ""))

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
    migrate_assignment_schema(assignments)
    validate_assignments(forms, assignments)

    # Do not mutate sidecar graph files until every assignment has validated. A bad assignment
    # must leave the whole pre-ID build intact rather than producing a half-rewritten graph.
    for name, columns in GRAPH_FILE_COLUMNS.items():
        rewrite_graph_file(args.forms.parent / name, columns, mapping)

    changed = apply_assignments(args.forms.parent / "edges.csv", forms, assignments)

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
