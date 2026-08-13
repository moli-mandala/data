#!/usr/bin/env python3
"""One-time migration from ``Language: Dialect`` rows to dialect tags.

The migration is deliberately mechanical: it preserves the browser builder's
existing parent-selection rule, writes every removed lect to ``dialects.csv``,
normalizes compiled forms, and updates the persistent form identity snapshots.
Future raw builds use the same registry through :mod:`dialects`.
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote

from dialects import dialect_tag, normalize_dialect
from backfill_language_coordinates import backfill


LANGUAGE_FIELDS = [
    "ID", "Name", "Glottocode", "Latitude", "Longitude", "Clade", "Location", "Quality"
]
DIALECT_FIELDS = [
    "ID", "Tag", "Language_ID", "Source_Language_ID", "Name", "Glottocode",
    "Latitude", "Longitude", "Clade", "Location", "Quality",
]
BASE_LANGUAGE_OVERRIDES = {"Hindi": "Hindi-Urdu"}


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def write_rows(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def build_registries(
    language_rows: list[dict[str, str]], form_rows: list[dict[str, str]]
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, dict[str, str]]]:
    exact = {row["Name"]: row for row in language_rows if ": " not in row["Name"]}
    colon_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in language_rows:
        if ": " in row["Name"]:
            colon_groups[row["Name"].split(": ", 1)[0]].append(row)

    parents: dict[str, dict[str, str]] = {}
    synthesized_ids: set[str] = set()
    for printed_base, members in colon_groups.items():
        base = BASE_LANGUAGE_OVERRIDES.get(printed_base, printed_base)
        parent = exact.get(base)
        if parent is None:
            parent = dict(members[0])
            parent.update({
                "Name": base, "Latitude": "", "Longitude": "", "Location": "", "Quality": ""
            })
            exact[base] = parent
            synthesized_ids.add(parent["ID"])
        parents[printed_base] = parent

    dialect_rows: list[dict[str, str]] = []
    aliases: dict[str, dict[str, str]] = {}
    for row in language_rows:
        if ": " not in row["Name"]:
            continue
        printed_base, name = row["Name"].split(": ", 1)
        parent_id = parents[printed_base]["ID"]
        entry = {
            "ID": row["ID"],
            "Tag": dialect_tag(parent_id, row["ID"], name),
            "Language_ID": parent_id,
            "Source_Language_ID": row["ID"],
            "Name": name,
            "Glottocode": row.get("Glottocode", ""),
            "Latitude": row.get("Latitude", ""),
            "Longitude": row.get("Longitude", ""),
            "Clade": row.get("Clade", ""),
            "Location": row.get("Location", ""),
            "Quality": row.get("Quality", ""),
        }
        dialect_rows.append(entry)
        aliases[row["ID"]] = entry

    base_rows: list[dict[str, str]] = []
    emitted_synthesized: set[str] = set()
    for row in language_rows:
        if ": " not in row["Name"]:
            base_rows.append(row)
        elif row["ID"] in synthesized_ids and row["ID"] not in emitted_synthesized:
            printed_base = row["Name"].split(": ", 1)[0]
            base_rows.append(parents[printed_base])
            emitted_synthesized.add(row["ID"])

    # Existing source parsers already emit useful simple dialect tags. Register
    # those too, so every dialect offered by the UI has an explicit parent even
    # when no former colon row supplied geographic metadata.
    tag_languages: dict[str, set[str]] = defaultdict(set)
    for row in form_rows:
        language_id, normalized_tags = normalize_dialect(
            row["Language_ID"], row.get("Tags", ""), aliases
        )
        for tag in (row.get("Tags") or "").split():
            if tag.startswith("dialect:") and tag in normalized_tags.split():
                tag_languages[tag].add(language_id)
    known_tags = {row["Tag"] for row in dialect_rows}
    clade_of = {row["ID"]: row.get("Clade", "") for row in base_rows}
    for tag, language_ids in sorted(tag_languages.items()):
        if tag in known_tags:
            continue
        if len(language_ids) != 1:
            raise ValueError(f"Dialect tag {tag!r} belongs to multiple languages: {language_ids}")
        language_id = next(iter(language_ids))
        dialect_rows.append({
            "ID": tag,
            "Tag": tag,
            "Language_ID": language_id,
            "Source_Language_ID": "",
            "Name": unquote(tag.rsplit(":", 1)[-1]),
            "Glottocode": "",
            "Latitude": "",
            "Longitude": "",
            "Clade": clade_of.get(language_id, ""),
            "Location": "",
            "Quality": "",
        })

    return base_rows, dialect_rows, aliases


def normalize_form_rows(
    rows: list[dict[str, str]], aliases: dict[str, dict[str, str]]
) -> int:
    changed = 0
    for row in rows:
        language_id, tags = normalize_dialect(row["Language_ID"], row.get("Tags", ""), aliases)
        if language_id != row["Language_ID"] or tags != row.get("Tags", ""):
            row["Language_ID"] = language_id
            row["Tags"] = tags
            changed += 1
    return changed


def migrate(root: Path) -> dict[str, int]:
    language_fields, language_rows = read_rows(root / "cldf/languages.csv")
    form_fields, form_rows = read_rows(root / "cldf/forms.csv")
    # This is a one-time migration, but make accidental reruns harmless. The
    # explicit registry becomes authoritative after the colon rows are gone.
    if not any(": " in row["Name"] for row in language_rows):
        _, dialect_rows = read_rows(root / "cldf/dialects.csv")
        return {
            "languages": len(language_rows),
            "dialects": len(dialect_rows),
            "forms": 0,
            "identities": 0,
        }
    base_rows, dialect_rows, aliases = build_registries(language_rows, form_rows)
    changed_forms = normalize_form_rows(form_rows, aliases)

    registry_path = root / "data/form-identities.csv"
    registry_fields, registry_rows = read_rows(registry_path)
    changed_identities = 0
    for row in registry_rows:
        target = aliases.get(row.get("Language_ID", ""))
        if target:
            row["Language_ID"] = target["Language_ID"]
            changed_identities += 1

    write_rows(root / "cldf/languages.csv", language_fields or LANGUAGE_FIELDS, base_rows)
    write_rows(root / "cldf/dialects.csv", DIALECT_FIELDS, dialect_rows)
    write_rows(root / "cldf/forms.csv", form_fields, form_rows)
    write_rows(registry_path, registry_fields, registry_rows)
    backfill(root / "cldf/languages.csv", root / "cldf/dialects.csv")
    return {
        "languages": len(base_rows),
        "dialects": len(dialect_rows),
        "forms": changed_forms,
        "identities": changed_identities,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    counts = migrate(args.root.resolve())
    print(", ".join(f"{key}={value}" for key, value in counts.items()))


if __name__ == "__main__":
    main()
