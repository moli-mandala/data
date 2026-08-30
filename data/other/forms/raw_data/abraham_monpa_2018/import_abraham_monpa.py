#!/usr/bin/env python3
"""Install the complete Western Arunachal / Monpa survey matrices.

The source is the CC-BY-4.0 Lexibank v3.0 digitisation of Abraham, Sako,
Kinny and Zeliang's SIL survey.  The release contains three compact source
matrices (Monpa, Kho-Bwa, and Hruso/Miji) plus a 307-concept list.

This importer deliberately reads those matrices rather than Lexibank's
generated ``cldf/forms.csv``.  The generated CLDF silently loses one of the
two source concepts labelled ``fat`` and roughly eighty Kho-Bwa rows whose
labels lost spaces (for example ``cookedrice``).  Reading the matrices by
their source order recovers every printed cell without guessing a form.

Run with ``--refresh-snapshot /path/to/abrahammonpa-v3.0.zip`` only when
refreshing the frozen upstream snapshot.  Ordinary runs are fully offline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
SNAPSHOT = HERE / "snapshot"
OUTPUT = ROOT / "data/other/forms/20260828-sil-western-arunachal-monpa.csv"
AUDIT = HERE.parent / "20260828-sil-western-arunachal-monpa-audit.csv"
MANIFEST = HERE.parent / "20260828-sil-western-arunachal-monpa-manifest.json"

SOURCE_KEY = "abraham-sako-kinny-zeliang2018"
RELEASE = "lexibank/abrahammonpa v3.0"
RELEASE_DOI = "10.5281/zenodo.5115885"
RELEASE_COMMIT = "d6b890a30c36e7fc4d38a9a6841e2dd2dd569521"
RELEASE_ZIP_SHA256 = "09a930bb46d1b43c512e83dbc13d7ebd30710a7ecdd895114b27f159bab60fbb"

SNAPSHOT_MEMBERS = {
    "abrahammonpa-3.0/raw/monpa.tsv": "monpa.tsv",
    "abrahammonpa-3.0/raw/khobwa.tsv": "khobwa.tsv",
    "abrahammonpa-3.0/raw/hruso.tsv": "hruso.tsv",
    "abrahammonpa-3.0/raw/abraham2018-concepts.tsv": "concepts.tsv",
    "abrahammonpa-3.0/etc/languages.tsv": "languages.tsv",
    "abrahammonpa-3.0/.zenodo.json": "zenodo.json",
    "abrahammonpa-3.0/LICENSE": "LICENSE",
}

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic",
    "Notes", "Source", "Cognateset", "Etymology", "Entry_Key",
    "Variant_Of_Key", "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "Snapshot_Table", "Source_Row", "Concept_Number", "Source_Item",
    "Source_Gloss", "Canonical_Gloss", "Source_Lect", "Upstream_Language_ID",
    "Jambu_Language_ID", "Dialect_ID", "Raw_Cell", "Form_Index",
    "Transcription", "Status", "Reason", "Entry_Key",
]

# Upstream language ID -> (Jambu parent language, site label).  The five Bugun
# sites already exist from Abraham & Sako (2021); all other sites get
# source-specific dialect records when this source is registered.
LECTS = {
    "MonpaKalaktang": ("KalaktangMonpa", "Kalaktang"),
    "MonpaBalemu": ("KalaktangMonpa", "Balemu"),
    "MonpaTomko": ("KalaktangMonpa", "Tomko"),
    "MonpaNamsu": ("KalaktangMonpa", "Namsu"),
    "MonpaTembang": ("Tshangla", "Tembang"),
    "MonpaDirangDum": ("Tshangla", "Dirang Dum"),
    "MonpaSangti": ("Tshangla", "Sangti"),
    "MonpaDirang": ("Tshangla", "Dirang"),
    "MonpaTawangMonastery": ("Dakpakha", "Tawang Monastery"),
    "MonpaZimithang": ("Dakpakha", "Zimithang"),
    "MonpaChangprong": ("Dakpakha", "Changprong"),
    "SartangKhoitam": ("Khoitam", "Khoitam"),
    "ChugParchu": ("Chug", "Parchu"),
    "SartangDarbuA": ("Sartang", "Darbu 1"),
    "MijiNafra": ("Miji", "Nafra"),
    "LishLish": ("Lish", "Lish"),
    "SherdukpenRupa": ("Sherdukpen", "Rupa"),
    "HrusoAkaJamiri": ("Hruso", "Jamiri"),
    "SartangDarbuB": ("Sartang", "Darbu 2"),
    "SartangKhoina": ("Khoina", "Khoina"),
    "BugunSingchung": ("Bugun", "Singchung"),
    "BugunWangho": ("Bugun", "Wangho"),
    "BugunBichom": ("Bugun", "Bichom"),
    "BugunKaspi": ("Bugun", "Kaspi"),
    "BugunNamphri": ("Bugun", "Namphri"),
    "NamreiNabolang": ("Miji", "Nabolang"),
    "NamreiBisai": ("Miji", "Bisai"),
    "DammaiRurang": ("Miji", "Rurang"),
    "DammaiDibin": ("Miji", "Dibin"),
    "SherdukpenShergaon": ("Sherdukpen", "Shergaon"),
}

EXISTING_DIALECTS = {
    "BugunSingchung": "bugun_singchung",
    "BugunWangho": "bugun_wangho",
    "BugunBichom": "bugun_bichom",
    "BugunKaspi": "bugun_kaspi",
    "BugunNamphri": "bugun_namphri",
}

HEADER_OVERRIDES = {"BugunSingchung": "Bugun (Singchung)"}
GAPS = {"", "-", "–"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def refresh_snapshot(zip_path: Path) -> None:
    payload = zip_path.read_bytes()
    digest = sha256(payload)
    if digest != RELEASE_ZIP_SHA256:
        raise ValueError(f"release zip SHA-256 {digest}, expected {RELEASE_ZIP_SHA256}")
    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for member, name in SNAPSHOT_MEMBERS.items():
            (SNAPSHOT / name).write_bytes(archive.read(member))


def normalise_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def slug(value: str) -> str:
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def dialect_id(upstream_id: str) -> str:
    return EXISTING_DIALECTS.get(upstream_id, f"abrahammonpa2018_{slug(upstream_id)}")


def dialect_tag(language_id: str, source_id: str, name: str) -> str:
    return (
        f"dialect:{quote(language_id, safe='')}:{quote(source_id, safe='')}:"
        f"{quote(name, safe='')}"
    )


def read_tsv(name: str) -> list[dict[str, str]]:
    with (SNAPSHOT / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def load_concepts() -> list[dict[str, str]]:
    concepts = read_tsv("concepts.tsv")
    assert len(concepts) == 307
    assert [int(row["NUMBER"]) for row in concepts] == list(range(1, 308))
    assert Counter(row["ENGLISH"] for row in concepts)["fat"] == 2
    return concepts


def canonical_gloss(concept: dict[str, str]) -> str:
    number = int(concept["NUMBER"])
    if number == 81:
        return "fat (organic substance)"
    if number == 82:
        return "fat (obese)"
    return concept["ENGLISH"]


def language_metadata() -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    languages = {row["ID"]: row for row in read_tsv("languages.tsv")}
    assert set(languages) == set(LECTS)
    headers: dict[str, str] = {}
    for upstream_id, row in languages.items():
        header = HEADER_OVERRIDES.get(upstream_id) or row["WiktionaryName"]
        if not header:
            raise ValueError(f"no source-table header for {upstream_id}")
        if header in headers:
            raise ValueError(f"duplicate source-table header {header}")
        headers[header] = upstream_id
    return languages, headers


def split_forms(raw: str | None) -> list[str]:
    if raw is None:
        return []
    # Two Monpa cells contain a `` || ̃ai za`` trailing annotation.  Lexibank's
    # source orthography profile likewise treats ``||`` as a comment boundary;
    # retain the complete cell in the audit but install only the lexical form.
    raw = raw.split(" || ", 1)[0]
    return [
        unicodedata.normalize("NFC", part.strip())
        for part in raw.split(";")
        if part.strip() not in GAPS
    ]


def concept_for_hruso(
    source_item: str,
    source_gloss: str,
    concepts_by_label: dict[str, list[dict[str, str]]],
) -> dict[str, str]:
    matches = concepts_by_label[normalise_label(source_gloss)]
    if len(matches) == 1:
        return matches[0]
    # In the numbered Hruso/Miji list, item 119 is anatomical/organic fat and
    # item 250 is the adjective 'obese'; the surrounding source headings make
    # that distinction explicit even though both English cells say 'fat'.
    if source_gloss == "fat" and source_item in {"119", "250"}:
        return matches[0 if source_item == "119" else 1]
    raise ValueError(f"ambiguous concept {source_item} {source_gloss!r}: {matches}")


def concept_for_unnumbered(
    source_gloss: str,
    concepts_by_label: dict[str, list[dict[str, str]]],
    duplicate_occurrences: Counter[str],
) -> dict[str, str]:
    """Resolve an unnumbered matrix row, including the two distinct fat rows.

    Monpa and Kho-Bwa contain the same 307 headings but do not keep identical
    row order throughout, so the normalised heading is authoritative.  Only
    ``fat`` is duplicated; in both tables its anatomical sense precedes its
    adjectival sense, matching the source concept inventory.
    """
    label = normalise_label(source_gloss)
    matches = concepts_by_label[label]
    if len(matches) == 1:
        return matches[0]
    occurrence = duplicate_occurrences[label]
    duplicate_occurrences[label] += 1
    if label == "fat" and occurrence < len(matches):
        return matches[occurrence]
    raise ValueError(f"ambiguous unnumbered concept {source_gloss!r}: {matches}")


def extract() -> tuple[list[list[str]], list[dict[str, str]], dict[str, object]]:
    concepts = load_concepts()
    languages, header_to_upstream = language_metadata()
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for concept in concepts:
        by_label[normalise_label(concept["ENGLISH"])].append(concept)

    installed: list[list[str]] = []
    audit: list[dict[str, str]] = []
    table_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    parent_counts: Counter[str] = Counter()
    cell_counts: Counter[str] = Counter()

    for table in ("monpa.tsv", "khobwa.tsv", "hruso.tsv"):
        rows = read_tsv(table)
        duplicate_occurrences: Counter[str] = Counter()
        headers = [key for key in rows[0] if key not in {"", "No.", "Gloss"}]
        unknown_headers = set(headers) - set(header_to_upstream)
        if unknown_headers:
            raise ValueError(f"unknown {table} headers: {sorted(unknown_headers)}")

        for data_index, row in enumerate(rows, start=1):
            source_row = data_index + 1  # include the TSV header line
            if table == "hruso.tsv":
                source_item = row["No."]
                source_gloss = row["Gloss"]
                concept = concept_for_hruso(source_item, source_gloss, by_label)
            else:
                source_item = ""
                source_gloss = row[""]
                concept = concept_for_unnumbered(
                    source_gloss, by_label, duplicate_occurrences
                )

            gloss = canonical_gloss(concept)
            for header in headers:
                upstream_id = header_to_upstream[header]
                parent, site = LECTS[upstream_id]
                source_dialect = dialect_id(upstream_id)
                raw_cell = row.get(header) or ""
                forms = split_forms(raw_cell)
                cell_counts["all"] += 1
                if " || " in raw_cell:
                    cell_counts["trailing_annotation"] += 1

                if not forms:
                    cell_counts["gaps"] += 1
                    audit.append({
                        "Snapshot_Table": table,
                        "Source_Row": str(source_row),
                        "Concept_Number": concept["NUMBER"],
                        "Source_Item": source_item,
                        "Source_Gloss": source_gloss,
                        "Canonical_Gloss": gloss,
                        "Source_Lect": header,
                        "Upstream_Language_ID": upstream_id,
                        "Jambu_Language_ID": parent,
                        "Dialect_ID": source_dialect,
                        "Raw_Cell": raw_cell,
                        "Form_Index": "",
                        "Transcription": "",
                        "Status": "excluded",
                        "Reason": "explicit dash or blank source cell",
                        "Entry_Key": "",
                    })
                    continue

                cell_counts["filled"] += 1
                if len(forms) > 1:
                    cell_counts["multi_form"] += 1
                for form_index, form in enumerate(forms, start=1):
                    source_code = languages[upstream_id]["Source_ID"]
                    entry_key = (
                        f"abrahammonpa2018:{table.removesuffix('.tsv')}:"
                        f"r{source_row:03d}:{source_code}:v{form_index}"
                    )
                    citation = (
                        f"{SOURCE_KEY}[Lexibank v3.0, {table}, row {source_row}, "
                        f"{header}]"
                    )
                    note = (
                        f"CC-BY Lexibank v3.0 digitization of the SIL survey; "
                        f"source table {table}; source gloss {source_gloss}; "
                        f"source lect {header}"
                    )
                    if len(forms) > 1:
                        note += f"; alternative {form_index} of {len(forms)} in source cell"
                    if " || " in raw_cell:
                        note += "; trailing source annotation retained in the audit"
                    installed.append([
                        parent, "", form, gloss, "", form, note, citation,
                        "", "", entry_key, "", "", "",
                        dialect_tag(parent, source_dialect, site),
                    ])
                    audit.append({
                        "Snapshot_Table": table,
                        "Source_Row": str(source_row),
                        "Concept_Number": concept["NUMBER"],
                        "Source_Item": source_item,
                        "Source_Gloss": source_gloss,
                        "Canonical_Gloss": gloss,
                        "Source_Lect": header,
                        "Upstream_Language_ID": upstream_id,
                        "Jambu_Language_ID": parent,
                        "Dialect_ID": source_dialect,
                        "Raw_Cell": raw_cell,
                        "Form_Index": str(form_index),
                        "Transcription": form,
                        "Status": "installed",
                        "Reason": (
                            "source-matrix form; trailing annotation audit-only"
                            if " || " in raw_cell else "source-matrix form"
                        ),
                        "Entry_Key": entry_key,
                    })
                    table_counts[table] += 1
                    language_counts[upstream_id] += 1
                    parent_counts[parent] += 1

    if cell_counts != Counter(
        all=9210, filled=9068, gaps=142, multi_form=210, trailing_annotation=2
    ):
        raise ValueError(f"source-cell topology changed: {cell_counts}")
    if len(installed) != 9279 or len(audit) != 9421:
        raise ValueError(f"expected 9279 installed / 9421 audit, got {len(installed)} / {len(audit)}")
    if len({row[10] for row in installed}) != len(installed):
        raise ValueError("duplicate entry key")
    if any(unicodedata.normalize("NFC", row[2]) != row[2] for row in installed):
        raise ValueError("non-NFC installed form")

    snapshot_hashes = {
        name: sha256((SNAPSHOT / name).read_bytes())
        for name in sorted(SNAPSHOT_MEMBERS.values())
    }
    manifest: dict[str, object] = {
        "source_key": SOURCE_KEY,
        "source_title": (
            "Sociolinguistic Research among Selected Groups in Western Arunachal "
            "Pradesh: Highlighting Monpa"
        ),
        "upstream_release": RELEASE,
        "upstream_release_doi": RELEASE_DOI,
        "upstream_commit": RELEASE_COMMIT,
        "upstream_release_zip_sha256": RELEASE_ZIP_SHA256,
        "license": "CC-BY-4.0",
        "snapshot_sha256": snapshot_hashes,
        "source_tables": 3,
        "source_concepts": 307,
        "source_lects": 30,
        "source_cells": dict(cell_counts),
        "installed_forms": len(installed),
        "audit_rows": len(audit),
        "installed_by_table": dict(sorted(table_counts.items())),
        "installed_by_upstream_language": dict(sorted(language_counts.items())),
        "installed_by_jambu_language": dict(sorted(parent_counts.items())),
        "upstream_cldf_forms": 8213,
        "forms_recovered_beyond_upstream_cldf": len(installed) - 8213,
        "upstream_cldf_loss_notes": [
            "concept 81 'fat' was overwritten by the duplicate label for concept 82",
            "Kho-Bwa labels with removed spaces failed the upstream exact-name lookup",
        ],
        "unparsed_rows": 0,
        "unmapped_concepts": 0,
        "unmapped_lects": 0,
    }
    return installed, audit, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-snapshot", type=Path,
        help="verified abrahammonpa v3.0 release zip (snapshot refresh only)",
    )
    args = parser.parse_args()
    if args.refresh_snapshot:
        refresh_snapshot(args.refresh_snapshot)

    missing = [name for name in SNAPSHOT_MEMBERS.values() if not (SNAPSHOT / name).exists()]
    if missing:
        raise SystemExit(f"missing frozen snapshot files: {', '.join(missing)}")

    installed, audit, manifest = extract()
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(installed)
    with AUDIT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(audit)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(
        f"installed={len(installed)} audit={len(audit)} gaps="
        f"{sum(row['Status'] == 'excluded' for row in audit)} "
        f"recovered_beyond_upstream_cldf={len(installed) - 8213}"
    )


if __name__ == "__main__":
    main()
