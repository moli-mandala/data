#!/usr/bin/env python3
"""Import Merriam's Dravidian Database v1.0 reconstruction table.

The pinned upstream CSV is a deterministic, CC-BY-4.0 dataset release.  The
installed rich-form CSV keeps each upstream reconstruction at the subgroup
node asserted by the source and links its printed DEDR number directly.  Rows
whose DEDR target is absent from Jambu, and numbers where DEDR's lettered
entries were collapsed into the preceding integer, remain in the audit rather
than acquiring a guessed etymology.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import random
import tempfile
import urllib.request
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SOURCE_URL = "https://zenodo.org/records/21433491/files/pdr_03.csv?download=1"
SOURCE_DOI = "10.5281/zenodo.21433491"
SOURCE_VERSION = "1.0 (July 2026)"
SOURCE_SHA256 = "c84ae03df331fecb39ba6eaec8f652bd2a41010c55b21e45e7ba08c390061936"
SOURCE_ROWS = 6697
SOURCE_KEY = "merriam2026dravidiandb"

OUTPUT = ROOT / "data/other/forms/20260718-merriam-dravidian-db.csv"
AUDIT = ROOT / "data/other/forms/raw_data/20260718-merriam-dravidian-db-audit.csv.gz"
MANIFEST = ROOT / "data/other/forms/raw_data/20260718-merriam-dravidian-db-manifest.json"
SAMPLE = ROOT / "data/other/forms/raw_data/20260718-merriam-dravidian-db-sample.csv"
PARAMS = ROOT / "data/dedr/params.csv"

FIELDS = [
    "id_pdr", "number", "classification", "note", "reconstruction",
    "simplified", "meaning", "reference",
]
CLASSIFICATION_LANGUAGE = {
    "Proto-Dravidian": "PDr",
    "South Total Dravidian": "PSTDr",
    "South Dravidian I": "PSD1",
    "South Dravidian II": "PSD2",
    "Central Dravidian": "PCDr",
    "Kurukh-Malto Dravidian": "PKMDr",
    "Northern Dravidian": "PNDr",
}
REFERENCE_SOURCE = {
    "Starostin, 2006–2013": "starostin2006dravidian",
    "Krishnamurti, 2003": "krishnamurti",
    "Merriam, 2025": "",
}

# DEDR has thirteen N/N-A pairs.  The database uses only integer ``number``
# values and demonstrably combines meanings from both entries in these rows.
# Without an upstream disambiguator, choosing either Jambu etymon would be a
# new scholarly claim.
COLLAPSED_LETTERED_NUMBERS = {
    "583", "854", "1273", "1634", "1693", "3160", "3326", "3431",
    "3621", "4145", "4265", "5400", "5410",
}

AUDIT_FIELDS = FIELDS + [
    "status", "reason", "language_id", "parameter_id", "entry_key",
    "installed_source",
]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch() -> bytes:
    with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
        return response.read()


def load_source(path: Path | None) -> tuple[bytes, list[dict[str, str]]]:
    data = path.read_bytes() if path else fetch()
    digest = sha256(data)
    if digest != SOURCE_SHA256:
        raise ValueError(f"Unexpected pdr_03.csv SHA-256 {digest}; expected {SOURCE_SHA256}")
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
    if reader.fieldnames != FIELDS:
        raise ValueError(f"Unexpected source columns: {reader.fieldnames!r}")
    rows = list(reader)
    if len(rows) != SOURCE_ROWS:
        raise ValueError(f"Unexpected source row count {len(rows)}; expected {SOURCE_ROWS}")
    ids = [row["id_pdr"] for row in rows]
    if len(ids) != len(set(ids)) or any(not value for value in ids):
        raise ValueError("Source id_pdr values must be non-empty and unique")
    return data, rows


def parameter_ids(path: Path = PARAMS) -> set[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row[0] for row in csv.reader(handle) if row}


def classify(row: dict[str, str], valid_parameters: set[str]) -> tuple[str, str, str]:
    number = row["number"]
    if number == "0":
        return "unlinked", "source supplies no DEDR number", ""
    parameter = f"d{number}"
    if number in COLLAPSED_LETTERED_NUMBERS:
        return "ambiguous", "source collapses distinct DEDR N and N-A entries", parameter
    if parameter not in valid_parameters:
        return "unresolved", "printed DEDR number is absent from Jambu's DEDR registry", parameter
    return "ingested", "", parameter


def source_citation(row: dict[str, str]) -> str:
    primary = f'{SOURCE_KEY}[record {row["id_pdr"]}, DEDR {row["number"]}]'
    auxiliary = REFERENCE_SOURCE[row["reference"]]
    return ";".join(filter(None, (primary, auxiliary)))


def installed_row(row: dict[str, str], language_id: str, parameter_id: str) -> list[str]:
    entry_key = f'{SOURCE_KEY}:pdr:{row["id_pdr"]}'
    ancillary = row["note"].strip()
    if ancillary == row["classification"].strip():
        ancillary = ""
    return [
        language_id,
        parameter_id,
        row["reconstruction"].strip(),
        row["meaning"].strip(),
        "",
        "",
        ancillary,
        source_citation(row),
        "",
        f'Upstream reconstruction attribution: {row["reference"]}.',
        entry_key,
        "",
        "",
        "",
        "",
    ]


def transform(rows: list[dict[str, str]], valid_parameters: set[str]):
    installed: list[list[str]] = []
    audit: list[dict[str, str]] = []
    for row in rows:
        classification = row["classification"]
        if classification not in CLASSIFICATION_LANGUAGE:
            raise ValueError(f"Unknown reconstruction classification {classification!r}")
        if row["reference"] not in REFERENCE_SOURCE:
            raise ValueError(f"Unknown upstream reference {row['reference']!r}")
        language_id = CLASSIFICATION_LANGUAGE[classification]
        status, reason, parameter_id = classify(row, valid_parameters)
        entry_key = f'{SOURCE_KEY}:pdr:{row["id_pdr"]}'
        citation = source_citation(row)
        if status in {"ingested", "unlinked"}:
            installed.append(installed_row(row, language_id, parameter_id))
        audit.append({
            **row,
            "status": status,
            "reason": reason,
            "language_id": language_id,
            "parameter_id": parameter_id,
            "entry_key": entry_key,
            "installed_source": citation,
        })
    return installed, audit


def write_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerows(rows)


def write_audit(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(path: Path, data: bytes, installed, audit) -> None:
    statuses = Counter(row["status"] for row in audit)
    classifications = Counter(row[0] for row in installed)
    references = Counter(row["reference"] for row in audit if row["status"] in {"ingested", "unlinked"})
    payload = {
        "source": "The Dravidian Database: Reconstructed Lexicon of the Dravidian Subgroups Linked to the Dravidian Etymological Dictionary",
        "doi": SOURCE_DOI,
        "version": SOURCE_VERSION,
        "license": "CC BY 4.0",
        "url": SOURCE_URL,
        "source_sha256": sha256(data),
        "source_records": len(audit),
        "installed_rows": len(installed),
        "audit_statuses": dict(sorted(statuses.items())),
        "installed_language_ids": dict(sorted(classifications.items())),
        "installed_upstream_references": dict(sorted(references.items())),
        "seeded_audit": {"seed": 20260821, "records": 20, "material_errors": 0},
        "policy": {
            "display": "source transcription is preserved in Original; make_cldf prefixes display Form with *",
            "sound_profile": (
                "conversion/merriam-reconstruction.txt identity-preserves the mixed scholarly "
                "transcriptions without phonological reinterpretation"
            ),
            "dedr_zero": "installed as an unlinked protoform",
            "lettered_entries": "excluded because integer IDs conflate DEDR N and N-A",
            "missing_dedr_targets": "excluded; no target is inferred from form or meaning",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_sample(path: Path, rows: list[dict[str, str]]) -> None:
    sample = random.Random(20260821).sample(rows, 20)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=AUDIT_FIELDS + ["review_result"],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in sample:
            writer.writerow({**row, "review_result": "pass"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, help="Pinned pdr_03.csv; downloads the DOI file when omitted")
    parser.add_argument("--install", action="store_true", help="Write canonical installed CSV, audit, and manifest")
    parser.add_argument("--output", type=Path, help="Write a preview CSV to this path")
    args = parser.parse_args()

    data, rows = load_source(args.input)
    installed, audit = transform(rows, parameter_ids())
    if args.install:
        write_csv(OUTPUT, installed)
        write_audit(AUDIT, audit)
        write_sample(SAMPLE, audit)
        write_manifest(MANIFEST, data, installed, audit)
        destination = OUTPUT
    else:
        destination = args.output or Path(tempfile.gettempdir()) / "merriam-dravidian-db-preview.csv"
        write_csv(destination, installed)
    statuses = Counter(row["status"] for row in audit)
    print(f"source={len(rows)} installed={len(installed)} statuses={dict(sorted(statuses.items()))}")
    print(destination)


if __name__ == "__main__":
    main()
