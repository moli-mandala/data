"""Shared durable-overlay support for OCR post-correction workflows.

The browser workbench writes one small corrections CSV per source. Importers
read it through this module, which rejects decisions made against an older
version of the audit. Generated audit files remain reproducible and are never
used as an editing surface.
"""

from __future__ import annotations

import csv
import hashlib
import unicodedata
from dataclasses import dataclass
from pathlib import Path


CORRECTION_FIELDS = (
    "Entry_Key",
    "Status",
    "Form",
    "POS",
    "Gloss",
    "Notes",
    "Audit_Fingerprint",
    "Updated_At",
)
REVIEW_STATUSES = frozenset({"accepted", "corrected", "illegible", "skipped"})


@dataclass(frozen=True)
class OcrCorrection:
    entry_key: str
    status: str
    form: str
    pos: str
    gloss: str
    notes: str
    audit_fingerprint: str
    updated_at: str


def audit_fingerprint(headers: list[str], row: dict[str, str]) -> str:
    """Match the workbench's stable fingerprint of one exact audit record."""
    payload = "\x1e".join(f"{header}\0{row.get(header, '')}" for header in headers)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def audit_fingerprints(path: Path, key_field: str | None = None) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"OCR audit has no header: {path}")
        key_field = key_field or next(
            (field for field in reader.fieldnames if field.casefold() == "entry_key"),
            None,
        )
        if not key_field:
            raise ValueError(f"OCR audit has no Entry_Key column: {path}")
        result: dict[str, str] = {}
        for row in reader:
            key = row.get(key_field, "").strip()
            if not key:
                raise ValueError(f"OCR audit contains an empty Entry_Key: {path}")
            if key in result:
                raise ValueError(f"duplicate OCR audit Entry_Key {key}: {path}")
            result[key] = audit_fingerprint(reader.fieldnames, row)
    return result


def load_corrections(
    path: Path, audit_path: Path, key_field: str | None = None
) -> dict[str, OcrCorrection]:
    """Read an overlay and prove every decision still matches its audit row."""
    if not path.exists():
        return {}
    fingerprints = audit_fingerprints(audit_path, key_field)
    corrections: dict[str, OcrCorrection] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(CORRECTION_FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"OCR correction overlay is missing {sorted(missing)}: {path}")
        for row in reader:
            key = row["Entry_Key"].strip()
            if not key:
                raise ValueError(f"OCR correction overlay contains an empty Entry_Key: {path}")
            if key in corrections:
                raise ValueError(f"duplicate OCR correction Entry_Key {key}: {path}")
            if key not in fingerprints:
                raise ValueError(f"OCR correction refers to unknown audit entry {key}: {path}")
            status = row["Status"].strip()
            if status not in REVIEW_STATUSES:
                raise ValueError(f"invalid OCR correction status {status!r} for {key}")
            form = unicodedata.normalize("NFC", row["Form"].strip())
            if status in {"accepted", "corrected"} and not form:
                raise ValueError(f"reviewed OCR entry has an empty form: {key}")
            expected = fingerprints[key]
            actual = row["Audit_Fingerprint"].strip()
            if actual != expected:
                raise ValueError(
                    f"stale OCR correction for {key}: audit is {expected}, overlay records {actual or 'no fingerprint'}"
                )
            corrections[key] = OcrCorrection(
                entry_key=key,
                status=status,
                form=form,
                pos=unicodedata.normalize("NFC", row["POS"].strip()),
                gloss=unicodedata.normalize("NFC", row["Gloss"].strip()),
                notes=unicodedata.normalize("NFC", row["Notes"].strip()),
                audit_fingerprint=actual,
                updated_at=row["Updated_At"].strip(),
            )
    return corrections
