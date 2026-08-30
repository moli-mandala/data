#!/usr/bin/env python3
"""Expand the frozen hand-keyed Kurux page-39 lines into site-cell evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
LINES = PACKAGE / "manual_chunks/p039-items001-010-lines.tsv"
CELLS = PACKAGE / "manual_chunks/p039-items001-010-cells.tsv"
MANIFEST = PACKAGE / "source_manifest.json"
WORKSPACE = PACKAGE.parents[5]
SOURCE_PDF = WORKSPACE / "tmp/pdfs/kurux_manual/silesr2011_040.pdf"
SOURCE_IMAGE = WORKSPACE / "tmp/pdfs/kurux_manual/page-39.png"
SOURCE_PDF_SHA256 = "f2f06c25ac55462d6a40843539d8417e24a647bd1eb0bbe3f24ea3e45f0b9e4b"
SOURCE_IMAGE_SHA256 = "946ff8df00c62586ba1daeab766503297deb13d890c0a257c2268dafd9641e35"

# Printed survey codes remain literal until the identity/schema review. Code 0
# is retained as audit-only control evidence and must not be staged as Kurux.
SITES = {
    "A": ("site-code-A", "target"),
    "B": ("site-code-B", "target"),
    "C": ("site-code-C", "target"),
    "D": ("site-code-D", "target"),
    "E": ("site-code-E", "target"),
    "0": ("Bangla", "control"),
}
SITE_ORDER = {code: index for index, code in enumerate("ABCDE0")}
ALLOWED_STATUSES = {"attested", "blank", "ambiguous", "illegible", "not_used"}
FIELDS = [
    "physical_page", "printed_page", "line_id", "item", "gloss", "group",
    "site_code", "site", "role", "site_variant", "status", "form",
    "visible_base", "note", "evidence_sha256",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_lines() -> list[dict[str, str]]:
    with LINES.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(rows) == 38
    assert [row["line_id"] for row in rows] == [f"P039-L{i:03d}" for i in range(1, 39)]
    assert {row["status"] for row in rows} <= ALLOWED_STATUSES
    for row in rows:
        assert row["site_codes"] and set(row["site_codes"]) <= set(SITES)
        assert len(row["site_codes"]) == len(set(row["site_codes"]))
        for field in ("form", "visible_base", "note"):
            assert row[field] == unicodedata.normalize("NFC", row[field])
            assert "�" not in row[field]
            assert not any(0xE000 <= ord(char) <= 0xF8FF for char in row[field])
        if row["status"] == "attested":
            assert row["form"] and not row["visible_base"]
        elif row["status"] == "blank":
            assert not row["form"] and not row["visible_base"] and row["note"]
        else:
            assert not row["form"] and row["note"]
    return rows


def expand(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    cells: list[dict[str, str]] = []
    variants: Counter[tuple[str, str]] = Counter()
    for row in rows:
        for code in row["site_codes"]:
            site, role = SITES[code]
            coordinate = (row["item"], code)
            variants[coordinate] += 1
            cells.append({
                "physical_page": "39",
                "printed_page": "38",
                "line_id": row["line_id"],
                "item": row["item"],
                "gloss": row["gloss"],
                "group": row["group"],
                "site_code": code,
                "site": site,
                "role": role,
                "site_variant": str(variants[coordinate]),
                "status": row["status"],
                "form": row["form"],
                "visible_base": row["visible_base"],
                "note": row["note"],
                "evidence_sha256": SOURCE_IMAGE_SHA256,
            })
    cells.sort(key=lambda row: (
        int(row["item"]), SITE_ORDER[row["site_code"]], int(row["site_variant"])
    ))
    by_item: defaultdict[str, set[str]] = defaultdict(set)
    coordinate_counts: Counter[tuple[int, str]] = Counter()
    for row in cells:
        by_item[row["item"]].add(row["site_code"])
        coordinate_counts[(int(row["item"]), row["site_code"])] += 1
    assert set(map(int, by_item)) == set(range(1, 11))
    assert all(codes == set(SITES) for codes in by_item.values())
    assert {key: count for key, count in coordinate_counts.items() if count > 1} == {(3, "A"): 2}
    assert len(coordinate_counts) == 60
    assert len(cells) == 61
    return cells


def write(cells: list[dict[str, str]], lines: list[dict[str, str]]) -> None:
    with CELLS.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(cells)

    status = Counter(row["status"] for row in cells)
    role_status = Counter((row["role"], row["status"]) for row in cells)
    conceptual = {(int(row["item"]), row["site_code"]) for row in cells}
    conceptual_status = {}
    for coordinate in conceptual:
        states = {row["status"] for row in cells
                  if (int(row["item"]), row["site_code"]) == coordinate}
        assert len(states) == 1
        conceptual_status[coordinate] = states.pop()

    assert status == {"attested": 58, "blank": 3}
    assert role_status == {("target", "attested"): 48, ("target", "blank"): 3,
                           ("control", "attested"): 10}
    assert Counter(conceptual_status.values()) == {"attested": 57, "blank": 3}
    assert sha256(SOURCE_PDF) == SOURCE_PDF_SHA256
    assert sha256(SOURCE_IMAGE) == SOURCE_IMAGE_SHA256

    manifest = {
        "report": "ESR 2011-040 The Kurux of Bangladesh",
        "state": "partial_manual_review",
        "policy": (
            "Rendered page supplied every reading; OCR, PDF text, raw legacy glyphs, "
            "installed forms, and earlier audits supplied or verified none."
        ),
        "source_pdf": "tmp/pdfs/kurux_manual/silesr2011_040.pdf",
        "source_pdf_sha256": SOURCE_PDF_SHA256,
        "source_pdf_bytes": SOURCE_PDF.stat().st_size,
        "source_pdf_pages": 90,
        "wayback_timestamp": "20170809124903",
        "wayback_original_url": "http://www-01.sil.org/silesr/2011/silesr2011-040.pdf",
        "wordlist_physical_pages": [39, 57],
        "wordlist_printed_pages": [38, 56],
        "render_dpi": 300,
        "rendered_page_count": 19,
        "source_image": "tmp/pdfs/kurux_manual/page-39.png",
        "source_image_sha256": SOURCE_IMAGE_SHA256,
        "source_image_dimensions": [2550, 3300],
        "physical_pages_reviewed": [39],
        "printed_pages_reviewed": [38],
        "items_reviewed": [1, 10],
        "response_lines": len(lines),
        "conceptual_cells": len(conceptual),
        "expanded_cell_rows": len(cells),
        "site_variant_extra_rows": len(cells) - len(conceptual),
        "status_counts_expanded": dict(sorted(status.items())),
        "status_counts_conceptual": dict(sorted(Counter(conceptual_status.values()).items())),
        "target_counts_expanded": {
            key: role_status[("target", key)] for key in ("attested", "blank")
        },
        "control_counts_expanded": {
            key: role_status[("control", key)] for key in ("attested", "blank")
        },
        "blank_coordinates": ["item-001/site-B", "item-009/site-A", "item-010/site-A"],
        "ambiguous_coordinates": [],
        "illegible_coordinates": [],
        "not_used_coordinates": [],
        "site_variant_coordinates": ["item-003/site-A"],
        "site_identity_state": "pending_identity_and_schema_review",
        "control_policy": "Printed code 0 (Bangla) is retained audit-only pending identity/schema review.",
        "pending_items": [11, 307],
        "manual_lines_sha256": sha256(LINES),
        "expanded_cells_sha256": sha256(CELLS),
        "reconciliation_state": "not_started_after_manual_freeze",
    }
    with MANIFEST.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> None:
    lines = read_lines()
    cells = expand(lines)
    write(cells, lines)
    print(f"wrote {len(cells)} rows for 60 conceptual cells from {len(lines)} manual lines")


if __name__ == "__main__":
    # Compatibility entry point: always rebuild the complete frozen chunk set so
    # a historical command cannot roll the cumulative manifest back to one block.
    from build_manual_chunks import main as build_all_chunks

    build_all_chunks()
