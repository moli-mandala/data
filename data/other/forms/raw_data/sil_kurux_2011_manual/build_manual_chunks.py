#!/usr/bin/env python3
"""Expand all frozen Kurux manual chunks and write the cumulative manifest."""

from __future__ import annotations

import csv
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
MANIFEST = PACKAGE / "source_manifest.json"
WORKSPACE = PACKAGE.parents[5]
SOURCE_PDF = WORKSPACE / "tmp/pdfs/kurux_manual/silesr2011_040.pdf"
SOURCE_PDF_SHA256 = "f2f06c25ac55462d6a40843539d8417e24a647bd1eb0bbe3f24ea3e45f0b9e4b"
IMAGES = {
    39: {
        "path": WORKSPACE / "tmp/pdfs/kurux_manual/page-39.png",
        "sha256": "946ff8df00c62586ba1daeab766503297deb13d890c0a257c2268dafd9641e35",
        "printed_page": 38,
    },
    40: {
        "path": WORKSPACE / "tmp/pdfs/kurux_manual/page-40.png",
        "sha256": "a943fd4d96a00cc1c76aca4409550a5c06dba65ffedf9a23b0f054fffaf1851e",
        "printed_page": 39,
    },
    41: {
        "path": WORKSPACE / "tmp/pdfs/kurux_manual/page-41.png",
        "sha256": "1ae2acbb1a628ea21e1ad44868f1511e503c75d776c9e8613dbead650eb86131",
        "printed_page": 40,
    },
    42: {
        "path": WORKSPACE / "tmp/pdfs/kurux_manual/page-42.png",
        "sha256": "691e3a92fc69930111ee1a5f4b66cdbca12dc8b182bb59075dfd36840ce8564c",
        "printed_page": 41,
    },
    43: {
        "path": WORKSPACE / "tmp/pdfs/kurux_manual/page-43.png",
        "sha256": "5e24b3f741d34b4cf197ca849cd221ca25f22bf1948f020cd630cc23a7a684df",
        "printed_page": 42,
    },
    44: {
        "path": WORKSPACE / "tmp/pdfs/kurux_manual/page-44.png",
        "sha256": "bdd0f705bf6e00ec853e3b910fc4b8f5f8c469afa8940604d42b4714e6afceaa",
        "printed_page": 43,
    },
    45: {
        "path": WORKSPACE / "tmp/pdfs/kurux_manual/page-45.png",
        "sha256": "e7fd9d9b2c86eb59c5c9e48c0901cd81c06c87e0127eec280b83e1b0dcb3538d",
        "printed_page": 44,
    },
    46: {
        "path": WORKSPACE / "tmp/pdfs/kurux_manual/page-46.png",
        "sha256": "d09898f5a931b2897783a6b517c18dca6dc8df464faa53ec717ea91700b6dbd4",
        "printed_page": 45,
    },
    47: {
        "path": WORKSPACE / "tmp/pdfs/kurux_manual/page-47.png",
        "sha256": "826017bd8f270c01f07130290e70b400796d90d7c4b4a857935eda230de6238a",
        "printed_page": 46,
    },
    48: {
        "path": WORKSPACE / "tmp/pdfs/kurux_manual/page-48.png",
        "sha256": "9b085fe9413daa4926c5004f910e9cf0f551d301d3cacf0c361cbc9b78a4582b",
        "printed_page": 47,
    },
    49: {
        "path": WORKSPACE / "tmp/pdfs/kurux_manual/page-49.png",
        "sha256": "3961f6abf6887b0b57f57651e7c9f795f057cc2140fd6a1fcad0ba556096698c",
        "printed_page": 48,
    },
    50: {
        "path": WORKSPACE / "tmp/pdfs/kurux_manual/page-50.png",
        "sha256": "27e740e67e861b0f7a1fd1c98e255b9d88ea4c910e577b677d5cf73718dce601",
        "printed_page": 49,
    },
    51: {
        "path": WORKSPACE / "tmp/pdfs/kurux_manual/page-51.png",
        "sha256": "f523942c92303f3df63204c0d74ae05f6d0213fb674c8006c0d98b2c5f32024e",
        "printed_page": 50,
    },
    52: {
        "path": WORKSPACE / "tmp/pdfs/kurux_manual/page-52.png",
        "sha256": "6ed56b95d7351fc9207a7e2688a5665ca9bd508efdcbf780480d903878031171",
        "printed_page": 51,
    },
    53: {
        "path": WORKSPACE / "tmp/pdfs/kurux_manual/page-53.png",
        "sha256": "63b558e55499b53ca1c6c110af674e555eb4522c86f98d65c36664cc5fbbf381",
        "printed_page": 52,
    },
    54: {
        "path": WORKSPACE / "tmp/pdfs/kurux_manual/page-54.png",
        "sha256": "883c0edbe3cc51069187ab56854259e757a38801444369680a3d35f7f479717a",
        "printed_page": 53,
    },
    55: {
        "path": WORKSPACE / "tmp/pdfs/kurux_manual/page-55.png",
        "sha256": "a21143378b48042d8fce359523ca632e3f096ee914d0f0d5dc4e0408a5f0c2d7",
        "printed_page": 54,
    },
    56: {
        "path": WORKSPACE / "tmp/pdfs/kurux_manual/page-56.png",
        "sha256": "c6088d6a0caa5d26d56f8d6df2b4a8eef1875e2c7f98fd408c98fe1473a71f38",
        "printed_page": 55,
    },
    57: {
        "path": WORKSPACE / "tmp/pdfs/kurux_manual/page-57.png",
        "sha256": "84c0739aeaef327886753b5c2dd4f95c48f48e14b885cbe39756f76e84badd1d",
        "printed_page": 56,
    },
}
CHUNKS = [
    {
        "id": "items001-010",
        "lines": PACKAGE / "manual_chunks/p039-items001-010-lines.tsv",
        "cells": PACKAGE / "manual_chunks/p039-items001-010-cells.tsv",
        "items": range(1, 11),
        "line_count": 38,
        "conceptual_count": 60,
        "expanded_count": 61,
        "duplicate_coordinates": {(3, "A"): 2},
    },
    {
        "id": "items011-020",
        "lines": PACKAGE / "manual_chunks/p039-040-items011-020-lines.tsv",
        "cells": PACKAGE / "manual_chunks/p039-040-items011-020-cells.tsv",
        "items": range(11, 21),
        "line_count": 34,
        "conceptual_count": 60,
        "expanded_count": 60,
        "duplicate_coordinates": {},
    },
    {
        "id": "items021-030",
        "lines": PACKAGE / "manual_chunks/p040-items021-030-lines.tsv",
        "cells": PACKAGE / "manual_chunks/p040-items021-030-cells.tsv",
        "items": range(21, 31),
        "line_count": 33,
        "conceptual_count": 60,
        "expanded_count": 63,
        "duplicate_coordinates": {(30, "B"): 2, (30, "D"): 2, (30, "E"): 2},
    },
    {
        "id": "items031-040",
        "lines": PACKAGE / "manual_chunks/p040-041-items031-040-lines.tsv",
        "cells": PACKAGE / "manual_chunks/p040-041-items031-040-cells.tsv",
        "items": range(31, 41),
        "line_count": 31,
        "conceptual_count": 60,
        "expanded_count": 60,
        "duplicate_coordinates": {},
    },
    {
        "id": "items041-050",
        "lines": PACKAGE / "manual_chunks/p041-042-items041-050-lines.tsv",
        "cells": PACKAGE / "manual_chunks/p041-042-items041-050-cells.tsv",
        "items": range(41, 51),
        "line_count": 37,
        "conceptual_count": 60,
        "expanded_count": 63,
        "duplicate_coordinates": {(50, "B"): 2, (50, "C"): 2, (50, "D"): 2},
    },
    {
        "id": "items051-060",
        "lines": PACKAGE / "manual_chunks/p042-items051-060-lines.tsv",
        "cells": PACKAGE / "manual_chunks/p042-items051-060-cells.tsv",
        "items": range(51, 61),
        "line_count": 33,
        "conceptual_count": 60,
        "expanded_count": 60,
        "duplicate_coordinates": {},
    },
    {
        "id": "items061-070",
        "lines": PACKAGE / "manual_chunks/p042-043-items061-070-lines.tsv",
        "cells": PACKAGE / "manual_chunks/p042-043-items061-070-cells.tsv",
        "items": range(61, 71),
        "line_count": 35,
        "conceptual_count": 60,
        "expanded_count": 61,
        "duplicate_coordinates": {(66, "B"): 2},
    },
    {
        "id": "items071-080",
        "lines": PACKAGE / "manual_chunks/p043-items071-080-lines.tsv",
        "cells": PACKAGE / "manual_chunks/p043-items071-080-cells.tsv",
        "items": range(71, 81),
        "line_count": 32,
        "conceptual_count": 60,
        "expanded_count": 62,
        "duplicate_coordinates": {(71, "A"): 2, (76, "D"): 2},
    },
    {
        "id": "items081-090",
        "lines": PACKAGE / "manual_chunks/p043-044-items081-090-lines.tsv",
        "cells": PACKAGE / "manual_chunks/p043-044-items081-090-cells.tsv",
        "items": range(81, 91),
        "line_count": 29,
        "conceptual_count": 60,
        "expanded_count": 60,
        "duplicate_coordinates": {},
    },
    {
        "id": "items091-100",
        "lines": PACKAGE / "manual_chunks/p044-045-items091-100-lines.tsv",
        "cells": PACKAGE / "manual_chunks/p044-045-items091-100-cells.tsv",
        "items": range(91, 101),
        "line_count": 34,
        "conceptual_count": 60,
        "expanded_count": 61,
        "duplicate_coordinates": {(96, "A"): 2},
    },
    {
        "id": "items101-110",
        "lines": PACKAGE / "manual_chunks/p045-items101-110-lines.tsv",
        "cells": PACKAGE / "manual_chunks/p045-items101-110-cells.tsv",
        "items": range(101, 111),
        "line_count": 30,
        "conceptual_count": 60,
        "expanded_count": 60,
        "duplicate_coordinates": {},
    },
    {
        "id": "items111-120",
        "lines": PACKAGE / "manual_chunks/p045-046-items111-120-lines.tsv",
        "cells": PACKAGE / "manual_chunks/p045-046-items111-120-cells.tsv",
        "items": range(111, 121),
        "line_count": 42,
        "conceptual_count": 60,
        "expanded_count": 64,
        "duplicate_coordinates": {
            (114, "A"): 2, (118, "D"): 2, (119, "B"): 2, (120, "0"): 2
        },
    },
    {
        "id": "items121-130",
        "lines": PACKAGE / "manual_chunks/p046-items121-130-lines.tsv",
        "cells": PACKAGE / "manual_chunks/p046-items121-130-cells.tsv",
        "items": range(121, 131),
        "line_count": 33,
        "conceptual_count": 60,
        "expanded_count": 60,
        "duplicate_coordinates": {},
    },
    {
        "id": "items131-140",
        "lines": PACKAGE / "manual_chunks/p047-items131-140-lines.tsv",
        "cells": PACKAGE / "manual_chunks/p047-items131-140-cells.tsv",
        "items": range(131, 141),
        "line_count": 36,
        "conceptual_count": 60,
        "expanded_count": 61,
        "duplicate_coordinates": {(131, "A"): 2},
    },
    {
        "id": "items141-150",
        "lines": PACKAGE / "manual_chunks/p047-048-items141-150-lines.tsv",
        "cells": PACKAGE / "manual_chunks/p047-048-items141-150-cells.tsv",
        "items": range(141, 151),
        "line_count": 33,
        "conceptual_count": 60,
        "expanded_count": 63,
        "duplicate_coordinates": {(147, "E"): 2, (149, "C"): 2, (150, "E"): 2},
    },
    {
        "id": "items151-160",
        "lines": PACKAGE / "manual_chunks/p048-items151-160-lines.tsv",
        "cells": PACKAGE / "manual_chunks/p048-items151-160-cells.tsv",
        "items": range(151, 161),
        "line_count": 25,
        "conceptual_count": 60,
        "expanded_count": 60,
        "duplicate_coordinates": {},
    },
    {
        "id": "items161-170",
        "lines": PACKAGE / "manual_chunks/p048-049-items161-170-lines.tsv",
        "cells": PACKAGE / "manual_chunks/p048-049-items161-170-cells.tsv",
        "items": range(161, 171),
        "line_count": 33,
        "conceptual_count": 60,
        "expanded_count": 61,
        "duplicate_coordinates": {(165, "C"): 2},
    },
    {
        "id": "items171-180",
        "lines": PACKAGE / "manual_chunks/p049-items171-180-lines.tsv",
        "cells": PACKAGE / "manual_chunks/p049-items171-180-cells.tsv",
        "items": range(171, 181),
        "line_count": 28,
        "conceptual_count": 60,
        "expanded_count": 60,
        "duplicate_coordinates": {},
    },
    {
        "id": "items181-190",
        "lines": PACKAGE / "manual_chunks/p049-050-items181-190-lines.tsv",
        "cells": PACKAGE / "manual_chunks/p049-050-items181-190-cells.tsv",
        "items": range(181, 191),
        "line_count": 28,
        "conceptual_count": 60,
        "expanded_count": 60,
        "duplicate_coordinates": {},
    },
    {
        "id": "items191-200",
        "lines": PACKAGE / "manual_chunks/p050-items191-200-lines.tsv",
        "cells": PACKAGE / "manual_chunks/p050-items191-200-cells.tsv",
        "items": range(191, 201),
        "line_count": 30,
        "conceptual_count": 60,
        "expanded_count": 60,
        "duplicate_coordinates": {},
    },
    {
        "id": "items201-210",
        "lines": PACKAGE / "manual_chunks/p051-items201-210-lines.tsv",
        "cells": PACKAGE / "manual_chunks/p051-items201-210-cells.tsv",
        "items": range(201, 211),
        "line_count": 31,
        "conceptual_count": 60,
        "expanded_count": 62,
        "duplicate_coordinates": {(202, "D"): 2, (202, "E"): 2},
    },
    {
        "id": "items211-220",
        "lines": PACKAGE / "manual_chunks/p051-052-items211-220-lines.tsv",
        "cells": PACKAGE / "manual_chunks/p051-052-items211-220-cells.tsv",
        "items": range(211, 221),
        "line_count": 33,
        "conceptual_count": 60,
        "expanded_count": 60,
        "duplicate_coordinates": {},
    },
    {
        "id": "items221-230",
        "lines": PACKAGE / "manual_chunks/p052-items221-230-lines.tsv",
        "cells": PACKAGE / "manual_chunks/p052-items221-230-cells.tsv",
        "items": range(221, 231),
        "line_count": 26,
        "conceptual_count": 60,
        "expanded_count": 60,
        "duplicate_coordinates": {},
    },
    {
        "id": "items231-240",
        "lines": PACKAGE / "manual_chunks/p052-053-items231-240-lines.tsv",
        "cells": PACKAGE / "manual_chunks/p052-053-items231-240-cells.tsv",
        "items": range(231, 241),
        "line_count": 19,
        "conceptual_count": 60,
        "expanded_count": 60,
        "duplicate_coordinates": {},
    },
    {
        "id": "items241-250",
        "lines": PACKAGE / "manual_chunks/p053-items241-250-lines.tsv",
        "cells": PACKAGE / "manual_chunks/p053-items241-250-cells.tsv",
        "items": range(241, 251),
        "line_count": 32,
        "conceptual_count": 60,
        "expanded_count": 62,
        "duplicate_coordinates": {(245, "A"): 2, (245, "D"): 2},
    },
    {
        "id": "items251-260",
        "lines": PACKAGE / "manual_chunks/p053-054-items251-260-lines.tsv",
        "cells": PACKAGE / "manual_chunks/p053-054-items251-260-cells.tsv",
        "items": range(251, 261),
        "line_count": 40,
        "conceptual_count": 60,
        "expanded_count": 60,
        "duplicate_coordinates": {},
    },
    {
        "id": "items261-270",
        "lines": PACKAGE / "manual_chunks/p054-055-items261-270-lines.tsv",
        "cells": PACKAGE / "manual_chunks/p054-055-items261-270-cells.tsv",
        "items": range(261, 271),
        "line_count": 40,
        "conceptual_count": 60,
        "expanded_count": 60,
        "duplicate_coordinates": {},
    },
    {
        "id": "items271-280",
        "lines": PACKAGE / "manual_chunks/p055-items271-280-lines.tsv",
        "cells": PACKAGE / "manual_chunks/p055-items271-280-cells.tsv",
        "items": range(271, 281),
        "line_count": 36,
        "conceptual_count": 60,
        "expanded_count": 61,
        "duplicate_coordinates": {(274, "D"): 2},
    },
    {
        "id": "items281-290",
        "lines": PACKAGE / "manual_chunks/p055-056-items281-290-lines.tsv",
        "cells": PACKAGE / "manual_chunks/p055-056-items281-290-cells.tsv",
        "items": range(281, 291),
        "line_count": 30,
        "conceptual_count": 60,
        "expanded_count": 62,
        "duplicate_coordinates": {(283, "D"): 2, (284, "D"): 2},
    },
    {
        "id": "items291-300",
        "lines": PACKAGE / "manual_chunks/p056-items291-300-lines.tsv",
        "cells": PACKAGE / "manual_chunks/p056-items291-300-cells.tsv",
        "items": range(291, 301),
        "line_count": 30,
        "conceptual_count": 60,
        "expanded_count": 60,
        "duplicate_coordinates": {},
    },
    {
        "id": "items301-307",
        "lines": PACKAGE / "manual_chunks/p056-057-items301-307-lines.tsv",
        "cells": PACKAGE / "manual_chunks/p056-057-items301-307-cells.tsv",
        "items": range(301, 308),
        "line_count": 17,
        "conceptual_count": 42,
        "expanded_count": 42,
        "duplicate_coordinates": {},
    },
]
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


def physical_page(line_id: str) -> int:
    assert line_id.startswith("P") and "-L" in line_id
    return int(line_id[1:4])


def read_lines(chunk: dict) -> list[dict[str, str]]:
    with chunk["lines"].open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(rows) == chunk["line_count"]
    assert len({row["line_id"] for row in rows}) == len(rows)
    assert {int(row["item"]) for row in rows} == set(chunk["items"])
    assert {row["status"] for row in rows} <= ALLOWED_STATUSES
    for row in rows:
        assert physical_page(row["line_id"]) in IMAGES
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
        if row.get("confidence"):
            assert row["confidence"] in {"high", "medium", "low"}
    return rows


def expand(chunk: dict, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    cells: list[dict[str, str]] = []
    variants: Counter[tuple[int, str]] = Counter()
    for row in rows:
        page = physical_page(row["line_id"])
        for code in row["site_codes"]:
            coordinate = (int(row["item"]), code)
            variants[coordinate] += 1
            site, role = SITES[code]
            cells.append({
                "physical_page": str(page),
                "printed_page": str(IMAGES[page]["printed_page"]),
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
                "evidence_sha256": IMAGES[page]["sha256"],
            })
    cells.sort(key=lambda row: (
        int(row["item"]), SITE_ORDER[row["site_code"]], int(row["site_variant"])
    ))
    by_item: defaultdict[int, set[str]] = defaultdict(set)
    coordinate_counts: Counter[tuple[int, str]] = Counter()
    for row in cells:
        item = int(row["item"])
        by_item[item].add(row["site_code"])
        coordinate_counts[(item, row["site_code"])] += 1
    assert set(by_item) == set(chunk["items"])
    assert all(codes == set(SITES) for codes in by_item.values())
    assert len(coordinate_counts) == chunk["conceptual_count"]
    assert len(cells) == chunk["expanded_count"]
    assert {key: count for key, count in coordinate_counts.items() if count > 1} == (
        chunk["duplicate_coordinates"]
    )
    return cells


def write_cells(chunk: dict, cells: list[dict[str, str]]) -> None:
    with chunk["cells"].open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(cells)


def write_manifest(results: list[tuple[dict, list[dict[str, str]], list[dict[str, str]]]]) -> None:
    all_lines = [row for _, lines, _ in results for row in lines]
    all_cells = [row for _, _, cells in results for row in cells]
    conceptual = {(int(row["item"]), row["site_code"]) for row in all_cells}
    status = Counter(row["status"] for row in all_cells)
    role_status = Counter((row["role"], row["status"]) for row in all_cells)
    conceptual_status = {}
    for coordinate in conceptual:
        states = {row["status"] for row in all_cells
                  if (int(row["item"]), row["site_code"]) == coordinate}
        assert len(states) == 1
        conceptual_status[coordinate] = states.pop()

    assert len(all_lines) == 988
    assert len(conceptual) == 1842
    assert len(all_cells) == 1869
    assert status == {"attested": 1661, "blank": 136, "not_used": 72}
    assert role_status == {
        ("target", "attested"): 1365,
        ("target", "blank"): 136,
        ("target", "not_used"): 60,
        ("control", "attested"): 296,
        ("control", "not_used"): 12,
    }
    assert Counter(conceptual_status.values()) == {
        "attested": 1634, "blank": 136, "not_used": 72
    }
    assert sha256(SOURCE_PDF) == SOURCE_PDF_SHA256
    for image in IMAGES.values():
        assert sha256(image["path"]) == image["sha256"]

    chunk_manifests = []
    for chunk, lines, cells in results:
        chunk_manifests.append({
            "id": chunk["id"],
            "items": [min(chunk["items"]), max(chunk["items"])],
            "response_lines": len(lines),
            "conceptual_cells": chunk["conceptual_count"],
            "expanded_cell_rows": len(cells),
            "status_counts_expanded": dict(sorted(Counter(row["status"] for row in cells).items())),
            "confidence_counts": dict(sorted(Counter(
                row.get("confidence", "") for row in lines if row.get("confidence", "")
            ).items())),
            "manual_lines": str(chunk["lines"].relative_to(PACKAGE)),
            "manual_lines_sha256": sha256(chunk["lines"]),
            "expanded_cells": str(chunk["cells"].relative_to(PACKAGE)),
            "expanded_cells_sha256": sha256(chunk["cells"]),
        })

    manifest = {
        "report": "ESR 2011-040 The Kurux of Bangladesh",
        "state": "manual_review_complete",
        "policy": (
            "Rendered pages supplied every reading; OCR, PDF text, raw legacy glyphs, "
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
        "reviewed_images": [
            {
                "physical_page": page,
                "printed_page": image["printed_page"],
                "path": str(image["path"].relative_to(WORKSPACE)),
                "sha256": image["sha256"],
                "dimensions": [2550, 3300],
            }
            for page, image in sorted(IMAGES.items())
        ],
        "physical_pages_reviewed": [39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57],
        "printed_pages_reviewed": [38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56],
        "items_reviewed": [1, 307],
        "response_lines": len(all_lines),
        "conceptual_cells": len(conceptual),
        "expanded_cell_rows": len(all_cells),
        "site_variant_extra_rows": len(all_cells) - len(conceptual),
        "status_counts_expanded": dict(sorted(status.items())),
        "status_counts_conceptual": dict(sorted(Counter(conceptual_status.values()).items())),
        "target_counts_expanded": {
            key: role_status[("target", key)] for key in ("attested", "blank", "not_used")
        },
        "control_counts_expanded": {
            key: role_status[("control", key)] for key in ("attested", "blank", "not_used")
        },
        "blank_coordinates": [
            "item-001/site-B", "item-009/site-A", "item-010/site-A",
            "item-011/site-A", "item-011/site-D", "item-011/site-E",
            "item-012/site-A", "item-015/site-A",
            "item-021/site-A", "item-023/site-A", "item-028/site-A",
            "item-030/site-A",
            "item-033/site-A", "item-034/site-A", "item-037/site-A",
            "item-042/site-A", "item-044/site-A", "item-048/site-A",
            "item-050/site-A",
            "item-051/site-A", "item-055/site-A", "item-056/site-A",
            "item-056/site-E", "item-057/site-A", "item-058/site-A",
            "item-059/site-A", "item-059/site-B",
            "item-067/site-A", "item-068/site-A", "item-069/site-A",
            "item-070/site-A",
            "item-072/site-A", "item-075/site-A", "item-076/site-A",
            "item-077/site-A", "item-079/site-A",
            "item-085/site-A", "item-086/site-A", "item-090/site-A",
            "item-091/site-A", "item-092/site-A", "item-093/site-A",
            "item-094/site-A", "item-095/site-A",
            "item-101/site-A", "item-106/site-A",
            "item-112/site-A", "item-116/site-A", "item-117/site-A",
            "item-122/site-A", "item-122/site-B", "item-125/site-A",
            "item-127/site-A",
            "item-136/site-A", "item-137/site-A", "item-138/site-A",
            "item-139/site-A", "item-140/site-A",
            "item-145/site-A", "item-147/site-A", "item-148/site-A",
            "item-149/site-A",
            "item-153/site-A", "item-154/site-A", "item-158/site-A",
            "item-162/site-A", "item-164/site-A", "item-168/site-A",
            "item-169/site-A",
            "item-174/site-A", "item-179/site-A", "item-180/site-A",
            "item-181/site-A", "item-184/site-A", "item-185/site-A",
            "item-186/site-A", "item-187/site-A", "item-187/site-B",
            "item-192/site-A", "item-195/site-A", "item-196/site-A",
            "item-198/site-A", "item-199/site-A", "item-200/site-A",
            "item-201/site-A", "item-202/site-A",
            "item-203/site-A", "item-203/site-B",
            "item-204/site-A", "item-204/site-B",
            "item-205/site-A", "item-206/site-A", "item-207/site-A",
            "item-208/site-A", "item-209/site-A", "item-209/site-E",
            "item-210/site-A",
            "item-211/site-A", "item-212/site-A", "item-213/site-A",
            "item-215/site-A", "item-216/site-A", "item-217/site-A",
            "item-218/site-A", "item-218/site-B", "item-219/site-A",
            "item-221/site-A", "item-223/site-A", "item-223/site-E",
            "item-224/site-A", "item-224/site-B", "item-224/site-E",
            "item-239/site-A",
            "item-241/site-A", "item-250/site-A",
            "item-251/site-A", "item-252/site-A", "item-253/site-A",
            "item-254/site-A", "item-255/site-A", "item-255/site-C",
            "item-256/site-A", "item-257/site-A", "item-260/site-A",
            "item-261/site-A", "item-262/site-A", "item-263/site-A",
            "item-264/site-A", "item-265/site-A", "item-266/site-A",
            "item-267/site-A",
            "item-275/site-A", "item-275/site-D",
            "item-288/site-A", "item-289/site-A",
            "item-294/site-A",
        ],
        "ambiguous_coordinates": [],
        "illegible_coordinates": [],
        "not_used_coordinates": [
            "item-031/site-A", "item-031/site-B", "item-031/site-C",
            "item-031/site-D", "item-031/site-E", "item-031/site-0",
            "item-074/site-A", "item-074/site-B", "item-074/site-C",
            "item-074/site-D", "item-074/site-E", "item-074/site-0",
            "item-107/site-A", "item-107/site-B", "item-107/site-C",
            "item-107/site-D", "item-107/site-E", "item-107/site-0",
            "item-124/site-A", "item-124/site-B", "item-124/site-C",
            "item-124/site-D", "item-124/site-E", "item-124/site-0",
            "item-152/site-A", "item-152/site-B", "item-152/site-C",
            "item-152/site-D", "item-152/site-E", "item-152/site-0",
            "item-163/site-A", "item-163/site-B", "item-163/site-C",
            "item-163/site-D", "item-163/site-E", "item-163/site-0",
            "item-171/site-A", "item-171/site-B", "item-171/site-C",
            "item-171/site-D", "item-171/site-E", "item-171/site-0",
            "item-194/site-A", "item-194/site-B", "item-194/site-C",
            "item-194/site-D", "item-194/site-E", "item-194/site-0",
            "item-240/site-A", "item-240/site-B", "item-240/site-C",
            "item-240/site-D", "item-240/site-E", "item-240/site-0",
            "item-247/site-A", "item-247/site-B", "item-247/site-C",
            "item-247/site-D", "item-247/site-E", "item-247/site-0",
            "item-301/site-A", "item-301/site-B", "item-301/site-C",
            "item-301/site-D", "item-301/site-E", "item-301/site-0",
            "item-306/site-A", "item-306/site-B", "item-306/site-C",
            "item-306/site-D", "item-306/site-E", "item-306/site-0"
        ],
        "site_variant_coordinates": [
            "item-003/site-A", "item-030/site-B", "item-030/site-D", "item-030/site-E",
            "item-050/site-B", "item-050/site-C", "item-050/site-D",
            "item-066/site-B",
            "item-071/site-A",
            "item-076/site-D",
            "item-096/site-A",
            "item-114/site-A", "item-118/site-D", "item-119/site-B", "item-120/site-0",
            "item-131/site-A",
            "item-147/site-E", "item-149/site-C", "item-150/site-E",
            "item-165/site-C",
            "item-202/site-D", "item-202/site-E",
            "item-245/site-A", "item-245/site-D",
            "item-274/site-D",
            "item-283/site-D", "item-284/site-D",
        ],
        "site_identity_state": "pending_identity_and_schema_review",
        "control_policy": "Printed code 0 (Bangla) is retained audit-only pending identity/schema review.",
        "pending_items": [],
        "chunks": chunk_manifests,
        "reconciliation_state": "not_started_after_manual_freeze",
    }
    with MANIFEST.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> None:
    results = []
    for chunk in CHUNKS:
        lines = read_lines(chunk)
        cells = expand(chunk, lines)
        write_cells(chunk, cells)
        results.append((chunk, lines, cells))
    write_manifest(results)
    print("wrote 1869 rows for 1842 conceptual cells from 988 frozen manual lines")


if __name__ == "__main__":
    main()
