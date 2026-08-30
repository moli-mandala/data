#!/usr/bin/env python3
"""Align item-level OCR with the ten fixed Appendix A.3 list rows."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

SITES = ("HRT", "HJN", "HFT", "HNG", "BPL", "HTR", "HLH", "PBG", "HIN", "PUN")
TARGET_SITES = SITES[:4] + SITES[5:7]  # six lists treated as Haryanvi in section 4.1
MARKER = re.compile(r"^@@ item(\d+)-pdf(\d+)-p(\d+)-c(\d+)$")
TOKEN = re.compile(r"^[^A-Za-z0-9]*(?:[-'‘’]\s*)?([A-Za-z0-9|!]{2,5})(.*)$")
HEADER = re.compile(r"^[^A-Za-z0-9]*\d{1,3}\s*[.]", re.ASCII)

FIELDS = [
    "Item", "PDF_Page", "Printed_Page", "Column", "Header_OCR", "Site",
    "Role", "Site_OCR", "Raw_OCR", "Raw_OCR_Primary", "Raw_OCR_Secondary",
    "Raw_OCR_Latin", "OCR_Pass", "Blank", "Alignment_Note",
    "Transcription", "Review", "Uncertainty",
]


def distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b))
            )
        previous = current
    return previous[-1]


def clean_token(value: str) -> str:
    return re.sub(r"[^A-Z]", "", value.upper().translate(str.maketrans("801|!", "BOIII")))


def site_candidate(line: str, expected_index: int) -> tuple[int, str, str, str] | None:
    match = TOKEN.match(line)
    if not match:
        return None
    raw_token, rest = match.groups()
    # Site labels are set in capitals. Requiring at least two literal capitals
    # prevents ordinary responses such as `hare` from becoming a spurious HRT
    # label merely because their upper-cased edit distance happens to be small.
    if sum(character.isupper() for character in raw_token) < 2:
        return None
    token = clean_token(raw_token)
    if not 2 <= len(token) <= 4:
        return None

    choices = []
    for index in range(expected_index, len(SITES)):
        edit = distance(token, SITES[index])
        # Strongly prefer local monotonic alignment. This maps an early OCR
        # `HIN` to HJN rather than jumping over seven printed rows, while an
        # exact HFT can still prove that a blank HJN label was dropped.
        score = edit + (index - expected_index) * 0.55
        choices.append((score, edit, index))
    score, edit, index = min(choices)
    if edit <= 1 or (index == expected_index and edit <= 2):
        return index, raw_token, rest.strip(), f"token={token};edit={edit};score={score:.2f}"
    return None


def blocks(path: Path) -> list[tuple[dict[str, int], list[str]]]:
    result = []
    metadata = None
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        marker = MARKER.match(raw.strip())
        if marker:
            if metadata is not None:
                result.append((metadata, lines))
            item, pdf_page, printed_page, column = map(int, marker.groups())
            metadata = {
                "item": item,
                "pdf_page": pdf_page,
                "printed_page": printed_page,
                "column": column,
            }
            lines = []
        elif metadata is not None:
            lines.append(raw.rstrip())
    if metadata is not None:
        result.append((metadata, lines))
    return result


def align_item(metadata: dict[str, int], lines: list[str]) -> list[dict[str, str | int]]:
    header_lines: list[str] = []
    aligned: dict[int, dict[str, str]] = {}
    expected = 0
    last_index: int | None = None

    for raw in lines:
        text = raw.strip()
        if not text:
            continue
        candidate = site_candidate(text, expected) if expected < len(SITES) else None
        if candidate:
            index, raw_token, rest, note = candidate
            for skipped in range(expected, index):
                aligned[skipped] = {
                    "site_ocr": "", "raw": "", "note": "site label absent from OCR",
                }
            aligned[index] = {"site_ocr": raw_token, "raw": rest, "note": note}
            expected = index + 1
            last_index = index
            continue
        if expected == 0 or (not aligned and HEADER.match(text)):
            header_lines.append(text)
        elif last_index is not None and not HEADER.match(text):
            # Long response annotations such as `(thatch)` wrap without a
            # repeated site code. Preserve them on the preceding source cell.
            aligned[last_index]["raw"] = (aligned[last_index]["raw"] + " " + text).strip()
            aligned[last_index]["note"] += ";continued OCR line"

    for index in range(expected, len(SITES)):
        aligned[index] = {
            "site_ocr": "", "raw": "", "note": "site label absent from OCR",
        }

    header = " | ".join(header_lines)
    rows = []
    for index, site in enumerate(SITES):
        cell = aligned[index]
        rows.append(
            {
                "Item": metadata["item"],
                "PDF_Page": metadata["pdf_page"],
                "Printed_Page": metadata["printed_page"],
                "Column": metadata["column"],
                "Header_OCR": header,
                "Site": site,
                "Role": "target" if site in TARGET_SITES else "comparison",
                "Site_OCR": cell["site_ocr"],
                "Raw_OCR": cell["raw"],
                "Blank": "1" if not cell["raw"] else "0",
                "Alignment_Note": cell["note"],
                "Transcription": "",
                "Review": "pending" if site in TARGET_SITES else "audit-only comparison",
                "Uncertainty": "",
            }
        )
    return rows


def parse(path: Path) -> list[dict[str, str | int]]:
    source_blocks = blocks(path)
    if [metadata["item"] for metadata, _ in source_blocks] != list(range(1, 211)):
        raise AssertionError("OCR blocks must account for items 1-210 in order")
    rows = []
    for metadata, lines in source_blocks:
        rows.extend(align_item(metadata, lines))
    if len(rows) != 2100:
        raise AssertionError(f"expected 2,100 aligned cells, found {len(rows)}")
    return rows


def merge_passes(
    primary: list[dict[str, str | int]],
    secondary: list[dict[str, str | int]],
    primary_name: str,
    secondary_name: str,
) -> list[dict[str, str | int]]:
    """Retain both OCR readings and use only conservative secondary fallbacks.

    PSM 6 keeps site labels and responses together on most cells. PSM 4 can
    recover a dropped response, but sometimes reads a response word as a site
    code. A secondary value is therefore selected only when its printed site
    label is an exact normalized match for the expected site.
    """
    if len(primary) != len(secondary):
        raise AssertionError("OCR passes produced different cell counts")
    merged = []
    for first, second in zip(primary, secondary):
        key = (first["Item"], first["Site"])
        if key != (second["Item"], second["Site"]):
            raise AssertionError(f"OCR pass alignment differs at {key}")
        first_raw = str(first["Raw_OCR"])
        second_raw = str(second["Raw_OCR"])
        chosen = first
        chosen_name = primary_name
        if (
            not first_raw
            and second_raw
            and clean_token(str(second["Site_OCR"])) == str(second["Site"])
        ):
            chosen = second
            chosen_name = secondary_name
        row = dict(chosen)
        row["Raw_OCR_Primary"] = first_raw
        row["Raw_OCR_Secondary"] = second_raw
        row["OCR_Pass"] = chosen_name
        if chosen is second:
            row["Alignment_Note"] = (
                str(row["Alignment_Note"]) + ";exact-label secondary fallback"
            )
        merged.append(row)
    return merged


def add_latin_pass(
    rows: list[dict[str, str | int]],
    latin: list[dict[str, str | int]],
) -> None:
    if len(rows) != len(latin):
        raise AssertionError("Latin OCR pass produced a different cell count")
    for row, alternate in zip(rows, latin):
        key = (row["Item"], row["Site"])
        if key != (alternate["Item"], alternate["Site"]):
            raise AssertionError(f"Latin OCR pass alignment differs at {key}")
        alternate_raw = str(alternate["Raw_OCR"])
        row["Raw_OCR_Latin"] = alternate_raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("--secondary", type=Path)
    parser.add_argument("--latin", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = parse(args.raw)
    if args.secondary:
        rows = merge_passes(
            rows,
            parse(args.secondary),
            args.raw.stem,
            args.secondary.stem,
        )
    else:
        for row in rows:
            row["Raw_OCR_Primary"] = row["Raw_OCR"]
            row["Raw_OCR_Secondary"] = ""
            row["OCR_Pass"] = args.raw.stem
    if args.latin:
        add_latin_pass(rows, parse(args.latin))
    else:
        for row in rows:
            row["Raw_OCR_Latin"] = ""
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    blanks = sum(row["Blank"] == "1" for row in rows)
    target_pending = sum(row["Review"] == "pending" for row in rows)
    print(f"cells={len(rows)} blank_ocr={blanks} target_pending={target_pending}")


if __name__ == "__main__":
    main()
