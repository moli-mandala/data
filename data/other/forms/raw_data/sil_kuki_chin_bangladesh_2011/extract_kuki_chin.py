#!/usr/bin/env python3
"""Extract Appendix A.3 of SIL ESR 2011-025 without OCR.

The public appendix PDF has a complete text layer, but its TrueType fonts were
subsetted into private-use code points. The checked-in decoder below follows
the PDF's embedded ``ToUnicode`` and font programs. Contextual SAG-IPA glyphs
which differ only in width or height map to the same Unicode combining mark.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
DEFAULT_PDF = Path("/tmp/kuki-chin-appendix-a.pdf")
DEFAULT_OUTPUT = HERE / "wordlists.tsv"

PDF_SHA256 = "d0506535e6040bafebe88a3f3db5217f68ffd42179651ef9316bcbcb2272b230"
PDF_PAGES = range(53, 92)
EXPECTED_PRINTED_RESPONSES = 2565
EXPECTED_EXPANDED_ATTESTATIONS = 3875
EXPECTED_NO_ENTRY = 53
EXPECTED_SITE_COUNTS = Counter({
    "e": 333, "c": 329, "h": 327, "l": 324, "i": 324, "m": 324,
    "a": 323, "b": 323, "j": 321, "g": 321, "k": 319, "0": 307,
})
FIELDS = [
    "Item", "Gloss", "Similarity_Group", "Raw_Form", "Form", "Site_Codes",
    "PDF_Page", "Printed_Page", "Response", "Review",
]


def pua_map(values: dict[int, str]) -> dict[str, str]:
    return {chr(0xF000 + key): value for key, value in values.items()}


SAG = pua_map({
    0x001: "ʃ", 0x002: "d", 0x003: "ʒ", 0x004: " ", 0x005: "a",
    0x006: "k", 0x007: "r", 0x008: "ɨ", 0x009: "v", 0x00A: "n",
    0x00B: "h", 0x00C: "ɔ", 0x00D: "ː", 0x00E: "i", 0x00F: "ə",
    0x010: "\u0301", 0x011: "u", 0x012: "o", 0x013: "t",
    0x014: "\u0303", 0x015: "l", 0x016: "\u032f", 0x017: "ɾ",
    0x018: "f", 0x019: "ɛ", 0x01A: "ɑ", 0x01B: "s", 0x01C: "m",
    0x01D: "g", 0x01E: "ŋ", 0x01F: "p", 0x020: "b", 0x021: "ɪ",
    0x022: "ʈ", 0x023: "ʔ", 0x024: "j", 0x025: "œ", 0x026: "e",
    0x027: "y", 0x028: "\u031a", 0x029: "\u0308", 0x02A: "/",
    0x02B: "ɽ", 0x02C: "\u0304", 0x02D: "ʌ", 0x02E: "\u0301",
    0x02F: "ʉ", 0x030: "w", 0x031: "ɖ", 0x032: "\u0300",
    0x033: ".", 0x034: "ɸ", 0x035: "ɴ", 0x036: "(", 0x037: ")",
    0x038: "\u0329", 0x039: "æ", 0x03A: "\u0300", 0x03B: "ɲ",
    0x03C: "ɬ", 0x03D: "ɶ", 0x03E: "\u0306", 0x03F: "\u0302",
    0x040: "ɯ", 0x041: "ɣ",
})

EXPECTED_SAG_COUNTS = Counter({
    chr(0xF000 + code): count for code, count in {
        0x001: 488, 0x002: 282, 0x003: 125, 0x004: 312, 0x005: 1470,
        0x006: 786, 0x007: 336, 0x008: 173, 0x009: 74, 0x00A: 873,
        0x00B: 915, 0x00C: 392, 0x00D: 321, 0x00E: 1278, 0x00F: 373,
        0x010: 162, 0x011: 928, 0x012: 558, 0x013: 869, 0x014: 141,
        0x015: 506, 0x016: 905, 0x017: 59, 0x018: 24, 0x019: 396,
        0x01A: 223, 0x01B: 57, 0x01C: 498, 0x01D: 39, 0x01E: 436,
        0x01F: 370, 0x020: 182, 0x021: 134, 0x022: 66, 0x023: 255,
        0x024: 64, 0x025: 3, 0x026: 208, 0x027: 53, 0x028: 63,
        0x029: 4, 0x02A: 13, 0x02B: 29, 0x02C: 11, 0x02D: 14,
        0x02E: 9, 0x02F: 11, 0x030: 38, 0x031: 10, 0x032: 72,
        0x033: 29, 0x034: 3, 0x035: 1, 0x036: 184, 0x037: 184,
        0x038: 1, 0x039: 8, 0x03A: 1, 0x03B: 2, 0x03C: 1,
        0x03D: 1, 0x03E: 1, 0x03F: 2, 0x040: 1, 0x041: 2,
    }.items()
})

CHARIS = pua_map({
    0x003: " ", 0x006: "L", 0x007: "B", 0x008: "a", 0x009: "n",
    0x00A: "g", 0x00B: "l", 0x00C: "d", 0x00D: "e", 0x00E: "s",
    0x00F: "h", 0x010: "(", 0x011: "o", 0x012: "c", 0x013: "i",
    0x014: "u", 0x015: "t", 0x017: "r", 0x018: ")", 0x01A: "m",
    0x01B: "y", 0x01C: "f", 0x01D: "2", 0x01E: "0", 0x01F: "4",
    0x020: "-", 0x022: "b", 0x023: ",", 0x025: "k", 0x028: "5",
    0x029: "w", 0x02C: "q", 0x02D: "6", 0x030: "p", 0x031: "8",
    0x032: "1", 0x033: "9", 0x036: "x", 0x037: "v", 0x038: "j",
    0x03A: "M", 0x03B: "E", 0x03D: ":", 0x03F: "/", 0x043: "7",
    0x044: "3", 0x045: "“", 0x046: "”", 0x047: "z", 0x048: "=",
})

ARIAL = pua_map({
    0x001: " ", 0x002: "1", 0x003: "[", 0x004: "0", 0x005: "]",
    0x006: "2", 0x007: "a", 0x008: "l", 0x009: "3", 0x00A: "b",
    0x00B: "i", 0x00C: "j", 0x00D: "m", 0x00E: "4", 0x00F: "c",
    0x010: "k", 0x011: "5", 0x012: "g", 0x013: "h", 0x014: "e",
    0x015: "6", 0x016: "7", 0x017: "8", 0x018: "9",
    0x019: "A", 0x01A: "B",
})


def decode(text: str, mapping: dict[str, str]) -> str:
    unknown = sorted({char for char in text if 0xF000 <= ord(char) <= 0xF0FF and char not in mapping})
    if unknown:
        raise AssertionError(f"unmapped embedded glyphs: {[f'U+{ord(c):04X}' for c in unknown]}")
    return "".join(mapping.get(char, char) for char in text).replace("\n", "")


def spans(page) -> list[dict]:
    found = []

    def visit(text, _cm, tm, font, _size):
        if not text or text == "\n" or not font:
            return
        found.append({
            "x": float(tm[4]), "y": float(tm[5]),
            "font": str(font.get("/BaseFont", "")), "text": text.replace("\n", ""),
        })

    page.extract_text(visitor_text=visit)
    return found


def grouped(spans_: list[dict], needle: str, mapping: dict[str, str]) -> list[dict]:
    groups = defaultdict(list)
    for span in spans_:
        if needle in span["font"]:
            column = 0 if span["x"] < 300 else 1
            groups[column, round(span["y"], 1)].append(span)
    result = []
    for (column, y), members in groups.items():
        members.sort(key=lambda row: row["x"])
        result.append({
            "column": column, "y": y,
            "text": decode("".join(row["text"] for row in members), mapping),
            "members": members,
        })
    return result


def source_pages(pdf_path: Path) -> list[tuple[int, list[dict]]]:
    if hashlib.sha256(pdf_path.read_bytes()).hexdigest() != PDF_SHA256:
        raise AssertionError("public Appendix A PDF fingerprint drift")
    reader = PdfReader(pdf_path)
    if len(reader.pages) != 127:
        raise AssertionError(f"appendix PDF page-count drift: {len(reader.pages)}")
    return [(number, spans(reader.pages[number - 1])) for number in PDF_PAGES]


def parse(pdf_path: Path) -> list[dict[str, str | int]]:
    pages = source_pages(pdf_path)
    events = []
    glyph_counts = Counter()
    for pdf_page, page_spans in pages:
        headings = []
        for line in grouped(page_spans, "Charis-SIL", CHARIS):
            match = re.fullmatch(r"\s*(\d{1,3})\s+(.+?)\s*", line["text"])
            if match and 1 <= int(match.group(1)) <= 306:
                headings.append({
                    "kind": "heading", "page": pdf_page, "column": line["column"],
                    "y": line["y"], "item": int(match.group(1)), "gloss": match.group(2),
                })

        sag_spans = [span for span in page_spans if "SAG-IPA-SILManuscript" in span["font"]]
        responses = []
        for line in grouped(page_spans, "Arial", ARIAL):
            match = re.search(r"\[([0a-m]+)\]", line["text"])
            if not match:
                continue
            before = line["text"][:match.start()]
            group = re.search(r"[0-9A-B]", before)
            if not group:
                raise AssertionError(f"response without similarity group on PDF p. {pdf_page}: {line['text']!r}")
            nearby = [
                span for span in sag_spans
                if (0 if span["x"] < 300 else 1) == line["column"]
                and abs(span["y"] - line["y"]) <= 6.0
            ]
            nearby.sort(key=lambda row: (row["x"], -row["y"]))
            raw_form = "".join(span["text"] for span in nearby).strip(chr(0xF004))
            glyph_counts.update(char for char in raw_form if char in SAG)
            form = unicodedata.normalize("NFC", decode(raw_form, SAG).strip()) if raw_form else ""
            if form == "no entry":
                form = ""
            responses.append({
                "kind": "response", "page": pdf_page, "column": line["column"],
                "y": line["y"], "group": group.group(), "codes": match.group(1),
                "raw_form": raw_form, "form": form,
            })

        events.extend(headings)
        events.extend(responses)

    events.sort(key=lambda row: (row["page"], row["column"], -row["y"], 0 if row["kind"] == "heading" else 1))
    rows = []
    headings_seen = []
    current_item = None
    current_gloss = None
    response_number = 0
    for event in events:
        if event["kind"] == "heading":
            current_item = event["item"]
            current_gloss = event["gloss"]
            headings_seen.append(current_item)
            response_number = 0
            continue
        if current_item is None:
            raise AssertionError(f"orphan response on PDF p. {event['page']}")
        response_number += 1
        review = (
            "public Appendix A PDF text layer; embedded SAG-IPA subset decoded from "
            "font outlines and SIL SAGIPA2Uni.map v1.0"
        )
        if not event["form"]:
            review += "; printed no entry"
        rows.append({
            "Item": current_item, "Gloss": current_gloss,
            "Similarity_Group": event["group"], "Raw_Form": event["raw_form"],
            "Form": event["form"], "Site_Codes": event["codes"],
            "PDF_Page": event["page"], "Printed_Page": event["page"] - 3,
            "Response": response_number, "Review": review,
        })

    if headings_seen != list(range(1, 307)):
        raise AssertionError(f"item topology drift: {headings_seen}")
    bad_zero = [row for row in rows if row["Similarity_Group"] == "0" and row["Form"]]
    if bad_zero:
        raise AssertionError(f"similarity group 0 unexpectedly has a transcription: {bad_zero[:5]}")
    if any(row["Similarity_Group"] != "0" and not row["Form"] for row in rows):
        raise AssertionError("nonzero similarity group lacks a transcription")
    if glyph_counts != EXPECTED_SAG_COUNTS:
        raise AssertionError(f"embedded SAG glyph census drift: {glyph_counts}")
    site_counts = Counter(code for row in rows for code in str(row["Site_Codes"]))
    if len(rows) != EXPECTED_PRINTED_RESPONSES or site_counts != EXPECTED_SITE_COUNTS:
        raise AssertionError(
            f"response topology drift: printed={len(rows)} site_counts={site_counts}"
        )
    if sum(not row["Form"] for row in rows) != EXPECTED_NO_ENTRY:
        raise AssertionError("printed no-entry count drift")
    if sum(site_counts.values()) != EXPECTED_EXPANDED_ATTESTATIONS:
        raise AssertionError("expanded-attestation count drift")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", nargs="?", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = parse(args.pdf)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"items=306 printed_responses={len(rows)} expanded="
        f"{sum(len(str(row['Site_Codes'])) for row in rows)} "
        f"no_entry={sum(not row['Form'] for row in rows)}"
    )


if __name__ == "__main__":
    main()
