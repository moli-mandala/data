"""Extract Eastern Magar wordlists from Hilty's 2013 SIL survey.

The separate, born-digital appendices PDF contains 218 concepts for four
survey sites on PDF pages 42--57 (printed pages 40--55).  Its text is real
Unicode IPA in a fixed two-column layout, so no OCR is used.

Download ``Eastern_Magar_Appendices.pdf`` to
``tmp/pdfs/eastern-magar`` before running.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

import fitz


HERE = Path(__file__).resolve().parent
PDF = HERE.parents[3] / "tmp" / "pdfs" / "eastern-magar" / "Eastern_Magar_Appendices.pdf"
OUTPUT = HERE.parent / "20260813-eastern-magar.csv"
SOURCE = "hilty2013eastern-magar"

SITES = ("Dhankuta", "Nawalparasi", "Panchthar", "Sarlahi")
LECTS = {
    "Dhankuta": "eastern_magar_dhankuta",
    "Nawalparasi": "eastern_magar_nawalparasi",
    "Panchthar": "eastern_magar_panchthar",
    "Sarlahi": "eastern_magar_sarlahi",
}


def _lines(page: fitz.Page) -> list[dict]:
    result = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            text = "".join(span["text"] for span in line["spans"]).strip()
            if text:
                result.append(
                    {
                        "x": line["bbox"][0],
                        "y": line["bbox"][1],
                        "x2": line["bbox"][2],
                        "text": text,
                    }
                )
    return result


def _clean_form(text: str) -> tuple[str, str]:
    text = re.sub(r"\s+", " ", text).strip()
    note = ""
    if text.endswith(" (low)"):
        text = text.removesuffix(" (low)")
        note = "Source marks this form as low."
    # In the PDF's mixed fonts, the combining unreleased-stop mark extracts
    # as a detached plus. Visual inspection confirms the printed glyph is ̚.
    text = re.sub(r"([ptkbdgʈɖ])\s*\+", r"\1̚", text)
    # A handful of overprinted breathy-voice marks extract twice at the same
    # position even though the rendered table shows a single diacritic.
    text = re.sub("̤+", "̤", text)
    return text, note


def extract(pdf_path: Path = PDF) -> list[tuple[int, str, str, str, str, int]]:
    """Return (concept, gloss, site, form, note, printed page) records."""
    document = fitz.open(pdf_path)
    records: list[tuple[int, str, str, str, str, int]] = []
    concept_number = 0

    # Zero-based 41:57 = PDF pages 42--57 = printed pages 40--55.
    for page_index in range(41, 57):
        page_lines = _lines(document[page_index])
        for left, right in ((80, 300), (300, 530)):
            column = [line for line in page_lines if left <= line["x"] < right]
            labels = sorted(
                (line for line in column if line["text"] in SITES),
                key=lambda line: line["y"],
            )
            if len(labels) % 4:
                raise ValueError(
                    f"Incomplete site group on PDF page {page_index + 1}, x={left}"
                )

            for group_index in range(0, len(labels), 4):
                group = labels[group_index : group_index + 4]
                if tuple(label["text"] for label in group) != SITES:
                    raise ValueError(
                        f"Bad site order on PDF page {page_index + 1}: {group}"
                    )

                previous_bottom = labels[group_index - 1]["y"] + 3 if group_index else 50
                gloss_candidates = [
                    line
                    for line in column
                    if abs(line["x"] - group[0]["x"]) < 1
                    and previous_bottom < line["y"] < group[0]["y"] - 3
                    and line["text"] not in SITES
                ]
                if not gloss_candidates:
                    raise ValueError(
                        f"Missing gloss on PDF page {page_index + 1}, x={left}"
                    )
                gloss = max(gloss_candidates, key=lambda line: line["y"])["text"]
                # One unusually long English gloss runs into the damaged
                # Devanagari extraction in the adjacent table cell.
                gloss_tokens = []
                for token in gloss.split():
                    if not token.isascii():
                        break
                    gloss_tokens.append(token)
                gloss = " ".join(gloss_tokens)
                concept_number += 1

                for label in group:
                    form_candidates = [
                        line
                        for line in column
                        if line["x"] > label["x2"] + 5
                        and line["x"] < left + 170
                        and abs(line["y"] - label["y"]) < 3
                        and not re.fullmatch(r"[\d Xx,]+", line["text"])
                    ]
                    if len(form_candidates) > 1:
                        raise ValueError(
                            f"Multiple forms for {gloss!r}/{label['text']} "
                            f"on PDF page {page_index + 1}"
                        )
                    # Seven Dhankuta cells are genuinely blank in the source:
                    # eggplant, groundnut, and the numerals six through ten.
                    if not form_candidates:
                        continue
                    form, note = _clean_form(form_candidates[0]["text"])
                    records.append(
                        (
                            concept_number,
                            gloss,
                            label["text"],
                            form,
                            note,
                            page_index - 1,
                        )
                    )

    if concept_number != 218:
        raise ValueError(f"Expected 218 concepts, got {concept_number}")
    if len(records) != 865:
        raise ValueError(f"Expected 865 printed forms, got {len(records)}")
    if Counter(site for _, _, site, _, _, _ in records) != {
        "Dhankuta": 211,
        "Nawalparasi": 218,
        "Panchthar": 218,
        "Sarlahi": 218,
    }:
        raise ValueError("Unexpected site totals")
    return records


def main() -> None:
    extracted = extract()
    occurrence: Counter[tuple[int, str]] = Counter()
    rows = []
    for concept, gloss, site, form, note, printed_page in extracted:
        occurrence[(concept, site)] += 1
        rows.append(
            [
                LECTS[site],
                "",
                form,
                gloss,
                "",
                form,
                "",
                f"{SOURCE}[p. {printed_page}]",
                "",
                note,
                f"eastern-magar:{concept}:{site.lower()}:{occurrence[(concept, site)]}",
            ]
        )

    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)
    print(f"Wrote {len(rows)} forms from {len(LECTS)} lects to {OUTPUT}")


if __name__ == "__main__":
    main()
