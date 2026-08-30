#!/usr/bin/env python3
"""Extract the embedded-text scaffold for Desia Appendix B.5.

This is a locating/transcription aid only.  ``import_desia.py`` deliberately
does not read its output; installation is gated on the separately completed
``manual_review.tsv`` visual-review ledger.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pdfplumber


HERE = Path(__file__).resolve().parent
WORKSPACE_ROOT = HERE.parents[5]
DEFAULT_PDF = WORKSPACE_ROOT / "tmp/pdfs/desia-2021-056/source.pdf"
SCAFFOLD = HERE / "text_layer_scaffold.txt"
REVIEW = HERE / "manual_review.tsv"

SITES = [
    "Konda Maliguda", "Patta Maliguda", "Dame side", "Sabhapatiguda",
    "Chhatrabor", "Sourakundi", "Sindhiguda", "Souraguda", "Kakalpoda",
    "Gagnapur", "Gemelput", "Gumalput", "Kantigad", "Bodgaon", "Potenda",
    "Jujhari", "Ghumar", "Burja", "Aunli",
]
FIELDS = [
    "PDF_Page", "Printed_Page", "Line", "Item", "Gloss", "Site",
    "Manual_Form", "Similarity_Group", "Confidence", "Review_Method",
    "Review_Status", "Source_Text",
]


def line_groups(words: list[dict]) -> list[list[dict]]:
    groups: list[list[dict]] = []
    for word in sorted(words, key=lambda w: (round(w["top"], 1), w["x0"])):
        if not groups or abs(groups[-1][0]["top"] - word["top"]) > 1.0:
            groups.append([word])
        else:
            groups[-1].append(word)
    return groups


def side_lines(page, side: str) -> list[tuple[float, str]]:
    if side == "left":
        x0, x1 = 65, 305
    else:
        x0, x1 = 315, 545
    words = [w for w in page.extract_words(x_tolerance=1, y_tolerance=2)
             if x0 <= w["x0"] < x1]
    return [(group[0]["top"], " ".join(w["text"] for w in group))
            for group in line_groups(words)]


def parse(pdf_path: Path) -> tuple[str, list[dict[str, str]]]:
    scaffold: list[str] = []
    rows: list[dict[str, str]] = []
    current_item: int | None = None
    current_gloss = ""
    with pdfplumber.open(pdf_path) as pdf:
        for pdf_page in range(80, 128):
            page = pdf.pages[pdf_page - 1]
            scaffold.append(f"===== PDF {pdf_page} / PRINTED {pdf_page - 9} =====")
            for side in ("left", "right"):
                scaffold.append(f"--- {side.upper()} ---")
                for line_number, (top, text) in enumerate(side_lines(page, side), 1):
                    scaffold.append(f"{line_number:02d}\t{top:07.2f}\t{text}")
                    header = re.fullmatch(r"(\d{1,3})\s+(.+)", text)
                    if header and 1 <= int(header.group(1)) <= 210:
                        current_item = int(header.group(1))
                        current_gloss = header.group(2).strip()
                        continue
                    matched_site = next(
                        (site for site in SITES if text == site or text.startswith(site + " ")),
                        None,
                    )
                    if not matched_site:
                        continue
                    if current_item is None:
                        raise ValueError(f"Response before item header: p{pdf_page} {side} {text!r}")
                    remainder = text[len(matched_site):].strip()
                    tokens = remainder.split()
                    group = ""
                    if tokens and re.fullmatch(r"[1-8]", tokens[-1]):
                        group = tokens.pop()
                    form = " ".join(tokens)
                    if not form:
                        raise ValueError(f"Empty response scaffold: p{pdf_page} {side} {text!r}")
                    rows.append({
                        "PDF_Page": str(pdf_page),
                        "Printed_Page": str(pdf_page - 9),
                        "Line": f"{side[0].upper()}{line_number:02d}@{top:.2f}",
                        "Item": str(current_item),
                        "Gloss": current_gloss,
                        "Site": matched_site,
                        "Manual_Form": form,
                        "Similarity_Group": group,
                        "Confidence": "pending",
                        "Review_Method": "embedded text scaffold only; not reviewed",
                        "Review_Status": "pending",
                        "Source_Text": text,
                    })
            scaffold.append("")
    return "\n".join(scaffold) + "\n", rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--write-review-scaffold", action="store_true")
    args = parser.parse_args()
    text, rows = parse(args.pdf)
    SCAFFOLD.write_text(text, encoding="utf-8")
    if args.write_review_scaffold:
        with REVIEW.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    print(f"scaffold response lines: {len(rows)}")


if __name__ == "__main__":
    main()
