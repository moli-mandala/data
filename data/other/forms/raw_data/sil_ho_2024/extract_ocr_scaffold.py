#!/usr/bin/env python3
"""Create a non-authoritative OCR locator scaffold for Appendix D.3.

Nothing emitted by this script is accepted transcription.  ``manual_review.tsv``
is maintained separately and must be populated/verified from rendered page images.
"""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
PDF = REPO / "tmp/pdfs/ho_2024/JLSR2024_009.pdf"
OCR_DIR = HERE / "ocr_raw"
OUTPUT = HERE / "ocr_scaffold.tsv"
SHA256 = "5ca30882dc5ed0f8480c9710e5fc0e08bf4d92e27d591582e3d953709ec1f9d1"

LEFT = ["HO1", "HTH", "HKA", "HKE", "HCH", "HCU", "HSU", "HSA", "HJO", "HDH", "HBG", "HO2", "HRA", "HO3"]
RIGHT = ["HOP", "HBA", "HNI", "BBG", "BMA", "BOP", "BRA", "BGH", "MU1", "MU2", "SA1", "SBA", "OCU"]
SITES = LEFT + RIGHT
ROW_SLOT = {code: slot for slot, code in enumerate(LEFT)}
ROW_SLOT.update({code: slot for code, slot in zip(RIGHT, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])})


def norm(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper()).replace("O", "0")


def distance(a: str, b: str) -> int:
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        nxt = [i]
        for j, cb in enumerate(b, 1):
            nxt.append(min(nxt[-1] + 1, dp[j] + 1, dp[j - 1] + (ca != cb)))
        dp = nxt
    return dp[-1]


def closest_code(token: str, choices: list[str]) -> str | None:
    key = norm(token)
    ranked = sorted((distance(key, norm(code)), code) for code in choices)
    return ranked[0][1] if ranked and ranked[0][0] <= 1 else None


def read_words(page: int) -> list[dict]:
    path = OCR_DIR / f"page-{page:03d}.tsv"
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    return [
        {**row, "left": int(row["left"]), "top": int(row["top"]),
         "width": int(row["width"]), "height": int(row["height"])}
        for row in rows if row["level"] == "5" and row["text"].strip()
    ]


def baselines(words: list[dict], choices: list[str], xlo: int, xhi: int) -> list[int]:
    hits = []
    for word in words:
        if xlo <= word["left"] <= xhi and closest_code(word["text"], choices):
            hits.append(word["top"])
    tops = sorted(set(hits))
    if len(tops) < 3:
        return []
    # Split at the two largest vertical gaps. This remains stable when one or
    # more code labels are missed by OCR inside a fourteen-row block.
    cuts = sorted(sorted(range(len(tops) - 1), key=lambda i: tops[i + 1] - tops[i], reverse=True)[:2])
    groups = [tops[:cuts[0] + 1], tops[cuts[0] + 1:cuts[1] + 1], tops[cuts[1] + 1:]]
    return [min(group) for group in groups if group]


def extract_page(page: int) -> list[dict[str, str | int]]:
    words = read_words(page)
    starts = baselines(words, LEFT, 175, 250)
    first_item = (page - 72) * 3 + 1
    if len(starts) != 3:
        raise AssertionError(f"OCR could not locate three item blocks on PDF {page}: {starts}")
    rows = []
    for block, start in enumerate(starts):
        item = first_item + block
        next_start = starts[block + 1] if block < 2 else 2050
        # Estimate the typewriter row pitch from recognized left codes.
        candidates = sorted({word["top"] for word in words if start - 8 <= word["top"] < next_start - 30
                             and 175 <= word["left"] <= 250 and closest_code(word["text"], LEFT)})
        diffs = [b - a for a, b in zip(candidates, candidates[1:]) if 20 <= b - a <= 45]
        pitch = sorted(diffs)[len(diffs) // 2] if diffs else 32
        for code in SITES:
            column = "left" if code in LEFT else "right"
            slot = ROW_SLOT[code]
            center = start + slot * pitch + 10
            x0, x1 = ((265, 700) if column == "left" else (855, 1260))
            # Capture OCR words in the row band, excluding the printed code.
            cell_words = [word for word in words if x0 <= word["left"] < x1 and abs((word["top"] + word["height"] / 2) - center) <= max(14, pitch * .48)]
            cell_words.sort(key=lambda word: word["left"])
            response = " ".join(word["text"] for word in cell_words)
            rows.append({
                "Item": item, "Site_Code": code, "PDF_Page": page,
                "Printed_Page": page - 9, "Column": column,
                "OCR_Candidate": response, "OCR_Row_Top": start + slot * pitch,
                "OCR_Pitch": pitch, "Review_Status": "unreviewed",
            })
    return rows


def main() -> None:
    if hashlib.sha256(PDF.read_bytes()).hexdigest() != SHA256:
        raise AssertionError("canonical PDF checksum drift")
    rows = [row for page in range(72, 142) for row in extract_page(page)]
    if len(rows) != 5670 or len({(row["Item"], row["Site_Code"]) for row in rows}) != 5670:
        raise AssertionError("source topology drift")
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} unreviewed OCR locator rows to {OUTPUT}")


if __name__ == "__main__":
    main()
