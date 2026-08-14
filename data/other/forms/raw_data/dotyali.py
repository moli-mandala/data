"""Extract the four target Dotyali wordlists from Eichentopf & Tupper.

Appendix A.4 of the born-digital SIL PDF (PDF pages 36--48, printed
pages 28--40) contains Unicode IPA for Doti, Baitadi, Darchula, Bajhang,
Nepali, and Kumaoni.  Only the four Dotyali varieties are emitted here;
Nepali and Kumaoni are comparison controls.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pdfplumber


HERE = Path(__file__).resolve().parent
PDF = HERE.parents[3] / "tmp" / "pdfs" / "dotyali" / "silesr2019_004.pdf"
OUTPUT = HERE.parent / "20260813-dotyali.csv"
SOURCE = "eichentopf-tupper2019dotyali"

LECTS = {
    "Doti": "dotyali_doti",
    "Baitadi": "dotyali_baitadi",
    "Darchula": "dotyali_darchula",
    "Bajhang": "dotyali_bajhang",
}

# The positioned text has a separate similarity-group column immediately
# after every transcription column.  These bands select only the IPA cells.
FORM_BANDS = {
    "Doti": (299.0, 349.0),
    "Baitadi": (376.0, 439.0),
    "Darchula": (466.0, 524.0),
    "Bajhang": (551.0, 632.0),
}
GLOSS_BAND = (101.0, 164.0)


def _join_words(words: list[dict]) -> str:
    """Reassemble a positioned cell without detaching combining marks."""
    lines: list[list[dict]] = []
    for word in sorted(words, key=lambda item: (round(item["top"], 1), item["x0"])):
        if not lines or abs(lines[-1][0]["top"] - word["top"]) > 1.0:
            lines.append([word])
        else:
            lines[-1].append(word)

    rendered = []
    for line in lines:
        text = ""
        previous_right = None
        for word in sorted(line, key=lambda item: item["x0"]):
            token = word["text"]
            attach = (
                not text
                or unicodedata.combining(token[0])
                or (previous_right is not None and word["x0"] - previous_right < 1.0)
            )
            text += token if attach else f" {token}"
            previous_right = word["x1"]
        rendered.append(text)
    return re.sub(r"\s+", " ", " ".join(rendered)).strip()


def extract(pdf_path: Path = PDF) -> list[tuple[int, str, str, str, int, str]]:
    """Return ``(number, gloss, site, form, printed_page, note)`` records."""
    records = []
    concepts = []

    with pdfplumber.open(pdf_path) as document:
        # Zero-based 35:48 = PDF pages 36--48 = printed pages 28--40.
        for page_index in range(35, 48):
            words = document.pages[page_index].extract_words(
                x_tolerance=1,
                y_tolerance=2,
                keep_blank_chars=False,
                use_text_flow=False,
            )
            anchors = sorted(
                (
                    (int(word["text"]), word["top"])
                    for word in words
                    if 70.0 <= word["x0"] < 90.0
                    and word["text"].isdigit()
                    and int(word["text"]) <= 325
                ),
                key=lambda item: item[1],
            )
            content_top = anchors[0][1] - 8.0
            content_bottom = anchors[-1][1] + 8.0
            for number, anchor_y in anchors:
                concepts.append(number)

                def nearest(word: dict) -> bool:
                    return min(anchors, key=lambda item: abs(item[1] - word["top"])) == (
                        number,
                        anchor_y,
                    )

                gloss = _join_words(
                    [
                        word
                        for word in words
                        if GLOSS_BAND[0] <= word["x0"] < GLOSS_BAND[1]
                        and content_top <= word["top"] <= content_bottom
                        and nearest(word)
                    ]
                )
                if not gloss:
                    raise ValueError(f"Missing English gloss for item {number}")

                printed_page = page_index - 7
                for site, (left, right) in FORM_BANDS.items():
                    cell = _join_words(
                        [
                            word
                            for word in words
                            if left <= word["x0"] < right
                            and content_top <= word["top"] <= content_bottom
                            and nearest(word)
                        ]
                    )
                    if not cell or cell == "-":
                        continue
                    note = ""
                    if number == 317 and site == "Baitadi" and cell == "(sɑme ɑs 316)":
                        cell = "təməro"
                        note = "Source: same as item 316"
                    else:
                        qualifier = re.search(r"\s*(\([^)]*\))$", cell)
                        if qualifier:
                            note = f"Source qualifier: {qualifier.group(1)}"
                            cell = cell[: qualifier.start()].strip()
                    for alternate in re.split(r"\s*/\s*", cell):
                        alternate = alternate.strip()
                        if alternate:
                            records.append(
                                (number, gloss, site, alternate, printed_page, note)
                            )

    expected = list(range(1, 197)) + list(range(211, 326))
    if concepts != expected:
        raise ValueError(f"Expected the 311 printed lexical items, got {concepts}")
    return records


def main() -> None:
    extracted = extract()
    occurrence: Counter[tuple[int, str]] = Counter()
    rows = []
    for concept, gloss, site, form, printed_page, note in extracted:
        occurrence[(concept, site)] += 1
        rows.append(
            [
                LECTS[site],
                "",
                form,
                gloss,
                "",
                form,
                note,
                f"{SOURCE}[p. {printed_page}]",
                "",
                "",
                f"dotyali:{concept}:{site.lower()}:{occurrence[(concept, site)]}",
            ]
        )

    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)
    print(f"Wrote {len(rows)} forms from {len(LECTS)} lects to {OUTPUT}")


if __name__ == "__main__":
    main()
