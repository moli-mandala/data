"""Extract the four target Thakali wordlists from Webster (2021 [1993])."""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

import pdfplumber


HERE = Path(__file__).resolve().parent
PDF = HERE.parents[3] / "tmp" / "pdfs" / "thakali" / "official.pdf"
OUTPUT = HERE.parent / "20260813-thakali.csv"
SOURCE = "webster2021thakali"
LECTS = {
    "TUK": "thakali_tukuche",
    "MAR": "thakali_marpha",
    "TIN": "thakali_thini",
    "SYA": "thakali_syang",
}
CONTROL_CODES = {"WTa", "GGu", "NEP"}
def _lines(words, left, right):
    selected = [word for word in words if left <= word["x0"] < right]
    selected.sort(key=lambda word: (round(word["top"], 1), word["x0"]))
    lines = []
    for word in selected:
        top = round(word["top"], 1)
        if not lines or abs(lines[-1][0] - top) > 1.5:
            lines.append((top, [word]))
        else:
            lines[-1][1].append(word)
    return lines


def _join(words, *, compact=False):
    if not words:
        return ""
    value = words[0]["text"]
    for previous, current in zip(words, words[1:]):
        gap = current["x0"] - previous["x1"]
        value += (" " if not compact or gap > 2 else "") + current["text"]
    return value.strip()


def extract(pdf_path: Path = PDF):
    records = []
    headers = []
    variants = Counter()
    active = []

    def parse_segment(words, intervals, top, bottom, page_index):
        for left, right, concept in intervals:
            current_lect = None
            selected = [word for word in words if top <= word["top"] < bottom]
            for _, line in _lines(selected, left, right):
                tokens = [word["text"] for word in line]
                if not tokens:
                    continue
                code = tokens[0]
                start = 0
                if code == "TU" and concept[0] == 219:
                    code = "TUK"  # one source typo in the final table
                if code in LECTS or code in CONTROL_CODES:
                    current_lect = code
                    start = 1
                if current_lect is None or start >= len(tokens):
                    continue
                if not re.fullmatch(r"\d+", tokens[start]):
                    continue
                group = int(tokens[start])
                form_words = line[start + 1 :]
                if current_lect not in LECTS or group == 0 or not form_words:
                    continue
                cell = _join(form_words, compact=True)
                cell = re.sub(r"\s+\((?:listen)\)$", "", cell)
                number, gloss = concept
                note = ""
                if cell.endswith("(N)"):
                    cell = cell[:-3].strip()
                    note = "Marked as a Nepali borrowing in the source"
                if cell.endswith("(sm.river)"):
                    cell = cell[: -len("(sm.river)")].strip()
                    note = "Source note: small river"
                if (number, current_lect) == (197, "TUK"):
                    # The initial palatal stop has a broken ToUnicode mapping;
                    # the rendered official PDF reads [ɟʌ̃lʌ].
                    cell = cell.replace("�ʌ̃lʌ", "ɟʌ̃lʌ")
                for form in (value.strip() for value in cell.split(";")):
                    if not form:
                        continue
                    variants[(number, current_lect)] += 1
                    records.append(
                        (
                            number,
                            gloss,
                            current_lect,
                            form,
                            page_index - 5,
                            variants[(number, current_lect)],
                            note,
                        )
                    )

    with pdfplumber.open(pdf_path) as document:
        # PDF pages 24--45 (printed pages 18--39) contain Appendix A.4.
        for page_index in range(23, 45):
            words = document.pages[page_index].extract_words(
                x_tolerance=1, y_tolerance=2, keep_blank_chars=False
            )
            header_words = [
                word for word in words if re.fullmatch(r"\d{1,3}\.", word["text"])
            ]
            header_rows = []
            for word in sorted(header_words, key=lambda item: (round(item["top"], 1), item["x0"])):
                top = round(word["top"], 1)
                if not header_rows or abs(header_rows[-1][0] - top) > 1.5:
                    header_rows.append((top, [word]))
                else:
                    header_rows[-1][1].append(word)

            cursor = 0
            for header_top, row_headers in header_rows:
                if active and cursor < header_top:
                    parse_segment(words, active, cursor, header_top - 1, page_index)
                starts = [word["x0"] for word in row_headers]
                new_active = []
                for index, header_word in enumerate(row_headers):
                    left = header_word["x0"] - 2
                    right = starts[index + 1] - 2 if index + 1 < len(starts) else 610
                    number = int(header_word["text"][:-1])
                    header_line = [
                        word
                        for word in words
                        if abs(word["top"] - header_word["top"]) <= 1.5
                        and header_word["x1"] <= word["x0"] < right
                    ]
                    gloss = _join(header_line).strip()
                    headers.append((number, gloss))
                    new_active.append((left, right, (number, gloss)))
                active = new_active
                cursor = header_top + 2
            if active:
                parse_segment(words, active, cursor, 792, page_index)

    if [number for number, _ in sorted(headers)] != list(range(1, 221)):
        raise ValueError(f"Unexpected concept headers: {sorted(headers)}")
    if any("�" in row[3] for row in records):
        raise ValueError(f"Replacement character in target form: {[r for r in records if '�' in r[3]]}")
    return records


def main():
    rows = []
    for number, gloss, code, form, page, variant, note in extract():
        lect = LECTS[code]
        rows.append(
            [
                lect,
                "",
                form,
                gloss,
                "",
                form,
                note,
                f"{SOURCE}[p. {page}]",
                "",
                "",
                f"thakali:{number}:{lect}:{variant}",
            ]
        )
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)
    print(f"Wrote {len(rows)} forms from {len(LECTS)} target Thakali lects to {OUTPUT}")


if __name__ == "__main__":
    main()
