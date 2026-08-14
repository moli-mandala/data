"""Extract the six target wordlists from Shackelford (2019).

Appendix A.4 (PDF pages 56--88, printed pages 48--80) contains 325
elicitation items from three Dewas Rai sites, Done/Danuwar, Danuwar,
and Kochariya.  The Nepali comparison list is excluded.  Six verb-table
pages have a broken PDF ToUnicode map; their embedded Charis SIL glyph
IDs are decoded directly, without OCR.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

import pymupdf as fitz


HERE = Path(__file__).resolve().parent
PDF = HERE.parents[3] / "tmp" / "pdfs" / "dewas-rai-2019" / "source.pdf"
OUTPUT = HERE.parent / "20260813-dewas-rai.csv"
SOURCE = "shackelford2019dewas-rai"

LECTS = {
    "Mahendra Jhyadi": "dewas_rai_mahendra_jhyadi",
    "Singoul (Rai)": "dewas_rai_singoul",
    "Majhgaun": "dewas_rai_majhgaun",
    "Jaretar": "done_danuwar_jaretar",
    "Chandanpur": "danuwar_chandanpur",
    "Singoul (Kochariya)": "kochariya_singoul",
}
SITE_STARTS = {"Mahendra", "Singoul", "Majhgaun", "Jaretar", "Chandanpur", "Nepali"}

# Glyph IDs from the subset Charis SIL 5.000 fonts used on PDF pages
# 80--85.  These were matched against SIL's original font outlines.
GLYPHS = {
    3: " ", 11: "(", 12: ")", 16: "-", 17: ".", 18: "/",
    19: "0", 20: "1", 21: "2", 22: "3", 23: "4", 24: "5",
    25: "6", 26: "7", 27: "8", 28: "9", 36: "A", 37: "B",
    38: "C", 39: "D", 45: "J", 46: "K", 48: "M", 49: "N",
    51: "P", 53: "R", 54: "S", 55: "T", 68: "a", 69: "b",
    70: "c", 71: "d", 72: "e", 73: "f", 74: "g", 75: "h",
    76: "i", 77: "j", 78: "k", 79: "l", 80: "m", 81: "n",
    82: "o", 83: "p", 85: "r", 86: "s", 87: "t", 88: "u",
    89: "v", 90: "w", 91: "x", 92: "y", 93: "z", 597: "β",
    109: " ", 125: " ", 1193: " ", 2683: " ",
    620: "ɔ", 709: "ɖ", 1092: "ʰ", 1292: "ɪ", 1943: "ɹ",
    1373: "ʲ", 1950: "ɾ", 2001: "ʃ", 2042: "ʈ", 2089: "ũ",
    2117: "ṳ", 2268: "ʌ", 2369: "ʎ",
    2451: "ʒ", 2643: "̯", 2723: "̤",
}


def _decoded_words(page: fitz.Page) -> list[tuple[float, float, float, float, str]]:
    """Reconstruct Charis words from glyph IDs on the six malformed pages."""
    words = []
    for span in page.get_texttrace():
        if span["font"] != "CharisSIL":
            continue
        current = []

        def finish() -> None:
            if not current:
                return
            chars = [item[4] for item in current]
            words.append(
                (
                    min(item[0] for item in current),
                    min(item[1] for item in current),
                    max(item[2] for item in current),
                    max(item[3] for item in current),
                    "".join(chars),
                )
            )
            current.clear()

        previous_y = None
        previous_right = None
        for _unicode, glyph, origin, bbox in span["chars"]:
            if glyph not in GLYPHS:
                raise ValueError(f"Unmapped Charis glyph ID {glyph}")
            char = GLYPHS[glyph]
            y = origin[1]
            if (
                char == " "
                or (previous_y is not None and abs(y - previous_y) > 2)
                or (
                    current
                    and previous_right is not None
                    and bbox[0] - previous_right > 2
                )
            ):
                finish()
            if char != " ":
                current.append((*bbox, char))
            previous_y = y
            previous_right = bbox[2] if char != " " else None
        finish()
    return words


def _words(page: fitz.Page, page_index: int) -> list[tuple[float, float, float, float, str]]:
    if page_index in {68, 79, 80, 81, 84}:
        return _decoded_words(page)
    return [(x0, y0, x1, y1, text) for x0, y0, x1, y1, text, *_ in page.get_text("words")]


def _join_cell(
    words: list[tuple[float, float, float, float, str]],
    left: float,
    right: float,
    top: float,
    bottom: float,
) -> str:
    # PDF floating-point coordinates occasionally land a few ten-thousandths
    # left of the printed table rule.
    selected = [
        w
        for w in words
        if left - 0.5 <= w[0] < right - 0.5 and top <= w[1] < bottom
    ]
    if not selected:
        return ""
    lines: list[list[tuple[float, float, float, float, str]]] = []
    for word in sorted(selected, key=lambda w: (w[1], w[0])):
        for line in lines:
            if abs(line[0][1] - word[1]) < 2:
                line.append(word)
                break
        else:
            lines.append([word])
    output = []
    for line in sorted(lines, key=lambda line: line[0][1]):
        text = ""
        previous_right = None
        for x0, _y0, x1, _y1, token in sorted(line, key=lambda w: w[0]):
            if text and previous_right is not None and x0 - previous_right > 2:
                text += " "
            text += token
            previous_right = x1
        if text:
            output.append(text)
    return " ".join(output).strip()


def _clean_form(form: str) -> str:
    """Remove the table's standalone A--D/x similarity-code cells."""
    return " ".join(token for token in form.split() if token not in {"A", "B", "C", "D", "x"})


def _english_gloss(gloss: str) -> str:
    """Keep the English prompt; the following Nepali prompt has a broken font map."""
    before_nepali = re.split(r"[\u0900-\u097f]", gloss, maxsplit=1)[0]
    match = re.match(r"[A-Za-z0-9 /()'.,!?-]*", before_nepali)
    return match.group(0).strip() if match else before_nepali.strip()


def _labels(
    words: list[tuple[float, float, float, float, str]],
    x: float,
    form_left: float,
    top: float,
    bottom: float,
) -> list[tuple[float, str]]:
    labels = []
    for x0, y0, _x1, _y1, text in words:
        if abs(x0 - x) > 2 or not (top < y0 < bottom) or text not in SITE_STARTS:
            continue
        label = _join_cell(words, x, form_left - 3, y0 - 2, y0 + 4)
        if label in {*LECTS, "Nepali"}:
            labels.append((y0, label))
    return sorted(set(labels))


def _ordinary_page(
    words: list[tuple[float, float, float, float, str]],
    page_index: int,
    x_starts: tuple[float, float],
    form_offset: float,
    form_width: float,
    form_ranges: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> list[tuple[int, str, str, str, int]]:
    records = []
    headers = sorted(
        (x0, y0, int(match.group(1)))
        for x0, y0, _x1, _y1, text in words
        if (match := re.fullmatch(r"(\d+)\.", text)) and 1 <= int(match.group(1)) <= 325
    )
    for x, top, concept in headers:
        side = 0 if x < 200 else 1
        base_x = x_starts[side]
        if abs(x - base_x) > 8:
            continue
        next_headers = [
            y
            for hx, y, _c in headers
            if (0 if hx < 200 else 1) == side and y > top
        ]
        bottom = min(next_headers) if next_headers else 770
        column_right = x_starts[1] - 3 if side == 0 else 555
        heading = _join_cell(words, x, column_right, top - 2, top + 5)
        gloss = _english_gloss(re.sub(r"^\d+\.\s*", "", heading).strip())
        if form_ranges is None:
            form_left = base_x + form_offset
            form_right = form_left + form_width
        else:
            form_left, form_right = form_ranges[side]
        label_xs = [
            round(x0, 1)
            for x0, y0, _x1, _y1, text in words
            if top < y0 < bottom
            and text in SITE_STARTS
            and ((x0 < 200) == (side == 0))
        ]
        label_x = Counter(label_xs).most_common(1)[0][0] if label_xs else x
        labels = _labels(words, label_x, form_left, top, bottom)
        for index, (label_top, label) in enumerate(labels):
            row_bottom = labels[index + 1][0] - 2 if index + 1 < len(labels) else bottom - 2
            if label == "Nepali":
                continue
            form = _clean_form(_join_cell(words, form_left, form_right, label_top - 2, row_bottom))
            if form and form != "-":
                records.append((concept, gloss, LECTS[label], form, page_index - 7))
    return records


def _verb_page(
    words: list[tuple[float, float, float, float, str]], page_index: int
) -> list[tuple[int, str, str, str, int]]:
    records = []
    headers = sorted(
        (x0, y0, int(match.group(1)), int(match.group(2)))
        for x0, y0, _x1, _y1, text in words
        if (match := re.fullmatch(r"(\d+)/(\d+)\.", text))
    )
    for _x, top, past_concept, negative_concept in headers:
        compact = _x < 100
        label_x = 95.0 if compact else 107.2
        root_right = 319 if compact else 318
        past_left = 323.0 if compact else 321.6
        negative_left = 401.9 if compact else 395.8
        next_headers = [y for _hx, y, _a, _b in headers if y > top]
        bottom = min(next_headers) if next_headers else 770
        heading = _join_cell(words, label_x, root_right, top - 2, top + 5)
        gloss = _english_gloss(re.sub(r"^\d+/\d+\.\s*", "", heading).strip())
        labels = _labels(words, label_x, 195 if compact else 201, top, bottom)
        for index, (label_top, label) in enumerate(labels):
            row_bottom = labels[index + 1][0] - 2 if index + 1 < len(labels) else bottom - 2
            if label == "Nepali":
                continue
            lect = LECTS[label]
            past = _clean_form(_join_cell(words, past_left, negative_left, label_top - 2, row_bottom))
            negative = _clean_form(_join_cell(words, negative_left, 555, label_top - 2, row_bottom))
            if past and past != "-":
                records.append((past_concept, f"{gloss} (3S-PT)", lect, past, page_index - 7))
            if negative and negative != "-":
                records.append((negative_concept, f"{gloss} (2S-neg)", lect, negative, page_index - 7))
    return records


def _single_verb(
    words: list[tuple[float, float, float, float, str]],
    concept: int,
    top: float,
    bottom: float,
    gloss: str,
    page_index: int,
) -> list[tuple[int, str, str, str, int]]:
    """Parse the two single-column stative predicates numbered 259 and 260."""
    records = []
    labels = _labels(words, 107.2, 201, top, bottom)
    for index, (label_top, label) in enumerate(labels):
        row_bottom = labels[index + 1][0] - 2 if index + 1 < len(labels) else bottom - 2
        if label == "Nepali":
            continue
        form = _clean_form(_join_cell(words, 321.6, 395.8, label_top - 2, row_bottom))
        if form and form != "-":
            records.append((concept, f"{gloss} (3S-PT)", LECTS[label], form, page_index - 7))
    return records


def extract(pdf_path: Path = PDF) -> list[tuple[int, str, str, str, int]]:
    document = fitz.open(pdf_path)
    records = []

    # Concepts 1--238: two-column nominal and function-word tables.
    for page_index in range(55, 79):
        words = _words(document[page_index], page_index)
        if page_index == 68:
            records.extend(_ordinary_page(words, page_index, (97.3, 308.0), 104.4, 66))
        else:
            records.extend(_ordinary_page(words, page_index, (84.6, 308.6), 111, 70))

    # Concepts 239--313: paired past and negative-imperative verb forms.
    for page_index in range(79, 87):
        words = _words(document[page_index], page_index)
        records.extend(_verb_page(words, page_index))

    # These adjacent stative predicates each have a single past-tense column,
    # rather than forming one past/negative pair like the surrounding rows.
    words = _words(document[81], 81)
    header_259 = next(w for w in words if w[4] == "259.")
    header_260 = next(w for w in words if w[4] == "260.")
    header_261 = next(w for w in words if w[4] == "261/262.")
    records.extend(_single_verb(words, 259, header_259[1], header_260[1], "be hungry", 81))
    records.extend(_single_verb(words, 260, header_260[1], header_261[1], "be thirsty", 81))

    # Concept 309 is the only unpaired verb and appears on page 87.
    words = _words(document[86], 86)
    header_309 = next(w for w in words if w[4] == "309.")
    next_header = next(w for w in words if w[4] == "310/311.")
    labels = _labels(words, 95, 195, header_309[1], next_header[1])
    for index, (label_top, label) in enumerate(labels):
        row_bottom = labels[index + 1][0] - 2 if index + 1 < len(labels) else next_header[1] - 2
        if label != "Nepali":
            form = _clean_form(_join_cell(words, 323, 401.9, label_top - 2, row_bottom))
            if form and form != "-":
                records.append((309, "watch/see (3S-PT)", LECTS[label], form, 79))

    # Concepts 314--325: two-column pronoun table on pages 87--88.
    for page_index in (86, 87):
        words = _words(document[page_index], page_index)
        ordinary = _ordinary_page(
            words,
            page_index,
            (84.2, 317.5),
            0,
            0,
            form_ranges=((184, 262), (420, 490)),
        )
        records.extend(row for row in ordinary if row[0] >= 314)

    coverage = {concept for concept, _gloss, _lect, _form, _page in records}
    expected_empty = {213, 215}  # Every target cell is explicitly "-" in the source.
    expected = set(range(1, 326)) - expected_empty
    if coverage != expected:
        raise ValueError(
            f"Unexpected concept coverage: missing {sorted(expected - coverage)}; "
            f"unexpected {sorted(coverage - expected)}"
        )
    return records


def main() -> None:
    extracted = extract()
    occurrence: Counter[tuple[int, str]] = Counter()
    rows = []
    for concept, gloss, lect, form, printed_page in extracted:
        # The clean page-66 Charis subset maps the visually printed /i/ in
        # bhainsi 'buffalo' to Cyrillic U+04E1; recover the source glyph.
        form = form.replace("ӡ", "i")
        occurrence[(concept, lect)] += 1
        rows.append(
            [
                lect, "", form, gloss, "", form, "",
                f"{SOURCE}[p. {printed_page}]", "", "",
                f"dewas-rai:{concept}:{lect}:{occurrence[(concept, lect)]}",
            ]
        )
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)
    print(f"Wrote {len(rows)} forms from {len(LECTS)} lects to {OUTPUT}")


if __name__ == "__main__":
    main()
