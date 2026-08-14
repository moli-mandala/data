"""Extract the four Nepal Kurux target wordlists from Shackelford et al. (2022)."""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

import pdfplumber


HERE = Path(__file__).resolve().parent
PDF = HERE.parents[3] / "tmp" / "pdfs" / "kurux" / "official.pdf"
OUTPUT = HERE.parent / "20260813-kurux-nepal.csv"
SOURCE = "shackelford-swenson-chaudhary-maggard2022kurux"
LECTS = (
    (264.0, 349.0, "kurux_lochani"),
    (349.0, 434.0, "kurux_bhokraha"),
    (434.0, 519.0, "kurux_siddhapur"),
    (519.0, 604.0, "kurux_tokla"),
)
GLYPH_FIXES = {
    (110, "kurux_lochani"): "bʰãĩ̯s",
    (110, "kurux_siddhapur"): "bʰʌhãĩ̯s",
    (110, "kurux_tokla"): "bʰʌhãĩ̯s",
    (183, "kurux_lochani"): "t͡ʃaĩ̯kʰa",
    (183, "kurux_bhokraha"): "t͡ʃʌĩ̯kʰa",
    (183, "kurux_siddhapur"): "t͡ʃʌĩ̯kʰa",
    (183, "kurux_tokla"): "t͡ʃaĩ̯kʰa",
    (189, "kurux_lochani"): "pʌĩ̯ja",
    (189, "kurux_bhokraha"): "pʌĩ̯ja",
    (189, "kurux_siddhapur"): "pʌĩ̯ja",
    (189, "kurux_tokla"): "pʌĩ̯nja",
    (288, "kurux_siddhapur"): "t͡ʃĩːχʌsə",
    (289, "kurux_lochani"): "amat͡ʃĩχa",
    (289, "kurux_bhokraha"): "mat͡ʃĩχa",
    (289, "kurux_siddhapur"): "mat͡ʃĩχa",
    (289, "kurux_tokla"): "amat͡ʃĩχa",
    (328, "kurux_tokla"): "nimhãĩ̯",
}


def _lines(words):
    lines = []
    for word in sorted(words, key=lambda item: (round(item["top"], 1), item["x0"])):
        if not lines or abs(lines[-1][0] - word["top"]) > 3:
            lines.append((word["top"], [word]))
        else:
            lines[-1][1].append(word)
    return [sorted(line, key=lambda item: item["x0"]) for _, line in lines]


def _cell(words, left, right, top, bottom):
    selected = [
        word for word in words
        if left <= word["x0"] < right and top - 4 <= word["top"] < bottom - 4
    ]
    rendered = []
    for line in _lines(selected):
        value = line[0]["text"]
        for previous, current in zip(line, line[1:]):
            gap = current["x0"] - previous["x1"]
            value += (" " if gap > 2.5 else "") + current["text"]
        rendered.append(value)
    value = ""
    for index, line_value in enumerate(rendered):
        if index:
            previous = _lines(selected)[index - 1]
            current = _lines(selected)[index]
            wrapped_token = previous[-1]["x1"] > right - 16 and current[0]["x0"] < left + 2
            value += "" if wrapped_token else " "
        value += line_value
    value = value.strip()
    value = re.sub(r"\s+(?=[\u0300-\u036f])", "", value)
    value = re.sub(r"(?<=[\u0300-\u036f])\s+(?=[\u0300-\u036f])", "", value)
    # The PDF positions a combining tilde after the following consonant in
    # its text layer, although the rendered mark is over the preceding vowel.
    value = re.sub(
        r"([aeiouəɛɔɑʌɪɘæ])([bcdfghjklmnpqrstvwxyzχʃʒɖɽɾŋɳʈ]+)̃",
        r"\1̃\2",
        value,
    )
    value = value.replace("dʒ͡", "d͡ʒ")
    return value


def extract(pdf_path: Path = PDF):
    records = []
    concepts = []
    variants = Counter()
    with pdfplumber.open(pdf_path) as pdf:
        for page_index in range(51, 64):
            words = pdf.pages[page_index].extract_words(x_tolerance=1, y_tolerance=2)
            headers = [
                (int(word["text"]), word["top"])
                for word in words
                if 70 <= word["x0"] < 100 and re.fullmatch(r"\d{1,3}", word["text"])
            ]
            for position, (number, top) in enumerate(headers):
                bottom = headers[position + 1][1] if position + 1 < len(headers) else 570
                gloss = _cell(words, 105, 193, top, bottom)
                concepts.append((number, gloss))
                for left, right, lect in LECTS:
                    cell = _cell(words, left, right, top, bottom)
                    cell = GLYPH_FIXES.get((number, lect), cell)
                    if not cell or cell == "---":
                        continue
                    for form in (part.strip() for part in cell.split(",")):
                        if not form or form == "---":
                            continue
                        variants[(number, lect)] += 1
                        records.append(
                            (number, gloss, lect, form, page_index - 7, variants[(number, lect)])
                        )
    if [number for number, _ in concepts] != list(range(1, 339)):
        raise ValueError(f"Unexpected concept sequence: {concepts}")
    return records


def main():
    rows = []
    for number, gloss, lect, form, page, variant in extract():
        rows.append(
            [
                lect,
                "",
                form,
                gloss,
                "",
                form,
                "",
                f"{SOURCE}[p. {page}]",
                "",
                "",
                f"kurux-nepal:{number}:{lect}:{variant}",
            ]
        )
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)
    print(f"Wrote {len(rows)} forms from {len(LECTS)} Nepal Kurux target lects to {OUTPUT}")


if __name__ == "__main__":
    main()
