"""Extract the ten Hajong wordlists from Kim et al. (2011).

Appendix B.3 (PDF pages 41--64, printed pages 40--63) presents 307
elicitation prompts.  Six Indian and four Bangladeshi Hajong site lists are
retained; site code ``0``, the Standard Bangla dictionary comparator, is
excluded.  The PDF is born digital, but its legacy phonetic font is exposed by
text extraction as glyph-name placeholders, which are decoded below.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from collections import Counter
from pathlib import Path

from pypdf import PdfReader


HERE = Path(__file__).resolve().parent
PDF = HERE.parents[4] / "tmp" / "pdfs" / "hajong-2011" / "source.pdf"
OUTPUT = HERE.parent / "20260813-hajong.csv"
SOURCE = "kim-ahmad-kim-sangma2011hajong"

LECTS = {
    "a": "hajong_nugapara",
    "b": "hajong_chilapara",
    "c": "hajong_nirghini",
    "d": "hajong_dalugau",
    "e": "hajong_balachanda",
    "f": "hajong_dhamor",
    "B": "hajong_gopalbari",
    "C": "hajong_gopalpur",
    "D": "hajong_bhalukapara",
    "E": "hajong_nokshi",
}

GLYPHS = {
    "/1007": "ɛ",
    "/1024": "ʃ",
    "/1035": "ʒ",
    "/1104": "ɔ",
    "/1106": "ə",
    "/1117": "ɯ",
    "/1122": "r",
    "/1409": "ɡ",
    "/1418": "ŋ",
    "/1426": "ʈ",
    "/1513": "ɾ",
    "/1522": "ɽ",
    "/1605": "ɖ",
    "/1711": "ɪ",
    "/ch208:0133+6001": "̃",
    "/ch230:0133+6642": "̪",
    "/ch235:0133+6608": "̯",
}

HEADING = re.compile(r"^\s*(\d{1,3})\s{2,}(.+?)\s*$")
ENTRY = re.compile(r"^\s*(?:(\d+)\s+)?(.+?)\s+\[([0B-Ea-f ]+)\]\s*$")


def _decode(value: str) -> str:
    for glyph, ipa in sorted(GLYPHS.items(), key=lambda item: -len(item[0])):
        value = value.replace(glyph, ipa)

    # Superscript aspiration was positioned independently in the source font,
    # so PDF extraction inserts a space before it: e.g. ``d hula`` = dʰula.
    value = re.sub(r"([bcdfgjkptɖɡʃʒʈ])\s+h", r"\1ʰ", value)
    value = " ".join(value.split())
    return unicodedata.normalize("NFC", value)


def extract(pdf_path: Path = PDF) -> list[tuple[int, str, str, str, int, int]]:
    """Return concept, gloss, lect, IPA, printed page, similarity group."""
    reader = PdfReader(pdf_path)
    records: list[tuple[int, str, str, str, int, int]] = []
    concepts: dict[int, str] = {}
    current_concept: int | None = None
    current_gloss = ""
    unmatched: list[tuple[int, str]] = []

    for page_index in range(40, 64):
        text = reader.pages[page_index].extract_text() or ""
        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            if "[" not in line:
                heading = HEADING.fullmatch(line)
                if heading and 1 <= int(heading.group(1)) <= 307:
                    current_concept = int(heading.group(1))
                    current_gloss = heading.group(2).strip()
                    concepts[current_concept] = current_gloss
                continue
            entry = ENTRY.fullmatch(line)
            if not entry or current_concept is None:
                unmatched.append((page_index + 1, line))
                continue

            group_text, raw_form, codes = entry.groups()
            form = _decode(raw_form)
            if form.casefold() == "no entry":
                continue
            # Concept 207's first entry visually belongs to group 1, but the
            # digit itself is absent from the PDF text layer.
            if group_text is None:
                if current_concept == 207 and form == "bana":
                    group = 1
                else:
                    unmatched.append((page_index + 1, line))
                    continue
            else:
                group = int(group_text)

            for code in codes.replace(" ", ""):
                if code == "0":
                    continue
                if code not in LECTS:
                    raise ValueError(f"Unexpected site code {code!r} on PDF page {page_index + 1}")
                records.append(
                    (
                        current_concept,
                        current_gloss,
                        LECTS[code],
                        form,
                        page_index,
                        group,
                    )
                )

    if unmatched:
        raise ValueError(f"Unparsed wordlist lines: {unmatched}")
    expected = set(range(1, 308))
    if set(concepts) != expected:
        raise ValueError(
            f"Unexpected headings: missing {sorted(expected - set(concepts))}; "
            f"unexpected {sorted(set(concepts) - expected)}"
        )
    if any(placeholder in form for *_, form, _page, _group in records for placeholder in GLYPHS):
        raise ValueError("Undecoded legacy-font placeholder remains")
    return records


def main() -> None:
    extracted = extract()
    occurrence: Counter[tuple[int, str]] = Counter()
    rows = []
    for concept, gloss, lect, form, printed_page, group in extracted:
        occurrence[(concept, lect)] += 1
        rows.append(
            [
                lect,
                "",
                form,
                gloss,
                "",
                form,
                f"lexical-similarity-group:{group}",
                f"{SOURCE}[p. {printed_page}]",
                "",
                "",
                f"hajong:{concept}:{lect}:{occurrence[(concept, lect)]}",
            ]
        )
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)
    print(f"Wrote {len(rows)} forms from {len(LECTS)} Hajong lects to {OUTPUT}")


if __name__ == "__main__":
    main()
