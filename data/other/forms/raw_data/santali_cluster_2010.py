"""Extract the sixteen Bangladesh wordlists from Kim et al. (2010).

Appendix B.3 (physical/printed PDF pages 82--111) contains 307 prompts
for seven Santali, three Mahali, three Mundari, two Koda, and one Kol
village in Bangladesh.  The two uppercase Indian Mundari comparison lists
and site ``0`` (Standard Bangla) are deliberately excluded.

The born-digital PDF embeds ``SAG-IPASILManuscript``.  Its ToUnicode map
exposes non-ASCII glyphs as U+F0xx characters, so ``SAG_IPA`` below applies
SIL's official SAGIPA2Uni mapping before any rows are written.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from collections import Counter
from pathlib import Path

from pypdf import PdfReader


HERE = Path(__file__).resolve().parent
PDF = HERE.parents[4] / "tmp" / "pdfs" / "santali-cluster-2010" / "source.pdf"
OUTPUT = HERE.parent / "20260813-santali-cluster.csv"
SOURCE = "kim-kim-ahmad-sangma2010santali-cluster"

LECTS = {
    "a": "santali_rajarampur",
    "b": "santali_rautnagar",
    "c": "mundari_nijpara",
    "d": "santali_paharpur",
    "e": "mahali_abirpara",
    "g": "mahali_matindor",
    "h": "santali_patichora",
    "i": "santali_jabri",
    "j": "mundari_begunbari",
    "k": "mahali_pachondor",
    "l": "koda_kundang",
    "m": "kol_babudaing",
    "n": "koda_krishnupur",
    "o": "santali_bodobelghoria",
    "p": "mundari_karimpur",
    "q": "santali_rashidpur",
}

# Relevant byte assignments from SIL's SAGIPA2Uni.map.  In this symbol-font
# PDF, legacy byte XX is represented in extracted text as U+F0XX.
SAG_IPA = {
    0x21: "ː",
    0x40: "̚",
    0x41: "ɑ",
    0x45: "ɛ",
    0x48: "ʔ",
    0x49: "ɪ",
    0x52: "ɾ",
    0x53: "ʃ",
    0x55: "ʊ",
    0x56: "ʋ",
    0x5A: "ʒ",
    0x67: "g",
    0x7D: "ɽ",
    0x81: "ɨ",
    0x86: "ɐ",
    0x89: "ʌ",
    0x8A: "ɔ",
    0x91: "ɹ",
    0x95: "ɳ",
    0x98: "ʈ",
    0x99: "ɖ",
    0x9B: "ɭ",
    0x9D: "ɲ",
    0xA4: "ŋ",
    0xD0: "̃",
    0xE6: "̪",
    0xE9: "̥",
    0xEB: "̯",
}

HEADING = re.compile(r"^\s*(\d{1,3})\s{2,}(.+?)\s*$")
ENTRY = re.compile(r"^\s*(?:(\d+)\s+)?(.+?)\s+\[([0EMabcdeghijklmnopq ]+)\]\s*$")
CODE_ONLY = re.compile(r"^\s*\[[0EMabcdeghijklmnopq ]+\]\s*$")


def _decode(value: str) -> str:
    for byte, ipa in SAG_IPA.items():
        value = value.replace(chr(0xF000 + byte), ipa)
    value = " ".join(value.split())
    value = unicodedata.normalize("NFC", value)
    remaining = [char for char in value if 0xE000 <= ord(char) <= 0xF8FF]
    if remaining:
        points = ", ".join(f"U+{ord(char):04X}" for char in sorted(set(remaining)))
        raise ValueError(f"Unmapped SAG-IPA character(s): {points}")
    return value


def _joined_lines(text: str) -> list[str]:
    """Rejoin the eight entries whose bracketed code list wraps alone."""
    lines: list[str] = []
    for line in text.splitlines():
        if CODE_ONLY.fullmatch(line) and lines:
            lines[-1] += " " + line.strip()
        else:
            lines.append(line)
    return lines


def extract(pdf_path: Path = PDF) -> list[tuple[int, str, str, str, int, int]]:
    """Return concept, gloss, lect, IPA, printed page, similarity group."""
    reader = PdfReader(pdf_path)
    records: list[tuple[int, str, str, str, int, int]] = []
    concepts: dict[int, str] = {}
    current_concept: int | None = None
    current_gloss = ""
    current_group: int | None = None
    unmatched: list[tuple[int, str]] = []

    for page_index in range(81, 111):
        for raw_line in _joined_lines(reader.pages[page_index].extract_text() or ""):
            line = raw_line.rstrip()
            if "[" not in line:
                heading = HEADING.fullmatch(line)
                if heading and 1 <= int(heading.group(1)) <= 307:
                    current_concept = int(heading.group(1))
                    current_gloss = heading.group(2).strip()
                    current_group = None
                    concepts[current_concept] = current_gloss
                continue

            entry = ENTRY.fullmatch(line)
            if not entry or current_concept is None:
                unmatched.append((page_index + 1, line))
                continue

            group_text, raw_form, codes = entry.groups()
            if group_text is not None:
                current_group = int(group_text)
            # Concept 142's second group-1 variant wraps without repeating the
            # group digit; its position under the preceding group-1 line is
            # unambiguous in both the text layer and rendered page.
            if current_group is None:
                unmatched.append((page_index + 1, line))
                continue

            form = _decode(raw_form)
            for code in codes.replace(" ", ""):
                if code in {"0", "E", "M"}:
                    continue
                if code not in LECTS:
                    raise ValueError(f"Unexpected site code {code!r} on PDF page {page_index + 1}")
                records.append(
                    (
                        current_concept,
                        current_gloss,
                        LECTS[code],
                        form,
                        page_index + 1,
                        current_group,
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
                f"santali-cluster:{concept}:{lect}:{occurrence[(concept, lect)]}",
            ]
        )

    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)
    print(f"Wrote {len(rows)} forms from {len(LECTS)} Bangladesh lects to {OUTPUT}")


if __name__ == "__main__":
    main()
