"""Extract the born-digital wordlist appendix in Abraham & Sako (2021).

The official SIL PDF is JLSR2021-063.  Appendix B.3 (PDF pages 32--64,
printed pages 25--57) contains 307 concepts for sixteen source lects.  The
appendix is Unicode text, not OCR.  Similarity-group numbers are intentionally
discarded; every printed form is retained, including alternate responses.

Download the PDF to ``tmp/pdfs/JLSR2021-063.pdf`` before running this script.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader


HERE = Path(__file__).resolve().parent
PDF = HERE.parents[3] / "tmp" / "pdfs" / "JLSR2021-063.pdf"
OUTPUT = HERE.parent / "20260813-tagin-puroik.csv"
SOURCE = "abraham-sako2021"

LECTS = {
    "a": "puroik_phereng",
    "b": "puroik_gari",
    "c": "puroik_chug",
    "e": "puroik_paji",
    "v": "bugun_singchung",
    "w": "bugun_wangho",
    "x": "bugun_bichom",
    "y": "bugun_kaspi",
    "z": "bugun_namphri",
    "j": "tagin_sippi",
    "k": "tagin_nacho",
    "l": "tagin_baki",
    "d": "tagin_taliha",
    "n": "tagin_maskia",
    "f": "tagin_takseng",
    "g": "nyishi_chimpu",
}

NO_ENTRY = {"no entry", "ɴo entry"}
ENTRY_RE = re.compile(r"([0-9a-z])\s+(.+?)\s+\[([a-z]+)\]")
HEAD_RE = re.compile(r"(\d{1,3})\s+(.+)")


def extract(pdf_path: Path = PDF) -> tuple[dict[int, str], list[tuple[int, str, str, int]]]:
    """Return concept glosses and (concept, lect, form, printed page) rows."""
    reader = PdfReader(pdf_path)
    concepts: dict[int, str] = {}
    rows: list[tuple[int, str, str, int]] = []
    current: int | None = None
    expected = 1

    # Zero-based 31:64 = PDF pages 32--64 = printed pages 25--57.
    for page_index in range(31, 64):
        lines = [(line or "").strip() for line in (reader.pages[page_index].extract_text() or "").splitlines()]
        line_index = 0
        while line_index < len(lines):
            line = lines[line_index]
            line_index += 1
            if not line or line == "B.3 Wordlist transcription":
                continue
            # Printed running page number.
            if line.isdigit() and 25 <= int(line) <= 57:
                continue

            entry = ENTRY_RE.fullmatch(line)
            if entry and current is not None:
                form, codes = entry.group(2), entry.group(3)
                if form.casefold() not in NO_ENTRY:
                    for code in codes:
                        rows.append((current, code, form, page_index - 6))
                continue

            head = HEAD_RE.fullmatch(line)
            if head and int(head.group(1)) == expected:
                current = expected
                concepts[current] = head.group(2)
                expected += 1
                continue

            # Two forms on printed p. 37 are split across PDF text objects.
            if (
                re.fullmatch(r"[0-9a-z]\s+.+", line)
                and line_index < len(lines)
                and re.search(r"\[[a-z]+\]$", lines[line_index])
            ):
                joined = line + lines[line_index]
                entry = ENTRY_RE.fullmatch(joined)
                if entry and current is not None:
                    line_index += 1
                    form, codes = entry.group(2), entry.group(3)
                    if form.casefold() not in NO_ENTRY:
                        for code in codes:
                            rows.append((current, code, form, page_index - 6))
                    continue

            raise ValueError(f"Unparsed line on PDF page {page_index + 1}: {line!r}")

    if set(concepts) != set(range(1, 308)):
        raise ValueError(f"Expected concepts 1--307, got {sorted(concepts)}")
    if len(rows) != 4939:
        raise ValueError(f"Expected 4,939 printed forms, got {len(rows)}")
    unknown = {code for _, code, _, _ in rows} - LECTS.keys()
    if unknown:
        raise ValueError(f"Unknown lect codes: {sorted(unknown)}")
    return concepts, rows


def main() -> None:
    concepts, extracted = extract()
    occurrence: Counter[tuple[int, str]] = Counter()
    output_rows: list[list[str]] = []
    for concept, code, form, printed_page in extracted:
        occurrence[(concept, code)] += 1
        entry_key = f"tagin-puroik:{concept}:{code}:{occurrence[(concept, code)]}"
        output_rows.append(
            [
                LECTS[code],
                "",
                form,
                concepts[concept],
                "",
                form,
                "",
                f"{SOURCE}[p. {printed_page}]",
                "",
                "",
                entry_key,
            ]
        )

    output_rows.sort(key=lambda row: (row[3], row[0], row[2], row[10]))
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(output_rows)
    print(f"Wrote {len(output_rows):,} forms from {len(LECTS)} lects to {OUTPUT}")


if __name__ == "__main__":
    main()
