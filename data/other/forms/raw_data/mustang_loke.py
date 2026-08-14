"""Extract only the five Loke target wordlists from Khadgi et al. (2021 [2003])."""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader


HERE = Path(__file__).resolve().parent
PDF = HERE.parents[3] / "tmp" / "pdfs" / "mustang" / "official.pdf"
OUTPUT = HERE.parent / "20260813-mustang-loke.csv"
SOURCE = "khadgi-marcuson-marcuson2021mustang"
LECTS = {
    "L": "loke_lo_manthang",
    "G": "loke_ghiling",
    "C": "loke_chhosher",
    "J": "loke_jharkot",
    "B": "loke_kagbeni",
}
ALL_CODES = set(LECTS) | {"M", "K", "D", "d", "n", "t"}


def extract(pdf_path: Path = PDF):
    reader = PdfReader(pdf_path)
    records = []
    headers = []
    variants = Counter()
    current = None

    # PDF pages 134--169 (printed pages 126--161) contain Appendix C.6.
    for page_index in range(133, 169):
        for raw_line in (reader.pages[page_index].extract_text() or "").splitlines():
            line = " ".join(raw_line.split())
            if not line:
                continue
            header = re.fullmatch(r"(\d{1,3})\.\s+(.+)", line)
            if header:
                current = (int(header.group(1)), header.group(2).strip())
                headers.append(current)
                continue
            if current is None:
                continue

            tokens = line.split()
            codes = []
            while tokens and tokens[-1] in ALL_CODES:
                codes.append(tokens.pop())
            codes.reverse()
            if not codes or not any(code in LECTS for code in codes):
                continue

            group = None
            if tokens and re.fullmatch(r"\d+", tokens[0]):
                group = int(tokens.pop(0))
            form = " ".join(tokens).strip()
            # Two vowel breathiness marks have spurious word-space glyphs in
            # the PDF's ToUnicode map (items 31 L and 56 G).  The rendered
            # source shows the marks attached to the preceding vowel.
            form = re.sub(r"\s+(?=[\u0300-\u036f])", "", form)
            if not form or form == "NO ENTRY" or group == 0:
                continue

            note_parts = []
            if form.startswith("(") and form.endswith(")"):
                form = form[1:-1].strip()
                note_parts.append("Alternative word in the source")
            if group is not None:
                note_parts.append(f"Source lexical-similarity group {group}")
            note = "; ".join(note_parts)

            for code in codes:
                if code not in LECTS:
                    continue
                number, gloss = current
                variants[(number, code)] += 1
                records.append(
                    (
                        number,
                        gloss,
                        code,
                        form,
                        page_index - 7,
                        variants[(number, code)],
                        note,
                    )
                )

    if [number for number, _ in headers] != list(range(1, 337)):
        raise ValueError(f"Unexpected concept sequence: {headers}")
    if any("�" in row[3] for row in records):
        raise ValueError(f"Replacement character in target forms: {[r for r in records if '�' in r[3]]}")
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
                f"mustang:{number}:{lect}:{variant}",
            ]
        )
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)
    print(f"Wrote {len(rows)} forms from {len(LECTS)} Loke target lects to {OUTPUT}")


if __name__ == "__main__":
    main()
