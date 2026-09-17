"""Extract Appendix A of Canvin, Joseph & Manoj (2025) and build Jambu rows.

This is the Markodi (formerly labelled "Mavilan Tulu") wordlist. The PDF (kept in
``tmp/pdfs`` as a working source) fixes the forms. Every row carries an immutable
``Entry_Key`` (``canvin2025:<site>:<concept>``) naming its wordlist cell, so a form
elicited for two concepts stays two attestations.

Etymologies are *not* written here. They live in the standard per-source sidecar
``data/other/forms/etymologies/20260723-markodi.csv`` (see ``etymology_assignments.py``),
keyed by persistent form ID, and are applied by ``assign_form_ids.py`` during the build.
``markodi_etyma.csv`` beside this script keeps the comparison forms (Tulu, Malayalam,
Kodava) and the curator's notes as a research record only.

Re-run this script (``make markodi``) only when the PDF extraction changes; the CLDF
build no longer regenerates the forms.
"""

from __future__ import annotations

import csv
from pathlib import Path

from pypdf import PdfReader


HERE = Path(__file__).resolve().parent
PDF = HERE.parents[3] / "tmp" / "pdfs" / "JLSR2025-005.pdf"
OUTPUT = HERE.parent / "20260723-markodi.csv"
SOURCE_KEY = "canvin2025"

LANGUAGES = {
    "MTP": "markodi_pannithadam",
    "MTV": "markodi_vannarkadav",
    "MTE": "markodi_ennappara",
}

NA_FORMS = {"nill", "nil", "na", "-"}


def extract_wordlist(pdf_path: Path = PDF) -> list[tuple[str, dict[str, str]]]:
    """Return the 208 concepts on PDF pages 28--38 and their six comparison forms."""
    import re

    reader = PdfReader(pdf_path)
    items: list[tuple[str, dict[str, str]]] = []
    current: tuple[str, dict[str, str]] | None = None
    for page_number in range(28, 39):
        text = reader.pages[page_number - 1].extract_text() or ""
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.isdigit() or line == "A.2 Wordlist data":
                continue
            match = re.match(r"^(MTP|MTV|MTE|MAL|TUL|l?KOD)\s*:?\s*(.*)$", line)
            if match:
                if current is None:
                    raise ValueError(f"Form before gloss on PDF page {page_number}: {line}")
                code = match.group(1).removeprefix("l")  # one PDF typo: lKOD
                current[1][code] = match.group(2).strip()
            else:
                current = (line, {})
                items.append(current)

    if len(items) != 208:
        raise ValueError(f"Expected 208 concepts, extracted {len(items)}")
    for gloss, forms in items:
        missing = {"MTP", "MTV", "MTE", "MAL", "TUL", "KOD"} - forms.keys()
        if missing:
            raise ValueError(f"{gloss}: missing {sorted(missing)}")
    return items


def entry_key(site: str, gloss: str) -> str:
    """Immutable per-cell record key: the site code and the concept label as printed."""
    return f"{SOURCE_KEY}:{site}:{gloss}"


def main() -> None:
    items = extract_wordlist()
    output_rows: list[list[str]] = []
    for gloss, forms in items:
        for code, language_id in LANGUAGES.items():
            form = forms[code]
            if form.lower() in NA_FORMS:
                continue
            # 15-column source layout; Parameter_ID stays blank because etymologies come from
            # the sidecar. Column 11 is the Entry_Key.
            output_rows.append(
                [language_id, "", form, gloss, "", form, "", SOURCE_KEY, "", "",
                 entry_key(code, gloss), "", "", "", ""]
            )

    keys = [row[10] for row in output_rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Entry_Key collision: a concept label repeats within a site")
    output_rows.sort(key=lambda row: (row[3], row[0], row[2]))
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(output_rows)
    print(f"Wrote {len(output_rows)} forms to {OUTPUT}")


if __name__ == "__main__":
    main()
