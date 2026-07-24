"""Extract Appendix A of Canvin, Joseph & Manoj (2025) and build Jambu rows.

This is the Markodi (formerly labelled "Mavilan Tulu") wordlist. The PDF (kept in
``tmp/pdfs`` as a working source) fixes the forms; the etymology decisions are
hand-curated in ``markodi_etyma.csv`` — one row per attested site-form, with an
``Etymon`` column you edit:

    * a DEDR / Proto-Dravidian id (e.g. ``d1159``)          -> inherited reflex
    * a CDIAL id prefixed with ``~`` (e.g. ``~4661``)       -> Indo-Aryan borrowing
    * blank                                                 -> kept as a lone
                                                              (unetymologised) node

Re-run this script (then ``make stage``) to regenerate the Jambu forms from the PDF
plus that CSV. The CSV is the source of truth for etyma; this script never overwrites it.
"""

from __future__ import annotations

import csv
from pathlib import Path

from pypdf import PdfReader


HERE = Path(__file__).resolve().parent
PDF = HERE.parents[3] / "tmp" / "pdfs" / "JLSR2025-005.pdf"
OUTPUT = HERE.parent / "20260723-markodi.csv"
ETYMA = HERE / "markodi_etyma.csv"

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


def load_etyma() -> dict[tuple[str, str], str]:
    """(Gloss, Site) -> etymon id, from the hand-curated CSV. Blank / absent = unresolved."""
    if not ETYMA.exists():
        raise FileNotFoundError(f"missing etymon table {ETYMA}; see the module docstring")
    etyma: dict[tuple[str, str], str] = {}
    with ETYMA.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            etymon = (row.get("Etymon") or "").strip()
            if etymon:
                etyma[(row["Gloss"], row["Site"])] = etymon
    return etyma


def main() -> None:
    items = extract_wordlist()
    etyma = load_etyma()
    output_rows: list[list[str]] = []
    resolved = borrowed = 0
    for gloss, forms in items:
        for code, language_id in LANGUAGES.items():
            form = forms[code]
            if form.lower() in NA_FORMS:
                continue
            etymon = etyma.get((gloss, code), "")  # blank Param_ID → a lone node in the DB build
            if etymon:
                resolved += 1
                borrowed += etymon.startswith("~")
            output_rows.append(
                [language_id, etymon, form, gloss, "", form, "", "canvin2025"]
            )

    output_rows.sort(key=lambda row: (row[3], row[0], row[2]))
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(output_rows)

    unresolved_forms = len(output_rows) - resolved
    print(f"Wrote {len(output_rows)} forms to {OUTPUT}")
    print(
        f"  {resolved} linked to an etymon ({borrowed} as borrowings); "
        f"{unresolved_forms} left blank → lone (unetymologised) nodes"
    )


if __name__ == "__main__":
    main()
