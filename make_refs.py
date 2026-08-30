"""
make_refs.py — build cldf/references.csv from sources.bib: a short citation key + a formatted
(markdown) source string + inclusion status, for each bibliography entry. Ports the reference
logic from neojambu/scripts/make_database.py so the jambu-static build can read references from a
CSV instead of re-running pybtex.

Run:  uv run --with pybtex python make_refs.py
"""

import csv
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pybtex
import pybtex.database


FORM_INPUTS = [
    "data/cdial/cdial.csv",
    "data/dedr/dedr_new.csv",
    "data/dedr/pdr.csv",
    "data/munda/forms.csv",
    "data/other/forms/*.csv",
]

DEFAULT_JAMBU_EDITOR = "Aryaman Arora"
JAMBU_EDITOR_OVERRIDES = {
    "fritz": "Adam Farris",
    "backstrom1992": "Aryaman Arora; OpenAI Codex",
    "lehr": "Aryaman Arora; OpenAI Codex",
    "schmidt": "Aryaman Arora; OpenAI Codex",
    "canvin2025": "Aryaman Arora; OpenAI Codex; Claude Opus 4.8",
    "zoller2005": "OpenAI Codex",
    "bhaskararao-toda2025": "OpenAI Codex",
}
OCR_OVERRIDES = {
    "berger-auto",
    "dbia",
}

# Reference-level summary of who supplied the etymological analysis represented in Jambu.
# ``source-mapped`` means that the publication supplies the etymon/analysis and Jambu only
# resolves it to a canonical CDIAL/DEDR entry.  Keep this deliberately conservative: an absent
# value is displayed as "Not recorded" rather than silently crediting either the author or us.
ETYMOLOGY_PROVENANCE_OVERRIDES = {
    "CDIAL": "source",
    "dedr": "source",
    "dbia": "source-mapped",
    "bashir2023": "source",
    "berger": "source",
    "berger-auto": "source",
    "liljegren": "source",
    "trail-cooper1999": "source",
    "shackle": "source",
    "shackle-auto": "source",
    "toulmin": "source",
    "schmidt": "source",
    "southworth2005m": "source",
    "southworth2006proto": "source",
    "krishnamurti": "source",
    "pfeiffer2018": "source",
    "rau": "source",
    "kobayashi2022": "source",
    "weinreich2008": "source",
    "emeneau1997brahui": "source",
    "burrow-emeneau1972den1": "source",
    "burrow-emeneau1972den2": "source",
    "patyal2": "source",
    "patyal3": "source",
    "patyal4": "source",
    "patyal5": "source",
    "nured": "source-mapped",
    "gandhari": "source-mapped",
    "kullui-org": "source-mapped",
    "andersen1990": "source-mapped",
    "paranavitana": "source-mapped",
    "zoller2005": "source-mapped",
    "tulpule1999": "source-mapped",
    "backstrom1992": "jambu",
    "boehm": "jambu",
    "bundeli": "jambu",
    "chattisgarhi": "jambu",
    "kannauji": "jambu",
    "mewari": "jambu",
    "hadothi": "jambu",
    "dhundari": "jambu",
    "marwari": "jambu",
    "mewati": "jambu",
    "bagri": "jambu",
    "gill": "jambu",
    "kholosi": "jambu",
    "lehr": "jambu",
    "maimani": "jambu",
    "srinivasa": "jambu",
    "thari": "jambu",
    "webster": "jambu",
    "zubair": "jambu",
    "strand": "mixed",
}

ETYMOLOGY_PROVENANCE_VALUES = {"source", "source-mapped", "jambu", "mixed", "none", ""}

CDIAL_REFERENCE_CATALOG = Path("data/cdial/reference_catalog.json")


def cdial_reference_catalog():
    """Descriptions transcribed from the main and addenda CDIAL prefaces."""
    if not CDIAL_REFERENCE_CATALOG.exists():
        return {}
    with CDIAL_REFERENCE_CATALOG.open(encoding="utf-8") as f:
        return json.load(f)


def source_ids(value):
    """Yield bare bibliography IDs from a CLDF Source cell (dropping optional [pages])."""
    for token in (value or "").split(";"):
        key = token.split("[", 1)[0].strip()
        if key:
            yield key


def collect_provenance():
    """Map each citation key to the source-data files through which it entered Jambu."""
    paths_by_ref = defaultdict(set)
    for pattern in FORM_INPUTS:
        for filename in glob.glob(pattern):
            with open(filename, encoding="utf-8") as f:
                for row in csv.reader(f):
                    if len(row) < 8:
                        continue
                    for key in source_ids(row[7]):
                        paths_by_ref[key].add(Path(filename).as_posix())
    return paths_by_ref


def used_references(path="cldf/forms.csv"):
    used = set()
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            used.update(source_ids(row.get("Source", "")))
    return used


def create_short_ref(entry):
    """Short key like 'T1962' from first-author initial + year (deduped downstream)."""
    year = entry.fields.get("year")
    authors = entry.persons.get("author", [])
    if year == "n.d.":
        year = "?"
    if authors and year:
        fa = authors[0]
        first_letter = fa.last_names[0][0].upper() if fa.last_names else fa.first()[0].upper()
        year = year.replace("--", "—")
        return f"{first_letter}{year}"
    return "?"


def main():
    sources = pybtex.database.parse_file("cldf/sources.bib")
    engine = pybtex.PybtexEngine()
    provenance_by_ref = collect_provenance()
    used_refs = used_references()
    cdial_catalog = cdial_reference_catalog()
    used = set()
    rows = []
    for key in sources.entries:  # insertion (file) order → stable dedup suffixes
        entry = sources.entries[key]
        try:
            formatted = engine.format_from_string(
                entry.to_string("bibtex"), "plain", output_backend="markdown"
            )
            formatted = formatted[3:].strip()
            # Pybtex preserves a URL tilde as a LaTeX command, including a separator space that
            # breaks the generated Markdown destination. Emit the actual URL character instead.
            formatted = re.sub(r"\\+textasciitilde\s*", "~", formatted)
            # Pybtex's "plain" style hardcodes the degree of a thesis and ignores
            # BibTeX's ``type`` field, so it calls a bachelor's thesis a master's
            # one. Honour the field the entry actually states.
            degree = entry.fields.get("type", "").strip()
            if degree and entry.type in {"mastersthesis", "phdthesis"}:
                formatted = re.sub(
                    r"(Master's|PhD) thesis", degree.replace("\\", ""), formatted, count=1
                )
        except Exception as e:  # noqa: BLE001
            print(f"format error for {key}: {e}", file=sys.stderr)
            formatted = ""
        short = create_short_ref(entry)
        while short in used and short != "?":
            if short[-1].isdigit() or short[-1] == "?":
                short += "a"
            else:
                short = short[:-1] + chr(ord(short[-1]) + 1)
        used.add(short)
        provenance = entry.fields.get("provenance", "").strip() or "; ".join(
            sorted(provenance_by_ref.get(key, ()))
        )
        if key in used_refs and not provenance:
            provenance = "cldf/forms.csv (upstream import path unavailable)"
        editor = entry.fields.get("jambu_editor", "").strip() or JAMBU_EDITOR_OVERRIDES.get(
            key, DEFAULT_JAMBU_EDITOR
        )
        ocr = (
            entry.fields.get("ocr", "").strip().lower() in {"yes", "true", "1"}
            or key in OCR_OVERRIDES
        )
        etymology_provenance = entry.fields.get("etymology_provenance", "").strip().lower()
        if not etymology_provenance:
            etymology_provenance = ETYMOLOGY_PROVENANCE_OVERRIDES.get(key, "")
        if etymology_provenance not in ETYMOLOGY_PROVENANCE_VALUES:
            raise ValueError(
                f"Invalid etymology_provenance for {key}: {etymology_provenance!r}"
            )
        rows.append(
            [key, short, formatted, entry.fields.get("included", "No"), provenance, editor,
             "Yes" if ocr else "No", etymology_provenance]
        )

    # A cited key without a BibTeX record must still be a first-class, traceable reference rather
    # than an id-only row silently synthesised by the database transform.
    for key in sorted(used_refs - set(sources.entries)):
        provenance = "; ".join(sorted(provenance_by_ref.get(key, ())))
        if not provenance:
            provenance = "cldf/forms.csv (upstream import path unavailable)"
        editor = JAMBU_EDITOR_OVERRIDES.get(key, DEFAULT_JAMBU_EDITOR)
        description = cdial_catalog.get(
            key,
            f"Reference abbreviation `{key}` cited in the source data; full citation not yet catalogued.",
        )
        rows.append([
            key, key, description, "No", provenance, editor,
            "Yes" if key in OCR_OVERRIDES else "No", "",
        ])

    with open("cldf/references.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "ID", "Short", "Source", "Progress", "Provenance", "Editor", "OCR",
            "Etymology_Provenance",
        ])
        w.writerows(rows)
    print(f"wrote cldf/references.csv ({len(rows)} references)", file=sys.stderr)


if __name__ == "__main__":
    main()
