"""
make_refs.py — build cldf/references.csv from sources.bib: a short citation key + a formatted
(markdown) source string + inclusion status, for each bibliography entry. Ports the reference
logic from neojambu/scripts/make_database.py so the jambu-static build can read references from a
CSV instead of re-running pybtex.

Run:  uv run --with pybtex python make_refs.py
"""

import csv
import sys

import pybtex
import pybtex.database


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
    used = set()
    rows = []
    for key in sources.entries:  # insertion (file) order → stable dedup suffixes
        entry = sources.entries[key]
        try:
            formatted = engine.format_from_string(
                entry.to_string("bibtex"), "plain", output_backend="markdown"
            )
            formatted = formatted[3:].strip()
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
        rows.append([key, short, formatted, entry.fields.get("included", "No")])

    with open("cldf/references.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ID", "Short", "Source", "Progress"])
        w.writerows(rows)
    print(f"wrote cldf/references.csv ({len(rows)} references)", file=sys.stderr)


if __name__ == "__main__":
    main()
