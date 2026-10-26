#!/usr/bin/env python3
"""Lint the display ``Form`` of attested rows in ``cldf/forms.csv`` against the house transcription.

``profile_policy.py`` checks what each sound profile's rules *output*; it cannot see letters a
profile passes through unconverted, parser leaks (citation text, markup) or rows whose conversion
failed.  This lint reads the built forms instead.  Each check is a pattern that should never
appear in a house Form; ``ALLOW`` records, per check and source, the reviewed exceptions and
open decisions, so a new source or a regression fails while known cases stay visible.

    uv run python form_lint.py            # summary per check and source, exit 1 on violations
    uv run python form_lint.py -v         # with examples
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata as ud
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FORMS = ROOT / "cldf/forms.csv"

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

# Patterns matched against the NFD Form.
CHECKS = {
    # aspiration is the modifier letter ʰ, never a plain h after the stop
    "aspirate-digraph": re.compile(r"[kgcjtdṭḍpbʦʣ][̀-ͯ]*h"),
    # Greek ε for the IPA ɛ every source writes
    "greek-epsilon": re.compile("ε"),
    # footnote markers, homonym indices, citation page numbers
    "digit": re.compile(r"[0-9⁰¹²³⁴-⁹]"),
    # HTML or source markup leaking into the display form
    "markup": re.compile(r"[<>]|&lt;|&gt;"),
    # letters with no single house value (ċ) or with a house spelling (ʁ ġ → ɣ)
    "non-house-letter": re.compile(r"ċ|ċ|ʁ|ġ|ġ"),
}

# Reviewed exceptions and open decisions, per check: source key -> reason.
ALLOW: dict[str, dict[str, str]] = {
    "aspirate-digraph": {
        "pinnow1959versuch": "Munda C+h may be a cluster; per-source decision pending",
        "munda1968proto": "Munda C+h may be a cluster; per-source decision pending",
        "DSKH": "Kharia C+h may be a cluster; per-source decision pending",
        "kim-kim-sangma-ahmad2011tripura": "source has no conversion profile",
        "CDIAL": "Turner's c̄h notation, unresolved",
        "dbia": "Sanskrit loans in DBIA; profile not yet reviewed",
        "sheth1923": "Prakrit ṇh/nh clusters",
    },
    "digit": {
        "hockings-pilotraichoor1992": "Badaga numerals in place names; source notation pending",
        "merriam2026dravidiandb": "root indices on reconstructions (*maɣ1-)",
        "dbia": "OCR residue (0'ṇa, vat1)",
        "trail-cooper1999": "OCR residue (wr45yak)",
        "bhattacharya1968bonda": "OCR residue (ɔ:6ɔnsɔ?)",
        "zoller2023": "Zoller's own footnote marker (əvak²²)",
        "dedr": "DEDR homonym suffixes (ki-1)",
    },
    "markup": {
        "dedr": "DEDR parser junk: Latin plant names and stray </i> in seven forms",
    },
    "non-house-letter": {
        "nured": "ċ value undocumented; research pending",
        "hockings-pilotraichoor1992": "Badaga ġ is not clearly a fricative",
        "tagin-puroik": "ʁ may be a genuine uvular in these Tibeto-Burman lists",
        "ghatage-western": "OCR-garbled ċ in two cells",
    },
}


def source_key(source: str) -> str:
    return source.split(";", 1)[0].split("[", 1)[0].strip()


def scan(path: Path = FORMS, limit: int = 5):
    """``{check: {source: (count, examples)}}`` over attested rows (etyma excluded)."""
    counts: dict[str, Counter] = defaultdict(Counter)
    examples: dict[tuple[str, str], list[str]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["Status"] == "entry":
                continue
            form = ud.normalize("NFD", row["Form"])
            src = source_key(row["Source"])
            for name, pattern in CHECKS.items():
                if pattern.search(form):
                    counts[name][src] += 1
                    if len(examples[name, src]) < limit:
                        examples[name, src].append(row["Form"])
    return counts, examples


def violations(counts) -> dict[str, dict[str, int]]:
    return {
        name: {src: n for src, n in by_source.items() if src not in ALLOW.get(name, {})}
        for name, by_source in counts.items()
        if any(src not in ALLOW.get(name, {}) for src in by_source)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true", help="print example forms")
    args = parser.parse_args()
    counts, examples = scan()
    bad = violations(counts)
    for name in CHECKS:
        by_source = counts.get(name, Counter())
        print(f"== {name}: {sum(by_source.values())} forms")
        for src, n in by_source.most_common():
            mark = "allowed" if src in ALLOW.get(name, {}) else "VIOLATION"
            line = f"   {n:6} {src:40} {mark}"
            if args.verbose:
                line += f"  {examples[name, src]}"
            print(line)
    total = sum(sum(v.values()) for v in bad.values())
    print(f"{total} forms violate the house transcription outside ALLOW")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
