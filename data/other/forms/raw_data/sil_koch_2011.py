#!/usr/bin/env python3
"""Install the wordlists of SIL Electronic Survey Report 2011-033.

Alexander Kondakov, *Koch survey wordlists and sociolinguistic questionnaire* (SIL International,
2011) -- the data appendix to "Koch Dialects of Meghalaya and Assam: A Sociolinguistic Survey".
The wordlists were elicited in 2006--2007 in Meghalaya and Assam and cover eleven sites: standard
Garo, the Rongdani dialect of Rabha, four Koch dialects (Margan, Harigaya, Wanang, Tintekiya) and
the Koch-Rabha variety.

The report is not redistributed: this importer requires the publisher PDF at
``tmp/pdfs/sil-surveys/silesr2011_033.pdf`` and verifies its SHA-256 before reading it.

Extraction notes
----------------
Unlike the Appendix B3 layout in ``sil_survey_wordlists.py``, this appendix is a **table** whose
site label is a cell vertically centred over its form group, so a form belonging to one label is
routinely printed on the line *above* it::

    kɑń ,           <- first form of B. RR
    B. RR           <- the label, centred between its two forms
    kɑnɡɑnd͡ʒi      <- second form of B. RR

No line-based reading can recover that, so the parser works from word coordinates: within each of
the two page columns it collects gloss headers (at the column's left margin), label anchors
(``A.``--``K.`` at the label x-position) and form fragments, then assigns every form to the label
whose vertical centre is nearest.  Assignment happens per gloss block, not per line.

The site key on the first wordlist page spans the full page width and therefore crosses the column
gutter; it is parsed separately for the site registry and skipped as data by starting that page
below the last key line.

Transcription is Unicode IPA in the publisher's text layer -- no OCR and no legacy font decoding.
As with the other SIL survey ingests the source IPA is installed unchanged in ``Form`` and
``Phonemic``; ``conversion/sil-survey.txt`` maps it to house transcription at build time.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ROOT = HERE.parents[4]
PDF = ROOT / "tmp/pdfs/sil-surveys/silesr2011_033.pdf"
PDF_SHA256 = "c5999a5df5cdffae048a5b7f081f4280002a10cb3df58d4808dbfbdc2c6ab749"

INSTALLED = HERE.parent / "20260826-sil-koch.csv"
AUDIT = HERE / "20260826-sil-koch-audit.csv"
SOURCE_KEY = "kondakov2011koch"
KEY_PREFIX = "silkoch2011"

FIRST_PAGE, LAST_PAGE = 9, 44   # section 5, the questionnaire, begins on p45
KEY_PAGE = 9
GUTTER = 300
WORDLIST_ITEMS = 210

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
    "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
    "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "PDF_Page", "Column", "Gloss", "Site_Code", "Lect", "Site", "Raw_Form",
    "Status", "Reason", "Entry_Key",
]

# The report treats Tintekiya, Wanang, Harigaya and Margan as dialects of one Koch language and
# Koch-Rabha as a further variety of it; Garo and Rongdani Rabha are the comparison languages.
LECT_LANGUAGE = {
    "Garo": "Garo",
    "Rongdani Rabha": "Rabha",
    "Margan Koch": "Koch",
    "Harigaya Koch": "Koch",
    "Wanang Koch": "Koch",
    "Tintekiya Koch": "Koch",
    "Koch-Rabha": "Koch",
}
NEW_LANGUAGES = {
    # ID: (Name, Glottocode, latitude, longitude)  -- district centroids, quality C
    "Koch": ("Koch", "koch1250", "25.52", "90.22"),
    "Garo": ("Garo", "garo1247", "25.51", "90.21"),
}

LABEL = re.compile(r"^([A-K])\.$")
KEY_LINE = re.compile(r"^([A-K])\.\s*([A-Z]{1,3}):\s*(.+?)\s*\((.+?)\)\s*$", re.M)


def verify_pdf(path: Path) -> None:
    if not path.exists():
        raise SystemExit(
            f"{path} is missing. SIL reports are not redistributed; download "
            "silesr2011_033.pdf from sil.org and place it there."
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != PDF_SHA256:
        raise SystemExit(f"{path} SHA-256 is {digest}, expected {PDF_SHA256}")


def dialect_tag(language_id: str, source_id: str, name: str) -> str:
    return (f"dialect:{quote(language_id, safe='')}:{quote(source_id, safe='')}:"
            f"{quote(name, safe='')}")


def site_id(lect: str, place: str) -> str:
    village = re.split(r"[ ,]", place.strip())[0].lower()
    slug = re.sub(r"[^a-z0-9]+", "-", f"{lect} {village}".lower()).strip("-")
    return f"koch2011-{slug}"


def read_key(page):
    """The site key: code -> (abbreviation, lect name, locality). Also its vertical extent."""
    sites = {}
    for m in KEY_LINE.finditer(page.extract_text() or ""):
        sites[m.group(1)] = (m.group(2), m.group(3).replace("‐", "-"), m.group(4))
    words = page.extract_words()
    bottom = max(
        (w["bottom"] for w in words
         if LABEL.match(w["text"]) and any(
             w["x0"] < o["x0"] < w["x0"] + 35 and o["text"].endswith(":") for o in words)),
        default=0,
    )
    return sites, bottom


def parse(pdf_path: Path):
    import pdfplumber

    records, unassigned = [], []
    with pdfplumber.open(str(pdf_path)) as pdf:
        sites, key_bottom = read_key(pdf.pages[KEY_PAGE])
        for index in range(FIRST_PAGE, LAST_PAGE + 1):
            page = pdf.pages[index]
            floor = key_bottom if index == KEY_PAGE else 0
            for column, (lo, hi) in enumerate(((0, GUTTER), (GUTTER, page.width))):
                words = [
                    w for w in page.extract_words()
                    if lo <= w["x0"] < hi and w["top"] > floor
                ]
                if not words:
                    continue
                margin = min(w["x0"] for w in words)
                rows = defaultdict(list)
                for w in words:
                    rows[round(w["top"])].append(w)

                gloss, labels, forms = None, [], []

                def flush():
                    for top, text in forms:
                        if gloss is None or not labels:
                            unassigned.append((index, column, gloss, text))
                            continue
                        code = min(labels, key=lambda l: abs(l[0] - top))[1]
                        records.append({
                            "page": index, "column": column, "gloss": gloss,
                            "code": code, "form": unicodedata.normalize("NFC", text.strip()),
                        })

                for top in sorted(rows):
                    parts = sorted(rows[top], key=lambda w: w["x0"])
                    text = " ".join(w["text"] for w in parts)
                    if re.fullmatch(r"\d+", text):
                        continue                      # running page number
                    anchor = next((w for w in parts if LABEL.match(w["text"])), None)
                    if anchor:
                        abbrev = next(
                            (w["text"] for w in parts
                             if anchor["x0"] < w["x0"] < anchor["x0"] + 35), "")
                        if abbrev.endswith(":"):
                            continue                  # a key line, not data
                        labels.append((top, anchor["text"][0]))
                        rest = [w for w in parts if w["x0"] > anchor["x0"] + 35]
                        if rest:
                            forms.append((top, " ".join(w["text"] for w in rest)))
                        continue
                    if parts[0]["x0"] < margin + 12:
                        flush()
                        gloss, labels, forms = text, [], []
                        continue
                    forms.append((top, text))
                flush()
    return sites, records, unassigned


def build(sites, records, unassigned):
    rows, audit, seen = [], [], Counter()
    for rec in records:
        site = sites.get(rec["code"])
        base = {
            "PDF_Page": rec["page"], "Column": rec["column"], "Gloss": rec["gloss"],
            "Site_Code": rec["code"], "Lect": site[1] if site else "",
            "Site": site[2] if site else "", "Raw_Form": rec["form"],
        }
        if not site:
            audit.append({**base, "Status": "unmapped", "Entry_Key": "",
                          "Reason": f"site code {rec['code']!r} is not in the key"})
            continue
        language = LECT_LANGUAGE.get(site[1])
        if not language:
            audit.append({**base, "Status": "unmapped", "Entry_Key": "",
                          "Reason": f"lect {site[1]!r} has no canonical language"})
            continue
        sid = site_id(site[1], site[2])
        key = ""
        # the source prints synonyms comma-separated; the compiler treats a comma as a form
        # separator, so split here and give each part its own record key
        for part in [p.strip() for p in rec["form"].split(",") if p.strip()] or [rec["form"]]:
            seen[(rec["gloss"], sid)] += 1
            key = (f"{KEY_PREFIX}:{rec['code']}:i{seen[(rec['gloss'], sid)]}"
                   f":{re.sub(r'[^a-z0-9]+', '-', rec['gloss'].lower()).strip('-')[:28]}")
            rows.append({
                "Language_ID": language, "Parameter_ID": "", "Form": part,
                "Gloss": rec["gloss"], "Native": "", "Phonemic": part,
                "Notes": "", "Source": f"{SOURCE_KEY}[wordlist item {rec['gloss']!r}, "
                                       f"site {rec['code']}, {site[1]}]",
                "Cognateset": "", "Etymology": "", "Entry_Key": key, "Variant_Of_Key": "",
                "Borrowed_From_Key": "", "Derivation_Parent_Keys": "",
                "Tags": dialect_tag(language, sid, f"{site[1]} ({site[2].split(',')[0]})"),
            })
        audit.append({**base, "Status": "installed", "Reason": "", "Entry_Key": key})
    for page, column, gloss, text in unassigned:
        audit.append({"PDF_Page": page, "Column": column, "Gloss": gloss or "", "Site_Code": "",
                      "Lect": "", "Site": "", "Raw_Form": text, "Status": "unparsed",
                      "Reason": "no site label could be resolved for this fragment",
                      "Entry_Key": ""})
    return rows, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--pdf", type=Path, default=PDF)
    args = parser.parse_args()

    verify_pdf(args.pdf)
    sites, records, unassigned = parse(args.pdf)
    rows, audit = build(sites, records, unassigned)

    status = Counter(a["Status"] for a in audit)
    glosses = {a["Gloss"] for a in audit if a["Gloss"]}
    by_site = Counter(r["Tags"].rsplit(":", 2)[1] for r in rows)
    print(f"sites in key      : {len(sites)}")
    print(f"installed rows    : {len(rows)}")
    print(f"audit rows        : {len(audit)}  {dict(status)}")
    print(f"distinct glosses  : {len(glosses)} / {WORDLIST_ITEMS}")
    for language in sorted({r['Language_ID'] for r in rows}):
        n = sum(1 for r in rows if r["Language_ID"] == language)
        s = len({r["Tags"].rsplit(":", 2)[1] for r in rows if r["Language_ID"] == language})
        print(f"   {language:8}{n:6} forms / {s} sites")

    if args.install:
        with INSTALLED.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            for row in rows:
                writer.writerow([row[f] for f in FORM_FIELDS])
        with AUDIT.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(audit)
        print(f"wrote {INSTALLED.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
