#!/usr/bin/env python3
"""Install the SIL Bangladesh survey wordlists printed in the bracketed-site-code layout.

Several reports in the SIL Bangladesh series print their appendix wordlists as::

    7 rainbow                        <- item number and gloss
    0 no entry [ v ]                 <- a printed gap for site v
    1 ɾɔŋdhɔnu [ 0s ]                <- similarity group, form, and every site that gave it
    2 tɔit̯ʃɔŋgli [ abjkl ]

One printed line therefore stands for as many attestations as it lists site codes, and the codes
are resolved through the key printed just before the wordlists.  Group ``0`` with the form
``no entry`` marks an item that was not elicited at that site.

Currently wired up:

* ``silesr2011_038`` -- *The Tripura of Bangladesh: A Sociolinguistic Survey* (Kim, Kim, Sangma
  and Ahmad, 2011).  306 items; codes ``a``--``t`` are the twenty Kok Borok sites, ``u``--``w``
  the three Garo wordlists (Abeng, Chibok and A'tong) and ``0`` the standard Bangla list.

The other reports in this series (2011-023 Koch, 2011-040 Kurux, 2012-007 Garo) use the same
layout but their PDFs carry no usable ToUnicode map, so their text layer decodes to ``(cid:NN)``
sequences; they need a font decoder before they can be added here.

The reports are not redistributed: this importer requires the publisher PDFs under
``tmp/pdfs/sil-surveys/`` and verifies each SHA-256 before reading it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import NamedTuple
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ROOT = HERE.parents[4]
PDF_DIR = ROOT / "tmp/pdfs/sil-surveys"

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
    "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
    "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "Report", "PDF_Page", "Column", "Item", "Gloss", "Group", "Site_Code", "Site",
    "Raw_Form", "Status", "Reason", "Entry_Key",
]


# Verified glyph table for the subsetted phonetic font in silesr2011-040.  Outline shapes alone
# were not reliable: `1426` reads as ʈ but is plain `t` (tiktikki, bristi, cokh in the Bangla
# comparison list), and `1101` is unreadable from its outline but is an a-quality vowel (hapta,
# alu, Kurux mandi).  It maps to ɑ rather than a because the font carries a separate plain `a`.
KURUX_GLYPHS = {
    "1007": "ɛ", "1024": "ʃ", "1035": "ʒ", "1101": "ɑ", "1104": "ɔ", "1106": "ə",
    "1122": "ɹ", "1128": "ʊ", "1129": "ʌ", "1409": "ɡ", "1418": "ŋ", "1426": "t",
    "1513": "ɾ", "1522": "ɽ", "1605": "ɖ", "1618": "ɲ", "1711": "ɪ", "1810": "ɦ",
}
# 2011-023 and 2012-007 use the same numbering with three further glyphs.  1041 and 7011 are
# unambiguous outlines (a dotless question mark, and an i crossed by a bar); 9086 is the corner
# stroke of the unreleased-stop diacritic, which these languages need for their final stops.
# 9064 remains unread and any form carrying it is held back.
BANGLADESH_GLYPHS = dict(KURUX_GLYPHS, **{"1041": "ʔ", "7011": "ɨ", "9086": "\u031a"})


class Report(NamedTuple):
    pdf: str
    sha256: str
    source_key: str
    key_prefix: str
    site_prefix: str
    installed: str
    audit: str
    key_page: int
    first_page: int
    last_page: int
    gutter: int
    items: int
    region: str
    languages: dict      # site code -> Jambu language ID
    key_pattern: str = "paren"   # how the site key prints: "paren" or "plain"
    glyphs: dict = {}            # SIL glyph id -> character, for fonts with no ToUnicode


TRIPURA_CODES = {c: "Kokborok" for c in "abcdefghijklmnopqrst"}
TRIPURA_CODES.update({"u": "Garo", "v": "Garo", "w": "Garo", "0": "B"})

REPORTS = {
    "tripura": Report(
        pdf="silesr2011_038.pdf",
        sha256="3a40dff97634fba2996fe81c9b54c9faa360f2b41a35f559ba4d10e102ca6a6a",
        source_key="kim-kim-sangma-ahmad2011tripura",
        key_prefix="siltripura2011",
        site_prefix="tripura2011",
        installed="20260826-sil-tripura.csv",
        audit="20260826-sil-tripura-audit.csv",
        key_page=81, first_page=82, last_page=157,
        gutter=300, items=306,
        region="Chittagong Hill Tracts and Greater Sylhet, Bangladesh",
        languages=TRIPURA_CODES,
    ),
    "kurux": Report(
        pdf="silesr2011_040.pdf",
        sha256="f2f06c25ac55462d6a40843539d8417e24a647bd1eb0bbe3f24ea3e45f0b9e4b",
        source_key="kim-ahmad-kim-sangma2011kurux",
        key_prefix="silkurux2011",
        site_prefix="kurux2011",
        installed="20260826-sil-kurux.csv",
        audit="20260826-sil-kurux-audit.csv",
        key_page=37, first_page=38, last_page=56,   # Appendix B, the questionnaires, begins on p57
        gutter=300, items=307,
        region="Rangpur and Dinajpur divisions, Bangladesh, and West Bengal, India",
        languages={"A": "Kurux", "B": "Kurux", "C": "Kurux", "D": "Kurux", "E": "Kurux",
                   "0": "B"},
        key_pattern="plain",
        glyphs=KURUX_GLYPHS,
    ),
    "kochbd": Report(
        pdf="silesr2011_023.pdf",
        sha256="d1b2d597c16fd0338ad47d2bf031566192c5ff4e26a6651de14a228df681fc10",
        source_key="kim-ahmad-kim-sangma2011kochbd",
        key_prefix="silkochbd2011",
        site_prefix="kochbd2011",
        installed="20260826-sil-kochbd.csv",
        audit="20260826-sil-kochbd-audit.csv",
        key_page=41, first_page=42, last_page=61,
        gutter=297, items=307,
        region="Greater Mymensingh, Bangladesh",
        # A'tong is a Garo variety; Tintekiya and Chapra are Koch
        languages={"b": "Koch", "c": "Koch", "r": "Koch", "q": "Koch",
                   "l": "Garo", "m": "Garo", "0": "B"},
        glyphs=BANGLADESH_GLYPHS,
    ),
    "garobd": Report(
        pdf="silesr2012_007.pdf",
        sha256="4248b409d816c153f95c09e50bf51f9e5ff90d456e3c8d9d13dc2eca6f8c4359",
        source_key="kim-kim-sangma2012garo",
        key_prefix="silgarobd2012",
        site_prefix="garobd2012",
        installed="20260826-sil-garobd.csv",
        audit="20260826-sil-garobd-audit.csv",
        key_page=50, first_page=51, last_page=92,
        gutter=300, items=307,
        region="Greater Mymensingh and the Garo Hills, Bangladesh and India",
        # Abeng, Brak, Chibok, Dual and A'tong are Garo varieties; Megam and Lyngngam are Khasian
        languages={"a": "Garo", "i": "Garo", "o": "Garo", "d": "Garo", "n": "Garo",
                   "e": "Garo", "f": "Garo", "g": "Garo", "h": "Garo",
                   "l": "Garo", "m": "Garo", "b": "Koch", "c": "Koch",
                   "j": "Megam", "k": "Megam", "p": "Lyngngam", "0": "B"},
        glyphs=BANGLADESH_GLYPHS,
    ),
}

# "a Boro Pharangsia (Usoi)" -- village, then the clan or Garo dialect in parentheses.  Two key
# columns share a line, and the Bangla entry carries no parenthesis.
KEY_LINE = re.compile(r"(?:^|\s)([a-z0])\s+([A-Z][^()]*?)\s*\(([^)]+)\)")
KEY_BARE = re.compile(r"(?:^|\s)([a-z0])\s+([A-Z][A-Za-z']*)\s*$")
KEY_PLAIN = re.compile(r"^([A-Z0])\s+\(?([A-Z][^()]*?)\)?\s*$")
# some entries print only a parenthesised name, e.g. "0 (Bangla)" or "p (Lyngngam)"
KEY_PARENONLY = re.compile(r"(?:^|\s)([a-z0])\s+\(([^)]+)\)\s*$")
# items 298-306 are pronouns whose glosses themselves start with a digit ("1s", "3p"),
# so the gloss may not be required to begin with a non-digit; entry lines are
# distinguished by their bracketed site codes and are matched first.
GLOSS = re.compile(r"^(\d{1,3})\s+([^\[]+?)\s*$")
ENTRY = re.compile(r"^([0-9A-Za-z])\s+(.+?)\s*\[\s*([A-Za-z0-9]+)\s*\]\s*\]?\s*$")
NO_ENTRY = re.compile(r"^no entry$", re.I)
# 2011-040 marks an item nobody was asked as "[not used]" on its own line, distinct from the
# per-site "no entry"; both are printed gaps rather than data.
NOT_USED = re.compile(r"^\[\s*not used\s*\]$", re.I)


def verify_pdf(path: Path, sha256: str) -> None:
    if not path.exists():
        raise SystemExit(
            f"{path} is missing. SIL reports are not redistributed; download {path.name} "
            "from sil.org and place it there."
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha256:
        raise SystemExit(f"{path} SHA-256 is {digest}, expected {sha256}")


def dialect_tag(language_id: str, source_id: str, name: str) -> str:
    return (f"dialect:{quote(language_id, safe='')}:{quote(source_id, safe='')}:"
            f"{quote(name, safe='')}")


def site_id(report: Report, code: str, village: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", village.lower()).strip("-")
    return f"{report.site_prefix}-{code}-{slug}"


CID = re.compile(r"^\(cid:(\d+)\)$")


def font_differences(pdf_path: Path) -> dict:
    """Font name -> {character code: glyph name}, from each font's /Differences array.

    These PDFs embed subsetted phonetic fonts with no ToUnicode map, so pdfminer reports their
    characters as ``(cid:NN)``.  The /Differences array is the only surviving link from a code to
    a glyph name, and the glyph names are what the verified table is keyed on.
    """
    from pdfminer.pdfdocument import PDFDocument
    from pdfminer.pdfpage import PDFPage
    from pdfminer.pdfparser import PDFParser
    from pdfminer.pdftypes import resolve1

    out: dict = {}
    with pdf_path.open("rb") as handle:
        for page in PDFPage.create_pages(PDFDocument(PDFParser(handle))):
            for _, ref in resolve1(page.resources.get("Font") or {}).items():
                font = resolve1(ref)
                enc = resolve1(font.get("Encoding"))
                if not isinstance(enc, dict) or "Differences" not in enc:
                    continue
                name = str(font.get("BaseFont")).strip("/'").split("+")[-1]
                table = out.setdefault(name, {})
                code = 0
                for item in resolve1(enc["Differences"]):
                    if isinstance(item, int):
                        code = item
                    else:
                        table[code] = str(item).lstrip("/").strip("'")
                        code += 1
    return out


def decoded_lines(page, lo, hi, glyphs, differences):
    """Rebuild the lines of one column, decoding (cid:NN) characters through the glyph table."""
    from fontTools.agl import toUnicode

    rows: dict = {}
    for ch in page.chars:
        if not (lo <= ch["x0"] < hi):
            continue
        text = ch["text"]
        match = CID.match(text)
        if match:
            font = ch["fontname"].split("+")[-1]
            name = differences.get(font, {}).get(int(match.group(1)))
            text = glyphs.get(name) or (toUnicode(name) if name else "") or "\ufffd"
        # cluster characters onto a line: the appendix leads to sub-point baseline jitter
        key = next((k for k in rows if abs(k - ch["top"]) <= 2.5), round(ch["top"], 1))
        rows.setdefault(key, []).append((ch["x0"], ch["x1"], text))
    lines = []
    for _, parts in sorted(rows.items()):
        parts.sort()
        out = []
        previous_end = None
        for x0, x1, text in parts:
            # pdfplumber inserts word spacing for us in the text path; rebuilding from raw
            # characters means restoring it from the horizontal gaps
            if previous_end is not None and x0 - previous_end > 1.2:
                out.append(" ")
            out.append(text)
            previous_end = x1
        lines.append("".join(out))
    return lines


def read_key(page, pattern="paren") -> dict:
    """Site code -> (village, clan or dialect)."""
    sites = {}
    if pattern == "plain":
        for line in (page.extract_text() or "").splitlines():
            m = KEY_PLAIN.match(line.strip())
            if m:
                place = m.group(2).strip()
                sites[m.group(1)] = (place.split(",")[0].strip(), place)
        return sites
    # match per line: run over the whole page and a match can straddle the line break
    for line in (page.extract_text() or "").splitlines():
        for m in KEY_LINE.finditer(line):
            sites[m.group(1)] = (m.group(2).strip(), m.group(3).strip())
        m = KEY_PARENONLY.search(line)
        if m and m.group(1) not in sites:
            sites[m.group(1)] = (m.group(2).strip(), m.group(2).strip())
        m = KEY_BARE.search(line)
        if m and m.group(1) not in sites:
            sites[m.group(1)] = (m.group(2).strip(), "")
    return sites


def parse(report: Report, pdf_path: Path):
    import pdfplumber

    records, unparsed, notused = [], [], []
    differences = font_differences(pdf_path) if report.glyphs else {}
    with pdfplumber.open(str(pdf_path)) as pdf:
        sites = read_key(pdf.pages[report.key_page], report.key_pattern)
        # a gloss block runs past the column and page break, so item/gloss carry across them
        item = gloss = None
        for index in range(report.first_page, report.last_page + 1):
            page = pdf.pages[index]
            for column, (lo, hi) in enumerate(((0, report.gutter), (report.gutter, page.width))):
                if report.glyphs:
                    lines = decoded_lines(page, lo, hi, report.glyphs, differences)
                else:
                    lines = (page.crop((lo, 0, hi, page.height)).extract_text() or "").splitlines()
                for raw in lines:
                    line = raw.strip()
                    if (not line or re.fullmatch(r"\d{1,3}", line)
                            or line.startswith("A.") or line.startswith("APPENDIX")):
                        continue           # blank, running page number, or a section heading
                    if NO_ENTRY.match(line):
                        notused.append((index, column, item, gloss))
                        continue           # a bare "no entry" with no bracketed site
                    match = ENTRY.match(line)
                    if match:
                        if gloss is None:
                            unparsed.append((index, column, line, "entry before any gloss"))
                            continue
                        records.append({
                            "page": index, "column": column, "item": item, "gloss": gloss,
                            "group": match.group(1),
                            "form": unicodedata.normalize("NFC", match.group(2).strip()),
                            "codes": match.group(3),
                        })
                        continue
                    match = GLOSS.match(line)
                    if match:
                        item, gloss = match.group(1), match.group(2).strip()
                        continue
                    if NOT_USED.match(line):
                        notused.append((index, column, item, gloss))
                        continue
                    if records and re.fullmatch(r"[^\[\]\d]{2,}", line):
                        # a form that wrapped onto its own line continues the previous record
                        records[-1]["form"] = f"{records[-1]['form']} {line}".strip()
                        continue
                    if any("\ue000" <= c <= "\uf8ff" for c in line):
                        # a font whose /Differences array is absent: the characters land in the
                        # private use area and the line cannot be read at all
                        unparsed.append((index, column, line,
                                         "characters from a font with no /Differences array"))
                    else:
                        unparsed.append((index, column, line, "matched no line shape"))
    return sites, records, unparsed, notused


def build(report: Report, sites, records, unparsed, notused=()):
    rows, audit, seen = [], [], Counter()
    for rec in records:
        for code in rec["codes"]:
            site = sites.get(code)
            base = {
                "Report": report.source_key, "PDF_Page": rec["page"], "Column": rec["column"],
                "Item": rec["item"], "Gloss": rec["gloss"], "Group": rec["group"],
                "Site_Code": code, "Site": site[0] if site else "", "Raw_Form": rec["form"],
            }
            language = report.languages.get(code)
            if not site or not language:
                audit.append({**base, "Status": "unmapped", "Entry_Key": "",
                              "Reason": f"site code {code!r} is not in the key"})
                continue
            # "!", "%", "*" and "$" are ASCII standing in for diacritics whose glyph names carry
            # no verified reading, so those forms are held back with the undecodable ones
            # characters left in the private use area belong to a font whose /Differences array
            # is absent, so their reading is unknown just as an undecodable glyph's is
            # in this layout one printed line is one form, so an embedded comma is a decoding
            # artifact rather than a separator; the compiler would also split on it
            if ("\ufffd" in rec["form"] or any(c in rec["form"] for c in "!%*$,")
                    or any("\ue000" <= c <= "\uf8ff" for c in rec["form"])):
                # a glyph whose reading is not established; installing it would invent a segment
                audit.append({**base, "Status": "excluded", "Entry_Key": "",
                              "Reason": "contains a glyph with no verified reading"})
                continue
            if rec["group"] == "0" or NO_ENTRY.match(rec["form"]):
                audit.append({**base, "Status": "excluded", "Entry_Key": "",
                              "Reason": "printed gap: the item was not elicited at this site"})
                continue
            sid = site_id(report, code, site[0])
            seen[(rec["item"], sid)] += 1
            key = f"{report.key_prefix}:i{int(rec['item']):03d}:{code}:{seen[(rec['item'], sid)]}"
            rows.append({
                "Language_ID": language, "Parameter_ID": "", "Form": rec["form"],
                "Gloss": rec["gloss"], "Native": "", "Phonemic": rec["form"],
                "Notes": f"lexical-similarity group {rec['group']}",
                "Source": (f"{report.source_key}[wordlist item {rec['item']}, "
                           f"site {code} {site[0]}]"),
                "Cognateset": "", "Etymology": "", "Entry_Key": key, "Variant_Of_Key": "",
                "Borrowed_From_Key": "", "Derivation_Parent_Keys": "",
                "Tags": dialect_tag(language, sid, f"{site[0]} ({site[1]})"),
            })
            audit.append({**base, "Status": "installed", "Reason": "", "Entry_Key": key})
    for page, column, item, gloss in notused:
        audit.append({"Report": report.source_key, "PDF_Page": page, "Column": column,
                      "Item": item or "", "Gloss": gloss or "", "Group": "", "Site_Code": "",
                      "Site": "", "Raw_Form": "[not used]", "Status": "excluded",
                      "Reason": "printed gap: the item was not elicited at any site",
                      "Entry_Key": ""})
    for page, column, line, reason in unparsed:
        unreadable = "no /Differences array" in reason
        audit.append({"Report": report.source_key, "PDF_Page": page, "Column": column,
                      "Item": "", "Gloss": "", "Group": "", "Site_Code": "", "Site": "",
                      "Raw_Form": line,
                      "Status": "excluded" if unreadable else "unparsed",
                      "Reason": reason, "Entry_Key": ""})
    return rows, audit


def run(name: str, install: bool):
    report = REPORTS[name]
    pdf = PDF_DIR / report.pdf
    verify_pdf(pdf, report.sha256)
    sites, records, unparsed, notused = parse(report, pdf)
    rows, audit = build(report, sites, records, unparsed, notused)

    status = Counter(a["Status"] for a in audit)
    items = {a["Item"] for a in audit if a["Item"]}
    by_lang = defaultdict(set)
    for row in rows:
        by_lang[row["Language_ID"]].add(row["Tags"].rsplit(":", 2)[1])
    print(f"[{name}] sites in key {len(sites)}; installed {len(rows)}; "
          f"audited {len(audit)} {dict(status)}")
    print(f"   items seen {len(items)} / {report.items}")
    for language, s in sorted(by_lang.items(), key=lambda kv: -len(kv[1])):
        n = sum(1 for r in rows if r["Language_ID"] == language)
        print(f"   {language:10}{n:6} forms / {len(s)} sites")

    if install:
        target = HERE.parent / report.installed
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            for row in rows:
                writer.writerow([row[f] for f in FORM_FIELDS])
        with (HERE / report.audit).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(audit)
        print(f"   wrote {target.relative_to(REPO)}")
    return rows, audit, sites


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="*", choices=sorted(REPORTS) + [[]], default=[])
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    for name in (args.reports or sorted(REPORTS)):
        run(name, args.install)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
