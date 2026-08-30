#!/usr/bin/env python3
"""Install the Appendix B3 wordlists of the SIL India survey reports that print them.

Two reports compiled by Bijumon Varghese share one Appendix B3 layout, "Phonetic Transcription
of Wordlists", which prints a standard 210-item list elicited at every survey site:

* ``silesr2015_029`` -- *Tribes of Idukki, Kerala* (Varghese and Mathew, 2015)
* ``silesr2015_028`` -- *The Tribes of Palakkad, Kerala: A Sociolinguistic Profile* (Varghese, 2015)

Tamil and Malayalam are elicited alongside the target lects at every item as controls for the
lexical-similarity comparison and are excluded deliberately.

The reports are not redistributed: this importer requires the publisher PDFs under
``tmp/pdfs/sil-surveys/`` and verifies each SHA-256 before reading it.

Extraction notes
----------------
Appendix B3 is set in **two columns**.  ``page.extract_text()`` reads straight across the page and
interleaves them, silently shuffling entries between adjacent glosses, so every page is cropped at
the x=305 gutter and each column is read separately.

Within a column the shapes are::

    14. elbow                          <- gloss header
    Muthuvan, Itticity   1 kʌimuʈʈɨ    <- lect, site, lexical-similarity group, form
                         4 mũːkəj      <- continuation: another form for the same site

Three wrapping behaviours have to be undone before parsing.  A trailing comma means the record
continues on the next line -- this covers both a wrapped lect label (``Muthuvan,`` /
``Chempakathozhu 2 …``) and the two-part answers elicited by glosses such as 184 "he is hungry,
was hungry".  A line that matches nothing else continues the previous record's form, which is how
long forms soft-wrap without a comma.  Group numbers run to 12, not 9.

Lexical-similarity group ``0`` marks a gap.  Sixty-two are printed ``No Entry``; one is set in a
small-caps font run that decodes to ``ɴø ɛɴθɾʏ``, so gaps are detected by the group number rather
than by matching the string.  They are audited and not installed.

Transcription is Unicode IPA in the publisher's text layer -- no OCR, no legacy font decoding, and
no sound profile.  As with the SSNP survey appendices in ``ssnp.py``, survey IPA is installed
unchanged in both ``Form`` and ``Phonemic`` rather than converted to house transcription.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from typing import NamedTuple
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]          # the data/ git root
ROOT = HERE.parents[4]          # jambu-all, which holds tmp/pdfs
PDF_DIR = ROOT / "tmp/pdfs/sil-surveys"

GUTTER = 305
WORDLIST_ITEMS = 210
# Tamil and Malayalam are elicited at every item purely as comparison controls.
CONTROLS = {"Tamil", "Malayalam"}

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
    "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
    "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "Report", "PDF_Page", "Column", "Gloss_Number", "Gloss", "Lect", "Site",
    "Group", "Raw_Form", "Status", "Reason", "Entry_Key",
]


class Report(NamedTuple):
    pdf: str
    sha256: str
    source_key: str
    key_prefix: str
    installed: str
    audit: str
    site_prefix: str         # short slug that namespaces this report's site ids
    first_page: int          # 0-indexed; Appendix B3 body
    last_page: int
    region: str
    # printed lect label -> Jambu language ID
    languages: dict


REPORTS = {
    "idukki": Report(
        pdf="silesr2015_029.pdf",
        sha256="5d2618e5ca91aba8c28f2397a28e799c68d212ff85eea8168a89227dba7471ba",
        source_key="varghese-mathew2015idukki",
        key_prefix="silidukki2015",
        installed="20260826-sil-idukki.csv",
        audit="20260826-sil-idukki-audit.csv",
        site_prefix="idukki",
        first_page=58, last_page=107,
        region="Idukki district, Kerala, India",
        languages={
            "Muthuvan": "Muthuvan", "Mannan": "Mannan", "Urali": "Urali",
            "Mala Pulaya": "MalaPulaya", "Paliya": "Paliya",
        },
    ),
    "palakkad": Report(
        pdf="silesr2015_028.pdf",
        sha256="0260a0e7ce77cdedcb9c220979036d2f662f5c9c2b74a6a056e9146abd163bf8",
        source_key="varghese2015palakkad",
        key_prefix="silpalakkad2015",
        installed="20260826-sil-palakkad.csv",
        audit="20260826-sil-palakkad-audit.csv",
        site_prefix="palakkad",
        first_page=55, last_page=110,
        region="Palakkad district, Kerala, India",
        # "Mala Malasa" and "Mala Malasar" are the same Ancham site, spelled two ways in the
        # printed appendix; "Kada" and "Eravalla" are the appendix's short labels for the
        # Kadar and Eravallan lects named in the report body.
        languages={
            "Irula": "Irula", "Muduga": "Muduga", "Kurumba": "Kurumba",
            "Kada": "Kadar", "Eravalla": "Eravallan", "Alu Kurumba": "AluKurumba",
            "Malasar": "Malasar", "Malasar pasha": "Malasar",
            "Mala Malasa": "MalaMalasar", "Mala Malasar": "MalaMalasar",
        },
    ),
}

GLOSS = re.compile(r"^(\d{1,3})\.\s+(\D.*?)\s*$")
def entry_pattern(labels):
    """Anchor entries on the report's own lect labels; site names may be several words."""
    alternation = "|".join(re.escape(l) for l in sorted(labels, key=len, reverse=True))
    return re.compile(rf"^({alternation}),\s*(.+?)\s+(\d{{1,2}})(?:\s+(.*\S))?\s*$")
CONTROL = re.compile(r"^(Tamil|Malayalam)\s+(\d{1,2})(?:\s+(.*\S))?\s*$")
CONT = re.compile(r"^(\d{1,2})(?:\s+(.*\S))?\s*$")
SKIP = re.compile(r"^(\s*|-+|\d{1,3}|Appendix B3.*|sts|Phonetic Transcription.*)$")


def verify_pdf(path: Path, sha256: str) -> None:
    if not path.exists():
        raise SystemExit(
            f"{path} is missing. SIL reports are not redistributed; download {path.name} "
            "from sil.org and place it there."
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha256:
        raise SystemExit(f"{path} SHA-256 is {digest}, expected {sha256}")


def site_id(report: "Report", lect: str, site: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", site.lower()).strip("-")
    return f"{report.site_prefix}-{report.languages[lect].lower()}-{slug}"


def dialect_tag(language_id: str, source_id: str, name: str) -> str:
    from urllib.parse import quote
    return (f"dialect:{quote(language_id, safe='')}:{quote(source_id, safe='')}:"
            f"{quote(name, safe='')}")


def columns(page):
    """Appendix B3 is two-column; read each column separately or entries interleave."""
    for lo, hi in ((0, GUTTER), (GUTTER, page.width)):
        yield page.crop((lo, 0, hi, page.height)).extract_text() or ""


def unwrap(text, entry):
    """A trailing comma continues the record: wrapped lect labels and two-part answers.

    Forms legitimately end in a comma too (the two-part answers are printed "a, b"), so a line
    that already starts a new record is never absorbed into the previous one.
    """
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        starts_record = bool(entry.match(line) or CONTROL.match(line) or GLOSS.match(line))
        if out and out[-1].endswith(",") and not starts_record:
            out[-1] = f"{out[-1]} {line}"
        else:
            out.append(line)
    return out


def _rec(gloss, owner, group, form, page, column):
    return {"gloss_no": gloss[0] if gloss else "", "gloss": gloss[1] if gloss else "",
            "lect": owner[0], "site": owner[1], "group": group,
            "form": unicodedata.normalize("NFC", form.strip()),
            "page": page, "column": column}


def parse(report, pdf_path):
    import pdfplumber
    records, audit = [], []
    # A gloss's entries run past the column break, so parser state carries across columns and
    # pages in reading order (left column, right column, next page) rather than resetting.
    gloss = owner = pending = None
    entry = entry_pattern(set(report.languages) | CONTROLS)
    with pdfplumber.open(str(pdf_path)) as pdf:
        for index in range(report.first_page, report.last_page + 1):
            for column, text in enumerate(columns(pdf.pages[index])):
                for line in unwrap(text, entry):
                    if SKIP.match(line):
                        continue
                    m = GLOSS.match(line)
                    if m and not entry.match(line):
                        gloss, owner, pending = (m.group(1), m.group(2)), None, None
                        continue
                    m = entry.match(line)
                    if m and m.group(1) not in CONTROLS:
                        owner = (m.group(1), m.group(2))
                        if m.group(4):
                            records.append(_rec(gloss, owner, m.group(3), m.group(4), index, column))
                        else:
                            pending = (owner, m.group(3))
                        continue
                    m = CONTROL.match(line)
                    if m:
                        owner = (m.group(1), "")
                        if m.group(3):
                            records.append(_rec(gloss, owner, m.group(2), m.group(3), index, column))
                        else:
                            pending = (owner, m.group(2))
                        continue
                    m = CONT.match(line)
                    if m and owner:
                        if m.group(2):
                            records.append(_rec(gloss, owner, m.group(1), m.group(2), index, column))
                        else:
                            pending = (owner, m.group(1))
                        continue
                    if pending is not None:
                        own, group = pending
                        pending = None
                        records.append(_rec(gloss, own, group, line, index, column))
                        continue
                    if records:
                        records[-1]["form"] += " " + line
                        continue
                    audit.append({"Report": report.source_key, "PDF_Page": index,
                                  "Column": column, "Gloss_Number": "", "Gloss": "", "Lect": "",
                                  "Site": "", "Group": "", "Raw_Form": line, "Status": "unparsed",
                                  "Reason": "matched no line shape", "Entry_Key": ""})
    return records, audit


def build(report, records):
    """Split parsed records into installable rows and a complete per-record audit."""
    rows, audit, seen = [], [], Counter()
    for rec in records:
        base = {"Report": report.source_key, "PDF_Page": rec["page"], "Column": rec["column"],
                "Gloss_Number": rec["gloss_no"], "Gloss": rec["gloss"], "Lect": rec["lect"],
                "Site": rec["site"], "Group": rec["group"], "Raw_Form": rec["form"]}
        if rec["lect"] in CONTROLS:
            audit.append({**base, "Status": "excluded", "Entry_Key": "",
                          "Reason": "control language for the lexical-similarity comparison"})
            continue
        if rec["lect"] not in report.languages:
            audit.append({**base, "Status": "unmapped", "Entry_Key": "",
                          "Reason": f"lect {rec['lect']!r} is not a target of this survey"})
            continue
        if rec["group"] == "0":
            audit.append({**base, "Status": "excluded", "Entry_Key": "",
                          "Reason": "lexical-similarity group 0 marks a printed gap (No Entry)"})
            continue
        language = report.languages[rec["lect"]]
        sid = site_id(report, rec["lect"], rec["site"])
        # Two-part prompts ("drink, he drank") are answered with two forms printed "a, b".  The
        # compiler treats a comma as a form separator, so split here instead and give each part
        # its own record key: otherwise one Entry_Key would cover two compiled rows.
        parts = [p.strip() for p in rec["form"].split(",") if p.strip()] or [rec["form"]]
        keys = []
        for part in parts:
            seen[(rec["gloss_no"], sid)] += 1
            key = (f"{report.key_prefix}:g{int(rec['gloss_no']):03d}:{sid}"
                   f":i{seen[(rec['gloss_no'], sid)]}")
            keys.append(key)
            rows.append({"Language_ID": language, "Parameter_ID": "", "Form": part,
                         "Gloss": rec["gloss"], "Native": "", "Phonemic": part,
                         "Notes": f"Appendix B3 lexical-similarity group {rec['group']}",
                         "Source": (f"{report.source_key}[Appendix B3, item {rec['gloss_no']}, "
                                    f"{rec['lect']}, {rec['site']}]"),
                         "Cognateset": "", "Etymology": "", "Entry_Key": key, "Variant_Of_Key": "",
                         "Borrowed_From_Key": "", "Derivation_Parent_Keys": "",
                         "Tags": dialect_tag(language, sid, rec["site"])})
        audit.append({**base, "Status": "installed", "Reason": "", "Entry_Key": " ".join(keys)})
    return rows, audit


def summarise(name, rows, audit, parse_audit):
    by_lang = defaultdict(Counter)
    for row in rows:
        by_lang[row["Language_ID"]][row["Tags"].rsplit(":", 2)[1]] += 1
    status = Counter(e["Status"] for e in audit) + Counter(e["Status"] for e in parse_audit)
    glosses = {e["Gloss_Number"] for e in audit if e["Gloss_Number"]}
    print(f"\n[{name}] installed {len(rows)}; audited {len(audit) + len(parse_audit)}; "
          f"glosses {len(glosses)}/{WORDLIST_ITEMS}  {dict(status)}")
    for language, sites in sorted(by_lang.items(), key=lambda kv: -sum(kv[1].values())):
        print(f"    {language:14}{sum(sites.values()):5} forms / {len(sites)} sites")
    assert not status.get("unparsed"), f"{name}: every line must be accounted for"
    assert len(glosses) == WORDLIST_ITEMS, f"{name}: saw {len(glosses)} glosses"


def run(name, install):
    report = REPORTS[name]
    pdf = PDF_DIR / report.pdf
    verify_pdf(pdf, report.sha256)
    records, parse_audit = parse(report, pdf)
    rows, audit = build(report, records)
    summarise(name, rows, audit, parse_audit)
    if install:
        target = HERE.parent / report.installed
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            for row in rows:
                writer.writerow([row[f] for f in FORM_FIELDS])
        with (HERE / report.audit).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(audit + parse_audit)
        print(f"    wrote {target.relative_to(REPO)}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="*", choices=sorted(REPORTS) + [[]], default=[])
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    for name in (args.reports or sorted(REPORTS)):
        run(name, args.install)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
