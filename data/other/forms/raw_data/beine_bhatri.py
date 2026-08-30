"""Extract the twelve Appendix A word lists from Beine (2017), the SIL Bhatri survey.

The source is SIL Electronic Survey Report 2017-005, *A Sociolinguistic Survey of the
Bhatri-speaking Communities of Central India*, compiled by Dave Beine from fieldwork carried
out between February and November 1989 and published in May 2017.  Appendix A (PDF pages
15--38 = printed pages 10--33) prints a 12 site x 210 concept comparative word list: nine
Bhatri survey points in Bastar (Madhya Pradesh, now Chhattisgarh) and Koraput (Orissa, now
Odisha), plus the three comparison lects the survey tested intelligibility against -- Halbi
from Bhatpal, standard Oriya from Cuttack, and Adivasi Oriya from the Araku Valley.

What is installed
-----------------
All twelve lects.  The three comparison lects are elicited field word lists from named
localities exactly like the nine Bhatri points, so they are ingested rather than discarded as
survey "controls"; each is filed under its own canonical base language.  Every attested cell
becomes one unetymologised (``unlinked``) Jambu form under a registered dialect for its survey
site.  A cell printed ``--`` (en dash) carries no elicited word and is recorded in the audit as
``missing`` rather than installed.  A cell that lists two responses separated by the source's
``ʔ`` (or, once, ``/``) becomes one row per response: these are distinct lexemes (``mutⁿ``
beside ``pise̠b`` for 'urine'), not spelling variants, so they are not linked as variants.

Extraction
----------
The 2017 PDF carries a positioned CharisSIL text layer, so no OCR is needed, but the layer is
not laid out as a table and ordinary text extraction garbles it: column x-positions shift on
printed page 13, and combining marks are emitted on their own y-band.  Each line is therefore
rebuilt by assigning every character to the nearest dominant row baseline and concatenating in
*content-stream* order, which is the only order that keeps a combining mark next to its base.
Lines are then split on the twelve fixed uppercase site codes, which no form can contain.

Transcription
-------------
``conversion/beine-bhatri.txt`` maps Beine's "modified International Phonetic Alphabet"
(his own description; the report prints no key) onto Jambu's house transcription.  The
values below are established distributionally across the 2,520 cells and against the
Indo-Aryan etyma, and were checked against the rendered page images:

* ``ˑ`` after ``t d n r l`` marks retroflexion (``tˑondˑ`` 'mouth', ``ɡodˑ`` 'leg') -> ``ṭ ḍ ṇ ṛ ḷ``.
* ``ⁿ`` after ``t d`` marks dentality (``dⁿatⁿ`` 'teeth').  Like Beine's Gondi lists the source
  marks it only sporadically -- ``hat``, ``hatⁿ`` and ``atⁿ`` all appear for 'arm' -- so both
  spellings render as the house dental while ``Original`` keeps the source's own diacritics.
* ``̽`` (combining x above) marks a nasal vowel (``mu̽dˑ`` 'head') -> combining tilde.
* ``̂`` marks aspiration on ``h`` and non-syllabicity on a vowel (``soîla`` beside ``soila``
  'sleep') -> ``ʰ`` and ``̯``.
* ``̠`` (combining minus below) marks a look-alike letter used as a vowel symbol:
  ``e̠`` -> ``ə``, ``v̠`` -> ``ʌ``, ``c̠`` -> ``ɔ``.  ``v̠`` and ``e̠`` alternate freely across
  sites for the inherent vowel (``ɡv̠ɡe̠r̩`` beside ``ɡaɡe̠r̩`` beside ``ɡaɡv̠r̩`` 'body'), but
  they are kept distinct rather than collapsed, since a single word may contain both.
* ``̩`` (combining vertical line below) is the source's under-tick.  On ``r`` it is the
  retroflex flap (``r̩`` -> ``ṛ``); on ``ʃ`` and ``ʒ`` it is obligatory (186/186 and 146/146)
  and therefore carries no information, so ``tʃ̩`` -> ``c`` and ``dʒ̩`` -> ``j``, with the four
  Oriya tatsamas that print ``ʃ̩`` alone (``nakʃ̩atⁿr̩a`` 'star') giving ``ṣ``.
* ``.̩`` is a sporadic juncture mark, not a segment: ``bol.̩a`` beside ``bola`` 'speak',
  ``am.̩a`` beside ``ama`` 'mango'.  It is kept as ``.``.

Two residues are deliberately *not* resolved and their rows are tagged ``uncertain``:
the under-tick on a vowel (``e̩k`` beside ``ek`` 'one', ``soi̩la`` beside ``soîla``), which is
kept verbatim, and ``ɵ̩``, printed as a barred o with an under-tick in exactly two BUM cells
where 'flower' and 'fruit' require /pʰ/, rendered ``ɸ`` so that no phonological reading is
imposed.

Mechanical repairs to the 2017 typesetting (all recorded per record in the audit):

* ``U+FFFD`` in 24 cells is a dotless i whose ToUnicode entry is missing; the page renders
  ``boı̽si`` 'buffalo', ``sı̽ɡ`` 'horns'.  Repaired to ``i`` before conversion.
* Printed page 11 labels the Cuttack Oriya row of concepts 13--15 ``OC`` instead of ``OCU``.
* Two Halbi cells (50 'rainbow', 105 'father') print a stray combining diaeresis below and no
  word at all; they are treated as missing.
* The English prompts print ``ɡ`` for g, ``ʏ`` for y, ``ɪ`` for I and ``ʡ`` for ``?``, and the
  parentheses of 'we (incl.)' are broken into stray combining marks.  Glosses are repaired.

Usage
-----
    uv run python data/other/forms/raw_data/beine_bhatri.py            # preview + audit only
    uv run python data/other/forms/raw_data/beine_bhatri.py --install  # write the installed CSV
    uv run python data/other/forms/raw_data/beine_bhatri.py --dialects # print cldf/dialects.csv rows
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import random
import re
import sys
import unicodedata
from pathlib import Path
from urllib.parse import quote

import pdfplumber
from segments.tokenizer import Tokenizer


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PDF = HERE.parents[4] / "tmp" / "pdfs" / "beine-bhatri" / "source.pdf"
PDF_SHA256 = "4a9fec255be6a5e9967b66c494a4be575599d39bce58609573733a51a09d2891"

PROFILE = ROOT / "conversion" / "beine-bhatri.txt"
INSTALLED = HERE.parent / "20260825-beine-bhatri.csv"
PREVIEW = HERE.parents[4] / "tmp" / "20260825-beine-bhatri-preview.csv"
AUDIT = HERE / "20260825-beine-bhatri-audit.csv"
MANIFEST = HERE / "20260825-beine-bhatri-manifest.json"
SAMPLE = HERE / "20260825-beine-bhatri-sample.csv"

SOURCE = "beine2017bhatri"
DIALECT_PREFIX = "beine_"
# Beine's Gondi survey word lists already own the ``beine:`` Entry_Key namespace, so this
# source qualifies its keys with the work rather than only the author.
ENTRY_PREFIX = "beine-bhatri:"

# Appendix A occupies zero-based PDF pages 15--38; printed page = PDF page - 5.
FIRST_PAGE, LAST_PAGE = 15, 38
PAGE_OFFSET = 5
EXPECTED_CONCEPTS = 210
EXPECTED_SITES = 12
EXPECTED_CELLS = EXPECTED_CONCEPTS * EXPECTED_SITES
MISSING_MARKER = "–"          # en dash: no word elicited
RESPONSE_SEPARATORS = "ʔ/"    # the source's ʔ (46 cells) and one /

# The twelve speech-variety codes, in the fixed order every block prints them.
LECTS = ["OAR", "OCU", "HBH", "BAU", "BSA", "BJE", "BKP", "BUM", "BCB", "BAN", "BAR", "BAG"]
CODE = re.compile("(" + "|".join(LECTS) + ")")
NUMBER = re.compile(r"(\d+)\s*\.")

# Canonical base language, clade and locality for each survey site.  The descriptions are the
# printed key on printed page 10; coordinates live in cldf/dialects.csv.
SITES = {
    "OAR": ("AdivasiOriya", "Eastern", "Araku Valley",
            "Adivasi Oriya from Araku Valley, Visak District, Andhra Pradesh"),
    "OCU": ("Or", "Eastern", "Cuttack",
            "Oriya from Cuttack Tahsil, Cuttack District, Orissa"),
    "HBH": ("hal", "Halbic", "Bhatpal",
            "Halbi from Bhatpal, Kondagaon Tahsil, Bastar District, Madhya Pradesh"),
    "BAU": ("Bhatri", "Halbic", "Auli",
            "Bhatri from Auli, Nowrangapur Tahsil, Koraput District, Orissa"),
    "BSA": ("Bhatri", "Halbic", "Sargipal",
            "Bhatri from Sargipal, Jagdalpur Tahsil, Bastar District, Madhya Pradesh"),
    "BJE": ("Bhatri", "Halbic", "Jeypore",
            "Bhatri from Jeypore Tahsil, Koraput District, Orissa"),
    "BKP": ("Bhatri", "Halbic", "Kotpad",
            "Bhatri from Kotpad Tahsil, Koraput District, Orissa"),
    "BUM": ("Bhatri", "Halbic", "Umarkot",
            "Bhatri from Umarkot Tahsil, Koraput District, Orissa"),
    "BCB": ("Bhatri", "Halbic", "Chote Badal",
            "Bhatri from Chote Badal, Jagdalpur Tahsil, Bastar District, Madhya Pradesh"),
    "BAN": ("Bhatri", "Halbic", "Anantpur",
            "Bhatri from Anantpur, Kondagaon Tahsil, Bastar District, Madhya Pradesh"),
    "BAR": ("Bhatri", "Halbic", "Karpaud",
            "Bhatri from Karpaud, Jagdalpur Tahsil, Bastar District, Madhya Pradesh"),
    "BAG": ("Bhatri", "Halbic", "Amaguda",
            "Bhatri from Amaguda, Jagdalpur Tahsil, Bastar District, Madhya Pradesh"),
}

# Font substitutions in the English prompts of the 2017 typesetting.  The parentheses of
# 'we (incl.)', 'we (excl.)' and 'you (pl.)' are printed as stray combining marks.
GLOSS_REPAIRS = {
    "ɡ": "g",   # LATIN SMALL LETTER SCRIPT G -> g
    "ʏ": "y",   # LATIN LETTER SMALL CAPITAL Y -> y
    "ɪ": "I",   # LATIN LETTER SMALL CAPITAL I -> I
    "ʡ": "?",   # LATIN LETTER GLOTTAL STOP WITH STROKE -> ?
    "̆": "(",   # COMBINING BREVE -> (
    "̃": ")",   # COMBINING TILDE -> )
}
GLOSS_OVERRIDES = {208: "we (excl.)", 209: "you (pl.)"}

# U+FFFD stands for a dotless i that the PDF's ToUnicode map does not cover; the page itself
# renders an ordinary dotless i carrying the source's nasalization mark.
DOTLESS_I = "�"

# Cells whose printed content is a stray combining mark and no word at all.
DIACRITIC_ONLY = re.compile(r"^[̀-ͯ\s]*$")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_pdf() -> None:
    """Fail loudly when the uncommitted source PDF is absent or is a different file."""
    if not PDF.is_file():
        raise SystemExit(
            f"missing source PDF {PDF}\n"
            "  download SIL Electronic Survey Report 2017-005 from\n"
            "  https://www.sil.org/resources/archives/71330 and save it there"
        )
    actual = digest(PDF)
    if actual != PDF_SHA256:
        raise SystemExit(
            f"{PDF} is not the pinned SIL 2017-005 PDF:\n"
            f"  expected {PDF_SHA256}\n  found    {actual}"
        )


def page_lines(page) -> list[tuple[float, str]]:
    """Return (baseline, text) for each printed line of an Appendix A page body.

    Characters are bucketed onto the nearest dominant baseline -- combining marks are emitted
    about 2pt below their base -- and then concatenated in content-stream order, which is the
    only order that keeps a mark adjacent to the letter it belongs to.  The running folio sits
    outside the 60--700pt body band (at the foot on the first page, at the head thereafter).
    """
    body = [(i, c) for i, c in enumerate(page.chars) if 60.0 <= c["top"] <= 700.0]
    counts = collections.Counter(round(c["top"], 1) for _, c in body)
    anchors = sorted(top for top, n in counts.items() if n >= 5)
    if not anchors:
        raise ValueError("no text rows found on an Appendix A page")
    buckets: dict[float, list[tuple[int, dict]]] = collections.defaultdict(list)
    for i, c in body:
        buckets[min(anchors, key=lambda top: abs(top - c["top"]))].append((i, c))
    lines = []
    for anchor in sorted(buckets):
        text = "".join(c["text"] for _, c in sorted(buckets[anchor]))
        lines.append((anchor, unicodedata.normalize("NFC", text)))
    return lines


def clean_gloss(value: str) -> str:
    for bad, good in GLOSS_REPAIRS.items():
        value = value.replace(bad, good)
    return re.sub(r"\s+", " ", value).strip()


def extract(pdf_path: Path = PDF) -> tuple[list[dict[str, str]], dict[str, str], list[str]]:
    """Return one record per printed cell, the printed site key, and repair notes."""
    blocks: list[dict] = []
    key: dict[str, str] = {}
    notes: list[str] = []
    current: dict | None = None

    with pdfplumber.open(pdf_path) as document:
        if len(document.pages) != 95:
            raise ValueError(f"expected the 95-page report, got {len(document.pages)}")
        for index in range(FIRST_PAGE, LAST_PAGE + 1):
            page = document.pages[index]
            printed = index - PAGE_OFFSET
            folio = "".join(
                c["text"] for c in sorted(
                    (c for c in page.chars if c["top"] < 60.0 or c["top"] > 700.0),
                    key=lambda c: c["x0"],
                )
            ).strip()
            if folio != str(printed):
                raise ValueError(
                    f"PDF page {index + 1} prints folio {folio!r}, expected {printed}"
                )
            for _, raw in page_lines(page):
                line = re.sub(r"\s+", " ", raw).strip()
                if not line:
                    continue
                if NUMBER.match(line):
                    parts = NUMBER.split(line)
                    current = {
                        "printed": printed,
                        "concepts": [
                            (int(parts[i]), clean_gloss(parts[i + 1]))
                            for i in range(1, len(parts), 2)
                        ],
                        "rows": [],
                    }
                    blocks.append(current)
                    continue
                # Printed page 11 drops the final U of OCU on one row.
                if re.match(r"^OC\s", line):
                    notes.append(
                        f"printed page {printed}: source labels an Oriya (Cuttack) row "
                        f"'OC'; read as OCU from its fixed position: {line!r}"
                    )
                    line = "OCU" + line[2:]
                if not CODE.match(line):
                    if current is None:
                        continue  # Appendix heading lines above the site key
                    raise ValueError(f"unparsed wordlist line on printed page {printed}: {line!r}")
                parts = CODE.split(line)
                pairs = [(parts[i], parts[i + 1].strip()) for i in range(1, len(parts), 2)]
                if current is None:
                    code, description = pairs[0]
                    key[code] = re.sub(r"\s+", " ", description).strip()
                    continue
                current["rows"].append(pairs)

    if len(key) != EXPECTED_SITES or list(key) != LECTS:
        raise ValueError(f"expected the printed key to list {LECTS}, got {list(key)}")

    records: list[dict[str, str]] = []
    numbers: list[int] = []
    for block in blocks:
        codes = [pairs[0][0] for pairs in block["rows"]]
        if codes != LECTS:
            raise ValueError(
                f"printed page {block['printed']} block {block['concepts']} lists {codes}"
            )
        for column, (number, gloss) in enumerate(block["concepts"]):
            numbers.append(number)
            gloss = GLOSS_OVERRIDES.get(number, gloss)
            if not gloss:
                raise ValueError(f"concept {number} has no English prompt")
            for pairs in block["rows"]:
                if len(pairs) != len(block["concepts"]):
                    raise ValueError(
                        f"printed page {block['printed']} row {pairs[0][0]} has "
                        f"{len(pairs)} cells, expected {len(block['concepts'])}"
                    )
                records.append({
                    "Concept": str(number),
                    "Gloss": gloss,
                    "Site_Code": pairs[column][0],
                    "Printed_Page": str(block["printed"]),
                    "Column": str(column + 1),
                    "Raw_Cell": pairs[column][1],
                })

    if numbers != list(range(1, EXPECTED_CONCEPTS + 1)):
        raise ValueError(f"expected concepts 1--{EXPECTED_CONCEPTS} in order, got {numbers}")
    if len(records) != EXPECTED_CELLS:
        raise ValueError(f"expected {EXPECTED_CELLS} printed cells, got {len(records)}")
    return records, key, notes


def build() -> tuple[list[list[str]], list[dict[str, str]], list[str]]:
    """Return the installed rows, the per-cell audit rows, and the repair notes."""
    records, key, notes = extract()
    convert = Tokenizer(str(PROFILE))
    forms: list[list[str]] = []
    audit: list[dict[str, str]] = []

    for record in records:
        code = record["Site_Code"]
        language, _clade, _locality, _description = SITES[code]
        raw = record["Raw_Cell"]
        citation = (
            f"{SOURCE}[p. {record['Printed_Page']}, item {record['Concept']}, site {code}]"
        )
        shared = {
            **record,
            "Site_Description": key[code],
            "Language_ID": DIALECT_PREFIX + code.lower(),
            "Base_Language": language,
            "Source": citation,
        }

        if raw == MISSING_MARKER or DIACRITIC_ONLY.match(raw):
            reason = (
                "the source prints an en dash: no word was elicited at this site"
                if raw == MISSING_MARKER
                else "the source prints a stray combining mark and no word"
            )
            audit.append({
                **shared, "Status": "missing", "Reason": reason, "Entry_Key": "",
                "Response_Index": "", "Response_Count": "0", "Response": "",
                "House_Form": "", "Tags": "",
            })
            continue

        repaired = raw.replace(DOTLESS_I, "i")
        responses = [
            unicodedata.normalize("NFC", part.strip())
            for part in re.split(f"[{RESPONSE_SEPARATORS}]", repaired)
        ]
        if not all(responses):
            raise ValueError(f"empty response in {raw!r} ({citation})")
        for position, response in enumerate(responses, start=1):
            # Convert exactly as make_cldf.py will: the installed CSV is NFC, so the
            # profile sees precomposed letters.  It covers both normalizations.
            house = unicodedata.normalize(
                "NFC",
                convert(response, column="IPA").replace(" ", "").replace("#", " "),
            )
            if "�" in house:
                raise ValueError(
                    f"conversion/beine-bhatri.txt does not cover {response!r} ({citation})"
                )
            # The under-tick on a vowel and the barred o have no established value in this
            # source; keep the row visible for review instead of guessing.
            tags = "uncertain" if re.search(r"[aeiou]̩|ɵ", response) else ""
            entry_key = f"{ENTRY_PREFIX}{code}:{record['Concept']}:{position}"
            forms.append([
                DIALECT_PREFIX + code.lower(),  # Language_ID
                "",                             # Parameter_ID: the source claims no etymology
                response,                       # Form (the profile converts it at build time)
                record["Gloss"],                # Gloss
                "",                             # Native
                response,                       # Phonemic: Beine's own transcription
                "",                             # Notes
                citation,                       # Source
                "",                             # Cognateset
                "",                             # Etymology
                entry_key,                      # Entry_Key
                "",                             # Variant_Of_Key
                "",                             # Borrowed_From_Key
                "",                             # Derivation_Parent_Keys
                tags,                           # Tags
            ])
            audit.append({
                **shared,
                "Status": "ingested",
                "Reason": (
                    "single response"
                    if len(responses) == 1
                    else f"response {position} of {len(responses)} printed in one cell"
                ),
                "Entry_Key": entry_key,
                "Response_Index": str(position),
                "Response_Count": str(len(responses)),
                "Response": response,
                "House_Form": house,
                "Tags": tags,
            })

    return forms, audit, notes


AUDIT_COLUMNS = [
    "Status", "Reason", "Entry_Key", "Site_Code", "Site_Description", "Language_ID",
    "Base_Language", "Concept", "Gloss", "Printed_Page", "Column", "Response_Index",
    "Response_Count", "Raw_Cell", "Response", "House_Form", "Tags", "Source",
]


def write_audit(audit: list[dict[str, str]]) -> None:
    with AUDIT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        writer.writerows(audit)


def write_sample(audit: list[dict[str, str]]) -> None:
    """Write the seeded raw-versus-parsed sample used for the manual 20-record audit."""
    picks = random.Random(20260825).sample(audit, 20)
    picks.sort(key=lambda row: (int(row["Concept"]), LECTS.index(row["Site_Code"])))
    with SAMPLE.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        writer.writerows(picks)


def write_manifest(forms, audit, notes) -> None:
    statuses: dict[str, int] = {}
    per_site: dict[str, int] = {}
    for row in audit:
        statuses[row["Status"]] = statuses.get(row["Status"], 0) + 1
        if row["Status"] == "ingested":
            per_site[row["Site_Code"]] = per_site.get(row["Site_Code"], 0) + 1
    manifest = {
        "source_key": SOURCE,
        "title": (
            "A Sociolinguistic Survey of the Bhatri-speaking Communities of Central India "
            "(SIL Electronic Survey Report 2017-005)"
        ),
        "url": "https://www.sil.org/resources/archives/71330",
        "fieldwork": "February--November 1989; published May 2017",
        "pdf": {"path": str(PDF), "sha256": PDF_SHA256, "pages": 95},
        "included": {
            "appendix": "Appendix A: Wordlists",
            "pdf_pages": f"{FIRST_PAGE + 1}--{LAST_PAGE + 1}",
            "printed_pages": f"{FIRST_PAGE - PAGE_OFFSET}--{LAST_PAGE - PAGE_OFFSET}",
            "sites": EXPECTED_SITES,
            "concepts": EXPECTED_CONCEPTS,
            "source_cells": EXPECTED_CELLS,
            "audit_rows": len(audit),
            "installed_rows": len(forms),
            "statuses": statuses,
            "ingested_per_site": per_site,
            "tagged_uncertain": sum(1 for row in forms if row[14]),
        },
        "excluded": [
            "the report body (printed pages 1--9): prose, intelligibility and bilingualism "
            "tables, and recommendations, which assert no lexical attestation",
            "Appendix B (printed pages 34--65): the interlinearised narrative texts and their "
            "comprehension questions, which are running text rather than elicited lexemes",
            "Appendix C (printed pages 66--88): recorded-text-test score sheets",
            "the References section (printed page 89)",
        ],
        "repairs": notes,
        "installed_file": str(INSTALLED.relative_to(ROOT)),
        "audit_file": str(AUDIT.relative_to(ROOT)),
        "profile": str(PROFILE.relative_to(ROOT)),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def dialect_rows() -> list[list[str]]:
    """Return cldf/dialects.csv rows for the twelve survey sites (coordinates filled later)."""
    _records, key, _notes = extract()
    rows = []
    for code in LECTS:
        language, clade, locality, _description = SITES[code]
        dialect_id = DIALECT_PREFIX + code.lower()
        name = f"{locality} ({code})"
        rows.append([
            dialect_id,
            f"dialect:{quote(language, safe='')}:{quote(dialect_id, safe='')}:{quote(name, safe='')}",
            language,
            dialect_id,
            name,
            "",  # Glottocode: Beine's survey points are not Glottolog languoids
            "",  # Latitude
            "",  # Longitude
            clade,
            f"{key[code]}; Beine (2017) survey site {code}",
            "",  # Quality
        ])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true", help="write the installed source CSV")
    parser.add_argument("--dialects", action="store_true", help="print cldf/dialects.csv rows")
    args = parser.parse_args()

    require_pdf()
    if args.dialects:
        csv.writer(sys.stdout).writerows(dialect_rows())
        return

    forms, audit, notes = build()
    target = INSTALLED if args.install else PREVIEW
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(forms)
    write_audit(audit)
    write_sample(audit)
    write_manifest(forms, audit, notes)

    ingested = sum(1 for row in audit if row["Status"] == "ingested")
    missing = sum(1 for row in audit if row["Status"] == "missing")
    print(
        f"Wrote {len(forms)} forms from {EXPECTED_CELLS} printed cells "
        f"({ingested} elicited responses, {missing} cells with no word) "
        f"across {EXPECTED_SITES} survey sites to {target}"
    )
    for note in notes:
        print(f"  repair: {note}")


if __name__ == "__main__":
    main()
