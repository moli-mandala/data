#!/usr/bin/env python3
"""Extract every Kalkoti form Liljegren's 2013 article prints.

Liljegren, Henrik. 2013. "Notes on Kalkoti: A Shina Language with Strong
Kohistani Influences." *Linguistic Discovery* 11(1): 129-160. The article is
open access at Dartmouth (doi 10.1349/PS1.1537-0852.A.423) and carries a real
Acrobat text layer, so no OCR is involved.

This importer supersedes the hand-entered 2022 snapshot of the same article.
The installed path and the ``kalkoti`` citation key are unchanged, so every
form ID, alias and hand-assigned CDIAL etymology survives the re-extraction:
``curated.py`` re-attaches the Parameter_ID values from the previous installed
CSV and fails loudly if the re-extraction drops one of them.

Extraction is keyed on the article's typeset structure rather than on its
reading order:

* Every lexical table is a fixed grid. ``TABLES`` records each table's PDF
  page, vertical band and column x-ranges, plus the number of body rows the
  printed table has. A table whose extraction does not produce exactly that
  many rows raises, so a silent column- or page-break error cannot pass.
* Characters are read in content-stream order, not sorted by position.
  Kalkoti tone is written with combining grave and acute accents that Acrobat
  emits on their own baseline a couple of points above the letters they mark;
  sorting by position detaches them from their vowel or moves them onto the
  wrong one, while stream order reproduces the printed word exactly.
* Numbered interlinear examples pair a form line with the gloss line beneath
  it by shared x-position, and small-capital category labels are folded back
  into the gloss cell above them.

Only the Kalkoti column is installed. The Palula, Gawri, Sawi and Shina
comparanda that the article quotes from Baart (1997, 1999a), Buddruss (1967),
Liljegren (2008) and Schmidt & Kohistani (2008) are secondary citations of
works that Jambu ingests separately; they are retained verbatim in the audit's
``Raw_Context`` and are not installed as forms.

Run from ``data/``:

    uv run python data/other/forms/raw_data/liljegren_kalkoti_2013.py --install
    uv run python data/other/forms/raw_data/liljegren_kalkoti_2013.py --pdf kalkoti.pdf
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


SOURCE_ID = "kalkoti"
SNAPSHOT_DATE = "2026-08-25"
PDF_SHA256 = "7eb1d21fb075e8c65b03924ba04b9bea90c2249b61aef56756f47e9f0c8f20c1"
PDF_SHA512 = (
    "a20b130999cfb82f6f2a2b0aa742dda22a2283708944192043e0b7f408310e56"
    "dcbffaaaa001ddf00e9e849fe230174f73f7dee590e53ba129210d2572b817df"
)
PDF_PAGES = 33
# PDF page 2 carries the printed page number 129.
PRINTED_PAGE_OFFSET = 127
LANGUAGE_ID = "Kalk"
DIALECT_TAG = "dialect:Kalk:HKAT-xka:Kalkot%20%28HKAT%29"

ROOT = Path(__file__).resolve().parents[4]
RAW_DIR = ROOT / "data/other/forms/raw_data"
FORM_OUTPUT = ROOT / "data/other/forms/20220913-kalkoti.csv"
AUDIT_OUTPUT = RAW_DIR / "20260825-liljegren-kalkoti-audit.csv"
SAMPLE_OUTPUT = RAW_DIR / "20260825-liljegren-kalkoti-sample.csv"
MANIFEST_OUTPUT = RAW_DIR / "20260825-liljegren-kalkoti-manifest.json"
RAW_SNAPSHOT = RAW_DIR / "20260825-liljegren-kalkoti-extract.psv"

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
    "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
    "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "Snapshot_Date", "Unit_ID", "Region", "PDF_Page", "Printed_Page", "Locator",
    "Raw_Form", "Raw_Gloss", "Raw_Context", "Status", "Reason", "Final_Form",
    "Final_Gloss", "Final_Phonemic", "Final_Tags", "Final_Parameter", "Emitted_Key",
    "Merged_Into", "Source", "Record_SHA256",
]

OPEN_Q, CLOSE_Q = "‘", "’"


# --------------------------------------------------------------------------
# PDF geometry
# --------------------------------------------------------------------------

def _chars(page) -> list[dict]:
    """Read a page in content-stream order, keeping combining marks attached.

    Kalkoti tone is written with combining grave and acute accents. Acrobat
    emits each accent as its own glyph on a baseline two points above the
    letters, and with an x-origin that sits over the *following* letter, so any
    position sort silently moves the tone onto the wrong vowel. The content
    stream, by contrast, still holds the logical order, so a combining mark is
    simply given the position of the character it follows.
    """
    out: list[dict] = []
    for char in page.chars:
        text = char["text"]
        if text and all(unicodedata.combining(ch) for ch in text) and out:
            out.append({**out[-1], "text": text})
            continue
        out.append({"text": text, "x0": char["x0"], "top": char["top"]})
    return out


def _lines(chars: list[dict], top: float, bottom: float, gap: float = 8.0):
    """Split a vertical band into visual lines, preserving stream order."""
    band = [c for c in chars if top <= c["top"] <= bottom]
    lines: list[list[dict]] = []
    current: list[dict] = []
    baseline: float | None = None
    for char in sorted(band, key=lambda c: (c["top"], c["x0"])):
        # Compare against the previous character's baseline rather than the
        # first one in the line: a wrapped gloss cell sits below the form it
        # belongs to, so a line is a run of baselines each close to the last.
        if baseline is None or char["top"] - baseline <= gap:
            current.append(char)
            baseline = char["top"]
        else:
            lines.append(current)
            current, baseline = [char], char["top"]
    if current:
        lines.append(current)
    # Acrobat emits runs of positioning whitespace between paragraphs; a line
    # made only of those would offset the form/gloss pairing of an example.
    return [_restream(chars, line) for line in lines
            if any(not c["text"].isspace() for c in line)]


def _restream(chars: list[dict], line: list[dict]) -> list[dict]:
    """Return the characters of one visual line back in content-stream order."""
    wanted = Counter((c["text"], round(c["x0"], 2), round(c["top"], 2)) for c in line)
    out = []
    for char in chars:
        key = (char["text"], round(char["x0"], 2), round(char["top"], 2))
        if wanted[key]:
            wanted[key] -= 1
            out.append(char)
    return out


def _cell(line: list[dict], lo: float, hi: float) -> str:
    text = "".join(c["text"] for c in line if lo <= c["x0"] < hi)
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


# --------------------------------------------------------------------------
# Table geometry
# --------------------------------------------------------------------------
# ``bands`` are (pdf page, top, bottom) rectangles holding the table body;
# ``columns`` are (name, x0, x1) ranges; ``rows`` is the number of body rows
# the printed table has, asserted after extraction.

C4 = [("kalkoti", 70, 190), ("palula", 190, 305), ("gawri", 305, 400), ("gloss", 400, 600)]
C5 = [("kalkoti", 65, 180), ("palula", 180, 295), ("gawri", 295, 410), ("gloss", 410, 600)]

GRIDS: dict[str, dict] = {
    "t1": {"table": 1, "rows": 24, "columns": C4,
           "bands": [(4, 550, 700), (5, 65, 320)]},
    "t2": {"table": 2, "rows": 13, "columns": C4,
           "bands": [(5, 600, 660), (6, 65, 240)]},
    "t3": {"table": 3, "rows": 5,
           "columns": [("kalkoti", 65, 225), ("structure", 225, 385), ("gloss", 385, 600)],
           "bands": [(6, 440, 525)]},
    "t4": {"table": 4, "rows": 6, "columns": C4,
           "bands": [(6, 670, 710), (7, 65, 135)]},
    "t5": {"table": 5, "rows": 12, "columns": C5, "bands": [(7, 228, 430)]},
    "t6": {"table": 6, "rows": 1, "columns": C5, "bands": [(7, 536, 552)]},
    "t8": {"table": 8, "rows": 6, "columns": C4, "bands": [(10, 82, 185)]},
    "t9": {"table": 9, "rows": 4,
           "columns": [("kohistani-shina", 65, 155), ("palula", 155, 250),
                       ("kalkoti", 250, 345), ("gawri", 345, 425), ("gloss", 425, 600)],
           "bands": [(10, 482, 550)]},
    "t13": {"table": 13, "rows": 8,
            "columns": [("tone", 65, 150), ("kalkoti", 150, 270),
                        ("palula", 270, 385), ("oia", 385, 600)],
            "bands": [(17, 372, 510)]},
    "t14": {"table": 14, "rows": 8,
            "columns": [("person", 65, 215), ("nom", 215, 295), ("obl1", 295, 380),
                        ("obl2", 380, 455), ("gen", 455, 600)],
            "bands": [(18, 84, 218)]},
    "t16": {"table": 16, "rows": 9,
            "columns": [("sg", 65, 195), ("pl", 195, 350), ("gloss", 350, 600)],
            "bands": [(20, 380, 530)]},
    "t17": {"table": 17, "rows": 3,
            "columns": [("label", 65, 180), ("l-verb", 180, 295),
                        ("t-verb", 295, 410), ("suppletive", 410, 600)],
            "bands": [(22, 84, 135)]},
    "t18": {"table": 18, "rows": 4,
            "columns": [("agr", 65, 180), ("kalkoti", 180, 295),
                        ("palula", 295, 410), ("sawi", 410, 600)],
            "bands": [(23, 170, 260)]},
    "t19": {"table": 19, "rows": 9,
            "columns": [("gloss", 65, 180), ("kalkoti", 180, 295),
                        ("palula", 295, 410), ("sawi", 410, 600)],
            "bands": [(24, 405, 610)]},
}

# Table 11 prints two independent half-tables side by side.
T11_HALVES = [
    [("vowel", 65, 100), ("ipa", 100, 160), ("gloss", 160, 285)],
    [("vowel", 285, 330), ("ipa", 330, 405), ("gloss", 405, 600)],
]
# Table 12 is read column by column: each melody heads a list of items.
T12_COLUMNS = [
    ("1", "high level", 65, 160), ("2", "low level", 160, 245),
    ("3", "low-rising", 245, 350), ("4", "high-rising", 350, 465),
    ("5", "high-falling", 465, 600),
]
# Table 20 pairs an aspect row with a tense-marking column; each cell holds a
# category label and then one form-plus-gloss pair per line.
T20_COLUMNS = [("unmarked", 175, 360), ("marked", 360, 600)]
T20_ROWS = [("imperfective", 320, 390), ("perfective", 390, 455)]


def _grid(page_chars: dict[int, list[dict]], spec: dict) -> list[dict]:
    """Read one printed table as a list of row dictionaries."""
    rows: list[dict] = []
    for page, top, bottom in spec["bands"]:
        for line in _lines(page_chars[page], top, bottom):
            cells = {name: _cell(line, lo, hi) for name, lo, hi in spec["columns"]}
            if not any(cells.values()):
                continue
            cells["pdf_page"] = str(page)
            rows.append(cells)
    return rows


def _halves(page_chars, page: int, top: float, bottom: float) -> list[dict]:
    """Read Table 11, which prints two independent half-tables side by side."""
    rows = []
    for half, columns in enumerate(T11_HALVES, start=1):
        for line in _lines(page_chars[page], top, bottom):
            cells = {name: _cell(line, lo, hi) for name, lo, hi in columns}
            if not all(cells.values()):
                continue
            rows.append({**cells, "half": str(half), "pdf_page": str(page)})
    return rows


def _melodies(page_chars, page: int, top: float, bottom: float) -> list[dict]:
    """Read Table 12, whose five tone melodies are columns, not rows."""
    rows = []
    for melody, name, lo, hi in T12_COLUMNS:
        for line in _lines(page_chars[page], top, bottom):
            cell = _cell(line, lo, hi)
            if not cell:
                continue
            gloss = _gloss(cell)
            form = _SLASHED.search(cell)
            if not form:
                continue
            rows.append({"melody": melody, "melody_name": name,
                         "ipa": form.group(0).replace(" ", ""),
                         "gloss": gloss, "pdf_page": str(page)})
    return rows


def _tma(page_chars, page: int) -> list[dict]:
    """Read the Kalkoti cells of Table 20 (tense/aspect intersection)."""
    rows = []
    for aspect, top, bottom in T20_ROWS:
        for name, lo, hi in T20_COLUMNS:
            label = ""
            for line in _lines(page_chars[page], top, bottom):
                cell = _cell(line, lo, hi)
                if not cell:
                    continue
                if cell.endswith(":"):
                    label = cell.rstrip(":").strip()
                    continue
                gloss = _gloss(cell)
                if not gloss:
                    continue
                rows.append({
                    "aspect": aspect, "tense": name, "category": label,
                    "form": _before_gloss(cell), "gloss": gloss,
                    "pdf_page": str(page),
                })
    return rows


_SLASHED = re.compile(r"/[^/]+/")


def _gloss(cell: str) -> str:
    """Return the quoted gloss in a cell.

    The article uses U+2019 both as the closing quotation mark and as the
    apostrophe of an English possessive, so a non-greedy match truncates
    'father's sister' to 'father'. The gloss therefore runs from the first
    opening mark to the last closing one.
    """
    if OPEN_Q not in cell or CLOSE_Q not in cell:
        return ""
    return cell[cell.index(OPEN_Q) + 1 : cell.rindex(CLOSE_Q)].strip()


def _before_gloss(cell: str) -> str:
    return cell[: cell.index(OPEN_Q)].strip() if OPEN_Q in cell else cell.strip()


# --------------------------------------------------------------------------
# Numbered interlinear examples
# --------------------------------------------------------------------------

EXAMPLE_NUMBER = re.compile(r"^\((\d+)\)")
# Example (9) is Biori Palula, quoted for contrast with the Kalkoti example above it.
EXAMPLE_LANGUAGES = {"9": "Palula"}


def _words(line: list[dict]) -> list[dict]:
    """Split one visual line into words, keeping each word's left edge."""
    words: list[dict] = []
    for char in line:
        if char["text"].isspace():
            if words and words[-1]["text"]:
                words.append({"text": "", "x0": char["x0"]})
            continue
        if not words:
            words.append({"text": "", "x0": char["x0"]})
        if not words[-1]["text"]:
            words[-1]["x0"] = char["x0"]
        words[-1]["text"] += char["text"]
    return [
        {"text": unicodedata.normalize("NFC", w["text"]), "x0": w["x0"]}
        for w in words if w["text"]
    ]


def _align(forms: list[dict], glosses: list[dict]) -> list[tuple[str, str]]:
    """Attach each gloss word to the form word standing above it.

    A gloss is typeset two points to the right of its form, and a category
    label in small capitals is set below the gloss it belongs to; both are
    resolved by taking the nearest form whose left edge is not to the right.
    """
    pairs = [[form["text"], []] for form in forms]
    if not pairs:
        return []
    for gloss in glosses:
        index = 0
        for position, form in enumerate(forms):
            if gloss["x0"] >= form["x0"] - 4:
                index = position
        pairs[index][1].append(gloss["text"])
    return [(form, " ".join(parts)) for form, parts in pairs]


def _examples(page_chars: dict[int, list[dict]]) -> list[dict]:
    """Read every numbered interlinear example in the article."""
    records: list[dict] = []
    for page in sorted(page_chars):
        lines = _lines(page_chars[page], 0, 800)
        index = 0
        while index < len(lines):
            head = _cell(lines[index], 0, 700)
            match = EXAMPLE_NUMBER.match(head)
            if not match:
                index += 1
                continue
            number = match.group(1)
            block, cursor = [], index
            # An example runs until its free translation, which is the first
            # line of the block that opens with a quotation mark.
            while cursor < len(lines):
                text = _cell(lines[cursor], 0, 700)
                block.append(lines[cursor])
                if text.lstrip("(0123456789) ").startswith(OPEN_Q):
                    break
                cursor += 1
            # A long free translation wraps onto further lines; read on until
            # its closing quotation mark so the audit keeps the whole sentence.
            tail = cursor
            while tail < len(lines) and CLOSE_Q not in _cell(lines[tail], 0, 700):
                tail += 1
            translation = _gloss(" ".join(
                _cell(lines[n], 0, 700) for n in range(cursor, min(tail + 1, len(lines)))
            ))
            for pair in range(0, len(block) - 1, 2):
                forms = [w for w in _words(block[pair]) if not EXAMPLE_NUMBER.match(w["text"])]
                glosses = _words(block[pair + 1])
                for position, (form, gloss) in enumerate(_align(forms, glosses), start=1):
                    records.append({
                        "example": number, "line": str(pair // 2 + 1),
                        "position": str(position), "form": form, "gloss": gloss,
                        "translation": translation, "pdf_page": str(page),
                        "language": EXAMPLE_LANGUAGES.get(number, LANGUAGE_ID),
                    })
            index = max(cursor, tail) + 1
    return records


# --------------------------------------------------------------------------
# Source IPA -> the article's own broad transcription
# --------------------------------------------------------------------------
# Liljegren prints some material in IPA (Tables 8, 11, 12 and the phonetic
# prose of pp. 135-136) and the rest in the broad transcription customary among
# Shina scholars. Tables 7 and 10 state the correspondence between the two
# notations explicitly, so the IPA can be rewritten into the same broad
# transcription the other tables use. That keeps one lexeme in one row with the
# printed IPA preserved in Phonemic, instead of splitting ``pitri`` from
# ``/pitri/``. Only mappings the article itself gives are listed here.

IPA_TO_BROAD: list[tuple[str, str]] = [
    # Table 10: vowel quality and length. Several qualities share one symbol in
    # the broad transcription; the article's chart licenses each merger.
    ("iː", "ii"), ("uː", "uu"), ("eː", "ee"), ("ɛː", "ee"), ("oː", "oo"),
    ("oˑ", "oo"), ("ɔː", "oo"), ("æː", "ää"), ("aː", "ää"), ("ɑː", "aa"),
    ("ɒː", "aa"), ("ɪ", "i"), ("ʊ", "u"), ("æ", "ä"), ("ɛ", "ä"), ("ə", "ä"),
    ("ɑ", "a"), ("ɔ", "a"),
    # Table 7: consonants, longest sequence first.
    ("ʈʂʰ", "c̣h"), ("ʈʂ", "c̣"), ("ʨʰ", "čh"), ("ʨ", "č"), ("ʥ", "ǰ"),
    ("pʰ", "ph"), ("tʰ", "th"), ("ʈʰ", "ṭh"), ("kʰ", "kh"),
    ("ʈ", "ṭ"), ("ɖ", "ḍ"), ("ɳ", "ṇ"), ("ʂ", "ṣ"), ("ɕ", "š"),
    ("ɽ", "ṛ"), ("ʋ", "w"), ("j", "y"), ("ʦ", "ts"), ("ɾ", "r"),
]
# The article writes an unstable or optional segment in parentheses. A
# parenthesised (ʔ) is the prosodic glottal element of melodies 3 and 4 and is
# a property of the tone, not a segment, so it is dropped; every other
# parenthesised segment is a real consonant that some speakers drop, and is
# kept, exactly as in the article's own broad-transcription spellings.
GLOTTAL = "(ʔ)"
# Combining grave and acute, the article's tone marks.
TONE_MARKS = {"\u0300", "\u0301"}


def toneless(form: str) -> str:
    """The same word without its tone marking."""
    return unicodedata.normalize(
        "NFC",
        "".join(c for c in unicodedata.normalize("NFD", form) if c not in TONE_MARKS),
    )


def to_broad(ipa: str) -> str:
    """Rewrite one bracketed IPA citation in the article's broad transcription."""
    text = unicodedata.normalize("NFD", ipa.strip().strip("/[]").strip())
    # Two citations, /ɑlɑ:l/ (p. 136 n. 5) and [ʨʊɳu:ns] (p. 136), set vowel
    # length with an ASCII colon instead of the length mark used everywhere
    # else. Repair that typographic slip before reading the string.
    text = unicodedata.normalize("NFC", text).replace(":", "ː")
    text = text.replace(GLOTTAL, "")
    text = text.replace("̥", "")  # voiceless diacritic on the final tap of [lɑːtɾ̥]
    text = text.replace("(", "").replace(")", "")
    out, index = [], 0
    while index < len(text):
        for source, target in IPA_TO_BROAD:
            if text.startswith(source, index):
                out.append(target)
                index += len(source)
                break
        else:
            out.append(text[index])
            index += 1
    return unicodedata.normalize("NFC", "".join(out))


# --------------------------------------------------------------------------
# Lexical citations in running prose and in footnotes
# --------------------------------------------------------------------------
# Each entry is (unit, pdf page, locator suffix, printed citation, gloss,
# tags, note). Where the article prints an alternation, the forms are given in
# printed order and the later ones become variants of the first. ``--pdf``
# checks that every citation below is still on the page it claims.

PROSE: list[dict] = [
    {"unit": "tea", "page": 8, "forms": ["/ʨeː/"], "gloss": "tea", "tags": "noun",
     "note": "cited as a minimal pair with čhee ‘ash’ for the palatal aspiration contrast"},
    {"unit": "ash", "page": 8, "forms": ["/ʨʰeː/"], "gloss": "ash", "tags": "noun",
     "note": "cited as a minimal pair with čee ‘tea’ for the palatal aspiration contrast"},
    {"unit": "vomit", "page": 8, "forms": ["[ʨʰəɖil]", "[ʨʰəɽil]"], "gloss": "to vomit",
     "tags": "verb", "note": "intervocalic [ɖ]~[ɽ] alternation"},
    {"unit": "heavy", "page": 8, "forms": ["[ʊɡʊr]", "[ʊɣʊr]"], "gloss": "heavy",
     "tags": "adj", "note": "intervocalic [ɡ]~[ɣ] alternation"},
    {"unit": "full", "page": 8, "forms": ["[ʨʊpʊʈ]", "[ʨʊbʊʈ]"], "gloss": "full",
     "tags": "adj", "note": "voicing alternation in the medial plosive"},
    {"unit": "cut", "page": 8, "forms": ["[krʊʈil]", "[krʊɖil]"], "gloss": "to cut",
     "tags": "verb", "note": "voicing alternation in the medial plosive"},
    {"unit": "small", "page": 8, "forms": ["[lʊkuʈ]", "[lʊɡuʈ]"], "gloss": "small",
     "tags": "adj", "note": "voicing alternation in the medial plosive"},
    {"unit": "dust", "page": 8, "forms": ["[dur]", "[duɽ]"], "gloss": "dust",
     "tags": "noun", "note": "word-final /r/~/ɽ/ alternation"},
    {"unit": "bone", "page": 8, "forms": ["[ɑɖ]", "[ɑɽ]"], "gloss": "bone",
     "tags": "noun", "note": "word-final /ɖ/~/ɽ/ alternation"},
    {"unit": "wife", "page": 9, "forms": ["/treːr/"], "gloss": "woman, wife", "tags": "noun f",
     "note": "cited for the stable word-initial /tr/ cluster"},
    {"unit": "village", "page": 9, "forms": ["/drɑːm/"], "gloss": "village", "tags": "noun",
     "note": "cited for the stable word-initial /dr/ cluster"},
    {"unit": "bad", "page": 9, "forms": ["[lɑːtɾ̥]", "[lɑːt]"], "gloss": "bad", "tags": "adj",
     "note": "word-final /tr/ cluster alternating with loss of the [r] segment"},
    {"unit": "writing", "page": 9, "forms": ["[ʨʊɳu:ns]"], "gloss": "was writing",
     "tags": "verb ipfv pret", "note": "printed as čuṇuun-s; cited for its final [ns] cluster"},
    {"unit": "path", "page": 9, "forms": ["[pɑːnd]", "[pɑːn]"], "gloss": "path", "tags": "noun",
     "note": "nasal-plosive cluster alternating with a plain nasal"},
    {"unit": "tie", "page": 9, "forms": ["[ɡɛɳɖil]", "[ɡɛɳil]"], "gloss": "to tie", "tags": "verb",
     "note": "nasal-plosive cluster alternating with a plain nasal"},
    {"unit": "snake", "page": 9, "forms": ["[nɑːŋɡ]", "[nɑːŋ]"], "gloss": "snake", "tags": "noun",
     "note": "nasal-plosive cluster alternating with a plain nasal"},
    {"unit": "halal", "page": 9, "footnote": "5", "forms": ["/ɑlɑ:l/"],
     "gloss": "lawful slaughter, halal", "tags": "noun loanword",
     "note": "Liljegren derives it from Urdu حلال, with the source /h/ dropped"},
    {"unit": "gain", "page": 9, "footnote": "5", "forms": ["/æːsil/"], "gloss": "gain",
     "tags": "noun loanword",
     "note": "Liljegren derives it from Urdu حاصل, with the source /h/ dropped"},
    {"unit": "brother", "page": 9, "footnote": "6", "forms": ["/drɑ/"], "gloss": "brother",
     "tags": "noun m", "note": "cited for the dental assimilation of the bilabial-plus-/r/ cluster"},
    {"unit": "eight", "page": 10, "forms": ["[eːʂ]"], "gloss": "eight", "tags": "num",
     "note": "cited as evidence that the final segment of a fricative-plosive coda is dropped"},
    {"unit": "eye", "page": 10, "forms": ["ic̣ii"], "gloss": "eye", "tags": "noun f",
     "note": "cited for its final open syllable, matching stressed-final Palula ac̣híi"},
    {"unit": "dative", "page": 17, "forms": ["maṭee", "maṭ"], "gloss": "to me",
     "tags": "pron personal 1sg dat", "note": "a separate first person singular dative form"},
]

# The article prints complex predicates (Table 3) and the compound numeral
# (Table 6) only as wholes, glossing the whole expression. Their non-verbal and
# numeral elements are real lexemes, and the 2022 snapshot recorded them
# separately; they are kept here as an explicitly editorial segmentation, with
# the parent expression cited as their locator. Table 6's own structure row,
# "3 - conj - 2 - 20", licenses the reading of tee and biš.
SEGMENTS: list[dict] = [
    {"unit": "aga", "parent": "t3:1", "page": 6, "form": "äɡa", "gloss": "rain",
     "tags": "noun"},
    {"unit": "work", "parent": "t3:2", "page": 6, "form": "traam", "gloss": "work",
     "tags": "noun"},
    {"unit": "sleep", "parent": "t3:3", "page": 6, "form": "niin", "gloss": "sleep",
     "tags": "noun"},
    {"unit": "flight", "parent": "t3:4", "page": 6, "form": "šiiš", "gloss": "flight",
     "tags": "noun uncertain"},
    {"unit": "cold", "parent": "t3:5", "page": 6, "form": "šidäl", "gloss": "cold",
     "tags": "noun"},
    {"unit": "and", "parent": "t6:43", "page": 7, "form": "tee", "gloss": "and",
     "tags": "conj"},
    {"unit": "twenty", "parent": "t6:43", "page": 7, "form": "biš", "gloss": "twenty",
     "tags": "num"},
]

# Four comparanda were typed into the 2022 snapshot of this file to give
# Jambu's own etyma e34-e36 a second member each, and to hold the Palula
# cognate of Table 9's 'worm'. Three of them are Palula dictionary forms rather
# than anything this article prints, and all four are already in Jambu's house
# transcription, so they are carried through verbatim: dropping them would
# leave e34-e36 with a single reflex apiece. make_cldf.py keeps rows of this
# file whose language is not Kalkoti on the preservation profile.
EDITORIAL_ANCHORS: list[list[str]] = [
    ["Phal", "e34", "cōṇṭō̂", "to write, embroider", "", "", "", "liljegren"],
    ["Phal", "3438", "krīmī̂", "worm", "", "", "", "kalkoti"],
    ["Phal", "e35", "tapō̌s", "question", "", "", "", "liljegren"],
    ["Phal", "e36", "típa", "now", "", "", "", "liljegren"],
]

# Footnote 15 lists the tone melodies of five polysyllabic words. These are the
# only polysyllabic forms in the article that carry its tone notation.
FOOTNOTE_15: list[tuple[str, str, str, str]] = [
    ("pitri", "0", "father’s brother", "noun m"),
    ("ḍä̀rin", "L", "earth", "noun"),
    ("ic̣ì", "L", "eye", "noun f"),
    ("lumaáṭ", "H", "tail", "noun"),
    ("bä̀kaál", "LH", "to kill", "verb tr"),
]

MELODY_NOTE = {
    "1": "high level tone (melody 1)", "2": "low level tone (melody 2)",
    "3": "low-rising tone (melody 3)", "4": "high-rising tone (melody 4)",
    "5": "high-falling tone (melody 5)",
}
TONE_NOTE = {
    "0": "no underlying tone", "L": "a low tone", "H": "a high tone",
    "LH": "a low-high tone",
}


# --------------------------------------------------------------------------
# Glosses and grammatical labels
# --------------------------------------------------------------------------

NUMERAL_GLOSS = {
    "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six",
    "7": "seven", "8": "eight", "9": "nine", "10": "ten", "11": "eleven",
    "12": "twelve", "20": "twenty", "43": "forty-three",
}
# Table 4 labels its rows by person and number only; Table 14 adds deixis.
PRONOUN_GLOSS = {
    "1 SG": "I", "2 SG": "you (singular)", "3 SG": "he, she, it",
    "1 PL": "we", "2 PL": "you (plural)", "3 PL": "they",
}
PRONOUN_TAGS = {
    "1 SG": "1sg", "2 SG": "2sg", "3 SG": "3sg",
    "1 PL": "1pl", "2 PL": "2pl", "3 PL": "3pl",
}
# Table 14's four columns; OBL I is the direct-object form and OBL II the
# ergative one, as the printed column headings state.
CASE_TAGS = {"nom": "nom", "obl1": "obl acc", "obl2": "obl erg", "gen": "gen"}
CASE_NAME = {"nom": "nominative", "obl1": "accusative", "obl2": "ergative", "gen": "genitive"}
DEIXIS_TAGS = {"near": "prox", "far": "dist"}
DEIXIS_GLOSS = {"near": "proximate", "far": "remote"}

# Table 17 names the two Kalkoti verb classes after Palula's L- and T-verbs.
VERB_CLASSES = {
    "l-verb": "Kalkoti-verb-class-L", "t-verb": "Kalkoti-verb-class-T",
    "suppletive": "Kalkoti-verb-class-suppletive",
}
STEM_TAGS = {"Non-perfective stem": "verb stem", "Perfective stem": "verb stem pfv"}

AGREEMENT_TAGS = {"MSG": "m sg", "MPL": "m pl", "FSG": "f sg", "FPL": "f pl"}
TMA_TAGS = {
    "Present Imperfective": "verb ipfv pres", "Past Imperfective": "verb ipfv pret",
    "Simple Past": "verb pfv pret", "Pluperfect": "verb pfv pp",
}

# Interlinear category labels, printed in small capitals.
INTERLINEAR_TAGS = {
    "SG": "sg", "PL": "pl", "M": "m", "F": "f", "MSG": "m sg", "MPL": "m pl",
    "FSG": "f sg", "FPL": "f pl", "NOM": "nom", "ACC": "acc", "OBL": "obl",
    "ERG": "erg", "GEN": "gen", "DAT": "dat", "IPFV": "ipfv", "PFV": "pfv",
    "PST": "pret", "PRS": "pres", "CV": "conjunctive-participle", "Q": "interr",
    "1SG": "1sg", "2SG": "2sg", "3SG": "3sg", "1PL": "1pl", "2PL": "2pl", "3PL": "3pl",
}
# Personal names and place names in the example sentences.
PROPER_NOUNS = {"ɡulus", "zumaan", "šilkin", "thäl-iǰ"}
LABEL = re.compile(r"^[A-Z0-9]+$")
# 'I' is the English first person pronoun, not a category label.
LEXICAL_CAPITALS = {"I"}
# The article states the gender of these two suppletive perfectives on p. 151.
UNIT_TAGS = {
    "t19:4": "m sg", "t19:4:a2": "f", "t19:9": "m sg", "t19:9:a2": "f",
}


def read_interlinear(gloss: str) -> tuple[str, list[str], list[str]]:
    """Split an interlinear gloss into its lexical part and its category labels.

    Object-language material is glossed with lower-case words joined by dots,
    while grammatical categories are printed in small capitals; a chunk written
    entirely in capitals is therefore a category label and everything else is
    part of the lexical meaning.
    """
    words, labels, unknown = [], [], []
    for chunk in re.split(r"[-=]", gloss):
        for piece in re.split(r"[.()]", chunk):
            piece = piece.strip()
            if not piece:
                continue
            if LABEL.match(piece) and piece not in LEXICAL_CAPITALS:
                if piece in INTERLINEAR_TAGS:
                    labels.extend(INTERLINEAR_TAGS[piece].split())
                else:
                    unknown.append(piece)
            else:
                words.append(piece)
    return " ".join(words), list(dict.fromkeys(labels)), unknown


# --------------------------------------------------------------------------
# One flat record list, snapshotted so the importer runs without the PDF
# --------------------------------------------------------------------------

SNAPSHOT_FIELDS = ["unit", "region", "pdf_page", "form", "gloss", "context"]


def _record(unit, region, page, form, gloss, **context) -> dict[str, str]:
    parts = "; ".join(f"{key}={value}" for key, value in context.items() if value)
    return {"unit": unit, "region": region, "pdf_page": str(page),
            "form": form, "gloss": gloss, "context": parts}


def extract(path: Path) -> list[dict[str, str]]:
    """Read every Kalkoti citation in the article straight from the PDF."""
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        if len(pdf.pages) != PDF_PAGES:
            raise ValueError(f"expected {PDF_PAGES} pages, found {len(pdf.pages)}")
        chars = {n: _chars(page) for n, page in enumerate(pdf.pages, start=1)}

    out: list[dict[str, str]] = []
    for key, spec in GRIDS.items():
        rows = _grid(chars, spec)
        if len(rows) != spec["rows"]:
            raise ValueError(
                f"table {spec['table']} extracted {len(rows)} rows, expected {spec['rows']}"
            )
        out.extend(READERS[key](rows, spec))

    for row in _halves(chars, 11, 355, 440):
        out.append(_record(
            f"t11:{row['half']}:{row['vowel']}", "t11", row["pdf_page"], row["ipa"],
            _gloss(row["gloss"]), vowel=row["vowel"],
        ))
    seen: Counter = Counter()
    for row in _melodies(chars, 15, 480, 635):
        seen[row["melody"]] += 1
        out.append(_record(
            f"t12:{row['melody']}:{seen[row['melody']]}", "t12", row["pdf_page"],
            row["ipa"], row["gloss"], melody=row["melody"], melody_name=row["melody_name"],
        ))
    for index, row in enumerate(_tma(chars, 25), start=1):
        out.append(_record(
            f"t20:{index}", "t20", row["pdf_page"], row["form"], row["gloss"],
            category=row["category"], aspect=row["aspect"], tense=row["tense"],
        ))
    for row in _examples(chars):
        out.append(_record(
            f"ex{row['example']}:{row['line']}:{row['position']}", "interlinear",
            row["pdf_page"], row["form"], row["gloss"], example=row["example"],
            language=row["language"], translation=row["translation"],
        ))
    return out


def serialise(records: list[dict[str, str]]) -> str:
    lines = ["|".join(SNAPSHOT_FIELDS)]
    for record in records:
        if any("|" in record[field] for field in SNAPSHOT_FIELDS):
            raise ValueError(f"a pipe character would break the snapshot: {record}")
        lines.append("|".join(record[field] for field in SNAPSHOT_FIELDS))
    return "\n".join(lines) + "\n"


def snapshot() -> list[dict[str, str]]:
    text = RAW_SNAPSHOT.read_text(encoding="utf-8").splitlines()
    return [dict(zip(SNAPSHOT_FIELDS, line.split("|"))) for line in text[1:]]


# --------------------------------------------------------------------------
# Per-table readers
# --------------------------------------------------------------------------

def _comparanda(row: dict, *names: str) -> dict[str, str]:
    return {name: row.get(name, "") for name in names}


def _read_t1(rows, spec):
    for index, row in enumerate(rows, start=1):
        yield _record(f"t1:{index}", "t1", row["pdf_page"], row["kalkoti"],
                      _gloss(row["gloss"]), **_comparanda(row, "palula", "gawri"))


def _read_t2(rows, spec):
    """Table 2 prints ``non-perfective (perfective)`` in one cell."""
    for index, row in enumerate(rows, start=1):
        forms = _paired(row["kalkoti"])
        glosses = _paired(_gloss(row["gloss"]))
        for aspect, form, gloss in zip(("ipfv", "pfv"), forms, glosses):
            yield _record(f"t2:{index}:{aspect}", "t2", row["pdf_page"], form, gloss,
                          aspect=aspect, printed=row["kalkoti"],
                          **_comparanda(row, "palula", "gawri"))


def _paired(cell: str) -> list[str]:
    """Split ``a (b)`` into ``[a, b]``."""
    match = re.match(r"^(.*?)\s*\((.*)\)$", cell.strip())
    return [match.group(1).strip(), match.group(2).strip()] if match else [cell.strip()]


def _read_t3(rows, spec):
    for index, row in enumerate(rows, start=1):
        yield _record(f"t3:{index}", "t3", row["pdf_page"], row["kalkoti"],
                      _gloss(row["gloss"]), structure=row["structure"])


def _read_t4(rows, spec):
    for row in rows:
        label = row["gloss"]
        forms = _paired(row["kalkoti"])
        for case, form in zip(("nom", "obl"), forms):
            yield _record(f"t4:{label.replace(' ', '').lower()}:{case}", "t4",
                          row["pdf_page"], form, label, case=case, person=label,
                          printed=row["kalkoti"], **_comparanda(row, "palula", "gawri"))


def _read_t5(rows, spec):
    for row in rows:
        yield _record(f"t5:{_gloss(row['gloss'])}", "t5", row["pdf_page"], row["kalkoti"],
                      _gloss(row["gloss"]), **_comparanda(row, "palula", "gawri"))


def _read_t6(rows, spec):
    for row in rows:
        yield _record("t6:43", "t6", row["pdf_page"], row["kalkoti"], _gloss(row["gloss"]),
                      structure="3 - conj - 2 - 20", **_comparanda(row, "palula", "gawri"))


def _read_t8(rows, spec):
    for index, row in enumerate(rows, start=1):
        yield _record(f"t8:{index}", "t8", row["pdf_page"], row["kalkoti"],
                      _gloss(row["gloss"]), **_comparanda(row, "palula", "gawri"))


def _read_t9(rows, spec):
    for index, row in enumerate(rows, start=1):
        yield _record(f"t9:{index}", "t9", row["pdf_page"], row["kalkoti"],
                      _gloss(row["gloss"]),
                      **_comparanda(row, "kohistani-shina", "palula", "gawri"))


def _read_t13(rows, spec):
    tone = ""
    for index, row in enumerate(rows, start=1):
        tone = row["tone"].split("(")[0].strip() or tone
        yield _record(f"t13:{index}", "t13", row["pdf_page"],
                      _before_gloss(row["kalkoti"]), _gloss(row["kalkoti"]),
                      tone=tone, palula=row["palula"], oia=row["oia"])


def _read_t14(rows, spec):
    person, deixis = "", ""
    for row in rows:
        label = row["person"]
        if label in DEIXIS_TAGS:
            deixis = label
        else:
            person, deixis = label.rsplit(" ", 1)[0] if label.endswith(("near", "far")) else label, ""
            if label.endswith(("near", "far")):
                person, deixis = label[:-4].strip(), label[-4:].strip()
        for case in ("nom", "obl1", "obl2", "gen"):
            for position, form in enumerate(_alternates(row[case]), start=1):
                unit = f"t14:{person.replace(' ', '').lower()}{':' + deixis if deixis else ''}:{case}"
                yield _record(unit + (f":a{position}" if position > 1 else ""), "t14",
                              row["pdf_page"], form, person, case=case, person=person,
                              deixis=deixis, printed=row[case])


def _alternates(cell: str) -> list[str]:
    return [part.strip() for part in cell.split(",") if part.strip()]


def _read_t16(rows, spec):
    for index, row in enumerate(rows, start=1):
        for number in ("sg", "pl"):
            for position, form in enumerate(_alternates(row[number]), start=1):
                yield _record(f"t16:{index}:{number}" + (f":a{position}" if position > 1 else ""),
                              "t16", row["pdf_page"], form, _gloss(row["gloss"]),
                              number=number, printed=row[number])


def _read_t17(rows, spec):
    header, *body = rows
    for row in body:
        for column in ("l-verb", "t-verb", "suppletive"):
            yield _record(f"t17:{column}:{row['label'].split()[0].lower()}", "t17",
                          row["pdf_page"], row[column], _gloss(header[column]),
                          verb_class=column, stem=row["label"])


def _read_t18(rows, spec):
    for row in rows:
        yield _record(f"t18:{row['agr'].lower()}", "t18", row["pdf_page"], row["kalkoti"],
                      "is/are sitting down", agreement=row["agr"],
                      **_comparanda(row, "palula", "sawi"))


def _read_t19(rows, spec):
    for index, row in enumerate(rows, start=1):
        for position, form in enumerate(_alternates(row["kalkoti"]), start=1):
            yield _record(f"t19:{index}" + (f":a{position}" if position > 1 else ""),
                          "t19", row["pdf_page"], form, _gloss(row["gloss"]),
                          printed=row["kalkoti"], **_comparanda(row, "palula", "sawi"))


READERS = {
    "t1": _read_t1, "t2": _read_t2, "t3": _read_t3, "t4": _read_t4, "t5": _read_t5,
    "t6": _read_t6, "t8": _read_t8, "t9": _read_t9, "t13": _read_t13, "t14": _read_t14,
    "t16": _read_t16, "t17": _read_t17, "t18": _read_t18, "t19": _read_t19,
}


# --------------------------------------------------------------------------
# Locators, tags and glosses per region
# --------------------------------------------------------------------------

TABLE_OF = {
    "t1": 1, "t2": 2, "t3": 3, "t4": 4, "t5": 5, "t6": 6, "t8": 8, "t9": 9,
    "t11": 11, "t12": 12, "t13": 13, "t14": 14, "t16": 16, "t17": 17, "t18": 18,
    "t19": 19, "t20": 20,
}
# Regions printed in IPA rather than in the article's broad transcription.
IPA_REGIONS = {"t8", "t11", "t12"}
REGION_TAGS = {
    "t1": "noun", "t2": "verb", "t3": "verb multiword-expression", "t4": "pron personal",
    "t5": "num", "t6": "num compound", "t14": "pron personal", "t17": "verb stem",
    "t18": "verb ipfv", "t19": "verb pfv", "t20": "verb",
}


def _context(record: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in record["context"].split("; "):
        if "=" in part:
            key, value = part.split("=", 1)
            out[key] = value
    return out


def printed_page(record: dict[str, str]) -> int:
    return int(record["pdf_page"]) + PRINTED_PAGE_OFFSET


def locator(record: dict[str, str]) -> str:
    """Build the CLDF locator for one record, keyed on the printed page."""
    page = f"p. {printed_page(record)}"
    region, context = record["region"], _context(record)
    if region in TABLE_OF:
        return f"{page}, table {TABLE_OF[region]}"
    if region == "segment":
        return f"{page}, table {TABLE_OF[context['parent'].split(':')[0]]}"
    if region == "interlinear":
        return f"{page}, example {context.get('example', '')}"
    if region == "prose":
        note = context.get("footnote")
        return f"{page}, n. {note}" if note else page
    if region == "fn15":
        return f"{page}, n. 15"
    return page


def interpret(record: dict[str, str]) -> dict:
    """Turn one raw record into its form, gloss, tags, notes and phonemic value."""
    region, context = record["region"], _context(record)
    raw = record["form"]
    gloss, tags, notes, etymology, unknown = record["gloss"], [], [], "", []
    parameter = ""

    if region in IPA_REGIONS or raw.startswith(("/", "[")):
        form, phonemic = to_broad(raw), raw.strip("/[]").strip()
    else:
        form, phonemic = raw, ""

    if region in REGION_TAGS:
        tags += REGION_TAGS[region].split()
    if region == "t2":
        tags.append(context["aspect"])
    elif region == "t4":
        gloss = PRONOUN_GLOSS[context["person"]]
        tags += [PRONOUN_TAGS[context["person"]], context["case"]]
    elif region in {"t5", "t6"}:
        gloss = NUMERAL_GLOSS[gloss]
    elif region == "t12":
        notes.append(MELODY_NOTE[context["melody"]])
    elif region == "t13":
        tone = context["tone"]
        notes.append(f"Liljegren analyses this word as carrying {TONE_NOTE[tone]}.")
        printed = re.search(r"\(T:\s*(\d+)\)", context.get("oia", ""))
        if printed:
            parameter = printed.group(1)
            etymology = (
                f"Liljegren compares Palula {context['palula']} and OIA "
                f"{context['oia'].split(' (T:')[0]} (Turner 1966: {parameter})."
            )
    elif region == "t14":
        gloss = PRONOUN_GLOSS[context["person"]]
        if context.get("deixis"):
            gloss = f"{gloss} ({DEIXIS_GLOSS[context['deixis']]})"
            tags.append(DEIXIS_TAGS[context["deixis"]])
        tags += [PRONOUN_TAGS[context["person"]], *CASE_TAGS[context["case"]].split()]
    elif region == "t16":
        tags += ["noun", context["number"]]
    elif region == "t17":
        tags += [VERB_CLASSES[context["verb_class"]], *STEM_TAGS[context["stem"]].split()]
        gloss = f"to {gloss}"
    elif region == "t18":
        tags += AGREEMENT_TAGS[context["agreement"]].split()
    elif region == "t20":
        tags += TMA_TAGS[context["category"]].split()
    elif region == "interlinear":
        gloss, labels, unknown = read_interlinear(record["gloss"])
        tags += labels
        # Pronouns in the examples are glossed by category alone; give them the
        # same lexical gloss the pronoun tables use.
        if not gloss:
            person = next((f"{l[0]} {l[1:].upper()}" for l in labels
                           if l in PRONOUN_TAGS.values()), "")
            gloss = PRONOUN_GLOSS.get(person, "")
            if gloss:
                tags.append("pron")
        if raw.lower().rstrip(".,") in PROPER_NOUNS:
            tags.append("proper-noun")
        notes.append(f"cited in example ({context['example']}): ‘{context['translation']}’")
    elif region in {"prose", "fn15", "segment"}:
        tags += context.get("tags", "").split()
        if context.get("note"):
            notes.append(context["note"])
        if context.get("tone"):
            notes.append(f"Liljegren analyses this word as carrying {TONE_NOTE[context['tone']]}.")

    if record["unit"] in UNIT_TAGS:
        tags += UNIT_TAGS[record["unit"]].split()

    # The article writes morpheme boundaries with hyphens and clitic boundaries
    # with '='; the headword is the unsegmented host word, without the clitic.
    # A zero morph is a gloss-line convention, not part of the written word.
    form = form.rstrip(".,").replace("-Ø", "")
    # Table 13 and footnote 15 write tone with combining grave and acute
    # accents; conversion/kalkoti.txt carries them into house
    # transcription. Most tables print the same words without tone, so the fold
    # below matches on the tone-free shape and keeps the marked spelling.
    if "=" in form:
        notes.append(f"printed as {form}, with the polar-question clitic =ää")
        form = form.split("=", 1)[0]
    # A parenthesised segment is one that some speakers drop. The article's own
    # broad spellings write it out, so it is kept and the printed spelling with
    # its parentheses is recorded.
    if "(" in form:
        notes.append(f"printed as {form}; the parenthesised segment is optional")
        form = form.replace("(", "").replace(")", "")
    form = form.replace("-", "")
    return {
        "form": unicodedata.normalize("NFC", form),
        "phonemic": unicodedata.normalize("NFC", phonemic),
        "gloss": gloss.replace("’", "'").strip(),
        "tags": list(dict.fromkeys(tags)),
        "notes": "; ".join(notes), "etymology": etymology,
        "parameter": parameter, "unknown_labels": unknown,
    }


def prose_records() -> list[dict[str, str]]:
    """The prose and footnote citations, in the same shape as the snapshot."""
    out = []
    for entry in PROSE:
        for position, form in enumerate(entry["forms"], start=1):
            unit = f"prose:{entry['unit']}" + (f":a{position}" if position > 1 else "")
            out.append(_record(
                unit, "prose", entry["page"], form, entry["gloss"],
                tags=entry["tags"], note=entry.get("note", ""),
                footnote=entry.get("footnote", ""),
                variant_of=f"prose:{entry['unit']}" if position > 1 else "",
                printed=" ~ ".join(entry["forms"]) if len(entry["forms"]) > 1 else "",
            ))
    for entry in SEGMENTS:
        out.append(_record(
            f"seg:{entry['unit']}", "segment", entry["page"], entry["form"],
            entry["gloss"], tags=entry["tags"], parent=entry["parent"],
            note=(f"the article prints this element only inside the expression it "
                  f"cites in table {TABLE_OF[entry['parent'].split(':')[0]]}; the "
                  f"segmentation is editorial"),
        ))
    for index, (form, tone, gloss, tags) in enumerate(FOOTNOTE_15, start=1):
        out.append(_record(f"fn15:{index}", "fn15", 17, form, gloss, tone=tone, tags=tags))
    return out


def records() -> list[dict[str, str]]:
    return snapshot() + prose_records()


# --------------------------------------------------------------------------
# Assembling the installed rows
# --------------------------------------------------------------------------

def _union(current: str, addition: str, joiner: str) -> str:
    parts = [p for p in (current.split(joiner) if current else []) if p]
    for part in (addition.split(joiner) if addition else []):
        if part and part not in parts:
            parts.append(part)
    return joiner.join(parts)


def collapse(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, str]]:
    """Merge repeated citations of one lexeme into a single installed row.

    The article cites many words more than once - ``pitri`` appears in the
    kinship table, in the consonant-cluster table and in footnote 15 - and each
    citation contributes a different part of the record. Rows that agree on
    language, form, gloss and complete lexical analysis are merged; their
    citations, tags, notes and phonemic values are unioned and the surviving
    Entry_Key is the first one printed.
    """
    survivors: dict[tuple[str, ...], dict[str, str]] = {}
    def key_of(row):
        return (row["Language_ID"], toneless(row["Form"]), row["Gloss"])

    aliases: dict[str, str] = {}
    order: list[tuple[str, ...]] = []
    for row in rows:
        key = key_of(row)
        if key in survivors:
            target = survivors[key]
            aliases[row["Entry_Key"]] = target["Entry_Key"]
            # Only some tables mark tone. When one citation of a word carries it
            # and another does not, keep the marked spelling for the whole row.
            if row["Form"] != toneless(row["Form"]) and target["Form"] == toneless(target["Form"]):
                target["Form"] = row["Form"]
            target["Source"] = _union(target["Source"], row["Source"], ";")
            target["Tags"] = _union(target["Tags"], row["Tags"], " ")
            target["Notes"] = _union(target["Notes"], row["Notes"], "; ")
            target["Phonemic"] = _union(target["Phonemic"], row["Phonemic"], "; ")
            target["Etymology"] = target["Etymology"] or row["Etymology"]
            target["Parameter_ID"] = target["Parameter_ID"] or row["Parameter_ID"]
            continue
        aliases[row["Entry_Key"]] = row["Entry_Key"]
        survivors[key] = dict(row)
        order.append(key)
    merged = [survivors[key] for key in order]
    for row in merged:
        if row["Variant_Of_Key"]:
            row["Variant_Of_Key"] = aliases.get(row["Variant_Of_Key"], row["Variant_Of_Key"])
            if row["Variant_Of_Key"] == row["Entry_Key"]:
                row["Variant_Of_Key"] = ""
    return merged, aliases


CURATED = RAW_DIR / "20260825-liljegren-kalkoti-etymology.csv"


def curated() -> dict[str, dict[str, str]]:
    """Hand-assigned CDIAL etymologies carried over from the 2022 snapshot.

    The 2022 ingest of this article was typed by hand straight into Jambu's own
    transcription and its Parameter_ID values are editorial decisions, not
    something the article prints. The crosswalk keys them to this importer's
    Entry_Key values so that re-extraction preserves every one of them.
    """
    if not CURATED.exists():
        return {}
    with CURATED.open(encoding="utf-8", newline="") as stream:
        return {row["Entry_Key"]: row for row in csv.DictReader(stream)}


def build() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    forms: list[dict[str, str]] = []
    audit: list[dict[str, str]] = []
    etymologies = curated()
    unit_keys: dict[str, str] = {}

    for record in records():
        unit, region = record["unit"], record["region"]
        context = _context(record)
        parsed = interpret(record)
        key = f"{SOURCE_ID}:{unit}"
        status, reason = "installed", ""

        if context.get("language", LANGUAGE_ID) != LANGUAGE_ID:
            status, reason = "skipped", (
                f"example ({context.get('example')}) is quoted in "
                f"{context['language']}, not Kalkoti"
            )
        elif not parsed["form"]:
            status, reason = "skipped", "the printed cell is empty"
        elif parsed["unknown_labels"]:
            status, reason = "installed", (
                "unrecognised interlinear label(s) "
                + ", ".join(parsed["unknown_labels"])
            )

        curated_row = etymologies.get(key, {})
        parameter = curated_row.get("Parameter_ID", "") or parsed["parameter"]
        notes = _union(parsed["notes"], curated_row.get("Notes", ""), "; ")
        etymology = parsed["etymology"]
        editorial = curated_row.get("Parameter_ID", "")
        if parsed["parameter"] and editorial and editorial != parsed["parameter"]:
            notes = _union(
                notes,
                f"Liljegren prints CDIAL {parsed['parameter']} for this word; "
                f"Jambu links it to {editorial} instead",
                "; ",
            )

        citation = f"{SOURCE_ID}[{locator(record)}]"
        tags = parsed["tags"]
        if status == "installed":
            unit_keys[unit] = key
            forms.append(dict(zip(FORM_FIELDS, [
                LANGUAGE_ID, parameter, parsed["form"], parsed["gloss"], "",
                parsed["phonemic"], notes, citation, "", etymology, key,
                f"{SOURCE_ID}:{context['variant_of']}" if context.get("variant_of") else "",
                "", "", " ".join(dict.fromkeys(tags)),
            ])))

        audit.append({
            "Snapshot_Date": SNAPSHOT_DATE, "Unit_ID": unit, "Region": region,
            "PDF_Page": record["pdf_page"], "Printed_Page": str(printed_page(record)),
            "Locator": locator(record), "Raw_Form": record["form"],
            "Raw_Gloss": record["gloss"], "Raw_Context": record["context"],
            "Status": status,
            "Reason": reason or "read from the PDF text layer and installed unchanged",
            "Final_Form": parsed["form"] if status == "installed" else "",
            "Final_Gloss": parsed["gloss"] if status == "installed" else "",
            "Final_Phonemic": parsed["phonemic"] if status == "installed" else "",
            "Final_Tags": " ".join(parsed["tags"]) if status == "installed" else "",
            "Final_Parameter": parameter if status == "installed" else "",
            "Emitted_Key": key if status == "installed" else "", "Merged_Into": "",
            "Source": citation,
            "Record_SHA256": hashlib.sha256(
                "|".join(record[f] for f in SNAPSHOT_FIELDS).encode()
            ).hexdigest(),
        })

    merged, aliases = collapse(forms)
    for row in merged:
        row["Tags"] = " ".join([*row["Tags"].split(), DIALECT_TAG])
    for row in audit:
        key = unit_keys.get(row["Unit_ID"])
        if key and aliases.get(key, key) != key:
            row["Merged_Into"] = aliases[key]
    missing = sorted(set(etymologies) - set(unit_keys.values()))
    if missing:
        raise ValueError(
            f"re-extraction dropped {len(missing)} curated etymologies; "
            f"examples: {missing[:5]}"
        )
    return merged, audit


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]], header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if header:
            writer.writeheader()
        writer.writerows(rows)


def digest(path: Path, algorithm: str) -> str:
    return hashlib.new(algorithm, path.read_bytes()).hexdigest()


def manifest(forms, audit) -> dict:
    return {
        "source_id": SOURCE_ID,
        "snapshot_date": SNAPSHOT_DATE,
        "bibliography": (
            "Liljegren, Henrik. 2013. Notes on Kalkoti: A Shina Language with Strong "
            "Kohistani Influences. Linguistic Discovery 11(1): 129-160."
        ),
        "doi": "10.1349/PS1.1537-0852.A.423",
        "acquisition": (
            "Open-access article PDF published by the Dartmouth College Library at "
            "journals.dartmouth.edu, article 423"
        ),
        "pdf_sha256": PDF_SHA256,
        "pdf_sha512": PDF_SHA512,
        "pdf_pages": PDF_PAGES,
        "pdf_redistributed": False,
        "rights": (
            "Copyright is held by the author; the journal is open access. Only the "
            "extracted linguistic facts are installed and the PDF is not checked in."
        ),
        "supersedes": (
            "the hand-typed 2022 snapshot of the same article at the same installed "
            "path; its editorial CDIAL assignments are carried forward through "
            "data/other/forms/raw_data/20260825-liljegren-kalkoti-etymology.csv"
        ),
        "extraction": {
            "method": (
                "deterministic pdfplumber extraction from the article's Acrobat text "
                "layer, read in content-stream order; no OCR"
            ),
            "structure_keys": [
                "each lexical table is a fixed grid of column x-ranges whose printed "
                "row count is asserted after extraction",
                "combining tone accents are kept on the vowel they follow in the "
                "content stream rather than the one they are drawn over",
                "interlinear glosses are matched to their form by shared x-position",
            ],
            "checked_in_layer": str(RAW_SNAPSHOT.relative_to(ROOT)),
            "prose_layer": "the PROSE and FOOTNOTE_15 tables in this importer",
            "regions": dict(Counter(row["Region"] for row in audit)),
        },
        "scope": {
            "included": (
                "every Kalkoti form the article prints: the Kalkoti column of Tables 1-6, "
                "8, 9, 13, 18 and 19, all of Tables 11, 12, 14, 16, 17 and 20, the "
                "phonetic citations in the prose of pp. 135-137 and p. 144 n. 15, the "
                "loanwords of p. 136 n. 5, and all seventeen Kalkoti interlinear examples"
            ),
            "excluded": (
                "the Palula, Gawri, Sawi and Kohistani Shina comparanda columns, which "
                "are secondary citations of Baart (1997, 1999a), Buddruss (1967), "
                "Liljegren (2008) and Schmidt & Kohistani (2008) and are ingested from "
                "those works directly; the phoneme inventories of Tables 7 and 10; the "
                "comparative tense/aspect Tables 15 and 21-24; example (9), which is "
                "Biori Palula; and the bibliography"
            ),
            "etymology_policy": (
                "Table 13 prints Turner (1966) numbers for eight words and those are "
                "linked directly; every other link is an editorial assignment carried "
                "over from the 2022 snapshot. No link is inferred from resemblance."
            ),
            "language_model": (
                "all forms belong to canonical Kalkoti (Kalk) and carry the registered "
                "Kalkot dialect tag; the three speakers of the acoustic study remain "
                "provenance in the audit"
            ),
        },
        "outputs": {
            "forms": str(FORM_OUTPUT.relative_to(ROOT)), "form_count": len(forms),
            "audit": str(AUDIT_OUTPUT.relative_to(ROOT)), "audit_count": len(audit),
            "sample": str(SAMPLE_OUTPUT.relative_to(ROOT)),
            "statuses": dict(Counter(row["Status"] for row in audit)),
        },
        "unresolved": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, help="verify the article PDF and re-extract")
    parser.add_argument("--refresh", action="store_true", help="rewrite the checked-in extraction")
    parser.add_argument("--install", action="store_true", help="write the installed CSV and audit")
    args = parser.parse_args()

    if args.pdf:
        if digest(args.pdf, "sha256") != PDF_SHA256 or digest(args.pdf, "sha512") != PDF_SHA512:
            raise ValueError("the PDF does not match the checksums this importer was built on")
        fresh = serialise(extract(args.pdf))
        if args.refresh:
            RAW_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
            RAW_SNAPSHOT.write_text(fresh, encoding="utf-8")
        elif fresh != RAW_SNAPSHOT.read_text(encoding="utf-8"):
            raise ValueError("a fresh extraction no longer reproduces the checked-in snapshot")
        import pdfplumber

        with pdfplumber.open(args.pdf) as pdf:
            for record in prose_records():
                page = pdf.pages[int(record["pdf_page"]) - 1]
                # Rebuild the page in content-stream order for the same reason
                # the tables are read that way: extract_text moves a combining
                # dot-below off the letter it belongs to.
                text = unicodedata.normalize(
                    "NFC", "".join(c["text"] for c in _chars(page))
                )
                needle = record["form"].strip("/[]")
                if needle not in text.replace(" ", "") and needle not in text:
                    raise ValueError(f"{record['unit']}: {needle!r} is not on that page")
        print(f"verified {len(snapshot())} extracted and {len(prose_records())} cited records")

    forms, audit = build()
    counts = Counter(row["Status"] for row in audit)
    print(f"{len(audit)} raw records -> {len(forms)} installed rows: {dict(counts)}")
    if not args.install:
        return
    assert len({row["Entry_Key"] for row in forms}) == len(forms)
    write_csv(FORM_OUTPUT, FORM_FIELDS, forms, header=False)
    with FORM_OUTPUT.open("a", encoding="utf-8", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerows(EDITORIAL_ANCHORS)
    write_csv(AUDIT_OUTPUT, AUDIT_FIELDS, audit, header=True)
    sample = sorted(audit, key=lambda row: row["Record_SHA256"])[:25]
    write_csv(SAMPLE_OUTPUT, AUDIT_FIELDS, sample, header=True)
    MANIFEST_OUTPUT.write_text(
        json.dumps(manifest(forms, audit), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"installed {len(forms)} Kalkoti rows from Liljegren (2013)")


if __name__ == "__main__":
    main()
