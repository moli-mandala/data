#!/usr/bin/env python3
"""Extract every Kalkoti form David Hultman's 2023 grammar sketch prints.

Hultman, David. 2023. *Topics in the grammar of Kalkoti*. BA thesis,
Department of Linguistics, Stockholm University (supervisor Henrik Liljegren).
The thesis is the fullest description of Kalkoti that exists: 65 pages built on
eleven datasets recorded from four speakers between 2006 and 2023, and the
first systematic account of the language's tone system.

Like Knobloch's Sauji sketch, it is a grammar rather than a dictionary, so its
lexicon lives in numbered tables, interlinear examples and citations in running
prose. The PDF is XeTeX output with a real text layer, so no OCR is involved,
but two properties of that layer drive the extraction:

* **The text layer contains no space characters at all.** Word boundaries exist
  only as horizontal gaps, so words are recovered by measuring the gap between
  glyphs against the font size rather than by splitting on whitespace.
* **Characters are read in content-stream order, not sorted by position.**
  Kalkoti tone is written with combining acute and grave accents, and sorting
  by position detaches them from their vowel or moves them onto the next one.

Every table is declared with its page, vertical band, column x-ranges and
printed row count, and a table that does not yield exactly that many rows
raises, so a missed page break cannot pass silently.

Run from ``data/``:

    uv run python data/other/forms/raw_data/hultman_kalkoti_2023.py --install
    uv run python data/other/forms/raw_data/hultman_kalkoti_2023.py --pdf kalkoti.pdf
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


SOURCE_ID = "hultman2023kalkoti"
LILJEGREN_ID = "kalkoti"
SNAPSHOT_DATE = "2026-08-25"
PDF_SHA256 = "fa2245e67a5e715312e698500df8cbd327b04ef1b4cdb4e9e03656ff30399810"
PDF_SHA512 = (
    "d1857ae14a7f9a340c5596587db7cb33acab29af4efc60f1e31b611fb96cd29e"
    "38957b3e85e68eccb37fa957b76d897f83782abe368c2dd140aee7629a5835f0"
)
PDF_PAGES = 65
# PDF page 13 carries the printed page number 9.
PRINTED_PAGE_OFFSET = -4
LANGUAGE_ID = "Kalk"
DIALECT_TAG = "dialect:Kalk:HKAT-xka:Kalkot%20%28HKAT%29"

ROOT = Path(__file__).resolve().parents[4]
RAW_DIR = ROOT / "data/other/forms/raw_data"
FORM_OUTPUT = ROOT / "data/other/forms/20260825-hultman-kalkoti.csv"
AUDIT_OUTPUT = RAW_DIR / "20260825-hultman-kalkoti-audit.csv"
SAMPLE_OUTPUT = RAW_DIR / "20260825-hultman-kalkoti-sample.csv"
MANIFEST_OUTPUT = RAW_DIR / "20260825-hultman-kalkoti-manifest.json"
RAW_SNAPSHOT = RAW_DIR / "20260825-hultman-kalkoti-extract.psv"

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
    "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
    "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "Snapshot_Date", "Unit_ID", "Region", "PDF_Page", "Printed_Page", "Locator",
    "Raw_Form", "Raw_Gloss", "Raw_Context", "Status", "Reason", "Final_Form",
    "Final_Gloss", "Final_Phonemic", "Final_Tags", "Emitted_Key", "Merged_Into",
    "Source", "Record_SHA256",
]

OPEN_Q, CLOSE_Q = "‘", "’"
TONE_MARKS = {"̀", "́"}


# --------------------------------------------------------------------------
# PDF geometry
# --------------------------------------------------------------------------

def _chars(page) -> list[dict]:
    """Read a page in content-stream order, keeping combining marks attached."""
    out: list[dict] = []
    for char in page.chars:
        text = char["text"]
        if text and all(unicodedata.combining(ch) for ch in text) and out:
            out.append({**out[-1], "text": text, "mark": True})
            continue
        out.append({"text": text, "x0": char["x0"], "x1": char["x1"],
                    "top": char["top"], "size": char["size"], "mark": False})
    return out


def _restream(chars: list[dict], line: list[dict]) -> list[dict]:
    wanted = Counter(
        (c["text"], round(c["x0"], 2), round(c["top"], 2), c["mark"]) for c in line
    )
    out = []
    for char in chars:
        key = (char["text"], round(char["x0"], 2), round(char["top"], 2), char["mark"])
        if wanted[key]:
            wanted[key] -= 1
            out.append(char)
    return out


def _lines(chars: list[dict], top: float, bottom: float, gap: float = 7.0):
    """Split a vertical band into visual lines, preserving stream order."""
    band = [c for c in chars if top <= c["top"] <= bottom]
    lines: list[list[dict]] = []
    current: list[dict] = []
    baseline: float | None = None
    for char in sorted(band, key=lambda c: (c["top"], c["x0"])):
        if baseline is None or char["top"] - baseline <= gap:
            current.append(char)
            baseline = char["top"]
        else:
            lines.append(current)
            current, baseline = [char], char["top"]
    if current:
        lines.append(current)
    return [_restream(chars, line) for line in lines
            if any(not c["text"].isspace() for c in line)]


# The thesis's text layer holds no space glyphs, so a word boundary is a
# horizontal gap. Intra-word gaps measure ~0 pt and inter-word gaps ~0.25 em.
WORD_GAP = 0.14


def _words(line: list[dict]) -> list[dict]:
    """Split one visual line into words by measuring inter-glyph gaps."""
    words: list[dict] = []
    right: float | None = None
    for char in line:
        if right is not None and char["x0"] - right > char["size"] * WORD_GAP:
            words.append({"text": "", "x0": char["x0"]})
        if not words:
            words.append({"text": "", "x0": char["x0"]})
        if not words[-1]["text"]:
            words[-1]["x0"] = char["x0"]
        words[-1]["text"] += char["text"]
        if not char["mark"]:
            right = char["x1"]
    return [{"text": unicodedata.normalize("NFC", w["text"]).strip(), "x0": w["x0"]}
            for w in words if w["text"].strip()]


def _cell(line: list[dict], lo: float, hi: float) -> str:
    """The text of one column of a line, with word gaps restored as spaces."""
    return " ".join(w["text"] for w in _words(line) if lo <= w["x0"] < hi).strip()


def _gloss(cell: str) -> str:
    """The quoted gloss in a cell, from the first opening to the last closing mark."""
    if OPEN_Q not in cell or CLOSE_Q not in cell:
        return ""
    return cell[cell.index(OPEN_Q) + 1: cell.rindex(CLOSE_Q)].strip()


def toneless(form: str) -> str:
    return unicodedata.normalize(
        "NFC",
        "".join(c for c in unicodedata.normalize("NFD", form) if c not in TONE_MARKS),
    )


# --------------------------------------------------------------------------
# Table geometry
# --------------------------------------------------------------------------
# Each entry gives the PDF page, the vertical band holding the table body, the
# column x-ranges, and the number of body lines the printed table has. The line
# count is asserted after extraction.

TABLES: dict[str, dict] = {
    # 4.1.1 Illustration of the three-way stop contrast: a form line and a gloss
    # line alternate under four place-of-articulation columns.
    "t3": {"table": 3, "page": 13, "band": (195, 285), "lines": 6,
           "columns": [("voice", 160, 235), ("labial", 235, 290),
                       ("dental", 290, 333), ("palatal", 333, 393),
                       ("velar", 393, 470)]},
    "t4": {"table": 4, "page": 15, "band": (115, 320), "lines": 14,
           "columns": [("phoneme", 125, 185), ("kalkoti", 185, 262),
                       ("donor", 262, 296), ("script", 296, 318),
                       ("roman", 318, 382), ("gloss", 382, 520)]},
    "t5": {"table": 5, "page": 15, "band": (540, 615), "lines": 5,
           "columns": [("kalkoti", 190, 244), ("donor", 244, 277),
                       ("roman", 277, 328), ("gloss", 328, 520)]},
    "t6": {"table": 6, "page": 16, "band": (158, 250), "lines": 6,
           "columns": [("gloss", 200, 258), ("kalkoti", 258, 305),
                       ("palula", 305, 354), ("gawri", 354, 470)]},
    "t7": {"table": 7, "page": 16, "band": (570, 600), "lines": 2,
           "columns": [("place", 150, 212), ("kalkoti", 212, 240),
                       ("phonetic", 240, 275), ("gloss", 275, 331),
                       ("kalkoti2", 331, 359), ("phonetic2", 359, 393),
                       ("gloss2", 393, 470)]},
    "t9": {"table": 9, "page": 20, "band": (448, 645), "lines": 13,
           "columns": [("kalkoti", 205, 254), ("palula", 254, 298),
                       ("gloss", 298, 470)]},
    "t10": {"table": 10, "page": 22, "band": (202, 350), "lines": 10,
            "columns": [("kalkoti", 205, 253), ("palula", 253, 309),
                        ("gloss", 309, 470)]},
    "t12": {"table": 12, "page": 27, "band": (618, 672), "lines": 4,
            "columns": [("sg", 222, 276), ("pl", 276, 321), ("gloss", 321, 470)]},
    "t13": {"table": 13, "page": 29, "band": (116, 155), "lines": 3,
            "columns": [("sg", 205, 258), ("pl", 258, 323), ("gloss", 323, 470)]},
    "t16": {"table": 16, "page": 32, "band": (483, 623), "lines": 10,
            "columns": [("n1", 60, 85), ("v1", 85, 126), ("n2", 126, 149),
                        ("v2", 149, 198), ("n3", 198, 221), ("v3", 221, 298),
                        ("n4", 298, 321), ("v4", 321, 412), ("n5", 412, 446),
                        ("v5", 446, 520)]},
    "t17": {"table": 17, "page": 39, "band": (267, 425), "lines": 11,
            "columns": [("gender", 140, 205), ("kalkoti", 205, 285),
                        ("comparison", 285, 470)]},
    "t19": {"table": 19, "page": 41, "band": (568, 672), "lines": 7,
            "columns": [("msg", 120, 227), ("mpl", 227, 333),
                        ("f", 333, 421), ("gloss", 421, 520)]},
    "t20": {"table": 20, "page": 42, "band": (430, 565), "lines": 9,
            "columns": [("ipfv", 190, 259), ("pfv", 259, 311), ("gloss", 311, 470)]},
    "t23": {"table": 23, "page": 45, "band": (116, 200), "lines": 6,
            "columns": [("pfv", 110, 257), ("pluperfect", 257, 470)]},
    "t24": {"table": 24, "page": 46, "band": (327, 396), "lines": 5,
            "columns": [("pfv", 175, 237), ("ipfv", 237, 310),
                        ("impv", 310, 375), ("gloss", 375, 470)]},
    "t25": {"table": 25, "page": 48, "band": (116, 272), "lines": 11,
            "columns": [("pfv", 130, 192), ("ipfv", 192, 265),
                        ("inf", 265, 322), ("vn", 322, 394), ("gloss", 394, 470)]},
    "t26": {"table": 26, "page": 49, "band": (476, 500), "lines": 2,
            "columns": [("gloss", 225, 271), ("ipfv", 271, 329), ("pfv", 329, 470)]},
    "t28": {"table": 28, "page": 56, "band": (568, 595), "lines": 2,
            "columns": [("polarity", 205, 275), ("pres", 275, 323),
                        ("past", 323, 470)]},
}
# Table 11 (declension) and Table 14 (pronouns) print two paradigms side by
# side under shared headers, and Table 15 pairs each adjective with a phrase
# illustrating it; each gets its own reader below.
T11 = {"table": 11, "page": 25, "band": (514, 553), "lines": 3,
       "columns": [("case", 110, 162), ("sg", 162, 215), ("pl", 215, 305),
                   ("case2", 305, 357), ("sg2", 357, 412), ("pl2", 412, 470)]}
T14 = {"table": 14, "page": 30, "band": (492, 546), "lines": 4,
       "columns": [("case", 128, 198), ("1sg", 198, 227), ("2sg", 227, 256),
                   ("3sg.near", 256, 290), ("3sg.far", 290, 319),
                   ("1pl", 319, 359), ("2pl", 359, 403),
                   ("3pl.near", 403, 437), ("3pl.far", 437, 470)]}
T15 = {"table": 15, "page": 32, "band": (116, 210), "lines": 6,
       "columns": [("gloss", 118, 174), ("m", 174, 280), ("f", 280, 381),
                   ("gawri", 381, 470)]}
T21 = {"table": 21, "page": 43, "band": (217, 405), "lines": 14,
       "columns": [("ipfv", 195, 260), ("pfv", 260, 330), ("gloss", 330, 470)]}


def _grid(chars: dict[int, list[dict]], spec: dict) -> list[dict]:
    page, (top, bottom) = spec["page"], spec["band"]
    rows = []
    for line in _lines(chars[page], top, bottom):
        cells = {name: _cell(line, lo, hi) for name, lo, hi in spec["columns"]}
        if not any(cells.values()):
            continue
        cells["pdf_page"] = str(page)
        cells["top"] = f"{min(c['top'] for c in line):.0f}"
        rows.append(cells)
    if len(rows) != spec["lines"]:
        raise ValueError(
            f"table {spec['table']} extracted {len(rows)} lines, expected {spec['lines']}"
        )
    return rows


# --------------------------------------------------------------------------
# Numbered interlinear examples
# --------------------------------------------------------------------------
# An example block opens with "(N)" in the left margin and every following line
# of the block is indented past it. Inside the block each sub-example is a form
# line, the gloss line beneath it, and a free translation that closes it and
# carries the dataset reference the thesis cites, e.g. [JK15d-3].

EXAMPLE_NUMBER = re.compile(r"^\((\d+)\)$")
SUB_LETTER = re.compile(r"^([a-h])\.$")
DATASET_REF = re.compile(r"\[([A-Z]{1,2}\d{2}[a-z]-[\w.]+)\]")
BODY_MARGIN = 96.0


def _blocks(lines: list[list[dict]]) -> list[tuple[str, list[list[dict]]]]:
    out, index = [], 0
    while index < len(lines):
        words = _words(lines[index])
        match = EXAMPLE_NUMBER.match(words[0]["text"]) if words else None
        if not match:
            index += 1
            continue
        block, cursor = [lines[index]], index + 1
        while cursor < len(lines) and min(w["x0"] for w in _words(lines[cursor])) >= BODY_MARGIN:
            block.append(lines[cursor])
            cursor += 1
        out.append((match.group(1), block))
        index = max(cursor, index + 1)
    return out


def _align(forms: list[dict], glosses: list[dict]) -> list[tuple[str, str]]:
    """Attach each gloss word to the form word standing above it."""
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


def _examples(chars: dict[int, list[dict]]) -> list[dict]:
    records: list[dict] = []
    for page in sorted(chars):
        for number, block in _blocks(_lines(chars[page], 0, 900)):
            letter, pending = "", []
            for line in block:
                words = _words(line)
                text = " ".join(w["text"] for w in words)
                if text.startswith(OPEN_Q) or (letter and text.lstrip().startswith(OPEN_Q)):
                    reference = DATASET_REF.search(text)
                    translation = _gloss(text)
                    for position, (form, gloss) in enumerate(_flush(pending), start=1):
                        records.append({
                            "example": number + letter, "position": str(position),
                            "form": form, "gloss": gloss, "translation": translation,
                            "reference": reference.group(1) if reference else "",
                            "pdf_page": str(page),
                        })
                    pending = []
                    continue
                stripped = [w for w in words if not EXAMPLE_NUMBER.match(w["text"])]
                head = SUB_LETTER.match(stripped[0]["text"]) if stripped else None
                if head:
                    letter, stripped = head.group(1), stripped[1:]
                if stripped:
                    pending.append(stripped)
    return records


def _flush(pending: list[list[dict]]) -> list[tuple[str, str]]:
    """Pair the accumulated lines of a sub-example as form/gloss rows."""
    pairs: list[tuple[str, str]] = []
    for index in range(0, len(pending) - 1, 2):
        pairs.extend(_align(pending[index], pending[index + 1]))
    return pairs


# --------------------------------------------------------------------------
# Lexical citations in running prose
# --------------------------------------------------------------------------
# Every entry is checked against the page it cites by ``--pdf``. Only citations
# that are unambiguously Kalkoti are listed: the thesis writes object-language
# material either as phonemic IPA in slashes, as phonetic transcription in
# brackets, or in its practical orthography.

PROSE: list[dict] = [
    {"unit": "tome", "page": 13, "form": "/mɒʈɛ/", "gloss": "to me",
     "tags": "pron personal 1sg dat"},
    {"unit": "toyou", "page": 13, "form": "/tʊʈɛ/", "gloss": "to you",
     "tags": "pron personal 2sg dat"},
    {"unit": "did", "page": 13, "form": "/tʰæ̌ːl/", "gloss": "did", "tags": "verb pfv"},
    {"unit": "eye", "page": 13, "form": "[ɨ́ʈʂʰɪ]", "gloss": "eye", "tags": "noun f"},
    {"unit": "walnut", "page": 13, "form": "[ʈʂʰòːɾ]", "gloss": "walnut", "tags": "noun"},
    {"unit": "bear", "page": 13, "form": "[ɨʈʂ]", "gloss": "bear", "tags": "noun"},
    {"unit": "much", "page": 14, "form": "[bæːɖ]", "gloss": "much, very", "tags": "adv",
     "variants": ["[bæːɽ]"]},
    {"unit": "removed", "page": 14, "form": "[gəɽɪl]", "gloss": "removed", "tags": "verb pfv"},
    {"unit": "boys", "page": 14, "form": "[ləɽkʊ́ɾ]", "gloss": "boys", "tags": "noun m pl"},
    {"unit": "gift", "page": 14, "form": "[ɖɑ̀ːleː]", "gloss": "gift", "tags": "noun loanword",
     "note": "Hultman identifies this as a Pashto loanword"},
    {"unit": "sunken", "page": 14, "form": "[ɖʊb]", "gloss": "sunken", "tags": "adj"},
    {"unit": "sheep", "page": 14, "form": "[eːˀɾ̥]", "gloss": "sheep", "tags": "noun"},
    {"unit": "puton", "page": 14, "form": "/ʂɑ̌ːl/", "gloss": "put on", "tags": "verb pfv"},
    {"unit": "every", "page": 15, "form": "/xɛɾ/", "gloss": "every", "tags": "determiner"},
    {"unit": "five", "page": 16, "form": "/pɑːndʑ/", "gloss": "five", "tags": "num"},
    {"unit": "fifteen", "page": 16, "form": "/pɛndʑěːɕ/", "gloss": "fifteen", "tags": "num"},
    {"unit": "path", "page": 16, "form": "[pɑːnd]", "gloss": "path", "tags": "noun f",
     "variants": ["[pɑːn]"]},
    {"unit": "boys2", "page": 17, "form": "/lɛɖkʊ́ɾ/", "gloss": "boys", "tags": "noun m pl"},
    {"unit": "my", "page": 17, "form": "/mɪ/", "gloss": "I, my", "tags": "pron personal 1sg erg gen"},
    {"unit": "saw", "page": 17, "form": "[dɾɨʂ]", "gloss": "saw", "tags": "verb pfv"},
    {"unit": "tied", "page": 17, "form": "[gəɳɪl]", "gloss": "tied", "tags": "verb pfv"},
    {"unit": "gift2", "page": 18, "form": "/ɖæ̀ːleː/", "gloss": "gift", "tags": "noun loanword",
     "note": "cited for its long vowel in a non-root-final syllable"},
    {"unit": "dayafter", "page": 18, "form": "/tɾiːdeː/", "gloss": "the day after tomorrow",
     "tags": "adv temporal"},
    {"unit": "money", "page": 18, "form": "[pæ̃ːs]", "gloss": "money", "tags": "noun loanword",
     "note": "Hultman derives it from Gawri pää~s ‘money’ (Baart and Sagar 2004: 280)"},
    {"unit": "from", "page": 20, "form": "=daa", "gloss": "from", "tags": "postp"},
    {"unit": "outof", "page": 20, "form": "=eel", "gloss": "out of", "tags": "postp"},
    {"unit": "to", "page": 49, "form": "=thä", "gloss": "to", "tags": "postp dat"},
    {"unit": "nextto", "page": 54, "form": "=dii", "gloss": "at, next to", "tags": "postp"},
    {"unit": "night", "page": 22, "form": "ràat", "gloss": "night", "tags": "noun",
     "note": "Hultman notes that this forms a minimal tone pair with raat ‘blood’"},
    {"unit": "meat", "page": 22, "form": "maas", "gloss": "meat", "tags": "noun"},
    {"unit": "near", "page": 22, "form": "niyär", "gloss": "near", "tags": "adj"},
    {"unit": "song", "page": 22, "form": "róo", "gloss": "song", "tags": "noun"},
    {"unit": "what", "page": 57, "form": "guwaá", "gloss": "what", "tags": "pron interr"},
    {"unit": "who", "page": 57, "form": "kii", "gloss": "who", "tags": "pron interr"},
    # The phonological illustrations of examples (2)-(4), which are lists of
    # citations rather than interlinear examples.
    {"unit": "ex2-sheep", "page": 14, "example": "2a", "form": "/ěːɾ/", "gloss": "sheep",
     "tags": "noun", "note": "realized [eːˀɾ̥]"},
    {"unit": "ex3-man", "page": 18, "example": "3a", "form": "/měːɕ/", "gloss": "man",
     "tags": "noun m"},
    {"unit": "ex3-men", "page": 18, "example": "3a", "form": "/mɪ́ɕɑːl/", "gloss": "men",
     "tags": "noun m pl"},
    {"unit": "ex3-woman", "page": 18, "example": "3a", "form": "/tɾeːɾ/", "gloss": "woman",
     "tags": "noun f"},
    {"unit": "ex3-women", "page": 18, "example": "3a", "form": "/tɾɪjɑːl/", "gloss": "women",
     "tags": "noun f pl"},
    {"unit": "ex3-boy", "page": 18, "example": "3b", "form": "/puː/", "gloss": "boy",
     "tags": "noun m"},
    {"unit": "ex3-littleboy", "page": 18, "example": "3b", "form": "/pʊʈoːɕ/",
     "gloss": "little boy", "tags": "noun m diminutive"},
    {"unit": "ex3-girl", "page": 18, "example": "3b", "form": "/peː/", "gloss": "girl",
     "tags": "noun f"},
    {"unit": "ex3-littlegirl", "page": 18, "example": "3b", "form": "/pɪʈeːɕ/",
     "gloss": "little girl", "tags": "noun f diminutive"},
    {"unit": "ex4-made", "page": 21, "example": "4a", "form": "/tɾɑ̌ːl/", "gloss": "made",
     "tags": "verb pfv", "note": "glottalized [tɾɑːˀɬ] utterance-finally"},
    {"unit": "ex4-died", "page": 21, "example": "4a", "form": "/mʊ́ɾ/", "gloss": "died",
     "tags": "verb pfv", "note": "glottalized [mʊˀɾ̥] utterance-finally"},
    {"unit": "ex4-became", "page": 21, "example": "4b", "form": "/bɪ̀l/", "gloss": "became",
     "tags": "verb pfv", "note": "not glottalized before the polar question clitic"},
]


# --------------------------------------------------------------------------
# Source IPA -> the thesis's own practical orthography
# --------------------------------------------------------------------------
# Tables 2 and 8 print each phoneme beside its orthographic spelling in angle
# brackets, so the phonemic and phonetic citations can be rewritten into the
# same orthography the rest of the thesis uses. Only correspondences the thesis
# itself states are listed, plus the purely phonetic symbols it uses in square
# brackets, which are noted where they are not a plain allophone.

# Table 8: each vowel with its short and, where the thesis has one, its long
# orthographic spelling. Length is written by doubling the letter.
VOWELS: dict[str, tuple[str, str]] = {
    "i": ("i", "ii"), "u": ("u", "uu"), "e": ("e", "ee"), "o": ("o", "oo"),
    "æ": ("ä", "ää"), "ɑ": ("a", "aa"), "a": ("a", "aa"),
    # Short vowels, which the thesis writes with their own symbols.
    "ɪ": ("i", "ii"), "ʊ": ("u", "uu"), "ɛ": ("ä", "ää"), "ɒ": ("a", "aa"),
    # Phonetic variants used only inside square brackets.
    "ə": ("ä", "ää"), "ɨ": ("i", "ii"),
}
# Table 2: consonants, longest sequence first.
CONSONANTS: list[tuple[str, str]] = [
    ("ʈʂʰ", "c̣h"), ("ʈʂ", "c̣"), ("tɕʰ", "čh"), ("tɕ", "č"), ("dʑ", "ǰ"),
    ("pʰ", "ph"), ("tʰ", "th"), ("kʰ", "kh"), ("ts", "ts"),
    ("ŋg", "ng"), ("ŋ", "ng"),
    ("ʈ", "ṭ"), ("ɖ", "ḍ"), ("ɳ", "ṇ"), ("ʂ", "ṣ"), ("ɕ", "š"),
    ("ɣ", "ġ"), ("ɾ", "r"), ("ɽ", "ṛ"), ("ʋ", "w"), ("j", "y"), ("ɬ", "l"),
]
# The thesis marks tone on the IPA vowel: a caron is the rising contour it
# writes VV́ in the orthography, an acute the high tone on the first mora, and a
# grave the low tone. Glottalization and devoicing are phonetic detail.
CARON, ACUTE, GRAVE, LENGTH = "̌", "́", "̀", "ː"
TILDE = "̃"
PHONETIC_ONLY = {"ˀ", "̥", "̬", "̪"}


def to_orthography(citation: str) -> str:
    """Rewrite one slashed or bracketed citation in the thesis's orthography."""
    text = unicodedata.normalize("NFD", citation.strip().strip("/[]").strip())
    out: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char in PHONETIC_ONLY or char == LENGTH:
            index += 1
            continue
        if char in VOWELS:
            # A tone mark sits between the vowel and its length mark, so read
            # the whole vowel-plus-diacritics sequence before spelling it.
            cursor, mark, long, nasal = index + 1, "", False, False
            while cursor < len(text):
                if unicodedata.combining(text[cursor]):
                    if text[cursor] in (CARON, ACUTE, GRAVE):
                        mark = text[cursor]
                    elif text[cursor] == TILDE:
                        nasal = True
                elif text[cursor] == LENGTH:
                    long = True
                else:
                    break
                cursor += 1
            short_form, long_form = VOWELS[char]
            spelling = _toned(long_form if long else short_form, mark)
            # The thesis treats a long nasal vowel as an allophone of /Vːn/ and
            # writes it with the consonant: [pæ̃ːs] is spelled pääns.
            out.append(spelling + "n" if nasal else spelling)
            index = cursor
            continue
        for source, target in CONSONANTS:
            if text.startswith(unicodedata.normalize("NFD", source), index):
                out.append(target)
                index += len(unicodedata.normalize("NFD", source))
                break
        else:
            out.append(char)
            index += 1
    return unicodedata.normalize("NFC", "".join(out))


def _toned(spelling: str, mark: str) -> str:
    """Place a tone mark on the orthographic vowel the thesis writes it on."""
    if not mark or not spelling:
        return spelling
    if len(spelling) == 2 and spelling[0] == spelling[1]:
        # A caron is the rising contour, written as an acute on the second
        # mora; an acute or grave stays on the first mora.
        if mark == CARON:
            return spelling + ACUTE
        return spelling[0] + mark + spelling[1]
    return spelling + (ACUTE if mark == CARON else mark)


# --------------------------------------------------------------------------
# One flat record list, snapshotted so the importer runs without the PDF
# --------------------------------------------------------------------------

SNAPSHOT_FIELDS = ["unit", "region", "pdf_page", "form", "gloss", "context"]


def _record(unit, region, page, form, gloss, **context) -> dict[str, str]:
    parts = "; ".join(f"{k}={v}" for k, v in context.items() if v)
    return {"unit": unit, "region": region, "pdf_page": str(page),
            "form": form, "gloss": gloss, "context": parts}


def _alternates(cell: str) -> list[str]:
    return [part.strip() for part in cell.split(",") if part.strip()]


def _split_gloss(cell: str) -> tuple[str, str]:
    """A cell of the form ``form ‘gloss’`` split into its two parts."""
    if OPEN_Q not in cell:
        return cell.strip(), ""
    return cell[: cell.index(OPEN_Q)].strip(), _gloss(cell)


def _emit(rows, unit_prefix, region, columns, gloss_key, **fixed):
    """Emit one record per non-empty form column of each row."""
    for index, row in enumerate(rows, start=1):
        gloss = _gloss(row.get(gloss_key, "")) or row.get(gloss_key, "")
        for column in columns:
            cell = row.get(column, "")
            if not cell:
                continue
            for position, form in enumerate(_alternates(cell), start=1):
                suffix = f":a{position}" if position > 1 else ""
                yield _record(f"{unit_prefix}:{index}:{column}{suffix}", region,
                              row["pdf_page"], form, gloss, column=column,
                              printed=cell if position > 1 else "", **fixed)


def extract(path: Path) -> list[dict[str, str]]:
    """Read every Kalkoti citation in the thesis straight from the PDF."""
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        if len(pdf.pages) != PDF_PAGES:
            raise ValueError(f"expected {PDF_PAGES} pages, found {len(pdf.pages)}")
        chars = {n: _chars(page) for n, page in enumerate(pdf.pages, start=1)}

    out: list[dict[str, str]] = []

    # Table 3: a form line and a gloss line alternate under four place columns.
    places = ["labial", "dental", "palatal", "velar"]
    rows = _grid(chars, TABLES["t3"])
    for pair in range(0, len(rows), 2):
        voice = rows[pair]["voice"].lower()
        for place in places:
            out.append(_record(f"t3:{voice}:{place}", "t3", rows[pair]["pdf_page"],
                               rows[pair][place], _gloss(rows[pair + 1][place]),
                               voice=voice, place=place))

    # Tables 4 and 5: a Kalkoti loanword beside the donor form it comes from.
    for key, region in (("t4", "t4"), ("t5", "t5")):
        for index, row in enumerate(_grid(chars, TABLES[key]), start=1):
            donor = " ".join(filter(None, [row.get("donor", ""), row.get("script", ""),
                                           row.get("roman", "")])).strip()
            form = row["kalkoti"].replace("?", "").strip()
            out.append(_record(f"{region}:{index}", region, row["pdf_page"], form,
                               _gloss(row["gloss"]), donor=donor,
                               phoneme=row.get("phoneme", ""),
                               uncertain="yes" if "?" in row["kalkoti"] else ""))

    for key, columns, gloss_key, region in (
        ("t6", ["kalkoti"], "gloss", "t6"),
        ("t9", ["kalkoti"], "gloss", "t9"),
        ("t10", ["kalkoti"], "gloss", "t10"),
        ("t12", ["sg", "pl"], "gloss", "t12"),
        ("t13", ["sg", "pl"], "gloss", "t13"),
        ("t20", ["ipfv", "pfv"], "gloss", "t20"),
        ("t24", ["pfv", "ipfv", "impv"], "gloss", "t24"),
        ("t25", ["pfv", "ipfv", "inf", "vn"], "gloss", "t25"),
        ("t26", ["ipfv", "pfv"], "gloss", "t26"),
    ):
        out.extend(_emit(_grid(chars, TABLES[key]), key, region, columns, gloss_key))

    # Table 7: two half-tables, each a form, its phonetic realization and a gloss.
    for index, row in enumerate(_grid(chars, TABLES["t7"]), start=1):
        for half, (form, phon, gloss) in enumerate(
            [("kalkoti", "phonetic", "gloss"), ("kalkoti2", "phonetic2", "gloss2")], start=1
        ):
            out.append(_record(f"t7:{index}:{half}", "t7", row["pdf_page"], row[form],
                               _gloss(row[gloss]), phonetic=row[phon],
                               place=row["place"].lower()))

    # Table 11: two declensions side by side.
    for row in _grid(chars, T11):
        case = row["case"].lower()
        for lemma, (sg, pl) in enumerate([("sg", "pl"), ("sg2", "pl2")], start=1):
            for number, column in (("sg", sg), ("pl", pl)):
                out.append(_record(f"t11:{lemma}:{case}:{number}", "t11", row["pdf_page"],
                                   row[column], "man" if lemma == 1 else "friend",
                                   case=case, number=number))

    # Table 14: the personal pronoun paradigm.
    for row in _grid(chars, T14):
        case = row["case"].lower()
        for cell, _, _ in T14["columns"][1:]:
            out.append(_record(f"t14:{cell}:{case}", "t14", row["pdf_page"],
                               row[cell], "", case=case, person=cell))

    # Table 15: each adjective is shown inside an illustrative phrase.
    rows = _grid(chars, T15)
    for pair in range(0, len(rows), 2):
        head, below = rows[pair], rows[pair + 1]
        for gender in ("m", "f"):
            out.append(_record(f"t15:{pair // 2 + 1}:{gender}", "t15", head["pdf_page"],
                               head[gender], _gloss(head["gloss"]), gender=gender,
                               phrase=_gloss(below.get(gender, "")),
                               headword=_gloss(head["gloss"])))

    # Table 16: sixty numerals in five blocks of ten.
    for row in _grid(chars, TABLES["t16"]):
        for block in range(1, 6):
            value, form = row[f"n{block}"], row[f"v{block}"]
            if not value or not form:
                continue
            out.append(_record(f"t16:{value}", "t16", row["pdf_page"], form, value,
                               value=value))

    # Table 17: gender, with the comparandum that supports it.
    gender = ""
    for index, row in enumerate(_grid(chars, TABLES["t17"]), start=1):
        gender = row["gender"].lower() or gender
        form, gloss = _split_gloss(row["kalkoti"])
        out.append(_record(f"t17:{index}", "t17", row["pdf_page"], form, gloss,
                           gender=gender, comparison=row["comparison"]))

    # Table 19: the imperfective paradigm in three agreement columns.
    for index, row in enumerate(_grid(chars, TABLES["t19"]), start=1):
        for column in ("msg", "mpl", "f"):
            form, _ = _split_gloss(row[column])
            out.append(_record(f"t19:{index}:{column}", "t19", row["pdf_page"], form,
                               _gloss(row["gloss"]), column=column))

    # Table 21: irregular perfectives, where a gendered pair straddles three lines.
    pending: list[dict] = []
    for row in _grid(chars, T21):
        pending.append(row)
    index = 0
    while index < len(pending):
        row = pending[index]
        if row["ipfv"] and row["pfv"]:
            out.append(_record(f"t21:{index}:ipfv", "t21", row["pdf_page"], row["ipfv"],
                               _gloss(row["gloss"]), column="ipfv"))
            out.append(_record(f"t21:{index}:pfv", "t21", row["pdf_page"], row["pfv"],
                               _gloss(row["gloss"]), column="pfv"))
            index += 1
            continue
        # A split cell: the perfective forms sit above and below their gloss.
        group = pending[index:index + 3]
        gloss = next((_gloss(r["gloss"]) for r in group if r["gloss"]), "")
        stem = next((r["ipfv"] for r in group if r["ipfv"]), "")
        out.append(_record(f"t21:{index}:ipfv", "t21", row["pdf_page"], stem, gloss,
                           column="ipfv"))
        for position, entry in enumerate(r for r in group if r["pfv"]):
            form, agreement = entry["pfv"], ""
            if "(" in form:
                form, agreement = form.split("(")[0].strip(), form.split("(")[1].strip(") ")
            out.append(_record(f"t21:{index}:pfv:{position + 1}", "t21", row["pdf_page"],
                               form, gloss, column="pfv", agreement=agreement))
        index += 3

    # Table 23: each perfective beside its pluperfect, both with their own gloss.
    for index, row in enumerate(_grid(chars, TABLES["t23"]), start=1):
        for column in ("pfv", "pluperfect"):
            form, gloss = _split_gloss(row[column])
            out.append(_record(f"t23:{index}:{column}", "t23", row["pdf_page"], form,
                               gloss, column=column))

    # Table 28: the copula paradigm.
    for row in _grid(chars, TABLES["t28"]):
        polarity = row["polarity"].lower()
        for tense in ("pres", "past"):
            for position, form in enumerate(_alternates(row[tense]), start=1):
                if form in {"∅", "?"} or "?" in form:
                    continue
                suffix = f":a{position}" if position > 1 else ""
                out.append(_record(f"t28:{polarity}:{tense}{suffix}", "t28",
                                   row["pdf_page"], form, "", polarity=polarity,
                                   tense=tense))

    for row in _examples(chars):
        out.append(_record(
            f"ex{row['example']}:{row['position']}", "interlinear", row["pdf_page"],
            row["form"], row["gloss"], example=row["example"],
            reference=row["reference"], translation=row["translation"],
        ))
    return out


def prose_records() -> list[dict[str, str]]:
    out = []
    for entry in PROSE:
        forms = [entry["form"], *entry.get("variants", [])]
        for position, form in enumerate(forms, start=1):
            unit = f"prose:{entry['unit']}" + (f":a{position}" if position > 1 else "")
            out.append(_record(
                unit, "prose", entry["page"], form, entry["gloss"],
                tags=entry["tags"], note=entry.get("note", ""),
                example=entry.get("example", ""),
                variant_of=f"prose:{entry['unit']}" if position > 1 else "",
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


def records() -> list[dict[str, str]]:
    return snapshot() + prose_records()


# --------------------------------------------------------------------------
# Glosses and grammatical labels
# --------------------------------------------------------------------------

TABLE_OF = {
    "t3": 3, "t4": 4, "t5": 5, "t6": 6, "t7": 7, "t9": 9, "t10": 10, "t11": 11,
    "t12": 12, "t13": 13, "t14": 14, "t15": 15, "t16": 16, "t17": 17, "t19": 19,
    "t20": 20, "t21": 21, "t23": 23, "t24": 24, "t25": 25, "t26": 26, "t28": 28,
}
# Regions the thesis prints in IPA or in phonetic transcription rather than in
# its practical orthography.
IPA_REGIONS = {"t3"}
REGION_TAGS = {
    "t9": "", "t10": "", "t11": "noun", "t12": "noun", "t13": "noun loanword",
    "t14": "pron personal", "t15": "adj", "t16": "num", "t17": "noun",
    "t19": "verb ipfv", "t20": "verb", "t21": "verb", "t23": "verb",
    "t24": "verb", "t25": "verb", "t26": "verb", "t28": "copula verb",
    "t4": "loanword", "t5": "",
}
COLUMN_TAGS = {
    ("t12", "sg"): "sg", ("t12", "pl"): "pl",
    ("t13", "sg"): "sg", ("t13", "pl"): "pl",
    ("t20", "ipfv"): "ipfv", ("t20", "pfv"): "pfv",
    ("t21", "ipfv"): "ipfv stem", ("t21", "pfv"): "pfv",
    ("t23", "pfv"): "pfv", ("t23", "pluperfect"): "pfv pp",
    ("t24", "pfv"): "pfv", ("t24", "ipfv"): "ipfv", ("t24", "impv"): "impv",
    ("t25", "pfv"): "pfv", ("t25", "ipfv"): "ipfv", ("t25", "inf"): "inf",
    ("t25", "vn"): "ger", ("t26", "ipfv"): "ipfv", ("t26", "pfv"): "pfv",
    ("t19", "msg"): "m sg", ("t19", "mpl"): "m pl", ("t19", "f"): "f",
}
CASE_TAGS = {
    "direct": "dir", "oblique": "obl", "erg-gen": "erg gen",
    "nominative": "nom", "accusative": "acc", "ergative": "erg", "genitive": "gen",
}
PRONOUN_GLOSS = {
    "1sg": "I", "2sg": "you (singular)", "1pl": "we", "2pl": "you (plural)",
    "3sg.near": "he, she, it (proximate)", "3sg.far": "he, she, it (remote)",
    "3pl.near": "they (proximate)", "3pl.far": "they (remote)",
}
PRONOUN_TAGS = {
    "1sg": "1sg", "2sg": "2sg", "1pl": "1pl", "2pl": "2pl",
    "3sg.near": "3sg prox", "3sg.far": "3sg dist",
    "3pl.near": "3pl prox", "3pl.far": "3pl dist",
}
AGREEMENT_TAGS = {"m": "m", "f": "f", "m.sg": "m sg", "m.pl, f": "m pl f"}

_ONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
_TEENS = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
          "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]


def numeral_name(value: str) -> str:
    number = int(value)
    if number >= 1000:
        return "one thousand"
    if number >= 100:
        rest = number % 100
        head = f"{_ONES[number // 100]} hundred"
        return head if not rest else f"{head} and {numeral_name(str(rest))}"
    if number < 10:
        return _ONES[number]
    if number < 20:
        return _TEENS[number - 10]
    tens, ones = divmod(number, 10)
    return _TENS[tens] + (f"-{_ONES[ones]}" if ones else "")


# The thesis lists its glossing abbreviations on its own Abbreviations page and
# writes them in lower case, so a gloss chunk is a category label exactly when
# it appears in that list. Person-number combinations are added, as the glosses
# use them throughout.
INTERLINEAR_TAGS = {
    "acc": "acc", "caus": "caus", "cop": "copula", "cvb": "conjunctive-participle",
    "dat": "dat", "dim": "diminutive", "erg": "erg", "f": "f", "gen": "gen",
    "imp": "impv", "indf": "indef", "inf": "inf", "ipfv": "ipfv", "irr": "modal",
    "m": "m", "neg": "neg", "nom": "nom", "obl": "obl", "pass": "pass",
    "pfv": "pfv", "pl": "pl", "prs": "pres", "pst": "pret", "q": "interr",
    "refl": "refl", "sg": "sg", "vn": "ger", "quot": "discourse-marker",
    "1sg": "1sg", "2sg": "2sg", "3sg": "3sg",
    "1pl": "1pl", "2pl": "2pl", "3pl": "3pl",
}
# ``quot`` is on the thesis's list but Jambu has no quotative tag, so it is
# reported in the audit rather than mapped onto something it does not mean.
UNMAPPED_LABELS: set[str] = set()
# A verb form is one whose gloss carries an aspect, mood or imperative label.
VERBAL_LABELS = {"ipfv", "pfv", "impv", "inf", "modal", "copula"}


def read_interlinear(gloss: str, host_only: bool = False) -> tuple[str, list[str], list[str]]:
    """Split a gloss into its lexical part and the thesis's category labels.

    With ``host_only`` the labels of an enclitic are left out: in
    ``friend-obl.pl=erg`` the ergative is marked by the clitic ``=ä``, so it is
    not a property of the host word that the headword records.
    """
    if host_only:
        gloss = gloss.split("=", 1)[0]
    words, labels, unknown = [], [], []
    for chunk in re.split(r"[-=]", gloss):
        for piece in re.split(r"[.()]", chunk):
            piece = piece.strip()
            if not piece:
                continue
            if piece in INTERLINEAR_TAGS:
                labels.extend(INTERLINEAR_TAGS[piece].split())
            elif piece in UNMAPPED_LABELS:
                unknown.append(piece)
            else:
                words.append(piece.replace("_", " "))
    if VERBAL_LABELS & set(labels):
        labels.append("verb")
    return " ".join(words), list(dict.fromkeys(labels)), unknown


# Words the thesis glosses only by category, with the meaning that category
# states and the extra part of speech it implies.
PERSON_GLOSS = {
    "1sg": "I", "2sg": "you (singular)", "3sg": "he, she, it",
    "1pl": "we", "2pl": "you (plural)", "3pl": "they",
}
CATEGORY_GLOSS = {
    "copula": ("is, are", "copula verb"), "indef": ("a, one", "indef determiner"),
    "neg": ("not", "negator"), "refl": ("his own, her own", "pron refl"),
    "dat": ("to", "postp"), "discourse-marker": ("quotative particle", ""),
}


def _category_gloss(labels: list[str]) -> tuple[str, str]:
    for person, gloss in PERSON_GLOSS.items():
        if person in labels:
            return gloss, "pron personal"
    for label, (gloss, extra) in CATEGORY_GLOSS.items():
        if label in labels:
            if label == "copula" and "pret" in labels:
                return "was, were", extra
            return gloss, extra
    return "", ""


def split_table_gloss(gloss: str) -> tuple[str, list[str]]:
    """Some table glosses carry the same lower-case labels, e.g. 'see.pfv'."""
    words, tags = [], []
    for part in gloss.split("."):
        if part in INTERLINEAR_TAGS and words:
            tags.extend(INTERLINEAR_TAGS[part].split())
        else:
            words.append(part)
    return " ".join(words).strip(), tags


# --------------------------------------------------------------------------
# Locators and interpretation
# --------------------------------------------------------------------------

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
    page = f"p. {printed_page(record)}"
    region, context = record["region"], _context(record)
    if region in TABLE_OF:
        return f"{page}, table {TABLE_OF[region]}"
    if region == "interlinear":
        reference = context.get("reference", "")
        example = f"{page}, example {context.get('example', '')}"
        return f"{example}, {reference}" if reference else example
    if region == "prose" and context.get("example"):
        return f"{page}, example {context['example']}"
    return page


def interpret(record: dict[str, str]) -> dict:
    """Turn one raw record into its form, gloss, tags, notes and phonemic value."""
    region, context = record["region"], _context(record)
    raw = record["form"]
    gloss, tags, notes, etymology, unknown = record["gloss"], [], [], "", []

    # A citation counts as IPA only when it is enclosed in slashes or brackets;
    # the examples also use a bare opening bracket to group a complex predicate.
    enclosed = bool(re.fullmatch(r"/.+/|\[.+\]", raw.strip()))
    if region in IPA_REGIONS or enclosed:
        form, phonemic = to_orthography(raw), raw.strip("/[]").strip()
    else:
        form, phonemic = raw.strip("[]"), ""

    tags += REGION_TAGS.get(region, "").split()
    if (region, context.get("column")) in COLUMN_TAGS:
        tags += COLUMN_TAGS[(region, context["column"])].split()

    if region in {"t4", "t5"}:
        donor = context.get("donor", "")
        if donor:
            language = donor.split()[0]
            etymology = (
                f"Hultman compares {donor}, and treats the Kalkoti word as a "
                f"loan from {language}." if region == "t4" else
                f"Hultman compares {donor}, which retains the /h/ Kalkoti has lost."
            )
        if region == "t4":
            notes.append(f"cited as evidence for the loan-exclusive phoneme "
                         f"{context.get('phoneme', '')}".strip())
        if context.get("uncertain"):
            tags.append("uncertain")
            notes.append("the thesis marks this comparison with a question mark")
    elif region == "t3":
        notes.append(
            f"cited for the {context['voice']} {context['place']} stop contrast"
        )
    elif region == "t7":
        phonemic = context.get("phonetic", "").strip("[]")
        notes.append(f"cited for the {context['place']} nasal")
    elif region == "t11":
        tags.append(context["number"])
        if "=" in raw:
            notes.append("the case is marked by the clitic =ä, not by the noun")
        else:
            tags += CASE_TAGS[context["case"]].split()
    elif region == "t14":
        person = context["person"]
        gloss = PRONOUN_GLOSS[person]
        tags += PRONOUN_TAGS[person].split() + CASE_TAGS[context["case"]].split()
    elif region == "t15":
        tags.append(context["gender"])
        notes.append(f"illustrated by the phrase ‘{context['phrase']}’")
    elif region == "t16":
        gloss = numeral_name(context["value"])
        if "-" in form or "kum" in form:
            tags.append("compound")
    elif region == "t17":
        tags.append(context["gender"][0])
        etymology = f"Hultman supports the gender with {context['comparison']}."
    elif region == "t21" and context.get("agreement"):
        tags += AGREEMENT_TAGS.get(context["agreement"], "").split()
    elif region == "t28":
        tags += ["pres" if context["tense"] == "pres" else "pret"]
        if context["polarity"] == "negative":
            tags.append("neg")
        gloss = {
            ("affirmative", "pres"): "is, are", ("affirmative", "past"): "was, were",
            ("negative", "pres"): "is not, are not",
            ("negative", "past"): "was not, were not",
        }[(context["polarity"], context["tense"])]
    elif region == "interlinear":
        # A clitic cited on its own keeps its own labels; only a host word
        # drops the categories that belong to the clitic attached to it.
        host_only = "=" in raw and not raw.lstrip("[(").startswith("=")
        gloss, labels, unknown = read_interlinear(record["gloss"], host_only=host_only)
        tags += labels
        if not gloss:
            # Pronouns, the copula, the negator and a few particles are glossed
            # by category alone. Give them the meaning that category states.
            gloss, extra = _category_gloss(labels)
            if gloss:
                tags += extra.split()
                notes.append("the thesis glosses this word by category alone")
        notes.append(f"cited in example ({context['example']}): "
                     f"‘{context['translation']}’")
    elif region == "prose":
        tags += context.get("tags", "").split()
        if context.get("note"):
            notes.append(context["note"])

    if region in TABLE_OF and "." in gloss:
        gloss, extra = split_table_gloss(gloss)
        if extra:
            tags = [*tags, *extra]
            if VERBAL_LABELS & set(extra):
                tags.append("verb")

    # The thesis appends a question mark to a gloss it is not sure of, and
    # writes a bare question mark for a word it could not gloss at all.
    if gloss.strip() == "?":
        gloss = ""
    elif gloss.endswith("?"):
        gloss = gloss.rstrip("? ")
        tags.append("uncertain")
        notes.append("the thesis marks this gloss as uncertain")
    # A gloss that is a single capitalised word names a person or a place.
    if re.fullmatch(r"[A-Z][a-z]+", gloss):
        tags.append("proper-noun")

    form = form.rstrip(".,")
    if "=" in form and not form.startswith("="):
        notes.append(f"printed as {form}")
        form = form.split("=", 1)[0]
    form = form.replace("-", "")
    return {
        "form": unicodedata.normalize("NFC", form),
        "phonemic": unicodedata.normalize("NFC", phonemic),
        "gloss": gloss.replace("’", "'").strip(),
        "tags": list(dict.fromkeys(t for t in tags if t)),
        "notes": "; ".join(n for n in notes if n.strip()),
        "etymology": etymology, "unknown_labels": unknown,
    }


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

    The thesis cites the same word many times over: driṣ 'see.pfv' appears in
    four tables and a dozen examples. Rows that agree on language, tone-free
    shape and gloss are merged, their citations, tags, notes and phonemic values
    unioned, and the marked spelling kept when only some citations carry tone.
    """
    survivors: dict[tuple[str, ...], dict[str, str]] = {}
    aliases: dict[str, str] = {}
    order: list[tuple[str, ...]] = []
    for row in rows:
        key = (row["Language_ID"], toneless(row["Form"]), row["Gloss"])
        if key in survivors:
            target = survivors[key]
            aliases[row["Entry_Key"]] = target["Entry_Key"]
            if row["Form"] != toneless(row["Form"]) and target["Form"] == toneless(target["Form"]):
                target["Form"] = row["Form"]
            target["Source"] = _union(target["Source"], row["Source"], ";")
            target["Tags"] = _union(target["Tags"], row["Tags"], " ")
            target["Notes"] = _union(target["Notes"], row["Notes"], "; ")
            target["Phonemic"] = _union(target["Phonemic"], row["Phonemic"], "; ")
            target["Etymology"] = target["Etymology"] or row["Etymology"]
            target["Variant_Of_Key"] = target["Variant_Of_Key"] or row["Variant_Of_Key"]
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


# Interlinear tokens that are punctuation or elision rather than words.
SKIP_FORMS = {"", "...", "…", "-", "=", "∅"}


def build() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    forms: list[dict[str, str]] = []
    audit: list[dict[str, str]] = []
    unit_keys: dict[str, str] = {}

    for record in records():
        unit, region = record["unit"], record["region"]
        context = _context(record)
        parsed = interpret(record)
        key = f"{SOURCE_ID}:{unit}"
        status, reason = "installed", ""

        if parsed["form"] in SKIP_FORMS or not parsed["form"].strip(".,=-"):
            status, reason = "skipped", "the token is punctuation or an elision mark"
        elif not parsed["gloss"]:
            status, reason = "skipped", "the thesis marks this word as unglossed"
        elif parsed["unknown_labels"]:
            reason = ("unrecognised interlinear label(s) "
                      + ", ".join(parsed["unknown_labels"]))

        citation = f"{SOURCE_ID}[{locator(record)}]"
        if status == "installed":
            unit_keys[unit] = key
            forms.append(dict(zip(FORM_FIELDS, [
                LANGUAGE_ID, "", parsed["form"], parsed["gloss"], "",
                parsed["phonemic"], parsed["notes"], citation, "", parsed["etymology"],
                key,
                f"{SOURCE_ID}:{context['variant_of']}" if context.get("variant_of") else "",
                "", "", " ".join(parsed["tags"]),
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
            "Hultman, David. 2023. Topics in the grammar of Kalkoti. BA thesis, "
            "Department of Linguistics, Stockholm University."
        ),
        "acquisition": "Open-access full text published in DiVA, Stockholm University",
        "pdf_sha256": PDF_SHA256,
        "pdf_sha512": PDF_SHA512,
        "pdf_pages": PDF_PAGES,
        "pdf_redistributed": False,
        "rights": (
            "Open-access student thesis; no explicit reuse licence is stated, so only "
            "the extracted lexical facts are installed and the PDF is not checked in."
        ),
        "extraction": {
            "method": (
                "deterministic pdfplumber extraction from the thesis's XeTeX text "
                "layer, read in content-stream order; no OCR"
            ),
            "structure_keys": [
                "the text layer holds no space glyphs, so words are recovered by "
                "measuring inter-glyph gaps against the font size",
                "combining tone accents stay on the vowel they follow in the content "
                "stream rather than the one they are drawn over",
                "each table is a fixed grid whose printed line count is asserted",
                "interlinear glosses are matched to their form by shared x-position",
            ],
            "checked_in_layer": str(RAW_SNAPSHOT.relative_to(ROOT)),
            "prose_layer": "the PROSE table in this importer",
            "regions": dict(Counter(row["Region"] for row in audit)),
        },
        "scope": {
            "included": (
                "every Kalkoti form the thesis prints in Tables 3-7, 9-17 and 19-21 "
                "and 23-26 and 28, all 129 numbered interlinear sub-examples, the "
                "phonological illustrations of examples (2)-(4), and the phonemic and "
                "phonetic citations in running prose"
            ),
            "excluded": (
                "the Palula, Gawri, Pashto and Urdu comparanda columns, which are "
                "secondary citations of Liljegren (2016), Baart (1997, 1999) and Baart "
                "and Sagar (2004) and are kept in the audit and in Etymology prose; "
                "Table 1, which lists the datasets rather than words; the phoneme "
                "inventories of Tables 2 and 8; Table 18, which prints bare "
                "inflectional suffixes; and Tables 22 and 27, whose cells repeat "
                "sentences the numbered examples already carry"
            ),
            "etymology_policy": (
                "the thesis makes no reconstruction and cites no CDIAL number, so every "
                "row is installed unlinked; the donor languages of Tables 4, 5 and 13 "
                "and the comparanda of Tables 6, 9, 10 and 17 are recorded as prose "
                "plus a loanword tag, not as graph edges"
            ),
            "language_model": (
                "all forms belong to canonical Kalkoti (Kalk) and carry the registered "
                "Kalkot dialect tag; the four consultants and eleven datasets remain "
                "provenance in the locator and the audit"
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
    parser.add_argument("--pdf", type=Path, help="verify the thesis PDF and re-extract")
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
                text = unicodedata.normalize(
                    "NFC", "".join(c["text"] for c in _chars(page))
                )
                needle = record["form"].strip("/[]").replace(" ", "")
                if needle not in text.replace(" ", ""):
                    raise ValueError(f"{record['unit']}: {needle!r} is not on that page")
        print(f"verified {len(snapshot())} extracted and {len(prose_records())} cited records")

    forms, audit = build()
    counts = Counter(row["Status"] for row in audit)
    print(f"{len(audit)} raw records -> {len(forms)} installed rows: {dict(counts)}")
    if not args.install:
        return
    assert len({row["Entry_Key"] for row in forms}) == len(forms)
    write_csv(FORM_OUTPUT, FORM_FIELDS, forms, header=False)
    write_csv(AUDIT_OUTPUT, AUDIT_FIELDS, audit, header=True)
    sample = sorted(audit, key=lambda row: row["Record_SHA256"])[:25]
    write_csv(SAMPLE_OUTPUT, AUDIT_FIELDS, sample, header=True)
    MANIFEST_OUTPUT.write_text(
        json.dumps(manifest(forms, audit), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"installed {len(forms)} Kalkoti rows from Hultman (2023)")


if __name__ == "__main__":
    main()
