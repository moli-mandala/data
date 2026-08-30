#!/usr/bin/env python3
"""Extract every glossed Sauji lexical item from Knobloch's 2020 grammar sketch.

The thesis is open access on DiVA (diva2:1440556) and ships a genuine LaTeX text
layer, so no OCR is involved. ``extract()`` reads the PDF with pdfplumber word
boxes and recovers the lexical material from eleven structurally distinct regions
(nine numbered tables, the interlinear examples, and the italic-plus-quoted-gloss
citations in running prose). Extraction is keyed on the typeset structure the
thesis actually uses: object-language material is italic, glosses are roman, and
each interlinear gloss shares its word's ``x0``.

The PDF itself is not redistributed. ``20260825-knobloch-sauji-extract.psv`` is
the checked-in snapshot of ``extract()`` so the importer runs without it, and the
``PROSE`` table below carries the handful of citations whose gloss precedes the
form. ``--pdf FILE`` verifies the file against the SHA-256/SHA-512 that DiVA
publishes, asserts that a fresh extraction reproduces the snapshot byte for byte,
and checks that every prose form is still on the page it cites.

Run from ``data/``:

    uv run python data/other/forms/raw_data/knobloch_sauji_2020.py --install
    uv run python data/other/forms/raw_data/knobloch_sauji_2020.py --pdf sauji.pdf
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


SOURCE_ID = "knobloch2020sauji"
BUDDRUSS_ID = "buddruss1967sau"
SNAPSHOT_DATE = "2026-08-25"
PDF_SHA256 = "38e02514183aa4b7d6a424e81a06f27d31d99b0935af87a49488ebbab859da38"
PDF_SHA512 = (
    "74d44376dae060d84819817c919157c439cdc894418dcc52754a303ef5b3097f"
    "d2fe0f857f5f8482908cf1a61f4b5ea08d945d8bcd09a23ba0cc35f8bb27905a"
)
PDF_PAGES = 55
PRINTED_PAGE_OFFSET = 4  # PDF page 17 carries the printed page number 13
LANGUAGE_ID = "Sv"
DIALECT_TAG = "dialect:Sv:HKAT-sdg:Sau"

ROOT = Path(__file__).resolve().parents[4]
RAW_DIR = ROOT / "data/other/forms/raw_data"
FORM_OUTPUT = ROOT / "data/other/forms/20260825-knobloch-sauji.csv"
AUDIT_OUTPUT = RAW_DIR / "20260825-knobloch-sauji-audit.csv"
SAMPLE_OUTPUT = RAW_DIR / "20260825-knobloch-sauji-sample.csv"
MANIFEST_OUTPUT = RAW_DIR / "20260825-knobloch-sauji-manifest.json"
RAW_SNAPSHOT = RAW_DIR / "20260825-knobloch-sauji-extract.psv"

ITALIC_FONTS = {"LinLibertineOI", "LinLibertineOBI"}
OPEN_Q, CLOSE_Q = "‘", "’"

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
    "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
    "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "Snapshot_Date", "Unit_ID", "Region", "PDF_Page", "Printed_Page", "Locator",
    "Raw_Form", "Raw_Gloss", "Raw_Context", "Status", "Reason", "Final_Form",
    "Final_Gloss", "Final_Phonemic", "Final_Tags", "Emitted_Key", "Merged_Into", "Source",
    "Record_SHA256",
]


# --------------------------------------------------------------------------
# PDF extraction
# --------------------------------------------------------------------------

def _tokens(page) -> list[dict]:
    """Merge pdfplumber words back into printed words, tracking italics.

    ``extract_words`` splits at every font change, so a single printed word such
    as ``maanuṣ-a`` (roman stem, bold-italic suffix) arrives as two boxes. Words
    typeset without an intervening space are rejoined; a word counts as italic
    when any constituent box is italic.
    """
    words = page.extract_words(
        x_tolerance=1.2, y_tolerance=2, keep_blank_chars=False,
        use_text_flow=False, extra_attrs=["fontname"],
    )
    by_line: dict[float, list[dict]] = defaultdict(list)
    for word in words:
        by_line[round(word["top"], 1)].append(word)
    lines = []
    for top in sorted(by_line):
        merged: list[dict] = []
        for word in sorted(by_line[top], key=lambda w: w["x0"]):
            italic = word["fontname"].split("+")[-1] in ITALIC_FONTS
            if merged and word["x0"] - merged[-1]["x1"] < 0.9:
                merged[-1]["text"] += word["text"]
                merged[-1]["x1"] = word["x1"]
                merged[-1]["italic"] = merged[-1]["italic"] or italic
            else:
                merged.append({
                    "text": word["text"], "x0": word["x0"], "x1": word["x1"],
                    "top": top, "italic": italic,
                })
        lines.append((top, merged))
    return lines


def _cell(line: list[dict], lo: float, hi: float) -> str:
    return " ".join(t["text"] for t in line if lo <= t["x0"] < hi).strip()


def _band(lines, lo: float, hi: float):
    return [(top, toks) for top, toks in lines if lo <= top <= hi]


_QUOTED = re.compile(f"{OPEN_Q}([^{OPEN_Q}{CLOSE_Q}]*){CLOSE_Q}")


def _split_gloss(cell: str) -> tuple[str, str]:
    """Split ``paːɬu 'leaf'`` into form and gloss, tolerating the source's typos."""
    match = _QUOTED.search(cell)
    if not match:
        return cell.strip(), ""
    form = cell[: match.start()].strip()
    gloss = match.group(1).strip()
    trailing = cell[match.end():].strip()
    if not form and trailing:
        # p. 13 prints ``'ʃaŋko' wood``: the quotes enclose the form, not the gloss.
        return gloss, trailing
    return form, (gloss + (" " + trailing if trailing and not form else "")).strip()


def _record(unit, region, pdf_page, form, gloss, context="") -> dict[str, str]:
    return {
        "unit": unit, "region": region, "pdf_page": str(pdf_page),
        "form": unicodedata.normalize("NFC", form.strip()),
        "gloss": unicodedata.normalize("NFC", gloss.strip()),
        "context": unicodedata.normalize("NFC", context.strip()),
    }


def _table2(pdf) -> list[dict]:
    """p. 13, Table 2: consonant phonemes with initial/medial/final examples."""
    lines = _band(_tokens(pdf.pages[16]), 214.0, 662.0)
    columns = [("initial", 121.0, 256.0), ("medial", 255.0, 418.0), ("final", 417.0, 999.0)]
    out: list[dict] = []
    pending: dict[str, dict] = {}
    consonant = ""
    for _, toks in lines:
        label = _cell(toks, 80.0, 121.0)
        if label:
            consonant = label
            pending = {}
        for name, lo, hi in columns:
            cell = _cell(toks, lo, hi)
            if not cell or cell == "—":
                continue
            if cell.startswith("(Buddruss"):
                if name in pending:
                    pending[name]["context"] = cell
                continue
            form, gloss = _split_gloss(cell)
            # ``qaziː 'judge' (Buddruss, 1967: 125)`` puts the citation inline.
            inline = re.search(r"\(Buddruss[^)]*\)", gloss)
            context = ""
            if inline:
                context = inline.group(0)
                gloss = gloss[: inline.start()].strip()
            record = _record(f"t2:{consonant}:{name}", "table2", 17, form, gloss, context)
            pending[name] = record
            out.append(record)
    return out


def _table4(pdf) -> list[dict]:
    """p. 14, Table 4: vowel phonemes with broadly transcribed examples."""
    out = []
    for _, toks in _band(_tokens(pdf.pages[17]), 677.0, 757.0):
        phoneme = _cell(toks, 68.0, 100.0)
        cell = _cell(toks, 320.0, 460.0)
        for index, chunk in enumerate(cell.split(",")):
            form, gloss = _split_gloss(chunk.strip())
            if not form:
                continue
            out.append(_record(
                f"t4:{phoneme}:{index + 1}", "table4", 18, form.strip("/"), gloss,
                f"phoneme /{phoneme}/",
            ))
    return out


def _phonotactics(pdf) -> list[dict]:
    """p. 15, example (1): syllable-structure examples in broad transcription."""
    out = []
    for _, toks in _band(_tokens(pdf.pages[18]), 140.0, 215.0):
        text = " ".join(t["text"] for t in toks if t["x0"] >= 110.0)
        shape, _, examples = text.partition(" - ")
        for index, chunk in enumerate(examples.split(",")):
            form, gloss = _split_gloss(chunk.strip())
            form = form.strip("/ ")
            if not form or not gloss:
                continue
            out.append(_record(
                f"phon:{shape}:{index + 1}", "phonotactics", 19, form, gloss,
                f"syllable structure {shape}",
            ))
    return out


def _paradigm(pdf, page_index, top, bottom, table, labels, columns, printed) -> list[dict]:
    """Read a fixed-column paradigm grid (Tables 6, 7 and 8)."""
    out = []
    row_label = ""
    seen: Counter[tuple[str, str]] = Counter()
    for _, toks in _band(_tokens(pdf.pages[page_index]), top, bottom):
        printed_label = _cell(toks, labels[0], labels[1])
        if printed_label:
            row_label = printed_label
        for name, lo, hi in columns:
            cell = _cell(toks, lo, hi)
            for chunk in cell.split():
                form = chunk.strip()
                if not form:
                    continue
                seen[(row_label, name)] += 1
                ordinal = seen[(row_label, name)]
                out.append(_record(
                    f"{table}:{row_label}:{name}:{ordinal}", table, printed + PRINTED_PAGE_OFFSET,
                    form, "", f"{row_label} {name}",
                ))
    return out


def _table9(pdf) -> list[dict]:
    """p. 24, Table 9: cardinal numerals and their composition."""
    out = []
    for _, toks in _band(_tokens(pdf.pages[27]), 132.0, 760.0):
        numeral = _cell(toks, 185.0, 240.0)
        form = _cell(toks, 240.0, 330.0)
        composition = _cell(toks, 330.0, 999.0)
        if not numeral or not form:
            continue
        out.append(_record(
            f"t9:{numeral}", "table9", 28, form, numeral, f"composition {composition}",
        ))
    return out


def _table10(pdf) -> list[dict]:
    """p. 28, Table 10: imperfective and perfective stem formation."""
    out = []
    verb_class = ""
    for _, toks in _band(_tokens(pdf.pages[31]), 119.0, 232.0):
        label = _cell(toks, 130.0, 200.0)
        if label:
            verb_class = label.replace(" ", "")
        gloss = _split_gloss(_cell(toks, 200.0, 258.0))[1]
        for name, lo, hi in (("ipfv", 258.0, 380.0), ("pfv", 380.0, 999.0)):
            cell = _cell(toks, lo, hi)
            if not cell:
                continue
            # ``thaan- (m)/ theen- (f)`` prints two gender-specific stems.
            parts = [p.strip() for p in cell.replace("/", " ").split() if p.strip()]
            gender = ""
            emitted = 0
            for part in parts:
                if part in {"(m)", "(f)"}:
                    if emitted:
                        out[-1]["context"] += f" {part}"
                    gender = part
                    continue
                emitted += 1
                out.append(_record(
                    f"t10:{verb_class}:{gloss}:{name}:{emitted}", "table10", 32,
                    part, gloss, f"{name} stem",
                ))
    return out


def _table11(pdf) -> list[dict]:
    """p. 29, Table 11: composition of the TMA categories."""
    lines = _band(_tokens(pdf.pages[32]), 118.0, 320.0)
    # Each block prints its category label on the middle of three lines, with one
    # example above and one below, so examples are assigned to the nearest label.
    labels = [(top, _cell(toks, 68.0, 160.0).replace(" ", ""))
              for top, toks in lines if _cell(toks, 68.0, 160.0)]
    out = []
    counts: Counter[str] = Counter()
    for top, toks in lines:
        cell = _cell(toks, 350.0, 999.0)
        if not cell or cell.isdigit():  # a bare digit is the raised footnote marker
            continue
        form, gloss = _split_gloss(cell)
        if not form:
            continue
        category = min(labels, key=lambda pair: abs(pair[0] - top))[1]
        counts[category] += 1
        out.append(_record(
            f"t11:{category}:{counts[category]}", "table11", 33, form, gloss,
            f"TMA category {category}",
        ))
    return out


def _table14(pdf) -> list[dict]:
    """p. 38, Table 14: the Sauji column of the Sauji/Palula/Kalkoti cognate set."""
    out = []
    for index, (_, toks) in enumerate(_band(_tokens(pdf.pages[41]), 384.0, 548.0), start=1):
        gloss_cell = _cell(toks, 140.0, 240.0)
        sauji = _cell(toks, 240.0, 314.0)
        glosses = _QUOTED.findall(gloss_cell)
        primary = glosses[0] if glosses else gloss_cell
        secondary = re.search(r"\(([^)]*)\)", gloss_cell)
        parts = sauji.split()
        for position, part in enumerate(parts):
            bracketed = part.startswith("(")
            gloss = (secondary.group(1) if bracketed and secondary else primary)
            out.append(_record(
                f"t14:{index}:{position + 1}", "table14", 42, part.strip("()"), gloss,
                f"cognate row {index}" + (" (bracketed alternate)" if bracketed else ""),
            ))
    return out


_EXAMPLE = re.compile(r"^\((\d{1,3})\)$")
_SUBLABEL = re.compile(r"^([a-f])\.$")
_REFERENCE = re.compile(r"^\(([0-9]{1,2}_[A-Z]+_[0-9]{6})\)[.,]?$")


def _interlinear(pdf) -> list[dict]:
    """Interlinear examples: italic object line over an x-aligned roman gloss line.

    Object lines are indented past the example number, so prose lines that merely
    contain italics never qualify. Glosses are matched on ``x0`` within 2.5 pt,
    which absorbs the small kerning offsets around bracketed ellipses.
    """
    out: list[dict] = []
    example = ""
    sublabel = ""
    pending: list[dict] = []
    tiers: Counter[str] = Counter()
    for page_index in range(18, PDF_PAGES):
        lines = _tokens(pdf.pages[page_index])
        for position, (top, toks) in enumerate(lines):
            for token in toks:
                reference = _REFERENCE.match(token["text"])
                if reference:
                    for record in pending:
                        record["context"] = f"{record['context']}; {reference.group(1)}".strip("; ")
                    pending = []
            for token in toks:
                if token["italic"]:
                    continue
                number = _EXAMPLE.match(token["text"])
                if number and token["x0"] < 115.0:
                    example, sublabel = number.group(1), ""
                    pending = []
                letter = _SUBLABEL.match(token["text"])
                if letter and token["x0"] < 115.0:
                    sublabel = letter.group(1)
            # A genuine object line is set entirely in italics apart from its
            # example number and sub-label; justified prose that merely mentions
            # an italic form never reaches that ratio.
            # The quotation marks that frame the quoted example (56) are typeset in
            # the object language's italics but are not words and take no gloss.
            body = [t for t in toks if t["x0"] >= 95.0 and t["text"] not in {"“", "”"}
                    and not _EXAMPLE.match(t["text"]) and not _SUBLABEL.match(t["text"])]
            italics = [t for t in body if t["italic"]]
            if not italics or len(italics) < 0.8 * len(body) or position + 1 >= len(lines):
                continue
            next_top, next_toks = lines[position + 1]
            if not 0 < next_top - top < 20:
                continue
            glosses = [t for t in next_toks if not t["italic"]]
            matched = 0
            paired = []
            for token in italics:
                best = min(glosses, key=lambda g: abs(g["x0"] - token["x0"]), default=None)
                if best is not None and abs(best["x0"] - token["x0"]) <= 2.5:
                    matched += 1
                    paired.append((token, best["text"]))
                else:
                    paired.append((token, ""))
            if matched < max(1, len(italics) - 2):
                continue
            # Long examples run over several object lines; number the tiers so
            # that a word's key stays unique and stable within its example.
            tiers[f"{example}{sublabel}"] += 1
            tier = tiers[f"{example}{sublabel}"]
            for index, (token, gloss) in enumerate(paired, start=1):
                out.append(_record(
                    f"ex{example}{sublabel}:t{tier}:w{index}", "interlinear", page_index + 1,
                    token["text"].rstrip(".,"), gloss,
                    f"example ({example}){sublabel} line {tier}",
                ))
            pending.extend(out[-len(paired):])
    return out


def _inline(pdf) -> list[dict]:
    """Running-prose citations: an italic run followed by a roman quoted gloss."""
    out = []
    for page_index in range(15, 44):
        lines = _tokens(pdf.pages[page_index])
        flat = [t for _, toks in lines for t in toks]
        counter = 0
        index = 0
        while index < len(flat):
            if not flat[index]["italic"]:
                index += 1
                continue
            end = index
            while (end + 1 < len(flat) and flat[end + 1]["italic"]
                   and flat[end + 1]["top"] == flat[index]["top"]):
                end += 1
            form = " ".join(t["text"] for t in flat[index:end + 1])
            cursor = end + 1
            if cursor < len(flat) and flat[cursor]["text"].startswith(OPEN_Q):
                parts = []
                while cursor < len(flat):
                    text = flat[cursor]["text"]
                    parts.append(text)
                    # Glosses contain apostrophes (``wife's sister``), so only a
                    # token that opens or closes with ’ ends the quotation.
                    if text.rstrip(".,;:)").endswith(CLOSE_Q) or text.startswith(CLOSE_Q):
                        break
                    cursor += 1
                gloss = " ".join(parts)
                if CLOSE_Q in gloss and len(gloss) <= 80:
                    counter += 1
                    out.append(_record(
                        f"p{page_index + 1}:c{counter}", "inline", page_index + 1,
                        form.rstrip(",.;:"),
                        gloss[gloss.index(OPEN_Q) + 1: gloss.rindex(CLOSE_Q)],
                    ))
            index = end + 1
    return out


def extract(pdf_path: Path) -> list[dict[str, str]]:
    """Deterministically recover every raw lexical record from the thesis PDF."""
    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        if len(pdf.pages) != PDF_PAGES:
            raise ValueError(f"expected {PDF_PAGES} pages, found {len(pdf.pages)}")
        records = [
            *_table2(pdf),
            *_table4(pdf),
            *_phonotactics(pdf),
            *_paradigm(pdf, 23, 206.0, 247.0, "t6", (190.0, 285.0), (
                ("1sg", 285.0, 313.0), ("2sg", 313.0, 342.0),
                ("1pl", 342.0, 374.0), ("2pl", 374.0, 410.0)), 20),
            *_paradigm(pdf, 24, 594.0, 636.0, "t7", (70.0, 100.0), (
                ("prox.det.sg", 100.0, 155.0), ("prox.det.pl", 155.0, 208.0),
                ("prox.pron.sg", 208.0, 262.0), ("prox.pron.pl", 262.0, 315.0),
                ("rem.det.sg", 315.0, 369.0), ("rem.det.pl", 369.0, 423.0),
                ("rem.pron.sg", 423.0, 477.0), ("rem.pron.pl", 477.0, 540.0)), 21),
            *_paradigm(pdf, 25, 671.0, 740.0, "t8", (84.0, 120.0), (
                ("prox.det.sg", 120.0, 176.0), ("prox.det.pl", 176.0, 219.0),
                ("prox.pron.sg", 219.0, 273.0), ("prox.pron.pl", 273.0, 316.0),
                ("rem.det.sg", 316.0, 370.0), ("rem.det.pl", 370.0, 413.0),
                ("rem.pron.sg", 413.0, 467.0), ("rem.pron.pl", 467.0, 540.0)), 22),
            *_table9(pdf),
            *_table10(pdf),
            *_table11(pdf),
            *_table14(pdf),
            *_interlinear(pdf),
            *_inline(pdf),
        ]
    if len({r["unit"] for r in records}) != len(records):
        duplicates = [u for u, n in Counter(r["unit"] for r in records).items() if n > 1]
        raise ValueError(f"duplicate unit ids: {duplicates[:10]}")
    return records


# --------------------------------------------------------------------------
# Prose citations the two automatic patterns cannot reach
# --------------------------------------------------------------------------

# The thesis also names forms in running prose without the italic-plus-quoted-gloss
# frame the ``inline`` extractor keys on -- usually because the gloss precedes the
# form ("the plural of 'tongue' ǰib is clearly ǰiba") or because a paradigm is
# listed bare. Each row was read off the printed page; ``--pdf`` re-checks that the
# form and its cited page still agree with the text layer.
# printed page | form | gloss | context
PROSE = r"""
17|brawu|brothers|irregular plural of bra 'brother'
17|ǰitaka|boys|suppletive plural of pu 'boy'
17|ǰib|tongue|Buddruss's example rechecked against the recent data
17|ǰiba|tongues|plural of ǰib
17|beeṇ|sister|Buddruss's example rechecked against the recent data
17|beeṇa|sisters|plural of beeṇ
20|maṭee|to me|dative pronoun reported by Buddruss (1967: 39)
20|tuṭee|to you (singular)|dative pronoun reported by Buddruss (1967: 39)
20|asonṭee|to us|dative pronoun reported by Buddruss (1967: 39)
20|tusoṇṭee|to you (plural)|dative pronoun reported by Buddruss (1967: 39)
21|asan|us|postpositional pronoun reported by Buddruss (1967: 39); final a nasalized
21|asa|us|postpositional pronoun reported by Buddruss (1967: 39); final a nasalized
21|tusan|you (plural)|postpositional pronoun reported by Buddruss (1967: 39); final a nasalized
21|tusa|you (plural)|postpositional pronoun reported by Buddruss (1967: 39); final a nasalized
21|asondiyo|from us|postpositional pronoun plus the postposition diyo
23|paanǰbisa|hundred|spelling of the 100-component in the prose analysis of 110
25|sawa|hundred|Pashto loan used for the 100-component of 700
29|khomno|will eat|assimilated pronunciation of khamno reported by Buddruss (1967: 54)
40|musafari|journey|f-variant of the same word
40|musaphari|journey|pʰ-variant of the same word
"""


def prose_records() -> list[dict[str, str]]:
    out = []
    counts: Counter[str] = Counter()
    for row in csv.reader(io.StringIO(PROSE.strip()), delimiter="|"):
        page, form, gloss, context = (cell.strip() for cell in row)
        counts[page] += 1
        out.append(_record(
            f"prose:p{page}:{counts[page]}", "prose",
            int(page) + PRINTED_PAGE_OFFSET, form, gloss, context,
        ))
    return out


# --------------------------------------------------------------------------
# Gloss and tag curation
# --------------------------------------------------------------------------

# Leipzig-style gloss abbreviations used in the thesis, mapped onto Jambu's
# canonical tag vocabulary in tags.py.
GRAMMATICAL = {
    "nom": "nom", "acc": "acc", "erg": "erg", "gen": "gen", "obl": "obl",
    "abl": "abl", "loc": "loc", "dat": "dat", "postp": "postp",
    "sg": "sg", "pl": "pl", "m": "m", "f": "f", "F": "f",
    "msg": "m sg", "fsg": "f sg", "mpl": "m pl", "fpl": "f pl",
    "ipfv": "ipfv", "pfv": "pfv", "prs": "pres", "pst": "pret", "fut": "fut",
    "imp": "impv", "neg": "neg", "red": "reduplicated",
    "1sg": "1sg pron", "2sg": "2sg pron", "3sg": "3sg pron",
    "1pl": "1pl pron", "2pl": "2pl pron", "3pl": "3pl pron",
    "det": "determiner demonstrative", "prox": "prox", "rem": "dist",
    "qm": "interr part",
}
# Glosses that carry no lexical component at all name a pronoun or a particle;
# the lexical value follows from the person/deixis combination, not from guesswork.
DEICTIC_GLOSS = {
    ("1sg",): "I", ("1pl",): "we",
    ("2sg",): "you (singular)", ("2pl",): "you (plural)",
    ("3sg", "prox"): "he, she, it (proximate)",
    ("3sg", "rem"): "he, she, it (remote)",
    ("3pl", "prox"): "they (proximate)", ("3pl", "rem"): "they (remote)",
    ("det", "prox"): "this", ("det", "rem"): "that",
    ("neg",): "not", ("qm",): "question marker",
}
# Author's glosses whose printed shape does not survive the mechanical split:
# line-break hyphenation, slashes inside a lexical gloss, and stray punctuation.
GLOSS_OVERRIDES = {
    "ques- tion": ("question", ""),
    "killed)": ("killed", ""),
    "seven-hundred": ("seven hundred", "num"),
    "some-also": ("some, also", ""),
    "pashto-language": ("Pashto language", "noun"),
    "Sauji.much-obl": ("Sauji language", "noun obl"),
    "neg-be.pst-f.sg": ("not be", "verb neg pret f sg"),
    "1sg.acc-to": ("to me", "1sg pron acc"),
    "hand-obl.pl-with": ("hand", "noun obl pl"),
    "battle/conflict(F)": ("battle, conflict", "noun f"),
    "bear(fsg.nom)": ("bear", "noun f sg nom"),
    "sit.pfv-m.sg-prs-m.sg": ("sit", "verb pfv m sg pres"),
    "sit.pfv-m.pl-prs-m.pl": ("sit", "verb pfv m pl pres"),
    "grow.pfv-m.pl-prs-m.pl": ("grow", "verb pfv m pl pres"),
    "eat.pfv-m.sg-prs-m.sg": ("eat", "verb pfv m sg pres"),
    "become/go.ipfv-pst-m.pl": ("become, go", "verb ipfv pret m pl"),
    "say/speak/read.ipfv-f.sg": ("say, speak, read", "verb ipfv f sg"),
    "two/both": ("two, both", "num"),
    "1.sg": ("I", "1sg pron"), "2.sg": ("you (singular)", "2sg pron"),
    "1.pl": ("we", "1pl pron"),
    "be (prs.msg)": ("be", "verb copula pres m sg"),
    "be (pst.msg)": ("be", "verb copula pret m sg"),
    "be (mpl)": ("be", "verb copula m pl"),
    "be (fpl)": ("be", "verb copula f pl"),
    "do (simple present)": ("do", "verb pres"),
    "see (simple past)": ("see", "verb pret"),
    "hear (prs.msg)": ("hear", "verb pres m sg"),
    "1pl.erg": ("we", "1pl pron erg"),
    "wood": ("wood", "noun"),
}
# Proper names glossed with an English rendering of the name itself.
PROPER_NOUNS = {
    "Sau", "Sau-valley", "Azgiral", "Shergal", "Torsan", "Zangabosho",
    "Kamtili", "Zahir", "Sha", "Dawud", "Saleem",
}
# Region-level tags that follow from where the thesis prints the form.
REGION_TAGS = {
    "table9": "num", "t6": "pron personal", "t7": "demonstrative",
    "t8": "demonstrative", "table10": "verb stem", "table11": "verb",
}
# Extra tags carried by curated groups of prose citations.
UNIT_TAGS = {
    **{f"p30:c{n}": "postp" for n in range(1, 21)},
    **{f"p38:c{n}": "interr pron" for n in range(1, 7)},
    "p23:c1": "postp", "p24:c1": "postp", "p27:c1": "postp", "p29:c2": "postp",
    "p25:c2": "postp", "p27:c6": "postp",
    "p27:c2": "adj", "p27:c3": "adj", "p27:c4": "adj degree", "p27:c5": "noun pl",
    "p29:c1": "num", "p33:c15": "verb fut", "p33:c16": "verb fut",
    "p33:c17": "verb fut", "p33:c18": "verb impv", "p33:c19": "verb impv",
    "p33:c20": "verb caus", "p33:c21": "verb", "p33:c22": "verb pfv",
    "p33:c23": "verb pfv caus", "p43:c3": "verb fut", "p43:c2": "verb copula pres",
    "p20:c7": "noun m", "p20:c8": "noun f", "p20:c9": "noun m", "p20:c10": "noun f",
    "p20:c11": "noun m", "p20:c12": "noun f", "p20:c13": "noun m", "p20:c14": "noun f",
    "p21:c2": "noun pl", "p21:c4": "noun pl", "p21:c6": "noun pl", "p21:c8": "noun pl",
    "p21:c10": "noun pl", "p21:c12": "noun pl", "p21:c14": "noun pl", "p21:c16": "noun pl",
    "p21:c1": "noun m", "p21:c3": "noun m", "p21:c5": "noun f", "p21:c7": "noun m",
    "p21:c9": "noun m", "p21:c11": "noun f", "p21:c13": "noun f", "p21:c15": "noun f",
    "p21:c17": "noun", "p21:c18": "noun",
    "p20:c1": "noun", "p20:c2": "noun", "p20:c3": "noun f", "p20:c4": "noun f",
    "p20:c5": "noun", "p20:c6": "noun", "p20:c15": "noun", "p20:c16": "noun",
    "p22:c1": "noun", "p22:c2": "noun",
    "p22:c3": "noun f", "p23:c2": "multiword-expression", "p23:c3": "multiword-expression",
    "p23:c4": "noun", "p23:c5": "noun",
    **{f"t14:{n}:1": "verb pres" for n in range(1, 6)},
    **{f"t14:{n}:2": "verb pret" for n in range(1, 6)},
    "t14:6:1": "noun", "t14:7:1": "noun",
    **{f"t14:{n}:1": "num" for n in range(11, 14)},
    "prose:p17:1": "noun pl", "prose:p17:2": "noun pl", "prose:p17:3": "noun",
    "prose:p17:4": "noun pl", "prose:p17:5": "noun", "prose:p17:6": "noun pl",
    "prose:p20:1": "pron 1sg dat", "prose:p20:2": "pron 2sg dat",
    "prose:p20:3": "pron 1pl dat", "prose:p20:4": "pron 2pl dat",
    "prose:p21:1": "pron 1pl postp", "prose:p21:2": "pron 1pl postp",
    "prose:p21:3": "pron 2pl postp", "prose:p21:4": "pron 2pl postp",
    "prose:p21:5": "pron 1pl postp", "prose:p23:1": "num", "prose:p25:1": "num loanword",
    "prose:p29:1": "verb fut", "prose:p40:1": "noun", "prose:p40:2": "noun",
    "p16:c2": "noun", "p16:c1": "noun",
}
# The thesis names the donor language for these forms in section 5.2.3.
LOANWORDS = {
    "p43:c4": "Gawarbati", "p43:c5": "Gawarbati", "p43:c6": "Gawarbati",
    "p43:c7": "Gawarbati", "p43:c8": "Pashto", "p43:c9": "Pashto",
    "p43:c10": "Pashto", "p43:c11": "Pashto", "p43:c12": "Pashto",
    "prose:p25:1": "Pashto",
}
# Records that must not enter the installed CSV, with the reason recorded in the audit.
EXCLUSIONS = {
    **{f"p32:c{n}": "table 10 cell misread by the prose pattern; installed from the t10 grid instead"
       for n in range(1, 5)},
    **{f"p33:c{n}": "table 11 cell misread by the prose pattern; installed from the t11 grid instead"
       for n in range(1, 15)},
    **{f"p42:c{n}": "table 14 cell misread by the prose pattern; installed from the t14 grid instead"
       for n in range(1, 13)},
    "p32:c5": "Palula phed-áan-u cited for comparison, not a Sauji form",
    "p32:c6": "Palula phedíl-u cited for comparison, not a Sauji form",
    "p43:c1": "cited as the Palula copula in a comparison of Palula perfect formation",
    "p22:c4": "free translation of example (13), not a citation form",
}
# Elicitation references the thesis supplies outside the example block itself.
EXTERNAL_REFERENCES = {
    "ex29": "46_FS_100213",
    **{f"ex{n}": "14_HR_000519" for n in (77, 78, 79)},
    **{f"ex{n}": "20_NU_000819" for n in (80, 81, 82, 83)},
    **{f"ex{n}": "10_MAN_000512" for n in range(84, 93)},
}
# Printed abbreviations that stand for two forms at once.
ALTERNATES = {
    "pa/patuno": ("pa", "patuno"),
    "yomo/yomi": ("yomo", "yomi"),
    "kaaree/karee": ("kaaree", "karee"),
    "pašu(w)anu": ("pašuwanu", "pašuanu"),
    "-u/-oo": ("-u", "-oo"),
}
# Verb and noun inflection classes the thesis assigns explicitly.
VERB_CLASSES = {
    "1(-il)": "Sauji-verb-class-1", "2(-al)": "Sauji-verb-class-2",
    "3(-t)": "Sauji-verb-class-3", "4(suppletive)": "Sauji-verb-class-4",
}
NOUN_CLASSES = {
    "p20:c1": "Sauji-noun-class-1", "p20:c4": "Sauji-noun-class-1",
    "p20:c2": "Sauji-noun-class-2", "p20:c5": "Sauji-noun-class-2",
    "p20:c3": "Sauji-noun-class-3", "p20:c6": "Sauji-noun-class-3",
}
IPA_REGIONS = {"table2", "table4", "phonotactics"}


def parse_gloss(raw: str) -> tuple[str, list[str], bool]:
    """Split an interlinear or table gloss into a lexical gloss and canonical tags.

    Returns ``(gloss, tags, uncertain)``. Only the first hyphen-delimited morpheme
    can contribute a lexical gloss; every later morpheme is affixal. A trailing
    question mark is the author's own hedge and becomes an ``uncertain`` tag rather
    than part of the definition.
    """
    if raw in GLOSS_OVERRIDES:
        gloss, tags = GLOSS_OVERRIDES[raw]
        return gloss, tags.split(), False
    uncertain = raw.endswith("?") or raw == "?"
    text = raw.rstrip("?").strip()
    if not text:
        return "", [], uncertain
    tags: list[str] = []
    identity: list[str] = []
    lexical: list[str] = []
    morphemes = text.split("-")
    for index, morpheme in enumerate(morphemes):
        # ``man(nom.sg)`` encodes categories in parentheses, but ``after (temporal)``
        # and ``you (singular)`` are part of the definition; only a parenthesis whose
        # whole content is a known abbreviation is read as grammar.
        def _categorical(match: re.Match) -> str:
            inner = match.group(1)
            if inner == "?":
                return "."
            if all(part in GRAMMATICAL for part in inner.split(".") if part):
                return "." + inner
            return match.group(0)

        morpheme = re.sub(r"\(([^)]*)\)", _categorical, morpheme)
        uncertain = uncertain or "(?)" in text
        for part in morpheme.split("."):
            part = part.strip()
            if not part:
                continue
            if part in GRAMMATICAL:
                tags.extend(GRAMMATICAL[part].split())
                if part in {"det", "prox", "rem", "neg", "qm"} or re.fullmatch(
                    r"[123](sg|pl)", part
                ):
                    identity.append(part)
            elif index == 0:
                lexical.append(part)
    gloss = " ".join(lexical).replace("/", ", ").replace(" )", ")").strip()
    if not gloss:
        gloss = DEICTIC_GLOSS.get(tuple(dict.fromkeys(identity)), "")
    cases = ("nom", "acc", "obl", "gen", "abl", "loc", "erg", "dat")
    if any(tag in tags for tag in ("ipfv", "pfv", "pres", "pret", "fut", "impv")):
        tags.append("verb")
    elif gloss and not identity and any(tag in tags for tag in cases):
        tags.append("noun")
    elif gloss and not identity and any(tag in tags for tag in ("m", "f")):
        # Gender and number without a case suffix is agreement on a modifier.
        tags.append("adj")
    if raw in PROPER_NOUNS or (raw and raw[0].isupper() and raw in PROPER_NOUNS):
        tags.append("proper-noun")
    return gloss, list(dict.fromkeys(tags)), uncertain


def locator(record: dict[str, str]) -> str:
    """Build the CLDF locator from the record's own printed position."""
    unit, region = record["unit"], record["region"]
    printed = int(record["pdf_page"]) - PRINTED_PAGE_OFFSET
    parts = unit.split(":")
    if region == "table2":
        return f"p. {printed}, Table 2, /{parts[1]}/ {parts[2]}"
    if region == "table4":
        return f"p. {printed}, Table 4, /{parts[1]}/ example {parts[2]}"
    if region == "phonotactics":
        return f"p. {printed}, example (1) {parts[1]}"
    if region in {"t6", "t7", "t8"}:
        number = {"t6": 6, "t7": 7, "t8": 8}[region]
        return f"p. {printed}, Table {number}, {parts[1]} {parts[2]}"
    if region == "table9":
        return f"p. {printed}, Table 9, numeral {parts[1]}"
    if region == "table10":
        return f"p. {printed}, Table 10, class {parts[1]} {parts[3]} stem"
    if region == "table11":
        return f"p. {printed}, Table 11, {parts[1]}"
    if region == "table14":
        return f"p. {printed}, Table 14, row {parts[1]}"
    if region == "interlinear":
        reference = re.search(r"([0-9]{1,2}_[A-Z]+_[0-9]{6})", record["context"])
        example = record["context"].split(" line ")[0].replace("example ", "")
        suffix = f", data {reference.group(1)}" if reference else ""
        return f"p. {printed}, example {example} word {parts[-1][1:]}{suffix}"
    return f"p. {printed}"


def serialise(records: list[dict[str, str]]) -> str:
    handle = io.StringIO()
    writer = csv.writer(handle, delimiter="|", lineterminator="\n")
    for record in records:
        writer.writerow([record[key] for key in ("unit", "region", "pdf_page", "form", "gloss", "context")])
    return handle.getvalue()


def deserialise(text: str) -> list[dict[str, str]]:
    keys = ("unit", "region", "pdf_page", "form", "gloss", "context")
    return [dict(zip(keys, row)) for row in csv.reader(io.StringIO(text), delimiter="|") if row]


def snapshot() -> list[dict[str, str]]:
    return deserialise(RAW_SNAPSHOT.read_text(encoding="utf-8"))


def records() -> list[dict[str, str]]:
    """Every raw source record: the checked-in PDF extraction plus prose citations."""
    return [*snapshot(), *prose_records()]


# --------------------------------------------------------------------------
# Form emission
# --------------------------------------------------------------------------

_ONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
         "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
         "seventeen", "eighteen", "nineteen"]
_TENS = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy"}


def numeral_gloss(value: str) -> str:
    number = int(value)
    if number < 20:
        return _ONES[number]
    if number in _TENS:
        return _TENS[number]
    if number < 40:
        return f"{_TENS[number // 10 * 10]}-{_ONES[number % 10]}"
    return {100: "hundred", 110: "hundred and ten", 120: "hundred and twenty"}[number]


# The source's own slips, corrected with the reason kept in the audit.
FORM_FIXES = {
    "t2:l:medial": ("alo", "be (pst.msg)",
                    "the printed cell leaves the gloss's closing quote open"),
    "p16:c2": ("musaaferi", "journey", "the printed citation opens with an unclosed parenthesis"),
}
BUDDRUSS_PAGE = re.compile(r"Buddruss,? 1967:\s*([0-9-]+)")


def collapse(forms: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, str]]:
    """Fold repeated attestations of one lexeme into a single row.

    The thesis prints the same word many times -- ``aw`` 'and' appears in two dozen
    examples -- and every occurrence is a real citation of the same lexeme. Rows
    are folded on form plus gloss, which keeps genuine homographs such as *si*
    'bridge' and *si* 'together with' apart, and unions their locators, tags,
    notes and phonemic transcriptions. The first record in printed order supplies
    the surviving ``Entry_Key``; the map of superseded keys lets the caller record
    where each one went and repoint variant links.
    """
    survivors: dict[tuple[str, str], dict[str, str]] = {}
    order: list[tuple[str, str]] = []
    aliases: dict[str, str] = {}

    def _union(current: str, extra: str, separator: str) -> str:
        parts = [p for p in (*current.split(separator), *extra.split(separator)) if p.strip()]
        return separator.join(dict.fromkeys(part.strip() for part in parts))

    for row in forms:
        key = (row["Form"], row["Gloss"])
        aliases[row["Entry_Key"]] = survivors[key]["Entry_Key"] if key in survivors else row["Entry_Key"]
        if key not in survivors:
            survivors[key] = dict(row)
            order.append(key)
            continue
        target = survivors[key]
        target["Source"] = _union(target["Source"], row["Source"], ";")
        target["Tags"] = _union(target["Tags"], row["Tags"], " ")
        target["Notes"] = _union(target["Notes"], row["Notes"], "; ")
        target["Phonemic"] = _union(target["Phonemic"], row["Phonemic"], "; ")
        target["Etymology"] = target["Etymology"] or row["Etymology"]
    merged = [survivors[key] for key in order]
    for row in merged:
        if row["Variant_Of_Key"]:
            row["Variant_Of_Key"] = aliases.get(row["Variant_Of_Key"], row["Variant_Of_Key"])
        # A variant that collapsed onto its own base is no longer a variant.
        if row["Variant_Of_Key"] == row["Entry_Key"]:
            row["Variant_Of_Key"] = ""
    return merged, aliases


def build() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    forms: list[dict[str, str]] = []
    audit: list[dict[str, str]] = []
    record_keys: dict[str, list[str]] = {}
    for record in records():
        unit, region = record["unit"], record["region"]
        raw_form, raw_gloss = record["form"], record["gloss"]
        context = record["context"]
        if region == "interlinear":
            example = re.match(r"ex(\d+[a-f]?)", unit).group(1)
            reference = EXTERNAL_REFERENCES.get(f"ex{example}")
            if reference and "_" not in context:
                context = f"{context}; {reference}"
                record = {**record, "context": context}

        status, reason = "installed", ""
        notes, etymology = "", ""
        form, gloss = raw_form, raw_gloss
        if unit in FORM_FIXES:
            form, gloss, reason = FORM_FIXES[unit]
            status = "installed_after_repair"
        if unit in EXCLUSIONS:
            status, reason = "skipped", EXCLUSIONS[unit]
        elif "…" in form or not form:
            status, reason = "skipped", "bracketed ellipsis marking omitted material"

        parsed_gloss, tags, uncertain = parse_gloss(gloss)
        if region == "table9":
            parsed_gloss, tags = numeral_gloss(raw_gloss), ["num"]
        elif region == "table10":
            parsed_gloss = raw_gloss.removeprefix("to ")
            tags = ["verb", "stem", VERB_CLASSES[unit.split(":")[1]]]
            tags += [t for t in ("m", "f") if f"({t})" in context]
        elif region in {"t6", "t7", "t8"}:
            case, cell = unit.split(":")[1], unit.split(":")[2]
            tags = _paradigm_tags(region, case, cell)
            parsed_gloss = _paradigm_gloss(region, case, cell)
        tags = list(dict.fromkeys([*tags, *REGION_TAGS.get(region, "").split(),
                                   *UNIT_TAGS.get(unit, "").split()]))
        if uncertain:
            tags.append("uncertain")
            reason = reason or "the author marks the gloss itself as uncertain"
        if unit in NOUN_CLASSES:
            tags.append(NOUN_CLASSES[unit])
        if unit in LOANWORDS:
            tags.append("loanword")
            etymology = f"Knobloch identifies this as a loan from {LOANWORDS[unit]}."

        citations = [f"{SOURCE_ID}[{locator(record)}]"]
        page = BUDDRUSS_PAGE.search(context)
        if page:
            citations.append(f"{BUDDRUSS_ID}[p. {page.group(1)}]")
        if region == "t7" and form.startswith("("):
            notes = "not confirmed in the field data; supplied from Buddruss (1967: 41-43)"
            citations.append(f"{BUDDRUSS_ID}[pp. 41-43]")
        if region == "t8":
            notes = "dative and postpositional paradigm reproduced from Buddruss (1967: 41-43)"
            citations.append(f"{BUDDRUSS_ID}[pp. 41-43]")
        if region == "prose" and "Buddruss (1967:" in context:
            citations.append(f"{BUDDRUSS_ID}[p. {context.split('Buddruss (1967:')[1].split(')')[0].strip()}]")
        form = form.strip("()")

        variants = ALTERNATES.get(form, (form,))
        if len(variants) > 1:
            notes = f"printed as {form}"
        emitted = []
        for index, variant in enumerate(variants, start=1):
            key = f"{SOURCE_ID}:{unit}" + (f":v{index}" if index > 1 else "")
            emitted.append((key, variant))
        phonemic = raw_form if region in IPA_REGIONS else ""

        if status.startswith("installed"):
            for index, (key, variant) in enumerate(emitted):
                forms.append(dict(zip(FORM_FIELDS, [
                    LANGUAGE_ID, "", variant.replace(".", ""), parsed_gloss, "",
                    phonemic, notes, ";".join(citations), "", etymology, key,
                    emitted[0][0] if index else "", "", "",
                    " ".join(dict.fromkeys([*tags, DIALECT_TAG])),
                ])))
                record_keys.setdefault(unit, []).append(key)
        payload = "|".join(record.values()).encode()
        audit.append({
            "Snapshot_Date": SNAPSHOT_DATE, "Unit_ID": unit, "Region": region,
            "PDF_Page": record["pdf_page"],
            "Printed_Page": str(int(record["pdf_page"]) - PRINTED_PAGE_OFFSET),
            "Locator": locator(record), "Raw_Form": raw_form, "Raw_Gloss": raw_gloss,
            "Raw_Context": context, "Status": status,
            "Reason": reason or "extracted from the PDF text layer and installed unchanged",
            "Final_Form": " / ".join(v for _, v in emitted) if status.startswith("installed") else "",
            "Final_Gloss": parsed_gloss if status.startswith("installed") else "",
            "Final_Phonemic": phonemic if status.startswith("installed") else "",
            "Final_Tags": " ".join(tags) if status.startswith("installed") else "",
            "Emitted_Key": ";".join(k for k, _ in emitted) if status.startswith("installed") else "",
            "Source": ";".join(citations),
            "Record_SHA256": hashlib.sha256(payload).hexdigest(),
        })

    merged, aliases = collapse(forms)
    for row in audit:
        keys = record_keys.get(row["Unit_ID"], [])
        row["Merged_Into"] = ";".join(
            dict.fromkeys(aliases[key] for key in keys if aliases[key] != key)
        )
    return merged, audit


_PARADIGM_CASE = {
    "Nominative": "nom", "Accusative": "acc", "Ergative/Genitive": "erg gen",
    "Postpositional": "postp", "nom": "nom", "acc": "acc", "erg": "erg",
    "gen": "gen", "dat": "dat", "postp": "postp",
}


def _paradigm_tags(region: str, case: str, cell: str) -> list[str]:
    tags = _PARADIGM_CASE[case].split()
    if region == "t6":
        return [*tags, cell, "pron", "personal"]
    prox, kind, number = cell.split(".")
    tags += ["prox" if prox == "prox" else "dist", "demonstrative", number]
    if kind == "pron":
        # Only the pronominal column carries person; the determiner column does not.
        return [*tags, "pron", f"3{number}"]
    return [*tags, "determiner"]


def _paradigm_gloss(region: str, case: str, cell: str) -> str:
    if region == "t6":
        return DEICTIC_GLOSS[(cell,)]
    prox, kind, number = cell.split(".")
    deixis = "proximate" if prox == "prox" else "remote"
    if kind == "det":
        return f"{'this' if prox == 'prox' else 'that'}" + ("" if number == "sg" else " (plural)")
    return f"he, she, it ({deixis})" if number == "sg" else f"they ({deixis})"


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]], header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if header:
            writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha512(path: Path) -> str:
    return hashlib.sha512(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, help="verify the DiVA PDF and re-extract from it")
    parser.add_argument("--refresh", action="store_true", help="rewrite the checked-in extraction")
    parser.add_argument("--install", action="store_true", help="write the installed CSV and audit")
    args = parser.parse_args()

    if args.pdf:
        if sha256(args.pdf) != PDF_SHA256 or sha512(args.pdf) != PDF_SHA512:
            raise ValueError("PDF does not match the checksums published in the DiVA record")
        fresh = serialise(extract(args.pdf))
        if args.refresh:
            RAW_SNAPSHOT.write_text(fresh, encoding="utf-8")
        elif fresh != RAW_SNAPSHOT.read_text(encoding="utf-8"):
            raise ValueError("a fresh extraction no longer reproduces the checked-in snapshot")
        import pdfplumber

        with pdfplumber.open(args.pdf) as pdf:
            for record in prose_records():
                text = pdf.pages[int(record["pdf_page"]) - 1].extract_text(x_tolerance=1.2)
                if record["form"] not in unicodedata.normalize("NFC", text or ""):
                    raise ValueError(f"{record['unit']}: {record['form']!r} is not on that page")
        print(f"verified {len(snapshot())} extracted and {len(prose_records())} prose records")

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
    MANIFEST_OUTPUT.write_text(json.dumps(manifest(forms, audit), ensure_ascii=False,
                                          indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"installed {len(forms)} Sauji rows from Knobloch (2020)")


def manifest(forms, audit) -> dict:
    return {
        "source_id": SOURCE_ID,
        "snapshot_date": SNAPSHOT_DATE,
        "bibliography": (
            "Knobloch, Nina. 2020. A grammar sketch of Sauji: An Indo-Aryan language of "
            "Afghanistan. MA thesis, Department of Linguistics, Stockholm University."
        ),
        "acquisition": (
            "Open-access full text downloaded from the DiVA record diva2:1440556 "
            "(urn:nbn:se:su:diva-182519) on 2026-08-25"
        ),
        "pdf_sha256": PDF_SHA256,
        "pdf_sha512": PDF_SHA512,
        "pdf_sha512_matches_diva_record": True,
        "pdf_pages": PDF_PAGES,
        "pdf_redistributed": False,
        "rights": (
            "Open access student thesis published in DiVA; no explicit reuse licence is "
            "stated, so only the extracted lexical facts are installed and the PDF is not "
            "checked in."
        ),
        "extraction": {
            "method": (
                "deterministic pdfplumber word-box extraction from the thesis's own LaTeX "
                "text layer; no OCR"
            ),
            "structure_keys": [
                "object-language material is italic and glosses are roman",
                "interlinear glosses share their word's x0 within 2.5 pt",
                "tables are read from fixed column x-ranges",
            ],
            "checked_in_layer": str(RAW_SNAPSHOT.relative_to(ROOT)),
            "prose_layer": "the PROSE table in data/other/forms/raw_data/knobloch_sauji_2020.py",
            "regions": dict(Counter(row["Region"] for row in audit)),
            "transcription_uncertainties_remaining": 0,
        },
        "scope": {
            "included": (
                "every glossed Sauji form the thesis prints: Tables 2, 4, 6, 7, 8, 9, 10, 11 "
                "and the Sauji column of Table 14, the syllable-structure examples in (1), all "
                "interlinear examples including the three appendix texts, and the italicised "
                "citations in running prose"
            ),
            "excluded": (
                "the Palula and Kalkoti comparanda in Table 14 and in the Palula comparisons "
                "(secondary citations of Liljegren 2013, 2016); the phoneme-inventory charts in "
                "Tables 1 and 3 and the case-suffix chart in Table 5; bare inflectional and "
                "derivational affixes cited in the morphological description; the Appendix B "
                "index of recordings; free translations; and the bibliography"
            ),
            "etymology_policy": (
                "the thesis makes no etymological claim about individual Sauji words, so every "
                "row is installed unlinked with a blank Parameter_ID; the donor languages named "
                "in section 5.2.3 are recorded as prose plus a loanword tag, not as graph edges"
            ),
            "language_model": (
                "all forms belong to canonical Sauji (Sv) and carry the registered Sau dialect "
                "tag; consultants, recording sites and elicitation sessions remain provenance in "
                "the locator and audit"
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


if __name__ == "__main__":
    main()
