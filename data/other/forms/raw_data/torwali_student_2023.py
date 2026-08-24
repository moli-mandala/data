#!/usr/bin/env python3
"""Extract Zubair Torwali's CC BY 4.0 student dictionary from its born-digital PDF.

The PDF is a two-column Microsoft Word export.  Its typography is unusually useful: native
headwords are blue 16-point Calibri Bold, pronunciations are red 12-point Doulos SIL, parts of
speech are 12-point Candara Italic, and English definitions are black 14-point Times New Roman.
This importer uses those source-authored distinctions and page coordinates rather than OCR.

The licensed PDF is available from Wikimedia Commons but is not checked into this repository.
Run from ``data/`` (or any directory) with::

    uv run --with pdfplumber python data/other/forms/raw_data/torwali_student_2023.py \
      --pdf ../tmp/pdfs/torwali/Torwali_Dictionary_PDF.pdf --install

Without ``--install`` the importer parses and reports its reconciliation counts without writing
canonical artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber


SOURCE_ID = "torwali2023student"
SNAPSHOT_DATE = "2026-08-20"
PDF_SHA256 = "da338088c2674f0eccdb426fbcac210ebd9d258b2cfa7f2870171b228f42e33f"
PDF_PAGES = 232
ENTRY_PDF_PAGES = range(14, 231)
EXPECTED_HEADWORD_RECORDS = 2269

ROOT = Path(__file__).resolve().parents[4]
RAW_DIR = ROOT / "data/other/forms/raw_data"
FORM_OUTPUT = ROOT / "data/other/forms/20260820-torwali-student-2023.csv"
AUDIT_OUTPUT = RAW_DIR / "20260820-torwali-student-2023-audit.csv"
SAMPLE_OUTPUT = RAW_DIR / "20260820-torwali-student-2023-sample.csv"
MANIFEST_OUTPUT = RAW_DIR / "20260820-torwali-student-2023-manifest.json"

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
    "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
    "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "Snapshot_Date", "Entry_Key", "PDF_Page", "Printed_Page", "Column", "Column_Entry",
    "Headword", "Homograph_Number", "IPA", "Raw_POS", "Gloss", "Native_Example",
    "English_Example", "Scientific_Name", "Relation_Label", "Relation_Target",
    "Relation_Target_Key", "Dialect", "Status", "Reason", "Tags", "Source", "Raw_Record",
    "Review", "Material_Error", "Record_SHA256",
]

BLUE = (0.0, 0.0, 1.0)
RED = (1.0, 0.0, 0.0)
BLACKS = {(0.0,), (0,), 0, None}
LEFT_X = (65.0, 230.0)
RIGHT_X = (252.0, 416.0)
HEADWORD_TOP = 90.0
CONTENT_BOTTOM = 665.0
BAHRAIN_TAG = (
    "dialect:Tor:torwali2023student-BAH:"
    "Bahrain%20%28Torwali%202023%29"
)
CHAIL_TAG = (
    "dialect:Tor:torwali2023student-CHL:"
    "Chail%20%28Torwali%202023%29"
)
REVIEWED_SAMPLE_KEYS = {
    "torwali2023student:p041:cR:e02", "torwali2023student:p042:cL:e01",
    "torwali2023student:p088:cL:e06", "torwali2023student:p111:cR:e03",
    "torwali2023student:p141:cR:e04", "torwali2023student:p142:cL:e03",
    "torwali2023student:p163:cR:e04", "torwali2023student:p164:cL:e02",
    "torwali2023student:p170:cL:e03", "torwali2023student:p172:cR:e05",
    "torwali2023student:p174:cL:e03", "torwali2023student:p181:cR:e03",
    "torwali2023student:p181:cR:e05", "torwali2023student:p194:cL:e01",
    "torwali2023student:p201:cL:e02", "torwali2023student:p209:cR:e05",
    "torwali2023student:p215:cR:e03", "torwali2023student:p217:cL:e02",
    "torwali2023student:p219:cL:e04", "torwali2023student:p229:cL:e02",
}


def nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def compact(value: str) -> str:
    return re.sub(r"\s+", " ", nfc(value)).strip()


def color_is(char: dict, expected) -> bool:
    value = char.get("non_stroking_color")
    if isinstance(value, list):
        value = tuple(value)
    return value == expected


def is_mark(text: str) -> bool:
    return bool(text) and all(unicodedata.category(ch).startswith("M") for ch in text)


def is_headword_char(char: dict) -> bool:
    return (
        "Calibri-Bold" in char["fontname"]
        and 15.5 <= char["size"] <= 16.5
        and color_is(char, BLUE)
        and char["top"] > HEADWORD_TOP
    )


def _rtl_text(chars: list[dict]) -> str:
    """Rebuild visually RTL source text while attaching combining marks to their base."""
    bases = [c for c in chars if c["text"].strip() and not is_mark(c["text"])]
    marks = [c for c in chars if is_mark(c["text"])]
    spaces = [
        c for c in chars
        if not c["text"].strip() and (c["x1"] - c["x0"]) > 1.0
    ]
    assignments: dict[int, list[dict]] = defaultdict(list)
    for mark in marks:
        center = (mark["x0"] + mark["x1"]) / 2
        candidates = [
            (abs(center - (base["x0"] + base["x1"]) / 2), index)
            for index, base in enumerate(bases)
            if base["x0"] - 1.0 <= center <= base["x1"] + 1.0
        ]
        if candidates:
            assignments[min(candidates)[1]].append(mark)
    clusters = []
    for index, base in enumerate(bases):
        attached = assignments[index]
        attached.sort(key=lambda c: (c["top"], -c["x0"]))
        clusters.append((base["x0"], base["text"] + "".join(c["text"] for c in attached)))
    clusters.extend((space["x0"], " ") for space in spaces)
    value = "".join(text for _, text in sorted(clusters, reverse=True))
    return compact(value)


def _line_groups(chars: list[dict], tolerance: float = 3.5) -> list[list[dict]]:
    groups: list[dict] = []
    for char in sorted(chars, key=lambda c: (c["top"], c["x0"])):
        group = next(
            (group for group in reversed(groups) if abs(char["top"] - group["top"]) <= tolerance),
            None,
        )
        if group is None:
            group = {"top": char["top"], "chars": []}
            groups.append(group)
        group["chars"].append(char)
    return [group["chars"] for group in groups]


def _ltr_line(chars: list[dict]) -> str:
    return compact("".join(char["text"] for char in sorted(chars, key=lambda c: c["x0"])))


def _ltr_lines(chars: list[dict]) -> list[str]:
    return [line for line in (_ltr_line(group) for group in _line_groups(chars)) if line]


@dataclass
class Anchor:
    pdf_page: int
    column: str
    top: float
    chars: list[dict]
    column_entry: int = 0

    @property
    def headword(self) -> str:
        return _rtl_text(self.chars)

    @property
    def printed_page(self) -> int:
        return self.pdf_page - 1

    @property
    def key(self) -> str:
        return (
            f"{SOURCE_ID}:p{self.printed_page:03d}:"
            f"c{self.column}:e{self.column_entry:02d}"
        )


@dataclass
class Record:
    anchor: Anchor
    ipa: str = ""
    raw_pos: str = ""
    gloss: str = ""
    native_example: str = ""
    english_example: str = ""
    scientific_name: str = ""
    relation_label: str = ""
    relation_target: str = ""
    relation_target_key: str = ""
    homograph_number: str = ""
    status: str = ""
    reason: str = ""
    tags: list[str] = field(default_factory=list)
    raw_record: str = ""
    chail_marker: bool = False

    @property
    def citation(self) -> str:
        col = 1 if self.anchor.column == "R" else 2
        return (
            f"{SOURCE_ID}[p. {self.anchor.printed_page}, col. {col}, "
            f"entry {self.anchor.column_entry}]"
        )

    def form_row(self) -> list[str]:
        variant_key = ""
        derivation_keys = ""
        relation = relation_kind(self.relation_label)
        if self.relation_target_key:
            if relation in {"free-variant", "spelling-variant", "dialect-variant", "variant"}:
                variant_key = self.relation_target_key
            elif relation in {"plural-of", "past-of", "inflection-of"}:
                derivation_keys = self.relation_target_key
        return [
            # The PDF renders Torwali correctly but its embedded ToUnicode map
            # corrupts some headwords. Keep that candidate in the audit only;
            # canonical Native must not silently publish unreliable spellings.
            "Tor", "", self.ipa, self.gloss, "", self.ipa, "",
            self.citation, "", "", self.anchor.key, variant_key, "", derivation_keys,
            " ".join(self.tags),
        ]


def headword_anchors(page, pdf_page: int) -> list[Anchor]:
    anchors: list[Anchor] = []
    chars = [char for char in page.chars if is_headword_char(char) and char["text"].strip()]
    for column, (x0, x1) in (("R", RIGHT_X), ("L", LEFT_X)):
        column_chars = [char for char in chars if x0 <= char["x0"] <= x1]
        bases = [char for char in column_chars if not is_mark(char["text"])]
        groups: list[dict] = []
        for char in sorted(bases, key=lambda c: c["top"]):
            group = next(
                (group for group in reversed(groups) if abs(char["top"] - group["top"]) <= 1.2),
                None,
            )
            if group is None:
                group = {"top": char["top"], "chars": []}
                groups.append(group)
            group["chars"].append(char)
        marks = [char for char in column_chars if is_mark(char["text"])]
        meaningful_spaces = [
            char for char in page.chars
            if x0 <= char["x0"] <= x1
            and "Calibri-Bold" in char["fontname"]
            and 15.5 <= char["size"] <= 16.5
            and color_is(char, BLUE)
            and not char["text"].strip()
            and (char["x1"] - char["x0"]) > 1.0
        ]
        for group in groups:
            group["chars"].extend(
                mark for mark in marks if abs(mark["top"] - group["top"]) < 10.0
                and any(
                    base["x0"] - 1 <= (mark["x0"] + mark["x1"]) / 2 <= base["x1"] + 1
                    for base in group["chars"]
                )
            )
            group["chars"].extend(
                space for space in meaningful_spaces if abs(space["top"] - group["top"]) < 6.0
            )
        anchors.extend(
            Anchor(pdf_page=pdf_page, column=column, top=group["top"], chars=group["chars"])
            for group in groups
        )
    anchors.sort(key=lambda anchor: (0 if anchor.column == "R" else 1, anchor.top))
    counters = Counter()
    for anchor in anchors:
        counters[anchor.column] += 1
        anchor.column_entry = counters[anchor.column]
    return anchors


def _is_black(char: dict) -> bool:
    value = char.get("non_stroking_color")
    if isinstance(value, list):
        value = tuple(value)
    return value in BLACKS


def _segment_chars(page, anchor: Anchor, next_top: float | None) -> list[dict]:
    x0, x1 = RIGHT_X if anchor.column == "R" else LEFT_X
    bottom = min(next_top - 1.0, CONTENT_BOTTOM) if next_top is not None else CONTENT_BOTTOM
    return [
        char for char in page.chars
        if x0 <= char["x0"] <= x1 and anchor.top - 2.0 <= char["top"] < bottom
    ]


def _first_bracketed_ipa(chars: list[dict]) -> str:
    red_chars = [char for char in chars if color_is(char, RED) and char["top"] < chars[0]["top"] + 18]
    for line in _ltr_lines(red_chars):
        match = re.search(r"\[([^\]]+)\]", line)
        if match:
            return compact(match.group(1))
    # Mixed fonts and bidirectional layout occasionally split a bracket onto a nearby baseline.
    value = "".join(char["text"] for char in sorted(red_chars, key=lambda c: c["x0"]))
    match = re.search(r"\[([^\]]+)\]", value)
    return compact(match.group(1)) if match else ""


def _extract_pos(chars: list[dict], anchor_top: float) -> str:
    # Most POS labels share the headword line, but entries whose definition
    # wraps put the same source-styled label on the following baseline.  The
    # segment is already bounded by the next headword, so retaining every
    # black Candara Italic glyph in it is both safer and more complete than a
    # fixed offset from the anchor.
    selected = [
        char for char in chars
        if "Candara-Italic" in char["fontname"] and _is_black(char)
    ]
    near_anchor = [
        char for char in selected
        if anchor_top - 1.0 <= char["top"] <= anchor_top + 10.0
    ]
    if near_anchor:
        return compact(" ".join(_ltr_lines(near_anchor))).strip(" .")

    # Scientific names elsewhere in the entry may use the same italic face.
    # A wrapped POS is therefore the longest recognized prefix, not the whole
    # italic line (e.g. ``n Rubus idaeus`` contributes only source label ``n``).
    value = compact(" ".join(_ltr_lines(selected))).strip(" .")
    words = value.split()
    for end in range(len(words), 0, -1):
        candidate = " ".join(words[:end]).strip(" .")
        if pos_tags(candidate):
            return candidate
    return ""


def _extract_gloss(chars: list[dict]) -> str:
    selected = [
        char for char in chars
        if char["fontname"] == "TimesNewRomanPSMT"
        and 13.5 <= char["size"] <= 14.5 and _is_black(char)
    ]
    lines = _ltr_lines(selected)
    clean = []
    for line in lines:
        line = re.sub(r"^\(?\d+\)?\s*", "", line).strip(" ;")
        if line and line not in clean:
            clean.append(line)
    return compact(" ".join(clean))


def _extract_native_example(chars: list[dict], anchor_top: float) -> str:
    selected = [
        char for char in chars
        if "Calibri" in char["fontname"] and "Bold" not in char["fontname"]
        and 13.5 <= char["size"] <= 14.5 and color_is(char, BLUE)
        and char["top"] > anchor_top + 7.0
    ]
    return " ".join(_rtl_text(line) for line in _line_groups(selected) if _rtl_text(line))


def _extract_english_example(chars: list[dict], anchor_top: float) -> str:
    selected = [
        char for char in chars
        if char["fontname"] == "TimesNewRomanPSMT"
        and 11.5 <= char["size"] <= 12.5 and _is_black(char)
        and char["top"] > anchor_top + 8.0
    ]
    lines = [line for line in _ltr_lines(selected) if line not in {".", ")", "("}]
    return " ".join(lines)


def _extract_relation(chars: list[dict]) -> tuple[str, str]:
    label_chars = [
        char for char in chars
        if "Candara" in char["fontname"] and "Italic" not in char["fontname"]
        and 11.5 <= char["size"] <= 12.5 and _is_black(char)
    ]
    label_groups = _line_groups(label_chars)
    labeled_lines = [(_ltr_line(group).strip(" ."), group) for group in label_groups]
    relation_lines = [(line, group) for line, group in labeled_lines if "of" in line.casefold()]
    if not relation_lines:
        return "", ""
    label, label_group = relation_lines[0]
    relation_top = sum(char["top"] for char in label_group) / len(label_group)
    # Pronunciations use red brackets; the source's [چ] dialect marker and
    # relation targets use black type. Restrict this test to the latter.
    bracket_chars = [
        char for char in chars
        if char["text"] in {"[", "]"} and _is_black(char)
    ]

    def is_marker_che(char: dict) -> bool:
        return char["text"] == "چ" and {
            bracket["text"] for bracket in bracket_chars
            if abs(bracket["top"] - char["top"]) <= 5.0
            and abs(bracket["x0"] - char["x0"]) <= 24.0
        } == {"[", "]"}

    target_chars = [
        char for char in chars
        if "Calibri-Bold" in char["fontname"] and 15.5 <= char["size"] <= 16.5
        and _is_black(char) and char["text"].strip()
        and abs(char["top"] - relation_top) <= 10.0
        and not is_marker_che(char)
    ]
    target = ""
    if target_chars:
        target = _rtl_text(target_chars)
    return label, target


def _extract_scientific_name(chars: list[dict]) -> str:
    lines = _ltr_lines([
        char for char in chars
        if _is_black(char) and 11.5 <= char["size"] <= 12.5
        and ("TimesNewRomanPS-Italic" in char["fontname"] or "TimesNewRomanPSMT" == char["fontname"])
    ])
    text = " ".join(lines)
    match = re.search(r"Sc\.?\s*Name:?\s*(.+?)(?:\(|$)", text, re.IGNORECASE)
    return compact(match.group(1)) if match else ""


POS_MAP = {
    "ad.f": ["adj", "f"], "ad.m": ["adj", "m"], "adj": ["adj"],
    "adp": ["postp"], "adv": ["adv"], "aux.v": ["auxiliary", "verb"],
    "aux. vf": ["auxiliary", "verb", "f"], "cardnum": ["num"],
    "con": ["conj"], "coon": ["conj"], "coord.conn": ["conj"],
    "interj": ["interj"], "rel.pro": ["pron", "relative"], "n": ["noun"],
    "n f": ["noun", "f"], "n.m": ["noun", "m"], "n m": ["noun", "m"],
    "n pl": ["noun", "pl"], "n.proper": ["noun", "proper-noun"],
    "or. n": ["ord", "num"], "pers": ["pron", "personal"],
    "poss": ["pron", "poss"], "post": ["postp"], "pro": ["pron"],
    "q.f": ["quantifier"], "q. f": ["quantifier"], "sing": ["sg"], "v": ["verb"],
    "vi": ["verb", "intr"], "vt": ["verb", "tr"],
    "nprop": ["noun", "proper-noun"], "n.prop": ["noun", "proper-noun"],
    "ad. f": ["adj", "f"], "ad. m": ["adj", "m"], "adj. m": ["adj", "m"],
    "n. m": ["noun", "m"], "conn": ["conj"], "aux. v": ["auxiliary", "verb"],
    "relpro": ["pron", "relative"], "card.num": ["num"], "car.dnum": ["num"],
    "dem": ["pron", "demonstrative"], "ques. wd": ["pron", "interr"],
    "salutation": ["interj"], "dp.p": ["discourse-marker"],
}


def pos_tags(raw_pos: str) -> list[str]:
    value = compact(raw_pos).casefold().replace("..", ".").strip(" .")
    value = re.sub(r"\s+", " ", value)
    if value in POS_MAP:
        return list(POS_MAP[value])
    # Some entries combine two source labels, e.g. ``adj./adv.``.
    tags = []
    for piece in re.split(r"[/,;]", value):
        for tag in POS_MAP.get(piece.strip(" ."), []):
            if tag not in tags:
                tags.append(tag)
    if tags:
        return tags

    # A handful of source entries print adjacent labels without punctuation
    # (``adj adv``, ``n adj``, ``pro Aux. Vf``). Segment them greedily while
    # preferring the longest labels in the source inventory.
    words = value.split()
    index = 0
    while index < len(words):
        matched = None
        for end in range(len(words), index, -1):
            candidate = " ".join(words[index:end]).strip(" .")
            if candidate in POS_MAP:
                matched = (end, POS_MAP[candidate])
                break
        if matched is None:
            return []
        index, piece_tags = matched
        for tag in piece_tags:
            if tag not in tags:
                tags.append(tag)
    return tags


def relation_kind(label: str) -> str:
    value = re.sub(r"\s+", "", label.casefold())
    if "dia.var.of" in value or "dial.var.of" in value:
        return "dialect-variant"
    if "fr.var.of" in value:
        return "free-variant"
    if "sp.var.of" in value:
        return "spelling-variant"
    if "unspec.var.of" in value:
        return "variant"
    if "var.of" in value:
        return "variant"
    if "pl.of" in value:
        return "plural-of"
    if "pst of" in label.casefold() or "pstof" in value:
        return "past-of"
    if "irre.infl" in value or "irreg.infl" in value:
        return "inflection-of"
    return ""


def relation_tags(kind: str) -> list[str]:
    return {
        "dialect-variant": ["alternate", "dialectal"],
        "free-variant": ["alternate"],
        "spelling-variant": ["alternate"],
        "variant": ["alternate"],
        "plural-of": ["noun", "pl"],
        "past-of": ["verb", "pret"],
        "inflection-of": ["alternate"],
    }.get(kind, [])


def ipa_review_reason(ipa: str) -> str:
    """Flag unusual source IPA without silently repairing it."""
    reasons = []
    if any(char in ipa for char in ';."'):
        reasons.append("editorial punctuation in source IPA")
    if "I" in ipa:
        reasons.append("ASCII capital I in source IPA")
    if re.search(r"[^aeiouæɑəɛàá]:", ipa):
        reasons.append("colon follows a non-vowel in source IPA")
    return "; ".join(reasons)


def record_text(chars: list[dict]) -> str:
    lines = []
    for group in _line_groups(chars):
        blue_arabic = any(color_is(char, BLUE) and "Calibri" in char["fontname"] for char in group)
        value = _rtl_text(group) if blue_arabic else _ltr_line(group)
        if value:
            lines.append(value)
    return " ⏎ ".join(lines)


def has_chail_marker(chars: list[dict]) -> bool:
    brackets = [char for char in chars if char["text"] in {"[", "]"} and _is_black(char)]
    ches = [char for char in chars if char["text"] == "چ" and _is_black(char)]
    for che in ches:
        nearby = [
            bracket for bracket in brackets
            if abs(bracket["top"] - che["top"]) <= 5.0
            and abs(bracket["x0"] - che["x0"]) <= 24.0
        ]
        if {bracket["text"] for bracket in nearby} == {"[", "]"}:
            return True
    return False


def parse_pdf(pdf_path: Path) -> list[Record]:
    if hashlib.sha256(pdf_path.read_bytes()).hexdigest() != PDF_SHA256:
        raise ValueError(f"Unexpected PDF identity: {pdf_path}")
    records: list[Record] = []
    with pdfplumber.open(pdf_path) as pdf:
        if len(pdf.pages) != PDF_PAGES:
            raise ValueError(f"Expected {PDF_PAGES} pages, got {len(pdf.pages)}")
        for pdf_page in ENTRY_PDF_PAGES:
            page = pdf.pages[pdf_page - 1]
            anchors = headword_anchors(page, pdf_page)
            by_column = defaultdict(list)
            for anchor in anchors:
                by_column[anchor.column].append(anchor)
            for column in ("R", "L"):
                column_anchors = sorted(by_column[column], key=lambda anchor: anchor.top)
                for index, anchor in enumerate(column_anchors):
                    next_top = column_anchors[index + 1].top if index + 1 < len(column_anchors) else None
                    chars = _segment_chars(page, anchor, next_top)
                    relation_label, relation_target = _extract_relation(chars)
                    record = Record(
                        anchor=anchor,
                        ipa=_first_bracketed_ipa(chars),
                        raw_pos=_extract_pos(chars, anchor.top),
                        gloss=_extract_gloss(chars),
                        native_example=_extract_native_example(chars, anchor.top),
                        english_example=_extract_english_example(chars, anchor.top),
                        scientific_name=_extract_scientific_name(chars),
                        relation_label=relation_label,
                        relation_target=relation_target,
                        raw_record=record_text(chars),
                        chail_marker=has_chail_marker(chars),
                    )
                    number_chars = [
                        char for char in chars
                        if "Calibri-Bold" in char["fontname"] and color_is(char, BLUE)
                        and 8.5 <= char["size"] <= 9.5 and char["top"] < anchor.top + 10
                    ]
                    record.homograph_number = compact(_ltr_line(number_chars))
                    records.append(record)
    if len(records) != EXPECTED_HEADWORD_RECORDS:
        raise ValueError(
            f"Expected {EXPECTED_HEADWORD_RECORDS} headword records, extracted {len(records)}"
        )
    return records


def reconcile(records: list[Record]) -> None:
    by_native: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        by_native[compact(record.anchor.headword)].append(record)

    for record in records:
        kind = relation_kind(record.relation_label)
        record.tags = pos_tags(record.raw_pos)
        for tag in relation_tags(kind):
            if tag not in record.tags:
                record.tags.append(tag)
        if record.chail_marker:
            record.tags.append(CHAIL_TAG)
        else:
            record.tags.append(BAHRAIN_TAG)
        ipa_issue = ipa_review_reason(record.ipa)
        if ipa_issue and "uncertain" not in record.tags:
            record.tags.append("uncertain")
        if kind and not record.gloss:
            label = kind.replace("-", " ")
            record.gloss = f"{label} {record.relation_target}".strip()

        if record.relation_target:
            candidates = by_native.get(compact(record.relation_target), [])
            if len(candidates) == 1:
                record.relation_target_key = candidates[0].anchor.key
            elif len(candidates) > 1:
                record.reason = f"ambiguous relation target ({len(candidates)} exact headwords)"
            else:
                record.reason = "unresolved relation target"

        if not record.ipa:
            record.status = "excluded_no_ipa"
            if not record.reason:
                record.reason = "source entry supplies no pronunciation"
        else:
            record.status = "installed"
            if not record.gloss and not kind:
                record.reason = "installed with source-blank English gloss"

        unknown_pos = bool(record.raw_pos and not pos_tags(record.raw_pos))
        if unknown_pos:
            suffix = f"unrecognized POS: {record.raw_pos}"
            record.reason = f"{record.reason}; {suffix}".strip("; ")
        if ipa_issue:
            record.reason = f"{record.reason}; {ipa_issue}".strip("; ")


def audit_row(record: Record) -> dict[str, str]:
    reviewed = record.anchor.key in REVIEWED_SAMPLE_KEYS
    payload = {
        "Snapshot_Date": SNAPSHOT_DATE,
        "Entry_Key": record.anchor.key,
        "PDF_Page": str(record.anchor.pdf_page),
        "Printed_Page": str(record.anchor.printed_page),
        "Column": record.anchor.column,
        "Column_Entry": str(record.anchor.column_entry),
        "Headword": record.anchor.headword,
        "Homograph_Number": record.homograph_number,
        "IPA": record.ipa,
        "Raw_POS": record.raw_pos,
        "Gloss": record.gloss,
        "Native_Example": record.native_example,
        "English_Example": record.english_example,
        "Scientific_Name": record.scientific_name,
        "Relation_Label": record.relation_label,
        "Relation_Target": record.relation_target,
        "Relation_Target_Key": record.relation_target_key,
        "Dialect": "Chail" if record.chail_marker else "Sinkaen/Bahrain",
        "Status": record.status,
        "Reason": record.reason,
        "Tags": " ".join(record.tags),
        "Source": record.citation,
        "Raw_Record": record.raw_record,
        "Review": (
            "source-image-verified: IPA, gloss, POS, dialect marker, and inclusion status; "
            "PDF ToUnicode headword candidate excluded from canonical Native"
            if reviewed else "parser-generated; pending source-image review"
        ),
        "Material_Error": "no" if reviewed else "pending",
    }
    digest_input = "\x1f".join(payload[field] for field in AUDIT_FIELDS if field != "Record_SHA256")
    payload["Record_SHA256"] = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
    return payload


def write_csv(path: Path, rows, fieldnames: list[str] | None = None) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        if fieldnames:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        else:
            csv.writer(handle).writerows(rows)


def install(records: list[Record], pdf_path: Path) -> None:
    installed = [record for record in records if record.status == "installed"]
    audits = [audit_row(record) for record in records]
    sample_records = random.Random(20260820).sample(records, 20)
    sample_keys = {record.anchor.key for record in sample_records}
    samples = [row for row in audits if row["Entry_Key"] in sample_keys]
    write_csv(FORM_OUTPUT, [record.form_row() for record in installed])
    write_csv(AUDIT_OUTPUT, audits, AUDIT_FIELDS)
    write_csv(SAMPLE_OUTPUT, samples, AUDIT_FIELDS)
    manifest = {
        "source_id": SOURCE_ID,
        "snapshot_date": SNAPSHOT_DATE,
        "canonical_title": "Torwali English Dictionary for Students with pictures",
        "compiler": "Zubair Torwali",
        "edition_year": 2023,
        "isbn": "978-969-9665-01-1",
        "license": "CC BY 4.0",
        "license_evidence": "Wikimedia Commons file page; author uploaded own work",
        "dialect_evidence": [
            {
                "pdf_page": 8,
                "printed_page": 7,
                "description": "English abbreviations define black [چ] as Chail dialect and name the other dialect Sinkaen or Bahrain.",
            },
            {
                "pdf_page": 11,
                "printed_page": 10,
                "description": "Urdu usage notes explain that [چ] marks the Chail realization in dialect-variant entries.",
            },
        ],
        "webonary": {
            "url": "https://www.webonary.org/torwali/en/",
            "reported_entries": 2271,
            "last_upload": "2023-08-21",
            "role": "live-edition validation only; not bulk-crawled",
        },
        "pdf": {
            "url": "https://commons.wikimedia.org/wiki/File:Torwali_Dictionary_PDF.pdf",
            "local_input": str(pdf_path),
            "sha256": PDF_SHA256,
            "pages": PDF_PAGES,
            "entry_pdf_pages": [ENTRY_PDF_PAGES.start, ENTRY_PDF_PAGES.stop - 1],
            "redistributed": False,
        },
        "extraction": {
            "method": "deterministic PDF glyph font/color/coordinate extraction; no OCR",
            "raw_headword_records": len(records),
            "status_counts": dict(Counter(record.status for record in records)),
            "relation_counts": dict(Counter(relation_kind(record.relation_label) or "none" for record in records)),
            "unrecognized_pos_counts": dict(Counter(
                record.raw_pos for record in records if record.raw_pos and not pos_tags(record.raw_pos)
            )),
            "unresolved_relation_targets": sum(
                bool(record.relation_target and not record.relation_target_key) for record in records
            ),
            "dialect_counts": dict(Counter(
                "Chail" if record.chail_marker else "Sinkaen/Bahrain"
                for record in records
            )),
            "installed_dialect_counts": dict(Counter(
                "Chail" if record.chail_marker else "Sinkaen/Bahrain"
                for record in installed
            )),
            "seeded_sample_size": 20,
            "seed": 20260820,
            "seeded_sample_material_errors": 0,
            "seeded_sample_review_scope": "IPA, gloss, POS, dialect marker, and inclusion status against rendered PDF pages; ToUnicode headword candidates are noncanonical",
        },
        "outputs": {
            "installed_forms": len(installed),
            "forms": str(FORM_OUTPUT.relative_to(ROOT)),
            "audit": str(AUDIT_OUTPUT.relative_to(ROOT)),
            "sample": str(SAMPLE_OUTPUT.relative_to(ROOT)),
        },
        "scope": {
            "included": "IPA, POS, English definition, source-explicit Chail versus default Sinkaen/Bahrain dialect assignment, and explicit source relations on PDF pp. 14-230 (printed pp. 13-229); PDF ToUnicode headword candidates retained in the audit as locators",
            "excluded": "Canonical Native values because the PDF ToUnicode map demonstrably corrupts some rendered Torwali headwords; front matter; illustrations; example sentences and scientific names from installed form fields (preserved in audit); source entries without IPA",
            "etymology_policy": "No etymological links inferred; the dictionary makes lexical and variant/inflection claims only",
            "dialect_policy": "The dictionary states that Torwali has Chail and Sinkaen/Bahrain dialects. Black [چ] marks Chail; unmarked records are assigned to the default Sinkaen/Bahrain variety. Source-specific dialect tags are deliberately distinct from SSNP survey tags.",
        },
    }
    MANIFEST_OUTPUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    records = parse_pdf(args.pdf)
    reconcile(records)
    summary = {
        "records": len(records),
        "status": dict(Counter(record.status for record in records)),
        "with_gloss": sum(bool(record.gloss) for record in records),
        "with_pos": sum(bool(record.raw_pos) for record in records),
        "relations": dict(Counter(relation_kind(record.relation_label) or "none" for record in records)),
        "unresolved_relations": sum(
            bool(record.relation_target and not record.relation_target_key) for record in records
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.install:
        install(records, args.pdf)


if __name__ == "__main__":
    main()
