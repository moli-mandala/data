#!/usr/bin/env python3
"""Extract Hockings & Pilot-Raichoor's 1992 Badaga-English dictionary.

The supplied PDF is a scan with a structurally useful but linguistically lossy
hidden OCR layer: retroflex dots disappear and text from the two columns is
sometimes fused. This importer renders the Badaga-English section (PDF pages
21-643; printed pages 1-621) at 300 dpi and runs a fixed Tesseract Latin pass.
Page OCR is cached as JSON, and every source article is retained in an audit.

The English-Badaga reverse glossary, appendices, references, front matter,
blank leaves, and publisher advertisement are deliberately excluded: they are
indexes or context rather than independent Badaga attestations.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from ocr_corrections import OcrCorrection, load_corrections


SOURCE_ID = "hockings-pilotraichoor1992"
LANGUAGE_ID = "Badaga"
FIRST_PDF_PAGE = 21
LAST_PDF_PAGE = 643
INSERTED_BLANK_PDF_PAGES = {443, 444}
SCALE = 300 / 72
RICH_COLUMNS = 15
PDFIUM_LOCK = threading.Lock()

HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parents[3]
DEFAULT_CACHE = DATA_ROOT / ".cache/ocr/hockings-badaga/pages"
DEFAULT_OUTPUT_DIR = DATA_ROOT / ".cache/ocr/hockings-badaga/output"
DEFAULT_INSTALL = DATA_ROOT / "data/other/forms/20260818-hockings-badaga.csv"
DEFAULT_AUDIT = HERE / "20260818-hockings-badaga-audit.csv"
DEFAULT_CORRECTIONS = HERE / "20260818-hockings-badaga-corrections.csv"

# Geometry in source PDF points (page size 439.44 x 652.08). Entry heads start
# at about 54/229 points; continuation lines start at about 60/234 points.
BODY_TOP = 67.0
BODY_BOTTOM = 600.0
LEFT_HEAD_BAND = (51.0, 56.0)
RIGHT_HEAD_BAND = (225.0, 231.0)

# Unambiguous OCR substitutions for source retroflex glyphs. Plain ASCII
# t/d/n/l remain unchanged: when a dot is lost altogether, the row stays
# explicitly marked for transcription review.
OCR_CHAR_FIXES = str.maketrans({
    "ţ": "ṭ", "ț": "ṭ", "Ţ": "Ṭ", "Ț": "Ṭ",
    "đ": "ḍ", "Đ": "Ḍ", "ð": "ḍ",
    "ņ": "ṇ", "Ņ": "Ṇ", "ñ": "ṇ",
    "ł": "ḷ", "Ł": "Ḷ", "ļ": "ḷ", "Ļ": "Ḷ",
    "ş": "ṣ", "Ş": "Ṣ",
    "ﬁ": "fi", "ﬂ": "fl",
})

# Source abbreviation list plus OCR confusions for the printed italic "n.".
LABEL_PATTERN = (
    r"(?:aux\.?\s*vb\.?|def\.?\s*vb\.?|irr\.?\s*vb\.?|"
    r"adj\.?\s*P\.?|adv\.?\s*P\.?|Neg\.?\s*P\.?|Pr\.?\s*P\.?|"
    r"PP\.?|np\.?|n\.?|[ηπλ]\.?|adj\.?|adv\.?|vi\.?|vt\.?|vf\.?|"
    r"vb\.?|pr[oa]\.?|num\.?|excl\.?|conj\.?|idiom\.?|idem\.?|cf\.?|sfx\.?)"
)
LABEL = re.compile(rf"\s+(?P<label>{LABEL_PATTERN})(?=\s|,|;|$)", re.I)
DEDR = re.compile(
    r"(?P<uncertain>\?\s*)?DEDR\s+(?P<ids>"
    r"(?:\d+[a-z]?(?:\s*,\s*\d+[a-z]?)*|App\.?\s*\d+[a-z]?))",
    re.I,
)
EXAMPLE_START = re.compile(r"^(?:[‘'\"“]|\.{2,}[‘'\"]?|—)")
DONOR = re.compile(
    r"(?:<|from)\s+(?:Eng\.|English|Fr\.|French|Skt\.|Sanskrit|"
    r"Ka\.|Kannada|Ta\.|Tamil|Te\.|Telugu|Ma\.|Malayalam|"
    r"Ar\.|Arabic|Pers\.|Persian|Port\.|Portuguese)\b",
    re.I,
)

POS_TAGS = {
    "n.": ("noun",),
    "np.": ("noun", "proper-noun"),
    "adj.": ("adj",),
    "adj.p.": ("adj", "participle"),
    "adv.": ("adv",),
    "adv.p.": ("adv", "participle"),
    "neg.p.": ("neg", "participle"),
    "pr.p.": ("pres", "participle"),
    "pp.": ("pp", "participle"),
    "vb.": ("verb",),
    "def. vb.": ("verb",),
    "irr. vb.": ("verb",),
    "aux. vb.": ("verb", "auxiliary"),
    "vi.": ("verb", "intr"),
    "vt.": ("verb", "tr"),
    "vf.": ("verb", "tr"),
    "pro.": ("pron",),
    "num.": ("num",),
    "excl.": ("interj",),
    "conj.": ("conj",),
    "idiom.": ("multiword-expression",),
    "sfx.": ("suffix",),
}


@dataclass
class OCRLine:
    block: int
    paragraph: int
    line: int
    text: str
    left: int
    top: int
    right: int
    bottom: int
    confidence: float

    @property
    def left_pt(self) -> float:
        return self.left / SCALE

    @property
    def top_pt(self) -> float:
        return self.top / SCALE

    @property
    def bottom_pt(self) -> float:
        return self.bottom / SCALE


@dataclass
class Entry:
    pdf_page: int
    printed_page: int
    column: int
    top: int
    lines: list[OCRLine] = field(default_factory=list)
    head: str = ""
    label: str = ""

    @property
    def key(self) -> str:
        return f"{SOURCE_ID}:p{self.printed_page}:c{self.column}:y{self.top:04d}"

    @property
    def raw_entry(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def confidence(self) -> float:
        values = [line.confidence for line in self.lines if line.confidence >= 0]
        return sum(values) / len(values) if values else 0.0


def canonical(text: str) -> str:
    text = unicodedata.normalize("NFC", text.translate(OCR_CHAR_FIXES))
    text = text.replace("–", "-").replace("‐", "-")
    text = re.sub(
        r"\b(n|np|adj|adv|vi|vt|vf|vb|pro|num|excl|conj|idiom|idem|cf|sfx)\."
        r"(?=[A-Za-z0-9(])",
        r"\1. ",
        text,
        flags=re.I,
    )
    return re.sub(r"\s+", " ", text).strip()


def normalize_label(value: str) -> str:
    label = canonical(value).lower().rstrip(".")
    if label in {"η", "π", "λ", "n"}:
        return "n."
    label = re.sub(r"\s+", " ", label)
    aliases = {
        "adj p": "adj.p", "adj. p": "adj.p",
        "adv p": "adv.p", "adv. p": "adv.p",
        "neg p": "neg.p", "neg. p": "neg.p",
        "pr p": "pr.p", "pr. p": "pr.p",
        "aux vb": "aux. vb", "aux.vb": "aux. vb", "aux. vb": "aux. vb",
        "def vb": "def. vb", "def.vb": "def. vb", "def. vb": "def. vb",
        "irr vb": "irr. vb", "irr.vb": "irr. vb", "irr. vb": "irr. vb",
        "pra": "pro.", "pro": "pro.",
    }
    label = aliases.get(label, label)
    return label if label.endswith(".") else label + "."


def join_lines(lines: Iterable[str]) -> str:
    result = ""
    for line in lines:
        line = canonical(line)
        if not line:
            continue
        if result.endswith("-") and line[:1].islower():
            result = result[:-1] + line
        else:
            result += (" " if result else "") + line
    result = re.sub(r"\s+([,.;:!?])", r"\1", result)
    result = re.sub(r"([([{])\s+", r"\1", result)
    return canonical(result)


def mechanical_head_repairs(value: str) -> str:
    value = canonical(value)
    # In this scan, spaces inside a geminate are produced by the printed
    # subscript dots. Ordinary dental geminates OCR without an internal space.
    value = re.sub(r"(?i)(?<=\w)t\s+t(?=\w)", "ṭṭ", value)
    value = re.sub(r"(?i)(?<=\w)d\s+d(?=\w)", "ḍḍ", value)
    value = re.sub(r"(?i)(?<=\w)n\s+n(?=\w)", "ṇṇ", value)
    value = re.sub(r"(?i)(?<=\w)l\s+l(?=\w)", "ḷḷ", value)
    value = re.sub(r"\s*/\s*", "/", value)
    value = re.sub(r"\s*:\s*", ":", value)
    value = re.sub(r"\s*-\s*", "-", value)
    value = re.sub(
        r"\s+(?:ACC|GEN|DAT|NEG|NOM|LOC|HORT|HYP|IMPER|OBLIG|OPT|S1|S2)$",
        "", value, flags=re.I,
    )
    return canonical(value)


def line_groups(tsv: str) -> list[OCRLine]:
    rows = csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE)
    groups: dict[tuple[int, int, int], list[dict[str, str]]] = {}
    order: list[tuple[int, int, int]] = []
    for row in rows:
        if row.get("level") != "5" or not row.get("text", "").strip():
            continue
        key = (int(row["block_num"]), int(row["par_num"]), int(row["line_num"]))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(row)

    result = []
    for key in order:
        words = sorted(groups[key], key=lambda row: int(row["word_num"]))
        confidences = [float(row["conf"]) for row in words if float(row["conf"]) >= 0]
        result.append(OCRLine(
            block=key[0],
            paragraph=key[1],
            line=key[2],
            text=canonical(" ".join(row["text"] for row in words)),
            left=min(int(row["left"]) for row in words),
            top=min(int(row["top"]) for row in words),
            right=max(int(row["left"]) + int(row["width"]) for row in words),
            bottom=max(int(row["top"]) + int(row["height"]) for row in words),
            confidence=sum(confidences) / len(confidences) if confidences else 0.0,
        ))
    return result


def ocr_page(job: tuple[str, int, str]) -> tuple[int, dict]:
    pdf_path, pdf_page, cache_dir = job
    cache_path = Path(cache_dir) / f"page-{pdf_page:03d}.json"
    if cache_path.exists():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        data["printed_page"] = printed_page_for(pdf_page)
        return pdf_page, data
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise RuntimeError("pypdfium2 is required for Badaga OCR") from exc

    with PDFIUM_LOCK:
        document = pdfium.PdfDocument(pdf_path)
        page = document[pdf_page - 1]
        image = page.render(scale=SCALE).to_pil()
        page.close()
        document.close()
    payload = io.BytesIO()
    image.save(payload, format="PNG")
    proc = subprocess.run(
        ["tesseract", "stdin", "stdout", "-l", "script/Latin", "--psm", "3", "tsv"],
        input=payload.getvalue(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    data = {
        "pdf_page": pdf_page,
        "printed_page": printed_page_for(pdf_page),
        "width": image.width,
        "height": image.height,
        "dpi": 300,
        "engine": "tesseract script/Latin --psm 3 tsv",
        "lines": [
            asdict(line)
            for line in line_groups(proc.stdout.decode("utf-8", errors="replace"))
        ],
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    temporary.replace(cache_path)
    return pdf_page, data


def load_page_lines(data: dict) -> list[OCRLine]:
    return [OCRLine(**{**row, "text": canonical(row["text"])}) for row in data["lines"]]


def column(line: OCRLine, width: int) -> int:
    return 1 if line.left < width / 2 else 2


def is_margin_head(line: OCRLine, width: int) -> bool:
    band = LEFT_HEAD_BAND if column(line, width) == 1 else RIGHT_HEAD_BAND
    return band[0] <= line.left_pt <= band[1]


def plausible_start(text: str) -> bool:
    value = canonical(text)
    if (
        not value or value.isdigit() or len(value) == 1
        or value.startswith((".", ",", ";", ")")) or EXAMPLE_START.match(value)
    ):
        return False
    if value.upper() in {
        "BADAGA-ENGLISH", "DICTIONARY", "INCORPORATING A GAZETTEER",
        "OF BADAGA PLACENAMES",
    }:
        return False
    if re.match(
        r"^(?:and |or |the |to |from |with |which |who |where |when |that |"
        r"join |belonging |formerly |now |name of |houses? |km\.? |DEDR\b)",
        value, re.I,
    ):
        return False
    return True


def head_and_label(lines: list[OCRLine]) -> tuple[str, str]:
    preview = join_lines(line.text for line in lines[:4])
    # Some labels include descriptive words before a generic abbreviation.
    # Compare their position with the first ordinary label: a later ``sfx``
    # inside an article must not swallow an earlier noun/verb label.
    candidates: list[tuple[int, int, str]] = []
    for pattern, label in (
        (r"\s+(?:deictic base|remote deictic|INTERR base)(?=\s|$)", "base."),
        (
            r"\s+(?:(?:ACC|GEN|DAT|NEG|NOM|LOC|HORT|HYP|IMPER|OBLIG|"
            r"OPT|S1|S2|infinitive)\s+)?sfx(?=\s|$)",
            "sfx.",
        ),
        (r"\s+the final of adjectival participles(?=\s|:|$)", "adj.p."),
    ):
        special = re.search(pattern, preview[:320], re.I)
        if special:
            candidates.append((special.start(), special.end(), label))
    match = LABEL.search(preview[:320])
    if match:
        candidates.append((match.start(), match.end(), normalize_label(match.group("label"))))
    if candidates:
        start, _end, label = min(candidates, key=lambda item: item[0])
        head = mechanical_head_repairs(preview[:start])
        # Dictionary heads use slash-separated alternates, never comma- or
        # semicolon-separated English prose. Those marks indicate that a
        # continuation line has drifted into the head indentation band.
        if head and len(head) <= 60 and "," not in head and ";" not in head:
            return head, label
    return "", ""


def extract_entries(page_data: Iterable[dict]) -> tuple[list[Entry], list[dict[str, str]]]:
    entries: list[Entry] = []
    layout: list[dict[str, str]] = []
    current: Entry | None = None
    for data in sorted(page_data, key=lambda row: row["pdf_page"]):
        width = int(data["width"])
        lines = [
            line for line in load_page_lines(data)
            if BODY_TOP <= line.top_pt and line.bottom_pt <= BODY_BOTTOM
        ]
        lines.sort(key=lambda line: (column(line, width), line.top, line.left))
        for line in lines:
            starts = is_margin_head(line, width) and plausible_start(line.text)
            if starts:
                if current is not None:
                    head, label = head_and_label(current.lines)
                    if head:
                        current.head, current.label = head, label
                        entries.append(current)
                    elif entries:
                        entries[-1].lines.extend(current.lines)
                    else:
                        layout.append({
                            "Status": "layout-excluded",
                            "Reason": "pre-lexicon or headless margin text",
                            "PDF_Page": str(current.pdf_page),
                            "Printed_Page": str(current.printed_page),
                            "Column": str(current.column),
                            "Top": str(current.top),
                            "Raw_OCR": current.raw_entry,
                        })
                current = Entry(
                    pdf_page=int(data["pdf_page"]),
                    printed_page=int(data["printed_page"]),
                    column=column(line, width),
                    top=round(line.top_pt),
                    lines=[line],
                )
            elif current is not None:
                current.lines.append(line)
    if current is not None:
        head, label = head_and_label(current.lines)
        if head:
            current.head, current.label = head, label
            entries.append(current)
        elif entries:
            entries[-1].lines.extend(current.lines)
    return entries, layout


def read_valid_dedr(path: Path) -> dict[str, str]:
    identifiers = []
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream):
            if row and row[0].startswith("d"):
                identifiers.append(row[0][1:])
    result = {identifier.lower(): identifier for identifier in identifiers}
    by_number: dict[str, list[str]] = {}
    for identifier in identifiers:
        match = re.match(r"\d+", identifier)
        if match:
            by_number.setdefault(match.group(), []).append(identifier)
    for number, candidates in by_number.items():
        if len(candidates) == 1:
            result.setdefault(number, candidates[0])
    return result


def dedr_links(text: str, valid: dict[str, str]) -> tuple[list[str], list[str], bool]:
    links: list[str] = []
    invalid: list[str] = []
    uncertain = False
    for match in DEDR.finditer(text):
        uncertain = uncertain or bool(match.group("uncertain"))
        value = match.group("ids")
        if value.lower().startswith("app"):
            invalid.append(canonical(value))
            continue
        for cited in re.findall(r"\d+[a-z]?", value, re.I):
            key = cited.lower().lstrip("0") or "0"
            canonical_id = valid.get(key) or valid.get(cited.lower())
            if not canonical_id and re.fullmatch(r"\d+[a-z]", key):
                canonical_id = valid.get(key[:-1])
            if canonical_id and canonical_id not in links:
                links.append(canonical_id)
            elif not canonical_id and cited not in invalid:
                invalid.append(cited)
    return links, invalid, uncertain


def split_forms(head: str) -> list[str]:
    result = []
    for form in mechanical_head_repairs(head).split("/"):
        form = canonical(form).strip(" ,;")
        if form and form not in result:
            result.append(form)
    return result


def definition_lines(entry: Entry) -> list[str]:
    result = []
    for index, line in enumerate(entry.lines):
        value = canonical(line.text)
        if index and (EXAMPLE_START.match(value) or "—" in value):
            break
        result.append(value)
    return result


def analysis_parentheticals(text: str) -> tuple[str, list[str], list[str]]:
    analyses: list[str] = []
    notes: list[str] = []

    def replace(match: re.Match[str]) -> str:
        value = canonical(match.group())
        plain = value.casefold()
        if (
            "<" in value or "cf." in plain or "q.v." in plain
            or "recorded" in plain or "archaic form" in plain
            or "modern form" in plain or re.search(r"\b\d{4}\s*:", value)
        ):
            analyses.append(value)
            return " "
        if "christian usage" in plain or "formal speech" in plain:
            notes.append(value.strip("()"))
            return " "
        return value

    gloss = re.sub(r"\([^()]*\)", replace, text)
    return canonical(gloss), analyses, notes


def definition_parts(entry: Entry) -> tuple[str, tuple[str, ...], str, str]:
    text = join_lines(definition_lines(entry))
    match = LABEL.search(text[:320])
    if entry.label == "base.":
        special = re.search(r"(?:deictic base|remote deictic|INTERR base)", text, re.I)
        definition = text[special.end():] if special else text[len(entry.head):]
    elif entry.label == "sfx.":
        special = re.search(
            r"(?:(?:ACC|GEN|DAT|NEG|NOM|LOC|HORT|HYP|IMPER|OBLIG|OPT|S1|S2|"
            r"infinitive)\s+)?sfx\.?",
            text,
            re.I,
        )
        definition = text[special.end():] if special else text[len(entry.head):]
    elif entry.label == "adj.p." and "final of adjectival participles" in text.casefold():
        definition = text[len(entry.head):]
    else:
        definition = text[match.end():] if match else text[len(entry.head):]
    definition = definition.lstrip(" ,;:-")
    tags = list(POS_TAGS.get(entry.label, ()))
    for label_match in LABEL.finditer(definition):
        for tag in POS_TAGS.get(normalize_label(label_match.group("label")), ()):
            if tag not in tags:
                tags.append(tag)
    if entry.label == "base.":
        source_text = entry.raw_entry.casefold()
        tags.append(
            "demonstrative"
            if "deictic" in source_text or "remote" in source_text
            else "interr"
        )
    if DONOR.search(definition):
        tags.append("loanword")
    lower = definition.casefold()
    for needle, tag in (
        ("archaic", "archaic"), ("modern", "modern"),
        ("colloquial", "colloquial"), ("poetic", "poetic"),
    ):
        if needle in lower:
            tags.append(tag)
    cleaned, analyses, notes = analysis_parentheticals(definition)
    cleaned = DEDR.sub("", cleaned)
    cleaned = LABEL.sub(" ", cleaned)
    cleaned = canonical(cleaned).strip(" ,;:.")
    if not cleaned and entry.label == "idem.":
        cleaned = "same as the preceding dictionary entry"
        tags.append("alternate")
    if not cleaned and entry.label == "sfx.":
        case = re.search(
            r"\b(ACC|GEN|DAT|NEG|NOM|LOC|HORT|HYP|IMPER|OBLIG|OPT|S1|S2|"
            r"infinitive)\s+sfx\b",
            entry.raw_entry,
            re.I,
        )
        if case:
            cleaned = canonical(case.group(1)).lower() + " suffix"
    return (
        cleaned,
        tuple(dict.fromkeys(tags)),
        "; ".join(dict.fromkeys(analyses)),
        "; ".join(notes),
    )


def cross_reference_key(value: str, *, collapse_geminates: bool = False) -> str:
    """Normalize a printed head only enough to match OCR variants of a cross-reference."""
    value = unicodedata.normalize("NFD", canonical(value).casefold())
    value = "".join(character for character in value if not unicodedata.combining(character))
    value = re.sub(r"[^a-z0-9]+", "", value)
    if collapse_geminates:
        value = re.sub(r"([bcdfghjklmnpqrstvwxyz])\1+", r"\1", value)
    return value


def above_reference_forms(entry: Entry, gloss: str) -> list[str]:
    """Return heads named by the dictionary's ``cf. above`` convention."""
    source_text = join_lines(definition_lines(entry))
    if not re.search(r"\bcf\.\s*above\b", source_text, re.I):
        return []
    match = re.match(r"above\s*,\s*(.*)", gloss, re.I)
    if not match:
        return []
    value = re.split(r"\b(?:DEDR|DBIA)\b|\(", match.group(1), maxsplit=1)[0]
    return [canonical(form).strip(" ,;") for form in value.split("/") if form.strip(" ,;")]


def preceding_gloss(
    entry: Entry,
    forms: Sequence[str],
    gloss: str,
    preceding: Sequence[tuple[Entry, Sequence[str], str]],
) -> tuple[str, str]:
    """Resolve a printed ``cf. above`` to the nearest named preceding article."""
    references = above_reference_forms(entry, gloss)
    if not references:
        return gloss, ""

    targets = [*references, *forms]
    for collapse_geminates in (False, True):
        wanted = {
            cross_reference_key(value, collapse_geminates=collapse_geminates)
            for value in targets
        } - {""}
        for prior_entry, prior_forms, prior_gloss in reversed(preceding):
            keys = {
                cross_reference_key(value, collapse_geminates=collapse_geminates)
                for value in prior_forms
            }
            if wanted & keys:
                return prior_gloss, prior_entry.key

    # Two articles in the OCR cache have a named cross-reference embedded in a
    # preceding article that was fused with another head. Preserve the source's
    # explicit match rather than treating the printed word "above" as a gloss.
    wanted = [cross_reference_key(value) for value in references]
    for prior_entry, _prior_forms, prior_gloss in reversed(preceding):
        raw = cross_reference_key(prior_entry.raw_entry)
        if any(len(key) >= 5 and key in raw for key in wanted):
            return prior_gloss, prior_entry.key
    return gloss, ""


def rich_row(
    parameter: str,
    form: str,
    gloss: str,
    source: str,
    *,
    notes: str = "",
    etymology: str = "",
    entry_key: str = "",
    variant_of_key: str = "",
    tags: Sequence[str] = (),
) -> list[str]:
    return [
        LANGUAGE_ID, parameter, form, gloss, "", "", notes, source, "", etymology,
        entry_key, variant_of_key, "", "", " ".join(dict.fromkeys(tags)),
    ]


def build_rows(
    entries: Sequence[Entry],
    valid_dedr: dict[str, str],
    corrections: dict[str, OcrCorrection] | None = None,
) -> tuple[list[list[str]], list[dict[str, str]]]:
    corrections = corrections or {}
    rows: list[list[str]] = []
    audit: list[dict[str, str]] = []
    preceding: list[tuple[Entry, Sequence[str], str]] = []
    for entry in entries:
        forms = split_forms(entry.head)
        gloss, parsed_tags, analysis, notes = definition_parts(entry)
        gloss, _ = preceding_gloss(entry, forms, gloss, preceding)
        correction = corrections.get(entry.key)
        reviewed = bool(
            correction and correction.status in {"accepted", "corrected"}
        )
        if reviewed and correction is not None:
            forms[0] = correction.form or forms[0]
            gloss = correction.gloss or gloss
            notes = correction.notes or notes
            if correction.pos:
                corrected_label = normalize_label(correction.pos)
                parsed_tags = tuple(POS_TAGS.get(corrected_label, parsed_tags))
        links, invalid, link_uncertain = dedr_links(entry.raw_entry, valid_dedr)
        tags = list(parsed_tags) + ([] if reviewed else ["uncertain"])
        if link_uncertain:
            tags.append("uncertain")
        if len(forms) > 1:
            tags.append("alternate")
        status = "ingested"
        reason = ""
        if not forms or not gloss:
            status = "corrupt"
            reason = "empty parsed form or lexical definition"
        if any("�" in value for value in forms):
            status = "corrupt"
            reason = "replacement character in OCR head"
        if correction and correction.status in {"illegible", "skipped"}:
            status = "correction-excluded"
            reason = f"review overlay status: {correction.status}"

        source = f"{SOURCE_ID}[p. {entry.printed_page}, col. {entry.column}]"
        source_etymology = analysis
        if links:
            statement = "Hockings and Pilot-Raichoor cite DEDR " + ", ".join(links)
            source_etymology = "; ".join(filter(None, (source_etymology, statement)))
        if invalid:
            unresolved = "Unresolved printed DEDR citation(s): " + ", ".join(invalid)
            source_etymology = "; ".join(filter(None, (source_etymology, unresolved)))

        if status == "ingested":
            for index, parameter in enumerate(links or [""], 1):
                key = entry.key if index == 1 else f"{entry.key}:link:{index}"
                rows.append(rich_row(
                    f"d{parameter}" if parameter else "",
                    forms[0], gloss, source, notes=notes, etymology=source_etymology,
                    entry_key=key, tags=tags,
                ))
            for index, variant in enumerate(forms[1:], 1):
                rows.append(rich_row(
                    "", variant, gloss, source, notes=notes,
                    etymology=f"Printed alternate of {forms[0]}",
                    entry_key=f"{entry.key}:variant:{index}",
                    variant_of_key=entry.key,
                    tags=tuple(tags) + ("alternate",),
                ))

        audit.append({
            "Status": status,
            "Reason": reason,
            "Review_State": "needs_transcription_review",
            "Review_Reason": (
                "fresh unreviewed OCR; plain t/d/n/l may conceal a lost printed retroflex dot"
            ),
            "Entry_Key": entry.key,
            "PDF_Page": str(entry.pdf_page),
            "Printed_Page": str(entry.printed_page),
            "Column": str(entry.column),
            "Top": str(entry.top),
            "Raw_OCR": entry.raw_entry,
            "Raw_Head": entry.head,
            "Form": forms[0] if forms else "",
            "Variants": "|".join(forms[1:]),
            "POS": entry.label,
            "Gloss": gloss,
            "Tags": " ".join(dict.fromkeys(tags)),
            "Reference": source,
            "DEDR_IDs": "|".join(links),
            "Unresolved_DEDR_IDs": "|".join(invalid),
            "Etymology": source_etymology,
            "OCR_Confidence": f"{entry.confidence:.2f}",
        })
        if gloss:
            preceding.append((entry, forms, gloss))
    return rows, audit


def write_csv(path: Path, rows: Sequence[Sequence[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)


def write_audit(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def report(rows: Sequence[Sequence[str]], audit: Sequence[dict[str, str]]) -> str:
    statuses = Counter(row["Status"] for row in audit)
    return "\n".join((
        f"raw dictionary articles: {len(audit):,}",
        f"installed rich rows: {len(rows):,}",
        f"ingested articles: {statuses['ingested']:,}",
        f"corrupt/excluded articles: {statuses['corrupt']:,}",
        f"linked DEDR articles: {sum(bool(row['DEDR_IDs']) for row in audit):,}",
        f"unlinked articles: {sum(not row['DEDR_IDs'] and row['Status'] == 'ingested' for row in audit):,}",
        f"articles with printed alternates: {sum(bool(row['Variants']) for row in audit):,}",
        f"unresolved DEDR citations: {sum(bool(row['Unresolved_DEDR_IDs']) for row in audit):,}",
        f"unique stable entry keys: {len({row['Entry_Key'] for row in audit}):,}",
    ))


def parse_page_spec(spec: str | None) -> list[int]:
    if not spec:
        return [
            page for page in range(FIRST_PDF_PAGE, LAST_PDF_PAGE + 1)
            if page not in INSERTED_BLANK_PDF_PAGES
        ]
    pages: set[int] = set()
    for part in spec.split(","):
        if "-" in part:
            start, end = map(int, part.split("-", 1))
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part))
    return sorted(pages - INSERTED_BLANK_PDF_PAGES)


def printed_page_for(pdf_page: int) -> int:
    """Map scan leaves to printed dictionary pages.

    The first two unnumbered leaves are dictionary pages 1-2. Two additional
    blank scan leaves occur at PDF pages 443-444 between printed pp. 422-423;
    they are not part of the printed pagination and are excluded from OCR.
    """
    if pdf_page in INSERTED_BLANK_PDF_PAGES:
        raise ValueError(f"PDF page {pdf_page} is an inserted blank leaf")
    return pdf_page - (20 if pdf_page < min(INSERTED_BLANK_PDF_PAGES) else 22)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--pages", help="PDF pages for diagnostics, e.g. 21-25,50")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dedr-params", type=Path, default=DATA_ROOT / "data/dedr/params.csv")
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--corrections", type=Path, default=DEFAULT_CORRECTIONS)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args(argv)
    args.pdf = args.pdf.expanduser().resolve()
    if not args.pdf.exists():
        parser.error(f"PDF not found: {args.pdf}")
    if not shutil.which("tesseract"):
        parser.error("tesseract is not installed or not on PATH")

    pages = parse_page_spec(args.pages)
    jobs = [(str(args.pdf), page, str(args.cache_dir)) for page in pages]
    page_data: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(ocr_page, job): job[1] for job in jobs}
        for completed, future in enumerate(as_completed(futures), 1):
            page, data = future.result()
            page_data[page] = data
            if completed % 10 == 0 or completed == len(jobs):
                print(f"OCR pages: {completed}/{len(jobs)}", file=sys.stderr, flush=True)

    entries, layout = extract_entries(page_data.values())
    valid_dedr = read_valid_dedr(args.dedr_params)
    rows, audit = build_rows(entries, valid_dedr)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preview = args.output_dir / "hockings_badaga_import.csv"
    write_csv(preview, rows)
    write_audit(args.audit, audit)
    corrections = load_corrections(args.corrections, args.audit)
    if corrections:
        rows, _ = build_rows(entries, valid_dedr, corrections)
    if layout:
        write_audit(args.output_dir / "hockings_badaga_layout_exclusions.csv", layout)
    (args.output_dir / "hockings_badaga_report.txt").write_text(
        report(rows, audit) + "\n", encoding="utf-8"
    )
    if args.install:
        write_csv(DEFAULT_INSTALL, rows)
        print(f"installed {DEFAULT_INSTALL}")
    print(report(rows, audit))
    print(f"saved review decisions: {len(corrections)}")
    print(f"audit: {args.audit}")
    print(f"preview: {preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
