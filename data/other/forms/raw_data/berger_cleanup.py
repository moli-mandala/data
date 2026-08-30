#!/usr/bin/env python3
"""Clean, translate, audit, and install Berger's Burushaski dictionary.

This is the editorial layer over :mod:`berger`, which owns reproducible 300 dpi
OCR.  The older importer trusted Tesseract paragraph blocks and consequently
merged adjacent entries.  This layer rebuilds the source from the much more
stable line indentation in each of the four printed columns, removes the weaker
of a duplicated scan spread, restores the previously omitted first dictionary
page, and derives printed locators from the physical scan sequence.

German definitions are retained verbatim in the checked audit.  The installed
``Gloss`` is an English editorial translation.  ``--translate`` uses the pinned
Argos German-English package recorded in the manifest; the checked editorial
CSV remains authoritative, so normal rebuilds do not require the 150 MB model.

Run from ``data/``.  A normal audit/rebuild uses the existing page cache.  Pass
``--pdf`` to fill a missing cache using the OCR routine in ``berger.py``.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import gzip
import hashlib
import importlib.util
import io
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LEGACY_SCRIPT = HERE / "berger.py"
SPEC = importlib.util.spec_from_file_location("berger_legacy_ocr", LEGACY_SCRIPT)
legacy = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = legacy
SPEC.loader.exec_module(legacy)

SOURCE_ID = "berger"
AUTO_SOURCE_ID = "berger-auto"
SNAPSHOT_DATE = "2026-08-28"
PDF_SHA256 = "864cd94f8c41237aae2408e154f7b8d9e21c911b90f7d3ff6dd261568f05c0dc"
ARGOS_MODEL = "translate-de_en-1_3.argosmodel"
ARGOS_MODEL_SHA256 = "becc2b0011f8249fcb89be9ecb75ba0d876b1fab93c28ee6ff0420936897d637"
ARGOS_MODEL_URL = "https://argos-net.com/v1/translate-de_en-1_3.argosmodel"
TRANSLATION_OVERRIDES = {
    "brünstig, brunftig": "in heat, rutting",
    "hüpfen, springen": "leap, jump",
    "Nebenfrau, Konkubine": "concubine",
    "Weber": "weaver",
    "Wiesel, Ichneumon (besucht Häuser und gilt als glückbringend)": (
        "weasel, mongoose (visits houses and is considered lucky)"
    ),
}
FORM_OVERRIDES = {
    "berger:p141:c2:e001": "di-áarċ- -ś-",
}
GLOSS_OVERRIDES = {
    "berger:p066:c2:e008": "Weber",
    "berger:p093:c2:e009": "(Muttermilch aus der Brust) saugen; saugen an",
    "berger:p256:c2:e004": (
        "mit verkrüppelten Händen oder Füßen, NH mit nur einer Hand; "
        "ng. auch: mit einem verbogenen Horn"
    ),
    "berger:p257:c2:e009": "kurz",
    "berger:p305:c1:e003": "Wiesel, Ichneumon (besucht Häuser und gilt als glückbringend)",
    "berger:p395:c2:e008": "Hauptpfosten des Hauses, spielt bei Riten eine Rolle",
}
VARIANT_OVERRIDES = {
    "berger:p066:c2:e008": [("Bur", "biéeço", "Nager")],
}

CACHE_DIR = ROOT / ".cache/ocr/berger/pages"
AUTO_OUTPUT = ROOT / "data/other/forms/20260726-berger-auto.csv"
GOLD_OUTPUT = ROOT / "data/other/forms/20220930-berger.csv"
EDITORIAL = HERE / "20260828-berger-editorial.csv"
IDENTITY_MAP = HERE / "20260828-berger-entry-map.csv"
AUDIT_OUTPUT = HERE / "20260828-berger-audit.csv.gz"
SAMPLE_OUTPUT = HERE / "20260828-berger-sample.csv"
MANIFEST_OUTPUT = HERE / "20260828-berger-manifest.json"
LEGACY_INDEX = HERE / "20260828-berger-legacy-index.csv.gz"
LEGACY_AUTO_BLOB = "dcc80048197e274819a2908e6d6795b6e9aaecfd"
LEGACY_GOLD_BLOB = "6a6d2b8bf7637ef58c2075337606846db85a1b66"
LEGACY_INDEX_FIELDS = [
    "key", "language", "parameter", "form", "pdf", "printed", "gold", "gold_row",
]

# PDF p. 7 has the start of the dictionary only on the right-hand printed page.
# PDF pp. 50 and 51 are duplicate scans of printed pp. 94--95; p. 50 has the
# slightly higher mean OCR confidence and is retained.
PDF_PAGES = tuple(range(7, 248))
EXCLUDED_PDF_PAGES = {51}

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic",
    "Notes", "Source", "Cognateset", "Etymology", "Entry_Key",
    "Variant_Of_Key", "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
EDITORIAL_FIELDS = [
    "Entry_Key", "Form", "Raw_Gloss_German", "English_Gloss", "Review",
    "Translator", "Model", "Source_SHA256",
]
MAP_FIELDS = [
    "Stable_Key", "Installed_Key", "PDF_Page", "Printed_Page", "Column",
    "Entry_Ordinal", "OCR_Form", "Legacy_Form", "Method", "Score",
]
AUDIT_FIELDS = [
    "Snapshot_Date", "Stable_Key", "Installed_Key", "PDF_Page", "Printed_Page",
    "Column", "Entry_Ordinal", "Top", "Raw_OCR", "OCR_Confidence", "Raw_Form",
    "Final_Form", "Raw_Gloss_German", "English_Gloss", "Direct_Turner_IDs",
    "Etymology", "Tags", "Status", "Review", "Emitted_Keys", "Record_SHA256",
]
SOURCE_IMAGE_QA = {
    "berger:p009:c1:e001": "source-image verified",
    "berger:p031:c2:e004": "source-image verified",
    "berger:p057:c2:e007": "source-image verified; Nager variant retained",
    "berger:p085:c2:e004": "source-image verified; derived cross-reference retained",
    "berger:p113:c1:e009": "source-image verified",
    "berger:p141:c2:e001": "source-image verified; headword suffix and vowel length repaired",
    "berger:p167:c1:e006": "source-image verified; OCR suffix -śo repaired",
    "berger:p190:c1:e001": "source-image verified",
    "berger:p216:c1:e006": "source-image verified",
    "berger:p216:c1:e007": "source-image verified; dollar-sign OCR repaired to ś",
    "berger:p242:c1:e009": "source-image verified",
    "berger:p256:c2:e004": "source-image verified; terminal zero repaired to o",
    "berger:p265:c2:e002": "source-image verified",
    "berger:p290:c2:e004": "source-image verified",
    "berger:p290:c2:e003": "source-image verified",
    "berger:p291:c2:e004": "source-image verified; duplicated OCR vowel removed",
    "berger:p316:c2:e007": "source-image verified; sound-symbolic series retained",
    "berger:p341:c2:e003": "source-image verified",
    "berger:p364:c2:e006": "source-image verified",
    "berger:p390:c1:e007": "source-image verified",
    "berger:p414:c1:e008": "source-image verified",
    "berger:p436:c2:e005": "source-image verified; noun-class notation retained",
    "berger:p461:c2:e003": "source-image verified; line-break continuation retained",
    "berger:p486:c1:e002": "source-image verified; gutter collision and zuṭú repaired",
}
SOURCE_IMAGE_SAMPLE_KEYS = (
    "berger:p009:c1:e001", "berger:p031:c2:e004", "berger:p057:c2:e007",
    "berger:p085:c2:e004", "berger:p113:c1:e009", "berger:p141:c2:e001",
    "berger:p167:c1:e006", "berger:p190:c1:e001", "berger:p216:c1:e006",
    "berger:p242:c1:e009", "berger:p265:c2:e002", "berger:p290:c2:e003",
    "berger:p316:c2:e007", "berger:p341:c2:e003", "berger:p364:c2:e006",
    "berger:p390:c1:e007", "berger:p414:c1:e008", "berger:p436:c2:e005",
    "berger:p461:c2:e003", "berger:p486:c1:e002",
)

GOLD_ROW_REPAIRS = {
    23: ("berger-entry-334-dialect-1", 24),
    29: ("berger:gold:cdial9237:biedo", 30),
    30: ("berger:gold:cdial11225:baidaar", 31),
    31: ("berger:gold:cdial8431:bakor", 31),
    33: ("berger:gold:cdial9408:bala", 32),
    34: ("berger:gold:cdial6658:balaac-man", 32),
    35: ("berger:gold:cdial11406:balan-man", 33),
    36: ("berger:gold:cdial11406:balanees-man", 33),
    37: ("berger:gold:cdial9166:balando", 33),
    39: ("berger-entry-450-dialect-1", 33),
}
GOLD_TEXT_REPAIRS = {
    "Sontag": "Sonntag",
    "seelisch labü": "seelisch labil",
    "(Emtejahr)": "(Erntejahr)",
    "Pilanzen": "Pflanzen",
    "Empfangsraurn": "Empfangsraum",
    "fruher": "früher",
    "Grübem": "Gräbern",
    "fUr": "für",
    "Lümmer": "Lämmer",
    "nutzlich": "nützlich",
    "herunterhüngen": "herunterhängen",
    "Wascheaufhüngen": "Wäscheaufhängen",
}

HEADER_RE = re.compile(r"^(?:\d+\s+)?Burushaski\s*[-~–—]+\s*Deutsch(?:\s+\d+)?$", re.I)
SOURCE_START_RE = re.compile(
    r"^\((?:u\.|sh\.|ys\.|kho\.|vgl\.|e\.|tü\.|ti\.|skt\.|pa\.|wa\.|balti\.)",
    re.I,
)
SOURCE_INLINE_RE = re.compile(
    r"\s*\((?:u\.|sh\.|ys\.|kho\.|vgl\.|e\.|tü\.|ti\.|skt\.|pa\.|wa\.|balti\.)[^)]*\)",
    re.I,
)
GERMAN_CONTINUATIONS = {
    "aber", "auch", "auf", "aus", "bei", "bis", "das", "davon", "dem",
    "den", "der", "des", "die", "durch", "ein", "eine", "einem", "einen",
    "einer", "eines", "er", "es", "für", "gegen", "im", "in", "ist", "man",
    "mit", "nach", "nicht", "noch", "oder", "ohne", "sehr", "sich", "so",
    "und", "unter", "vom", "von", "vor", "wenn", "wie", "wird", "zu", "zum",
    "zur",
}
GERMAN_CAPITAL_CONTINUATIONS = {
    "Art", "Antwort", "Befehl", "Bevölkerung", "Blick", "Dorf", "Ende",
    "Frau", "Geld", "Gegend", "Gericht", "Gott", "Hilfe", "Krankheit", "Land",
    "Leichtigkeit", "Linderung", "Mann", "Mensch", "Musik", "Person", "Rast",
    "Rede", "Ruhe", "Sache", "Schaf", "Schwierigkeit", "Tier", "Unglück",
    "Weg", "Ziege", "Zustand",
}
BURUSHASKI_MARK_RE = re.compile(
    r"[áàâãåćċčçḍéèêíìîṅńṇóòôõqśšṣṭúùûźžżẓħγļŗŘőűŭ]", re.I
)


@dataclass
class RawUnit:
    pdf_page: int
    printed_page: int
    column: int
    entry_ordinal: int
    left: int
    top: int
    lines: list[dict]
    stable_key: str

    @property
    def raw_ocr(self) -> str:
        return "\n".join(line["text"] for line in self.lines)

    @property
    def text(self) -> str:
        return legacy.canonical(" ".join(line["text"] for line in self.lines))

    @property
    def confidence(self) -> float:
        values = [float(line["confidence"]) for line in self.lines]
        return sum(values) / len(values) if values else 0.0


@dataclass
class CleanEntry:
    unit: RawUnit
    language: str
    form: str
    gloss_de: str
    parameter_ids: list[str]
    etymology: str
    tags: list[str]
    installed_key: str
    variant_of_stable: str = ""
    derivation_parent_stable: str = ""
    english_gloss: str = ""
    review: list[str] = field(default_factory=list)
    gold_row: int = 0


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def printed_page(pdf_page: int, side: int) -> int:
    if pdf_page <= 50:
        return 2 * pdf_page - 6 + side
    return 2 * pdf_page - 8 + side


def physical_column(left: int, width: int) -> int:
    if left < width / 2:
        return 0 if left < width * 0.22 else 1
    return 2 if left < width * 0.67 else 3


def allowed_columns(pdf_page: int) -> tuple[int, ...]:
    if pdf_page == 7:
        return (2, 3)
    if pdf_page == 247:
        return (0, 1)
    return (0, 1, 2, 3)


def is_header_or_heading(text: str) -> bool:
    normalized = legacy.canonical(text)
    return bool(
        not normalized
        or HEADER_RE.match(normalized)
        or normalized in {"Burushaski", "Burushaski - Deutsch", "Burushaski ~- Deutsch", "fE; L 391"}
        or re.fullmatch(r"(?:Burushaski\s*)?[-~]?\s*Deutsch(?:\s+\d+)?", normalized, re.I)
        or re.fullmatch(r"\d{1,3}", normalized)
        or re.fullmatch(r"[a-zäöüśćčž]", normalized, re.I)
    )


def load_pages(cache_dir: Path, pdf: Path | None = None) -> list[dict]:
    missing = [page for page in PDF_PAGES if not (cache_dir / f"page-{page:03}.json").exists()]
    if missing and not pdf:
        raise FileNotFoundError(
            f"Missing Berger OCR cache pages {missing[:10]}; pass --pdf to render/OCR them"
        )
    if missing:
        for page in missing:
            legacy.ocr_page((str(pdf), page, str(cache_dir)))
    pages = []
    for page in PDF_PAGES:
        data = json.loads((cache_dir / f"page-{page:03}.json").read_text(encoding="utf-8"))
        if data["pdf_page"] != page:
            raise ValueError(f"Cache page mismatch: expected {page}, got {data['pdf_page']}")
        pages.append(data)
    return pages


def reconstruct_units(pages: Iterable[dict]) -> list[RawUnit]:
    units: list[RawUnit] = []
    current: RawUnit | None = None
    ordinals: Counter[tuple[int, int]] = Counter()
    for data in sorted(pages, key=lambda row: row["pdf_page"]):
        pdf_page = data["pdf_page"]
        if pdf_page in EXCLUDED_PDF_PAGES:
            continue
        width = data["width"]
        columns: dict[int, list[dict]] = defaultdict(list)
        for raw_line in data["lines"]:
            line = dict(raw_line)
            line["text"] = legacy.canonical(line["text"])
            line["text"] = re.sub(r"-\$0|-śŚ0|-ś0", "-śo", line["text"])
            if pdf_page == 36:
                line["text"] = line["text"].replace(
                    "buyéeço und biéeço -tiù ng. Weber",
                    "buyéeço und biéeço -tiň ng. Weber",
                ).replace("(sh. buyé&ēço, T 11307)", "(sh. buyéēço, T 11307)")
            if pdf_page == 119:
                line["text"] = line["text"].replace(
                    "Gsh. juṅ \"Bergschlucht\", T 6429)",
                    "(sh. juṅ \"Bergschlucht\", T 6429)",
                )
            if pdf_page == 150:
                line["text"] = line["text"].replace(
                    "(ys. -muśt, vgl. ng. muśtí unter ?múçi; sh.",
                    "(ys. -muṣṭ, vgl. ng. muṣṭí unter ²múçi; sh.",
                )
            if pdf_page == 177:
                line["text"] = line["text"].replace("{ys. qumá,", "(ys. qumá,")
            if pdf_page == 201:
                line["text"] = line["text"].replace(
                    "Siridáko -muċ, ng. surdúň dáko",
                    "śiridáko -muć, ng. śurdúň ḍáko",
                ).replace("T 12708 + gáko", "T 12708 + ḍáko")
            if pdf_page == 112 and line["text"].startswith("di-ísqis- -$-"):
                line["text"] = line["text"].replace("-$-", "-ś-", 1)
            if pdf_page == 247:
                line["text"] = line["text"].replace(
                    "zułtú hz.ng. unrein durch Pollution, zuzáq y Hölle L 391",
                    "zuṭú hz.ng. unrein durch Pollution,",
                ).replace(
                    "Menstruation, oder wenn man (sh. zozák, u. dōzax)",
                    "Menstruation, oder wenn man",
                )
            columns[physical_column(line["left"], width)].append(line)
        # Several adjacent scans have shifted crop boxes (most conspicuously PDF
        # pp. 18--23). Infer each column's entry margin from its own lower-tail
        # line positions instead of assuming one global x coordinate.
        starts = []
        for physical in range(4):
            positions = sorted(
                line["left"] for line in columns[physical]
                if line["top"] > 430 and not is_header_or_heading(line["text"])
            )
            if not positions:
                starts.append(width * (0.045, 0.24, 0.505, 0.69)[physical])
                continue
            starts.append(positions[max(0, round((len(positions) - 1) * 0.08))])
        for physical in allowed_columns(pdf_page):
            page_side = int(physical >= 2)
            source_column = physical % 2 + 1
            actual_page = printed_page(pdf_page, page_side)
            for line in sorted(columns[physical], key=lambda row: (row["top"], row["left"])):
                text = line["text"]
                if is_header_or_heading(text):
                    continue
                at_margin = line["left"] <= starts[physical] + 28
                hyphen_continuation = bool(
                    current and current.lines[-1]["text"].rstrip().endswith("-")
                    and text[:1].islower()
                )
                lexical = at_margin and not hyphen_continuation and legacy._looks_lexical(text)
                if lexical:
                    ordinals[(actual_page, source_column)] += 1
                    ordinal = ordinals[(actual_page, source_column)]
                    stable = f"berger:p{actual_page:03d}:c{source_column}:e{ordinal:03d}"
                    current = RawUnit(
                        pdf_page, actual_page, source_column, ordinal,
                        line["left"], line["top"], [line], stable,
                    )
                    units.append(current)
                elif current is not None:
                    # Top-of-column lines often close the last entry in the preceding column.
                    current.lines.append(line)
    return units


def dehyphenate_lines(lines: Sequence[str]) -> str:
    result = ""
    for line in lines:
        line = legacy.canonical(line)
        if not result:
            result = line
        elif result.endswith("-") and line[:1].islower():
            result = result[:-1] + line
        else:
            result += " " + line
    return legacy.canonical(result)


def likely_subentry(line: str) -> bool:
    text = legacy.canonical(line)
    if not text or SOURCE_START_RE.match(text):
        return bool(text)
    if re.match(r"^(?:davon|dazu)\b", text, re.I):
        return True
    if re.match(r"^-\S+\s+", text):
        return False
    first = text.split(maxsplit=1)[0].strip(".,;:()\"")
    bare = legacy.normalize_key(first)
    if not bare or bare in GERMAN_CONTINUATIONS or first in GERMAN_CAPITAL_CONTINUATIONS:
        return False
    # Citation-only and closing semantic-parenthesis lines continue the definition.
    if re.match(r"^(?:K|L)\s*\d|^\d+(?:\.\d+)*\b|^\)", text):
        return False
    marked = bool(re.search(r"[ÁÉÍÓÚáéíóúÀÈÌÒÙàèìòùĀāĪīŪūṬṭḌḍṄṅŅņṚṛŚśṢṣĆćČčɢɣ]", first))
    stem = first.endswith("-") or "-" in first
    return marked or stem


def repair_form(form: str, unit: RawUnit, dominant_initial: str) -> str:
    value = legacy.canonical(form)
    value = {
        "khóoś0": "khóośo",
        "móőtṭis": "móṭis",
    }.get(value, value)
    # A superscript homonym 1 is often OCRed as l/I.  Use the alphabetical page
    # context so genuine l-initial entries in the L section remain untouched.
    if len(value) > 1 and value[0] in "lI" and dominant_initial and dominant_initial != "l":
        remainder = legacy.normalize_key(value[1:])
        if remainder.startswith(dominant_initial):
            value = value[1:]
    if dominant_initial and dominant_initial != "z":
        remainder = re.sub(r"^[ZŽ]{1,2}", "", value)
        if remainder != value and legacy.normalize_key(remainder).startswith(dominant_initial):
            value = remainder
    value = value.replace("!", "")
    return legacy.canonical(value).strip(" ,.;:")


def extract_core_gloss(unit: RawUnit, form: str) -> str:
    kept = [unit.lines[0]["text"]]
    for line in unit.lines[1:]:
        if len(legacy.normalize_key(line["text"])) <= 1:
            continue
        if likely_subentry(line["text"]):
            break
        kept.append(line["text"])
    text = dehyphenate_lines(kept)
    prefix = legacy._clean_head_prefix(text)
    raw_heads = legacy._head_forms(prefix)
    raw_head = raw_heads[0] if raw_heads else form
    head_at = text.find(raw_head) if raw_head else -1
    body = text[head_at + len(raw_head):] if head_at >= 0 else text
    body = re.sub(
        r"^(?:\s*(?:hz\.ng\.?|hz\.?|ng\.?|NH\b|fem\.?\s*[-\w]+|"
        r"mask\.?\s*[-\w]+|sg\.?|pl\.?|D\.pl\.?|[hxy]\b|hm\b|adj\.?|"
        r"adv\.?|Pron\.?|Postp\.?|Konj\.?|Interj\.?|-[-\wćśṭḍṅ]+|,|;))+",
        "", body, flags=re.I,
    )
    source = SOURCE_INLINE_RE.search(body)
    if source:
        body = body[:source.start()]
    body = re.sub(r"\b(?:K|L)\s*\d+(?:[.;,:]\d+)*(?:\s*u\.a\.)?", "", body)
    body = re.sub(r"\b\d{1,2}\.\d{1,2}(?:[.;]\d{1,2})*\b", "", body)
    body = re.sub(r"\s+", " ", body).strip(" ,.;:-")
    # Do not let an OCRed form prefix leak back into the gloss when form repair changed it.
    if form and body.startswith(form):
        body = body[len(form):].lstrip(" ,.;:-")
    return legacy.canonical(body)


def direct_turner_ids(text: str, valid: set[str]) -> list[str]:
    values = []
    for match in legacy.TURNER_RE.finditer(text):
        before = text[max(0, match.start() - 14):match.start()]
        after = text[match.end():match.end() + 3]
        if re.search(r"(?:vgl\.?|\bzu)\s*$", before, re.I) or "?" in after:
            continue
        value, method = legacy.repair_id(match.group(1), valid)
        if value and method in {"exact", "repaired"} and value not in values:
            values.append(value)
    # Berger occasionally places several unhedged Turner articles after a
    # single ``T`` (e.g. ``T 3315; 14398``).  The legacy regex captures only
    # the first article, so recover the explicitly coordinated continuations.
    for match in re.finditer(r"\bT\s*(\d+[a-z]?(?:\.\d+)?)((?:\s*;\s*\d+[a-z]?(?:\.\d+)?)+)", text):
        before = text[max(0, match.start() - 14):match.start()]
        if re.search(r"(?:vgl\.?|\bzu)\s*$", before, re.I):
            continue
        for raw in re.findall(r"\d+[a-z]?(?:\.\d+)?", match.group(2)):
            value, method = legacy.repair_id(raw, valid)
            if value and method in {"exact", "repaired"} and value not in values:
                values.append(value)
    return values


def plausible_alphabetic_outlier(form: str, gloss: str) -> bool:
    """Keep marked Burushaski forms while rejecting column-flow German prose."""
    return bool(
        BURUSHASKI_MARK_RE.search(form)
        or re.match(r"^(?:[+?]|d[:+-]|di[-:])", form, re.I)
        or ("-" in form and len(form) <= 18)
        or re.match(r"^(?:s|ds)\.\s+", gloss, re.I)
    )


def plausible_variant(head: str, variant: str) -> bool:
    """Reject OCR/parser captures of German gloss words as dialect variants."""
    if not variant or variant[:1].isupper() or re.search(r"[0-9$<>]", variant):
        return False
    source = legacy.normalize_key(head)
    target = legacy.normalize_key(variant)
    if not source or not target:
        return False
    return difflib.SequenceMatcher(None, source, target).ratio() >= 0.35


def parse_entries(units: Sequence[RawUnit], valid_ids: set[str]) -> list[CleanEntry]:
    provisional = []
    initials: dict[int, Counter[str]] = defaultdict(Counter)
    for unit in units:
        prefix = legacy._clean_head_prefix(unit.text)
        form = legacy._head_forms(prefix)[0] if legacy._head_forms(prefix) else ""
        key = legacy.normalize_key(form)
        if key:
            initials[unit.printed_page][key[0]] += 1
        provisional.append((unit, form))
    dominant = {
        page: counts.most_common(1)[0][0] for page, counts in initials.items() if counts
    }
    legacy_initials: dict[int, Counter[str]] = defaultdict(Counter)
    for row in load_legacy_rows():
        key = legacy.normalize_key(row["form"])
        if row["printed"] and key:
            legacy_initials[row["printed"]][key[0]] += 1
    expected_initials = {}
    for page, counts in legacy_initials.items():
        peak = counts.most_common(1)[0][1]
        expected_initials[page] = {
            initial for initial, count in counts.items() if count >= max(2, peak * 0.2)
        }
    # The restored first printed page was absent from the old extraction.
    expected_initials[9] = expected_initials.get(10, {"a"})

    entries: list[CleanEntry] = []
    parent_by_column: dict[tuple[int, int], str] = {}
    for unit, raw_form in provisional:
        form = repair_form(raw_form, unit, dominant.get(unit.printed_page, ""))
        form = FORM_OVERRIDES.get(unit.stable_key, form)
        review = []
        obvious_nonentry = (
            legacy.normalize_key(form) in {"burushaski", "deutsch"}
            or form.casefold() in {"séance"}
            or bool(re.search(r"[0-9$<>]", form))
            or bool(form[:1].isupper() and not BURUSHASKI_MARK_RE.search(form))
        )
        if not form or obvious_nonentry or not legacy._looks_lexical(f"{form} Bedeutung"):
            review.append("suspicious-form")
        normalized = legacy.normalize_key(form)
        page_initial = dominant.get(unit.printed_page, "")
        expected = expected_initials.get(unit.printed_page, {page_initial} if page_initial else set())
        alphabetic_outlier = (
            normalized and expected and normalized[0] not in expected
        )
        if alphabetic_outlier:
            review.append("alphabetic-outlier")
        gloss = GLOSS_OVERRIDES.get(unit.stable_key, extract_core_gloss(unit, form))
        if (
            alphabetic_outlier
            and unit.stable_key not in SOURCE_IMAGE_QA
            and not plausible_alphabetic_outlier(form, gloss)
        ):
            if "suspicious-form" not in review:
                review.append("suspicious-form")
        if not gloss:
            review.append("missing-gloss")
        derived = bool(re.match(r"^(?:davon|dazu)\b", unit.text, re.I))
        parent = parent_by_column.get((unit.printed_page, unit.column), "") if derived else ""
        ids = direct_turner_ids(unit.text, valid_ids)
        tags = legacy._grammar_tags(unit.text, form)
        if derived and "derived" not in tags:
            tags.append("derived")
        # The complete auto tranche remains OCR-derived.  The audit carries the
        # typed reason while ``uncertain`` makes the state discoverable in Jambu.
        if "uncertain" not in tags:
            tags.append("uncertain")
        entry = CleanEntry(
            unit=unit, language="Bur", form=form, gloss_de=gloss,
            parameter_ids=ids, etymology=legacy._free_etymology(unit.text),
            tags=tags, installed_key=unit.stable_key,
            derivation_parent_stable=parent, review=[*review, "ocr-unreviewed"],
        )
        entries.append(entry)
        if not derived:
            parent_by_column[(unit.printed_page, unit.column)] = unit.stable_key

        variants = [
            item for item in legacy._dialect_variants(unit.text, form)
            if item[0] == "Bur" and plausible_variant(form, item[1])
        ]
        yasin = legacy.yasin_variant(unit.text)
        if (
            yasin and legacy.normalize_key(yasin) != legacy.normalize_key(form)
            and plausible_variant(form, yasin)
        ):
            variants.insert(0, ("Werch", yasin, "Yasin"))
        variants = VARIANT_OVERRIDES.get(unit.stable_key, variants)
        for index, (language, variant, dialect) in enumerate(variants, 1):
            variant_stable = f"{unit.stable_key}:dialect:{index}"
            variant_tags = legacy._grammar_tags(unit.text, variant, dialect=dialect)
            variant_tags.extend(tag for tag in ("alternate", "uncertain") if tag not in variant_tags)
            variant_review = ["ocr-unreviewed"]
            if not gloss:
                variant_review.insert(0, "missing-gloss")
            entries.append(CleanEntry(
                unit=unit, language=language, form=variant, gloss_de=gloss,
                parameter_ids=ids, etymology=entry.etymology, tags=variant_tags,
                installed_key=variant_stable, variant_of_stable=unit.stable_key,
                review=variant_review,
            ))
    return entries


def parse_source_page(source: str) -> tuple[int, int]:
    pdf = re.search(r"\[p\.\s*(\d+)\s*\(printed p\.\s*(\d+)\)\]", source)
    if pdf:
        return int(pdf.group(1)), int(pdf.group(2))
    printed = re.search(r"\[p\.\s*(\d+)\]", source)
    return (0, int(printed.group(1))) if printed else (0, 0)


def normalize_gold_rows(source_rows: Iterable[Sequence[str]]) -> list[list[str]]:
    rows = []
    for index, original in enumerate(source_rows, 1):
        row = list(original) + [""] * (15 - len(original))
        if index in GOLD_ROW_REPAIRS:
            key, page = GOLD_ROW_REPAIRS[index]
            row[10] = key
            row[7] = f"berger[p. {page}]"
        for bad, good in GOLD_TEXT_REPAIRS.items():
            row[3] = row[3].replace(bad, good)
        rows.append(row)
    return rows


def normalized_gold_rows() -> list[list[str]]:
    with GOLD_OUTPUT.open(encoding="utf-8") as stream:
        return normalize_gold_rows(csv.reader(stream))


def baseline_csv(blob: str) -> list[list[str]]:
    """Read the pinned pre-cleanup CSV blob used only for identity continuity."""
    data = subprocess.check_output(
        ["git", "cat-file", "blob", blob], cwd=ROOT, text=True,
    )
    return list(csv.reader(io.StringIO(data)))


def load_legacy_rows() -> list[dict]:
    if LEGACY_INDEX.exists():
        with gzip.open(LEGACY_INDEX, "rt", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            row["pdf"] = int(row["pdf"] or 0)
            row["printed"] = int(row["printed"] or 0)
            row["gold"] = row["gold"] == "1"
            row["gold_row"] = int(row["gold_row"] or 0)
        return rows
    rows = []
    for source_rows, gold in (
        (baseline_csv(LEGACY_AUTO_BLOB), False),
        (normalize_gold_rows(baseline_csv(LEGACY_GOLD_BLOB)), True),
    ):
        if gold:
            source_rows = normalize_gold_rows(source_rows)
        for index, row in enumerate(source_rows, 1):
            row.extend([""] * (15 - len(row)))
            pdf, printed = parse_source_page(row[7])
            rows.append({
                "key": row[10], "language": row[0], "parameter": row[1],
                "form": row[2], "pdf": pdf, "printed": printed,
                "gold": gold, "gold_row": index,
            })
    rows = [row for row in rows if row["key"]]
    with gzip.open(LEGACY_INDEX, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, LEGACY_INDEX_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "gold": "1" if row["gold"] else "0"})
    return rows


def catalog_evidence_keys() -> set[str]:
    path = ROOT / "data/burushaski_cognates.csv"
    with path.open(encoding="utf-8") as stream:
        return {
            key
            for row in csv.DictReader(stream)
            for key in row["Evidence_Keys"].split("|")
            if key.startswith("berger-")
        }


def legacy_catalog_key(key: str) -> str:
    """Return the pinned pre-cleanup key behind an explicit graph compatibility alias."""
    return key.removesuffix(":legacy-graph")


def baseline_auto_by_key() -> dict[str, list[str]]:
    rows = (list(row) + [""] * 15 for row in baseline_csv(LEGACY_AUTO_BLOB))
    return {row[10]: row[:15] for row in rows if row[10]}


def catalog_source_glosses(baseline: dict[str, list[str]]) -> dict[str, str]:
    """Inherit the source definition across each curated correspondence set."""
    result = {}
    with (ROOT / "data/burushaski_cognates.csv").open(encoding="utf-8") as stream:
        for catalog in csv.DictReader(stream):
            keys = [key for key in catalog["Evidence_Keys"].split("|") if key.startswith("berger-")]
            candidates = [
                baseline[legacy_catalog_key(key)][3]
                for key in keys
                if legacy_catalog_key(key) in baseline
                and baseline[legacy_catalog_key(key)][3]
            ]
            if catalog["Gloss"]:
                candidates.append(catalog["Gloss"])
            gloss = candidates[0] if candidates else "curated dialect correspondence"
            for key in keys:
                result[key] = baseline.get(
                    legacy_catalog_key(key), ["", "", "", gloss]
                )[3] or gloss
    return result


def sequence_map(news: list[CleanEntry], olds: list[dict]) -> list[tuple[int, int, float]]:
    """Order-preserving page-local alignment with conservative fuzzy matching."""
    n, m = len(news), len(olds)
    gap = -0.28
    scores = [[0.0] * (m + 1) for _ in range(n + 1)]
    steps = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        scores[i][0] = i * gap
        steps[i][0] = "n"
    for j in range(1, m + 1):
        scores[0][j] = j * gap
        steps[0][j] = "o"
    ratios = {}
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            ratio = difflib.SequenceMatcher(
                None, legacy.normalize_key(news[i - 1].form),
                legacy.normalize_key(olds[j - 1]["form"]),
            ).ratio()
            ratios[(i, j)] = ratio
            match = scores[i - 1][j - 1] + (ratio if ratio >= 0.55 else -0.8)
            skip_new = scores[i - 1][j] + gap
            skip_old = scores[i][j - 1] + gap
            scores[i][j], steps[i][j] = max(
                (match, "m"), (skip_new, "n"), (skip_old, "o"), key=lambda item: item[0]
            )
    result = []
    i, j = n, m
    while i or j:
        step = steps[i][j]
        if step == "m":
            ratio = ratios[(i, j)]
            if ratio >= 0.55:
                result.append((i - 1, j - 1, ratio))
            i -= 1
            j -= 1
        elif step == "n":
            i -= 1
        else:
            j -= 1
    return list(reversed(result))


def build_identity_map(entries: Sequence[CleanEntry]) -> list[dict]:
    bases = [entry for entry in entries if not entry.variant_of_stable]
    legacy_rows = load_legacy_rows()
    used_old = set()
    records = []
    by_pdf_new: dict[int, list[CleanEntry]] = defaultdict(list)
    by_pdf_old: dict[int, list[dict]] = defaultdict(list)
    for entry in bases:
        by_pdf_new[entry.unit.pdf_page].append(entry)
    for row in legacy_rows:
        if row["pdf"] and re.fullmatch(r"berger-entry-\d+", row["key"]):
            by_pdf_old[row["pdf"]].append(row)
    for page in sorted(by_pdf_new):
        news = by_pdf_new[page]
        olds = by_pdf_old.get(page, [])
        for ni, oi, score in sequence_map(news, olds):
            new, old = news[ni], olds[oi]
            if old["key"] in used_old:
                continue
            stable = new.installed_key
            new.installed_key = old["key"]
            used_old.add(old["key"])
            records.append(map_record(new, old, "page-sequence", score, stable))

    # The hand-entered tranche cites printed rather than PDF pages and is the
    # lexical authority. It may replace a weaker auto alignment.
    gold_rows = [row for row in legacy_rows if row["gold"]]
    used_gold_stables: set[str] = set()
    for old in gold_rows:
        candidates = [
            entry for entry in bases
            if not old["printed"] or entry.unit.printed_page == old["printed"]
            if entry.unit.stable_key not in used_gold_stables
        ]
        if not candidates:
            continue
        best = max(candidates, key=lambda entry: difflib.SequenceMatcher(
            None, legacy.normalize_key(entry.form), legacy.normalize_key(old["form"])
        ).ratio())
        score = difflib.SequenceMatcher(
            None, legacy.normalize_key(best.form), legacy.normalize_key(old["form"])
        ).ratio()
        if score < 0.72:
            continue
        replaced = best.installed_key
        if replaced != best.unit.stable_key:
            used_old.discard(replaced)
            records = [row for row in records if row["Installed_Key"] != replaced]
        stable = best.unit.stable_key
        best.installed_key = old["key"]
        used_gold_stables.add(stable)
        used_old.add(old["key"])
        records.append(map_record(best, old, "printed-form", score, stable))

    # Recover further unmatched auto identities by a unique page-local fuzzy
    # match. This catches corrections whose order was obscured by old paragraph
    # merges without forcing ambiguous homographs.
    for old in [row for row in legacy_rows if not row["gold"] and row["key"] not in used_old]:
        candidates = [
            entry for entry in bases
            if entry.installed_key == entry.unit.stable_key
            and (not old["pdf"] or entry.unit.pdf_page == old["pdf"])
        ]
        scored = sorted(((
            difflib.SequenceMatcher(
                None, legacy.normalize_key(entry.form), legacy.normalize_key(old["form"])
            ).ratio(), entry
        ) for entry in candidates), key=lambda item: item[0])
        if not scored or scored[-1][0] < 0.72:
            continue
        if len(scored) > 1 and scored[-1][0] - scored[-2][0] < 0.08:
            continue
        score, best = scored[-1]
        stable = best.installed_key
        best.installed_key = old["key"]
        used_old.add(old["key"])
        records.append(map_record(best, old, "unique-page-form", score, stable))

    # Preserve old variant keys through an exact page/form/language match.
    variants = [entry for entry in entries if entry.variant_of_stable]
    for entry in variants:
        candidates = [
            old for old in legacy_rows if old["key"] not in used_old
            and old["language"] == entry.language
            and old["printed"] == entry.unit.printed_page
            and legacy.normalize_key(old["form"]) == legacy.normalize_key(entry.form)
        ]
        if len(candidates) == 1:
            old = candidates[0]
            stable = entry.installed_key
            entry.installed_key = old["key"]
            used_old.add(old["key"])
            records.append(map_record(entry, old, "exact-variant", 1.0, stable))
    return sorted(records, key=lambda row: row["Stable_Key"])


def map_record(entry: CleanEntry, old: dict, method: str, score: float, stable: str) -> dict:
    return {
        "Stable_Key": stable,
        "Installed_Key": old["key"], "PDF_Page": entry.unit.pdf_page,
        "Printed_Page": entry.unit.printed_page, "Column": entry.unit.column,
        "Entry_Ordinal": entry.unit.entry_ordinal, "OCR_Form": entry.form,
        "Legacy_Form": old["form"], "Method": method, "Score": f"{score:.4f}",
    }


def write_csv(path: Path, rows: Iterable[Sequence], header: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        if header:
            writer.writerow(header)
        writer.writerows(rows)


def write_dict_csv(path: Path, rows: Iterable[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def apply_identity_map(entries: Sequence[CleanEntry], rebuild: bool = False) -> None:
    if rebuild or not IDENTITY_MAP.exists():
        records = build_identity_map(entries)
        write_dict_csv(IDENTITY_MAP, records, MAP_FIELDS)
    mapping = {}
    with IDENTITY_MAP.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            mapping[row["Stable_Key"]] = row["Installed_Key"]
    stable_to_installed = {}
    for entry in entries:
        if entry.variant_of_stable:
            installed = mapping.get(entry.installed_key, entry.installed_key)
        else:
            installed = mapping.get(entry.unit.stable_key, entry.installed_key)
        entry.installed_key = installed
        if not entry.variant_of_stable:
            stable_to_installed[entry.unit.stable_key] = installed
    for entry in entries:
        if entry.variant_of_stable and entry.installed_key.startswith("berger:p"):
            parent = stable_to_installed.get(entry.variant_of_stable, entry.variant_of_stable)
            suffix = entry.installed_key.removeprefix(entry.unit.stable_key)
            entry.installed_key = parent + suffix


def load_editorial() -> dict[str, dict]:
    if not EDITORIAL.exists():
        return {}
    with EDITORIAL.open(encoding="utf-8") as stream:
        return {row["Entry_Key"]: row for row in csv.DictReader(stream)}


def translation_source(entry: CleanEntry) -> str:
    value = entry.gloss_de.strip()
    if re.match(r"^s\.\s+", value, re.I):
        rest = re.sub(r"^s\.\s+", "", value, flags=re.I)
        if re.fullmatch(r"das Fg\.?", rest, re.I):
            return "Siehe den folgenden Eintrag."
        return f"Siehe {rest.rstrip('.')} ."
    if re.match(r"^ds\.\s+", value, re.I):
        rest = re.sub(r"^ds\.\s+(?:wie\s+)?", "", value, flags=re.I)
        return f"Dasselbe wie {rest.rstrip('.')} ."
    return f"Das Wort bedeutet: {value.rstrip('.')} ."


def clean_translation(source: str, target: str) -> str:
    value = re.sub(r"^The word means\s*:?\s*", "", target.strip(), flags=re.I)
    value = value.rstrip().removesuffix(".").strip()
    value = re.sub(r"\bsmall cattle\b", "small livestock", value, flags=re.I)
    value = re.sub(r"\bthat is, cattle\b", "that is, livestock", value, flags=re.I)
    if value[:1].isupper() and not re.match(r"(?:God|Allah|Indus|Hunza|Nager)\b", value):
        value = value[:1].lower() + value[1:]
    return postprocess_translation(value)


def postprocess_translation(value: str) -> str:
    def decode(match: re.Match[str]) -> str:
        try:
            return bytes(int(value, 16) for value in re.findall(r"0x([0-9A-Fa-f]{2})", match.group())).decode()
        except (ValueError, UnicodeDecodeError):
            return match.group()
    value = re.sub(r"(?:<0x[0-9A-Fa-f]{2}>)+", decode, value)
    value = re.sub(r"\bnH\b", "NH", value)
    value = re.sub(r"\bsee das Fg\.?", "see following entry", value, flags=re.I)
    value = re.sub(r"\bds\.?\b", "same as preceding entry", value, flags=re.I)
    if value.strip().casefold().rstrip(".") == "ds":
        return "same as preceding entry"
    return value


def generate_editorial(entries: Sequence[CleanEntry], argos_package: Path) -> None:
    try:
        import ctranslate2
        from argostranslate.package import Package
    except ImportError as exc:
        raise RuntimeError("Install argostranslate and ctranslate2 to use --translate") from exc
    package = Package(argos_package)
    translator = ctranslate2.Translator(
        str(package.package_path / "model"), device="cpu", inter_threads=1, intra_threads=8
    )
    items = {
        entry.installed_key: (entry.form, entry.gloss_de)
        for entry in entries if entry.gloss_de
    }
    # Clean hand-entered definitions override OCR text for their durable keys;
    # orphan component/variant rows are translated here too.
    for row in normalized_gold_rows():
        if row[3] and row[10]:
            items[row[10]] = (row[2], row[3])
    baseline = baseline_auto_by_key()
    catalog_glosses = catalog_source_glosses(baseline)
    for key in catalog_evidence_keys():
        row = baseline.get(legacy_catalog_key(key))
        if row and row[2]:
            gloss = catalog_glosses[key]
            items.setdefault(key, (row[2], gloss))
            items[f"catalog-preserved:{key}"] = (row[2], gloss)
    unique: dict[str, tuple[str, str]] = {}
    for _key, (_form, gloss) in items.items():
        if not gloss:
            continue
        digest = hashlib.sha256(gloss.encode()).hexdigest()
        pseudo = type("TranslationItem", (), {"gloss_de": gloss})()
        unique.setdefault(digest, (gloss, translation_source(pseudo)))
    previous = load_editorial()
    reusable = {
        row["Source_SHA256"]: postprocess_translation(row["English_Gloss"])
        for row in previous.values()
        if row.get("Source_SHA256") and row.get("English_Gloss")
    }
    digests = [digest for digest in unique if digest not in reusable]
    sources = [unique[digest][1] for digest in digests]
    translated = []
    batch_size = 256
    for start in range(0, len(sources), batch_size):
        batch = sources[start:start + batch_size]
        results = translator.translate_batch(
            [package.tokenizer.encode(text) for text in batch], replace_unknowns=True,
            max_batch_size=128, beam_size=1, num_hypotheses=1, length_penalty=0.2,
        )
        translated.extend(
            clean_translation(source, package.tokenizer.decode(result.hypotheses[0]).strip())
            for source, result in zip(batch, results)
        )
        print(f"Translated {min(start + batch_size, len(sources))}/{len(sources)}", file=sys.stderr)
    by_digest = {**reusable, **dict(zip(digests, translated))}
    records = []
    for key, (form, gloss) in items.items():
        if not gloss:
            continue
        digest = hashlib.sha256(gloss.encode()).hexdigest()
        english = TRANSLATION_OVERRIDES.get(gloss, by_digest[digest])
        records.append({
            "Entry_Key": key, "Form": form,
            "Raw_Gloss_German": gloss,
            "English_Gloss": english,
            "Review": "editorial-override" if gloss in TRANSLATION_OVERRIDES else "machine-translated-unreviewed",
            "Translator": "Argos Translate", "Model": "de_en 1.3",
            "Source_SHA256": digest,
        })
    write_dict_csv(EDITORIAL, records, EDITORIAL_FIELDS)


def apply_editorial(entries: Sequence[CleanEntry]) -> None:
    editorial = load_editorial()
    for entry in entries:
        record = editorial.get(entry.installed_key)
        if not record:
            entry.review.append("missing-translation")
            continue
        digest = hashlib.sha256(entry.gloss_de.encode()).hexdigest()
        if record["Source_SHA256"] != digest:
            entry.review.append("stale-translation")
            continue
        entry.english_gloss = TRANSLATION_OVERRIDES.get(
            entry.gloss_de, postprocess_translation(record["English_Gloss"])
        )
        if record["Review"]:
            entry.review.append(record["Review"])


def align_gold(entries: Sequence[CleanEntry]) -> list[list[str]]:
    by_key = {entry.installed_key: entry for entry in entries}
    editorial = load_editorial()
    output = []
    for index, original in enumerate(normalized_gold_rows(), 1):
        row = list(original)
        entry = by_key.get(row[10])
        record = editorial.get(row[10])
        source_hash = hashlib.sha256(row[3].encode()).hexdigest()
        translated_gold = (
            postprocess_translation(record["English_Gloss"])
            if record and record["Source_SHA256"] == source_hash else ""
        )
        if entry:
            entry.gold_row = index
            if translated_gold:
                row[3] = translated_gold
                entry.english_gloss = translated_gold
            elif entry.english_gloss:
                row[3] = entry.english_gloss
            else:
                # The small legacy file is the hand-reviewed layer: retain its
                # already-English definition when OCR cleanup changed the
                # corresponding German extraction after the translation pass.
                entry.english_gloss = row[3]
                entry.review = [reason for reason in entry.review if reason != "stale-translation"]
            row[7] = f"berger[p. {entry.unit.printed_page}]"
            row[9] = entry.etymology
            row[14] = " ".join(dict.fromkeys(entry.tags))
        elif translated_gold:
            row[3] = translated_gold
        output.append(row)
    return output


def resolve_key(stable: str, entries: Sequence[CleanEntry]) -> str:
    if not stable:
        return ""
    for entry in entries:
        if entry.unit.stable_key == stable and not entry.variant_of_stable:
            return entry.installed_key
    return ""


def import_rows(entries: Sequence[CleanEntry]) -> list[list[str]]:
    rows = []
    for entry in entries:
        if (
            entry.gold_row or "suspicious-form" in entry.review
            or "missing-gloss" in entry.review or not entry.english_gloss
        ):
            continue
        ids = entry.parameter_ids or [""]
        for link_index, parameter in enumerate(ids, 1):
            key = entry.installed_key if link_index == 1 else f"{entry.installed_key}:cdial:{link_index}"
            rows.append([
                entry.language, parameter, entry.form, entry.english_gloss, "", "", "",
                f"berger-auto[p. {entry.unit.pdf_page} (printed p. {entry.unit.printed_page})]",
                "", entry.etymology, key,
                resolve_key(entry.variant_of_stable, entries), "",
                resolve_key(entry.derivation_parent_stable, entries),
                " ".join(dict.fromkeys(entry.tags)),
            ])
    return rows


def catalog_preservation_rows(rows: Sequence[Sequence[str]], gold: Sequence[Sequence[str]]) -> list[list[str]]:
    """Retain exact keys used by accepted graph overlays after OCR reparsing."""
    present = {row[10] for row in [*rows, *gold]}
    baseline = baseline_auto_by_key()
    catalog_glosses = catalog_source_glosses(baseline)
    editorial = load_editorial()
    by_digest = {
        record["Source_SHA256"]: postprocess_translation(record["English_Gloss"])
        for record in editorial.values() if record["Source_SHA256"] and record["English_Gloss"]
    }
    wanted = sorted(catalog_evidence_keys() - present)
    preserved = []
    available = present | {key for key in wanted if key in baseline}
    for key in wanted:
        old = baseline.get(legacy_catalog_key(key))
        if not old or not old[2]:
            continue
        german = catalog_glosses[key]
        digest = hashlib.sha256(german.encode()).hexdigest()
        english = by_digest.get(digest, "")
        if not english:
            continue
        variant = old[11] if old[11] in available else ""
        derivation = "|".join(
            parent for parent in old[13].split("|") if parent in available
        )
        tags = " ".join(dict.fromkeys([
            *old[14].split(), "graph-evidence", "uncertain",
        ]))
        preserved.append([
            old[0], "", old[2], english, "", "", "", old[7], "", old[9],
            key, variant, "", derivation, tags,
        ])
    return preserved


def audit_rows(entries: Sequence[CleanEntry], emitted: Sequence[Sequence[str]]) -> list[dict]:
    emitted_by_prefix: dict[str, list[str]] = defaultdict(list)
    for row in emitted:
        emitted_by_prefix[row[10].split(":cdial:", 1)[0]].append(row[10])
    output = []
    for entry in entries:
        status = "gold" if entry.gold_row else "installed"
        if "suspicious-form" in entry.review or "missing-gloss" in entry.review:
            status = "excluded"
        elif not entry.english_gloss:
            status = "excluded"
        payload = "|".join((entry.unit.stable_key, entry.unit.raw_ocr, entry.form, entry.gloss_de))
        output.append({
            "Snapshot_Date": SNAPSHOT_DATE, "Stable_Key": entry.unit.stable_key,
            "Installed_Key": entry.installed_key, "PDF_Page": entry.unit.pdf_page,
            "Printed_Page": entry.unit.printed_page, "Column": entry.unit.column,
            "Entry_Ordinal": entry.unit.entry_ordinal, "Top": entry.unit.top,
            "Raw_OCR": entry.unit.raw_ocr, "OCR_Confidence": f"{entry.unit.confidence:.2f}",
            "Raw_Form": legacy._clean_head_prefix(entry.unit.text), "Final_Form": entry.form,
            "Raw_Gloss_German": entry.gloss_de, "English_Gloss": entry.english_gloss,
            "Direct_Turner_IDs": "|".join(entry.parameter_ids), "Etymology": entry.etymology,
            "Tags": " ".join(entry.tags), "Status": status,
            "Review": ";".join(dict.fromkeys(entry.review)),
            "Emitted_Keys": "|".join(emitted_by_prefix.get(entry.installed_key, [])),
            "Record_SHA256": hashlib.sha256(payload.encode()).hexdigest(),
        })
    return output


def preservation_audit(rows: Sequence[Sequence[str]]) -> list[dict]:
    baseline = baseline_auto_by_key()
    output = []
    for row in rows:
        old = baseline[legacy_catalog_key(row[10])]
        pdf, printed = parse_source_page(old[7])
        payload = "|".join((row[10], old[2], old[3], row[3]))
        output.append({
            "Snapshot_Date": SNAPSHOT_DATE, "Stable_Key": f"legacy:{row[10]}",
            "Installed_Key": row[10], "PDF_Page": pdf, "Printed_Page": printed,
            "Column": "", "Entry_Ordinal": "", "Top": "", "Raw_OCR": old[3],
            "OCR_Confidence": "", "Raw_Form": old[2], "Final_Form": row[2],
            "Raw_Gloss_German": catalog_source_glosses(baseline)[row[10]], "English_Gloss": row[3],
            "Direct_Turner_IDs": "", "Etymology": row[9], "Tags": row[14],
            "Status": "installed-preserved",
            "Review": "curated-graph-evidence-preserved;legacy-ocr-unreviewed",
            "Emitted_Keys": row[10], "Record_SHA256": hashlib.sha256(payload.encode()).hexdigest(),
        })
    return output


def write_gzip_audit(rows: Sequence[dict]) -> None:
    with gzip.open(AUDIT_OUTPUT, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_sample(rows: Sequence[dict]) -> None:
    # Keep the checked source-image sample stable as nearby OCR variants are
    # repaired or added. Each key names one printed entry unit; use its first
    # installed row, since all dialect rows share the same source image.
    installed_rows = [
        row for row in rows
        if row["Status"] in {"installed", "installed-untranslated", "gold"}
    ]
    by_stable: dict[str, dict] = {}
    for row in installed_rows:
        by_stable.setdefault(row["Stable_Key"], row)
    missing = [key for key in SOURCE_IMAGE_SAMPLE_KEYS if key not in by_stable]
    if missing:
        raise ValueError(f"source-image sample keys are no longer installed: {missing}")
    installed = [by_stable[key] for key in SOURCE_IMAGE_SAMPLE_KEYS]
    fields = [
        "Stable_Key", "Installed_Key", "PDF_Page", "Printed_Page", "Column",
        "Top",
        "Raw_OCR", "Final_Form", "Raw_Gloss_German", "English_Gloss",
        "Source_Image_Check", "Material_Error", "Resolution",
    ]
    previous = {}
    if SAMPLE_OUTPUT.exists():
        with SAMPLE_OUTPUT.open(encoding="utf-8") as stream:
            previous = {
                (row["Stable_Key"], row["Installed_Key"]): row
                for row in csv.DictReader(stream)
            }
    records = []
    for row in installed:
        record = {key: row.get(key, "") for key in fields}
        old = previous.get((record["Stable_Key"], record["Installed_Key"]), {})
        for field in ("Source_Image_Check", "Material_Error", "Resolution"):
            record[field] = old.get(field, "")
        if resolution := SOURCE_IMAGE_QA.get(record["Stable_Key"]):
            record["Source_Image_Check"] = "checked"
            record["Material_Error"] = "no"
            record["Resolution"] = resolution
        records.append(record)
    write_dict_csv(SAMPLE_OUTPUT, records, fields)


def write_manifest(units: Sequence[RawUnit], entries: Sequence[CleanEntry], rows: Sequence[Sequence], audit: Sequence[dict]) -> None:
    manifest = {
        "source": "Hermann Berger, Die Burushaski-Sprache von Hunza und Nager, Teil III: Wörterbuch (1998)",
        "source_id": SOURCE_ID,
        "snapshot_date": SNAPSHOT_DATE,
        "pdf_sha256": PDF_SHA256,
        "coverage": "printed pp. 9--486, Burushaski--German dictionary",
        "included_pdf_pages": "7 right half; 8--50; 52--246; 247 left half",
        "graph_compatibility": (
            "Pinned legacy evidence keys used by reviewed Burushaski cognate sets are retained as "
            "a separately marked compatibility tranche; their locators may include the former "
            "proper-name boundary and are not counted as reparsed scan coverage."
        ),
        "excluded": {
            "pdf_page_7_left": "dictionary preface, printed p. 8",
            "pdf_page_51": "weaker duplicate scan of printed pp. 94--95",
            "pdf_page_247_right": "proper-name appendix",
            "pdf_pages_248_327": "proper names, German--Burushaski index, and back matter",
        },
        "ocr": {
            "engine": "Tesseract script/Latin",
            "render_dpi": 300,
            "segmentation": "line indentation within four printed columns",
            "cache": str(CACHE_DIR.relative_to(ROOT)),
        },
        "translation": {
            "engine": "Argos Translate",
            "model": "German--English 1.3",
            "model_url": ARGOS_MODEL_URL,
            "model_file": ARGOS_MODEL,
            "model_sha256": ARGOS_MODEL_SHA256,
            "policy": "dictionary-context wrapper; checked editorial CSV is authoritative; automated translations remain explicitly unreviewed",
        },
        "counts": {
            "raw_units": len(units), "parsed_entries_including_variants": len(entries),
            "audited_records": len(audit),
            "installed_auto_rows": len(rows),
            "preserved_graph_evidence_rows": sum(
                row["Status"] == "installed-preserved" for row in audit
            ),
            "gold_rows": len(normalized_gold_rows()),
            "excluded_entries": sum(row["Status"] == "excluded" for row in audit),
            "untranslated_entries": sum(row["Status"] == "installed-untranslated" for row in audit),
            "direct_turner_links": sum(bool(row[1]) for row in rows),
        },
        "files": {
            "auto": str(AUTO_OUTPUT.relative_to(ROOT)),
            "gold": str(GOLD_OUTPUT.relative_to(ROOT)),
            "editorial": str(EDITORIAL.relative_to(ROOT)),
            "identity_map": str(IDENTITY_MAP.relative_to(ROOT)),
            "audit": str(AUDIT_OUTPUT.relative_to(ROOT)),
            "sample": str(SAMPLE_OUTPUT.relative_to(ROOT)),
        },
    }
    MANIFEST_OUTPUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, help="source PDF, used only to fill missing OCR cache pages")
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--rebuild-identity-map", action="store_true")
    parser.add_argument("--translate", action="store_true")
    parser.add_argument("--argos-package", type=Path, help="installed/extracted Argos de_en 1.3 package directory")
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args(argv)
    if args.pdf:
        args.pdf = args.pdf.expanduser().resolve()
        if sha256_path(args.pdf) != PDF_SHA256:
            raise ValueError("Berger PDF SHA-256 does not match the pinned scan")
    pages = load_pages(args.cache_dir, args.pdf)
    units = reconstruct_units(pages)
    valid_ids = legacy.load_valid_ids(ROOT / "data/cdial/params.csv")
    entries = parse_entries(units, valid_ids)
    apply_identity_map(entries, rebuild=args.rebuild_identity_map)
    if args.translate:
        if not args.argos_package:
            parser.error("--translate requires --argos-package")
        generate_editorial(entries, args.argos_package)
    apply_editorial(entries)
    gold = align_gold(entries)
    rows = import_rows(entries)
    preserved = catalog_preservation_rows(rows, gold)
    rows.extend(preserved)
    audit = [*audit_rows(entries, rows), *preservation_audit(preserved)]
    write_gzip_audit(audit)
    write_sample(audit)
    write_manifest(units, entries, rows, audit)
    if args.install:
        write_csv(AUTO_OUTPUT, rows)
        write_csv(GOLD_OUTPUT, gold)
    print(json.dumps({
        "raw_units": len(units), "entries": len(entries), "auto_rows": len(rows),
        "gold_rows": len(gold),
        "excluded": sum(row["Status"] == "excluded" for row in audit),
        "installed_untranslated": sum(row["Status"] == "installed-untranslated" for row in audit),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
