#!/usr/bin/env python3
"""Extract Berger's complete Burushaski--German dictionary for Jambu.

The source is a four-column scanned spread.  Its hidden text is adequate for
search, but systematically loses Berger's diacritics, so this script renders the
Burushaski--German dictionary at 300 dpi and OCRs it with Tesseract's Latin-script
model. Every lexical entry is emitted; an explicit ``T <number>``
(Turner/CDIAL) reference supplies ``Parameter_ID``, while other entries retain a
blank parameter. Grammatical and dialect labels, free-text etymologies, derived
forms, and dialect/orthographic variants are represented in Jambu's rich manual
import columns.

Outputs under ``--output-dir``:

* ``berger_entries.csv``: every candidate and its raw OCR/provenance;
* ``berger_review.csv``: candidates needing human review;
* ``berger_auto_import.csv``: rich fifteen-column Jambu rows not already in the manual
  ``20220930-berger.csv`` gold tranche;
* ``berger_gold_enriched.csv`` and ``berger_gold_grammar_audit.csv``: the manual
  tranche with grammatical evidence recovered from its aligned source entries;
* ``berger_report.md``: coverage and quality statistics.

OCR is cached page-by-page.  ``--install`` copies only the import delta into the
forms directory; it never changes the hand-entered file.
"""

from __future__ import annotations

import argparse
import csv
import difflib
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


DEFAULT_PDF = Path(
    "~/Documents/Linguistics/Burushaski/Die Burushaski-Sprache von Hunza und Nager. "
    "Teil III. Wörterbuch. Burushaski — Deutsch, Deutsch — Burushaski (Hermann "
    "Berger) (z-lib.org).pdf"
).expanduser()
DICTIONARY_PAGES = range(8, 248)  # PDF spreads; printed pp. 10--486
SCALE = 300 / 72
PDFIUM_LOCK = threading.Lock()
TURNER_RE = re.compile(
    r"(?<![A-Za-z])T\s*[.:]?\s*((?=[0-9Il|]*\d)[0-9Il|]{1,7}[a-z]?)(?![A-Za-z])"
)

# Frequent character confusions in this typeface.  These are deliberately
# conservative: the manual rows remain the authority for specialist glyphs.
CHAR_FIXES = str.maketrans(
    {
        "ĉ": "ć", "Ĉ": "Ć", "ġ": "g", "ñ": "ṅ", "ď": "ḍ",
        "ț": "ṭ", "ş": "ś", "ŝ": "ś", "Ŝ": "Ś", "º": "o",
        "ﬁ": "fi", "ﬂ": "fl", "„": '"', "“": '"', "”": '"',
    }
)

GRAMMAR_BOUNDARY = re.compile(
    r"\s+(?=(?:hz\.?|ng\.?|hz\.ng\.?|NH\b|fem\.?|mask\.?|sg\.?|pl\.?|"
    r"[hxy]\b|-[a-zćśṭḍṅ]+\b))",
    re.I,
)
GERMAN_BOUNDARY = re.compile(
    r"\s+(?=(?:der|die|das|ein(?:e[rmns]?)?|sich|sehr|Art|Mann|Frau|"
    r"Person|klein|groß|leicht|schwach|stark|schlecht|gut|nicht|ohne|"
    r"mit|von|für|nach|bei|auf|in|ja|nein|wo|wie|was|Himmel|Hagel|"
    r"Dienstag|Sonntag|Schwierigkeit|Unverschämtheit)\b)",
    re.I,
)
SOURCE_PAREN = re.compile(r"\((?:ys\.|sh\.|kho\.|u\.|pe\.|vgl\.).*?\)", re.I | re.S)

# Source-verified repairs for OCR column/entry-boundary failures. These are keyed by the stable
# local entry IDs, leave ``raw_entry`` untouched in the audit, and change only text read directly
# from the cited scan. Printed p. 367 places ``brummen, dröhnen`` at the end of entry 7806 and
# begins the following Turner-linked entry with ``rúus, ng. rúuś``. Its cached OCR carried a
# pre-existing two-page printed-locator offset, corrected here for these reviewed records.
REVIEWED_ENTRY_REPAIRS = {
    "berger-entry-7806": {
        "gloss": "(Flugzeug) brummen, dröhnen",
        "printed_page": 367,
    },
    "berger-entry-7807": {
        "form": "rúus",
        "gloss": "Vergeltung, Rache, Heimzahlen",
        "printed_page": 367,
    },
    "berger-entry-7807-dialect-1": {
        "gloss": "Vergeltung, Rache, Heimzahlen",
        "printed_page": 367,
    },
}


@dataclass
class OCRParagraph:
    text: str
    left: int
    top: int
    confidence: float


@dataclass
class OCRLine:
    block: int
    text: str
    left: int
    top: int
    confidence: float


@dataclass
class Entry:
    language: str
    pdf_page: int
    printed_page: int
    form: str
    gloss: str
    cdial_id: str
    dialects: str
    etymology: str
    raw_entry: str
    ocr_confidence: float
    id_method: str
    confidence: float = 0.0
    review_reasons: list[str] = field(default_factory=list)
    gold_row: int = 0
    tags: list[str] = field(default_factory=list)
    entry_key: str = ""
    variant_of_key: str = ""
    derivation_parent_keys: str = ""


def canonical(text: str) -> str:
    text = unicodedata.normalize("NFC", text.translate(CHAR_FIXES))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_key(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def tesseract_layout(image) -> tuple[list[OCRParagraph], list[OCRLine]]:
    payload = io.BytesIO()
    image.save(payload, format="PNG")
    proc = subprocess.run(
        ["tesseract", "stdin", "stdout", "-l", "script/Latin", "--psm", "3", "tsv"],
        input=payload.getvalue(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )
    rows = csv.DictReader(
        io.StringIO(proc.stdout.decode("utf-8", errors="replace")),
        delimiter="\t", quoting=csv.QUOTE_NONE,
    )
    groups: dict[tuple[int, int, int], list[dict[str, str]]] = {}
    line_groups: dict[tuple[int, int, int, int], list[dict[str, str]]] = {}
    order: list[tuple[int, int, int]] = []
    for row in rows:
        if row.get("level") != "5" or not row.get("text", "").strip():
            continue
        key = (int(row["block_num"]), int(row["par_num"]), int(row["page_num"]))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(row)
        line_groups.setdefault(
            (int(row["block_num"]), int(row["par_num"]), int(row["line_num"]), int(row["page_num"])), []
        ).append(row)

    result = []
    for key in order:
        words = groups[key]
        # Tesseract already numbers blocks/paragraphs in reading order. Within a
        # paragraph, sort by its recognised line and horizontal position.
        words.sort(key=lambda row: (int(row["line_num"]), int(row["word_num"])))
        conf = [float(row["conf"]) for row in words if float(row["conf"]) >= 0]
        result.append(
            OCRParagraph(
                text=canonical(" ".join(row["text"] for row in words)),
                left=min(int(row["left"]) for row in words),
                top=min(int(row["top"]) for row in words),
                confidence=sum(conf) / len(conf) if conf else 0.0,
            )
        )
    lines = []
    for key, words in line_groups.items():
        words.sort(key=lambda row: int(row["word_num"]))
        conf = [float(row["conf"]) for row in words if float(row["conf"]) >= 0]
        lines.append(
            OCRLine(
                block=key[0], text=canonical(" ".join(row["text"] for row in words)),
                left=min(int(row["left"]) for row in words),
                top=min(int(row["top"]) for row in words),
                confidence=sum(conf) / len(conf) if conf else 0.0,
            )
        )
    lines.sort(key=lambda line: (line.block, line.top, line.left))
    return result, lines


def ocr_page(job: tuple[str, int, str]) -> tuple[int, dict]:
    pdf_path, page_number, cache_dir = job
    cache_path = Path(cache_dir) / f"page-{page_number:03d}.json"
    if cache_path.exists():
        return page_number, json.loads(cache_path.read_text(encoding="utf-8"))
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pypdfium2 is required for Berger OCR") from exc

    with PDFIUM_LOCK:
        document = pdfium.PdfDocument(pdf_path)
        page = document[page_number - 1]
        image = page.render(scale=SCALE).to_pil()
        page.close()
        document.close()
    paragraphs, lines = tesseract_layout(image)
    data = {
        "pdf_page": page_number,
        "width": image.width,
        "paragraphs": [asdict(paragraph) for paragraph in paragraphs],
        "lines": [asdict(line) for line in lines],
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    temporary.replace(cache_path)
    return page_number, data


def load_valid_ids(path: Path) -> set[str]:
    with path.open(encoding="utf-8") as stream:
        return {row[0].lower() for row in csv.reader(stream) if row and row[0]}


def repair_id(raw: str, valid: set[str]) -> tuple[str, str]:
    candidate = raw.translate(str.maketrans({"I": "1", "l": "1", "|": "1"})).lower()
    if candidate in valid:
        return candidate, "exact"
    variants = set()
    # A question mark or closing parenthesis is occasionally recognised as one
    # extra final digit (e.g. printed 11433? -> OCR 114337).
    for count in (1, 2):
        if len(candidate) > count and candidate[:-count] in valid:
            variants.add(candidate[:-count])
    for index in range(len(candidate)):
        shortened = candidate[:index] + candidate[index + 1 :]
        if shortened in valid:
            variants.add(shortened)
    if len(variants) == 1:
        return variants.pop(), "repaired"
    return "", "ambiguous" if variants else "invalid"


def extract_form(text: str) -> str:
    first = text.split(" (", 1)[0]
    first = re.sub(r"^[¹²³⁴⁵⁶⁷⁸⁹⁰\d]+\s*", "", first).lstrip("|¦!.,;:—- ")
    # Superscript homonym 1 is often read as a lowercase l attached to a
    # consonant-initial lemma (``lbirál`` -> ``birál``).
    first = re.sub(r"^l(?=[bcćčdfgjkmpqrsśštṭvwxyzž])", "", first, flags=re.I)
    boundary = GRAMMAR_BOUNDARY.search(first) or GERMAN_BOUNDARY.search(first)
    head = first[: boundary.start()] if boundary else first
    tokens = head.split()
    if len(tokens) > 1:
        allowed_tail = re.compile(
            r"^(?:-?[a-zćśṭḍṅ]+-|(?:man|et|sen|ju|th|ar|n|t)-?́?|\([^)]+\))$", re.I
        )
        kept = [tokens[0]]
        for token in tokens[1:]:
            if allowed_tail.fullmatch(token):
                kept.append(token)
            else:
                break
        head = " ".join(kept)
    return canonical(head).strip(" ,.;:")


def extract_gloss(text: str, form: str) -> str:
    body = text[len(form) :] if text.startswith(form) else text
    # The comparison/etymology parenthesis is provenance, not the German gloss.
    turner = TURNER_RE.search(body)
    if turner:
        opening = body.rfind("(", 0, turner.start())
        body = body[: opening if opening >= 0 else turner.start()]
    body = SOURCE_PAREN.sub("", body)
    body = re.sub(r"^(?:\s*(?:hz\.?|ng\.?|hz\.ng\.?|NH\b|[hxy]\b|sg\.?|pl\.?)\s*)+", "", body, flags=re.I)
    body = re.sub(r"\b(?:L|K)\s*\d+(?:[.;,:]\d+)*(?:\s*u\.a\.)?", "", body)
    body = re.sub(r"\b\d{1,2}\.\d{1,2}(?:[.;]\d{1,2})*\b", "", body)
    return canonical(body).strip(" ,.;:-")


def extract_dialects(text: str) -> str:
    found = []
    for label, pattern in (("NH", r"\bNH\b"), ("hz.", r"\bhz\."), ("ng.", r"\bng\.")):
        if re.search(pattern, text, re.I):
            found.append(label)
    return " ".join(found)


def yasin_variant(text: str) -> str:
    match = re.search(r"\bys\.\s*([^,;()]+)", text, re.I)
    if not match:
        return ""
    value = canonical(match.group(1))
    # Stop before prose/source markers accidentally swallowed by OCR.
    value = re.split(r"\s+(?:vgl\.|L\s*\d|T\s*\d|und\b|oder\b)", value, maxsplit=1, flags=re.I)[0]
    value = re.sub(r"\s+-[a-z]+$", "", value, flags=re.I)
    tokens = value.strip(" .").split()
    return tokens[0] if tokens else ""


GERMAN_LINE_STARTS = {
    "auch", "auf", "bei", "das", "davon", "der", "die", "ein", "eine", "einen",
    "einer", "es", "für", "in", "mit", "nach", "ng", "oder", "ohne", "sich", "und",
    "von", "vor", "wenn", "wie", "zu",
}


def candidate_chunks(data: dict) -> Iterable[tuple[str, int, int, float]]:
    """Yield one compact entry context per Turner citation.

    Tesseract sometimes merges several adjacent dictionary entries into a single
    paragraph. Lines are more stable: Turner citations occur in an indented source
    line, and the corresponding bold headword begins a nearby less-indented line.
    """
    paragraphs = [OCRParagraph(**row) for row in data.get("paragraphs", [])]
    for index, paragraph in enumerate(paragraphs):
        if not TURNER_RE.search(paragraph.text):
            continue
        text = paragraph.text
        left, top, confidence = paragraph.left, paragraph.top, paragraph.confidence
        # A source parenthesis is occasionally made a paragraph of its own. The
        # immediately preceding OCR paragraph is then the dictionary entry body.
        if re.match(r"^\((?:ys|sh|kho|vgl)\.", text, re.I) and index:
            previous = paragraphs[index - 1]
            # Avoid crossing the wide gutter between the two scanned book pages.
            if (previous.left < data["width"] / 2) == (left < data["width"] / 2):
                text = canonical(previous.text + " " + text)
                left, top = previous.left, previous.top
                confidence = (previous.confidence + confidence) / 2
        yield text, left, top, confidence


def parse_pages(page_data: Iterable[dict], valid_ids: set[str]) -> list[Entry]:
    entries = []
    for data in sorted(page_data, key=lambda item: item["pdf_page"]):
        for chunk, left, _top, ocr_confidence in candidate_chunks(data):
            matches = list(TURNER_RE.finditer(chunk))
            if not matches:
                continue
            form = extract_form(chunk)
            right_page = left >= data["width"] / 2
            printed_page = 2 * data["pdf_page"] - 6 + int(right_page)
            dialects = extract_dialects(chunk)
            for match in matches:
                cdial_id, method = repair_id(match.group(1), valid_ids)
                entry = Entry(
                    language="Bur", pdf_page=data["pdf_page"], printed_page=printed_page,
                    form=form, gloss=extract_gloss(chunk, form), cdial_id=cdial_id,
                    dialects=dialects, etymology=chunk[chunk.rfind("(", 0, match.start()) + 1 :].rstrip(")"),
                    raw_entry=chunk, ocr_confidence=ocr_confidence, id_method=method,
                )
                entries.append(entry)
                variant = yasin_variant(entry.etymology)
                if variant and normalize_key(variant) != normalize_key(form):
                    entries.append(
                        Entry(
                            language="Werch", pdf_page=entry.pdf_page, printed_page=entry.printed_page,
                            form=variant, gloss=entry.gloss, cdial_id=cdial_id, dialects="ys.",
                            etymology=entry.etymology, raw_entry=entry.raw_entry,
                            ocr_confidence=entry.ocr_confidence, id_method=method,
                        )
                    )
    return entries


HEADER_RE = re.compile(r"^(?:\d+\s+)?Burushaski\s*[-–—]\s*Deutsch(?:\s+\d+)?$", re.I)
SOURCE_ONLY_RE = re.compile(r"^\((?:ys|sh|kho|u|pe|tü|ti|e|skt|vgl)\.", re.I)
DERIVED_RE = re.compile(r"^(davon|dazu)\s+", re.I)
PROSE_STARTS = {
    "als", "auch", "auf", "bei", "bis", "da", "das", "dass", "den", "der", "die",
    "dies", "ein", "eine", "einem", "einen", "einer", "eines", "er", "es", "für",
    "hat", "im", "in", "ist", "man", "mit", "nach", "nicht", "oder", "ruf", "schwache",
    "sehr", "sie", "so", "und", "von", "vor", "wenn", "wie", "wird", "wo", "zu",
}
NON_FORMS = PROSE_STARTS | {
    "art", "bildung", "deutsch", "ds", "ernte", "faulheit", "felsen", "gewand",
    "himmel", "jahresertrag", "kopflos", "land", "person", "pferdeknecht", "rast",
    "sonntag", "schwach", "schwierigkeit", "vgl", "wh",
}
POS_PATTERNS = (
    ("adj", re.compile(r"\badj\.", re.I)),
    ("adv", re.compile(r"\badv\.", re.I)),
    ("pron", re.compile(r"\bPron(?:omen)?\.?", re.I)),
    ("num", re.compile(r"\b(?:Zahlwort|num\.)", re.I)),
    ("postp", re.compile(r"\bPostp\.", re.I)),
    ("prep", re.compile(r"\bPräp\.", re.I)),
    ("conj", re.compile(r"\b(?:Konj|conj)\.", re.I)),
    ("interj", re.compile(r"\b(?:Interj|Ausruf)\b", re.I)),
    ("part", re.compile(r"\bPartikel\b", re.I)),
)
ETYMOLOGY_MARKER = re.compile(
    r"(?:\b(?:ys|sh|kho|u|pe|tü|ti|west-ti|e|skt|pa|wa|balti|lad)\.\s|"
    r"\bT\s*\d|[<>]=?|\bvgl\.?(?:\s|$)|\bzu\s+(?:T\s*\d|sh\.|ys\.|kho\.|u\.)|"
    r"\b(?:Herkunft|Ursprung|Etymologie|entlehnt|Lehnwort|rückgebildet|verwandt|wörtl\.)\b|"
    r"\baus\s+(?:dem\s+)?(?:Shina|Khowar|Urdu|Hindi|Persischen|Tibetischen|Balti|Sanskrit)\b|\+)", re.I
)
FORM_CHARS = r"A-Za-zÀ-žāīūṭḍṅṇṛśṣćčc̣ɣʒʓʑʐʂʃʦʣáéíóúàèìòùâêîôûäëïöüåøæœḷḥṁṃñ"


def _clean_head_prefix(text: str) -> str:
    text = DERIVED_RE.sub("", text).strip()
    text = re.sub(r"^[¹²³⁴⁵⁶⁷⁸⁹⁰\d]+\s*", "", text).lstrip("|¦!.,;:—- ")
    text = re.sub(r"^l(?=[bcćčdfgjkmpqrsśštṭvwxyzž])", "", text, flags=re.I)
    boundaries = [
        re.search(r"\s+(?=(?:hz\.ng\.|hz\.|ng\.|NH\b|Mu\.|Kr\.|Gan\.|Alt\.|Hay\.|"
                  r"fem\.|mask\.|sg\.|pl\.|D\.pl\.|[hxy]\b|hm\b|adv\.|adj\.|Postp\.|"
                  r"Pron\.|Konj\.|Interj\.|s\.\s))", text, re.I),
        re.search(r"\s+(?=\d{1,2}(?:\.\d+)+\b|K\s*\d|L\s*\d)", text),
        re.search(r"\s+\(", text),
        GERMAN_BOUNDARY.search(text),
    ]
    positions = [match.start() for match in boundaries if match]
    prefix = text[: min(positions)] if positions else text
    # A comma after the first token normally begins inflectional or semantic
    # material. Explicit dialect variants are parsed separately from their labels.
    comma = prefix.find(",")
    if comma > 0:
        prefix = prefix[:comma]
    if len(prefix) > 100:
        prefix = " ".join(prefix.split()[:3])
    return canonical(prefix).strip(" ,.;:")


def _head_forms(prefix: str) -> list[str]:
    if not prefix:
        return []
    piece = canonical(prefix).strip(" ,.;:")
    if not piece:
        return []
    tokens = piece.split()
    kept = [tokens[0]]
    for token in tokens[1:]:
        if re.fullmatch(r"(?:-?[\wćśṭḍṅ]+-|(?:man|mán|et|ét|sen|ju|th|ar|n|t)-?́?|\([^)]+\))", token, re.I):
            kept.append(token)
        else:
            break
    return [" ".join(kept)]


def _looks_lexical(text: str) -> bool:
    if not text or HEADER_RE.match(text) or SOURCE_ONLY_RE.match(text):
        return False
    if text.startswith(("Eigennamen", "Abkürzungen:")):
        return False
    first = normalize_key(text.split(maxsplit=1)[0])
    if not first or first in NON_FORMS:
        return False
    forms = _head_forms(_clean_head_prefix(text))
    if not forms:
        return False
    form = forms[0]
    return bool(
        len(form) <= 80
        and re.search(rf"[{FORM_CHARS}]", form)
        and not re.search(r"[=<>\[\]{}\"]", form)
        and len(form.split()) <= 5
        and not (len(normalize_key(form)) <= 2 and not re.search(r"[aeiouáéíóúàèìòù]", form, re.I))
    )


def _split_embedded_entries(text: str) -> list[str]:
    """Split obvious entry boundaries which Tesseract merged into one paragraph."""
    # Berger explicitly introduces derivations with davon/dazu; they are lexical
    # rows and carry a derivational edge to the preceding head.
    text = re.sub(r"\s+(?=(?:davon|dazu)\s+)", "\n", text, flags=re.I)
    # A closed etymological parenthesis is commonly followed by the next bold
    # headword. Require a nearby dictionary label to avoid splitting examples.
    text = re.sub(
        rf"(?<=\))\s+(?=(?:[¹²³⁴⁵⁶⁷⁸⁹⁰\d]+)?[{FORM_CHARS}(][^\n]{{0,45}}?\s"
        r"(?:hz\.ng\.|hz\.|ng\.|NH\b|[hxy]\b|fem\.|s\.\s))",
        "\n",
        text,
        flags=re.I,
    )
    return [canonical(part) for part in text.splitlines() if canonical(part)]


def _grammar_tags(text: str, form: str, *, dialect: str = "") -> list[str]:
    probe = text[:350]
    tags = []
    for tag, pattern in POS_PATTERNS:
        if pattern.search(probe):
            tags.append(tag)
    verbish = bool(
        re.search(r"(?:^|\s)(?:[dgiue]-)?(?:[\wćśṭḍṅ]+-[/|])?[-\wćśṭḍṅ]*(?:mán|man|ét|et|sen|ju|th)-", probe, re.I)
        or re.search(r"(?:^|\s)-[cćstn]-", probe)
    )
    nounish = bool(re.search(r"\b(?:[hxy]\b|hm\b|sg\.|pl\.|D\.pl\.|-iṅ|-miċ|-muċ|-anċ)\b", probe, re.I))
    if verbish and "verb" not in tags:
        tags.append("verb")
    elif nounish and not any(tag in tags for tag in ("pron", "num", "postp", "verb")):
        tags.append("noun")
    if re.search(r"\bfem\.", probe, re.I):
        tags.append("f")
    if re.search(r"\bmask\.", probe, re.I):
        tags.append("m")
    if re.search(r"\bsg\.", probe, re.I):
        tags.append("sg")
    if re.search(r"\b(?:pl\.|D\.pl\.)", probe, re.I):
        tags.append("pl")
    if re.search(r"\b(?:trs|trans)\.", probe, re.I):
        tags.append("tr")
    if re.search(r"\b(?:intr|intrans)\.", probe, re.I):
        tags.append("intr")
    if DERIVED_RE.match(text):
        tags.append("derived")
    dialect_tags = []
    if dialect:
        dialect_tags.append(f"dialect:{dialect}")
    else:
        form_end = text.find(form) + len(form) if form and form in text else 0
        label_tail = text[form_end:].lstrip(" ,")[:25]
        # Gender/class and number labels can intervene between a headword and
        # its dialect label (e.g. ``abáś y hz.ng.``).
        label_tail = re.sub(
            r"^(?:(?:[hxy]|hm|sg\.|pl\.|D\.pl\.|fem\.|mask\.)\s+)*",
            "", label_tail, flags=re.I,
        )
    if not dialect and re.match(r"hz\.ng\.", label_tail, re.I):
        dialect_tags.extend(("dialect:Hunza", "dialect:Nager"))
    elif not dialect and re.match(r"ng\.", label_tail, re.I):
        dialect_tags.append("dialect:Nager")
    elif not dialect:
        dialect_tags.append("dialect:Hunza")
    if re.search(r"\bNH\b", probe):
        dialect_tags.append("dialect:NH")
    return list(dict.fromkeys([*tags, *dialect_tags]))


def _free_etymology(text: str) -> str:
    values = []
    for match in re.finditer(r"\(([^()]*(?:\([^()]*\)[^()]*)*)\)", text):
        value = canonical(match.group(1))
        if ETYMOLOGY_MARKER.search(value) and value not in values:
            values.append(value)
    # Retain Turner citations even when OCR lost the surrounding parenthesis.
    for match in TURNER_RE.finditer(text):
        value = f"T {match.group(1)}"
        if not any(value in existing for existing in values):
            values.append(value)
    return "; ".join(values)


def _gloss_for_full_entry(text: str, prefix: str) -> str:
    body = text[len(prefix):] if prefix and text.startswith(prefix) else text
    # Remove compact morphology/dialect labels before the definition.
    body = re.sub(
        r"^(?:\s*(?:hz\.ng\.|hz\.|ng\.|NH\b|Mu\.|Kr\.|Gan\.|Alt\.|Hay\.|"
        r"fem\.\s*[-\w]+|mask\.\s*[-\w]+|sg\.|pl\.|D\.pl\.|[hxy]\b|hm\b|"
        r"-[-\w]+|\d+(?:\.\d+)+|K\s*\d+|L\s*\d+|,|;))+",
        "", body, flags=re.I,
    )
    etym_start = next(
        (match.start() for match in re.finditer(r"\(", body) if ETYMOLOGY_MARKER.search(body[match.start():match.start()+200])),
        len(body),
    )
    body = body[:etym_start]
    body = re.sub(r"\b(?:K|L)\s*\d+(?:[.;,:]\d+)*(?:\s*u\.a\.)?", "", body)
    body = re.sub(r"\b\d{1,2}\.\d{1,2}(?:[.;]\d{1,2})*\b", "", body)
    return canonical(body).strip(" ,.;:-")


def _dialect_variants(text: str, main_form: str) -> list[tuple[str, str, str]]:
    variants = []
    # Yasin forms are a separate language in Jambu.
    for match in re.finditer(r"\bys\.\s*([^;()]+)", text, re.I):
        span = re.split(r"\s+(?:vgl\.|T\s*\d|sh\.|kho\.|u\.)", match.group(1), maxsplit=1, flags=re.I)[0]
        for form in re.split(r"\s*,\s*|\s+(?:oder|und)\s+", span):
            form = canonical(form).strip(" .,:;\"")
            form = form.split()[0] if len(form.split()) > 1 else form
            if form and normalize_key(form) not in NON_FORMS and normalize_key(form) != normalize_key(main_form):
                variants.append(("Werch", form, "Yasin"))
    # Explicit Nager/Hunza alternants occur directly after the label. Do not
    # manufacture a variant when the label is followed by prose/class marking.
    for label, dialect in (("ng", "Nager"), ("hz", "Hunza")):
        for match in re.finditer(rf"\b{label}\.\s+([^;()]+)", text[:220], re.I):
            token = canonical(match.group(1)).split(maxsplit=1)[0].strip(" .,:;")
            normalized = normalize_key(token)
            marked = bool(re.search(r"[áéíóúàèìòùāīūṭḍṅṇṛśṣćčc̣ɣ]", token))
            if (
                token and not token.startswith("-") and normalized not in NON_FORMS
                and re.fullmatch(rf"[{FORM_CHARS}()/'-]+", token)
                and normalized != normalize_key(main_form) and marked
            ):
                variants.append(("Bur", token, dialect))
    return list(dict.fromkeys(variants))


def parse_lexicon_pages(page_data: Iterable[dict], valid_ids: set[str]) -> list[Entry]:
    """Parse all lexical rows, not only entries carrying a Turner citation."""
    parsed: list[Entry] = []
    serial = 0
    parent_by_column: dict[tuple[int, int], Entry] = {}
    for data in sorted(page_data, key=lambda item: item["pdf_page"]):
        width = data["width"]
        for paragraph in (OCRParagraph(**row) for row in data["paragraphs"]):
            # PDF p.247 changes to the proper-name appendix in the right half.
            if data["pdf_page"] == 247 and paragraph.left >= width / 2:
                continue
            text = canonical(paragraph.text)
            if not text or HEADER_RE.match(text) or text in {"Burushaski - Deutsch", "fE; L 391"}:
                continue
            edges = tuple(ratio * width for ratio in (0.045, 0.24, 0.505, 0.69))
            column = min(range(4), key=lambda candidate: abs(paragraph.left - edges[candidate]))
            expected_left = edges[column]
            if SOURCE_ONLY_RE.match(text) and (parent := parent_by_column.get((data["pdf_page"], column))):
                close = text.find(")")
                source_text = text[: close + 1] if close >= 0 else text
                remainder = text[close + 1 :].strip() if close >= 0 else ""
                parent.raw_entry = canonical(parent.raw_entry + " " + source_text)
                parent.etymology = _free_etymology(parent.raw_entry)
                attached_ids = []
                for match in TURNER_RE.finditer(source_text):
                    cdial_id, method = repair_id(match.group(1), valid_ids)
                    if cdial_id and cdial_id not in attached_ids:
                        attached_ids.append(cdial_id)
                        if not parent.cdial_id:
                            parent.cdial_id = cdial_id
                            parent.id_method = method
                # Variants may already have been emitted from the head line;
                # keep them synchronized when the etymology is a separate
                # typographic paragraph.
                for entry in parsed:
                    if entry.variant_of_key == parent.entry_key:
                        entry.raw_entry = parent.raw_entry
                        entry.etymology = parent.etymology
                        if parent.cdial_id and not entry.cdial_id:
                            entry.cdial_id = parent.cdial_id
                            entry.id_method = parent.id_method
                existing_variants = {
                    (entry.language, normalize_key(entry.form), entry.variant_of_key)
                    for entry in parsed
                }
                for variant_index, (language, variant, dialect) in enumerate(_dialect_variants(source_text, parent.form), 1):
                    if (language, normalize_key(variant), parent.entry_key) in existing_variants:
                        continue
                    parsed.append(
                        Entry(
                            language=language, pdf_page=parent.pdf_page,
                            printed_page=parent.printed_page, form=variant,
                            gloss=parent.gloss, cdial_id=parent.cdial_id,
                            dialects=dialect, etymology=parent.etymology,
                            raw_entry=parent.raw_entry, ocr_confidence=parent.ocr_confidence,
                            id_method=parent.id_method,
                            tags=[*_grammar_tags(parent.raw_entry, variant, dialect=dialect), "alternate"],
                            entry_key=f"{parent.entry_key}-attached-{paragraph.top}-{variant_index}",
                            variant_of_key=parent.entry_key,
                        )
                    )
                if not remainder:
                    continue
                text = remainder
            # Indented OCR paragraphs are continuations/examples, not headwords.
            if abs(paragraph.left - expected_left) > 110:
                if parent := parent_by_column.get((data["pdf_page"], column)):
                    parent.raw_entry = canonical(parent.raw_entry + " " + text)
                    parent.etymology = _free_etymology(parent.raw_entry)
                continue
            for chunk in _split_embedded_entries(text):
                if not _looks_lexical(chunk):
                    continue
                prefix = _clean_head_prefix(chunk)
                forms = _head_forms(prefix)
                if not forms:
                    continue
                main_form = forms[0]
                serial += 1
                key = f"berger-entry-{serial}"
                cited = []
                id_methods = []
                for match in TURNER_RE.finditer(chunk):
                    cdial_id, method = repair_id(match.group(1), valid_ids)
                    if cdial_id and cdial_id not in cited:
                        cited.append(cdial_id)
                        id_methods.append(method)
                right_page = paragraph.left >= width / 2
                printed_page = 2 * data["pdf_page"] - 6 + int(right_page)
                derived = bool(DERIVED_RE.match(chunk))
                parent = parent_by_column.get((data["pdf_page"], column))
                base = Entry(
                    language="Bur", pdf_page=data["pdf_page"], printed_page=printed_page,
                    form=main_form, gloss=_gloss_for_full_entry(chunk, prefix),
                    cdial_id=cited[0] if cited else "", dialects=extract_dialects(chunk),
                    etymology=_free_etymology(chunk), raw_entry=chunk,
                    ocr_confidence=paragraph.confidence,
                    id_method=id_methods[0] if id_methods else "unlinked",
                    tags=_grammar_tags(chunk, main_form), entry_key=key,
                    derivation_parent_keys=(parent.entry_key if derived and parent else ""),
                )
                if not any(tag in base.tags for tag in ("noun", "verb", "adj", "adv", "pron", "num", "postp", "prep", "conj", "interj", "part")):
                    first_gloss = base.gloss.lstrip("('[] ")[:1]
                    if first_gloss and first_gloss.isupper():
                        base.tags.insert(0, "noun")
                parsed.append(base)
                if not derived:
                    parent_by_column[(data["pdf_page"], column)] = base

                # Alternate spellings printed in the head line.
                for variant_index, variant in enumerate(forms[1:], 1):
                    parsed.append(
                        Entry(
                            language="Bur", pdf_page=base.pdf_page, printed_page=base.printed_page,
                            form=variant, gloss=base.gloss, cdial_id=base.cdial_id,
                            dialects=base.dialects, etymology=base.etymology,
                            raw_entry=base.raw_entry, ocr_confidence=base.ocr_confidence,
                            id_method=base.id_method, tags=[*base.tags, "alternate"],
                            entry_key=f"{key}-variant-{variant_index}", variant_of_key=key,
                        )
                    )
                for variant_index, (language, variant, dialect) in enumerate(_dialect_variants(chunk, main_form), 1):
                    parsed.append(
                        Entry(
                            language=language, pdf_page=base.pdf_page, printed_page=base.printed_page,
                            form=variant, gloss=base.gloss, cdial_id=base.cdial_id,
                            dialects=dialect, etymology=base.etymology,
                            raw_entry=base.raw_entry, ocr_confidence=base.ocr_confidence,
                            id_method=base.id_method,
                            tags=[*_grammar_tags(chunk, variant, dialect=dialect), "alternate"],
                            entry_key=f"{key}-dialect-{variant_index}", variant_of_key=key,
                        )
                    )
                # Multiple secure CDIAL citations require scalar Parameter_ID rows.
                for extra_index, cdial_id in enumerate(cited[1:], 2):
                    parsed.append(
                        Entry(
                            language=base.language, pdf_page=base.pdf_page, printed_page=base.printed_page,
                            form=base.form, gloss=base.gloss, cdial_id=cdial_id,
                            dialects=base.dialects, etymology=base.etymology,
                            raw_entry=base.raw_entry, ocr_confidence=base.ocr_confidence,
                            id_method=id_methods[extra_index - 1], tags=base.tags,
                            entry_key=f"{key}-cdial-{extra_index}", variant_of_key=key,
                        )
                    )
    return parsed


def load_gold(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    if any(len(row) not in (8, 15) for row in rows):
        raise ValueError(f"Expected eight or fifteen columns in Berger gold file: {path}")
    return rows


def align_gold(entries: Sequence[Entry], gold: Sequence[Sequence[str]]) -> int:
    used: set[int] = set()
    matched = 0
    for gold_index, row in enumerate(gold, 1):
        language, cdial_id, form = row[:3]
        candidates = []
        for index, entry in enumerate(entries):
            if index in used or entry.language != language or entry.cdial_id != cdial_id:
                continue
            ratio = difflib.SequenceMatcher(None, normalize_key(form), normalize_key(entry.form)).ratio()
            if ratio >= 0.55:
                candidates.append((ratio, -abs(gold_index - index), index))
        if not candidates:
            continue
        _, _, best = max(candidates)
        entry = entries[best]
        entry.form = form
        entry.gloss = row[3] or entry.gloss
        entry.dialects = row[6]
        entry.gold_row = gold_index
        entry.id_method = "gold"
        used.add(best)
        matched += 1
    return matched


def enrich_gold_rows(
    entries: Sequence[Entry], gold: Sequence[Sequence[str]]
) -> tuple[list[list[str]], list[list[str]]]:
    """Recover structured tags for the hand-entered tranche from OCR evidence."""
    aligned = {entry.gold_row: entry for entry in entries if entry.gold_row}
    used = {entry.entry_key for entry in aligned.values()}
    enriched = []
    audit = []
    for gold_index, original in enumerate(gold, 1):
        row = list(original)
        row.extend([""] * (15 - len(row)))
        language, cdial_id, form, gloss = row[:4]
        evidence = aligned.get(gold_index)
        strategy = "aligned-source" if evidence else ""

        if not evidence:
            exact = [
                entry for entry in entries
                if entry.entry_key not in used
                and normalize_key(entry.form) == normalize_key(form)
                and entry.language == language
            ]
            if not exact and language == "Werch":
                exact = [
                    entry for entry in entries
                    if normalize_key(entry.form) == normalize_key(form)
                    and entry.language == "Bur"
                ]
            if exact:
                evidence = max(
                    exact,
                    key=lambda entry: (
                        entry.cdial_id == cdial_id,
                        difflib.SequenceMatcher(None, gloss, entry.gloss).ratio(),
                    ),
                )
                strategy = "exact-form-source"

        if not evidence:
            candidates = [
                entry for entry in entries
                if entry.language == language and entry.cdial_id == cdial_id
            ]
            if candidates:
                candidate = max(
                    candidates,
                    key=lambda entry: difflib.SequenceMatcher(
                        None, normalize_key(form), normalize_key(entry.form)
                    ).ratio(),
                )
                ratio = difflib.SequenceMatcher(
                    None, normalize_key(form), normalize_key(candidate.form)
                ).ratio()
                if ratio >= 0.78:
                    evidence = candidate
                    strategy = "fuzzy-form-source"

        if evidence:
            used.add(evidence.entry_key)
            tags = [tag for tag in evidence.tags if not tag.startswith("dialect:")]
            if language == "Werch":
                tags.extend(("dialect:Yasin", "alternate"))
            else:
                tags.extend(tag for tag in evidence.tags if tag.startswith("dialect:"))
            row[7] = f"berger[p. {evidence.printed_page}]"
            row[10] = evidence.entry_key
            evidence_form = evidence.form
            evidence_raw = evidence.raw_entry
        else:
            dialect = "Yasin" if language == "Werch" else ""
            probe = " ".join((form, row[6], gloss))
            tags = _grammar_tags(probe, form, dialect=dialect)
            if gloss[:1].isupper() and "noun" not in tags:
                tags.insert(0, "noun")
            if language == "Werch" and "alternate" not in tags:
                tags.append("alternate")
            strategy = "legacy-printed-evidence"
            evidence_form = ""
            evidence_raw = probe

        existing = row[14].split()
        row[14] = " ".join(dict.fromkeys([*existing, *tags]))
        enriched.append(row)
        audit.append([
            gold_index, form, strategy, evidence_form, evidence_raw, row[14]
        ])
    return enriched, audit


def apply_reviewed_repairs(entries: Sequence[Entry]) -> None:
    """Apply scan-verified, entry-keyed repairs while preserving raw OCR evidence."""
    by_key = {entry.entry_key: entry for entry in entries}
    missing = sorted(set(REVIEWED_ENTRY_REPAIRS) - set(by_key))
    if missing:
        raise ValueError(f"Reviewed Berger repair keys disappeared: {missing}")
    for entry_key, values in REVIEWED_ENTRY_REPAIRS.items():
        entry = by_key[entry_key]
        for field_name, value in values.items():
            setattr(entry, field_name, value)


def assess(entries: Sequence[Entry]) -> None:
    plausible_form = re.compile(r"^[()A-Za-zÀ-žāīūṭḍṅṇṛśṣćčc̣ɣʒʓʑʐʂʃʦʣ -]+[́]?$", re.UNICODE)
    page_initials: dict[int, Counter[str]] = {}
    for entry in entries:
        key = normalize_key(entry.form)
        if key:
            page_initials.setdefault(entry.pdf_page, Counter())[key[0]] += 1
    prose_heads = {
        "auch", "dazu", "davon", "deutsch", "erblatt", "felsblocken", "fur", "gelander",
        "gen", "gern", "leopard", "lich", "men", "mit", "nach", "naturlcher", "oder",
        "ren", "scheinend", "schlauen", "tig", "trocknete", "und", "usw", "werden",
        "weiter", "wickelt",
    }
    for entry in entries:
        if entry.gold_row:
            entry.confidence = 1.0
            continue
        reasons = []
        score = 0.55
        if entry.id_method == "exact":
            score += 0.2
        elif entry.id_method == "repaired":
            reasons.append("repaired_cdial_id")
            score += 0.08
        elif entry.id_method == "unlinked":
            score += 0.05
        else:
            reasons.append("invalid_cdial_id")
            score -= 0.35
        if entry.ocr_confidence >= 80:
            score += 0.1
        elif entry.ocr_confidence < 60:
            reasons.append("low_ocr_confidence")
            score -= 0.1
        source_as_head = bool(re.match(r"^\(?(?:ys|sh|kho|vgl|u|t)\.?\b", entry.form, re.I))
        normalized_form = normalize_key(entry.form)
        german_as_head = normalized_form in prose_heads
        if (
            not entry.form or len(entry.form) > 80 or not plausible_form.fullmatch(entry.form)
            or source_as_head or german_as_head or entry.form[:1].isupper()
        ):
            reasons.append("suspicious_headword")
            score -= 0.25
        initials = page_initials.get(entry.pdf_page, Counter())
        if normalized_form and initials:
            dominant_initial, dominant_count = initials.most_common(1)[0]
            own_count = initials[normalized_form[0]]
            if dominant_count >= 2 and normalized_form[0] != dominant_initial and own_count < 2:
                reasons.append("alphabetic_outlier")
                score -= 0.18
        if not entry.gloss:
            reasons.append("missing_gloss")
            score -= 0.12
        elif len(entry.gloss) > 500:
            reasons.append("suspicious_gloss")
            score -= 0.12
        if len(TURNER_RE.findall(entry.raw_entry)) > 1:
            reasons.append("multi_entry_paragraph")
            score -= 0.15
        entry.review_reasons = reasons
        entry.confidence = max(0.0, min(1.0, score))


def import_rows(entries: Sequence[Entry]) -> Iterable[list[str]]:
    eligible = [
        entry for entry in entries
        if not entry.gold_row and entry.form and not any(
            reason in entry.review_reasons for reason in ("suspicious_headword", "alphabetic_outlier")
        )
    ]
    selected = []
    signature_to_key = {}
    aliases = {}
    for entry in eligible:
        signature = (entry.language, entry.cdial_id, normalize_key(entry.form), entry.variant_of_key)
        if signature in signature_to_key:
            aliases[entry.entry_key] = signature_to_key[signature]
            continue
        signature_to_key[signature] = entry.entry_key
        selected.append(entry)
    available_keys = {entry.entry_key for entry in selected}

    def resolved(key: str) -> str:
        while key in aliases:
            key = aliases[key]
        return key if key in available_keys else ""

    for entry in selected:
        variant_of_key = resolved(entry.variant_of_key)
        derivation_parent_keys = " ".join(
            resolved(key) for key in entry.derivation_parent_keys.split() if resolved(key)
        )
        source = f"berger-auto[p. {entry.pdf_page} (printed p. {entry.printed_page})]"
        yield [
            entry.language, entry.cdial_id, entry.form, entry.gloss, "", "",
            "", source, "", entry.etymology, entry.entry_key,
            variant_of_key, "", derivation_parent_keys,
            " ".join(dict.fromkeys(entry.tags)),
        ]


def write_csv(path: Path, rows: Iterable[Sequence[object]], header: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        if header:
            writer.writerow(header)
        writer.writerows(rows)


def write_outputs(output_dir: Path, entries: Sequence[Entry], gold: Sequence[Sequence[str]], matched: int) -> None:
    columns = list(asdict(entries[0]).keys()) if entries else list(Entry.__dataclass_fields__)
    entry_rows = []
    for entry in entries:
        row = asdict(entry)
        row["review_reasons"] = ";".join(entry.review_reasons)
        entry_rows.append([row[column] for column in columns])
    write_csv(output_dir / "berger_entries.csv", entry_rows, columns)
    write_csv(
        output_dir / "berger_review.csv",
        (row for row, entry in zip(entry_rows, entries) if entry.review_reasons), columns,
    )
    auto_rows = list(import_rows(entries))
    write_csv(output_dir / "berger_auto_import.csv", auto_rows)
    enriched_gold, gold_audit = enrich_gold_rows(entries, gold)
    write_csv(output_dir / "berger_gold_enriched.csv", enriched_gold)
    write_csv(
        output_dir / "berger_gold_grammar_audit.csv",
        gold_audit,
        ("Gold_Row", "Form", "Strategy", "Evidence_Form", "Evidence_Raw", "Tags"),
    )
    methods = Counter(entry.id_method for entry in entries)
    report = f"""# Berger extraction report

- Parsed lexical rows (including variants): {len(entries):,}
- Burushaski rows: {sum(entry.language == 'Bur' for entry in entries):,}
- Werchikwar (``ys.``) variants: {sum(entry.language == 'Werch' for entry in entries):,}
- Exact CDIAL IDs: {methods['exact']:,}
- Safely repaired CDIAL IDs: {methods['repaired']:,}
- Rows without a CDIAL ID: {methods['unlinked']:,}
- Invalid/ambiguous CDIAL IDs: {methods['invalid'] + methods['ambiguous']:,}
- Manual rows aligned: {matched:,} / {len(gold):,}
- Installed/import rows: {len(auto_rows):,}
- Variant rows: {sum(bool(entry.variant_of_key) for entry in entries):,}
- Derived rows: {sum(bool(entry.derivation_parent_keys) for entry in entries):,}
- Rows carrying free-text etymology: {sum(bool(entry.etymology) for entry in entries):,}
- Rows carrying structured tags: {sum(bool(entry.tags) for entry in entries):,}
- Review queue: {sum(bool(entry.review_reasons) for entry in entries):,}

The import delta excludes rows aligned to ``20220930-berger.csv``. Under ``--install``
the hand-entered lexical content remains authoritative while its rich-schema metadata
and grammatical tags are refreshed from the row-level alignment audit.
"""
    (output_dir / "berger_report.md").write_text(report, encoding="utf-8")


def parse_page_spec(spec: str | None) -> list[int]:
    if not spec:
        return list(DICTIONARY_PAGES)
    pages: set[int] = set()
    for part in spec.split(","):
        if "-" in part:
            start, end = map(int, part.split("-", 1))
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part))
    return sorted(pages)


def main(argv: Sequence[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    data_root = here.parents[3]
    work_dir = data_root / ".cache/ocr/berger"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--pages", help="PDF pages, e.g. 7-12,40")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--cache-dir", type=Path, default=work_dir / "pages")
    parser.add_argument("--output-dir", type=Path, default=work_dir / "output")
    parser.add_argument("--install", action="store_true", help="install the generated import delta")
    args = parser.parse_args(argv)
    args.pdf = args.pdf.expanduser().resolve()
    if not args.pdf.exists():
        parser.error(f"PDF not found: {args.pdf}")
    if not shutil.which("tesseract"):
        parser.error("tesseract is not installed or not on PATH")

    valid_ids = load_valid_ids(data_root / "data/cdial/params.csv")
    gold = load_gold(data_root / "data/other/forms/20220930-berger.csv")
    pages = parse_page_spec(args.pages)
    jobs = [(str(args.pdf), page, str(args.cache_dir)) for page in pages]
    page_data = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(ocr_page, job): job[1] for job in jobs}
        for completed, future in enumerate(as_completed(futures), 1):
            page, data = future.result()
            page_data[page] = data
            if completed % 10 == 0 or completed == len(jobs):
                print(f"OCR pages: {completed}/{len(jobs)}", file=sys.stderr, flush=True)

    entries = parse_lexicon_pages(page_data.values(), valid_ids)
    matched = align_gold(entries, gold)
    apply_reviewed_repairs(entries)
    assess(entries)
    write_outputs(args.output_dir, entries, gold, matched)
    if args.install:
        destination = data_root / "data/other/forms/20260726-berger-auto.csv"
        shutil.copyfile(args.output_dir / "berger_auto_import.csv", destination)
        print(f"Installed {destination}")
        gold_destination = data_root / "data/other/forms/20220930-berger.csv"
        shutil.copyfile(args.output_dir / "berger_gold_enriched.csv", gold_destination)
        audit_destination = here / "20220930-berger-grammar-audit.csv"
        shutil.copyfile(args.output_dir / "berger_gold_grammar_audit.csv", audit_destination)
        print(f"Installed {gold_destination}")
        print(f"Installed {audit_destination}")
    print((args.output_dir / "berger_report.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
