#!/usr/bin/env python3
"""Extract every lexical entry from Christopher Shackle's Guru Nanak glossary.

The scan has a usable, but poor, hidden OCR layer.  This extractor instead renders
the lexical pages and runs Tesseract's Latin-script model on the roman column.  On
the main glossary it also OCRs the Gurmukhi column independently and aligns the
native headword by vertical position.  Existing hand-entered rows are treated as
gold: they are aligned to the OCR entries, retained verbatim, and excluded from the
generated import delta.

Outputs (under --output-dir):

* shackle_entries.csv      every parsed printed entry, with provenance
* shackle_review.csv       entries needing human attention
* shackle_auto_import.csv  eight-column Jambu rows not already represented by gold
* shackle_report.md        extraction and gold-alignment statistics

The checked-in import CSV is only replaced when --install is passed.  OCR results
are cached by page, so interrupted runs resume cheaply.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import html
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_PDF = Path(
    "~/Documents/Linguistics/Indo-European/Indo-Aryan/NIA/Northwestern/"
    "A_GURU_NANAK_GLOSSARY_2ND_EDITION.pdf"
).expanduser()
MAIN_PAGES = range(36, 312)       # PDF pages; printed pages 1--276
SUPPLEMENT_PAGES = range(315, 351)  # printed pages 280--315
SCALE = 300 / 72

# Point-coordinate crops, converted after rendering.  The main glossary has a
# Gurmukhi column at x~=52 and roman text at x~=111.  The supplement has no native
# column and uses nearly the full text width.
MAIN_LATIN_CROP = (103, 103, 426, 590)
MAIN_NATIVE_CROP = (42, 103, 104, 590)
SUPPLEMENT_CROP = (48, 82, 426, 590)

POS_PATTERN = re.compile(
    r",\s*(?P<pos>"
    r"v\s*\.\s*[its]\s*\.?|v\s*\.\s*s\s*\.?|"
    r"adj\s*\.?|adv\s*\.?|m\s*\.?|f\s*\.?|"
    r"pr\s*\.?|pron\s*\.?|ppn\s*\.?|pp\s*\.?|ptc\s*\.?|"
    r"num\s*\.?|intj\s*\.?|conj\s*\.?|cj\s*\.?|mp\s*\.?|"
    r"pd\s*\.?|fp\s*\.?|fsd\s*\.?|vsf\s*\.?|emph\s*\.?|"
    r"poss\s+(?:pr|ppn)\s*\.?|prepn\s*\.?|conditional\s+suf\s*\.?|"
    r"fut\s+3s\s*\.?|inf\s*\.?|ger\s*\.?|part\s*\.?|"
    r"postp\s*\.?|interr\s*\.?)",
    re.IGNORECASE,
)
CROSSREF_PATTERN = re.compile(r"^(.{1,100}?)(?::|,)\s*(?:also\s+)?see\s+", re.I)
QUOTE_PATTERN = re.compile(r"[‘“']([^’”']{1,500})[’”']")
BRACKET_PATTERN = re.compile(r"[\[(]([^\])]{1,1000})[\])]", re.S)
ASCII_ID_PATTERN = re.compile(r"(?<![A-Za-z0-9])(\d{1,5}[a-z]?)(?![A-Za-z0-9])", re.I)
GURMUKHI_PATTERN = re.compile(r"[\u0A00-\u0A7F]")
PDFIUM_LOCK = threading.Lock()

# Verified directly against the scan. These supplement forms have no Gurmukhi
# column, so a tiny page-scoped correction is safer than guessing from OCR.
PRINTED_FORM_CORRECTIONS = {
    (318, "adhalu"): "āḍhalu",
    (334, "dhahaa"): "dhāhā",
    (349, "vaca"): "vāṛā",
}

POS_TAGS = {
    "m": ("m",), "f": ("f",), "n": ("n",),
    "vt": ("verb", "tr"), "vi": ("verb", "intr"), "vs": ("verb",),
    "adj": ("adj",), "adv": ("adv",), "pr": ("pron",), "pron": ("pron",),
    "ppn": ("postp",), "prepn": ("prep",), "pp": ("pp",),
    "ptc": ("part",), "part": ("part",), "cj": ("conj",), "conj": ("conj",),
    "num": ("num",), "intj": ("interj",), "ger": ("ger",), "inf": ("inf",),
    "mp": ("m", "pl"), "fp": ("f", "pl"), "pd": ("pl", "dir"),
    "fsd": ("f", "sg", "dir"), "vsf": ("verb", "sg", "f"),
    "poss pr": ("pron", "poss"), "poss ppn": ("postp", "poss"),
    "conditional suf": ("conditional", "suffix"),
    "conditionalsuf": ("conditional", "suffix"),
    "fut 3s": ("verb", "fut", "3sg"), "emph": ("emph",),
    "interr": ("interr",),
}


def tags_for_pos(pos: str) -> list[str]:
    tags: list[str] = []
    for part in re.split(r"\s*[,;]\s*", pos.lower().strip().rstrip(".?")):
        tags.extend(POS_TAGS.get(part, ()))
    return list(dict.fromkeys(tags))


def tagged_notes(pos: str, notes: str) -> str:
    tag_field = " ".join(tags_for_pos(pos))
    return "; ".join(part for part in (tag_field, notes) if part)


@dataclass
class OCRLine:
    text: str
    top: int
    bottom: int
    left: int
    confidence: float


@dataclass
class Entry:
    section: str
    pdf_page: int
    printed_page: int
    top: int
    raw_head: str
    form: str
    pos: str
    gloss: str
    native: str
    cdial_ids: list[str]
    link_method: str
    etymology: str
    raw_entry: str
    ocr_confidence: float
    confidence: float = 0.0
    review_reasons: list[str] = field(default_factory=list)
    gold_rows: list[int] = field(default_factory=list)


def point_crop(image, crop: tuple[int, int, int, int]):
    return image.crop(tuple(round(value * SCALE) for value in crop))


def tesseract_lines(image, language: str, psm: int = 6) -> list[OCRLine]:
    """Run Tesseract TSV on a PIL image and reconstruct its lines."""
    payload = io.BytesIO()
    image.save(payload, format="PNG")
    proc = subprocess.run(
        ["tesseract", "stdin", "stdout", "-l", language, "--psm", str(psm), "tsv"],
        input=payload.getvalue(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    # OCR text often contains literal quotation marks. TSV is not CSV-quoted, so
    # Python's default quote handling can otherwise consume hundreds of lines as
    # one malformed field.
    rows = csv.DictReader(
        io.StringIO(proc.stdout.decode("utf-8", errors="replace")),
        delimiter="\t",
        quoting=csv.QUOTE_NONE,
    )
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("level") == "5" and row.get("text", "").strip():
            groups[(row["block_num"], row["par_num"], row["line_num"])].append(row)

    result = []
    for words in groups.values():
        words.sort(key=lambda row: int(row["left"]))
        confidences = [float(row["conf"]) for row in words if float(row["conf"]) >= 0]
        result.append(
            OCRLine(
                text=" ".join(row["text"] for row in words),
                top=min(int(row["top"]) for row in words),
                bottom=max(int(row["top"]) + int(row["height"]) for row in words),
                left=min(int(row["left"]) for row in words),
                confidence=sum(confidences) / len(confidences) if confidences else 0.0,
            )
        )
    return sorted(result, key=lambda line: (line.top, line.left))


def ocr_page(job: tuple[str, int, str, bool]) -> tuple[int, dict]:
    pdf_path, pdf_page, cache_dir, include_native = job
    cache_path = Path(cache_dir) / f"page-{pdf_page:03d}.json"
    if cache_path.exists():
        return pdf_page, json.loads(cache_path.read_text())

    try:
        import pypdfium2 as pdfium
    except ImportError as exc:  # pragma: no cover - exercised by CLI environments
        raise RuntimeError(
            "pypdfium2 is required. Run with the Codex workspace Python runtime or "
            "install pypdfium2."
        ) from exc

    # PDFium's page loader is not reliably thread-safe on this scanned Acrobat
    # document. Rendering is quick relative to OCR, so only this small section is
    # serialised.
    with PDFIUM_LOCK:
        document = pdfium.PdfDocument(pdf_path)
        page = document[pdf_page - 1]
        image = page.render(scale=SCALE).to_pil()
        page.close()
        document.close()
    section = "main" if pdf_page in MAIN_PAGES else "later_gurus"
    latin_crop = MAIN_LATIN_CROP if section == "main" else SUPPLEMENT_CROP
    latin = tesseract_lines(point_crop(image, latin_crop), "script/Latin")
    native: list[OCRLine] = []
    if include_native and section == "main":
        native = tesseract_lines(point_crop(image, MAIN_NATIVE_CROP), "pan")

    data = {
        "section": section,
        "pdf_page": pdf_page,
        "latin": [asdict(line) for line in latin],
        "native": [asdict(line) for line in native],
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False))
    temporary.replace(cache_path)
    return pdf_page, data


def canonical_ocr(text: str) -> str:
    replacements = {
        "ă": "ā", "ã": "ā", "â": "ā", "ä": "ā",
        "ĕ": "e", "ě": "e", "ĩ": "ī", "î": "ī",
        "ŏ": "o", "õ": "o", "ŭ": "u", "ũ": "ū",
        "ż": "z", "§": "s", "¢": "c", "@": "a",
        # Systematic Latin OCR glyph confusions. Marked consonants are first
        # reduced to their base and then restored from the Gurmukhi skeleton.
        "ħ": "h", "ț": "t", "đ": "d", "ř": "r", "ņ": "n", "ń": "n",
        "ù": "n", "ł": "l", "ç": "c", "ĉ": "c", "ı": "i", "ø": "o",
        "ġ": "g", "ğ": "g", "ĝ": "g", "š": "s",
        "{": "[", "}": "]", "j": "j",
    }
    return "".join(replacements.get(char, char) for char in unicodedata.normalize("NFC", text))


def normalize_key(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def normalize_words(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def normalize_pos(pos: str) -> str:
    compact = re.sub(r"\s+", "", pos.lower()).rstrip(".")
    mapping = {"v.t": "vt", "v.i": "vi", "v.s": "vs", "pron": "pr"}
    return mapping.get(compact, compact)


def start_of_entry(text: str) -> tuple[str, str] | None:
    text = canonical_ocr(text.strip()).lstrip("|¦!.,;:—- ")
    match = POS_PATTERN.search(text)
    if match and match.start() <= 105:
        head = text[: match.start()].strip()
        if 1 <= len(head) <= 100 and re.search(r"[A-Za-zÀ-ž]", head):
            return head, normalize_pos(match.group("pos"))
    match = CROSSREF_PATTERN.match(text)
    if match and re.search(r"[A-Za-zÀ-ž]", match.group(1)):
        return match.group(1).strip(), "xref"
    # A small but real class of dictionary entries has no printed POS label.
    match = re.match(r"^(.{1,100}?),\s*[‘“']", text)
    if match and re.search(r"[A-Za-zÀ-ž]", match.group(1)):
        return match.group(1).strip(), ""
    return None


def clean_form(raw_head: str) -> str:
    head = canonical_ocr(raw_head).strip()
    head = re.sub(r"^[†‡°*]+\s*", "", head)
    # Parentheses following whitespace contain inflectional shorthand, not the lemma.
    head = re.split(r"\s+\(", head, maxsplit=1)[0]
    head = re.split(r"\s*[\[(]\s*-", head, maxsplit=1)[0]
    head = re.sub(r"[¹²³⁴⁵⁶⁷⁸⁹⁰]+$", "", head)
    head = re.sub(r"(?<=[A-Za-zÀ-ž])\d+$", "", head)
    # Superscript homonym numbers are commonly read as ? or !.
    head = re.sub(r"[?!]+$", "", head)
    return head.strip(" ,.;:")


def nearest_native(entry_top: int, native_lines: Sequence[OCRLine]) -> str:
    candidates = [line for line in native_lines if GURMUKHI_PATTERN.search(line.text)]
    if not candidates:
        return ""
    line = min(candidates, key=lambda item: abs(item.top - entry_top))
    if abs(line.top - entry_top) > round(10 * SCALE):
        return ""
    return "".join(GURMUKHI_PATTERN.findall(line.text))


def extract_bracket(text: str) -> str:
    # The closing bracket is a frequent OCR casualty; the opening square bracket
    # is much more stable. Taking the final bracketed tail is also safer than a
    # generic parenthesis regex, which can mistake a parenthetical gloss for the
    # etymology and promote the frequency count to a CDIAL id.
    positions = [match.start() for match in re.finditer(r"\[", text)]
    if not positions:
        return ""
    start = positions[-1]
    # An opening brace before a parenthetical comparison is often OCRed as a
    # second square bracket: ``[1673 ... {x 386 ...]``. In that one case the
    # preceding bracket is the actual etymology start.
    if len(positions) > 1 and re.match(r"\[\s*(?:x|cf\.)\s*\d", text[start:], re.I):
        start = positions[-2]
    # In the supplement, printed ``[??]`` is repeatedly OCRed as ``[22]``.
    # If such an unknown-etymology bracket follows a real bracket in a merged
    # OCR entry, retain the earlier etymology instead (e.g. vāṛā on PDF p. 349).
    final_tail = text[start + 1 :].rstrip("]").strip()
    if len(positions) > 1 and re.fullmatch(r"22(?:\s*:\s*(?:e?f|cf)\..*)?", final_tail, re.I):
        start = positions[-2]
    return text[start + 1 :].rstrip("]").strip()


def restore_unknown_markers(etymology: str) -> str:
    """Restore supplement ``??`` sequences that Latin OCR reads as ``22``."""
    etymology = re.sub(r"(?<!\d)22(?=\s*$)", "??", etymology)
    etymology = re.sub(
        r"(?<!\d)22(?=\s*:\s*(?:e?f|cf)\.)", "??", etymology, flags=re.I
    )
    return re.sub(r"(?<=\?\?\s:\s)ef\.", "cf.", etymology, flags=re.I)


def normalize_etymon(text: str) -> str:
    text = canonical_ocr(html.unescape(text))
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]+", "", text)


def adjacent_etymon(etymology: str, end: int) -> str:
    """Return the IA form printed immediately after a candidate CDIAL number."""
    tail = etymology[end:]
    match = re.match(
        r"\s*(?:(?:\bor\s+Sk\.|[<>=~x×:+;,()?/_Ž$#-]|\b(?:cf|or)\b)\s*){0,5}"
        r"[*°]?\s*([A-Za-zÀ-žṃṇṛṭḍṅñḥüïḷśṣṁḕ]+(?:-[A-Za-zÀ-žṃṇṛṭḍṅñḥüïḷśṣṁḕ]+)*)",
        tail,
        re.I,
    )
    return match.group(1) if match else ""


def etymon_agrees(candidate: str, printed: str, cdial_etyma: dict[str, set[str]]) -> bool:
    observed = normalize_etymon(printed)
    expected = {normalize_etymon(value) for value in cdial_etyma.get(candidate, set())}
    expected.discard("")
    if not observed or not expected:
        return False
    if len(observed) <= 3:
        return observed in expected
    return max(difflib.SequenceMatcher(None, observed, value).ratio() for value in expected) >= 0.68


def resolve_cdial_candidate(
    candidate: str, printed: str, cdial_etyma: dict[str, set[str]]
) -> str | None:
    """Accept a printed number only when its adjacent IA form verifies that same id."""
    return candidate if etymon_agrees(candidate, printed, cdial_etyma) else None


def extract_ids(
    etymology: str,
    valid_ids: set[str],
    cdial_etyma: dict[str, set[str]] | None = None,
) -> list[str]:
    result = []
    # A leading 1 is often recognised as E/l/I in this typeface (e.g. 1661 ->
    # E661). Recover it only when the corrected value is a real CDIAL entry.
    for match in re.finditer(r"(?<![A-Za-z0-9])[EIl](\d{3,4}[a-z]?)(?![A-Za-z0-9])", etymology):
        candidate = "1" + match.group(1).lower()
        printed = adjacent_etymon(etymology, match.end())
        resolved = (
            candidate if cdial_etyma is None and candidate in valid_ids
            else resolve_cdial_candidate(candidate, printed, cdial_etyma or {})
        )
        if resolved in valid_ids and resolved not in result:
            result.append(resolved)
    for match in ASCII_ID_PATTERN.finditer(etymology):
        candidate = match.group(1).lower()
        # Verse locations such as ``Sr5.2`` can leak into an etymology when a
        # closing bracket is lost; the number after the dot is not a CDIAL id.
        if match.start() >= 2 and re.search(r"\d\.$", etymology[: match.start()]):
            continue
        # Printed question marks in uncertain etymologies are systematically
        # recognised as the valid but unrelated CDIAL id 22. All of these
        # contexts mean ``??``, not entry 22.
        if candidate == "22" and (
            re.fullmatch(r"\s*22(?:\s*:\s*(?:e?f|cf)\..*)?\s*", etymology, re.I)
            or re.search(r"(?:^|\s)22\s*$", etymology)
            or re.search(r"(?<![A-Za-z0-9])22\s*:\s*(?:e?f|cf)\.", etymology, re.I)
        ):
            continue
        printed = adjacent_etymon(etymology, match.end())
        resolved = (
            candidate if cdial_etyma is None and candidate in valid_ids
            else resolve_cdial_candidate(candidate, printed, cdial_etyma or {})
        )
        if resolved in valid_ids and resolved not in result:
            result.append(resolved)
    return result


def extract_gloss(text: str, head: str, pos: str) -> str:
    start = text.find(head) + len(head) if head in text else 0
    match = QUOTE_PATTERN.search(text, start)
    if match:
        gloss = match.group(1)
        leaked = re.search(r"[.?!]\s*(?:\d+[.'’”]?|M\d+[a-z.'’”]*|F[.'’”]*)\s*\[", gloss)
        if leaked:
            gloss = gloss[: leaked.start() + 1]
    else:
        opening = text.find("‘", start)
        if opening < 0:
            return ""
        tail = text[opening + 1 :]
        # Typical damaged closing quote: "(of the suny. 7." or "(of door]. M2."
        end = re.search(r"[.\]]\s*(?:\d+[.'’”]?|M\d+[a-z.'’”]*|F[.'’”]*)\s*(?:\[|$)", tail)
        if not end:
            return ""
        gloss = tail[: end.start() + (1 if tail[end.start()] == "]" else 0)]
    gloss = re.sub(r"\s+", " ", gloss).strip().rstrip("]")
    # A missing closing quote can make OCR consume most of an illustrative sentence.
    if len(gloss) > 240:
        return ""
    return gloss


def parse_pages(
    page_data: Sequence[dict],
    valid_ids: set[str],
    cdial_etyma: dict[str, set[str]] | None = None,
) -> list[Entry]:
    entries: list[Entry] = []
    current: dict | None = None

    def finish() -> None:
        nonlocal current
        if not current:
            return
        raw = re.sub(r"\s+", " ", " ".join(current["parts"])).strip()
        raw_head = current["raw_head"]
        etymology = restore_unknown_markers(extract_bracket(raw))
        raw_cdial_ids = extract_ids(etymology, valid_ids)
        verified_cdial_ids = extract_ids(etymology, valid_ids, cdial_etyma)
        native_lines = current["native_lines"]
        entry = Entry(
            section=current["section"],
            pdf_page=current["pdf_page"],
            printed_page=current["pdf_page"] - 35,
            top=current["top"],
            raw_head=raw_head,
            form=clean_form(raw_head),
            pos=current["pos"],
            gloss=extract_gloss(raw, raw_head, current["pos"]),
            native=nearest_native(current["top"], native_lines),
            cdial_ids=verified_cdial_ids,
            link_method=(
                "explicit" if verified_cdial_ids
                else "rejected" if raw_cdial_ids
                else "unlinked"
            ),
            etymology=etymology,
            raw_entry=raw,
            ocr_confidence=sum(current["confidences"]) / len(current["confidences"]),
        )
        entries.append(entry)
        current = None

    for data in sorted(page_data, key=lambda item: item["pdf_page"]):
        native_lines = [OCRLine(**item) for item in data.get("native", [])]
        for line_dict in data["latin"]:
            line = OCRLine(**line_dict)
            text = canonical_ocr(line.text.strip())
            if not text or re.fullmatch(r"[\d\W_]+", text):
                continue
            start = start_of_entry(text)
            if start:
                finish()
                current = {
                    "section": data["section"],
                    "pdf_page": data["pdf_page"],
                    "top": line.top,
                    "raw_head": start[0],
                    "pos": start[1],
                    "parts": [text],
                    "confidences": [line.confidence],
                    "native_lines": native_lines,
                }
            elif current:
                current["parts"].append(text)
                current["confidences"].append(line.confidence)
    finish()
    return entries


def resolve_crossrefs(entries: Sequence[Entry]) -> None:
    """Inherit CDIAL ids through Shackle's explicit ``=`` and ``see`` links."""
    by_head: dict[str, list[Entry]] = defaultdict(list)
    for entry in entries:
        for value in (entry.form, clean_form(entry.raw_head)):
            key = normalize_key(value)
            if key:
                by_head[key].append(entry)
    pattern = re.compile(
        r"(?:=|\bsee\b)\s*[†°t0]?\s*([A-ZĀĪŪṬḌṆṚṄÑḤÜ][A-ZĀĪŪṬḌṆṚṄÑḤÜ0-9-]{1,40})"
    )
    for _ in range(12):
        changed = 0
        for entry in entries:
            if entry.cdial_ids or not entry.etymology:
                continue
            inherited: list[str] = []
            for target in pattern.findall(entry.etymology):
                candidates = by_head.get(normalize_key(target), [])
                candidate_ids = {param for candidate in candidates for param in candidate.cdial_ids}
                if len(candidate_ids) == 1:
                    inherited.extend(candidate_ids)
            inherited = list(dict.fromkeys(inherited))
            if inherited:
                entry.cdial_ids = inherited
                entry.link_method = "crossref"
                changed += 1
        if not changed:
            break


def load_valid_cdial_ids(path: Path) -> set[str]:
    with path.open() as stream:
        return {row[0].lower() for row in csv.reader(stream) if row and row[0]}


def load_cdial_etyma(path: Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    with path.open() as stream:
        for row in csv.reader(stream):
            if len(row) < 4:
                continue
            result[row[0].lower()].update(part.strip() for part in row[1].split(","))
            # Secondary stems and morphological bases are marked bold or italic
            # in CDIAL and are also legitimate forms printed beside an entry id.
            result[row[0].lower()].update(
                html.unescape(value) for value in re.findall(r"<(?:b|i)>(.*?)</(?:b|i)>", row[3])
            )
    return result


def load_gold(path: Path) -> list[list[str]]:
    with path.open() as stream:
        rows = list(csv.reader(stream))
    if any(len(row) != 5 for row in rows):
        raise ValueError(f"Expected five columns in gold file: {path}")
    return rows


def similarity(left: str, right: str) -> float:
    return difflib.SequenceMatcher(None, normalize_words(left), normalize_words(right)).ratio()


def gold_score(row: Sequence[str], entry: Entry) -> float:
    gold_form, _pos, gloss, param, _notes = row
    form_score = max(
        (similarity(candidate.strip(), entry.form) for candidate in gold_form.split(",")),
        default=0.0,
    )
    gloss_score = similarity(gloss, entry.gloss) if gloss and entry.gloss else 0.0
    id_score = 1.0 if param and param.lower() in entry.cdial_ids else 0.0
    pos_score = 1.0 if normalize_pos(_pos) == entry.pos else 0.0
    return 5.0 * id_score + 3.0 * form_score + 2.0 * gloss_score + 0.5 * pos_score


def align_gold(gold: Sequence[Sequence[str]], entries: Sequence[Entry]) -> dict[int, int]:
    """Globally align selected gold rows to the complete printed entry stream.

    The manual file contains a few local reorderings, while OCR can destroy either
    the headword or the CDIAL number. A strict subsequence alignment therefore
    cascades after one miss. We instead index three independent anchors (headword,
    gloss, CDIAL id), then use source order only as a soft disambiguator.
    """
    matches: dict[int, int] = {}
    head_index: dict[str, set[int]] = defaultdict(set)
    gloss_index: dict[str, set[int]] = defaultdict(set)
    id_index: dict[str, set[int]] = defaultdict(set)

    # The hand-entered tranche ends at colī on PDF p.149. Infer its endpoint from
    # the final gold head instead of baking that page into the algorithm.
    final_forms = {normalize_key(part) for part in gold[-1][0].split(",") if part.strip()}
    endpoint_indices = [
        index for index, entry in enumerate(entries)
        if entry.section == "main" and normalize_key(entry.form) in final_forms
    ]
    endpoint = min(endpoint_indices) if endpoint_indices else 2700
    max_page = entries[endpoint].pdf_page + 6 if endpoint_indices else 170
    eligible = [
        index for index, entry in enumerate(entries)
        if entry.section == "main" and entry.pdf_page <= max_page
    ]
    for index in eligible:
        entry = entries[index]
        for value in (entry.form, clean_form(entry.raw_head)):
            if normalize_key(value):
                head_index[normalize_key(value)].add(index)
        if normalize_words(entry.gloss):
            gloss_index[normalize_words(entry.gloss)].add(index)
        for param in entry.cdial_ids:
            id_index[param].add(index)

    used: Counter[int] = Counter()
    for gold_index, row in enumerate(gold):
        expected = round(endpoint * gold_index / max(1, len(gold) - 1))
        gold_forms = {normalize_key(part) for part in row[0].split(",") if part.strip()}
        gold_gloss = normalize_words(row[2])
        pool: set[int] = set()
        for form in gold_forms:
            pool.update(head_index.get(form, set()))
        if gold_gloss:
            pool.update(gloss_index.get(gold_gloss, set()))
        if row[3]:
            pool.update(id_index.get(row[3], set()))
        had_exact_anchor = bool(pool)

        # Only the small residue with no exact anchor needs fuzzy local search.
        if not pool:
            pool.update(index for index in eligible if expected - 180 <= index <= expected + 180)

        ranked: list[tuple[float, int]] = []
        for index in pool:
            entry = entries[index]
            entry_heads = {normalize_key(entry.form), normalize_key(clean_form(entry.raw_head))}
            exact_head = bool(gold_forms & entry_heads)
            exact_gloss = bool(gold_gloss and gold_gloss == normalize_words(entry.gloss))
            exact_id = bool(row[3] and row[3] in entry.cdial_ids)
            score = gold_score(row, entry)
            score += 6.0 * exact_head + 5.0 * exact_gloss + 3.0 * exact_id
            # Gold is a selected subset, but its broad progression follows the
            # book. A gentle global-position prior resolves repeated pronouns and
            # common etyma without allowing one bad match to cascade.
            score -= 0.004 * abs(index - expected)
            if used[index] and not (exact_head and (exact_gloss or exact_id)):
                score -= 3.0 * used[index]
            ranked.append((score, index))

        score, best = max(ranked, default=(0.0, -1))
        threshold = 5.0 if had_exact_anchor else 4.0
        if best >= 0 and score >= threshold:
            matches[gold_index] = best
            entries[best].gold_rows.append(gold_index + 1)
            used[best] += 1
    return matches


def native_consonants(native: str) -> list[str]:
    mapping = {
        "ਕ": "k", "ਖ": "kh", "ਗ": "g", "ਘ": "gh", "ਙ": "ṅ",
        "ਚ": "c", "ਛ": "ch", "ਜ": "j", "ਝ": "jh", "ਞ": "ñ",
        "ਟ": "ṭ", "ਠ": "ṭh", "ਡ": "ḍ", "ਢ": "ḍh", "ਣ": "ṇ",
        "ਤ": "t", "ਥ": "th", "ਦ": "d", "ਧ": "dh", "ਨ": "n",
        "ਪ": "p", "ਫ": "ph", "ਬ": "b", "ਭ": "bh", "ਮ": "m",
        "ਯ": "y", "ਰ": "r", "ਲ": "l", "ਵ": "v", "ੜ": "ṛ",
        "ਸ਼": "ś", "ਸ": "s", "ਹ": "h", "ਖ਼": "kh", "ਗ਼": "g",
        "ਜ਼": "z", "ਫ਼": "f", "ਲ਼": "ḷ",
    }
    result = []
    for index, char in enumerate(native):
        if char in mapping:
            result.append(mapping[char])
        elif char in "ੰਂ":
            following = next((mapping[c] for c in native[index + 1 :] if c in mapping), "")
            if following[:1] in "kg":
                result.append("ṅ")
            elif following[:1] in "cj":
                result.append("ñ")
            elif following[:1] in "ṭḍ":
                result.append("ṇ")
            elif following[:1] in "td":
                result.append("n")
            elif following[:1] in "pb":
                result.append("m")
            else:
                result.append("ṃ")
    return result


def restore_retroflex(form: str, native: str) -> str:
    """Use the native consonant skeleton to restore robust retroflex marks."""
    if not native:
        return form
    native_units = native_consonants(native)
    if not native_units:
        return form
    independent_vowels = "ਅਆਇਈਉਊਏਐਓਔ"
    # A printed dagger before a headword is commonly OCRed as an attached "t".
    if form.startswith("t") and native[:1] in independent_vowels:
        form = form[1:]
    elif form.startswith("t") and len(native_units) >= 1:
        latin_after_t = [char for char in normalize_key(form[1:]) if char not in "aeiou"]
        if latin_after_t and latin_after_t[0] == normalize_key(native_units[0])[:1]:
            form = form[1:]
    native_base = [normalize_key(unit) for unit in native_units]
    # Treat aspirates as one consonant unit.  Comparing individual characters
    # made Gurmukhi ਠ (Shackle ``ṭh``) fail to align with OCR ``th``.
    consonants = set("bcdfgjklmnpqrstvwxyzṭḍṇṛṅñḷśṣ")
    positions: list[int] = []
    latin_base: list[str] = []
    index = 0
    while index < len(form):
        char = form[index].lower()
        if char in consonants:
            unit = char
            if index + 1 < len(form) and form[index + 1].lower() == "h" and char != "h":
                unit += "h"
                index += 1
            positions.append(index - len(unit) + 1)
            latin_base.append(normalize_key(unit))
        index += 1
    matcher = difflib.SequenceMatcher(None, latin_base, native_base)
    chars = list(form)
    marked = {
        "ṭ": "ṭ", "ḍ": "ḍ", "ṇ": "ṇ", "ṛ": "ṛ", "ḷ": "ḷ",
        "ṅ": "ṅ", "ñ": "ñ", "ś": "ś", "ṣ": "ṣ",
    }
    for a0, b0, size in matcher.get_matching_blocks():
        for offset in range(size):
            native_unit = native_units[b0 + offset]
            base = native_base[b0 + offset]
            base_letter = base[:1]
            desired = native_unit[:1]
            if desired in marked and normalize_key(desired) == base_letter:
                pos = positions[a0 + offset]
                chars[pos] = marked[desired]
                # Gurmukhi does not repeat a consonant merely to mark gemination.
                for neighbour in (pos - 1, pos + 1):
                    if 0 <= neighbour < len(chars) and normalize_key(chars[neighbour]) == base_letter:
                        chars[neighbour] = marked[desired]
    return "".join(chars)


def apply_corrections(entries: Sequence[Entry], gold: Sequence[Sequence[str]]) -> None:
    for entry in entries:
        if entry.gold_rows:
            rows = [gold[index - 1] for index in entry.gold_rows]
            forms = []
            ids = []
            for row in rows:
                if row[0] not in forms:
                    forms.append(row[0])
                if row[3] and row[3] not in ids:
                    ids.append(row[3])
            entry.form = ", ".join(forms)
            entry.pos = rows[0][1]
            entry.gloss = next((row[2] for row in rows if row[2]), entry.gloss)
            entry.cdial_ids = ids
            entry.link_method = "gold"
            continue
        entry.form = restore_retroflex(canonical_ocr(entry.form), entry.native)
        entry.form = PRINTED_FORM_CORRECTIONS.get(
            (entry.pdf_page, normalize_key(entry.raw_head)), entry.form
        )


def assess(entries: Sequence[Entry]) -> None:
    allowed = re.compile(r"^[A-Za-zāīūṃṇṛṭḍṅñḥüïḷśṣ' ,.-]+$")
    for entry in entries:
        reasons = []
        score = 0.42
        if entry.gold_rows:
            entry.review_reasons = []
            entry.confidence = 1.0
            continue
        if entry.ocr_confidence >= 80:
            score += 0.12
        elif entry.ocr_confidence < 55:
            reasons.append("low_ocr_confidence")
            score -= 0.12
        if entry.gloss:
            score += 0.15
            if "[" in entry.gloss or len(entry.gloss) > 220:
                reasons.append("suspicious_gloss")
                score -= 0.12
        elif entry.pos != "xref":
            reasons.append("missing_gloss")
            score -= 0.12
        if entry.pos:
            score += 0.08
        if entry.native:
            score += 0.06
        elif entry.section == "main":
            reasons.append("missing_native")
        if entry.cdial_ids:
            score += 0.08
        elif entry.link_method == "rejected":
            reasons.append("unverified_cdial_candidate")
            score -= 0.08
        if (
            not entry.form
            or not allowed.fullmatch(entry.form)
            or len(entry.form) > 90
            or len(entry.form.split()) > 9
            or any(char in entry.form for char in "[]{}")
        ):
            reasons.append("suspicious_headword")
            score -= 0.18
        if entry.form[:1] == "t" and entry.native.startswith(("ਉ", "ਅ", "ਇ", "ਏ", "ਓ")):
            reasons.append("possible_dagger_as_t")
            score -= 0.12
        entry.review_reasons = reasons
        entry.confidence = max(0.0, min(1.0, score))


def notes_for_entry(entry: Entry) -> str:
    notes = [f"Shackle PDF p. {entry.pdf_page} (printed p. {entry.printed_page})"]
    if entry.etymology:
        notes.append(f"etym. [{entry.etymology}]")
    if entry.review_reasons:
        notes.append("auto-review: " + ", ".join(entry.review_reasons))
    return "; ".join(notes)


def import_rows(entries: Sequence[Entry]) -> Iterable[list[str]]:
    for entry in entries:
        # Keep every recognised printed entry in shackle_entries.csv, but do not
        # put a continuation line that merely *resembles* a headword into CLDF.
        # These remain visible in shackle_review.csv for correction.
        if entry.gold_rows or "suspicious_headword" in entry.review_reasons:
            continue
        params = entry.cdial_ids or [""]
        for param in params:
            yield [
                "OP",
                param,
                entry.form,
                entry.gloss,
                entry.native,
                "",
                tagged_notes(entry.pos if entry.pos != "xref" else "", notes_for_entry(entry)),
                "shackle-auto",
            ]


def write_csv(path: Path, rows: Iterable[Sequence[str]], header: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        if header:
            writer.writerow(header)
        writer.writerows(rows)


def write_outputs(
    output_dir: Path,
    entries: Sequence[Entry],
    gold: Sequence[Sequence[str]],
    matches: dict[int, int],
) -> None:
    columns = [
        "section", "pdf_page", "printed_page", "top", "native", "raw_head", "form",
        "pos", "gloss", "cdial_ids", "link_method", "etymology", "ocr_confidence", "confidence",
        "review_reasons", "gold_rows", "raw_entry",
    ]
    entry_rows = []
    for entry in entries:
        row = asdict(entry)
        row["cdial_ids"] = ";".join(entry.cdial_ids)
        row["review_reasons"] = ";".join(entry.review_reasons)
        row["gold_rows"] = ";".join(map(str, entry.gold_rows))
        entry_rows.append([row[column] for column in columns])
    write_csv(output_dir / "shackle_entries.csv", entry_rows, columns)
    review_rows = [row for row, entry in zip(entry_rows, entries) if entry.confidence < 0.72]
    write_csv(output_dir / "shackle_review.csv", review_rows, columns)
    write_csv(output_dir / "shackle_auto_import.csv", import_rows(entries))

    matched_entries = {index for index in matches.values()}
    standard_ids = sum(bool(entry.cdial_ids) for entry in entries)
    link_counts = Counter(entry.link_method for entry in entries)
    review_only = sum("suspicious_headword" in entry.review_reasons for entry in entries)
    report = f"""# Shackle extraction report

- Parsed printed entries: {len(entries):,}
- Main glossary entries: {sum(entry.section == 'main' for entry in entries):,}
- Later-Guru supplement entries: {sum(entry.section == 'later_gurus' for entry in entries):,}
- Entries with a CDIAL link after safe resolution: {standard_ids:,}
  - Directly printed links: {link_counts['explicit']:,}
  - Safe ``=``/``see`` cross-reference links: {link_counts['crossref']:,}
  - Manual-gold links: {link_counts['gold']:,}
- Entries remaining unlinked: {sum(not entry.cdial_ids for entry in entries):,}
- Numeric candidates rejected for IA-etymon mismatch: {link_counts['rejected']:,}
- Gold rows aligned: {len(matches):,} / {len(gold):,}
- Distinct printed entries represented by gold: {len(matched_entries):,}
- New import rows: {sum(1 for _ in import_rows(entries)):,}
- Review queue: {sum(entry.confidence < 0.72 for entry in entries):,}
- OCR continuation candidates held out of CLDF: {review_only:,}

The import delta deliberately excludes printed entries aligned to the manual gold
file. Existing manual rows therefore remain authoritative, including cases where
one printed entry was split across multiple CDIAL etyma.
"""
    (output_dir / "shackle_report.md").write_text(report)


def parse_page_spec(spec: str | None) -> list[int]:
    if not spec:
        return list(MAIN_PAGES) + list(SUPPLEMENT_PAGES)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--pages", help="PDF pages, e.g. 36-40,315")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--cache-dir", type=Path, default=data_root / "tmp/pdfs/shackle-ocr")
    parser.add_argument("--output-dir", type=Path, default=here / "shackle_output")
    parser.add_argument("--no-native", action="store_true", help="skip Gurmukhi OCR")
    parser.add_argument(
        "--install",
        action="store_true",
        help="copy the generated delta into data/other/forms for CLDF inclusion",
    )
    args = parser.parse_args(argv)
    args.pdf = args.pdf.expanduser().resolve()
    if not args.pdf.exists():
        parser.error(f"PDF not found: {args.pdf}")
    if not shutil.which("tesseract"):
        parser.error("tesseract is not installed or not on PATH")

    cdial_params = data_root / "data/cdial/params.csv"
    valid_ids = load_valid_cdial_ids(cdial_params)
    cdial_etyma = load_cdial_etyma(cdial_params)
    gold_path = here / "old_punjabi"
    gold = load_gold(gold_path)
    pages = parse_page_spec(args.pages)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(str(args.pdf), page, str(args.cache_dir), not args.no_native) for page in pages]
    page_data: dict[int, dict] = {}
    # Tesseract does the CPU-heavy work in child processes, while PDFium renders in
    # native code. Threads avoid macOS sandbox semaphore restrictions without
    # serialising either workload.
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(ocr_page, job): job[1] for job in jobs}
        completed = 0
        for future in as_completed(futures):
            page, data = future.result()
            page_data[page] = data
            completed += 1
            if completed % 10 == 0 or completed == len(jobs):
                print(f"OCR pages: {completed}/{len(jobs)}", file=sys.stderr, flush=True)

    entries = parse_pages(list(page_data.values()), valid_ids, cdial_etyma)
    resolve_crossrefs(entries)
    matches = align_gold(gold, entries)
    apply_corrections(entries, gold)
    assess(entries)
    write_outputs(args.output_dir, entries, gold, matches)

    if args.install:
        destination = data_root / "data/other/forms/20260724-old-punjabi-shackle-auto.csv"
        shutil.copyfile(args.output_dir / "shackle_auto_import.csv", destination)
        print(f"Installed {destination}")
    print((args.output_dir / "shackle_report.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
