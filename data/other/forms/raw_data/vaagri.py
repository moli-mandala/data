#!/usr/bin/env python3
"""Extract the numbered dictionary in Srinivasa Varma's *Vaagri Boli*.

The PDF has a usable image layer but its old OCR is badly displaced on warped
pages.  A fresh Tesseract pass recovers the 2,436 numbered slots (plus the
author's 1798a insertion) reliably.  The generated raw CSV retains the full OCR
entry for review; the forms CSV uses immutable dictionary item keys and keeps
the older, manually normalised CDIAL-linked rows as trusted overrides.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import re
import shutil
import subprocess
import tempfile
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path


FIRST_PDF_PAGE = 186
LAST_PDF_PAGE = 283
PRINTED_PAGE_OFFSET = 14
SOURCE = "srinivasa"
LANGUAGE = "VB"
LAST_ITEM = 2436

# These four item numbers are the only ones whose fresh OCR contains no digit.
# Their position in the otherwise complete sequence makes the corrections
# unambiguous.  The first entry's headword is also visually clear in the scan.
MALFORMED_PREFIXES = {"t.", "ES.", "Sk.", "Sl."}
FORM_OVERRIDES = {"1": "i"}
ENTRY_TEXT_OVERRIDES = {
    "15": "ine, pro. - him (acc.) (G. ine: he)",
    "74": "urtho:, part. - near",
    "2436": "yo:, pro. - he, she, it (G. o:)",
}


@dataclass
class Entry:
    item: str
    pdf_page: int
    printed_page: int
    form: str
    morphology: str
    gloss: str
    etymology: str
    raw_entry: str


def expected_items() -> list[str]:
    # Four visibly lettered insertions occur in addition to 1798a. The source
    # prints the second 1897 without a suffix; use 1897a internally so the
    # immutable source keys remain unique while preserving its position.
    insertions = {526: "526a", 1086: "1086a", 1417: "1417a", 1798: "1798a", 1897: "1897a"}
    items = []
    for number in range(1, LAST_ITEM + 1):
        items.append(str(number))
        if number in insertions:
            items.append(insertions[number])
    return items


def is_entry_start(line: str) -> bool:
    text = line.strip()
    first = text.split(maxsplit=1)[0] if text.split() else ""
    has_entry_punctuation = "," in text or bool(re.search(r"\s(?:—|–|~|-)\s", text))
    return first in MALFORMED_PREFIXES or (
        bool(re.search(r"\d", first)) and has_entry_punctuation
    )


def strip_item_prefix(line: str) -> str:
    parts = line.strip().split(maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"dictionary entry lacks a headword: {line!r}")
    return parts[1].strip()


def clean_continuation(line: str) -> str:
    text = " ".join(line.split())
    if not text or re.fullmatch(r"\d+", text):
        return ""
    if re.fullmatch(r"(?:\[[^]]{0,5}]|/[^/]{0,5}/)", text):
        return ""
    return text


def split_entry(item: str, text: str, pdf_page: int) -> Entry:
    text = " ".join(text.split())
    text = ENTRY_TEXT_OVERRIDES.get(item, text)
    if "," in text:
        form, rest = (part.strip() for part in text.split(",", 1))
    else:
        # A handful of source entries omit part-of-speech metadata entirely.
        bare_separator = re.search(r"\s(?:—|–|~|=|-)\s", text)
        if not bare_separator:
            raise ValueError(f"item {item} has no form/morphology boundary: {text!r}")
        form = text[: bare_separator.start()].strip()
        rest = text[bare_separator.start() :].strip()
    form = FORM_OVERRIDES.get(item, form)
    form = re.sub(r"\s*/\s*", "/", form).strip(" .;:")

    separator = re.search(
        r"[’']?(?:—|–|~|=|\+|«)|(?<![A-Za-z])-(?![A-Za-z])|(?<=[.,;])-(?=[A-Za-z])",
        rest,
    )
    if separator:
        morphology = rest[: separator.start()].strip(" ,.;")
        definition = rest[separator.end() :].strip()
    else:
        markers = list(re.finditer(
            r"\b(?:neut|mas|fem|nc|vc|intr|tr|adj|adv|part|pro|onom|sg|pl)\b[.,;']*",
            rest[:100],
            re.I,
        ))
        if markers:
            boundary = markers[-1].end()
            morphology = rest[:boundary].strip(" ,.;")
            definition = rest[boundary:].strip(" ,.;")
        else:
            morphology, definition = "", rest.strip(" ,.;")

    # Comparative/source matter begins with a conventional parenthetical.
    # Keep it verbatim in Etymology while leaving the English definition clean.
    etymology = ""
    marker = re.search(
        r"(?:^|\s)\((?:S\.|Pk\.|Pa\.|P\.|G(?:\.|,|\s)|M\.|H\.|N\.|A\.|B\.|C\.|L\.|"
        r"Ta\.|Te\.|Ko\.|Ka\.|DED\b|Eng\.|English\b|cf\.)",
        definition,
        re.I,
    )
    if marker:
        etymology = definition[marker.start() :].strip()
        definition = definition[: marker.start()].strip(" ,.;")

    return Entry(
        item=item,
        pdf_page=pdf_page,
        printed_page=pdf_page - PRINTED_PAGE_OFFSET,
        form=unicodedata.normalize("NFC", form),
        morphology=morphology,
        gloss=definition,
        etymology=etymology,
        raw_entry=text,
    )


def parse_ocr_pages(pages: dict[int, str]) -> list[Entry]:
    candidates: list[tuple[int, int, str]] = []
    page_lines: dict[int, list[str]] = {}
    for pdf_page in range(FIRST_PDF_PAGE, LAST_PDF_PAGE + 1):
        lines = pages[pdf_page].splitlines()
        page_lines[pdf_page] = lines
        candidates.extend(
            (pdf_page, index, line)
            for index, line in enumerate(lines)
            if is_entry_start(line)
        )

    items = expected_items()
    if len(candidates) != len(items):
        raise ValueError(
            f"expected {len(items)} dictionary entries, found {len(candidates)}; "
            "inspect the fresh OCR before ingesting"
        )

    entries = []
    for position, (item, (pdf_page, line_index, first_line)) in enumerate(
        zip(items, candidates)
    ):
        next_page, next_index = (
            candidates[position + 1][:2]
            if position + 1 < len(candidates)
            else (LAST_PDF_PAGE + 1, 0)
        )
        parts = [strip_item_prefix(first_line)]
        if next_page == pdf_page:
            continuation = page_lines[pdf_page][line_index + 1 : next_index]
        else:
            continuation = page_lines[pdf_page][line_index + 1 :]
        parts.extend(filter(None, (clean_continuation(line) for line in continuation)))
        entries.append(split_entry(item, " ".join(parts), pdf_page))
    return entries


def ocr_pdf(pdf: Path, pdftoppm: str, tesseract: str) -> dict[int, str]:
    pages: dict[int, str] = {}
    with tempfile.TemporaryDirectory(prefix="vaagri-ocr-") as directory:
        work = Path(directory)
        for page in range(FIRST_PDF_PAGE, LAST_PDF_PAGE + 1):
            prefix = work / f"page-{page}"
            subprocess.run(
                [
                    pdftoppm, "-f", str(page), "-l", str(page), "-png", "-r", "300",
                    "-singlefile", str(pdf), str(prefix),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            result = subprocess.run(
                [tesseract, str(prefix) + ".png", "stdout", "-l", "eng", "--psm", "6"],
                check=True,
                capture_output=True,
                text=True,
            )
            pages[page] = result.stdout
    return pages


def _plain(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return "".join(re.findall(r"[a-z0-9]+", text.replace(":", "")))


def read_manual_seed(path: Path) -> list[list[str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return [
            row for row in csv.reader(stream)
            if len(row) >= 8 and row[0] == LANGUAGE and "OCR-derived" not in row[6]
        ]


def match_manual(entries: list[Entry], manual: list[list[str]]) -> dict[str, list[list[str]]]:
    """Match the old curated subset to OCR entries without trusting OCR numbers alone."""
    matches: dict[str, list[list[str]]] = {}
    for row in manual:
        param, form, gloss = row[1], row[2], row[3]
        candidates = []
        for index, entry in enumerate(entries):
            has_reference = bool(
                param and re.search(rf"(?<!\d){re.escape(param)}(?!\d)", entry.raw_entry)
            )
            ref_bonus = 1.0 if has_reference else 0
            form_score = difflib.SequenceMatcher(None, _plain(form), _plain(entry.form)).ratio()
            gloss_score = difflib.SequenceMatcher(None, _plain(gloss), _plain(entry.gloss)).ratio()
            score = 0.55 * form_score + 0.45 * gloss_score + ref_bonus
            if has_reference or score >= 0.55:
                candidates.append((score, index))
        if not candidates:
            raise ValueError(f"could not reconcile trusted Vaagri seed row: {row!r}")
        _score, index = max(candidates)
        matches.setdefault(entries[index].item, []).append(row)
    return matches


def grammatical_tags(morphology: str) -> list[str]:
    value = morphology.lower()
    tags = []
    for pattern, tag in [
        (r"\bn\b", "noun"), (r"\bv\b", "verb"), (r"\badj\b", "adj"),
        (r"\badv\b", "adv"), (r"\bpro\b", "pron"), (r"\bpart\b", "part"),
        (r"\bconj\b", "conj"), (r"\bonom\b", "indecl"),
        (r"\bmas\b", "m"), (r"\bfem\b", "f"), (r"\bneut\b", "n"),
        (r"\bsg\b", "sg"), (r"\bpl\b", "pl"), (r"\bintr\b", "intr"),
        (r"\btr\b", "tr"), (r"(?:^|[,.; ])c\b", "caus"),
    ]:
        if re.search(pattern, value) and tag not in tags:
            tags.append(tag)
    return tags


def import_rows(entries: list[Entry], manual: list[list[str]]) -> list[list[str]]:
    trusted = match_manual(entries, manual)
    rows = []
    for entry in entries:
        citation = (
            f"{SOURCE}[p. {entry.pdf_page} (printed p. {entry.printed_page}), "
            f"item {entry.item}]"
        )
        seeds = trusted.get(entry.item) or [None]
        for seed_index, seed in enumerate(seeds, 1):
            if seed:
                param, form, gloss = seed[1], seed[2], seed[3]
                notes = seed[6]
            else:
                param, form, gloss = "", entry.form, entry.gloss
                notes = "auto-review: OCR-derived"
            # Existing curated rows predate source keys and already have durable
            # registry identities. Keep their legacy fingerprint path so adding
            # page/item provenance does not remint their public IDs.
            key = "" if seed else f"{SOURCE}:{entry.item}"
            rows.append([
                LANGUAGE, param, form, gloss, "", "", notes, citation, "", entry.etymology,
                key, "", "", "", " ".join(grammatical_tags(entry.morphology)),
            ])
    return rows


def write_raw(entries: list[Entry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(entries[0])), lineterminator="\n")
        writer.writeheader()
        writer.writerows(asdict(entry) for entry in entries)


def write_rows(rows: list[list[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--ocr-dir", type=Path, help="reuse page-N.txt files from a prior OCR pass")
    parser.add_argument("--raw-output", type=Path, default=here / "vaagri_dictionary.csv")
    parser.add_argument("--forms-output", type=Path, default=here.parent / "20220913-vaagri.csv")
    parser.add_argument("--manual-seed", type=Path)
    args = parser.parse_args()

    manual = read_manual_seed(args.manual_seed or args.forms_output)
    if args.ocr_dir:
        pages = {
            page: (args.ocr_dir / f"page-{page}.txt").read_text(encoding="utf-8")
            for page in range(FIRST_PDF_PAGE, LAST_PDF_PAGE + 1)
        }
    else:
        pdftoppm = shutil.which("pdftoppm")
        tesseract = shutil.which("tesseract")
        if not pdftoppm or not tesseract:
            raise SystemExit("pdftoppm and tesseract are required")
        pages = ocr_pdf(args.pdf, pdftoppm, tesseract)

    entries = parse_ocr_pages(pages)
    rows = import_rows(entries, manual)
    write_raw(entries, args.raw_output)
    write_rows(rows, args.forms_output)
    print(
        f"extracted {len(entries)} Vaagri dictionary records; "
        f"preserved {len(manual)} trusted CDIAL-linked rows"
    )


if __name__ == "__main__":
    main()
