#!/usr/bin/env python3
"""Extract a review queue for the remainder of Bhawnani's Thari vocabulary.

The thesis is distributed both as a standalone vocabulary PDF and as the same
pages embedded in the annexures PDF.  Both copies contain an old ABBYY OCR
layer.  Its text is imperfect, but its word coordinates preserve the original
four-column layout (form, gloss, form, gloss) very well.  This importer uses
those coordinates, keeps the previously reviewed/CDIAL-linked CSV untouched,
and emits only entries not represented there.

Every OCR decision is written to a sidecar audit CSV.  The embedded ABBYY text
is suitable for locating rows but not for transcribing Thari: a held-out page
showed systematic base-letter errors and loss of contrastive diacritics.  The
script therefore refuses to install these forms as lexical data.  Stable
page/column/entry keys are retained so reviewed transcriptions can later be
installed without changing form identity.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pdfplumber

from ocr_corrections import OcrCorrection, load_corrections


HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parents[3]
DEFAULT_REVIEWED = DATA_ROOT / "data/other/forms/20220913-thari.csv"
DEFAULT_OUTPUT = DATA_ROOT / "data/other/forms/20260817-thari-remaining.csv"
DEFAULT_AUDIT = HERE / "20260817-thari-audit.csv"
DEFAULT_CORRECTIONS = HERE / "20260817-thari-corrections.csv"

# The standalone PDF starts at printed p. 186.  In the annexures copy, the
# vocabulary begins on PDF p. 26.  The two scans occasionally differ by one
# OCR line, so they are compared by printed page rather than PDF object number.
PRINTED_PAGE_FIRST = 186
ANNEX_VOCAB_FIRST = 26
# The hand-reviewed legacy file ends at gā̃ 'village', this exact source row.
# Earlier unreviewed words are deliberately outside this continuation ingest.
CONTINUATION_AFTER_KEY = "thari:p201:c1:e10"

FORM_BANDS = ((90.0, 200.0), (340.0, 450.0))
GLOSS_BANDS = ((200.0, 330.0), (450.0, 590.0))
TOP = 90.0
BOTTOM = 925.0
REFERENCE_WIDTH = 697.9
REFERENCE_HEIGHT = 993.1

POS_START = re.compile(
    r"(?:\(|\{)\s*(?:n\s*[.']?\s*(?:m|f)|n|a|adv|pp|pro|conj|interj|tr|intr)\b",
    re.IGNORECASE,
)


@dataclass
class Entry:
    pdf_page: int
    printed_page: int
    column: int
    column_entry: int
    top: float
    raw_form_pos: str
    raw_gloss: str
    form: str
    pos: str
    gloss: str
    duplicate_raw_form_pos: str = ""
    duplicate_raw_gloss: str = ""
    duplicate_agrees: bool = False
    reviewed_form: str = ""
    reviewed_param: str = ""
    review_match_score: float = 0.0

    @property
    def key(self) -> str:
        return f"thari:p{self.printed_page}:c{self.column}:e{self.column_entry}"


def compact(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "").replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip(" .|•♦")


def normalized_words(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).casefold()
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("£", "g").replace("©", "a").replace("&", "a")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def normalized_form_key(text: str) -> str:
    text = normalized_words(text).replace(" ", "")
    # These are recurrent ABBYY/typewriter confusions, not linguistic changes.
    return (
        text.replace("0", "o")
        .replace("1", "i")
        .replace("5", "s")
        .replace("vv", "w")
    )


def split_form_pos(raw: str) -> tuple[str, str]:
    raw = compact(raw)
    match = POS_START.search(raw)
    if match:
        form = raw[: match.start()]
        pos_raw = raw[match.start() :]
    else:
            # ABBYY often loses only the opening parenthesis: ``fekntn.f)``.
        match = re.search(
            r"(?i)(?<![a-z])(?:n\s*[.']?\s*[mf]|adv|pp|pro|conj|interj|intr|tr|a)\s*[)}`;]?$",
            raw,
        )
        if not match:
            match = re.search(r"(?i)(?:n\s*[.']?\s*[mf])\s*[)}`;]?$", raw)
        if match:
            form = raw[: match.start()]
            pos_raw = raw[match.start() :]
        else:
            form, pos_raw = raw, ""

    form = clean_form(form)
    return form, clean_pos(pos_raw)


def clean_form(text: str) -> str:
    text = compact(text)
    substitutions = str.maketrans(
        {
            "&": "a",
            "©": "a",
            "£": "g",
            "@": "a",
            "U": "u",
            "K": "k",
            "B": "b",
            "M": "m",
            "N": "n",
            "I": "i",
        }
    )
    text = text.translate(substitutions).casefold()
    text = text.replace("0", "o").replace("1", "i")
    text = re.sub(r"[|_^*~=+<>\[\]{}'’`\"!?;:,\\/]", "", text)
    text = re.sub(r"\s+", " ", text).strip(" .-•")
    return unicodedata.normalize("NFC", text)


def clean_pos(text: str) -> str:
    key = normalized_words(text)
    if "intr" in key:
        return "verb intr"
    if re.search(r"\btr\b", key):
        return "verb tr"
    if "adv" in key:
        return "adv"
    if "conj" in key:
        return "conj"
    if "interj" in key:
        return "interj"
    if "pro" in key:
        return "pronoun"
    if "pp" in key:
        return "postposition"
    if re.search(r"\bn\s*f\b", key):
        return "noun feminine"
    if re.search(r"\bn\s*m\b", key):
        return "noun masculine"
    if re.search(r"\bn\b", key):
        return "noun"
    if re.search(r"\ba\b", key):
        return "adjective"
    return ""


def tags_for_pos(pos: str) -> list[str]:
    """Map the source's printed part-of-speech labels to canonical tags."""
    return {
        "noun feminine": ["noun", "f"],
        "noun masculine": ["noun", "m"],
        "noun": ["noun"],
        "adjective": ["adj"],
        "adv": ["adv"],
        "pronoun": ["pron"],
        "postposition": ["postp"],
        "conj": ["conj"],
        "interj": ["interj"],
        "verb tr": ["verb", "tr"],
        "verb intr": ["verb", "intr"],
        "": [],
    }[pos]


def enrich_reviewed_tags(
    reviewed_path: Path, audit_path: Path, output_path: Path | None = None
) -> tuple[int, int]:
    """Restore POS tags for legacy rows already aligned in the OCR audit."""
    with audit_path.open(encoding="utf-8", newline="") as stream:
        audit_rows = [
            row for row in csv.DictReader(stream) if row["Status"] == "already_reviewed"
        ]
    evidence: dict[tuple[str, str], dict[str, str]] = {}
    for row in audit_rows:
        key = (row["Reviewed_Form"], row["Reviewed_Parameter_ID"])
        # One duplicated OCR row maps to khurpī. Prefer the more specific printed
        # label (noun feminine) and retain a single stable source location.
        if key not in evidence or len(row["POS"]) > len(evidence[key]["POS"]):
            evidence[key] = row

    with reviewed_path.open(encoding="utf-8", newline="") as stream:
        reviewed_rows = list(csv.reader(stream))
    tagged = 0
    aligned = 0
    for row in reviewed_rows:
        row.extend([""] * (15 - len(row)))
        match = evidence.get((row[2], row[1]))
        if not match:
            continue
        aligned += 1
        tags = tags_for_pos(match["POS"])
        if tags:
            tagged += 1
        row[7] = (
            f"thari[p. {match['Printed_Page']}, col. {match['Column']}, "
            f"entry {match['Column_Entry']}]"
        )
        row[10] = match["Entry_Key"]
        row[14] = " ".join(tags)

    target = output_path or reviewed_path
    with target.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(reviewed_rows)
    return aligned, tagged


def clean_gloss(text: str) -> str:
    text = compact(text)
    text = text.replace("£", "g").replace("^", "g")
    text = re.sub(r"\s+([,;.])", r"\1", text)
    text = re.sub(r"([,;])(?=[A-Za-z])", r"\1 ", text)
    return text


def _lines(page, band: tuple[float, float]) -> list[tuple[float, str]]:
    x_scale = page.width / REFERENCE_WIDTH
    y_scale = page.height / REFERENCE_HEIGHT
    crop = page.crop(
        (band[0] * x_scale, TOP * y_scale, band[1] * x_scale, BOTTOM * y_scale)
    )
    result = []
    for line in crop.extract_text_lines(strip=True, x_tolerance=2, y_tolerance=3):
        text = compact(line["text"])
        if text:
            # Normalize coordinates to the standalone PDF so duplicate scans
            # use the same row-alignment thresholds.
            result.append((float(line["top"]) / y_scale, text))
    return result


def extract_column(page, pdf_page: int, printed_page: int, column: int) -> list[Entry]:
    forms = _lines(page, FORM_BANDS[column - 1])
    glosses = _lines(page, GLOSS_BANDS[column - 1])
    used_glosses: set[int] = set()
    starts: list[tuple[float, str, int]] = []

    # A true entry start has form-column and gloss-column text on the same
    # baseline.  This is more reliable than recognizing the damaged POS label.
    for form_top, form_text in forms:
        candidates = [
            (abs(gloss_top - form_top), index)
            for index, (gloss_top, _) in enumerate(glosses)
            if index not in used_glosses and abs(gloss_top - form_top) <= 4.2
        ]
        if not candidates:
            continue
        _, gloss_index = min(candidates)
        if not re.search(r"[A-Za-z]", form_text) or not re.search(
            r"[A-Za-z]", glosses[gloss_index][1]
        ):
            continue
        used_glosses.add(gloss_index)
        starts.append((form_top, form_text, gloss_index))

    entries: list[Entry] = []
    for entry_index, (top, raw_form, gloss_index) in enumerate(starts, 1):
        next_top = starts[entry_index][0] if entry_index < len(starts) else BOTTOM
        parts = [glosses[gloss_index][1]]
        for index in range(gloss_index + 1, len(glosses)):
            gloss_top, gloss_text = glosses[index]
            if gloss_top >= next_top - 3.0:
                break
            # Page numbers, bullets, and scan debris do not belong to a gloss.
            if re.search(r"[A-Za-z]", gloss_text):
                parts.append(gloss_text)
        form, pos = split_form_pos(raw_form)
        gloss = clean_gloss(" ".join(parts))
        if len(form) < 1 or len(gloss) < 1:
            continue
        entries.append(
            Entry(
                pdf_page=pdf_page,
                printed_page=printed_page,
                column=column,
                column_entry=entry_index,
                top=top,
                raw_form_pos=raw_form,
                raw_gloss=" ".join(parts),
                form=form,
                pos=pos,
                gloss=gloss,
            )
        )
    return entries


def extract_pdf(path: Path) -> list[Entry]:
    entries: list[Entry] = []
    with pdfplumber.open(path) as document:
        for page_index, page in enumerate(document.pages):
            printed_page = PRINTED_PAGE_FIRST + page_index
            for column in (1, 2):
                entries.extend(
                    extract_column(page, page_index + 1, printed_page, column)
                )
    return entries


def extract_annex(path: Path, page_count: int) -> list[Entry]:
    entries: list[Entry] = []
    with pdfplumber.open(path) as document:
        first = ANNEX_VOCAB_FIRST - 1
        for offset, page in enumerate(document.pages[first : first + page_count]):
            printed_page = PRINTED_PAGE_FIRST + offset
            for column in (1, 2):
                entries.extend(
                    extract_column(page, first + offset + 1, printed_page, column)
                )
    return entries


def similarity(left: str, right: str) -> float:
    return difflib.SequenceMatcher(None, left, right).ratio()


def merge_duplicate_ocr(primary: list[Entry], duplicate: list[Entry]) -> None:
    by_page_col: dict[tuple[int, int], list[Entry]] = {}
    for entry in duplicate:
        by_page_col.setdefault((entry.printed_page, entry.column), []).append(entry)

    for entry in primary:
        candidates = by_page_col.get((entry.printed_page, entry.column), [])
        if not candidates:
            continue
        scored = [
            (
                0.75
                * similarity(normalized_words(entry.gloss), normalized_words(candidate.gloss))
                + 0.25 * similarity(normalized_form_key(entry.form), normalized_form_key(candidate.form)),
                candidate,
            )
            for candidate in candidates
        ]
        score, candidate = max(scored, key=lambda item: item[0])
        if score < 0.58:
            continue
        entry.duplicate_raw_form_pos = candidate.raw_form_pos
        entry.duplicate_raw_gloss = candidate.raw_gloss
        entry.duplicate_agrees = (
            normalized_form_key(entry.form) == normalized_form_key(candidate.form)
            and normalized_words(entry.gloss) == normalized_words(candidate.gloss)
        )

def read_reviewed(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


def match_reviewed(entries: list[Entry], rows: list[list[str]]) -> set[int]:
    """Return indexes of source entries already represented by reviewed rows."""
    matched: set[int] = set()
    # The old partial transcription ends in the first part of G.  Restricting
    # candidates to those pages prevents a common gloss such as ``one`` from
    # matching an unrelated later entry.
    candidates = [i for i, entry in enumerate(entries) if entry.printed_page <= 202]

    for row in rows:
        reviewed_form, reviewed_gloss = row[2], row[3]
        gloss_key = normalized_words(reviewed_gloss)
        form_key = normalized_form_key(reviewed_form)
        options = []
        for index in candidates:
            if index in matched:
                continue
            entry = entries[index]
            gloss_score = similarity(gloss_key, normalized_words(entry.gloss))
            hidden_form, _ = split_form_pos(entry.raw_form_pos)
            form_score = max(
                similarity(form_key, normalized_form_key(candidate_form))
                for candidate_form in (entry.form, hidden_form)
                if candidate_form
            )
            exact_bonus = 0.25 if gloss_key == normalized_words(entry.gloss) else 0.0
            score = 0.72 * gloss_score + 0.28 * form_score + exact_bonus
            options.append((score, gloss_score, form_score, index))
        if not options:
            continue
        score, gloss_score, form_score, index = max(options)
        accepted = (
            (gloss_score >= 0.82 and score >= 0.65)
            or (gloss_score >= 0.64 and score >= 0.73)
            or (form_score >= 0.72 and score >= 0.52)
        )
        if not accepted:
            continue
        matched.add(index)
        entries[index].reviewed_form = reviewed_form
        entries[index].reviewed_param = row[1]
        entries[index].review_match_score = score
    return matched


def write_import(
    path: Path,
    entries: Iterable[Entry],
    corrections: dict[str, OcrCorrection] | None = None,
) -> int:
    """Install only entries explicitly accepted in the correction overlay."""
    entries = list(entries)
    corrections = corrections or {}
    reviewed = [
        (entry, corrections[entry.key])
        for entry in entries
        if entry.key in corrections
        and corrections[entry.key].status in {"accepted", "corrected"}
    ]
    if not reviewed:
        raise RuntimeError(
            "Thari OCR forms are unreviewed and failed calibration; "
            "correct the audit rows before installing them"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        for entry, correction in reviewed:
            source = (
                f"thari[p. {entry.printed_page}, col. {entry.column}, "
                f"entry {entry.column_entry}]"
            )
            writer.writerow(
                [
                    "thar",
                    "",
                    correction.form,
                    correction.gloss,
                    "",
                    "",
                    "",
                    source,
                    "",
                    "",
                    entry.key,
                    "",
                    "",
                    "",
                    " ".join(tags_for_pos(correction.pos)),
                ]
            )
            count += 1
    return count


def write_audit(
    path: Path, entries: list[Entry], skipped: set[int], prior_section: set[int]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "Status",
        "Entry_Key",
        "PDF_Page",
        "Printed_Page",
        "Column",
        "Column_Entry",
        "Top",
        "Raw_Form_POS",
        "Form",
        "POS",
        "Raw_Gloss",
        "Gloss",
        "Duplicate_Raw_Form_POS",
        "Duplicate_Raw_Gloss",
        "Duplicate_Agrees",
        "Reviewed_Form",
        "Reviewed_Parameter_ID",
        "Review_Match_Score",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, entry in enumerate(entries):
            writer.writerow(
                {
                    "Status": (
                        "already_reviewed"
                        if index in skipped
                        else "prior_section"
                        if index in prior_section
                        else "needs_review"
                    ),
                    "Entry_Key": entry.key,
                    "PDF_Page": entry.pdf_page,
                    "Printed_Page": entry.printed_page,
                    "Column": entry.column,
                    "Column_Entry": entry.column_entry,
                    "Top": f"{entry.top:.1f}",
                    "Raw_Form_POS": entry.raw_form_pos,
                    "Form": entry.form,
                    "POS": entry.pos,
                    "Raw_Gloss": entry.raw_gloss,
                    "Gloss": entry.gloss,
                    "Duplicate_Raw_Form_POS": entry.duplicate_raw_form_pos,
                    "Duplicate_Raw_Gloss": entry.duplicate_raw_gloss,
                    "Duplicate_Agrees": "yes" if entry.duplicate_agrees else "no",
                    "Reviewed_Form": entry.reviewed_form,
                    "Reviewed_Parameter_ID": entry.reviewed_param,
                    "Review_Match_Score": (
                        f"{entry.review_match_score:.3f}" if entry.review_match_score else ""
                    ),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, type=Path, help="standalone vocabulary PDF")
    parser.add_argument("--annex", type=Path, help="annexures PDF containing the duplicate pages")
    parser.add_argument("--reviewed", type=Path, default=DEFAULT_REVIEWED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--corrections", type=Path, default=DEFAULT_CORRECTIONS)
    parser.add_argument(
        "--install",
        action="store_true",
        help="refused while remaining form transcriptions are unreviewed",
    )
    args = parser.parse_args()

    entries = extract_pdf(args.pdf)
    duplicate: list[Entry] = []
    if args.annex:
        duplicate = extract_annex(args.annex, len({entry.pdf_page for entry in entries}))
        merge_duplicate_ocr(entries, duplicate)
    reviewed_rows = read_reviewed(args.reviewed)
    skipped = match_reviewed(entries, reviewed_rows)
    try:
        anchor = next(index for index, entry in enumerate(entries) if entry.key == CONTINUATION_AFTER_KEY)
    except StopIteration as error:
        raise RuntimeError(f"continuation anchor missing: {CONTINUATION_AFTER_KEY}") from error
    prior_section = set(range(anchor + 1)) - skipped
    remaining = [
        entry
        for index, entry in enumerate(entries)
        if index > anchor and index not in skipped
    ]
    write_audit(args.audit, entries, skipped, prior_section)
    corrections = load_corrections(args.corrections, args.audit)
    if args.install:
        installed = write_import(args.output, remaining, corrections)
        print(f"installed reviewed corrections: {installed}")

    print(f"primary entries: {len(entries)}")
    print(f"duplicate entries: {len(duplicate)}")
    print(f"already reviewed: {len(skipped)} / {len(reviewed_rows)}")
    print(f"prior unreviewed entries outside continuation: {len(prior_section)}")
    print(f"remaining entries: {len(remaining)}")
    print(f"saved review decisions: {len(corrections)}")
    print(f"audit: {args.audit}")
    print("install status: blocked pending transcription review")


if __name__ == "__main__":
    main()
