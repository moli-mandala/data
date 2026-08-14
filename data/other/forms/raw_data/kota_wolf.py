"""Extract Richard K. Wolf's working Kota dictionary and its DEDR links.

The source is a born-digital, one-column PDF.  Its retroflex dots are separate
small glyphs rather than Unicode characters, so extraction repairs them from
their page coordinates before parsing entries.

Run from ``data/``::

    uv run python data/other/forms/raw_data/kota_wolf.py \
      tmp/pdfs/kota/kota-dictionary-2014.pdf
"""

from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber


SOURCE_ID = "wolf-kota"
LANGUAGE_ID = "Kota"
RICH_COLUMNS = 15
LAST_ALPHABETICAL_PAGE = 44

# A few source entries omit the otherwise consistent dash between head and
# definition.  Listing their head strings is safer than guessing a boundary in
# prose.  The keys are prefixes after retroflex-dot repair.
NO_DASH_HEADS = {
    2: ("anān/anōn, anōḷ, anōr, and",),
    3: ("arg gī- (gic-)",),
    10: ("eṛt pay",),
    13: ("im",),
    18: ("kāl at- (ac-)/ iṭ- (ic-)/ eṭ-", "kāṇ-"),
    20: ("kiṭkac",),
    26: ("munguṭ",),
    28: ("nert-/nerty-",),
    32: ("pat- (pac-)",),
    33: ("pergūcd kuṛm",),
    38: ("tēy- (tēc-)", "tic pac mog"),
    42: ("vatm",),
}

# These are genuine wrapped continuations which happen to return to the left
# entry margin in the source file.
LEFT_MARGIN_CONTINUATIONS = {(1, 407.5), (31, 557.0), (37, 598.5)}

ENTRY_DASH = re.compile(r"\s*(?:—|--|(?<=\s)–(?=\s))\s*")
DIRECT_DEDR = re.compile(r"\((?:DEDR\s*)?(\d+[a-z]?)\)", re.I)
APPROX_DEDR = re.compile(r"\(app\.\s*(\d+[a-z]?)\)", re.I)
NAMED_DEDR = re.compile(r"(?<![\w])DEDR\s*(\d+[a-z]?)", re.I)
BRACKET = re.compile(r"\[([^\]]+)\]")


@dataclass
class Line:
    page: int
    x0: float
    top: float
    text: str
    bold: bool = False


@dataclass
class Entry:
    page: int
    ordinal: int
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(part.strip() for part in self.lines if part.strip())

    @property
    def key(self) -> str:
        return f"wolf-kota:p{self.page}:e{self.ordinal}"


def _attach_underdots(line: dict, dots: list[dict]) -> str:
    """Attach the PDF's displaced small full stops to overlapping base glyphs."""
    chars = line.get("chars") or []
    targets: set[int] = set()
    removed: set[int] = set()
    # Italic appendix forms use a second encoding: a full-size dot is placed
    # inside the preceding glyph rather than on a separate tiny text line.
    for index, char in enumerate(chars):
        if index == 0 or char["text"] != ".":
            continue
        previous = chars[index - 1]
        if (
            1.0 <= char["top"] - previous["top"] <= 3.0
            and char["x0"] < previous["x1"]
            and previous["text"].strip()
        ):
            targets.add(index - 1)
            removed.add(index)
    for dot in dots:
        if not 3.0 <= dot["top"] - line["top"] <= 6.5:
            continue
        center = (dot["x0"] + dot["x1"]) / 2
        candidates = [
            (index, abs((char["x0"] + char["x1"]) / 2 - center))
            for index, char in enumerate(chars)
            if char["text"].strip()
            and char["x0"] - 1 <= center <= char["x1"] + 1
        ]
        if candidates:
            targets.add(min(candidates, key=lambda item: item[1])[0])

    # pdfplumber's line text contains inferred spaces which have no matching
    # char object.  Map the target by its ordinal among non-space glyphs.
    target_ordinals = {
        sum(bool(char["text"].strip()) for char in chars[: index + 1]) - 1
        for index in targets
    }
    removed_ordinals = {
        sum(bool(char["text"].strip()) for char in chars[: index + 1]) - 1
        for index in removed
    }
    ordinal = -1
    repaired: list[str] = []
    for char in line["text"]:
        if not char.isspace():
            ordinal += 1
        if ordinal in removed_ordinals and not char.isspace():
            continue
        if ordinal in target_ordinals and not char.isspace():
            char = unicodedata.normalize("NFC", char + "\N{COMBINING DOT BELOW}")
        repaired.append(char)
    return unicodedata.normalize("NFC", "".join(repaired))


def extract_lines(page, page_number: int) -> list[Line]:
    raw_lines = page.extract_text_lines(return_chars=True)
    dots = [
        char
        for line in raw_lines
        for char in line.get("chars", [])
        if char["text"] == "." and char.get("size", 99) < 10
    ]
    result = []
    for line in raw_lines:
        chars = line.get("chars") or []
        if chars and all(char.get("size", 99) < 10 for char in chars):
            continue
        result.append(
            Line(
                page_number,
                line["x0"],
                line["top"],
                _attach_underdots(line, dots),
                bool(chars) and all("Bold" in char.get("fontname", "") for char in chars),
            )
        )
    return result


def _no_dash_head(page: int, text: str) -> str:
    for head in NO_DASH_HEADS.get(page, ()):
        if text.startswith(head) and (len(text) == len(head) or text[len(head)].isspace()):
            return head
    return ""


def extract_entries(pdf_path: Path) -> list[Entry]:
    entries: list[Entry] = []
    current: Entry | None = None
    page_ordinals: dict[int, int] = {}

    def start(page: int, text: str) -> None:
        nonlocal current
        if current is not None:
            entries.append(current)
        page_ordinals[page] = page_ordinals.get(page, 0) + 1
        current = Entry(page, page_ordinals[page], [text])

    def finish() -> None:
        nonlocal current
        if current is not None:
            entries.append(current)
            current = None

    with pdfplumber.open(pdf_path) as pdf:
        if len(pdf.pages) < 50:
            raise ValueError(f"expected a 50-page Kota dictionary, got {len(pdf.pages)} pages")
        for page_number, page in enumerate(pdf.pages, 1):
            for line in extract_lines(page, page_number):
                if not 65 <= line.top <= 695:
                    continue
                if page_number == 1 and line.top < 340:
                    continue
                if page_number == 44 and line.top >= 490:
                    break
                if page_number <= LAST_ALPHABETICAL_PAGE:
                    if line.x0 >= 100:
                        if current is not None:
                            current.lines.append(line.text)
                        continue
                    if not 68 <= line.x0 <= 76:
                        continue
                    if (page_number, round(line.top, 1)) in LEFT_MARGIN_CONTINUATIONS:
                        if current is not None:
                            current.lines.append(line.text)
                        continue
                    start(page_number, line.text)
                    continue

                # The thematic appendices mix lexical entries with headings and
                # running examples. Retain only explicit head—definition lines.
                if 68 <= line.x0 <= 78 and ENTRY_DASH.search(line.text):
                    start(page_number, line.text)
                elif page_number == 49 and 68 <= line.x0 <= 78:
                    day = re.fullmatch(
                        r"(Monday|Tuesday|Wed|Thurs|Fri|Sat|Sun):\s*(.+)", line.text
                    )
                    if day:
                        form = day.group(2)
                        alternate = re.fullmatch(r"(.+?)\s+\(([^)]+)\)", form)
                        if alternate:
                            form = f"{alternate.group(1)}/{alternate.group(2)}"
                        start(page_number, f"{form} — {day.group(1)}")
                    else:
                        finish()
                elif line.x0 >= 100 and current is not None:
                    current.lines.append(line.text)
                elif line.x0 < 100:
                    finish()

    finish()
    return entries


def split_entry(entry: Entry) -> tuple[str, str]:
    text = entry.text
    match = ENTRY_DASH.search(text)
    if match:
        return text[: match.start()].strip(), text[match.end() :].strip()
    head = _no_dash_head(entry.page, text)
    if head:
        return head, text[len(head) :].strip()
    return "", text


def _dedr_links(
    text: str, valid: set[str] | dict[str, str]
) -> tuple[list[tuple[str, str]], list[str], bool]:
    found: list[tuple[str, str]] = []
    checked_unlinked = bool(re.search(r"\[\s*[-–](?:\s*\+[^\]]+)?\s*\]", text))

    for match in DIRECT_DEDR.finditer(text):
        found.append((match.group(1).lower(), "direct"))
    for match in APPROX_DEDR.finditer(text):
        found.append((match.group(1).lower(), "related"))
    for bracket in BRACKET.finditer(text):
        content = bracket.group(1).strip()
        if "Kota In" in content:
            continue
        if content.startswith(("-", "–")):
            # ``[- + 1957]`` still cites the second compound member.
            numbers = re.findall(r"(?<=\+)\s*(\d+[a-z]?)", content, re.I)
        elif re.match(r"(?:DEDR\s*)?\d", content, re.I):
            numbers = re.findall(r"\d+[a-z]?", content, re.I)
        else:
            numbers = []
        for number in numbers:
            found.append((number.lower(), "related"))
    for match in NAMED_DEDR.finditer(text):
        found.append((match.group(1).lower(), "related"))

    deduplicated: list[tuple[str, str]] = []
    for number, relation in found:
        candidate = (number, relation)
        if candidate not in deduplicated and not any(n == number for n, _ in deduplicated):
            deduplicated.append(candidate)
    canonical_ids = valid if isinstance(valid, dict) else {value: value for value in valid}
    resolved: list[tuple[str, str]] = []
    invalid: list[str] = []
    for number, relation in deduplicated:
        canonical = canonical_ids.get(number, "")
        if not canonical and re.fullmatch(r"\d+[a-z]", number):
            canonical = canonical_ids.get(number[:-1], "")
        # One entry prints the impossible five-digit ``44111``; resolving the
        # repeated final digit recovers DEDR 4411 without a form-based guess.
        if not canonical and len(number) >= 5 and number[-1] == number[-2]:
            canonical = canonical_ids.get(number[:-1], "")
        if canonical:
            item = (canonical, relation)
            if item not in resolved:
                resolved.append(item)
        else:
            invalid.append(number)
    return resolved, invalid, checked_unlinked


def read_dedr_ids(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as stream:
        identifiers = [row[0][1:] for row in csv.reader(stream) if row and row[0].startswith("d")]
    result = {identifier.lower(): identifier for identifier in identifiers}
    by_number: dict[str, list[str]] = {}
    for identifier in identifiers:
        match = re.match(r"(\d+)", identifier)
        if match:
            by_number.setdefault(match.group(1), []).append(identifier)
    for number, candidates in by_number.items():
        if len(candidates) == 1:
            result.setdefault(number, candidates[0])
    return result


def _head_forms(head: str) -> tuple[str, list[str], str]:
    annotation = ""
    # Remove only source/etymon annotations, retaining lexical parentheses such
    # as nōm(b). Conjugational stems are emitted below as variants.
    cleaned = re.sub(r"\s+\((?:DEDR\s*)?\d+[a-z]?\)\s*$", "", head, flags=re.I)
    cleaned = re.sub(r"\s+\(app\.\s*\d+[a-z]?\)\s*$", "", cleaned, flags=re.I)
    inflections: list[str] = []
    trailing = re.search(r"\s+\(([^()]*)\)\s*$", cleaned)
    if trailing:
        content = trailing.group(1).strip()
        if cleaned[: trailing.start()].rstrip().endswith("-") and all(
            token.strip(" []").endswith("-") for token in re.split(r"[/,]", content)
        ):
            inflections = [token.strip(" []") for token in re.split(r"[/,]", content)]
            annotation = f"Conjugational stem(s): {content}"
            cleaned = cleaned[: trailing.start()].strip()

    forms = [part.strip(" []") for part in re.split(r"\s*/\s*|\s*,\s*", cleaned)]
    forms = [form for form in forms if form and not form.lower().startswith("or with ")]
    if not forms:
        return "", [], annotation
    variants = list(dict.fromkeys(forms[1:] + inflections))
    variants = [form for form in variants if form != forms[0]]
    return forms[0], variants, annotation


def _clean_gloss(gloss: str) -> str:
    gloss = re.sub(r"^(?:\((?:DEDR\s*)?\d+[a-z]?\)|\(app\.\s*\d+[a-z]?\)|\[[^\]]+\])\s*", "", gloss, flags=re.I)
    return re.sub(r"\s+", " ", gloss).strip(" ;")


def _rich_row(
    parameter: str,
    form: str,
    gloss: str,
    source: str,
    *,
    notes: str = "",
    etymology: str = "",
    entry_key: str = "",
    variant_of_key: str = "",
    derivation_parent_keys: tuple[str, ...] = (),
    tags: tuple[str, ...] = (),
) -> list[str]:
    return [
        LANGUAGE_ID, parameter, form, gloss, "", "", notes, source, "",
        etymology, entry_key, variant_of_key, "", "|".join(derivation_parent_keys),
        " ".join(tags),
    ]


def build_rows(
    entries: list[Entry], valid_dedr: set[str] | dict[str, str]
) -> tuple[list[list[str]], list[dict]]:
    rows: list[list[str]] = []
    audit: list[dict] = []
    for entry in entries:
        printed_head, raw_gloss = split_entry(entry)
        if not printed_head:
            audit.append({
                "Status": "skipped", "Entry_Key": entry.key, "PDF_Page": entry.page,
                "Printed_Head": "", "Form": "", "Variants": "", "Gloss": "",
                "DEDR_IDs": "", "DEDR_Relations": "", "Checked_Unlinked": "",
                "Unresolved_DEDR_IDs": "", "Raw_Entry": entry.text,
            })
            continue
        form, variants, morphology = _head_forms(printed_head)
        if not form:
            continue
        links, invalid, checked_unlinked = _dedr_links(entry.text, valid_dedr)
        gloss = _clean_gloss(raw_gloss)
        source = f"{SOURCE_ID}[p. {entry.page}, entry {entry.ordinal}]"
        parameters = links or [("", "")]
        notes = ""
        if invalid:
            notes = "unresolved DEDR citation(s): " + ", ".join(f"DEDR {x}" for x in invalid)
        for index, (number, relation) in enumerate(parameters, 1):
            key = entry.key if index == 1 else f"{entry.key}:link:{index}"
            etymology_parts = [morphology] if morphology else []
            if number:
                etymology_parts.append(f"Wolf's {relation} reference to DEDR {number}")
            tags = ("inherited",) if number else ()
            rows.append(
                _rich_row(
                    f"d{number}" if number else "", form, gloss, source,
                    notes=notes, etymology="; ".join(etymology_parts),
                    entry_key=key, tags=tags,
                )
            )
        for index, variant in enumerate(variants, 1):
            variant_key = f"{entry.key}:variant:{index}"
            rows.append(
                _rich_row(
                    "", variant, gloss, source,
                    etymology=f"Variant or conjugational stem of {form}",
                    entry_key=variant_key, variant_of_key=entry.key,
                    derivation_parent_keys=(entry.key,), tags=("variant",),
                )
            )
        audit.append({
            "Status": "ingested", "Entry_Key": entry.key, "PDF_Page": entry.page,
            "Printed_Head": printed_head, "Form": form, "Variants": "|".join(variants),
            "Gloss": gloss, "DEDR_IDs": "|".join(number for number, _ in links),
            "DEDR_Relations": "|".join(relation for _, relation in links),
            "Checked_Unlinked": "yes" if checked_unlinked else "",
            "Unresolved_DEDR_IDs": "|".join(invalid), "Raw_Entry": entry.text,
        })
    return rows, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--dedr-params", type=Path, default=Path("data/dedr/params.csv"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/other/forms/20260813-wolf-kota.csv"),
    )
    parser.add_argument(
        "--audit", type=Path,
        default=Path("data/other/forms/raw_data/20260813-wolf-kota-audit.csv"),
    )
    args = parser.parse_args()
    entries = extract_entries(args.pdf)
    rows, audit = build_rows(entries, read_dedr_ids(args.dedr_params))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    with args.audit.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(audit)
    print(f"wrote {len(rows)} form rows from {len(entries)} entries")


if __name__ == "__main__":
    main()
