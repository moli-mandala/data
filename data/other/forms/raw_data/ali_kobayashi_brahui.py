"""Extract Ali and Kobayashi's corpus-derived Brahui glossary.

The glossary is printed on pp. 687--733 of *Brahui Texts* (2024, minor
revision 2025).  Its PDF text layer distinguishes bold headwords from italic
parts of speech, so this importer reads the two columns directly without OCR.

Run from ``data/``::

    uv run --with pymupdf python \
      data/other/forms/raw_data/ali_kobayashi_brahui.py \
      ../tmp/pdfs/brahui/BrahuiTexts.pdf
"""

from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

SOURCE_ID = "ali-kobayashi2024"
LANGUAGE_ID = "Brahui"
FIRST_GLOSSARY_PDF_PAGE = 698
LAST_GLOSSARY_PDF_PAGE = 744
PDF_TO_PRINTED_OFFSET = 11
RICH_COLUMNS = 15
BODY_TOP = 78.0
BODY_BOTTOM = 655.0
COLUMN_SPLIT = 245.0

LOAN_LANGUAGES = {
    "A": "Arabic",
    "B": "Balochi",
    "E": "English",
    "F": "Farsi",
    "H": "Hindi-Urdu",
    "Pash.": "Pashto",
    "Si.": "Sindhi",
    "Sk.": "Sanskrit",
    "U": "Urdu",
}
LOAN_CODE = re.compile(r"\s*\[((?:A|B|E|F|H|U|Pash\.|Si\.|Sk\.)+)\]\s*$")
POS_TAGS = (
    ("int. adv.", ("interr", "adv")),
    ("int. pron.", ("interr", "pron")),
    ("int. a.", ("interr", "adj")),
    ("filler.", ("discourse-marker",)),
    ("postp.", ("postp",)),
    ("onom.", ("onomatopoeia",)),
    ("pron.", ("pron",)),
    ("conj.", ("conj",)),
    ("prop.", ("proper-noun",)),
    ("prep.", ("prep",)),
    ("phr.", ("multiword-expression",)),
    ("num.", ("num",)),
    ("adv.", ("adv",)),
    ("itj.", ("interj",)),
    ("vt.", ("verb", "tr")),
    ("vi.", ("verb", "intr")),
    ("v.", ("verb",)),
    ("n.", ("noun",)),
    ("a.", ("adj",)),
)


@dataclass
class Line:
    x0: float
    y0: float
    headword: str
    remainder: str
    text: str


@dataclass
class Entry:
    printed_page: int
    ordinal: int
    headword: str
    parts: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{SOURCE_ID}:p{self.printed_page}:e{self.ordinal}"

    @property
    def definition(self) -> str:
        return _join_parts(self.parts)


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace(";", ";")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def _join_parts(parts: list[str]) -> str:
    result = ""
    for part in parts:
        part = _clean(part)
        if not part:
            continue
        if result.endswith("-") and re.match(r"^[a-zāīūṛṭḍšžɣɬ]", part):
            if re.search(r"=[^\s]+-$", result):
                result += part
            else:
                result = result[:-1] + part
        else:
            result = f"{result} {part}".strip()
    return _clean(result)


def _line_from_spans(line: dict) -> Line | None:
    spans = [span for span in line["spans"] if span["text"]]
    if not spans:
        return None
    text = _clean("".join(span["text"] for span in spans))
    if not text:
        return None

    head = []
    remainder = []
    in_head = True
    previous_x1 = None
    for span in spans:
        value = span["text"]
        bold = "Bold" in span["font"]
        if in_head and bold:
            head.append(value)
        else:
            in_head = False
            if previous_x1 is not None and span["bbox"][0] - previous_x1 > 2.0:
                remainder.append(" ")
            remainder.append(value)
        previous_x1 = span["bbox"][2]

    x0, y0, _, _ = line["bbox"]
    return Line(x0, y0, _clean("".join(head)), _clean("".join(remainder)), text)


def page_columns(page: fitz.Page) -> tuple[list[Line], list[Line]]:
    columns: tuple[list[Line], list[Line]] = ([], [])
    for block in page.get_text("dict")["blocks"]:
        for raw_line in block.get("lines", []):
            x0, y0, _, _ = raw_line["bbox"]
            if not BODY_TOP <= y0 <= BODY_BOTTOM:
                continue
            line = _line_from_spans(raw_line)
            if line is None:
                continue
            columns[0 if x0 < COLUMN_SPLIT else 1].append(line)
    for column in columns:
        column.sort(key=lambda line: (line.y0, line.x0))
    return columns


def extract_entries(pdf_path: Path) -> list[Entry]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF (`fitz`) is required to extract the glossary PDF") from exc
    document = fitz.open(pdf_path)
    entries: list[Entry] = []
    current: Entry | None = None
    for pdf_page in range(FIRST_GLOSSARY_PDF_PAGE, LAST_GLOSSARY_PDF_PAGE + 1):
        printed_page = pdf_page - PDF_TO_PRINTED_OFFSET
        ordinal = 0
        for column in page_columns(document[pdf_page - 1]):
            for line in column:
                if line.headword:
                    if current is not None:
                        entries.append(current)
                    ordinal += 1
                    current = Entry(printed_page, ordinal, line.headword, [line.remainder])
                elif current is not None:
                    current.parts.append(line.text)
    if current is not None:
        entries.append(current)
    return entries


def split_loan_codes(definition: str) -> tuple[str, list[str]]:
    match = LOAN_CODE.search(definition)
    if not match:
        return definition, []
    raw = match.group(1)
    codes = []
    while raw:
        code = next((candidate for candidate in LOAN_LANGUAGES if raw.startswith(candidate)), None)
        if code is None:
            return definition, []
        codes.append(code)
        raw = raw[len(code):]
    return definition[: match.start()].strip(), codes


def grammatical_tags(definition: str) -> list[str]:
    tags: list[str] = []
    for prefix, values in POS_TAGS:
        if definition.startswith(prefix):
            tags.extend(values)
            break

    if definition.startswith("pred a.") or definition == "stunned":
        tags.append("adj")
    elif definition.startswith("(vi)COP"):
        tags.extend(("verb", "intr"))
    elif definition.startswith("rel."):
        tags.append("adv")
    elif definition.startswith("(pref)"):
        tags.append("prefix")
    elif definition.startswith("(phrase)") or definition == "be upon you":
        tags.append("multiword-expression")
    elif definition == "=too":
        tags.extend(("part", "emph"))
    elif definition == "COMP":
        tags.append("conj")
    elif definition == "PROPR":
        tags.append("suffix")
    elif definition.startswith("let’s go"):
        tags.append("verb")

    morphology = {
        "COP": "copula",
        "IMP": "impv",
        "PST": "pret",
        "PRS": "pres",
        "PRF": "perfect",
        "NEG": "neg",
        "SBJV": "subj",
    }
    for marker, tag in morphology.items():
        if re.search(rf"(?<![A-Z]){marker}(?![A-Z])", definition):
            if tag != "copula" and "verb" not in tags:
                tags.insert(0, "verb")
            tags.append(tag)
    for person, tag in (("1SG", "1sg"), ("2SG", "2sg"), ("3SG", "3sg"),
                        ("1PL", "1pl"), ("2PL", "2pl"), ("3PL", "3pl")):
        if person in definition:
            tags.append(tag)
    if re.search(r"(?:^|[.-])PL(?:[.;-]|$)", definition) and not any(
        tag.endswith("pl") for tag in tags
    ):
        tags.append("pl")
    return list(dict.fromkeys(tags))


def entry_row(entry: Entry) -> tuple[list[str], dict[str, str]]:
    definition, loan_codes = split_loan_codes(entry.definition)
    etymology = ""
    if loan_codes:
        names = ", ".join(LOAN_LANGUAGES[code] for code in loan_codes)
        marks = ", ".join(f"[{code}]" for code in loan_codes)
        etymology = f"Ali and Kobayashi mark this entry {marks} ({names})."
    tags = grammatical_tags(definition)
    if loan_codes:
        tags.append("loanword")
    if "Rakhshān" in definition:
        tags.extend(("dialectal", "dialect:Brahui:brahui_rakhshan:Rakhsh%C4%81n"))
    tags_text = " ".join(dict.fromkeys(tags))
    source = f"{SOURCE_ID}[p. {entry.printed_page}]"
    row = [
        LANGUAGE_ID, "", entry.headword, definition, "", entry.headword, "", source,
        "", etymology, entry.key, "", "", "", tags_text,
    ]
    audit = {
        "Printed_Page": str(entry.printed_page),
        "Entry": str(entry.ordinal),
        "Form": entry.headword,
        "Definition": definition,
        "Loan_Codes": "|".join(loan_codes),
        "Tags": tags_text,
        "Entry_Key": entry.key,
    }
    assert len(row) == RICH_COLUMNS
    return row, audit


def write_outputs(entries: list[Entry], output: Path, audit_path: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    rows_and_audit = [entry_row(entry) for entry in entries]
    with output.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(row for row, _ in rows_and_audit)
    fields = ["Printed_Page", "Entry", "Form", "Definition", "Loan_Codes", "Tags", "Entry_Key"]
    with audit_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audit for _, audit in rows_and_audit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/other/forms/20260813-ali-kobayashi-brahui.csv"),
    )
    parser.add_argument(
        "--audit", type=Path,
        default=Path("data/other/forms/raw_data/20260813-ali-kobayashi-brahui-audit.csv"),
    )
    args = parser.parse_args()
    entries = extract_entries(args.pdf)
    write_outputs(entries, args.output, args.audit)
    print(f"wrote {len(entries)} Brahui glossary entries to {args.output}")


if __name__ == "__main__":
    main()
