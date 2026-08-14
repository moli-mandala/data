"""Extract Bhaskararao and Kobayashi's 2025 Toda dictionary.

The repository PDF is image-only to ordinary PDF libraries, but it contains a
high-quality Unicode text layer outside the visible crop box.  Ghostscript's
``txtwrite`` device exposes that layer losslessly, including Toda's combining
low lines, retroflex dots, ogoneks, and vowel-length marks.  The PDF also puts
the following page's text to the right of the crop box; this importer retains
only the visible left page.

Run from ``data/``::

    uv run python data/other/forms/raw_data/bhaskararao_toda.py \
      /path/to/B600_AAL68_Toda_Dictionary.pdf
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import tempfile
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


SOURCE_ID = "bhaskararao-toda2025"
LANGUAGE_ID = "Toda"
FIRST_DICTIONARY_PDF_PAGE = 17
LAST_DICTIONARY_PDF_PAGE = 295
RICH_COLUMNS = 15
VISIBLE_RIGHT_EDGE = 450.0
ENTRY_LEFT_MIN = 70.0
ENTRY_LEFT_MAX = 78.0
BODY_TOP = 94.0
BODY_BOTTOM = 735.0
DEDR = re.compile(r"\bDEDR\s+(\d+[a-z]?(?:\s*,\s*\d+[a-z]?)*)(?!\d)", re.I)
S2_OF = re.compile(r"\bS\s*2\s+of\s+([^,;.)]+-)", re.I)
POS = re.compile(
    r"\s+(?:"
    r"dem\.(?:a|adv|n|pron)\.|int\.(?:adv|pron)\.|idf\.(?:adv|pron)\."
    r"|int\.(?:n|phr)\.|prop\.n\.|rel\.adv\.|med-caus\."
    r"|n\.,(?:a|adv|echo|vi)\.|a\.,(?:n|adv)\.|kin\.,n\."
    r"|postp\.,suff\.|adv\.,(?:conj|postp)\.|pron\.,adv\."
    r"|vt\.,vi\.|vi\.,vt\.|n\.,\s*echo|n\.echo|abl\.|onom\.|n(?=\s)"
    r"|adj\.|adv\.|caus\.|conj\.|echo\.|emph\.|excl\.|idf\.|imp\."
    r"|impers\.|inst\.|itj\.|kin\.|loc\.|n\.|neg\.|num\.|obl\."
    r"|part\.|pers\.|phr\.|pl\.|postp\.|pron\.|prop\.|rel\.|sg\."
    r"|suff\.|v\.|vi\.|voc\.|vt\.|a\."
    r")(?=\s|$)",
    re.I,
)


@dataclass
class Glyph:
    x0: float
    x1: float
    char: str
    font: str


@dataclass
class Line:
    x0: float
    y: float
    text: str
    italic_prefix: str


@dataclass
class Entry:
    pdf_page: int
    printed_page: int
    ordinal: int
    head: str
    lines: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{SOURCE_ID}:p{self.printed_page}:e{self.ordinal}"

    @property
    def text(self) -> str:
        return _clean_spacing(" ".join(self.lines))


def _clean_spacing(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([([{])\s+", r"\1", text)
    text = re.sub(r"\b(DEDR|TGT|PB|RSLST|Sak\.)(?=\d)", r"\1 ", text)
    text = re.sub(r";(?=[A-Za-z])", "; ", text)
    text = re.sub(r",(?=[A-Za-z])", ", ", text)
    return unicodedata.normalize("NFC", text).strip()


def _bbox(value: str) -> tuple[float, float, float, float]:
    return tuple(float(part) for part in value.split())  # type: ignore[return-value]


def _attach_combining(glyphs: list[Glyph]) -> list[Glyph]:
    bases = [glyph for glyph in glyphs if not unicodedata.combining(glyph.char)]
    marks = [glyph for glyph in glyphs if unicodedata.combining(glyph.char)]
    suffixes: dict[int, list[str]] = {}
    for mark in marks:
        candidates = [
            (index, abs((base.x0 + base.x1) / 2 - mark.x0))
            for index, base in enumerate(bases)
            if base.char.strip() and base.x0 - 1.5 <= mark.x0 <= base.x1 + 1.5
        ]
        if not candidates:
            candidates = [
                (index, abs((base.x0 + base.x1) / 2 - mark.x0))
                for index, base in enumerate(bases)
                if base.char.strip()
                and abs((base.x0 + base.x1) / 2 - mark.x0) <= 4.0
            ]
        if candidates:
            target = min(candidates, key=lambda item: item[1])[0]
            suffixes.setdefault(target, []).append(mark.char)
    result = []
    for index, base in enumerate(bases):
        if index in suffixes:
            base = Glyph(base.x0, base.x1, base.char + "".join(suffixes[index]), base.font)
        result.append(base)
    return result


def _render_glyphs(glyphs: list[Glyph]) -> tuple[str, str]:
    glyphs = _attach_combining(sorted(glyphs, key=lambda glyph: (glyph.x0, glyph.x1)))
    rendered: list[str] = []
    italic: list[str] = []
    previous: Glyph | None = None
    in_italic_prefix = True
    for glyph in glyphs:
        gap = glyph.x0 - previous.x1 if previous is not None else 0
        inferred_space = previous is not None and gap > 2.2 and not (
            previous.char.endswith(" ") or glyph.char.startswith(" ")
        )
        if inferred_space:
            rendered.append(" ")
            if in_italic_prefix:
                italic.append(" ")
        rendered.append(glyph.char)
        if in_italic_prefix:
            if "Italic" in glyph.font or not glyph.char.strip():
                italic.append(glyph.char)
            else:
                in_italic_prefix = False
        previous = glyph
    return _clean_spacing("".join(rendered)), _clean_spacing("".join(italic))


def page_lines(page: ET.Element) -> list[Line]:
    by_y: dict[float, list[Glyph]] = {}
    for line in page.findall("./block/line"):
        for span in line.findall("span"):
            font = span.attrib.get("font", "")
            for char in span.findall("char"):
                x0, y, x1, _ = _bbox(char.attrib["bbox"])
                if x0 < 0 or x0 >= VISIBLE_RIGHT_EDGE or not BODY_TOP <= y <= BODY_BOTTOM:
                    continue
                value = char.attrib.get("c", "")
                if value:
                    by_y.setdefault(round(y, 1), []).append(Glyph(x0, x1, value, font))
    result = []
    for y, glyphs in sorted(by_y.items()):
        base_glyphs = [glyph for glyph in glyphs if not unicodedata.combining(glyph.char)]
        if not base_glyphs:
            continue
        text, italic_prefix = _render_glyphs(glyphs)
        if text:
            result.append(Line(min(glyph.x0 for glyph in base_glyphs), y, text, italic_prefix))
    return result


def extract_xml(pdf_path: Path, output: Path) -> None:
    command = [
        "gs", "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=txtwrite", "-dTextFormat=1",
        f"-dFirstPage={FIRST_DICTIONARY_PDF_PAGE}",
        f"-dLastPage={LAST_DICTIONARY_PDF_PAGE}",
        f"-sOutputFile={output}", str(pdf_path),
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("Ghostscript (`gs`) is required to extract the PDF text layer") from exc


def read_pages(xml_path: Path) -> list[ET.Element]:
    # txtwrite emits adjacent <page> roots, so wrap them in one document node.
    text = xml_path.read_text(encoding="utf-8")
    return list(ET.fromstring(f"<document>{text}</document>"))


def extract_entries(xml_path: Path) -> list[Entry]:
    entries: list[Entry] = []
    current: Entry | None = None
    for offset, page in enumerate(read_pages(xml_path)):
        pdf_page = FIRST_DICTIONARY_PDF_PAGE + offset
        printed_page = pdf_page - FIRST_DICTIONARY_PDF_PAGE + 1
        ordinal = 0
        for line in page_lines(page):
            is_start = (
                ENTRY_LEFT_MIN <= line.x0 <= ENTRY_LEFT_MAX
                and bool(line.italic_prefix)
                and line.text.startswith(line.italic_prefix)
            )
            if is_start:
                if current is not None:
                    entries.append(current)
                ordinal += 1
                head = line.italic_prefix
                pos = POS.search(line.text)
                # Several source pages expose each visual line as one italic text-layer span.
                # The lexical-category boundary is still explicit and is more trustworthy than
                # treating the entire line as a lemma.
                if pos and (head == line.text or pos.start() < len(head)):
                    head = line.text[: pos.start()].strip()
                current = Entry(pdf_page, printed_page, ordinal, head, [line.text])
            elif current is not None:
                current.lines.append(line.text)
    if current is not None:
        entries.append(current)
    return entries


def read_dedr_ids(path: Path) -> dict[str, str]:
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


def _head_forms(head: str) -> tuple[str, list[str], str]:
    head = _clean_spacing(head)
    sense = ""
    sense_match = re.search(r"\s+\(([1-9]\d*)\)$", head)
    if sense_match:
        sense = sense_match.group(1)
        head = head[: sense_match.start()].strip()

    variants: list[str] = []
    trailing = re.search(r"\s+\(([^()]*)\)$", head)
    if trailing and head[: trailing.start()].rstrip().endswith("-"):
        content = re.sub(r"^also\s+", "", trailing.group(1).strip(), flags=re.I)
        candidates = [part.strip() for part in re.split(r"\s*[/~,]\s*", content)]
        if candidates and all(candidate.endswith("-") for candidate in candidates):
            variants = candidates
            head = head[: trailing.start()].strip()
    return head, list(dict.fromkeys(form for form in variants if form != head)), sense


def _dedr_links(text: str, valid: dict[str, str]) -> tuple[list[str], list[str]]:
    links: list[str] = []
    invalid: list[str] = []
    for match in DEDR.finditer(text):
        for cited in re.findall(r"\d+[a-z]?", match.group(1), re.I):
            number = cited.lower()
            canonical = valid.get(number)
            if not canonical and number.isdigit():
                canonical = valid.get(number.lstrip("0") or "0")
            if not canonical and re.fullmatch(r"\d+[a-z]", number):
                canonical = valid.get(number[:-1])
            if canonical and canonical not in links:
                links.append(canonical)
            elif not canonical and number not in invalid:
                invalid.append(number)
    return links, invalid


def _definition(entry: Entry) -> str:
    first = entry.lines[0]
    remainder = first[len(entry.head):].strip() if first.startswith(entry.head) else first
    return _clean_spacing(" ".join([remainder, *entry.lines[1:]]))


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
        LANGUAGE_ID, parameter, form, gloss, "", "", notes, source, "", etymology,
        entry_key, variant_of_key, "", "|".join(derivation_parent_keys), " ".join(tags),
    ]


def build_rows(
    entries: list[Entry], valid_dedr: dict[str, str]
) -> tuple[list[list[str]], list[dict[str, str]]]:
    rows: list[list[str]] = []
    audit: list[dict[str, str]] = []
    previous_keys: dict[str, list[str]] = {}
    for entry in entries:
        form, variants, sense = _head_forms(entry.head)
        definition = _definition(entry)
        links, invalid = _dedr_links(entry.text, valid_dedr)
        source = f"{SOURCE_ID}[p. {entry.printed_page}, entry {entry.ordinal}]"
        s2 = S2_OF.search(definition)
        target_form = _clean_spacing(s2.group(1)) if s2 else ""
        target_candidates = previous_keys.get(target_form, [])
        variant_of = target_candidates[-1] if target_candidates else ""
        notes = ""
        if invalid:
            notes = "Unresolved DEDR citation(s): " + ", ".join(f"DEDR {item}" for item in invalid)

        parameters = links or [""]
        for index, parameter in enumerate(parameters, 1):
            key = entry.key if index == 1 else f"{entry.key}:link:{index}"
            etymology = ""
            tags: tuple[str, ...] = ()
            parent_key = ""
            parents: tuple[str, ...] = ()
            if variant_of and not links:
                etymology = f"Printed S2 stem of {target_form}"
                tags = ("variant",)
                parent_key = variant_of
                parents = (variant_of,)
            elif parameter:
                etymology = f"Bhaskararao and Kobayashi cite DEDR {parameter}"
                tags = ("inherited",)
            rows.append(
                _rich_row(
                    f"d{parameter}" if parameter else "", form, definition, source,
                    notes=notes, etymology=etymology, entry_key=key,
                    variant_of_key=parent_key, derivation_parent_keys=parents, tags=tags,
                )
            )

        previous_keys.setdefault(form, []).append(entry.key)
        for index, variant in enumerate(variants, 1):
            variant_key = f"{entry.key}:variant:{index}"
            rows.append(
                _rich_row(
                    "", variant, definition, source,
                    etymology=f"Printed S2 or alternate stem of {form}",
                    entry_key=variant_key, variant_of_key=entry.key,
                    derivation_parent_keys=(entry.key,), tags=("variant",),
                )
            )
            previous_keys.setdefault(variant, []).append(variant_key)

        audit.append({
            "Status": "ingested",
            "Entry_Key": entry.key,
            "PDF_Page": str(entry.pdf_page),
            "Printed_Page": str(entry.printed_page),
            "Printed_Head": entry.head,
            "Form": form,
            "Variants": "|".join(variants),
            "Sense": sense,
            "Definition": definition,
            "DEDR_IDs": "|".join(links),
            "Unresolved_DEDR_IDs": "|".join(invalid),
            "Variant_Of_Key": variant_of if not links else "",
            "Raw_Entry": entry.text,
        })
    return rows, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--xml", type=Path, help="reuse a previously extracted txtwrite XML file")
    parser.add_argument("--dedr-params", type=Path, default=Path("data/dedr/params.csv"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/other/forms/20260813-bhaskararao-toda.csv"),
    )
    parser.add_argument(
        "--audit", type=Path,
        default=Path("data/other/forms/raw_data/20260813-bhaskararao-toda-audit.csv"),
    )
    args = parser.parse_args()

    if args.xml:
        entries = extract_entries(args.xml)
    else:
        with tempfile.TemporaryDirectory(prefix="toda-dictionary-") as directory:
            xml_path = Path(directory) / "dictionary.xml"
            extract_xml(args.pdf, xml_path)
            entries = extract_entries(xml_path)
    rows, audit = build_rows(entries, read_dedr_ids(args.dedr_params))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    with args.audit.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(audit)
    print(f"wrote {len(rows)} form rows from {len(entries)} dictionary entries")


if __name__ == "__main__":
    main()
