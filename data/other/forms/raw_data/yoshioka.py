#!/usr/bin/env python3
"""Extract Yoshioka's Eastern Burushaski vocabulary for Jambu ingestion.

The vocabulary occupies PDF pages 505--618 (printed pp. CLXXIX--CCXCII) of
Yoshioka's 2012 dissertation.  The PDF has useful embedded text and typography,
but a minority of Burushaski glyphs have no usable ToUnicode mapping.  This
extractor therefore uses the native text for structure and diacritics, and
repairs only those unmapped glyphs from cached, page-level Tesseract OCR.

Outputs under ``--output-dir``:

* ``yoshioka_entries.csv``: parsed entries with raw text and provenance;
* ``yoshioka_review.csv``: entries requiring manual review;
* ``yoshioka_auto_import.csv``: rich fifteen-column Jambu ingestion rows;
* ``yoshioka_report.md``: extraction and tagging statistics.

``--install`` copies the import CSV to ``data/other/forms``.  The source PDF is
never copied or modified.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import shutil
import subprocess
import threading
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_PDF = Path(
    "~/Documents/Linguistics/Burushaski/"
    "Burushaski, A Reference Grammar of Eastern (Yoshioka).pdf"
).expanduser()
VOCABULARY_PAGES = range(505, 619)
SCALE = 300 / 72
SOURCE_ID = "yoshioka2012"
PDFIUM_LOCK = threading.Lock()

# Yoshioka's abbreviations (thesis pp. xiii--xiv).  H/HM/HF/X/Y/Z are noun
# classes, not dialects; they deliberately do not occur in DIALECTS.
DIALECTS = {
    "HZ": "Hunza",
    "NG": "Nager",
    "HO": "Hopar",
    "GA": "Ganish",
    "AL": "Altit",
    "HS": "Hillside",
    "RF": "Riverfront",
}
LOAN_LANGUAGES = {
    "AR": "Arabic",
    "EN": "English",
    "KH": "Khowar",
    "PE": "Persian",
    "SH": "Shina",
    "UR": "Urdu",
}
FORM_OVERRIDES = {
    # Four roots are composed almost entirely of glyphs lacking ToUnicode
    # mappings, leaving too little native text for automatic OCR alignment.
    525: [("sprout, come up", "c̣ík")],
    544: [("furrow (a field), dig", "γajaγajám @-t-")],
    577: [("sniff, smell", "nas d-@-l-")],
    608: [("sneeze", "thiãũ")],
}
POS_TAGS = {
    "ADJ": ("adj",),
    "ADV": ("adv",),
    "CONJ": ("conj",),
    "COP": ("verb",),
    "DITR": ("verb", "tr"),
    "INTERJ": ("interj",),
    "INTR": ("verb", "intr"),
    "NUM": ("num",),
    "PRN": ("pron",),
    "TR": ("verb", "tr"),
}
GRAMMATICAL_TAGS = {
    # Number and inflection.  Yoshioka's ``DOUBLE PL`` denotes a second
    # plural formation, not dual number.
    "SG": ("sg",),
    "PL": ("pl",),
    "DOUBLE": ("double-plural",),
    # Verbal categories.
    "IPFV": ("ipfv",),
    "PFV": ("pfv",),
    "CP": ("participle", "conjunctive-participle"),
    "P": ("participle",),
    "PP": ("participle",),
    "INF": ("inf",),
    "IMP": ("impv",),
    "NEG": ("neg",),
    # Syntactic roles.
    "SUBJ": ("subj",),
    "OBJ": ("obj",),
    "DO": ("obj", "direct-object"),
    "IO": ("obj", "indirect-object"),
    # Case.
    "ABS": ("abs",),
    "ERG": ("erg",),
    "NOM": ("nom",),
    "GEN": ("gen",),
    "DAT": ("dat",),
    "ABL": ("abl",),
    "LOC": ("loc",),
    "INS": ("instr",),
    "OBL": ("obl",),
    "ADE": ("ade",),
    "INE": ("ine",),
    "ESS": ("ess",),
    # Deixis and other dictionary labels found in Appendix II.
    "PROX": ("prox",),
    "DIST": ("dist",),
    "INDEF": ("indef",),
    "FINALIS": ("finalis",),
}
NOUN_CLASSES = {"H", "HM", "HF", "X", "Y", "Z", "HX", "XY", "XZ", "YZ", "HXY"}
NOUN_CLASS_MEMBERS = {
    "H": ("H",),
    "HM": ("HM",),
    "HF": ("HF",),
    "X": ("X",),
    "Y": ("Y",),
    "Z": ("Z",),
    "HX": ("H", "X"),
    "XY": ("X", "Y"),
    "XZ": ("X", "Z"),
    "YZ": ("Y", "Z"),
    "HXY": ("H", "X", "Y"),
}
GRAMMAR_CODES = set(POS_TAGS) | set(GRAMMATICAL_TAGS) | NOUN_CLASSES | set(DIALECTS) | {
    "ONO",
    "RF", "RMND", "WB", "YS", "EB", "SH", "UR", "PE", "EN", "AR",
}


@dataclass
class OCRWord:
    text: str
    left: int
    right: int
    top: int
    bottom: int


@dataclass
class NativeLine:
    page: int
    top: float
    left: float
    text: str
    chars: list[dict]
    head: str = ""
    head_right: float = 0.0
    gloss_start: int = 0


@dataclass
class Entry:
    pdf_page: int
    printed_page: int
    form: str
    gloss: str
    raw_entry: str
    reference_note: str
    etymology: str
    tags: list[str]
    entry_key: str
    confidence: float = 1.0
    review_reasons: list[str] = field(default_factory=list)


def canonical(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    return re.sub(r"\s+", " ", text).strip()


def normalize_ocr_notation(text: str) -> str:
    """Restore Yoshioka's underdots when OCR mistakes them for cedillas.

    The Latin-script Tesseract model consistently reads the thesis's retroflex
    ``c̣``, ``ṣ``, and ``ṭ`` glyphs as ç, ş, and ț/ţ.  Cedillas are not part of
    Yoshioka's Burushaski notation (see the consonant inventory on PDF p. 33),
    so these substitutions are unambiguous in extracted headwords and notes.
    """
    return canonical(text.translate(str.maketrans({
        "ç": "c̣", "Ç": "C̣",
        "ş": "ṣ", "Ş": "Ṣ",
        "ț": "ṭ", "Ț": "Ṭ", "ţ": "ṭ", "Ţ": "Ṭ",
    })))


def _is_gentium(char: dict) -> bool:
    return "Gentium" in char.get("fontname", "")


def _is_italic(char: dict) -> bool:
    return "Italic" in char.get("fontname", "")


def _char_text(char: dict) -> str:
    value = char.get("text", "")
    # Unmapped Gentium glyphs are exposed as full letter-width spaces.  Real
    # spaces in this font are much narrower.
    if value == " " and _is_gentium(char) and char["x1"] - char["x0"] >= 4.0:
        return "�"
    return value


def native_lines(page, page_number: int) -> list[NativeLine]:
    """Reconstruct body lines while excluding the diagonal thesis watermark."""
    chars = [
        char for char in page.chars
        if 75 < char["top"] < 752
        and char.get("size", 0) > 2
        and "MSPGothic" not in char.get("fontname", "")
    ]
    groups: list[list[dict]] = []
    for char in sorted(chars, key=lambda item: (item["bottom"], item["x0"])):
        for group in reversed(groups[-5:]):
            if abs(group[0]["bottom"] - char["bottom"]) < 1.25:
                group.append(char)
                break
        else:
            groups.append([char])

    result = []
    for group in sorted(groups, key=lambda item: min(char["top"] for char in item)):
        group.sort(key=lambda item: item["x0"])
        visible = [char for char in group if _char_text(char).strip()]
        if not visible:
            continue
        text = canonical("".join(_char_text(char) for char in group))
        result.append(
            NativeLine(
                page=page_number,
                top=min(char["top"] for char in group),
                left=min(char["x0"] for char in visible),
                text=text,
                chars=group,
            )
        )
    return result


def _token_spans(chars: Sequence[dict]) -> list[tuple[str, int, int, list[dict]]]:
    spans = []
    start = None
    current: list[dict] = []
    for index, char in enumerate(chars):
        value = _char_text(char)
        if value == " ":
            if current:
                spans.append((canonical("".join(_char_text(c) for c in current)), start, index, current))
                current = []
                start = None
            continue
        if start is None:
            start = index
        current.append(char)
    if current:
        spans.append((canonical("".join(_char_text(c) for c in current)), start, len(chars), current))
    return spans


def analyze_candidate(line: NativeLine) -> bool:
    """Identify an entry line and recover its typographically marked root."""
    gentium = [char for char in line.chars if _is_gentium(char) and _char_text(char).strip()]
    if not gentium:
        return False
    first = gentium[0]
    if first["x0"] > 138 or line.top > 748:
        return False
    spans = _token_spans(line.chars)
    first_span_index = next(
        (index for index, (_, _, _, chars) in enumerate(spans) if any(_is_gentium(c) for c in chars)),
        None,
    )
    if first_span_index is None:
        return False
    first_span = spans[first_span_index]
    initial_italic = _is_italic(next(c for c in first_span[3] if _is_gentium(c)))

    chosen = []
    for token, start, end, token_chars in spans[first_span_index:]:
        gentium_chars = [c for c in token_chars if _is_gentium(c)]
        if not gentium_chars:
            break
        token_italic = any(_is_italic(c) for c in gentium_chars)
        if not initial_italic and token_italic:
            break
        if initial_italic and not token_italic:
            break
        chosen.append((token, start, end, token_chars))
        # A comma-separated stem list belongs to a roman root, never to the
        # bold-italic entry item selected here.
        if not initial_italic:
            break
    if not chosen:
        return False
    line.head = canonical(" ".join(token for token, *_ in chosen)).strip(" ,;")
    line.head_right = max(char["x1"] for _, _, _, chars in chosen for char in chars)

    # Locate the first English meaning token after the root/stems and grammar
    # codes. Gentium tokens are morphological forms; Times tokens are labels or
    # the definition.
    last_end = chosen[-1][2]
    gloss_start = 0
    for token, start, _end, token_chars in spans:
        if start < last_end:
            continue
        plain = token.strip("(),.;:[]")
        if not plain or plain in GRAMMAR_CODES or re.fullmatch(r"[-@=<>+/]+", plain):
            continue
        if any(_is_gentium(c) for c in token_chars):
            continue
        if re.search(r"[A-Za-z]", plain):
            gloss_start = start
            break
    line.gloss_start = gloss_start
    return bool(line.head)


def ocr_page(job: tuple[str, int, str]) -> tuple[int, list[list[OCRWord]]]:
    pdf_path, page_number, cache_dir = job
    cache_path = Path(cache_dir) / f"page-{page_number:03d}.json"
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return page_number, [[OCRWord(**word) for word in line] for line in payload]
    import pypdfium2 as pdfium

    # PDFium's document/page creation is not thread-safe on this source PDF.
    # Rendering is brief; OCR itself still runs concurrently outside the lock.
    with PDFIUM_LOCK:
        document = pdfium.PdfDocument(pdf_path)
        page = document[page_number - 1]
        image = page.render(scale=SCALE).to_pil()
        page.close()
        document.close()
    png = io.BytesIO()
    image.save(png, format="PNG")
    proc = subprocess.run(
        ["tesseract", "stdin", "stdout", "-l", "script/Latin", "--psm", "6", "tsv"],
        input=png.getvalue(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )
    rows = csv.DictReader(
        io.StringIO(proc.stdout.decode("utf-8", errors="replace")),
        delimiter="\t", quoting=csv.QUOTE_NONE,
    )
    grouped: dict[tuple[int, int, int], list[OCRWord]] = {}
    for row in rows:
        if row.get("level") != "5" or not row.get("text", "").strip():
            continue
        left, top = int(row["left"]), int(row["top"])
        width, height = int(row["width"]), int(row["height"])
        grouped.setdefault(
            (int(row["block_num"]), int(row["par_num"]), int(row["line_num"])), []
        ).append(OCRWord(row["text"], left, left + width, top, top + height))
    lines = list(grouped.values())
    for line in lines:
        line.sort(key=lambda word: word.left)
    lines.sort(key=lambda line: (min(word.top for word in line), min(word.left for word in line)))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps([[asdict(word) for word in line] for line in lines], ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(cache_path)
    return page_number, lines


def _ocr_head(line: NativeLine, ocr_lines: Sequence[Sequence[OCRWord]]) -> str:
    target = line.top * SCALE
    candidates = [
        words for words in ocr_lines
        if abs(min(word.top for word in words) - target) < 24
    ]
    if not candidates:
        return ""
    words = min(candidates, key=lambda row: abs(min(word.top for word in row) - target))
    boundary = line.head_right * SCALE + 8
    selected = [word.text for word in words if (word.left + word.right) / 2 <= boundary]
    return canonical(" ".join(selected)).strip(" ,;")


def repair_line_text(line: NativeLine, ocr_lines: Sequence[Sequence[OCRWord]]) -> str:
    """Use OCR for an audit-note line only when native text has unmapped glyphs."""
    if "�" not in line.text:
        return line.text
    target = line.top * SCALE
    candidates = [
        words for words in ocr_lines
        if abs(min(word.top for word in words) - target) < 24
    ]
    if not candidates:
        return normalize_ocr_notation(line.text.replace("�", "?"))
    words = min(candidates, key=lambda row: abs(min(word.top for word in row) - target))
    repaired = fill_holes(
        line.text, canonical(" ".join(word.text for word in words)), fallback_to_ocr=False
    )
    return normalize_ocr_notation(repaired.replace("�", "?"))


def fill_holes(native: str, ocr: str, *, fallback_to_ocr: bool = True) -> str:
    """Replace only unmapped native glyphs, retaining native PDF diacritics."""
    native = canonical(native)
    ocr = canonical(ocr)
    if "�" not in native or not ocr:
        return native
    native_words, ocr_words = native.split(), ocr.split()
    if len(native_words) != len(ocr_words):
        return ocr if fallback_to_ocr else native
    repaired = []
    for source, fallback in zip(native_words, ocr_words):
        if "�" not in source:
            repaired.append(source)
        elif len(source) == len(fallback):
            repaired.append("".join(b if a == "�" else a for a, b in zip(source, fallback)))
        else:
            repaired.append(fallback)
    return canonical(" ".join(repaired))


def grammatical_tags(text: str) -> list[str]:
    prefix = text.split("||", 1)[0].split("¶", 1)[0]
    tokens = set(re.findall(r"(?<![A-Za-z])([A-Z]{1,7})(?![A-Za-z])", prefix))
    tags = []
    if tokens & NOUN_CLASSES:
        tags.append("noun")
    for source_class in NOUN_CLASS_MEMBERS:
        if source_class in tokens:
            tags.extend(
                f"Burushaski-class-{member}"
                for member in NOUN_CLASS_MEMBERS[source_class]
            )
    if "HM" in tokens:
        tags.append("m")
    if "HF" in tokens:
        tags.append("f")
    for token in POS_TAGS:
        if token in tokens:
            tags.extend(POS_TAGS[token])
    for token in GRAMMATICAL_TAGS:
        if token in tokens:
            tags.extend(GRAMMATICAL_TAGS[token])
    # ONO has no dedicated canonical Jambu token; retain it verbatim in Notes
    # and use the system's indeclinable grammatical category.
    if "ONO" in tokens:
        tags.append("indecl")
    return list(dict.fromkeys(tags))


def dialect_tags(text: str) -> list[str]:
    prefix = text.split("¶", 1)[0]
    tags = ["dialect:Eastern%20Burushaski"]
    tokens = set(re.findall(r"(?<![A-Za-z])([A-Z]{2})(?![A-Za-z])", prefix))
    tags.extend(f"dialect:{DIALECTS[token]}" for token in DIALECTS if token in tokens)
    return list(dict.fromkeys(tags))


def loan_tags(text: str) -> list[str]:
    if "¶" not in text:
        return []
    note = text.split("¶", 1)[1]
    tokens = set(re.findall(r"(?<![A-Za-z])([A-Z]{2})(?![A-Za-z])", note))
    loans = [f"loan:{LOAN_LANGUAGES[token]}" for token in LOAN_LANGUAGES if token in tokens]
    return (["loanword"] + loans) if loans else []


def extract_gloss(first_line: NativeLine, raw_entry: str) -> str:
    lexical = raw_entry.split("||", 1)[0].split("¶", 1)[0]
    first_text = first_line.text
    if first_line.gloss_start:
        prefix = canonical("".join(_char_text(char) for char in first_line.chars[:first_line.gloss_start]))
        if lexical.startswith(prefix):
            lexical = lexical[len(prefix):]
        else:
            # Native line normalization can slightly alter spacing.
            lexical = first_text[len(prefix):] + raw_entry[len(first_text):]
    else:
        lexical = lexical.replace(first_line.head, "", 1)
    return canonical(lexical).strip(" ,;:-")


def parse_pages(document, pages: Sequence[int], ocr: dict[int, list[list[OCRWord]]]) -> list[Entry]:
    entries = []
    serial = 0
    for page_number in pages:
        lines = native_lines(document.pages[page_number - 1], page_number)
        candidates = []
        for index, line in enumerate(lines):
            if analyze_candidate(line):
                candidates.append((index, line))
        for position, (line_index, line) in enumerate(candidates):
            end = candidates[position + 1][0] if position + 1 < len(candidates) else len(lines)
            raw = canonical(" ".join(item.text for item in lines[line_index:end]))
            clean_raw = canonical(
                " ".join(repair_line_text(item, ocr.get(page_number, [])) for item in lines[line_index:end])
            )
            form = normalize_ocr_notation(
                fill_holes(line.head, _ocr_head(line, ocr.get(page_number, [])))
            )
            serial += 1
            tags = list(dict.fromkeys(
                grammatical_tags(raw) + dialect_tags(raw) + loan_tags(raw)
            ))
            references = clean_raw.split("||", 1)[1].split("¶", 1)[0].strip() if "||" in clean_raw else ""
            etymology = clean_raw.split("¶", 1)[1].strip() if "¶" in clean_raw else ""
            # English meanings are native PDF text; any unmapped glyphs here
            # occur only inside cited Burushaski forms embedded in a gloss.
            # Keep the meaning ingestible and make the damaged locus explicit.
            gloss = extract_gloss(line, raw).replace("�", "?")
            reasons = []
            for gloss_probe, corrected_form in FORM_OVERRIDES.get(page_number, []):
                if "�" in form and gloss_probe in gloss:
                    form = corrected_form
                    break
            if "�" in form:
                reasons.append("unresolved_glyph")
            if not re.search(r"[^\W\d_]", form, flags=re.UNICODE):
                reasons.append("suspicious_headword")
            if not gloss or gloss in GRAMMAR_CODES:
                reasons.append("missing_gloss")
            entries.append(
                Entry(
                    pdf_page=page_number,
                    printed_page=page_number - 326,
                    form=form,
                    gloss=gloss,
                    raw_entry=clean_raw,
                    reference_note=references,
                    etymology=etymology,
                    tags=tags,
                    entry_key=f"yoshioka-entry-{serial}",
                    confidence=max(0.0, 1.0 - 0.25 * len(reasons)),
                    review_reasons=reasons,
                )
            )
    return entries


def import_rows(entries: Iterable[Entry]) -> Iterable[list[str]]:
    for entry in entries:
        # Review candidates are retained in yoshioka_entries/review.csv but are
        # not installed. In practice these are isolated source-language or
        # parenthetical continuation fragments, not lexical entries.
        if entry.review_reasons:
            continue
        source = f"{SOURCE_ID}[p. {entry.pdf_page} (printed p. {entry.printed_page})]"
        yield [
            "Bur", "", entry.form, entry.gloss, "", "", "", source, "",
            entry.etymology, entry.entry_key, "", "", "",
            " ".join(entry.tags),
        ]


def write_csv(path: Path, rows: Iterable[Sequence[object]], header: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        if header:
            writer.writerow(header)
        writer.writerows(rows)


def write_outputs(output_dir: Path, entries: Sequence[Entry]) -> None:
    columns = list(Entry.__dataclass_fields__)
    rows = []
    for entry in entries:
        value = asdict(entry)
        value["tags"] = " ".join(entry.tags)
        value["review_reasons"] = ";".join(entry.review_reasons)
        rows.append([value[column] for column in columns])
    write_csv(output_dir / "yoshioka_entries.csv", rows, columns)
    write_csv(
        output_dir / "yoshioka_review.csv",
        (row for row, entry in zip(rows, entries) if entry.review_reasons),
        columns,
    )
    write_csv(output_dir / "yoshioka_auto_import.csv", import_rows(entries))
    counts = Counter(tag for entry in entries for tag in entry.tags)
    report = f"""# Yoshioka extraction report

- Vocabulary range: PDF pp. 505--618 (printed pp. 179--292)
- Parsed lexical entries: {len(entries):,}
- Installed ingestion rows: {sum(not entry.review_reasons for entry in entries):,}
- Entries with grammatical tags: {sum(any(not tag.startswith(('dialect:', 'loan:')) and tag != 'loanword' for tag in entry.tags) for entry in entries):,}
- Entries with explicit locality/dialect tags: {sum(any(tag != 'dialect:Eastern%20Burushaski' and tag.startswith('dialect:') for tag in entry.tags) for entry in entries):,}
- Entries marked as loanwords: {counts['loanword']:,}
- Entries with source etymology/cognate notes: {sum(bool(entry.etymology) for entry in entries):,}
- Review queue: {sum(bool(entry.review_reasons) for entry in entries):,}
- Unresolved embedded-font glyphs: {sum('unresolved_glyph' in entry.review_reasons for entry in entries):,}

H/HM/HF/X/Y/Z are treated as nominal classes. HZ/NG/HO/GA/AL/HS/RF are
mapped to Jambu ``dialect:`` tags. The original abbreviations and morphology are
retained in each row's dictionary-entry note.
"""
    (output_dir / "yoshioka_report.md").write_text(report, encoding="utf-8")


def parse_page_spec(spec: str | None) -> list[int]:
    if not spec:
        return list(VOCABULARY_PAGES)
    pages = set()
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
    work_dir = data_root / ".cache/ocr/yoshioka"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--pages", help="PDF pages, e.g. 505-510")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache-dir", type=Path, default=work_dir / "pages")
    parser.add_argument("--output-dir", type=Path, default=work_dir / "output")
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args(argv)
    args.pdf = args.pdf.expanduser().resolve()
    if not args.pdf.exists():
        parser.error(f"PDF not found: {args.pdf}")
    if not shutil.which("tesseract"):
        parser.error("tesseract is not installed or not on PATH")
    pages = parse_page_spec(args.pages)

    jobs = [(str(args.pdf), page, str(args.cache_dir)) for page in pages]
    ocr = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(ocr_page, job) for job in jobs]
        for future in as_completed(futures):
            page_number, lines = future.result()
            ocr[page_number] = lines

    import pdfplumber
    with pdfplumber.open(args.pdf) as document:
        entries = parse_pages(document, pages, ocr)
    write_outputs(args.output_dir, entries)
    if args.install:
        destination = data_root / "data/other/forms/20260726-yoshioka-eastern-burushaski.csv"
        shutil.copyfile(args.output_dir / "yoshioka_auto_import.csv", destination)
        print(f"Installed {destination}")
    print(f"Parsed {len(entries):,} entries; review {sum(bool(e.review_reasons) for e in entries):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
