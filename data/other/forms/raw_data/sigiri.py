#!/usr/bin/env python3
"""Extract the glossary-index from Paranavitana's *Sigiri Graffiti* II.

The source is an image-only scan.  Pages 431--480 of the PDF (printed pages
423--472) contain a three-column Old Sinhala glossary.  Each column is OCRed
separately so that Tesseract cannot interleave neighbouring entries.  The
checked-in Jambu import is deliberately conservative: the printed headword,
an automatically isolated English gloss when one is clear, and the complete
OCR entry with page/column provenance are retained.  Entries whose headword
or gloss needs inspection are also written to ``sigiri_review.csv``.

Run, for example::

    uv run --with pypdfium2 --with pillow python sigiri.py \
        --pdf /path/to/sigiri.pdf --install

Tesseract with the ``script/Latin`` model must be available on PATH.  OCR is
cached by page under the output directory, so interrupted runs resume.
"""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
import re
import shutil
import subprocess
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_PDF = Path(
    "~/Documents/Linguistics/Indo-European/Indo-Aryan/NIA/Insular/sigiri.pdf"
).expanduser()
DEFAULT_OUTPUT_DIR = HERE.parents[3] / ".cache/ocr/sigiri/output"
INSTALL_PATH = HERE.parent / "20260726-paranavitana-sigiri.csv"

PDF_PAGES = range(431, 481)
PRINTED_PAGE_OFFSET = 8
SCALE = 300 / 72

LANGUAGE = "OSi"
SOURCE = "paranavitana"
CDIAL_PARAMS = HERE.parents[3] / "data/cdial/params.csv"

ENTRY_START_MIN_LEFT = 0
ENTRY_START_MAX_LEFT = round(63 * SCALE)
HEAD_PATTERN = re.compile(r"^([^,]{1,90}),\s*(.*)$")
ENTRY_DESCRIPTOR_PATTERN = re.compile(
    r"^(?:"
    r"s(?:\.[fm])?\.?|a\.?|adv\.?|vb\.?|v\.l\.?|indec\.?|prt\.?|pron\.?|"
    r"num\.?|pref\.?|suf\.?|interj\.?|cond\.?|abs\.?|inf\.?|op\.?|cd\.?|"
    r"nom\.?|acc\.?|gen\.?|dat\.?|loc\.?|inst\.?|abl\.?|voc\.?|"
    r"orth\.?|var\.?|e\.f\.?|l\.f\.?|neg\.?"
    r")(?:\s|,|$)",
    re.IGNORECASE,
)
GRAMMAR_PATTERN = re.compile(
    r"^(?:"
    r"s(?:\.[fm])?\.|a\.|adv\.|vb\.|v\.l\.|indec\.|prt\.|pron\.|"
    r"num\.|pref\.|suf\.|interj\.|cond\.|abs\.|inf\.|"
    r"nom\.|acc\.|gen\.|dat\.|loc\.|inst\.|abl\."
    r")(?:\s|,|$)",
    re.IGNORECASE,
)
REFERENCE_START = re.compile(
    r"(?:,\s*\d{1,3}(?:\s*[,;.]|\s|$)|"
    r";\s*(?:loc|acc|nom|gen|dat|inst|abl|voc|cf)\.?\b|"
    r"\.\s+See\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class OCRLine:
    text: str
    left: int
    top: int
    confidence: float


@dataclass
class Entry:
    pdf_page: int
    printed_page: int
    column: int
    top: int
    headword: str
    gloss: str
    raw_entry: str
    confidence: float
    review_reasons: list[str] = field(default_factory=list)
    sanskrit_etyma: list[str] = field(default_factory=list)
    cdial_ids: list[str] = field(default_factory=list)


def normalize_sanskrit_etymon(text: str) -> str:
    """Return a forgiving key shared by Sigiri OCR and CDIAL headwords.

    Accent, vowel length, homonym numbers, and punctuation are removed, but
    segmental spelling is retained. A match is accepted only when this key
    identifies one CDIAL entry, so normalization cannot choose a homonym.
    """
    text = html.unescape(text).casefold()
    text = text.translate(
        str.maketrans({
            "ı": "i", "ſ": "s", "ʰ": "h", "ṃ": "m", "ṁ": "m",
            "ŋ": "n", "ṅ": "n", "ñ": "n", "ṇ": "n",
        })
    )
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"m(?=[kg])", "n", text)
    return re.sub(r"[^a-z]+", "", text)


def extract_sanskrit_etyma(raw_entry: str) -> list[str]:
    """Extract Sanskrit source forms from Paranavitana's bracketed note."""
    match = re.search(r"\bSkt\s*[.,:]\s*", raw_entry, re.IGNORECASE)
    if not match:
        return []
    tail = raw_entry[match.end() :].split("]", 1)[0]
    # Pali/comparative citations and inflectional explanations start a new
    # part of the bracket. Plus signs and hyphens belong to compounds.
    tail = re.split(
        r"\s*;\s*|,\s*(?:e\.f|l\.f|lit|var)\.|"
        r"\b(?:cf|P|Pk|Pr|T|mod\.\s*S)\s*[.,:]",
        tail,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    tail = tail.split("=", 1)[0]
    tail = re.sub(r"\([^)]*\)", " ", tail)
    result = []
    for part in re.split(r"\s+or\s+|/", tail, flags=re.IGNORECASE):
        part = part.strip(" ,;:.[]{}()")
        if part and normalize_sanskrit_etymon(part):
            result.append(part)
    return result


def load_cdial_headword_index(path: Path = CDIAL_PARAMS) -> dict[str, set[str]]:
    """Map normalized Sanskrit headwords to all compatible CDIAL IDs."""
    index: dict[str, set[str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 2 or not row[0]:
                continue
            forms = [part.strip() for part in row[1].split(",")]
            if len(row) > 3:
                forms.extend(html.unescape(x) for x in re.findall(r"<b>(.*?)</b>", row[3]))
            for form in forms:
                key = normalize_sanskrit_etymon(form)
                if key:
                    index.setdefault(key, set()).add(row[0].lower())
    return index


def match_cdial_ids(
    sanskrit_etyma: list[str], index: dict[str, set[str]]
) -> tuple[list[str], bool]:
    """Return unique exact-headword matches and whether a key was ambiguous."""
    result: list[str] = []
    ambiguous = False
    for etymon in sanskrit_etyma:
        key = normalize_sanskrit_etymon(etymon)
        # In this scan the initial h of hasta is consistently OCRed as k
        # (both occurrences gloss 'hand'). Do not accidentally attach those
        # rows to unrelated CDIAL kaṣṭa; repaired hasta remains ambiguous in
        # CDIAL and is therefore correctly left for review.
        if key == "kasta" and "ṣ" not in etymon:
            key = "hasta"
        candidates = index.get(key, set())
        if len(candidates) > 1:
            ambiguous = True
        elif len(candidates) == 1:
            candidate = next(iter(candidates))
            if candidate not in result:
                result.append(candidate)
    return result, ambiguous


def plausible_rules(rules: tuple[int, int] | list[int], width: int, pdf_page: int) -> bool:
    first, second = rules
    first_fraction = first / width
    spacing_fraction = (second - first) / width
    first_ok = (0.35 <= first_fraction <= 0.44) if pdf_page % 2 == 0 else (
        0.31 <= first_fraction <= 0.40
    )
    return first_ok and 0.22 <= spacing_fraction <= 0.30


def detect_column_crops(
    image, pdf_page: int | None = None
) -> tuple[list[tuple[int, int, int, int]], tuple[int, int]]:
    """Locate the two printed rules and return non-overlapping pixel crops.

    Recto/verso registration differs by more than 100 pixels at 300 DPI, so
    fixed page coordinates either clip headwords or admit a neighbouring
    column.  The rules are the longest near-vertical dark runs in their broad
    thirds of the page and remain detectable even on the short final page.
    """
    gray = image.convert("L")
    y0 = round(75 * SCALE)
    y1 = min(round(760 * SCALE), image.height - 1)

    def longest_run(x: int) -> int:
        run = best = 0
        for y in range(y0, y1):
            if gray.getpixel((x, y)) < 160:
                run += 1
                best = max(best, run)
            else:
                run = 0
        return best

    def rule_between(left: int, right: int) -> int:
        scores = {x: longest_run(x) for x in range(left, right)}
        peak = max(scores, key=scores.get)
        # Rules are several pixels thick and slightly slanted in the scan.
        # Crop from the left edge of that dark cluster, not from its peak.
        threshold = scores[peak] * 0.5
        cluster = [
            x for x in range(max(left, peak - 30), min(right, peak + 31))
            if scores[x] >= threshold
        ]
        return min(cluster)

    if pdf_page is None:
        first_band = (0.25, 0.50)
    elif pdf_page % 2 == 0:
        first_band = (0.35, 0.44)
    else:
        first_band = (0.31, 0.40)
    first = rule_between(
        round(image.width * first_band[0]), round(image.width * first_band[1])
    )
    second = rule_between(
        first + round(image.width * 0.22),
        min(image.width, first + round(image.width * 0.30) + 1),
    )
    column_width = second - first
    if pdf_page is not None and not plausible_rules((first, second), image.width, pdf_page):
        raise ValueError(f"implausible Sigiri column rules: {first}, {second}")

    outer_margin = round(20 * SCALE)
    right_gap = round(2 * SCALE)
    crops = [
        (max(0, first - column_width - outer_margin), y0, first - right_gap, y1),
        (first + right_gap, y0, second - right_gap, y1),
        (
            second + right_gap,
            y0,
            min(image.width, second + column_width - right_gap),
            y1,
        ),
    ]
    return crops, (first, second)


def _tesseract_lines(image) -> list[OCRLine]:
    payload = io.BytesIO()
    image.save(payload, format="PNG")
    process = subprocess.run(
        [
            "tesseract", "stdin", "stdout", "-l", "script/Latin",
            "--psm", "6", "tsv",
        ],
        input=payload.getvalue(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    rows = csv.DictReader(
        io.StringIO(process.stdout.decode("utf-8", errors="replace")),
        delimiter="\t",
        quoting=csv.QUOTE_NONE,
    )
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        if row.get("level") != "5" or not row.get("text", "").strip():
            continue
        key = (row["block_num"], row["par_num"], row["line_num"])
        groups.setdefault(key, []).append(row)

    result: list[OCRLine] = []
    for words in groups.values():
        words.sort(key=lambda word: int(word["left"]))
        confidences = [float(word["conf"]) for word in words if float(word["conf"]) >= 0]
        result.append(
            OCRLine(
                text=" ".join(word["text"] for word in words),
                left=min(int(word["left"]) for word in words),
                top=min(int(word["top"]) for word in words),
                confidence=sum(confidences) / len(confidences) if confidences else 0.0,
            )
        )
    return sorted(result, key=lambda line: (line.top, line.left))


def ocr_page(pdf_path: Path, pdf_page: int, cache_dir: Path) -> dict:
    cache_path = cache_dir / f"page-{pdf_page:03d}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        # All source pages have the same rendered width at a given SCALE.
        rendered_width = round(549 * SCALE)
        if plausible_rules(cached.get("column_rules", (0, 0)), rendered_width, pdf_page):
            return cached

    try:
        import pypdfium2 as pdfium
    except ImportError as exc:  # pragma: no cover - CLI environment dependent
        raise RuntimeError("pypdfium2 is required to render the source PDF") from exc

    document = pdfium.PdfDocument(str(pdf_path))
    page = document[pdf_page - 1]
    image = page.render(scale=SCALE).to_pil()
    crops, rules = detect_column_crops(image, pdf_page)
    columns = []
    for number, crop in enumerate(crops, 1):
        lines = _tesseract_lines(image.crop(crop))
        columns.append({"column": number, "lines": [asdict(line) for line in lines]})
    page.close()
    document.close()

    data = {"pdf_page": pdf_page, "column_rules": rules, "columns": columns}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    temporary.replace(cache_path)
    return data


def normalize_headword(raw: str) -> str:
    """Apply only high-confidence OCR repairs to a printed headword."""
    text = unicodedata.normalize("NFC", raw)
    if text.strip(" |;:.`'\"[]{}") in {"€”,", "€”", "€"}:
        return "e"
    text = re.sub(r"^[^0-9A-Za-zÀ-žĀ-ž]+", "", text)
    text = text.strip(" |;:.`'\"[]{}")
    text = text.replace("ä", "ā").replace("Ä", "Ā")
    text = text.replace("ã", "ā").replace("Ã", "Ā")
    text = re.sub(r"(?:ńñ|ùñ|uñ|nñ|ñ)(?=g)", "ṅ", text, flags=re.IGNORECASE)
    # Tesseract's readings of the tiny homonym superscripts are inconsistent.
    # Dropping an uncertain terminal mark is safer than inventing the wrong ID;
    # duplicate forms remain separate source entries in CLDF.
    text = re.sub(r"[!?®ð°]+$", "", text)
    return re.sub(r"\s+", " ", text).strip()


def is_entry_start(line: OCRLine) -> bool:
    """Recognise the hanging-indent first line of a glossary entry."""
    if not (ENTRY_START_MIN_LEFT <= line.left <= ENTRY_START_MAX_LEFT):
        return False
    match = HEAD_PATTERN.match(line.text.strip(" |"))
    if not match:
        return False
    head, remainder = match.groups()
    if (
        len(head) > 45
        or re.search(r"[\[\]{};|\d]", head)
        or re.match(r"^[a-z]\.", head, re.IGNORECASE)
    ):
        return False
    return bool(ENTRY_DESCRIPTOR_PATTERN.match(remainder))


def _dehyphenate(lines: list[str]) -> str:
    text = ""
    for line in lines:
        line = re.sub(r"\s+", " ", line).strip(" |")
        if not line:
            continue
        if text.endswith("-") and line[:1].islower():
            text = text[:-1] + line
        else:
            text = f"{text} {line}".strip()
    return text


def extract_gloss(raw_entry: str, headword: str) -> str:
    """Return a conservative English definition, or empty text if uncertain."""
    remainder = raw_entry[len(raw_entry.split(",", 1)[0]) + 1 :].strip()
    closing = remainder.find("]")
    candidate = remainder[closing + 1 :].lstrip(" ,;") if closing >= 0 else ""

    if not candidate:
        # Without a bracketed etymology, discard a leading grammatical label.
        match = GRAMMAR_PATTERN.match(remainder)
        if match:
            candidate = remainder[match.end() :].lstrip(" ,;")

    if not candidate:
        return ""

    # A definition may be explicitly quoted.  Prefer it when it occurs in the
    # definition portion, never an etymological gloss inside the brackets.
    quoted = re.search(r"[‘'\"]([^’'\"]{2,180})[’'\"]", candidate)
    if quoted and re.search(r"[A-Za-z]", quoted.group(1)):
        candidate = quoted.group(1)
    else:
        stop = REFERENCE_START.search(candidate)
        if stop:
            candidate = candidate[: stop.start()]
        candidate = re.sub(r"\s+", " ", candidate).strip(" ,;:.")

    if re.match(r"^(?:cf\.|See\b|var\. of\b|e\.f\. of\b|l\.f\. of\b)", candidate, re.I):
        return ""
    if candidate == headword or not re.search(r"[A-Za-z]", candidate):
        return ""
    return candidate[:300]


def parse_pages(
    pages: list[dict], cdial_index: dict[str, set[str]] | None = None
) -> list[Entry]:
    entries: list[Entry] = []
    current: dict | None = None

    def finish() -> None:
        nonlocal current
        if current is None:
            return
        raw_entry = _dehyphenate(current["lines"])
        match = HEAD_PATTERN.match(raw_entry)
        if not match:
            current = None
            return
        headword = normalize_headword(match.group(1))
        if not entries and current["pdf_page"] == 431 and current["column"] == 1:
            # The oversized title touches the first headword in Tesseract's
            # segmentation on this one page; the printed entry is a¹.
            headword = "a"
        confidence = sum(current["confidences"]) / len(current["confidences"])
        gloss = extract_gloss(raw_entry, headword)
        sanskrit_etyma = extract_sanskrit_etyma(raw_entry)
        cdial_ids, ambiguous_etymon = (
            match_cdial_ids(sanskrit_etyma, cdial_index)
            if cdial_index is not None
            else ([], False)
        )
        reasons: list[str] = []
        if not gloss:
            reasons.append("missing_gloss")
        if confidence < 75:
            reasons.append("low_ocr_confidence")
        if not headword or len(headword) > 50 or re.search(r"\d{2,}|[\[\]{}|]", headword):
            reasons.append("suspicious_headword")
        if re.search(r"[?®ð]", headword):
            reasons.append("uncertain_glyph")
        if ambiguous_etymon:
            reasons.append("ambiguous_sanskrit_etymon")
        entries.append(
            Entry(
                pdf_page=current["pdf_page"],
                printed_page=current["pdf_page"] - PRINTED_PAGE_OFFSET,
                column=current["column"],
                top=current["top"],
                headword=headword,
                gloss=gloss,
                raw_entry=raw_entry,
                confidence=confidence,
                review_reasons=reasons,
                sanskrit_etyma=sanskrit_etyma,
                cdial_ids=cdial_ids,
            )
        )
        current = None

    for page in pages:
        for column in page["columns"]:
            for raw_line in column["lines"]:
                line = OCRLine(**raw_line)
                text = line.text.strip(" |")
                if not text or "GLOSSARY" in text.upper() or re.fullmatch(r"\d{3}", text):
                    continue
                starts_entry = is_entry_start(line)
                if starts_entry:
                    finish()
                    current = {
                        "pdf_page": page["pdf_page"],
                        "column": column["column"],
                        "top": line.top,
                        "lines": [text],
                        "confidences": [line.confidence],
                    }
                elif current is not None:
                    current["lines"].append(text)
                    current["confidences"].append(line.confidence)
    finish()
    return entries


def write_outputs(entries: list[Entry], output_dir: Path, install: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "sigiri_entries.csv"
    review_path = output_dir / "sigiri_review.csv"
    import_path = output_dir / "sigiri_auto_import.csv"

    fields = [
        "pdf_page", "printed_page", "column", "top", "headword", "gloss",
        "confidence", "review_reasons", "raw_entry", "sanskrit_etyma", "cdial_ids",
    ]
    with audit_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for entry in entries:
            row = asdict(entry)
            row["review_reasons"] = ";".join(entry.review_reasons)
            row["sanskrit_etyma"] = ";".join(entry.sanskrit_etyma)
            row["cdial_ids"] = ";".join(entry.cdial_ids)
            writer.writerow(row)

    with review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for entry in entries:
            if entry.review_reasons:
                row = asdict(entry)
                row["review_reasons"] = ";".join(entry.review_reasons)
                row["sanskrit_etyma"] = ";".join(entry.sanskrit_etyma)
                row["cdial_ids"] = ";".join(entry.cdial_ids)
                writer.writerow(row)

    with import_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        import_rows = 0
        for entry in entries:
            source = (
                f"{SOURCE}[p. {entry.pdf_page} (printed p. {entry.printed_page}), "
                f"col. {entry.column}]"
            )
            for cdial_id in entry.cdial_ids or [""]:
                writer.writerow(
                    [LANGUAGE, cdial_id, entry.headword, entry.gloss, "", "", "", source]
                )
                import_rows += 1

    if install:
        shutil.copyfile(import_path, INSTALL_PATH)
        print(f"Installed {import_rows} rows at {INSTALL_PATH}")
    print(
        f"Wrote {len(entries)} entries; "
        f"{sum(bool(entry.review_reasons) for entry in entries)} need review; "
        f"{sum(bool(entry.sanskrit_etyma) for entry in entries)} have Sanskrit etyma and "
        f"{sum(bool(entry.cdial_ids) for entry in entries)} link to CDIAL"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="discard cached OCR")
    args = parser.parse_args()

    if not args.pdf.exists():
        parser.error(f"source PDF not found: {args.pdf}")
    if shutil.which("tesseract") is None:
        parser.error("tesseract is required on PATH")

    cache_dir = args.output_dir / "cache"
    if args.refresh and cache_dir.exists():
        shutil.rmtree(cache_dir)
    pages = [ocr_page(args.pdf, page, cache_dir) for page in PDF_PAGES]
    entries = parse_pages(pages, load_cdial_headword_index())
    write_outputs(entries, args.output_dir, args.install)


if __name__ == "__main__":
    main()
