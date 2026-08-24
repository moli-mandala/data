#!/usr/bin/env python3
"""Extract Andersen's Minor Rock Edict dictionary into Jambu rows.

The source is an image-only scan.  PDF pages 134--177 (printed pages
136--179) contain the one-column dictionary and concordance.  Tesseract's
line coordinates recover the printed hanging indent: bold dictionary heads
begin at the left margin, while comparison, morphology, and concordance lines
are indented.  The full OCR entry is retained with page provenance so every
automatic transcription remains auditable.

Run with Tesseract and ``pypdfium2`` available::

    python andersen_mre.py --pdf /path/to/andersen.pdf --install
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import quote


HERE = Path(__file__).resolve().parent
DEFAULT_PDF = Path(
    "~/Documents/Linguistics/Indo-European/Indo-Aryan/MIA/"
    "Studies in the Minor Rock Edicts of Aśoka — Andersen (1990) (1).pdf"
).expanduser()
DEFAULT_CACHE = HERE.parents[3] / ".cache/ocr/andersen-mre/pages"
DEFAULT_OUTPUT = HERE.parents[3] / ".cache/ocr/andersen-mre/output"
INSTALL_PATH = HERE.parent / "20260804-andersen-mre.csv"
CDIAL_PARAMS = HERE.parents[3] / "data/cdial/params.csv"

PDF_PAGES = range(134, 178)
PRINTED_PAGE_DELTA = 2
SCALE = 250 / 72
LANGUAGE = "As"
SOURCE = "andersen1990"

DESCRIPTOR = re.compile(
    r"^(?:m\.|f\.|n\.|a\.|adv\.|conj\.|pron\.|num\.|part\.|prep\.|"
    r"interj\.|indec\.|vb\.|pres\.|aor\.|pp\.|cf\.)(?=\s|$)",
    re.IGNORECASE,
)
ENTRY_STOPWORDS = {
    "a", "cf.", "dictionary", "in", "of", "parenthetical", "rv", "ry",
    "see", "skt", "turner",
}
HEADWORD_OVERRIDES = {
    "aḍdhatiya-": "aḍhatiya-",
    "diyaqhiya-": "diyaḍhiya-",
    "jaānapada-": "jānapada-",
    "Ppiyadasi-": "piyadasi-",
    "Piyadasinama-": "piyadasināma-",
    "māānüsa-": "mānusa-",
    "vampa-": "vaṃṇa-",
    "vayajana-": "vayaṃjana-",
    "suvamṇa-": "suvaṃṇa-",
    "suvamṇagiri-": "suvaṃṇagiri-",
    "ka-": "śaka-",
    "Saca-": "śaca-",
    "Svaga-": "śvaga-",
}
GLOSS_OVERRIDES = {"kam": "well", "hakam": "I", "mātā-": "mother"}
MRE_SITES = {
    # Andersen abbreviation: (CLDF dialect ID, canonical dialect label).
    # Four sites reuse pre-existing Ashokan metadata rows; the others are
    # Andersen MRE additions in cldf/languages.csv.
    "Ah": ("as-ahraura", "Ahraura"),
    "Bh": ("as-bahapur", "Bahapur"),
    "Bi": ("bh", "Bairat Bhabhru"),
    "Br": ("as-brahmagiri", "Brahmagiri"),
    "Er": ("as-erragudi", "Erragudi"),
    "Ga": ("gav", "Gavimath"),
    "Gu": ("as-gujarra", "Gujarra"),
    "Jt": ("as-jatinga-ramesvara", "Jatinga-Ramesvara"),
    "Ms": ("as-maski", "Maski"),
    "Ni": ("as-nittur", "Nittur"),
    "Pa": ("as-panguraria", "Panguraria"),
    "Pn": ("as-panguraria", "Panguraria"),
    "Pl": ("as-palkigundu", "Palkigundu"),
    "P1": ("as-palkigundu", "Palkigundu"),
    "Ra": ("as-rajula-mandagiri", "Rajula-Mandagiri"),
    "Ru": ("ru", "Rupnath"),
    "Sa": ("sah", "Sahasram"),
    "Sd": ("as-siddapura", "Siddapura"),
    "Ud": ("as-udegolam", "Udegolam"),
}
SITE_PATTERN = re.compile(
    r"(?<![A-Za-z])(" + "|".join(map(re.escape, MRE_SITES)) + r")(?![A-Za-z])"
)
SANSKRIT_SOURCE = re.compile(
    r"(?:^|\s)(?:RV|AVP?|AA|JB|MS|TS|Skt)\.?\s*:?\s*"
    r"([*A-Za-z\u00c0-\u024f\u1e00-\u1eff\u0300-\u036f-]+)",
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
    top: int
    headword: str
    gloss: str
    raw_entry: str
    confidence: float
    pos: str = ""
    sanskrit_etymon: str = ""
    cdial_id: str = ""
    dialects: list[str] = field(default_factory=list)
    review_reasons: list[str] = field(default_factory=list)


def _tesseract_lines(image) -> list[OCRLine]:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    result = subprocess.run(
        ["tesseract", "stdin", "stdout", "-l", "script/Latin", "--psm", "6", "tsv"],
        input=buffer.getvalue(), capture_output=True, check=True,
    )
    # Tesseract does not escape literal quotation marks in the TSV text field,
    # so ordinary CSV quote handling can accidentally merge many physical
    # lines into one record.
    rows = csv.DictReader(
        io.StringIO(result.stdout.decode("utf-8", "replace")),
        delimiter="\t", quoting=csv.QUOTE_NONE,
    )
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        if row.get("level") != "5" or not row.get("text", "").strip():
            continue
        key = (row["page_num"], row["block_num"], row["par_num"], row["line_num"])
        groups.setdefault(key, []).append(row)
    lines = []
    for words in groups.values():
        words.sort(key=lambda word: int(word["left"]))
        confidences = [float(word["conf"]) for word in words if float(word["conf"]) >= 0]
        lines.append(OCRLine(
            " ".join(word["text"] for word in words),
            min(int(word["left"]) for word in words),
            min(int(word["top"]) for word in words),
            sum(confidences) / len(confidences) if confidences else 0,
        ))
    return sorted(lines, key=lambda line: (line.top, line.left))


def ocr_page(pdf_path: Path, page_number: int, cache_dir: Path) -> dict:
    cache_path = cache_dir / f"page-{page_number:03d}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("cache_version") == 2:
            return cached
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(str(pdf_path))
    page = document[page_number - 1]
    image = page.render(scale=SCALE).to_pil()
    lines = _tesseract_lines(image)
    data = {
        "cache_version": 2,
        "pdf_page": page_number,
        "width": image.width,
        "lines": [asdict(line) for line in lines],
    }
    page.close()
    document.close()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    temporary.replace(cache_path)
    return data


def normalize_headword(text: str) -> str:
    """Apply only recurring, visually verified OCR substitutions."""
    text = unicodedata.normalize("NFC", text)
    text = text.translate(str.maketrans({
        "ț": "ṭ", "ţ": "ṭ", "ș": "ṣ", "ş": "ṣ", "ņ": "ṇ", "ä": "ā",
        "Ț": "Ṭ", "Ţ": "Ṭ", "Ș": "Ṣ", "Ş": "Ṣ", "Ņ": "Ṇ", "Ä": "Ā",
    }))
    text = re.sub(r"qd(?=h)", "ḍ", text, flags=re.IGNORECASE)
    text = re.sub(r"q(?=h)", "ḍ", text, flags=re.IGNORECASE)
    text = re.sub(r"^[^A-Za-z\u00c0-\u024f\u1e00-\u1eff*]+", "", text)
    text = text.strip(" ,;:.\"'`[]{}")
    return HEADWORD_OVERRIDES.get(text, text)


def normalize_etymon(text: str) -> str:
    text = html.unescape(text).casefold().rstrip("-")
    text = text.translate(str.maketrans({"ṃ": "m", "ṁ": "m", "ṅ": "n", "ñ": "n", "ṇ": "n"}))
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]+", "", text)


def load_cdial_index(path: Path = CDIAL_PARAMS) -> dict[str, set[str]]:
    index: dict[str, set[str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 2 or not row[0]:
                continue
            forms = [part.strip() for part in row[1].split(",")]
            if len(row) > 3:
                forms.extend(html.unescape(x) for x in re.findall(r"<b>(.*?)</b>", row[3]))
            for form in forms:
                key = normalize_etymon(form)
                if key:
                    index.setdefault(key, set()).add(row[0].lower())
    return index


def is_entry_candidate(line: OCRLine) -> bool:
    text = line.text.strip(" |.")
    if re.match(r"A\s+vipula-\s+a\.", text):
        return True
    if len(text) > 180:
        return False
    match = re.match(r"^(\S{1,60})\s+(.+)$", text)
    if not match:
        return False
    head, remainder = match.groups()
    if re.search(r"\d|[\[\]()]", head) or head.casefold() in ENTRY_STOPWORDS:
        return False
    return bool(
        DESCRIPTOR.match(remainder)
        or re.search(r'["\u201c][^"\u201d]+["\u201d]', remainder)
        or re.match(r"(?:see\b|\d[A-Z]\b)", remainder, re.IGNORECASE)
    )


def entry_start_keys(page: dict) -> set[tuple[int, int]]:
    """Select the candidate line following the page's sloping left margin.

    The photographed pages are mildly trapezoidal, so a fixed x threshold
    either drops heads near the foot or admits indented prose.  True heads form
    a nearly straight margin; a tiny RANSAC fit recovers that line.
    """
    candidates = [OCRLine(**raw) for raw in page["lines"] if is_entry_candidate(OCRLine(**raw))]
    if not candidates:
        return set()
    models: list[tuple[float, float]] = [(float(line.left), 0.0) for line in candidates]
    for first in candidates:
        for second in candidates:
            if first.top == second.top:
                continue
            slope = (second.left - first.left) / (second.top - first.top)
            if abs(slope) <= 0.10:
                models.append((first.left - slope * first.top, slope))
    best = max(
        models,
        key=lambda model: (
            sum(abs(line.left - (model[0] + model[1] * line.top)) <= 24 for line in candidates),
            -sum(min(abs(line.left - (model[0] + model[1] * line.top)), 50) for line in candidates),
        ),
    )
    selected = {
        (line.top, line.left)
        for line in candidates
        if abs(line.left - (best[0] + best[1] * line.top)) <= 30
    }
    # A torn corner is OCRed as a leading capital A before the real first head.
    for line in candidates:
        if re.match(r"A\s+vipula-\b", line.text):
            selected.add((line.top, line.left))
    return selected


def dehyphenate(lines: list[str]) -> str:
    text = ""
    for line in lines:
        line = re.sub(r"\s+", " ", line).strip(" |")
        if text.endswith("-") and line[:1].islower():
            text = text[:-1] + line
        else:
            text = f"{text} {line}".strip()
    return text


def parse_head(first_line: str) -> tuple[str, str]:
    first_line = re.sub(r"^A\s+(?=vipula-\b)", "", first_line)
    match = re.match(r"^(\S{1,60})\s+(.+)$", first_line.strip(" |."))
    if not match:
        return "", ""
    head, remainder = match.groups()
    pos = DESCRIPTOR.match(remainder)
    return normalize_headword(head), (pos.group(0).rstrip(".").lower() if pos else "")


def tags_for_pos(pos: str) -> list[str]:
    """Map Andersen's printed dictionary descriptors to canonical Jambu tags."""
    return {
        "m": ["noun", "m"],
        "f": ["noun", "f"],
        "n": ["noun", "n"],
        "a": ["adj"],
        "adv": ["adv"],
        "conj": ["conj"],
        "pron": ["pron"],
        "num": ["num"],
        "part": ["part"],
        "prep": ["prep"],
        "interj": ["interj"],
        "indec": ["indecl"],
        "vb": ["verb"],
        "pres": ["verb", "pres"],
        "aor": ["verb", "aor"],
        "pp": ["verb", "pp"],
        "cf": [],
        "": [],
    }[pos]


def extract_gloss(first_line: str) -> str:
    match = re.search(r'["\u201c]([^"\u201d]{1,240})["\u201d]', first_line)
    return re.sub(r"\s+", " ", match.group(1)).strip() if match else ""


def extract_dialects(raw_entry: str) -> list[str]:
    """Return the printed MRE findspots cited in an entry's concordance."""
    found = {MRE_SITES[match.group(1)][1] for match in SITE_PATTERN.finditer(raw_entry)}
    ordered = dict.fromkeys(label for _, label in MRE_SITES.values())
    return [name for name in ordered if name in found]


def dialect_tag(dialect: str) -> str:
    """Encode a findspot using the browser DB's canonical dialect-token schema."""
    dialect_id, label = next(
        metadata for metadata in MRE_SITES.values() if metadata[1] == dialect
    )
    return (
        f"dialect:{quote(LANGUAGE, safe='')}:{quote(dialect_id, safe='')}:"
        f"{quote(label, safe='')}"
    )


def parse_pages(pages: list[dict], cdial_index: dict[str, set[str]]) -> list[Entry]:
    entries: list[Entry] = []
    current: dict | None = None

    def finish() -> None:
        nonlocal current
        if current is None:
            return
        raw_entry = dehyphenate(current["lines"])
        headword, pos = parse_head(current["lines"][0])
        if not headword:
            current = None
            return
        gloss = GLOSS_OVERRIDES.get(headword, extract_gloss(current["lines"][0]))
        source_match = SANSKRIT_SOURCE.search(raw_entry)
        sanskrit = source_match.group(1).strip(" ,;:.") if source_match else ""
        candidates = cdial_index.get(normalize_etymon(sanskrit), set()) if sanskrit else set()
        cdial_id = next(iter(candidates)) if len(candidates) == 1 else ""
        confidence = sum(current["confidences"]) / len(current["confidences"])
        reasons = []
        if not gloss and pos != "cf":
            reasons.append("missing_gloss")
        if confidence < 75:
            reasons.append("low_ocr_confidence")
        if re.search(r"[?@|]", headword) or len(headword) > 50:
            reasons.append("suspicious_headword")
        if sanskrit and len(candidates) > 1:
            reasons.append("ambiguous_sanskrit_etymon")
        entries.append(Entry(
            pdf_page=current["pdf_page"], printed_page=current["pdf_page"] + PRINTED_PAGE_DELTA,
            top=current["top"], headword=headword, gloss=gloss, raw_entry=raw_entry,
            confidence=confidence, pos=pos, sanskrit_etymon=sanskrit,
            cdial_id=cdial_id, dialects=extract_dialects(raw_entry),
            review_reasons=reasons,
        ))
        current = None

    for page in sorted(pages, key=lambda value: value["pdf_page"]):
        starts = entry_start_keys(page)
        for raw_line in page["lines"]:
            line = OCRLine(**raw_line)
            text = line.text.strip(" |")
            if not text or text == "Dictionary" or re.fullmatch(r"\d{3}", text):
                continue
            if (line.top, line.left) in starts:
                finish()
                current = {
                    "pdf_page": page["pdf_page"], "top": line.top,
                    "lines": [text], "confidences": [line.confidence],
                }
            elif current is not None:
                current["lines"].append(text)
                current["confidences"].append(line.confidence)
    finish()
    if not any(entry.headword == "saṃyata-" for entry in entries):
        sanskrit = "saṃyatá-"
        candidates = cdial_index.get(normalize_etymon(sanskrit), set())
        entries.append(Entry(
            pdf_page=169, printed_page=171, top=1605,
            headword="saṃyata-", gloss="restrained",
            raw_entry=(
                'saṃyata- "restrained" RV saṃyatá- (saṃ + √ yam), '
                'Pāli saṃyata-, saññata -eṇā (InstrSg) 1I [Gu]'
            ),
            confidence=0, pos="a", sanskrit_etymon=sanskrit,
            cdial_id=next(iter(candidates)) if len(candidates) == 1 else "",
            dialects=["Gujarra"],
            review_reasons=["manually_recovered_scan_damage"],
        ))
        entries.sort(key=lambda entry: (entry.pdf_page, entry.top))
    return entries


def write_outputs(entries: list[Entry], output_dir: Path, install: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(entries[0])) if entries else []
    with (output_dir / "andersen_mre_entries.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for entry in entries:
            row = asdict(entry)
            row["dialects"] = ";".join(entry.dialects)
            row["review_reasons"] = ";".join(entry.review_reasons)
            writer.writerow(row)
    with (output_dir / "andersen_mre_review.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for entry in entries:
            if entry.review_reasons:
                row = asdict(entry)
                row["dialects"] = ";".join(entry.dialects)
                row["review_reasons"] = ";".join(entry.review_reasons)
                writer.writerow(row)
    import_path = output_dir / "andersen_mre_import.csv"
    with import_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for entry in entries:
            source = f"{SOURCE}[p. {entry.pdf_page} (printed p. {entry.printed_page})]"
            tags = " ".join([
                *tags_for_pos(entry.pos),
                *(dialect_tag(dialect) for dialect in entry.dialects),
            ])
            writer.writerow([
                LANGUAGE, entry.cdial_id, entry.headword, entry.gloss, "", "",
                "", source, "", "", "", "", "", "", tags,
            ])
    if install:
        shutil.copyfile(import_path, INSTALL_PATH)
        print(f"Installed {len(entries)} rows at {INSTALL_PATH}")
    print(
        f"Wrote {len(entries)} entries; "
        f"{sum(bool(entry.review_reasons) for entry in entries)} need review; "
        f"{sum(bool(entry.sanskrit_etymon) for entry in entries)} have Sanskrit comparisons; "
        f"{sum(bool(entry.cdial_id) for entry in entries)} link uniquely to CDIAL"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    pdf = args.pdf.expanduser().resolve()
    if not pdf.exists():
        parser.error(f"PDF not found: {pdf}")
    if not shutil.which("tesseract"):
        parser.error("tesseract is not installed")
    page_data = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(ocr_page, pdf, page, args.cache_dir) for page in PDF_PAGES]
        for completed, future in enumerate(as_completed(futures), 1):
            page_data.append(future.result())
            if completed % 10 == 0 or completed == len(futures):
                print(f"OCR pages: {completed}/{len(futures)}")
    entries = parse_pages(page_data, load_cdial_index())
    write_outputs(entries, args.output_dir, args.install)


if __name__ == "__main__":
    main()
