"""Extract Table B.1 of Kumari et al. (2026), A Grammar of Gaddi.

The PDF is a local, non-committed input. Extraction uses its selectable text;
no OCR or linguistic reconstruction is performed here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[4]
WORKSPACE = DATA.parent
PDF = WORKSPACE / "tmp/pdfs/gaddi-2026/grammar.pdf"
PDF_SHA256 = "e4ad8cf8a452bee58f609c3c5e015a0446fa91f77f72a065b36f5a9633fb48b8"
SOURCE = "kumari2026gaddi"
CSV = DATA / "data/other/forms/20260925-kumari-gaddi.csv"
AUDIT = HERE / "audit.jsonl"
PATTERN = re.compile(r"(?m)^\s*(\d{1,3}(?:\.\d)?)\s+(\[[^\]]+\])\s+([^\n]+)")
BROKEN_SPACES = re.compile(r"(?<=[a-zɑəɛɔɡʃʒʈɖɳɭɽʋ])\s+(?=ʰ)")
BROKEN_MARK_SPACE = re.compile(r"(?<=ː)\s+(?=̃)")
FLAGS = {
    "44": "transcription: source has combining tilde after long mark",
    "88": "transcription: source has voiceless diacritic on vowel",
    "132": "transcription: source prints colon instead of IPA length sign",
    "155.3": "structure: split is grouped with spit under prompt 155",
    "207.2": "gloss: printed form also occurs for rain at item 121",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(pdf: Path = PDF) -> list[dict]:
    from pypdf import PdfReader
    if not pdf.exists():
        raise FileNotFoundError(f"Required Gaddi PDF missing: {pdf}")
    if sha256(pdf) != PDF_SHA256:
        raise ValueError("Gaddi PDF hash differs from pinned UCL Press copy")
    document = PdfReader(pdf)
    if len(document.pages) != 152:
        raise ValueError(f"Expected 152 PDF pages, got {len(document.pages)}")
    rows: list[dict] = []
    for pdf_page in range(136, 142):
        page = document.pages[pdf_page - 1]
        page_text = page.extract_text()
        for match in PATTERN.finditer(page_text):
            item, bracketed, gloss = match.groups()
            raw = bracketed[1:-1]
            repairs: list[str] = []
            if "\n" in raw:
                repairs.append("joined PDF-wrapped glyph inside IPA brackets")
            if BROKEN_SPACES.search(raw):
                repairs.append("joined PDF-spaced aspiration")
            if BROKEN_MARK_SPACE.search(raw):
                repairs.append("joined PDF-spaced combining tilde")
            # Wrapped glyphs inside brackets have no phonemic word break; real
            # spaces in the printed multiword forms remain untouched.
            form = unicodedata.normalize("NFC", BROKEN_SPACES.sub("", BROKEN_MARK_SPACE.sub("", raw.replace("\n", "").strip())))
            if item == "73":
                # pypdf alone separates the last letter of "heart" into its
                # own text line. The rendered PDF and PyMuPDF both show heart.
                if gloss != "hear" or "\n73 [dil] hear\nt\n74 " not in page_text:
                    raise ValueError("Unexpected PDF layout at item 73")
                gloss = "heart"
                repairs.append("rejoined detached final t of printed gloss heart")
            if not form or not gloss:
                raise ValueError(f"Empty row at PDF p.{pdf_page}, item {item}")
            rows.append(
                {
                    "entry_key": f"kumari-gaddi:B1:{item}",
                    "item": item,
                    "pdf_page": pdf_page,
                    "printed_page": pdf_page - 17,
                    "raw_bracketed": bracketed,
                    "raw_line": match.group(0).strip(),
                    "extraction_repairs": repairs,
                    "form": form,
                    "source_gloss": gloss.strip(),
                    "uncertainty": FLAGS.get(item, ""),
                }
            )
    items = {r["item"] for r in rows}
    if len(rows) != 254 or len(items) != 254:
        raise ValueError(f"Expected 254 distinct B.1 attestations, got {len(rows)} rows/{len(items)} keys")
    if {int(i.split(".")[0]) for i in items} != set(range(1, 222)):
        raise ValueError("Table B.1 prompt numbers 1–221 are incomplete")
    return rows


def read_appendix(pdf: Path = PDF) -> list[dict]:
    """Read all four Appendix III tables; coordinates preserve the two columns."""
    import pdfplumber
    from collections import Counter
    if sha256(pdf) != PDF_SHA256:
        raise ValueError("Gaddi PDF hash differs from pinned UCL Press copy")
    rows = []
    counts = Counter()
    table = ""
    with pdfplumber.open(pdf) as document:
        for pdf_page in range(142, 146):
            text = document.pages[pdf_page - 1].extract_text()
            for line in text.splitlines():
                if line.startswith("Table C."):
                    table = line.split()[1]
                    continue
                if line.startswith(("Gloss ", "Appendix ", "APPeNDIx", "126 ", "128 ", "1 The ", "research ", "Mamta ")):
                    continue
                # Each data line has an English capitalized prompt followed by
                # lowercase IPA. Spaces within distributive phrases are real.
                match = re.fullmatch(r"([A-Z][A-Za-z -]*?) ([^A-Z].*)", line)
                if not match:
                    raise ValueError(f"Unaccounted Appendix III line: {line!r}")
                gloss, raw = match.groups()
                if table == "C.4":
                    # English prompts have spaces; never treat their suffix as
                    # part of the IPA language column.
                    gloss = ("One each", "Three each", "Five each", "One by one", "Two by two")[counts[table]]
                    if not line.startswith(gloss + " "):
                        raise ValueError(f"Changed distributive prompt: {line}")
                    raw = line[len(gloss) + 1:]
                counts[table] += 1
                item = str(counts[table])
                form = BROKEN_SPACES.sub("", raw)
                repairs = []
                if form != raw:
                    repairs.append("joined PDF-spaced aspiration")
                # pdfplumber's geometric sort places the tilde after the next
                # consonant. Both are checked against the printed page 126.
                if table == "C.1" and gloss in {"Forty-five", "Sixty-five"}:
                    expected = {"Forty-five": ("pɛt̃ɑli", "pɛ̃tɑli"), "Sixty-five": ("pɛʈ̃ʰ", "pɛ̃ʈʰ")}[gloss]
                    if form != expected[0]:
                        raise ValueError(f"Changed positioned-tilde extraction: {gloss} {form}")
                    form = expected[1]
                    repairs.append("restored visually verified tilde to preceding vowel after coordinate sorting")
                uncertainty = ""
                if table == "C.3" and gloss == "Quarter":
                    uncertainty = "transcription: printed colon retained rather than interpreted as IPA length"
                if table == "C.3" and gloss in {"Tenth", "Sixteenth"}:
                    uncertainty = "gloss: source pairs tenth with solɑ and sixteenth with dəs; printed pairing preserved without swapping"
                if table == "C.3" and gloss == "Three-quarters":
                    uncertainty = "transcription: printed Latin c in coutʰɑ retained, not expanded to tʃ"
                rows.append({
                    "entry_key": f"kumari-gaddi:{table.replace('.', '')}:{item}",
                    "item": item, "table": table, "pdf_page": pdf_page,
                    "printed_page": pdf_page - 17, "raw_line": line,
                    "raw_form": raw, "form": unicodedata.normalize("NFC", form),
                    "source_gloss": gloss.lower(), "extraction_repairs": repairs,
                    "uncertainty": uncertainty,
                    "numeral_type": {"C.1":"cardinal", "C.2":"ordinal", "C.3":"fractional", "C.4":"distributive"}[table],
                    "collector": "Kumari Mamta; doctoral numeral research, credited in Appendix III footnote 1 (p. 125)",
                    "excluded": form == "–",
                    "reason": "Source prints a dash, no form supplied" if form == "–" else "Printed source attestation",
                })
    if dict(counts) != {"C.1":105, "C.2":22, "C.3":7, "C.4":5}:
        raise ValueError(f"Incomplete numeral appendix: {counts}")
    return rows


def analyse(row: dict) -> tuple[str, str]:
    gloss = row["source_gloss"]
    tags: list[str] = (["num", "ord"] if row.get("table") == "C.2" else ["num"]) if row.get("table") else []
    if gloss.endswith(" (ma)"):
        gloss, tags = gloss[:-5], ["m"]
    elif gloss.endswith(" (fe)"):
        gloss, tags = gloss[:-5], ["f"]
    elif gloss.endswith(" (noun)"):
        gloss, tags = gloss[:-7], ["noun"]
    elif gloss.endswith(" (verb)"):
        gloss, tags = gloss[:-7], ["verb"]
    if row["uncertainty"]:
        tags.append("uncertain")
    return gloss, " ".join(tags)


def output(rows: list[dict]) -> tuple[list[list[str]], list[dict]]:
    csv_rows: list[list[str]] = []
    audits: list[dict] = []
    for row in rows:
        gloss, tags = analyse(row)
        citation = f"{SOURCE}[p. {row['printed_page']}, table {row.get('table', 'B.1')}, item {row['item']}]"
        fields = [
            "ga", "", row["form"], gloss, "", "", "", citation,
            "", "", row["entry_key"], "", "", "", tags,
        ]
        if not row.get("excluded", False):
            csv_rows.append(fields)
        audits.append(
            {
                **row,
                "status": "excluded_blank" if row.get("excluded", False) else "ingested",
                "language_id": "ga",
                "source": citation,
                "gloss": gloss,
                "tags": tags,
                "excluded": row.get("excluded", False),
            }
        )
    return csv_rows, audits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, default=PDF)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    rows = read_rows(args.pdf) + read_appendix(args.pdf)
    csv_rows, audits = output(rows)
    AUDIT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in audits))
    if args.install:
        with CSV.open("w", newline="") as handle:
            csv.writer(handle).writerows(csv_rows)
    print(f"Tables B.1 and C.1–C.4: {len(rows)} raw rows, {len(csv_rows)} proposed rows; audit: {AUDIT}")
    if args.install:
        print(f"Installed: {CSV}")


if __name__ == "__main__":
    main()
