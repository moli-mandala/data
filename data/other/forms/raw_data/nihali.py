#!/usr/bin/env python3
"""Reconstruct the superseded Mundlay OCR and Nagaraja/Wiktionary imports.

Mundlay's PDF embeds a lossy text layer which drops most diacritics.  The
importer therefore consumes per-page, per-column Tesseract output and labels
every resulting record as unreviewed OCR.  ``--prepare-mundlay`` can reproduce
those files from the PDF when Ghostscript, Pillow, and Tesseract's Latin model
are installed.

The Wiktionary input is a pinned raw-wikitext snapshot.  Its 1,694 table rows
are derived from Nagaraja (2014: 250--332); raw markup and source-attribution
labels are retained in the audit sidecar.

The reviewed ``nihali_database.py`` import is canonical.  To prevent an
accidental rollback, this historical tool writes only explicitly named legacy
preview files under ``raw_data``.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import re
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parents[3]
FORMS = DATA_ROOT / "data/other/forms"
DEFAULT_MUNDLAY_OUTPUT = HERE / "superseded-20260817-mundlay-nihali.csv"
DEFAULT_MUNDLAY_AUDIT = HERE / "superseded-20260817-mundlay-nihali-audit.csv"
DEFAULT_WIKTIONARY_OUTPUT = HERE / "superseded-20260817-nagaraja-nihali-wiktionary.csv"
DEFAULT_WIKTIONARY_AUDIT = HERE / "superseded-20260817-nagaraja-nihali-wiktionary-audit.csv"

WIKTIONARY_REVISION = 88143027
WIKTIONARY_TIMESTAMP = "2025-11-12T18:53:49Z"
MUNDLAY_STARTS = [
    1, 21, 41, 60, 83, 108, 135, 167, 196, 224, 251, 272,
    302, 326, 356, 392, 423, 452, 476, 513, 548, 580, 619, 655,
    687, 726, 757, 797, 834, 884, 927, 968, 1007, 1046, 1089,
    1133, 1171, 1209, 1249, 1290, 1334, 1376, 1419, 1464, 1512,
    1553, 1597, 1637,
]
# These labels are absent in the printed lexicon (not merely missed by OCR).
MUNDLAY_PRINTED_GAPS = {34, 421, *range(845, 855), 1101}
MUNDLAY_INSERTIONS = {123: "123/2", 654: "654a"}

RICH_COLUMNS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic",
    "Notes", "Source", "Cognateset", "Etymology", "Entry_Key",
    "Variant_Of_Key", "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]


def compact(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value)).strip()


def strip_wiki(value: str) -> str:
    value = re.sub(r"\[\[[^]|]+\|([^]]+)\]\]", r"\1", value)
    value = re.sub(r"\[\[([^]]+)\]\]", r"\1", value)
    value = re.sub(r"<[^>]+>", "", value)
    value = re.sub(r"\{\{[^{}]*\}\}", "", value)
    return compact(value).replace("'''", "").replace("''", "")


def split_top_level_commas(value: str) -> list[str]:
    parts, current, depth = [], [], 0
    for char in value:
        if char == "(":
            depth += 1
        elif char == ")" and depth:
            depth -= 1
        if char in ",;" and depth == 0:
            parts.append(compact("".join(current)))
            current = []
        else:
            current.append(char)
    parts.append(compact("".join(current)))
    return [part for part in parts if part]


def clean_wiktionary_form(value: str) -> tuple[str, str]:
    value = strip_wiki(value)
    notes = []
    # Parenthesized prose follows the headword; parentheses embedded in a
    # headword are retained because they encode optional source segments.
    parenthesis = value.find(" (")
    if parenthesis >= 0:
        notes.append(value[parenthesis + 1 :])
        value = value[:parenthesis].strip()
    quote_annotation = re.search(r"\s+[‘ʻ][^’]+[’]?\s*$", value)
    if quote_annotation:
        notes.append(quote_annotation.group(0).strip())
        value = value[: quote_annotation.start()].strip()
    match = re.search(r"\s+(<.+)$", value)
    if match:
        notes.append(match.group(1))
        value = value[: match.start()].strip()
    return value.strip(" .;"), "; ".join(notes)


def parse_wiktionary(path: Path) -> list[dict[str, str]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.startswith("|") or line.startswith("|-"):
            continue
        cells = [cell.strip() for cell in line[1:].split("||")]
        if len(cells) < 3:
            continue
        # One upstream row contains an accidental cell separator inside its
        # explanatory parenthesis: ``(-ku ‘pl.’ in || Korku)``.
        if "(-ku" in cells[1] and len(cells) == 4:
            cells = [cells[0], cells[1].split("(-ku", 1)[0].strip(), "", cells[3]]
        if len(cells) != 4:
            raise ValueError(f"unexpected Wiktionary row at line {line_number}: {cells!r}")
        gloss_raw, form_raw, origin_raw, page_raw = cells
        if not form_raw:
            continue
        rows.append(
            {
                "line": str(line_number),
                "gloss_raw": gloss_raw,
                "form_raw": form_raw,
                "origin_raw": origin_raw,
                "page_raw": page_raw,
                "gloss": strip_wiki(gloss_raw),
                "origin": strip_wiki(origin_raw),
            }
        )
    return rows


def build_wiktionary(path: Path, output: Path, audit: Path) -> tuple[int, int]:
    source_rows = parse_wiktionary(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("w", encoding="utf-8", newline="") as forms, audit.open(
        "w", encoding="utf-8", newline=""
    ) as audits:
        writer = csv.writer(forms)
        audit_fields = [
            "Status", "Entry_Key", "Snapshot_Revision", "Snapshot_Timestamp",
            "Wikitext_Line", "Raw_Gloss", "Gloss", "Raw_Form", "Form",
            "Raw_Origin", "Origin", "Raw_Page_Marker", "Variant_Number",
        ]
        aw = csv.DictWriter(audits, fieldnames=audit_fields)
        aw.writeheader()
        for source_index, row in enumerate(source_rows, 1):
            raw_variants = split_top_level_commas(row["form_raw"])
            base_key = f"nagaraja2014-wiktionary:{source_index}"
            for variant_number, raw_variant in enumerate(raw_variants, 1):
                form, form_note = clean_wiktionary_form(raw_variant)
                key = base_key if variant_number == 1 else f"{base_key}:v{variant_number}"
                status = "ingested" if form else "skipped_blank_form"
                tags = []
                if row["origin"]:
                    tags.append("loanword")
                if "?" in row["origin"]:
                    tags.append("uncertain")
                notes = [
                    f"Wiktionary snapshot revision {WIKTIONARY_REVISION}; derived from Nagaraja (2014: 250–332)"
                ]
                if form_note:
                    notes.append(f"source-form annotation: {form_note}")
                if row["page_raw"]:
                    notes.append(f"Wiktionary page marker: {row['page_raw']}")
                if form:
                    writer.writerow(
                        [
                            "Ni", "", form, row["gloss"], "", raw_variant,
                            "; ".join(notes),
                            f"nagaraja2014[pp. 250–332];wiktionary-nihali[revision {WIKTIONARY_REVISION}, row {source_index}]",
                            "", (
                                f"Nagaraja/Wiktionary source attribution: {row['origin']}"
                                if row["origin"] else ""
                            ), key, base_key if variant_number > 1 else "", "", "", "; ".join(tags),
                        ]
                    )
                    written += 1
                aw.writerow(
                    {
                        "Status": status,
                        "Entry_Key": key,
                        "Snapshot_Revision": WIKTIONARY_REVISION,
                        "Snapshot_Timestamp": WIKTIONARY_TIMESTAMP,
                        "Wikitext_Line": row["line"],
                        "Raw_Gloss": row["gloss_raw"],
                        "Gloss": row["gloss"],
                        "Raw_Form": raw_variant,
                        "Form": form,
                        "Raw_Origin": row["origin_raw"],
                        "Origin": row["origin"],
                        "Raw_Page_Marker": row["page_raw"],
                        "Variant_Number": variant_number,
                    }
                )
    return len(source_rows), written


@dataclass
class MundlayEntry:
    source_label: str
    number: int
    printed_page: int
    column: int
    raw_label: str
    raw_text: str
    form: str
    gloss: str
    pos: str
    loan: bool
    uncertain: bool

    @property
    def key(self) -> str:
        return f"mundlay1996:{self.source_label}"


def label_digits(raw: str) -> str:
    translated = raw.translate(str.maketrans({"I": "1", "l": "1", "i": "1", "]": "1", "T": "7"}))
    return "".join(re.findall(r"\d", translated))


def plausible_label(raw: str, expected: int) -> bool:
    if expected == 77 and raw.upper().endswith("T1"):
        return True
    digits = label_digits(raw)
    target = str(expected)
    if not digits:
        return False
    if target in digits or digits in target:
        return True
    return difflib.SequenceMatcher(None, digits, target).ratio() >= 0.60


def looks_like_ocr_label(raw: str) -> bool:
    raw = raw.rstrip(".,")
    return bool(re.fullmatch(r"[?ŁLlMC/0-9IiT\]]+(?:/[0-9]+)?a?", raw))


def split_mundlay_blocks(text: str, start: int, end: int) -> list[tuple[str, str, str]]:
    """Return (canonical label, raw label, OCR block) for one source column."""
    lines = text.splitlines()
    blocks: list[tuple[str, str, list[str]]] = []
    expected = start
    current: tuple[str, str, list[str]] | None = None
    insertion_done = False
    for line in lines:
        match = re.match(r"^\s*(\S{1,18}[.,])\s+(.*)$", line)
        accepted: tuple[str, int] | None = None
        if match and looks_like_ocr_label(match.group(1)):
            raw_label = match.group(1).rstrip(".,")
            while expected in MUNDLAY_PRINTED_GAPS:
                expected += 1
            insertion = MUNDLAY_INSERTIONS.get(expected - 1)
            if insertion and not insertion_done and insertion.replace("a", "") in raw_label:
                accepted = (insertion, expected - 1)
                insertion_done = True
            elif expected <= end and plausible_label(raw_label, expected):
                accepted = (str(expected), expected)
                expected += 1
                insertion_done = False
        if accepted:
            if current:
                blocks.append(current)
            current = (accepted[0], match.group(1).rstrip(".,"), [match.group(2)])
        elif current and compact(line):
            current[2].append(compact(line))
    if current:
        blocks.append(current)
    expected_numbers = [n for n in range(start, end + 1) if n not in MUNDLAY_PRINTED_GAPS]
    found_numbers = [int(label) for label, _, _ in blocks if label.isdigit()]
    if found_numbers != expected_numbers:
        missing = sorted(set(expected_numbers) - set(found_numbers))
        raise ValueError(f"Mundlay column {start}-{end}: missing {missing[:12]} (found {len(blocks)} blocks)")
    return [(label, raw_label, compact(" ".join(body))) for label, raw_label, body in blocks]


def parse_mundlay_body(body: str) -> tuple[str, str, str]:
    if body.startswith(("‘", "ʻ", "\"", "“")):
        quote = re.search(r"[‘ʻ\"]([^’\"”]+)[’\"”]", body)
        return "", compact(quote.group(1)) if quote else "", ""
    quote_start = min(
        (index for index in (body.find(" ‘"), body.find(" ʻ"), body.find(' "')) if index >= 0),
        default=-1,
    )
    see_match = re.match(r"(\S+?)(?:[.,]?\s+see\b)", body, re.I)
    candidate_pos = None if see_match else re.match(r"(.+?)\s*\(([^)}]{1,35})[)}]", body)
    pos_match = candidate_pos if candidate_pos and (quote_start < 0 or candidate_pos.end() <= quote_start) else None
    if pos_match:
        form = pos_match.group(1)
        pos = compact(pos_match.group(2))
    else:
        form = see_match.group(1) if see_match else body[:quote_start] if quote_start >= 0 else body.split(" ", 1)[0]
        pos = ""
    form = compact(form).strip(" .,;:‘’ʻ\"“”")
    quote = re.search(r"[‘ʻ\"]([^’\"”]+)[’\"”]", body)
    gloss = compact(quote.group(1)) if quote else ""
    return form, gloss, pos


def parse_mundlay(ocr_dir: Path) -> list[MundlayEntry]:
    paths = sorted(ocr_dir.glob("page-??-c?.txt"))
    if len(paths) != 48:
        raise ValueError(f"expected 48 Mundlay OCR columns, found {len(paths)} in {ocr_dir}")
    entries = []
    for index, path in enumerate(paths):
        start = MUNDLAY_STARTS[index]
        end = MUNDLAY_STARTS[index + 1] - 1 if index + 1 < len(paths) else 1660
        blocks = split_mundlay_blocks(path.read_text(encoding="utf-8"), start, end)
        for label, raw_label, body in blocks:
            form, gloss, pos = parse_mundlay_body(body)
            prefix = raw_label[: raw_label.find(label_digits(raw_label))] if label_digits(raw_label) else ""
            entries.append(
                MundlayEntry(
                    label, int(re.match(r"\d+", label).group()), 17 + index // 2,
                    1 + index % 2, raw_label, body, form, gloss, pos,
                    "L" in prefix.upper() or "Ł" in prefix.upper(), "?" in prefix,
                )
            )
    return entries


def build_mundlay(ocr_dir: Path, output: Path, audit: Path) -> tuple[int, int]:
    entries = parse_mundlay(ocr_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as forms, audit.open(
        "w", encoding="utf-8", newline=""
    ) as audits:
        writer = csv.writer(forms)
        fields = [
            "Status", "Entry_Key", "Printed_Page", "Column", "Source_Label",
            "Raw_OCR_Label", "Raw_OCR_Block", "Form", "Gloss", "POS", "Loan", "Uncertain",
        ]
        aw = csv.DictWriter(audits, fieldnames=fields)
        aw.writeheader()
        written = 0
        for entry in entries:
            valid = bool(entry.form and re.search(r"[A-Za-zÀ-ž]", entry.form))
            if valid:
                tags = [tag for tag, active in (("loanword", entry.loan), ("uncertain", entry.uncertain)) if active]
                if entry.pos:
                    tags.append(entry.pos)
                writer.writerow(
                    [
                        "Ni", "", entry.form, entry.gloss, "", entry.form,
                        "Unreviewed OCR transcription; consult the source scan before analysis",
                        f"mundlay1996[p. {entry.printed_page}, col. {entry.column}, entry {entry.source_label}]",
                        "", "", entry.key, "", "", "", "; ".join(tags),
                    ]
                )
                written += 1
            aw.writerow(
                {
                    "Status": "ocr_unreviewed" if valid else "skipped_blank_or_illegible_form",
                    "Entry_Key": entry.key,
                    "Printed_Page": entry.printed_page,
                    "Column": entry.column,
                    "Source_Label": entry.source_label,
                    "Raw_OCR_Label": entry.raw_label,
                    "Raw_OCR_Block": entry.raw_text,
                    "Form": entry.form,
                    "Gloss": entry.gloss,
                    "POS": entry.pos,
                    "Loan": "yes" if entry.loan else "no",
                    "Uncertain": "yes" if entry.uncertain else "no",
                }
            )
    return len(entries), written


def prepare_mundlay(pdf: Path, ocr_dir: Path) -> None:
    from PIL import Image

    ocr_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mundlay-pages-") as temporary:
        pattern = str(Path(temporary) / "page-%02d.png")
        subprocess.run(
            [
                "gs", "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=pnggray", "-r400",
                "-dFirstPage=19", "-dLastPage=42", f"-sOutputFile={pattern}", str(pdf),
            ], check=True,
        )
        for page_number, page_path in enumerate(sorted(Path(temporary).glob("page-*.png")), 1):
            image = Image.open(page_path)
            for column, (left, right) in enumerate(((0.08, 0.49), (0.51, 0.92)), 1):
                crop = image.crop((int(image.width * left), int(image.height * .10), int(image.width * right), int(image.height * .91)))
                base = ocr_dir / f"page-{page_number:02d}-c{column}"
                crop.save(base.with_suffix(".png"))
                subprocess.run(["tesseract", str(base.with_suffix(".png")), str(base), "-l", "script/Latin", "--psm", "6"], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiktionary", type=Path)
    parser.add_argument("--mundlay-pdf", type=Path)
    parser.add_argument("--mundlay-ocr-dir", type=Path)
    parser.add_argument("--prepare-mundlay", action="store_true")
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    if args.prepare_mundlay:
        if not args.mundlay_pdf or not args.mundlay_ocr_dir:
            parser.error("--prepare-mundlay requires --mundlay-pdf and --mundlay-ocr-dir")
        prepare_mundlay(args.mundlay_pdf, args.mundlay_ocr_dir)
    if args.wiktionary:
        source_count, written = build_wiktionary(
            args.wiktionary, DEFAULT_WIKTIONARY_OUTPUT,
            DEFAULT_WIKTIONARY_AUDIT,
        )
        print(f"Wiktionary source rows: {source_count}; wrote: {written}")
    if args.mundlay_ocr_dir:
        source_count, written = build_mundlay(
            args.mundlay_ocr_dir, DEFAULT_MUNDLAY_OUTPUT,
            DEFAULT_MUNDLAY_AUDIT,
        )
        print(f"Mundlay source entries: {source_count}; wrote: {written}")


if __name__ == "__main__":
    main()
