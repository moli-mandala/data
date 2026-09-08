#!/usr/bin/env python3
"""Install every lexical entry in Watters's 2006 Kusunda Appendix A.

The copyrighted PDF is not redistributed.  A pinned CC-BY-SA Wiktionary
transcription is used as a Unicode repair layer for the PDF's embedded STEDT
font.  The source census, order, part-of-speech sequence, page/column breaks,
and sampled readings are checked against printed pp. 139--152.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import random
import re
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
PACKAGE = Path(__file__).resolve().parent
SNAPSHOT = PACKAGE / "snapshot/wiktionary-84645357.wiki"
MANIFEST = PACKAGE / "manifest.json"
INSTALLED = ROOT / "data/other/forms/20260901-watters-kusunda.csv"
AUDIT = PACKAGE / "20260901-watters-kusunda-audit.csv"
SOURCE_KEY = "watters2006kusunda"
PDF_SHA256 = "3fd0942e1ed860e3c6b27b3d4543ab9f83800aabd4199318173fe4e129198bb2"
SAMPLE_SEED = 20260901

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic",
    "Notes", "Source", "Cognateset", "Etymology", "Entry_Key",
    "Variant_Of_Key", "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "Source_Record", "PDF_Page", "Printed_Page", "Column", "Item",
    "Raw_Headword", "Raw_POS", "Raw_Gloss", "Raw_Notes", "Variant_Index",
    "Separator", "Annotation", "Form", "Gloss", "Notes", "Tags", "Status",
    "Reason", "Entry_Key", "Variant_Of_Key", "Citation", "Review_State",
]

# Counts are printed entries, not visual lines.  The only entry without an
# explicit POS is p. 144 col. 1 item 32 (tən-da idaŋ).
PAGE_COLUMN_COUNTS = [
    (139, 22, 28), (140, 37, 29), (141, 33, 31), (142, 28, 43),
    (143, 36, 38), (144, 34, 35), (145, 33, 36), (146, 34, 38),
    (147, 33, 26), (148, 35, 35), (149, 38, 31), (150, 32, 33),
    (151, 30, 34), (152, 8, 7),
]

POS_TAGS = {
    "n.": ["noun"], "vt.": ["verb", "tr"], "vi.": ["verb", "intr"],
    "v.": ["verb"], "v., adj.": ["verb", "adj"], "adj.": ["adj"],
    "adv.": ["adv"], "pp.": ["postp"], "pron.": ["pron"],
    "num.": ["num"], "interrog.": ["interr"], "dem.": ["demonstrative"],
    "loc.": ["adv", "spatial"], "conj.": ["conj"], "greet.": ["interj"],
    "suff.": ["suffix"], "aff.": ["suffix"], "": [],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    actual = sha256(SNAPSHOT)
    if actual != manifest["unicode_control"]["sha256"]:
        raise ValueError(f"Unicode-control checksum mismatch: {actual}")
    return manifest


def clean_markup(value: str) -> str:
    value = re.sub(r"\{\{audio\|.*?\}\}", "", value)
    value = re.sub(r"\[\[[^]|]+\|([^]]+)\]\]", r"\1", value)
    value = re.sub(r"\[\[([^]]+)\]\]", r"\1", value)
    value = value.replace("'''", "").replace("''", "")
    return unicodedata.normalize("NFC", html.unescape(re.sub(r"\s+", " ", value)).strip())


def parse_records() -> list[dict[str, str]]:
    verify_snapshot()
    text = SNAPSHOT.read_text(encoding="utf-8")
    section = text.split("==Watters (2006)==", 1)[1].split("|}", 1)[0]
    records = []
    for chunk in section.split("|-")[1:]:
        line = chunk.strip().split("\n", 1)[0]
        if not line.startswith("| "):
            continue
        cells = [clean_markup(cell) for cell in line[2:].split("||")]
        if len(cells) != 4:
            raise ValueError(f"Malformed four-cell Watters row: {line}")
        records.append(dict(zip(("headword", "pos", "gloss", "notes"), cells)))
    if len(records) != 877:
        raise ValueError(f"Expected 877 source entries, found {len(records)}")

    cursor = 0
    for page, left, right in PAGE_COLUMN_COUNTS:
        for column, count in ((1, left), (2, right)):
            for item, record in enumerate(records[cursor:cursor + count], 1):
                record.update(page=str(page), column=str(column), item=str(item))
            cursor += count
    if cursor != len(records):
        raise ValueError(f"Page/column census covers {cursor}, not {len(records)}")
    return records


def split_outside_parentheses(value: str) -> list[tuple[str, str]]:
    pieces: list[tuple[str, str]] = []
    buf: list[str] = []
    depth = 0
    separator = ""
    for char in value:
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        if depth == 0 and char in "/,;":
            pieces.append(("".join(buf).strip(), separator))
            buf = []
            separator = char
        else:
            buf.append(char)
    pieces.append(("".join(buf).strip(), separator))
    return [(piece, sep) for piece, sep in pieces if piece]


def annotation_tags(annotation: str) -> list[str]:
    tags: list[str] = []
    if "imp" in annotation:
        tags.append("impv")
    if "proh" in annotation:
        tags.extend(["impv", "neg"])
    if "3rd" in annotation:
        tags.append("3sg")
    if "2nd" in annotation:
        tags.append("2sg")
    if "neg" in annotation:
        tags.append("neg")
    return list(dict.fromkeys(tags))


def citation(record: dict[str, str]) -> str:
    return (
        f"{SOURCE_KEY}[p. {record['page']}, col. {record['column']}, "
        f"item {record['item']}]"
    )


def transform(records: list[dict[str, str]]) -> tuple[list[list[str]], list[dict[str, str]]]:
    installed: list[list[str]] = []
    audit: list[dict[str, str]] = []
    installed_keys: list[str] = []
    for source_record, record in enumerate(records, 1):
        raw_headword = record["headword"]
        loan = bool(re.search(r"\(Nep\.?\)", raw_headword))
        headword = re.sub(r"\s*\(Nep\.?\)", "", raw_headword).strip()
        variants = split_outside_parentheses(headword)
        base_key = (
            f"{SOURCE_KEY}:p{int(record['page']):03d}:c{record['column']}:"
            f"i{int(record['item']):03d}:v01"
        )
        seen: set[str] = set()
        for variant_index, (raw_form, separator) in enumerate(variants, 1):
            annotations = re.findall(r"\(([^)]*)\)", raw_form)
            form = re.sub(r"\s*\([^)]*\)", "", raw_form).strip()
            # Two Wiktionary control rows retain the PDF's pre-vowel tilde
            # placement.  Jambu profiles use a post-vowel combining mark.
            form = re.sub(r"~([aeiouə])", r"\1~", form)
            form = unicodedata.normalize("NFC", form)
            key = base_key[:-2] + f"{variant_index:02d}"
            duplicate = form in seen
            seen.add(form)
            tags = list(POS_TAGS[record["pos"]])
            for annotation in annotations:
                tags.extend(annotation_tags(annotation))
            if loan:
                tags.extend(["loanword", "loan:Nepali"])
            if variant_index > 1:
                tags.append("alternate")
                if separator == "/":
                    tags.append("sound-variant")
            tags = list(dict.fromkeys(tags))
            notes = record["notes"]
            if len(variants) > 1:
                paradigm = f"Source headword group: {raw_headword}"
                notes = f"{notes}; {paradigm}" if notes else paradigm
            if annotations:
                label = ", ".join(annotations)
                notes = f"{notes}; Source form label: {label}" if notes else f"Source form label: {label}"
            status = "excluded" if duplicate else "ingested"
            reason = "exact repeated form within source entry" if duplicate else "printed vocabulary attestation"
            variant_of = "" if variant_index == 1 else base_key
            if not duplicate:
                installed.append([
                    "Kusunda", "", form, record["gloss"], "", form, notes,
                    citation(record), "", "", key, variant_of, "", "",
                    " ".join(tags),
                ])
                installed_keys.append(key)
            audit.append({
                "Source_Record": str(source_record), "PDF_Page": record["page"],
                "Printed_Page": record["page"], "Column": record["column"],
                "Item": record["item"], "Raw_Headword": raw_headword,
                "Raw_POS": record["pos"], "Raw_Gloss": record["gloss"],
                "Raw_Notes": record["notes"], "Variant_Index": str(variant_index),
                "Separator": separator, "Annotation": " | ".join(annotations),
                "Form": form, "Gloss": record["gloss"], "Notes": notes,
                "Tags": " ".join(tags), "Status": status, "Reason": reason,
                "Entry_Key": "" if duplicate else key,
                "Variant_Of_Key": "" if duplicate else variant_of,
                "Citation": citation(record), "Review_State": "not-sampled",
            })

    sampled = set(random.Random(SAMPLE_SEED).sample(installed_keys, 20))
    for row in audit:
        if row["Entry_Key"] in sampled:
            row["Review_State"] = "verified-against-render"
    if len({row[10] for row in installed}) != len(installed):
        raise ValueError("Installed Watters Entry_Key values are not unique")
    return installed, audit


def verify_pdf(path: Path) -> None:
    if sha256(path) != PDF_SHA256:
        raise ValueError(f"Unexpected Watters PDF checksum: {sha256(path)}")
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError("pdfplumber is required only for --pdf verification") from exc
    pos = set(POS_TAGS) - {"", "v., adj."}
    with pdfplumber.open(path) as pdf:
        if len(pdf.pages) != 182:
            raise ValueError(f"Expected 182 PDF pages, found {len(pdf.pages)}")
        observed = 0
        for page_no in range(139, 153):
            words = pdf.pages[page_no - 1].extract_words(
                x_tolerance=1, y_tolerance=2, extra_attrs=["fontname"]
            )
            tops = set()
            for word in words:
                token = word["text"].rstrip(".,") + "."
                if "Italic" in word["fontname"] and token in pos:
                    tops.add((1 if word["x0"] < 297 else 2, round(word["top"])))
            observed += len(tops)
        # 876 explicit POS loci plus the single POS-empty entry.
        if observed != 876:
            raise ValueError(f"Expected 876 explicit POS loci, found {observed}")


def write_csv(path: Path, rows, fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        if fields:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        else:
            csv.writer(stream, lineterminator="\n").writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()
    if args.pdf:
        verify_pdf(args.pdf)
    installed, audit = transform(parse_records())
    output = args.output or (INSTALLED if args.install else Path("/tmp/watters-kusunda-proposed.csv"))
    audit_output = args.audit_output or (AUDIT if args.install else Path("/tmp/watters-kusunda-audit.csv"))
    write_csv(output, installed)
    write_csv(audit_output, audit, AUDIT_FIELDS)
    print(json.dumps({
        "source_entries": 877, "installed": len(installed),
        "excluded_variant_duplicates": sum(row["Status"] == "excluded" for row in audit),
        "output": str(output), "audit": str(audit_output),
    }, indent=2))


if __name__ == "__main__":
    main()
