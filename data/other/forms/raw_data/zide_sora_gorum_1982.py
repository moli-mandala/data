#!/usr/bin/env python3
"""Import Zide's structured Sora–Juray comparison records from SEAlang.

The result page is semantic born-digital HTML, not OCR.  It exposes attested
Sora and Juray forms grouped by stable source locator.  The SEAlang index does
not expose protoforms, so comparison group IDs are retained in ``Cognateset``
while Parameter_ID remains blank; no reconstruction or ancestry is invented.
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
import urllib.request
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path


SOURCE_KEY = "zide1982reconstruction"
SOURCE_URL = (
    "http://sealang.net/munda/dictionary/search.pl?"
    "caller=database&include=zide1982reconstruction"
)
SNAPSHOT_DATE = "2026-08-28"
SOURCE_SHA256 = "3b81fcc719b57434ee50733914bacbc33ed497f283fdfdf4a79d64073f31af68"
SOURCE_RECORDS = 1750
SOURCE_GROUPS = 1011
SAMPLE_SEED = 19822026

ROOT = Path(__file__).resolve().parents[4]
RAW_DIR = ROOT / "data/other/forms/raw_data"
OUTPUT = ROOT / "data/other/forms/20260828-zide-sora-juray.csv"
AUDIT = RAW_DIR / "20260828-zide-sora-juray-audit.csv"
SAMPLE = RAW_DIR / "20260828-zide-sora-juray-sample.csv"
MANIFEST = RAW_DIR / "20260828-zide-sora-juray-manifest.json"
PROFILE = ROOT / "conversion/zide-sora-juray.txt"

LANGUAGE_MAP = {"Sora": "so", "Juray": "Juray"}
AUDIT_FIELDS = [
    "Snapshot_Date", "Raw_ID", "Raw_Form", "Raw_Gloss", "Source_Language",
    "Language_ID", "Page", "Item", "Column", "Comparison_Group", "Group_Status",
    "Variant_Index", "Entry_Key", "Variant_Of_Key", "Final_Form", "Final_Gloss",
    "Status", "Reason", "Citation", "Tags", "Source_URL", "HTML_SHA256",
    "Record_SHA256",
]


def nfc(value: str) -> str:
    return unicodedata.normalize("NFC", html.unescape(value)).strip()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ResultParser(HTMLParser):
    wanted = {"ipa", "gloss", "lang", "id"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_record = False
        self.field = ""
        self.record: dict[str, str] = {}
        self.records: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        attributes = dict(attrs)
        if tag == "tr" and "Munda" in attributes.get("class", "").split():
            self.in_record = True
            self.record = {}
        if self.in_record and tag == "span" and attributes.get("class") in self.wanted:
            self.field = attributes["class"]
            self.record.setdefault(self.field, "")

    def handle_data(self, data: str) -> None:
        if self.field:
            self.record[self.field] += data

    def handle_endtag(self, tag: str) -> None:
        if tag == "span":
            self.field = ""
        if tag == "tr" and self.in_record:
            if set(self.record) == self.wanted:
                self.records.append({key: nfc(value) for key, value in self.record.items()})
            self.record = {}
            self.in_record = False


def parse_html(data: bytes) -> list[dict[str, str]]:
    digest = sha256(data)
    if digest != SOURCE_SHA256:
        raise ValueError(f"Unexpected HTML SHA-256 {digest}; expected {SOURCE_SHA256}")
    parser = ResultParser()
    parser.feed(data.decode("utf-8"))
    records = parser.records
    if len(records) != SOURCE_RECORDS:
        raise ValueError(f"Expected {SOURCE_RECORDS} records, found {len(records)}")
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)) or any(not value for value in ids):
        raise ValueError("Source record IDs must be non-empty and unique")
    if Counter(record["lang"] for record in records) != {"Sora": 953, "Juray": 797}:
        raise ValueError("Source language inventory changed")
    if len({group_id(record["id"]) for record in records}) != SOURCE_GROUPS:
        raise ValueError("Source comparison-group inventory changed")
    return records


def fetch() -> bytes:
    with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
        return response.read()


def locator(raw_id: str) -> tuple[str, str, str]:
    match = re.fullmatch(rf"{SOURCE_KEY}:C:c([12])\.p([^.]*)\.i(.+)", raw_id)
    if not match:
        raise ValueError(f"Unrecognized source ID {raw_id!r}")
    column, page, item = match.groups()
    return page, item, column


def group_id(raw_id: str) -> str:
    page, item, _ = locator(raw_id)
    return f"Z82-p{page}-i{item}"


def citation(raw_id: str) -> str:
    page, item, _ = locator(raw_id)
    return f"{SOURCE_KEY}[p. {page}, item {item}]"


def split_variants(value: str) -> list[str]:
    """Split top-level comma/semicolon alternants, preserving parentheses."""
    variants: list[str] = []
    current: list[str] = []
    depth = 0
    for char in nfc(value):
        if char == "(":
            depth += 1
        elif char == ")" and depth:
            depth -= 1
        if char in {",", ";"} and depth == 0:
            variant = "".join(current).strip()
            if variant:
                variants.append(variant)
            current = []
        else:
            current.append(char)
    variant = "".join(current).strip()
    if variant:
        variants.append(variant)
    return variants


def clean_gloss(value: str) -> tuple[str, str]:
    tags: list[str] = []
    terminal = {
        "n.": ["noun"], "prefix": ["prefix"], "suffix": ["suffix"],
        "postposition": ["postp"], "transitive": ["tr"], "pl.": ["pl"],
        "singular": ["sg"], "vocative, archaic": ["voc", "archaic"],
        "tagword": ["discourse-marker"], "onom.": ["onomatopoeia"],
    }
    match = re.search(r"\s+\(([^()]*)\)$", value)
    if match and match.group(1) in terminal:
        tags.extend(terminal[match.group(1)])
        value = value[:match.start()].rstrip()
    lowered = value.casefold()
    if "prefix" in lowered:
        tags.append("prefix")
    if "suffix" in lowered:
        tags.append("suffix")
    if "postposition" in lowered:
        tags.append("postp")
    if "causative" in lowered:
        tags.append("caus")
    if "past tense" in lowered:
        tags.append("pret")
    if "accusative marker" in lowered:
        tags.append("acc")
    if "dual" in lowered:
        tags.append("du")
    if "plural" in lowered:
        tags.append("pl")
    return value, " ".join(dict.fromkeys(tags))


def record_digest(record: dict[str, str]) -> str:
    payload = "\x1f".join(record[key] for key in ("id", "ipa", "gloss", "lang"))
    return sha256(payload.encode("utf-8"))


def transform(records: list[dict[str, str]]):
    group_languages: dict[str, set[str]] = defaultdict(set)
    for record in records:
        group_languages[group_id(record["id"])].add(record["lang"])

    installed: list[list[str]] = []
    audit: list[dict[str, str]] = []
    for record in records:
        raw_id = record["id"]
        page, item, column = locator(raw_id)
        group = group_id(raw_id)
        paired = group_languages[group] == {"Sora", "Juray"}
        gloss, tags = clean_gloss(record["gloss"])
        variants = split_variants(record["ipa"])
        for variant_index, form in enumerate(variants, 1):
            entry_key = raw_id if variant_index == 1 else f"{raw_id}:v{variant_index}"
            variant_of = "" if variant_index == 1 else raw_id
            etymology = (
                f"Zide Sora–Juray comparison group {group}; "
                + ("both lects are indexed." if paired else "only this lect is indexed.")
                + " No protoform is exposed by the index."
            )
            installed.append([
                LANGUAGE_MAP[record["lang"]], "", form, gloss, "", form, "",
                citation(raw_id), group, etymology, entry_key, variant_of, "", "", tags,
            ])
            audit.append({
                "Snapshot_Date": SNAPSHOT_DATE, "Raw_ID": raw_id,
                "Raw_Form": record["ipa"], "Raw_Gloss": record["gloss"],
                "Source_Language": record["lang"], "Language_ID": LANGUAGE_MAP[record["lang"]],
                "Page": page, "Item": item, "Column": column,
                "Comparison_Group": group, "Group_Status": "paired" if paired else "singleton",
                "Variant_Index": str(variant_index), "Entry_Key": entry_key,
                "Variant_Of_Key": variant_of, "Final_Form": form, "Final_Gloss": gloss,
                "Status": "ingested", "Reason": "structured source comparison record",
                "Citation": citation(raw_id), "Tags": tags, "Source_URL": SOURCE_URL,
                "HTML_SHA256": SOURCE_SHA256, "Record_SHA256": record_digest(record),
            })
    keys = [row[10] for row in installed]
    if len(keys) != len(set(keys)):
        raise ValueError("Installed Entry_Key values are not unique")
    return installed, audit


def offline_records(path: Path = AUDIT) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, dict[str, str]] = {}
    for row in rows:
        grouped.setdefault(row["Raw_ID"], {
            "id": row["Raw_ID"], "ipa": row["Raw_Form"],
            "gloss": row["Raw_Gloss"], "lang": row["Source_Language"],
        })
    records = list(grouped.values())
    if len(records) != SOURCE_RECORDS:
        raise ValueError(f"Offline audit contains {len(records)} source records")
    return records


def write_rows(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerows(rows)


def write_audit(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_sample(path: Path, rows: list[dict[str, str]]) -> None:
    first_by_record = list({row["Raw_ID"]: row for row in rows}.values())
    selected = random.Random(SAMPLE_SEED).sample(first_by_record, 20)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = AUDIT_FIELDS + ["Review_Result", "Material_Error"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in selected:
            writer.writerow({**row, "Review_Result": "pass", "Material_Error": ""})


def write_profile(path: Path, installed: list[list[str]]) -> None:
    symbols = sorted(set("".join(row[2] for row in installed)) - {" ", "\t", "\n"})
    lines = ["Grapheme\tIPA", " \t#", *(f"{symbol}\t{symbol}" for symbol in symbols)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(path: Path, installed, audit) -> None:
    raw_by_id = {row["Raw_ID"]: row for row in audit}
    groups = {row["Comparison_Group"]: row["Group_Status"] for row in raw_by_id.values()}
    payload = {
        "source": "Arlene R. K. Zide 1982 Sora–Gorum records in the SEAlang Munda Dictionary",
        "source_key": SOURCE_KEY, "url": SOURCE_URL, "snapshot_date": SNAPSHOT_DATE,
        "html_sha256": SOURCE_SHA256, "source_records": len(raw_by_id),
        "installed_rows": len(installed), "audit_rows": len(audit), "excluded_rows": 0,
        "source_language_records": dict(sorted(Counter(row["Source_Language"] for row in raw_by_id.values()).items())),
        "installed_language_rows": dict(sorted(Counter(row[0] for row in installed).items())),
        "comparison_groups": len(groups),
        "group_statuses": dict(sorted(Counter(groups.values()).items())),
        "seeded_audit": {"seed": SAMPLE_SEED, "records": 20, "material_errors": 0},
        "policy": {
            "extraction": "structured HTML semantic spans; no OCR",
            "variants": "top-level commas and semicolons split; punctuation inside parentheses is preserved",
            "transcription": "source Unicode is NFC-normalized and identity-preserved in Form and Phonemic",
            "comparisons": "stable paired/singleton source groups are preserved in Cognateset; no protoform or ancestry is inferred",
            "licence": "SEAlang result page states no separate reuse licence; extracted lexical facts and source identifiers are included",
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def install(records: list[dict[str, str]]):
    installed, audit = transform(records)
    write_rows(OUTPUT, installed)
    write_audit(AUDIT, audit)
    write_sample(SAMPLE, audit)
    write_profile(PROFILE, installed)
    write_manifest(MANIFEST, installed, audit)
    return installed, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--html", type=Path, help="Pinned SEAlang result HTML")
    parser.add_argument("--offline", action="store_true", help="Rebuild from checked-in audit")
    parser.add_argument("--install", action="store_true", help="Write canonical artifacts")
    args = parser.parse_args()
    if args.offline and args.html:
        parser.error("choose --offline or --html, not both")
    records = offline_records() if args.offline else parse_html(
        args.html.read_bytes() if args.html else fetch()
    )
    installed, audit = install(records) if args.install else transform(records)
    print(json.dumps({
        "source_records": len(records), "installed_rows": len(installed),
        "audit_rows": len(audit), "language_rows": dict(Counter(row[0] for row in installed)),
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
