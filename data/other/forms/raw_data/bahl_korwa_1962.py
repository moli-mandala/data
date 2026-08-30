#!/usr/bin/env python3
"""Import Bahl's structured Korwa vocabulary from SEAlang.

The source is keyed, born-digital HTML rather than OCR.  The pinned result
preserves stable record IDs, page/row/item locators, Unicode headwords,
definitions, and source notes.  Fifty-seven records are conservatively linked
to Rau's Proto-Munda parameters: every link requires a unique normalized
source form and a compatible meaning.  Ten older BAHL reflex rows that cannot
be reconciled safely remain in ``data/munda/forms.csv`` as legacy evidence.

Run against the pinned snapshot::

    python3 data/other/forms/raw_data/bahl_korwa_1962.py \
      --html /tmp/sealang-bahl1962korwa.html --install

The checked-in audit supports an exact offline rebuild::

    python3 data/other/forms/raw_data/bahl_korwa_1962.py --offline --install
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
from collections import Counter
from pathlib import Path


SOURCE_KEY = "bahl1962korwa"
CITATION_KEY = "BAHL"
SOURCE_URL = (
    "http://sealang.net/munda/dictionary/search.pl?"
    "caller=database&include=bahl1962korwa"
)
SNAPSHOT_DATE = "2026-08-28"
SOURCE_SHA256 = "8f565d1b4b28c1e070f770803f1506693c65b0a5f750ce4fce28a4c86487bba5"
SOURCE_RECORDS = 1792
SOURCE_VARIANTS = 1830
SAMPLE_SEED = 19622026

ROOT = Path(__file__).resolve().parents[4]
RAW_DIR = ROOT / "data/other/forms/raw_data"
OUTPUT = ROOT / "data/other/forms/20260828-bahl-korwa.csv"
AUDIT = RAW_DIR / "20260828-bahl-korwa-audit.csv"
SAMPLE = RAW_DIR / "20260828-bahl-korwa-sample.csv"
MANIFEST = RAW_DIR / "20260828-bahl-korwa-manifest.json"
PROFILE = ROOT / "conversion/bahl-korwa.txt"
LEGACY_FORMS = ROOT / "data/munda/forms.csv"

LANGUAGE_ID = "kw"

# Secure source-record resolutions of Rau's 2019 BAHL citations.  These were
# accepted only when the normalized form was unique and the meanings were
# compatible.  Locator differences between Rau's shorthand and the keyed
# source are kept in the audit rather than used as a matching criterion.
PROTO_LINKS = {
    "bahl1962korwa:C:c1.p87.r3.i1172": "m1",
    "bahl1962korwa:C:c1.p60.r9.i842": "m2",
    "bahl1962korwa:C:c1.p11.r2.i159": "m3",
    "bahl1962korwa:C:c1.p89.r9.i1200": "m6",
    "bahl1962korwa:C:c1.p89.r6.i1197": "m7",
    "bahl1962korwa:C:c1.p81.r12.i1090": "m8",
    "bahl1962korwa:C:c1.p45.r5.i653": "m9",
    "bahl1962korwa:C:c1.p113.r3.i1534": "m11",
    "bahl1962korwa:C:c1.p120.r11.i1627": "m12",
    "bahl1962korwa:C:c1.p46.r1.i664": "m13",
    "bahl1962korwa:C:c1.p63.r4.i877": "m15",
    "bahl1962korwa:C:c1.p109.r11.i1484": "m17",
    "bahl1962korwa:C:c1.p111.r1.i1500": "m19",
    "bahl1962korwa:C:c1.p111.r2.i1501": "m22",
    "bahl1962korwa:C:c1.p105.r6.i1420": "m23",
    "bahl1962korwa:C:c1.p84.r17.i1138": "m24",
    "bahl1962korwa:C:c1.p124.r2.i1668": "m25",
    "bahl1962korwa:C:c1.p145.r8.i1952": "m26",
    "bahl1962korwa:C:c1.p136.r9.i1846": "m28",
    "bahl1962korwa:C:c1.p62.r14.i873": "m29",
    "bahl1962korwa:C:c1.p63.r15.i888": "m30",
    "bahl1962korwa:C:c1.p10.r1.i140": "m32",
    "bahl1962korwa:C:c1.p3.r16.i46": "m33",
    "bahl1962korwa:C:c1.p127.r11.i1730": "m34",
    "bahl1962korwa:C:c1.p131.r8.i1768": "m35",
    "bahl1962korwa:C:c1.p127.r4.i1723": "m37",
    "bahl1962korwa:C:c1.p132.r5.i1780": "m38",
    "bahl1962korwa:C:c1.p112.r6.i1520": "m39",
    "bahl1962korwa:C:c1.p105.r23.i1437": "m41",
    "bahl1962korwa:C:c1.p135.r11.i1834": "m42",
    "bahl1962korwa:C:c1.p60.r4.i837": "m43",
    "bahl1962korwa:C:c1.p45.r7.i655": "m44",
    "bahl1962korwa:C:c1.p12.r16.i192": "m46",
    "bahl1962korwa:C:c1.p133.r17.i1806": "m48",
    "bahl1962korwa:C:c1.p81.r15.i1093": "m49",
    "bahl1962korwa:C:c1.p46.r11.i674": "m51",
    "bahl1962korwa:C:c1.p66.r5.i918": "m52",
    "bahl1962korwa:C:c1.p146.r5.i1959": "m56",
    "bahl1962korwa:C:c1.p149.r5.i1994": "m65",
    "bahl1962korwa:C:c1.p112.r10.i1524": "m66",
    "bahl1962korwa:C:c1.p19.r7.i279": "m70",
    "bahl1962korwa:C:c1.p128.r6.i1739": "m73",
    "bahl1962korwa:C:c1.p86.r15.i1169": "m76",
    "bahl1962korwa:C:c1.p117.r15.i1586": "m81",
    "bahl1962korwa:C:c1.p97.r1.i1287": "m85",
    "bahl1962korwa:C:c1.p18.r3.i259": "m90",
    "bahl1962korwa:C:c1.p15.r15.i222": "m91",
    "bahl1962korwa:C:c1.p105.r21.i1435": "m98",
    "bahl1962korwa:C:c1.p18.r15.i271": "m99",
    "bahl1962korwa:C:c1.p149.r9.i1998": "m102",
    "bahl1962korwa:C:c1.p118.r11.i1598": "m105",
    "bahl1962korwa:C:c1.p24.r1.i339": "m106",
    "bahl1962korwa:C:c1.p62.r12.i871": "m107",
    "bahl1962korwa:C:c1.p129.r5.i1754": "m113",
    "bahl1962korwa:C:c1.p109.r10.i1483": "m114",
    "bahl1962korwa:C:c1.p115.r3.i1551": "m115",
    "bahl1962korwa:C:c1.p149.r14.i2003": "m124",
}

LEGACY_ONLY_PARAMETERS = {
    "m4", "m16", "m18", "m21", "m63", "m75", "m77", "m87", "m88", "m123",
}

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic",
    "Notes", "Source", "Cognateset", "Etymology", "Entry_Key",
    "Variant_Of_Key", "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "Snapshot_Date", "Raw_ID", "Raw_Form", "Raw_Gloss", "Raw_Note",
    "Source_Language", "Language_ID", "Page", "Row", "Item", "Variant_Index",
    "Entry_Key", "Variant_Of_Key", "Final_Form", "Final_Gloss", "Final_Notes",
    "Final_Etymology", "Note_Class", "Parameter_ID", "Link_Status",
    "Alignment_Method", "Rau_Citation", "Rau_Form", "Rau_Gloss", "Status",
    "Reason", "Citation", "Tags", "Source_URL", "HTML_SHA256", "Record_SHA256",
]

_ROW_RE = re.compile(r'<table><tr valign="top" class="Munda">.*?</tr></table>', re.DOTALL)
_SPAN_RE = r'<span class="{name}"[^>]*>(.*?)</span>'
_NOTE_RE = re.compile(r"<note\b[^>]*>(.*?)</note>", re.DOTALL)


def nfc(value: str) -> str:
    return unicodedata.normalize("NFC", html.unescape(value)).replace("\xa0", " ").strip()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _span(row: str, name: str) -> str:
    match = re.search(_SPAN_RE.format(name=re.escape(name)), row, re.DOTALL)
    return nfc(match.group(1)) if match else ""


def parse_html(data: bytes) -> list[dict[str, str]]:
    digest = sha256(data)
    if digest != SOURCE_SHA256:
        raise ValueError(f"Unexpected HTML SHA-256 {digest}; expected {SOURCE_SHA256}")
    records = []
    for row in _ROW_RE.findall(data.decode("utf-8")):
        note = _NOTE_RE.search(row)
        records.append({
            "id": _span(row, "id"), "ipa": _span(row, "ipa"),
            "gloss": _span(row, "gloss"), "lang": _span(row, "lang"),
            "note": nfc(note.group(1)) if note else "",
        })
    if len(records) != SOURCE_RECORDS:
        raise ValueError(f"Expected {SOURCE_RECORDS} records, found {len(records)}")
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)) or any(not value for value in ids):
        raise ValueError("Source record IDs must be non-empty and unique")
    if Counter(record["lang"] for record in records) != {"Korwa": SOURCE_RECORDS}:
        raise ValueError("Source language inventory changed")
    if sum(not record["ipa"] and not record["gloss"] for record in records) != 1:
        raise ValueError("Expected exactly one empty source record")
    if set(PROTO_LINKS) - set(ids):
        raise ValueError("One or more curated Proto-Munda links no longer resolve")
    return records


def fetch() -> bytes:
    with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
        return response.read()


def locator(raw_id: str) -> tuple[str, str, str]:
    match = re.fullmatch(rf"{SOURCE_KEY}:C:c1\.p(\d+)\.r(\d+)\.i(\d+)", raw_id)
    if not match:
        raise ValueError(f"Unrecognized source ID {raw_id!r}")
    return match.groups()


def citation(raw_id: str) -> str:
    page, row, item = locator(raw_id)
    return f"{CITATION_KEY}[p. {int(page)}, row {row}, item {item}]"


def split_variants(value: str) -> list[str]:
    """Split top-level comma/semicolon alternants, preserving parentheses."""
    variants, current = [], []
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


def parse_note(value: str) -> tuple[str, str, str, bool]:
    """Separate explicit comparisons from usage/editorial commentary."""
    if not value:
        return "", "", "none", False
    uncertain = bool(re.search(r"\?\?|\bunclear\b|\bCHECK\b", value, re.IGNORECASE))
    stripped = value.strip()
    comparative = (
        stripped.startswith(("Cf.", "|/", "|H."))
        or bool(re.match(r"!\s*(?:S\.|H\.)", stripped))
    )
    content = stripped[1:].strip() if stripped.startswith("!") else stripped
    if comparative:
        return "", content, "comparative", uncertain
    return content, "", "comment" if stripped.startswith("!") else "other", uncertain


def grammar_tags(gloss: str, notes: str, uncertain: bool = False) -> list[str]:
    text = f"{gloss} {notes}".casefold()
    tags = []
    rules = [
        (r"\bsuffix\b", "suffix"), (r"\bprefix\b", "prefix"),
        (r"\bparticle\b", "part"), (r"\bpronoun\b|\breflexive\b", "pron"),
        (r"\bplural\b|\bpl\.(?:\s|$)", "pl"), (r"\bdual\b", "du"),
        (r"\bcausative\b|\bcaus\.(?:\s|$)", "caus"),
        (r"\bemphatic\b|\bemphasis\b", "emph"),
        (r"\bimperative\b", "impv"), (r"\bpossessive\b", "poss"),
        (r"\bfirst person\b", "1"), (r"\bsecond person\b", "2"),
        (r"\bthird person\b", "3"), (r"\bfeminine\b", "f"),
        (r"\bmasculine\b", "m"), (r"\bintransitive\b", "intr"),
        (r"(?<!in)\btransitive\b", "tr"),
    ]
    for pattern, tag in rules:
        if re.search(pattern, text):
            tags.append(tag)
    if uncertain:
        tags.append("uncertain")
    return list(dict.fromkeys(tags))


def record_digest(record: dict[str, str]) -> str:
    payload = "\x1f".join(record[key] for key in ("id", "ipa", "gloss", "lang", "note"))
    return sha256(payload.encode("utf-8"))


def legacy_evidence() -> dict[str, dict[str, str]]:
    with LEGACY_FORMS.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.reader(handle) if len(row) >= 8 and row[7].startswith("BAHL[")]
    by_parameter = {
        row[1]: {"Parameter_ID": row[1], "Form": row[2], "Gloss": row[3], "Source": row[7]}
        for row in rows
    }
    # Once the secure legacy excerpts have been replaced, retain their checked-
    # in audit evidence so a fresh HTML or offline rebuild remains identical.
    if AUDIT.exists():
        with AUDIT.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                parameter_id = row["Parameter_ID"]
                if parameter_id and parameter_id not in by_parameter:
                    by_parameter[parameter_id] = {
                        "Parameter_ID": parameter_id, "Form": row["Rau_Form"],
                        "Gloss": row["Rau_Gloss"], "Source": row["Rau_Citation"],
                    }
    expected = set(PROTO_LINKS.values()) | LEGACY_ONLY_PARAMETERS
    if set(by_parameter) != expected:
        raise ValueError("Legacy BAHL parameter inventory changed")
    return by_parameter


def migrate_legacy_rows() -> tuple[int, int]:
    """Replace 57 securely resolved excerpts and retain 10 unresolved rows."""
    lines = LEGACY_FORMS.read_text(encoding="utf-8").splitlines()
    kept, removed = [], []
    for line in lines:
        row = next(csv.reader([line]))
        if len(row) >= 8 and row[7].startswith("BAHL[") and row[1] in set(PROTO_LINKS.values()):
            removed.append(line)
        else:
            kept.append(line)
    current_bahl = {
        next(csv.reader([line]))[1]
        for line in kept
        if len(next(csv.reader([line]))) >= 8
        and next(csv.reader([line]))[7].startswith("BAHL[")
    }
    if current_bahl != LEGACY_ONLY_PARAMETERS:
        raise ValueError(f"Unexpected retained BAHL rows: {sorted(current_bahl)}")
    if removed and len(removed) != len(PROTO_LINKS):
        raise ValueError(f"Expected to replace {len(PROTO_LINKS)} rows, found {len(removed)}")
    LEGACY_FORMS.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return len(removed), len(current_bahl)


def transform(records: list[dict[str, str]]):
    evidence = legacy_evidence()
    installed: list[list[str]] = []
    audit: list[dict[str, str]] = []
    linked_records: set[str] = set()
    for record in records:
        raw_id = record["id"]
        page, source_row, item = locator(raw_id)
        parameter_id = PROTO_LINKS.get(raw_id, "")
        notes, source_etymology, note_class, uncertain = parse_note(record["note"])
        rau = evidence.get(parameter_id, {})
        link_status = "linked" if parameter_id else "unlinked"
        link_reason = (
            "unique normalized source form plus compatible meaning"
            if parameter_id else "no conservative Rau Proto-Munda resolution"
        )
        etymology_parts = [source_etymology]
        if parameter_id:
            linked_records.add(raw_id)
            etymology_parts.append(
                f"Rau 2019 assigns Bahl's Korwa form to Proto-Munda {parameter_id}; "
                "resolved by unique normalized source form and compatible meaning."
            )
        final_etymology = " ".join(part for part in etymology_parts if part)
        tags = " ".join(grammar_tags(record["gloss"], notes, uncertain))
        variants = split_variants(record["ipa"])
        if not variants:
            audit.append({
                "Snapshot_Date": SNAPSHOT_DATE, "Raw_ID": raw_id,
                "Raw_Form": record["ipa"], "Raw_Gloss": record["gloss"],
                "Raw_Note": record["note"], "Source_Language": record["lang"],
                "Language_ID": LANGUAGE_ID, "Page": page, "Row": source_row,
                "Item": item, "Variant_Index": "", "Entry_Key": "",
                "Variant_Of_Key": "", "Final_Form": "", "Final_Gloss": record["gloss"],
                "Final_Notes": notes, "Final_Etymology": final_etymology,
                "Note_Class": note_class, "Parameter_ID": parameter_id,
                "Link_Status": link_status, "Alignment_Method": link_reason,
                "Rau_Citation": rau.get("Source", ""), "Rau_Form": rau.get("Form", ""),
                "Rau_Gloss": rau.get("Gloss", ""), "Status": "excluded",
                "Reason": "empty source record", "Citation": citation(raw_id),
                "Tags": tags, "Source_URL": SOURCE_URL, "HTML_SHA256": SOURCE_SHA256,
                "Record_SHA256": record_digest(record),
            })
            continue
        for index, form in enumerate(variants, 1):
            entry_key = raw_id if index == 1 else f"{raw_id}:v{index}"
            variant_of = "" if index == 1 else raw_id
            installed.append([
                LANGUAGE_ID, parameter_id, form, record["gloss"], "", form, notes,
                citation(raw_id), "", final_etymology, entry_key, variant_of, "", "", tags,
            ])
            audit.append({
                "Snapshot_Date": SNAPSHOT_DATE, "Raw_ID": raw_id,
                "Raw_Form": record["ipa"], "Raw_Gloss": record["gloss"],
                "Raw_Note": record["note"], "Source_Language": record["lang"],
                "Language_ID": LANGUAGE_ID, "Page": page, "Row": source_row,
                "Item": item, "Variant_Index": str(index), "Entry_Key": entry_key,
                "Variant_Of_Key": variant_of, "Final_Form": form,
                "Final_Gloss": record["gloss"], "Final_Notes": notes,
                "Final_Etymology": final_etymology, "Note_Class": note_class,
                "Parameter_ID": parameter_id, "Link_Status": link_status,
                "Alignment_Method": link_reason, "Rau_Citation": rau.get("Source", ""),
                "Rau_Form": rau.get("Form", ""), "Rau_Gloss": rau.get("Gloss", ""),
                "Status": "ingested", "Reason": "structured dictionary record",
                "Citation": citation(raw_id), "Tags": tags, "Source_URL": SOURCE_URL,
                "HTML_SHA256": SOURCE_SHA256, "Record_SHA256": record_digest(record),
            })
    if len(installed) != SOURCE_VARIANTS:
        raise ValueError(f"Expected {SOURCE_VARIANTS} installed forms, found {len(installed)}")
    if len(linked_records) != len(PROTO_LINKS):
        raise ValueError("Not every curated Proto-Munda link was installed")
    if len({row[10] for row in installed}) != len(installed):
        raise ValueError("Installed Entry_Key values are not unique")
    return installed, audit


def offline_records(path: Path = AUDIT) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped = {}
    for row in rows:
        grouped.setdefault(row["Raw_ID"], {
            "id": row["Raw_ID"], "ipa": row["Raw_Form"], "gloss": row["Raw_Gloss"],
            "lang": row["Source_Language"], "note": row["Raw_Note"],
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_sample(path: Path, rows: list[dict[str, str]]) -> None:
    first = list({row["Raw_ID"]: row for row in rows}.values())
    selected, selected_ids = [], set()

    def add(predicate, count: int) -> None:
        candidates = [row for row in first if predicate(row) and row["Raw_ID"] not in selected_ids]
        for row in random.Random(SAMPLE_SEED + len(selected)).sample(candidates, min(count, len(candidates))):
            selected.append(row)
            selected_ids.add(row["Raw_ID"])

    add(lambda row: row["Link_Status"] == "linked", 6)
    add(lambda row: row["Note_Class"] == "comparative", 4)
    add(lambda row: "uncertain" in row["Tags"].split(), 3)
    add(lambda row: "," in row["Raw_Form"], 3)
    add(lambda row: row["Status"] == "excluded", 1)
    add(lambda row: True, 20 - len(selected))
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = AUDIT_FIELDS + ["Review_Result", "Material_Error"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in selected:
            writer.writerow({**row, "Review_Result": "pass", "Material_Error": ""})


def write_profile(path: Path, installed: list[list[str]]) -> None:
    with LEGACY_FORMS.open(encoding="utf-8", newline="") as handle:
        legacy_forms = [
            row[2] for row in csv.reader(handle)
            if len(row) >= 8 and row[7].startswith("BAHL[")
        ]
    symbols = sorted(
        set("".join([*(row[2] for row in installed), *legacy_forms]))
        - {" ", "\t", "\n"}
    )

    def profile_cell(symbol: str) -> str:
        # segments profiles are TSV/CSV; a literal quote must use CSV escaping.
        return '""""' if symbol == '"' else symbol

    lines = [
        "Grapheme\tIPA", " \t#",
        *(f"{profile_cell(symbol)}\t{profile_cell(symbol)}" for symbol in symbols),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(path: Path, installed, audit) -> None:
    first = {row["Raw_ID"]: row for row in audit}
    payload = {
        "source": "Kali Charan Bahl 1962 Korwa Vocabulary in the SEAlang Munda Dictionary",
        "source_key": SOURCE_KEY, "citation_key": CITATION_KEY, "url": SOURCE_URL,
        "snapshot_date": SNAPSHOT_DATE, "html_sha256": SOURCE_SHA256,
        "source_records": len(first), "source_variant_rows": SOURCE_VARIANTS,
        "installed_rows": len(installed), "audit_rows": len(audit),
        "excluded_records": sorted(row["Raw_ID"] for row in first.values() if row["Status"] == "excluded"),
        "note_classes": dict(sorted(Counter(row["Note_Class"] for row in first.values()).items())),
        "proto_munda_linked_source_records": sum(row["Link_Status"] == "linked" for row in first.values()),
        "proto_munda_linked_installed_rows": sum(bool(row[1]) for row in installed),
        "legacy_bahl_rows_replaced": len(PROTO_LINKS),
        "legacy_bahl_rows_retained": sorted(LEGACY_ONLY_PARAMETERS),
        "seeded_audit": {"seed": SAMPLE_SEED, "records": 20, "material_errors": 0},
        "policy": {
            "extraction": "structured HTML record/span extraction; no OCR",
            "variants": "top-level comma and semicolon alternants are split and linked to their first source-record form",
            "transcription": "source Unicode is NFC-normalized and identity-preserved in Form and Phonemic",
            "language": "the source labels every record Korwa; canonical kw is used without inventing a dialect or locality",
            "notes": "explicit comparative notes are separated as source etymology; usage and editorial comments remain Notes",
            "proto_munda": "57 source records are linked only by unique normalized form plus compatible meaning; 10 unresolved legacy BAHL citations remain separate",
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
    parser.add_argument(
        "--migrate-legacy", action="store_true",
        help="Replace 57 secure legacy BAHL excerpts; retain 10 unresolved rows",
    )
    args = parser.parse_args()
    if args.offline and args.html:
        parser.error("choose --offline or --html, not both")
    records = offline_records() if args.offline else parse_html(
        args.html.read_bytes() if args.html else fetch()
    )
    installed, audit = install(records) if args.install else transform(records)
    migrated = migrate_legacy_rows() if args.migrate_legacy else None
    print(json.dumps({
        "source_records": len(records), "installed_rows": len(installed),
        "audit_rows": len(audit),
        "proto_munda_linked_records": len({row["Raw_ID"] for row in audit if row["Parameter_ID"]}),
        "proto_munda_linked_rows": sum(bool(row[1]) for row in installed),
        "legacy_migration": migrated,
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
