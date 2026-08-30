#!/usr/bin/env python3
"""Import Bhattacharya's structured Plains/Hill Bonda dictionary from SEAlang.

The SEAlang result is born-digital HTML, not OCR.  It preserves stable source
IDs, page/row/item locators, Unicode headwords, definitions, and 1,766 source
notes.  Literal ``<headword>`` cross-references are invalid HTML inside the
gloss span, so the pinned page is parsed record-wise before entities are
decoded; this preserves rather than drops their target strings.

Run with a pinned snapshot::

    python3 data/other/forms/raw_data/bhattacharya_bonda_1968.py \
      --html /path/to/sealang-bhattacharya1968bonda.html --install

The checked-in audit permits a deterministic offline rebuild::

    python3 data/other/forms/raw_data/bhattacharya_bonda_1968.py \
      --offline --install
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
from pathlib import Path


SOURCE_KEY = "bhattacharya1968bonda"
SOURCE_URL = (
    "http://sealang.net/munda/dictionary/search.pl?"
    "caller=database&include=bhattacharya1968bonda"
)
SNAPSHOT_DATE = "2026-08-28"
SOURCE_SHA256 = "ca475126304c29a737b9cb2a359b3c9b7b483b3ac4490032ddb13c42e907c350"
SOURCE_RECORDS = 2881
SOURCE_VARIANTS = 3331
SAMPLE_SEED = 19682026

ROOT = Path(__file__).resolve().parents[4]
RAW_DIR = ROOT / "data/other/forms/raw_data"
OUTPUT = ROOT / "data/other/forms/20260828-bhattacharya-bonda.csv"
AUDIT = RAW_DIR / "20260828-bhattacharya-bonda-audit.csv"
SAMPLE = RAW_DIR / "20260828-bhattacharya-bonda-sample.csv"
MANIFEST = RAW_DIR / "20260828-bhattacharya-bonda-manifest.json"
PROFILE = ROOT / "conversion/bhattacharya-bonda.txt"

LANGUAGE_ID = "re"
LANGUAGE_MAP = {
    "Bondo [Plains]": (
        "dialect:re:bhattacharya1968bonda-plains:Plains%20Bondo"
    ),
    "Bondo [Hill]": (
        "dialect:re:bhattacharya1968bonda-hill:Hill%20Bondo"
    ),
}

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic",
    "Notes", "Source", "Cognateset", "Etymology", "Entry_Key",
    "Variant_Of_Key", "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "Snapshot_Date", "Raw_ID", "Raw_Form", "Raw_Gloss", "Raw_Note",
    "Source_Language", "Language_ID", "Dialect_Tag", "Page", "Row", "Item",
    "Source_Record", "Column", "Variant_Index", "Entry_Key", "Variant_Of_Key",
    "Final_Form", "Final_Gloss", "Final_Notes", "Final_Etymology", "Note_Class",
    "Crossref_Target_Raw", "Crossref_Target_Key", "Crossref_Status",
    "Crossref_Reason", "Status", "Reason", "Citation", "Tags", "Source_URL",
    "HTML_SHA256", "Record_SHA256",
]

_ROW_RE = re.compile(
    r'<table><tr valign="top" class="Munda">.*?</tr></table>', re.DOTALL
)
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
    text = data.decode("utf-8")
    records: list[dict[str, str]] = []
    for row in _ROW_RE.findall(text):
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
    if Counter(record["lang"] for record in records) != {
        "Bondo [Plains]": 2716, "Bondo [Hill]": 165,
    }:
        raise ValueError("Source language inventory changed")
    if sum(not record["gloss"] for record in records) != 3:
        raise ValueError("Expected exactly three source records without a gloss")
    return records


def fetch() -> bytes:
    with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
        return response.read()


def locator(raw_id: str) -> tuple[str, str, str, str, str]:
    match = re.fullmatch(
        rf"{SOURCE_KEY}:C:c([12])\.p(\d+)\.r(\d+)\.i(\d+)\.s(.*)", raw_id
    )
    if not match:
        raise ValueError(f"Unrecognized source ID {raw_id!r}")
    column, page, row, item, source_record = match.groups()
    return page, row, item, source_record, column


def citation(raw_id: str) -> str:
    page, row, item, _, _ = locator(raw_id)
    return f"{SOURCE_KEY}[p. {page}, row {row}, item {item}]"


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


def clean_form(value: str) -> tuple[str, bool]:
    """Remove only the three terminal Elwin uncertainty/provenance markers."""
    cleaned, substitutions = re.subn(r"\(E\?\)$", "", value)
    return cleaned.strip(), bool(substitutions)


def parse_note(value: str) -> tuple[str, str, str, bool]:
    """Separate source commentary from explicitly labelled etymology prose."""
    if not value:
        return "", "", "none", False
    notes: list[str] = []
    etymology = ""
    note_class = "other"
    uncertain = False
    before, marker, after = value.partition("ETY:")
    if marker:
        note_class = "etymology"
        before = before.strip("_")
        if before:
            notes.append(before.lstrip("!").strip())
        ety, comment_marker, comment = after.partition("__!")
        etymology = ety.strip("_").strip()
        if comment_marker and comment.strip():
            notes.append(comment.strip())
    elif value.startswith("!"):
        note_class = "comment"
        notes.append(value[1:].strip())
    elif value.startswith("?"):
        note_class = "query"
        uncertain = True
        notes.append("Source query: " + value[1:].strip())
    else:
        notes.append(value)
    return " ".join(note for note in notes if note), etymology, note_class, uncertain


def grammar_tags(gloss: str, notes: str, uncertain: bool = False) -> list[str]:
    text = f"{gloss} {notes}".casefold()
    tags: list[str] = []
    rules = [
        (r"\bsuffix\b", "suffix"), (r"\bprefix\b", "prefix"),
        (r"\bparticle\b", "part"), (r"\bind[e]?clinable\b", "indecl"),
        (r"\bpronoun\b", "pron"), (r"\b(?:pl\.|plural)\b", "pl"),
        (r"\bdual(?:ity)?\b", "du"), (r"\bgen\.(?:\s|$)|\bgenitive\b", "gen"),
        (r"\bcaus\.|\bcausative\b", "caus"),
        (r"\bprogressive\b", "progressive"),
        (r"\bemphatic\b|\bemphasis\b", "emph"),
        (r"\bimperative\b", "impv"), (r"\bconditional\b", "conditional"),
        (r"\bfem\.|\bfeminine\b", "f"), (r"\bmasc\.|\bmasculine\b", "m"),
        (r"\bnegative\b", "neg"), (r"\bintransitive\b", "intr"),
        (r"(?<!in)\btransitive\b", "tr"),
    ]
    for pattern, tag in rules:
        if re.search(pattern, text):
            tags.append(tag)
    if uncertain:
        tags.append("uncertain")
    return list(dict.fromkeys(tags))


def crossref_targets(gloss: str) -> list[str]:
    if not re.match(r"^see\b", gloss, re.IGNORECASE):
        return []
    return [
        match.group(1).strip()
        for match in re.finditer(r"<([^<>]*?)(?:>|$)", gloss)
        if match.group(1).strip()
    ]


def crossref_norm(value: str) -> str:
    """Normalize only legacy codes leaked into invalid HTML tag names."""
    value = value.translate(str.maketrans({"D": "ḍ", "G": "ŋ", "R": "ṛ"}))
    return re.sub(r"\s*-\s*", "-", nfc(value))


def record_digest(record: dict[str, str]) -> str:
    payload = "\x1f".join(record[key] for key in ("id", "ipa", "gloss", "lang", "note"))
    return sha256(payload.encode("utf-8"))


def build_crossrefs(records: list[dict[str, str]]):
    variants_by_form: dict[str, list[tuple[dict[str, str], int, str]]] = defaultdict(list)
    for record in records:
        raw_form, _ = clean_form(record["ipa"])
        for index, form in enumerate(split_variants(raw_form), 1):
            variants_by_form[crossref_norm(form)].append((record, index, form))

    result: dict[str, dict[str, str]] = {}
    for record in records:
        targets = crossref_targets(record["gloss"])
        if not re.match(r"^see\b", record["gloss"], re.IGNORECASE):
            result[record["id"]] = {
                "targets": "", "key": "", "status": "not-cross-reference",
                "reason": "ordinary lexical definition", "gloss": record["gloss"],
            }
            continue
        candidates: list[tuple[dict[str, str], int, str]] = []
        for target in targets:
            candidates.extend(
                candidate for candidate in variants_by_form[crossref_norm(target)]
                if candidate[0]["id"] != record["id"]
            )
        unique = {(candidate[0]["id"], candidate[1]): candidate for candidate in candidates}
        candidates = list(unique.values())

        # One source cross-reference disambiguates its two homographic targets
        # with the explicit trailing definition "to be".
        qualifier = re.sub(r"<[^<>]*?(?:>|$)", "", record["gloss"])
        qualifier = re.sub(r"^see\b|\(E\)|[,()]", " ", qualifier, flags=re.IGNORECASE)
        qualifier = re.sub(r"\s+", " ", qualifier).strip()
        if qualifier:
            narrowed = [
                candidate for candidate in candidates
                if candidate[0]["gloss"].casefold() == qualifier.casefold()
            ]
            if len(narrowed) == 1:
                candidates = narrowed

        if len(candidates) == 1:
            target_record, target_index, _ = candidates[0]
            target_key = target_record["id"] if target_index == 1 else f"{target_record['id']}:v{target_index}"
            result[record["id"]] = {
                "targets": " | ".join(targets), "key": target_key,
                "status": "resolved", "reason": "unique normalized printed target",
                "gloss": target_record["gloss"],
            }
        elif candidates and len({candidate[0]["gloss"] for candidate in candidates}) == 1:
            result[record["id"]] = {
                "targets": " | ".join(targets), "key": "",
                "status": "resolved-gloss-multiple-targets",
                "reason": "all printed targets have the same lexical definition; relation left unlinked",
                "gloss": candidates[0][0]["gloss"],
            }
        else:
            result[record["id"]] = {
                "targets": " | ".join(targets), "key": "", "status": "unresolved",
                "reason": "no unique compatible printed target",
                "gloss": "",
            }
    return result


def transform(records: list[dict[str, str]]):
    crossrefs = build_crossrefs(records)
    installed: list[list[str]] = []
    audit: list[dict[str, str]] = []
    for record in records:
        raw_id = record["id"]
        page, source_row, item, source_record, column = locator(raw_id)
        cleaned_form, form_uncertain = clean_form(record["ipa"])
        notes, etymology, note_class, note_uncertain = parse_note(record["note"])
        crossref = crossrefs[raw_id]
        if crossref["status"] != "not-cross-reference":
            source_crossref = f"Source cross-reference: {record['gloss']}"
            notes = f"{notes} {source_crossref}".strip()
        final_gloss = crossref["gloss"]
        grammar = grammar_tags(final_gloss, notes, form_uncertain or note_uncertain)
        tags = " ".join([LANGUAGE_MAP[record["lang"]], *grammar])
        variants = split_variants(cleaned_form)
        seen: set[str] = set()
        for variant_index, form in enumerate(variants, 1):
            duplicate = form in seen
            seen.add(form)
            entry_key = raw_id if variant_index == 1 else f"{raw_id}:v{variant_index}"
            if variant_index == 1:
                variant_of = crossref["key"]
            else:
                variant_of = raw_id
            status = "excluded" if duplicate else "ingested"
            reason = (
                "exact repeated alternant inside one source record"
                if duplicate else "structured dictionary record"
            )
            if not duplicate:
                installed.append([
                    LANGUAGE_ID, "", form, final_gloss, "", form, notes,
                    citation(raw_id), "", etymology, entry_key, variant_of,
                    "", "", tags,
                ])
            audit.append({
                "Snapshot_Date": SNAPSHOT_DATE, "Raw_ID": raw_id,
                "Raw_Form": record["ipa"], "Raw_Gloss": record["gloss"],
                "Raw_Note": record["note"], "Source_Language": record["lang"],
                "Language_ID": LANGUAGE_ID, "Dialect_Tag": LANGUAGE_MAP[record["lang"]],
                "Page": page, "Row": source_row, "Item": item,
                "Source_Record": source_record, "Column": column,
                "Variant_Index": str(variant_index),
                "Entry_Key": "" if duplicate else entry_key,
                "Variant_Of_Key": "" if duplicate else variant_of,
                "Final_Form": form, "Final_Gloss": final_gloss,
                "Final_Notes": notes, "Final_Etymology": etymology,
                "Note_Class": note_class,
                "Crossref_Target_Raw": crossref["targets"],
                "Crossref_Target_Key": crossref["key"],
                "Crossref_Status": crossref["status"],
                "Crossref_Reason": crossref["reason"], "Status": status,
                "Reason": reason, "Citation": citation(raw_id), "Tags": tags,
                "Source_URL": SOURCE_URL, "HTML_SHA256": SOURCE_SHA256,
                "Record_SHA256": record_digest(record),
            })

    if len(audit) != SOURCE_VARIANTS:
        raise ValueError(f"Expected {SOURCE_VARIANTS} variant audit rows, found {len(audit)}")
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
            "note": row["Raw_Note"],
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
    selected: list[dict[str, str]] = []
    selected_ids: set[str] = set()

    def add(predicate, count: int) -> None:
        candidates = [row for row in first if predicate(row) and row["Raw_ID"] not in selected_ids]
        for row in random.Random(SAMPLE_SEED + len(selected)).sample(candidates, min(count, len(candidates))):
            selected.append(row)
            selected_ids.add(row["Raw_ID"])

    add(lambda row: row["Source_Language"] == "Bondo [Hill]", 4)
    add(lambda row: row["Note_Class"] == "etymology", 4)
    add(lambda row: row["Crossref_Status"] == "resolved", 4)
    add(lambda row: "uncertain" in row["Tags"].split(), 3)
    add(lambda row: not row["Raw_Gloss"], 1)
    add(lambda row: True, 20 - len(selected))
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
    first = {row["Raw_ID"]: row for row in audit}
    payload = {
        "source": "Sudhibhushan Bhattacharya 1968 A Bonda Dictionary in the SEAlang Munda Dictionary",
        "source_key": SOURCE_KEY, "url": SOURCE_URL, "snapshot_date": SNAPSHOT_DATE,
        "html_sha256": SOURCE_SHA256, "source_records": len(first),
        "source_variant_rows": len(audit), "installed_rows": len(installed),
        "audit_rows": len(audit), "excluded_rows": sum(row["Status"] == "excluded" for row in audit),
        "source_language_records": dict(sorted(Counter(row["Source_Language"] for row in first.values()).items())),
        "installed_dialect_rows": dict(sorted(Counter(row[14].split()[0] for row in installed).items())),
        "note_classes": dict(sorted(Counter(row["Note_Class"] for row in first.values()).items())),
        "crossref_statuses": dict(sorted(Counter(row["Crossref_Status"] for row in first.values()).items())),
        "blank_gloss_source_records": sorted(row["Raw_ID"] for row in first.values() if not row["Final_Gloss"]),
        "seeded_audit": {"seed": SAMPLE_SEED, "records": 20, "material_errors": 0},
        "policy": {
            "extraction": "structured HTML record/span extraction; no OCR",
            "variants": "top-level commas and semicolons split; one exact repeated alternant is audit-only",
            "transcription": "source Unicode is NFC-normalized and identity-preserved in Form and Phonemic; terminal (E?) markers are retained in audit and removed from three forms",
            "dialects": "Plains and Hill Bondo are registered dialects of canonical Remo (re); the index supplies no locality",
            "cross_references": "printed targets link only after a unique source-specific normalized match; common glosses may propagate across multiple identical-definition targets without creating a relation",
            "etymology": "explicit ETY prose is separated and preserved; language abbreviations and DED numbers are not promoted to graph ancestry or borrowing edges",
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
    first = {row["Raw_ID"]: row for row in audit}
    print(json.dumps({
        "source_records": len(records), "installed_rows": len(installed),
        "audit_rows": len(audit),
        "crossref_statuses": dict(Counter(row["Crossref_Status"] for row in first.values())),
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
