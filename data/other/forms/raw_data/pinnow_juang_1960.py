#!/usr/bin/env python3
"""Import Pinnow's structured Juang vocabulary from SEAlang (no OCR)."""

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


SOURCE_KEY = "pinnow1960beitraege"
CITATION_KEY = "PJDW"
SOURCE_URL = (
    "http://sealang.net/munda/dictionary/search.pl?"
    "caller=database&include=pinnow1960beitraege"
)
SNAPSHOT_DATE = "2026-08-28"
SOURCE_SHA256 = "6fc00f2689630d5bed5e5c04f6efd15654a40f11cd8a66b186d54fea5aede596"
SOURCE_RECORDS = 1658
SOURCE_VARIANTS = 1824
INSTALLED_ROWS = 1818
SAMPLE_SEED = 19602026

ROOT = Path(__file__).resolve().parents[4]
RAW_DIR = ROOT / "data/other/forms/raw_data"
OUTPUT = ROOT / "data/other/forms/20260828-pinnow-juang.csv"
AUDIT = RAW_DIR / "20260828-pinnow-juang-audit.csv"
SAMPLE = RAW_DIR / "20260828-pinnow-juang-sample.csv"
MANIFEST = RAW_DIR / "20260828-pinnow-juang-manifest.json"
PROFILE = ROOT / "conversion/pinnow-juang.txt"
LEGACY_FORMS = ROOT / "data/munda/forms.csv"
LANGUAGE_ID = "ju"

PROTO_LINKS = {
    "pinnow1960beitraege:C:i407": "m1", "pinnow1960beitraege:C:i697": "m2",
    "pinnow1960beitraege:C:i477": "m3", "pinnow1960beitraege:C:i848": "m4",
    "pinnow1960beitraege:C:i418": "m6", "pinnow1960beitraege:C:i684": "m8",
    "pinnow1960beitraege:C:i492": "m9", "pinnow1960beitraege:C:i674": "m10",
    "pinnow1960beitraege:C:i223": "m11", "pinnow1960beitraege:C:i478": "m12",
    "pinnow1960beitraege:C:i1186": "m14", "pinnow1960beitraege:C:i714": "m15",
    "pinnow1960beitraege:C:i279": "m17", "pinnow1960beitraege:C:i194": "m18",
    "pinnow1960beitraege:C:i248": "m19", "pinnow1960beitraege:C:i1421": "m21",
    "pinnow1960beitraege:C:i180": "m22", "pinnow1960beitraege:C:i1557": "m24",
    "pinnow1960beitraege:C:i1324": "m25", "pinnow1960beitraege:C:i799": "m26",
    "pinnow1960beitraege:C:i1570": "m27", "pinnow1960beitraege:C:i1418": "m28",
    "pinnow1960beitraege:C:i62": "m32", "pinnow1960beitraege:C:i1181": "m35",
    "pinnow1960beitraege:C:i357": "m36", "pinnow1960beitraege:C:i999": "m37",
    "pinnow1960beitraege:C:i482": "m38", "pinnow1960beitraege:C:i430": "m40",
    "pinnow1960beitraege:C:i258": "m41", "pinnow1960beitraege:C:i1427": "m42",
    "pinnow1960beitraege:C:i568": "m44", "pinnow1960beitraege:C:i199": "m45",
    "pinnow1960beitraege:C:i1447": "m48", "pinnow1960beitraege:C:i1543": "m49",
    "pinnow1960beitraege:C:i1552": "m50", "pinnow1960beitraege:C:i542": "m51",
    "pinnow1960beitraege:C:i913": "m55", "pinnow1960beitraege:C:i679": "m59",
    "pinnow1960beitraege:C:i1459": "m60", "pinnow1960beitraege:C:i1347": "m68",
    "pinnow1960beitraege:C:i647": "m70", "pinnow1960beitraege:C:i1386": "m72",
    "pinnow1960beitraege:C:i1022": "m73", "pinnow1960beitraege:C:i1084": "m74",
    "pinnow1960beitraege:C:i1194": "m77", "pinnow1960beitraege:C:i504": "m80",
    "pinnow1960beitraege:C:i1045": "m81", "pinnow1960beitraege:C:i691": "m82",
    "pinnow1960beitraege:C:i1536": "m83", "pinnow1960beitraege:C:i1434": "m84",
    "pinnow1960beitraege:C:i1384": "m87", "pinnow1960beitraege:C:i572": "m89",
    "pinnow1960beitraege:C:i1623": "m90", "pinnow1960beitraege:C:i1335": "m94",
    "pinnow1960beitraege:C:i720": "m95", "pinnow1960beitraege:C:i205": "m98",
    "pinnow1960beitraege:C:i793": "m101", "pinnow1960beitraege:C:i1422": "m103",
    "pinnow1960beitraege:C:i635": "m104", "pinnow1960beitraege:C:i1070": "m105",
    "pinnow1960beitraege:C:i528": "m108", "pinnow1960beitraege:C:i1550": "m110",
    "pinnow1960beitraege:C:i1513": "m111", "pinnow1960beitraege:C:i257": "m123",
    "pinnow1960beitraege:C:i57": "m126", "pinnow1960beitraege:C:i1089": "m127",
}
LEGACY_ONLY_PARAMETERS = {"m16", "m31", "m75", "m78", "m79", "m93", "m102"}

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic",
    "Notes", "Source", "Cognateset", "Etymology", "Entry_Key",
    "Variant_Of_Key", "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "Snapshot_Date", "Raw_ID", "Raw_Form", "Raw_Gloss", "Raw_Note",
    "Source_Language", "Language_ID", "Item", "Variant_Index", "Entry_Key",
    "Variant_Of_Key", "Final_Form", "Final_Gloss", "Final_Notes",
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
    if Counter(record["lang"] for record in records) != {"Juang": SOURCE_RECORDS}:
        raise ValueError("Source language inventory changed")
    if any(not record["ipa"] for record in records):
        raise ValueError("Unexpected blank source form")
    if set(PROTO_LINKS) - set(ids):
        raise ValueError("One or more curated Proto-Munda links no longer resolve")
    return records


def fetch() -> bytes:
    with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
        return response.read()


def item(raw_id: str) -> str:
    match = re.fullmatch(rf"{SOURCE_KEY}:C:i(\d+)", raw_id)
    if not match:
        raise ValueError(f"Unrecognized source ID {raw_id!r}")
    return match.group(1)


def citation(raw_id: str) -> str:
    return f"{CITATION_KEY}[item {item(raw_id)}]"


def split_variants(value: str) -> list[str]:
    variants, current = [], []
    depth = 0
    for char in nfc(value):
        if char in "([":
            depth += 1
        elif char in ")]" and depth:
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


def clean_form(value: str) -> tuple[str, bool, str]:
    marker_match = re.search(r"(\(E[\d,]+\))$", value)
    marker = marker_match.group(1) if marker_match else ""
    if marker:
        value = value[:-len(marker)]
    cleaned, substitutions = re.subn(r"\s*(?:\(\?\?\)|\?\?)$", "", value)
    return cleaned.strip(), bool(substitutions), marker


def parse_note(value: str) -> tuple[str, str, str, bool]:
    if not value:
        return "", "", "none", False
    uncertain = "??" in value
    notes, comparisons = value, ""
    if "#" in notes:
        notes, marker, comparison = notes.partition("#")
        comparisons = marker + comparison.strip()
    cf = re.search(r"(?:^|\s)(Cf\.)", notes)
    if cf:
        comparison = notes[cf.start(1):].strip()
        notes = notes[:cf.start(1)].strip()
        comparisons = " ".join(part for part in (comparisons, comparison) if part)
    return notes.strip(), comparisons, "comparative" if comparisons else "comment", uncertain


def clean_gloss(value: str) -> tuple[str, bool]:
    if not value or value == "?":
        return "", value == "?"
    return re.sub(r"_+", " ", value), False


def grammar_tags(gloss: str, notes: str, uncertain: bool = False) -> list[str]:
    text = f"{gloss} {notes}".casefold()
    tags = []
    rules = [
        (r"\bsuffix\b", "suffix"), (r"\bprefix\b", "prefix"),
        (r"\bparticle\b", "part"), (r"\bpronoun\b|\breflexive\b", "pron"),
        (r"\bplural\b|\bpl\.(?:\s|$)", "pl"), (r"\bdual\b", "du"),
        (r"\bgenitive\b|\bpossessive\b", "gen"), (r"\blocative\b", "loc"),
        (r"\bcausative\b|\bcause to\b", "caus"), (r"\bimperative\b", "impv"),
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
        rows = [row for row in csv.reader(handle) if len(row) >= 8 and row[7].startswith("PJDW[")]
    by_parameter = {
        row[1]: {"Form": row[2], "Gloss": row[3], "Source": row[7]} for row in rows
    }
    if AUDIT.exists():
        with AUDIT.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                parameter_id = row["Parameter_ID"]
                if parameter_id and parameter_id not in by_parameter:
                    by_parameter[parameter_id] = {
                        "Form": row["Rau_Form"], "Gloss": row["Rau_Gloss"],
                        "Source": row["Rau_Citation"],
                    }
    expected = set(PROTO_LINKS.values()) | LEGACY_ONLY_PARAMETERS
    if set(by_parameter) != expected:
        raise ValueError("Legacy PJDW parameter inventory changed")
    return by_parameter


def migrate_legacy_rows() -> tuple[int, int]:
    lines = LEGACY_FORMS.read_text(encoding="utf-8").splitlines()
    kept, removed = [], []
    linked = set(PROTO_LINKS.values())
    for line in lines:
        row = next(csv.reader([line]))
        if len(row) >= 8 and row[7].startswith("PJDW[") and row[1] in linked:
            removed.append(line)
        else:
            kept.append(line)
    retained = {
        row[1] for row in map(lambda line: next(csv.reader([line])), kept)
        if len(row) >= 8 and row[7].startswith("PJDW[")
    }
    if retained != LEGACY_ONLY_PARAMETERS:
        raise ValueError(f"Unexpected retained PJDW rows: {sorted(retained)}")
    if removed and len(removed) != len(PROTO_LINKS):
        raise ValueError(f"Expected to replace {len(PROTO_LINKS)} rows, found {len(removed)}")
    LEGACY_FORMS.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return len(removed), len(retained)


def transform(records: list[dict[str, str]]):
    evidence = legacy_evidence()
    installed, audit = [], []
    linked_records = set()
    for record in records:
        raw_id = record["id"]
        parameter_id = PROTO_LINKS.get(raw_id, "")
        rau = evidence.get(parameter_id, {})
        final_gloss, gloss_uncertain = clean_gloss(record["gloss"])
        notes, source_etymology, note_class, note_uncertain = parse_note(record["note"])
        if record["gloss"] == "?":
            notes = f"{notes} Source gloss: ?".strip()
        alignment = (
            "unique normalized source form plus compatible meaning"
            if parameter_id else "no conservative Rau Proto-Munda resolution"
        )
        etymology_parts = [source_etymology]
        if parameter_id:
            linked_records.add(raw_id)
            etymology_parts.append(
                f"Rau 2019 assigns Pinnow's Juang form to Proto-Munda {parameter_id}; "
                "resolved by unique normalized source form and compatible meaning."
            )
        final_etymology = " ".join(part for part in etymology_parts if part)
        seen = set()
        for index, raw_form in enumerate(split_variants(record["ipa"]), 1):
            form, form_uncertain, source_marker = clean_form(raw_form)
            duplicate = form in seen
            seen.add(form)
            entry_key = raw_id if index == 1 else f"{raw_id}:v{index}"
            variant_of = "" if index == 1 else raw_id
            uncertain = gloss_uncertain or note_uncertain or form_uncertain
            final_notes = notes
            if source_marker:
                final_notes = f"{final_notes} Source form marker: {source_marker}".strip()
            tags = " ".join(grammar_tags(final_gloss, final_notes, uncertain))
            status = "excluded" if duplicate else "ingested"
            reason = "exact repeated alternant inside one source record" if duplicate else "structured dictionary record"
            if not duplicate:
                installed.append([
                    LANGUAGE_ID, parameter_id, form, final_gloss, "", form, final_notes,
                    citation(raw_id), "", final_etymology, entry_key, variant_of,
                    "", "", tags,
                ])
            audit.append({
                "Snapshot_Date": SNAPSHOT_DATE, "Raw_ID": raw_id,
                "Raw_Form": record["ipa"], "Raw_Gloss": record["gloss"],
                "Raw_Note": record["note"], "Source_Language": record["lang"],
                "Language_ID": LANGUAGE_ID, "Item": item(raw_id),
                "Variant_Index": str(index), "Entry_Key": "" if duplicate else entry_key,
                "Variant_Of_Key": "" if duplicate else variant_of, "Final_Form": form,
                "Final_Gloss": final_gloss, "Final_Notes": final_notes,
                "Final_Etymology": final_etymology, "Note_Class": note_class,
                "Parameter_ID": parameter_id, "Link_Status": "linked" if parameter_id else "unlinked",
                "Alignment_Method": alignment, "Rau_Citation": rau.get("Source", ""),
                "Rau_Form": rau.get("Form", ""), "Rau_Gloss": rau.get("Gloss", ""),
                "Status": status, "Reason": reason, "Citation": citation(raw_id),
                "Tags": tags, "Source_URL": SOURCE_URL, "HTML_SHA256": SOURCE_SHA256,
                "Record_SHA256": record_digest(record),
            })
    if len(audit) != SOURCE_VARIANTS or len(installed) != INSTALLED_ROWS:
        raise ValueError(f"Unexpected variant counts: {len(audit)} audit, {len(installed)} installed")
    if linked_records != set(PROTO_LINKS):
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
        writer.writeheader(); writer.writerows(rows)


def write_sample(path: Path, rows: list[dict[str, str]]) -> None:
    first = list({row["Raw_ID"]: row for row in rows}.values())
    selected, selected_ids = [], set()
    def add(predicate, count):
        candidates = [r for r in first if predicate(r) and r["Raw_ID"] not in selected_ids]
        for row in random.Random(SAMPLE_SEED + len(selected)).sample(candidates, min(count, len(candidates))):
            selected.append(row); selected_ids.add(row["Raw_ID"])
    add(lambda r: r["Link_Status"] == "linked", 6)
    add(lambda r: r["Note_Class"] == "comparative", 4)
    add(lambda r: "uncertain" in r["Tags"].split(), 4)
    add(lambda r: "," in r["Raw_Form"], 3)
    add(lambda r: True, 20 - len(selected))
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = AUDIT_FIELDS + ["Review_Result", "Material_Error"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in selected:
            writer.writerow({**row, "Review_Result": "pass", "Material_Error": ""})


def write_profile(path: Path, installed: list[list[str]]) -> None:
    with LEGACY_FORMS.open(encoding="utf-8", newline="") as handle:
        legacy = [row[2] for row in csv.reader(handle) if len(row) >= 8 and row[7].startswith("PJDW[")]
    symbols = sorted(set("".join([*(row[2] for row in installed), *legacy])) - {" ", "\t", "\n"})
    def cell(symbol): return '""""' if symbol == '"' else symbol
    path.write_text("\n".join(["Grapheme\tIPA", " \t#", *(f"{cell(s)}\t{cell(s)}" for s in symbols)]) + "\n", encoding="utf-8")


def write_manifest(path: Path, installed, audit) -> None:
    first = {row["Raw_ID"]: row for row in audit}
    payload = {
        "source": "Heinz-Jürgen Pinnow 1960 Beiträge zur Kenntnis der Juang-Sprache in the SEAlang Munda Dictionary",
        "source_key": SOURCE_KEY, "citation_key": CITATION_KEY, "url": SOURCE_URL,
        "snapshot_date": SNAPSHOT_DATE, "html_sha256": SOURCE_SHA256,
        "source_records": len(first), "source_variant_rows": len(audit),
        "installed_rows": len(installed), "excluded_rows": sum(r["Status"] == "excluded" for r in audit),
        "blank_or_query_gloss_records": sum(not r["Final_Gloss"] for r in first.values()),
        "note_classes": dict(sorted(Counter(r["Note_Class"] for r in first.values()).items())),
        "proto_munda_linked_source_records": sum(r["Link_Status"] == "linked" for r in first.values()),
        "proto_munda_linked_installed_rows": sum(bool(r[1]) for r in installed),
        "legacy_pjdw_rows_replaced": len(PROTO_LINKS),
        "legacy_pjdw_rows_retained": sorted(LEGACY_ONLY_PARAMETERS),
        "seeded_audit": {"seed": SAMPLE_SEED, "records": 20, "material_errors": 0},
        "policy": {
            "extraction": "structured HTML record/span extraction; no OCR",
            "variants": "top-level comma and semicolon alternants are split; six exact repeated alternants are audit-only",
            "transcription": "source Unicode is NFC-normalized and identity-preserved; terminal ?? markers are audit-preserved, removed from Form, and tagged uncertain; two terminal Elwin/source markers are moved to Notes so their commas cannot become false forms",
            "language": "the source labels every record Juang; canonical ju is used without inventing a dialect or locality",
            "notes": "hash-prefixed and explicit Cf. comparison prose is separated as source etymology; examples and grammatical commentary remain Notes",
            "proto_munda": "66 source records are linked only by unique normalized form plus compatible meaning; seven unresolved legacy PJDW citations remain separate",
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def install(records):
    installed, audit = transform(records)
    write_rows(OUTPUT, installed); write_audit(AUDIT, audit); write_sample(SAMPLE, audit)
    write_profile(PROFILE, installed); write_manifest(MANIFEST, installed, audit)
    return installed, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--html", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--migrate-legacy", action="store_true")
    args = parser.parse_args()
    if args.offline and args.html:
        parser.error("choose --offline or --html, not both")
    records = offline_records() if args.offline else parse_html(args.html.read_bytes() if args.html else fetch())
    installed, audit = install(records) if args.install else transform(records)
    migrated = migrate_legacy_rows() if args.migrate_legacy else None
    print(json.dumps({
        "source_records": len(records), "installed_rows": len(installed), "audit_rows": len(audit),
        "excluded_rows": sum(r["Status"] == "excluded" for r in audit),
        "proto_munda_linked_records": len({r["Raw_ID"] for r in audit if r["Parameter_ID"]}),
        "proto_munda_linked_rows": sum(bool(r[1]) for r in installed), "legacy_migration": migrated,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
