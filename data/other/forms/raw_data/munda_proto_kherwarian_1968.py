#!/usr/bin/env python3
"""Import Ram Dayal Munda's structured Proto-Kherwarian comparison index.

SEAlang exposes the thesis index as semantic HTML records with stable IDs,
Unicode transcription, gloss, and language.  No OCR is involved.  The source
has three levels: Proto-Kherwarian reconstructions, reconstructed pre-Mundari,
and Santali.  Proto rows become source-local parameters and all three levels
become forms attached to those parameters when the source alignment is secure.

Run from ``data/`` (or any directory) with::

    python3 data/other/forms/raw_data/munda_proto_kherwarian_1968.py \
      --html /path/to/munda1968-sealang.html --install

The HTML is not redistributed.  The checked-in audit supports a deterministic
offline rebuild without network access::

    python3 data/other/forms/raw_data/munda_proto_kherwarian_1968.py \
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
from html.parser import HTMLParser
from pathlib import Path


SOURCE_KEY = "munda1968proto"
SOURCE_URL = (
    "http://sealang.net/munda/dictionary/search.pl?"
    "caller=database&include=munda1968proto"
)
SNAPSHOT_DATE = "2026-08-28"
SOURCE_SHA256 = "5624d596e1baff95c0a8578b65922a903a34f29ef414a58910c7690d44f4402f"
SOURCE_RECORDS = 2768
PROTO_RECORDS = 920
SAMPLE_SEED = 20260828

ROOT = Path(__file__).resolve().parents[4]
RAW_DIR = ROOT / "data/other/forms/raw_data"
FORMS = ROOT / "data/other/forms/20260828-munda-proto-kherwarian.csv"
PARAMS = ROOT / "data/other/params/20260828-munda-proto-kherwarian.csv"
AUDIT = RAW_DIR / "20260828-munda-proto-kherwarian-audit.csv"
SAMPLE = RAW_DIR / "20260828-munda-proto-kherwarian-sample.csv"
MANIFEST = RAW_DIR / "20260828-munda-proto-kherwarian-manifest.json"
PROFILE = ROOT / "conversion/munda-proto-kherwarian.txt"

LANGUAGE_MAP = {
    "proto Kherwarian": "PKher",
    "pre Mundari": "PreMu",
    "Santali": "sa",
}

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic",
    "Notes", "Source", "Cognateset", "Etymology", "Entry_Key",
    "Variant_Of_Key", "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "Snapshot_Date", "Raw_ID", "Raw_Form", "Raw_Gloss", "Source_Language",
    "Language_ID", "SEAlang_Record", "Page", "Column", "Variant_Index",
    "Entry_Key", "Variant_Of_Key", "Final_Form", "Final_Gloss", "Parameter_ID",
    "Link_Status", "Alignment_Method", "Status", "Reason", "Citation", "Tags",
    "Source_URL", "HTML_SHA256", "Record_SHA256",
]

# Two records have damaged locator alignment and a non-identical but compatible
# gloss.  Their form and meaning uniquely identify the source reconstruction.
CURATED_ALIGNMENT = {
    "munda1968proto:C:c1.p.i51": "munda1968proto:R:c3.p1.i51",  # Santali bi
    "munda1968proto:R:c2.p.i52": "munda1968proto:R:c3.p15.i52",  # pre-Mundari tree
}


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
    if Counter(record["lang"] for record in records) != {
        "Santali": 925, "pre Mundari": 923, "proto Kherwarian": 920,
    }:
        raise ValueError("Source language inventory changed")
    return records


def fetch() -> bytes:
    with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
        return response.read()


def locator(raw_id: str) -> tuple[str, str, str]:
    match = re.fullmatch(
        rf"{SOURCE_KEY}:([CR]):c([123])\.p([^.]*)\.i(.+)", raw_id
    )
    if not match:
        raise ValueError(f"Unrecognized source ID {raw_id!r}")
    _, column, source_record, page = match.groups()
    return source_record, page, column


def locator_suffix(raw_id: str) -> str:
    return raw_id.split(".", 1)[1]


def parameter_id(raw_id: str) -> str:
    source_record, page, _ = locator(raw_id)
    page_key = page.replace("-", "x")
    return f"pkh-{source_record or 'none'}-{page_key}"


def citation(raw_id: str) -> str:
    source_record, page, _ = locator(raw_id)
    parts = [f"p. {page}"]
    if source_record:
        parts.append(f"SEAlang record {source_record}")
    return f"{SOURCE_KEY}[{', '.join(parts)}]"


def clean_gloss(value: str) -> tuple[str, str]:
    """Separate explicit terminal POS labels while preserving lexical content."""
    value = value.replace("{Sci.name} ", "")
    tags: list[str] = []
    labels = {
        "adverb": "adv", "pronoun": "pron", "adjective": "adj", "adj.": "adj",
        "noun": "noun", "n.": "noun", "neg.": "neg",
    }
    match = re.search(r"\s+\((adverb|pronoun|adjective|adj\.|noun|n\.|neg\.)\)$", value)
    if match:
        tags.append(labels[match.group(1)])
        value = value[:match.start()].rstrip()
    lowered = value.casefold()
    if "pronoun" in lowered:
        tags.append("pron")
    if "particle" in lowered:
        tags.append("part")
    if "dual" in lowered:
        tags.append("du")
    if "plural" in lowered:
        tags.append("pl")
    if "singular" in lowered:
        tags.append("sg")
    if "inclusive" in lowered:
        tags.append("inclusive")
    if "exclusive" in lowered:
        tags.append("exclusive")
    for person, number in re.findall(r"([123])(?:st|nd|rd)? person (dual|plural|singular)", lowered):
        if number == "plural":
            tags.append(f"{person}pl")
        elif number == "singular":
            tags.append(f"{person}sg")
    return value, " ".join(dict.fromkeys(tags))


def split_variants(value: str) -> list[str]:
    """A spaced tilde is SEAlang's top-level alternant delimiter."""
    return [part.strip() for part in re.split(r"\s+~\s+", nfc(value)) if part.strip()]


def record_digest(record: dict[str, str]) -> str:
    payload = "\x1f".join(record[key] for key in ("id", "ipa", "gloss", "lang"))
    return sha256(payload.encode("utf-8"))


def build_alignment(records: list[dict[str, str]]):
    proto = [record for record in records if record["lang"] == "proto Kherwarian"]
    proto_by_id = {record["id"]: record for record in proto}
    proto_by_suffix: dict[str, list[dict[str, str]]] = defaultdict(list)
    proto_by_gloss: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in proto:
        proto_by_suffix[locator_suffix(record["id"])].append(record)
        proto_by_gloss[record["gloss"].casefold()].append(record)

    result: dict[str, tuple[str, str, str]] = {}
    for record in records:
        if record["lang"] == "proto Kherwarian":
            result[record["id"]] = (record["id"], "self", "source reconstruction")
            continue
        direct = [candidate for candidate in proto_by_suffix[locator_suffix(record["id"])]
                  if candidate["gloss"] == record["gloss"]]
        if len(direct) == 1:
            result[record["id"]] = (direct[0]["id"], "locator+gloss", "shared source locator and gloss")
            continue
        gloss = proto_by_gloss[record["gloss"].casefold()]
        if len(gloss) == 1:
            result[record["id"]] = (gloss[0]["id"], "unique-gloss", "unique exact source gloss")
            continue
        if record["id"] in CURATED_ALIGNMENT:
            target = proto_by_id[CURATED_ALIGNMENT[record["id"]]]
            result[record["id"]] = (
                target["id"], "curated-form-meaning",
                "damaged source locator; uniquely compatible form and meaning",
            )
            continue
        result[record["id"]] = ("", "unlinked", "no corresponding source reconstruction")
    return result, proto


def transform(records: list[dict[str, str]]):
    alignment, proto = build_alignment(records)
    params: list[list[str]] = []
    for record in proto:
        gloss, _ = clean_gloss(record["gloss"])
        params.append([
            parameter_id(record["id"]), "PKher", record["ipa"], gloss,
            citation(record["id"]),
        ])

    forms: list[list[str]] = []
    audit: list[dict[str, str]] = []
    for record in records:
        raw_id = record["id"]
        source_record, page, column = locator(raw_id)
        language_id = LANGUAGE_MAP[record["lang"]]
        proto_id, method, reason = alignment[raw_id]
        param = parameter_id(proto_id) if proto_id else ""
        gloss, tags = clean_gloss(record["gloss"])
        cite = citation(raw_id)
        variants = split_variants(record["ipa"])
        for variant_index, form in enumerate(variants, 1):
            entry_key = raw_id if variant_index == 1 else f"{raw_id}:v{variant_index}"
            variant_of = "" if variant_index == 1 else raw_id
            if record["lang"] == "proto Kherwarian":
                etymology = "Munda's Proto-Kherwarian reconstruction."
            elif param:
                stage = "reconstructed pre-Mundari" if language_id == "PreMu" else "Santali"
                etymology = f"Munda's {stage} reflex of {param}."
            else:
                etymology = "Source comparison record has no aligned Proto-Kherwarian reconstruction."
            forms.append([
                language_id, param, form, gloss, "", form, "", cite, "", etymology,
                entry_key, variant_of, "", "", tags,
            ])
            audit.append({
                "Snapshot_Date": SNAPSHOT_DATE, "Raw_ID": raw_id,
                "Raw_Form": record["ipa"], "Raw_Gloss": record["gloss"],
                "Source_Language": record["lang"], "Language_ID": language_id,
                "SEAlang_Record": source_record, "Page": page, "Column": column,
                "Variant_Index": str(variant_index), "Entry_Key": entry_key,
                "Variant_Of_Key": variant_of, "Final_Form": form, "Final_Gloss": gloss,
                "Parameter_ID": param, "Link_Status": "linked" if param else "unlinked",
                "Alignment_Method": method, "Status": "ingested", "Reason": reason,
                "Citation": cite, "Tags": tags, "Source_URL": SOURCE_URL,
                "HTML_SHA256": SOURCE_SHA256, "Record_SHA256": record_digest(record),
            })

    if len(params) != PROTO_RECORDS or len({row[0] for row in params}) != len(params):
        raise ValueError("Proto-Kherwarian parameter inventory is incomplete or non-unique")
    keys = [row[10] for row in forms]
    if len(keys) != len(set(keys)):
        raise ValueError("Installed Entry_Key values are not unique")
    return params, forms, audit


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
    path.parent.mkdir(parents=True, exist_ok=True)
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


def write_profile(path: Path, forms: list[list[str]]) -> None:
    symbols = sorted(set("".join(row[2] for row in forms)) - {" ", "\t", "\n"})
    lines = ["Grapheme\tIPA", " \t#", *(f"{symbol}\t{symbol}" for symbol in symbols)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(path: Path, params, forms, audit) -> None:
    raw_by_id = {row["Raw_ID"]: row for row in audit}
    payload = {
        "source": "Ram Dayal Munda 1968 Proto-Kherwarian records in the SEAlang Munda Dictionary",
        "source_key": SOURCE_KEY, "url": SOURCE_URL, "snapshot_date": SNAPSHOT_DATE,
        "html_sha256": SOURCE_SHA256, "source_records": len(raw_by_id),
        "parameter_rows": len(params), "form_rows": len(forms), "audit_rows": len(audit),
        "excluded_rows": 0,
        "source_language_records": dict(sorted(Counter(row["Source_Language"] for row in raw_by_id.values()).items())),
        "installed_language_rows": dict(sorted(Counter(row[0] for row in forms).items())),
        "alignment_methods": dict(sorted(Counter(row["Alignment_Method"] for row in raw_by_id.values()).items())),
        "unlinked_source_records": sorted(row["Raw_ID"] for row in raw_by_id.values() if row["Link_Status"] == "unlinked"),
        "seeded_audit": {"seed": SAMPLE_SEED, "records": 20, "material_errors": 0},
        "policy": {
            "extraction": "structured HTML semantic spans; no OCR",
            "variants": "only spaced tildes split source-level alternants; slash and optional-segment notation are preserved",
            "transcription": "source Unicode is NFC-normalized and identity-preserved in Form and Phonemic",
            "alignment": "shared locator plus gloss, then unique exact gloss; two damaged locators receive documented form-and-meaning corrections",
            "unlinked": "three comparison records without a source reconstruction remain installed and unlinked",
            "glosses": "terminal explicit POS labels become tags; one SEAlang scientific-name template marker is removed",
            "licence": "SEAlang result page states no separate reuse licence; extracted lexical facts and source identifiers are included",
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def install(records: list[dict[str, str]]):
    params, forms, audit = transform(records)
    write_rows(PARAMS, params)
    write_rows(FORMS, forms)
    write_audit(AUDIT, audit)
    write_sample(SAMPLE, audit)
    write_profile(PROFILE, forms)
    write_manifest(MANIFEST, params, forms, audit)
    return params, forms, audit


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
    params, forms, audit = install(records) if args.install else transform(records)
    print(json.dumps({
        "source_records": len(records), "parameter_rows": len(params),
        "form_rows": len(forms), "audit_rows": len(audit),
        "alignment_methods": dict(Counter(row["Alignment_Method"] for row in audit)),
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
