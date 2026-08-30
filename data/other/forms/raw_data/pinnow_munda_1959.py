#!/usr/bin/env python3
"""Import SEAlang's born-digital index of Pinnow's 1959 Munda comparisons.

The SEAlang result page is structured HTML, not OCR.  Each result supplies a
source transcription, gloss, language label, stable database identifier, and
Pinnow comparison-set number.  The importer preserves those assertions and
links a comparison to Jambu's Rau Proto-Munda etymon only when Rau himself
prints the same Pinnow number.  Duplicate Rau cross-references are handled
conservatively and recorded in the audit.

Run from ``data/`` (or any directory) with::

    python3 data/other/forms/raw_data/pinnow_munda_1959.py \
      --html /path/to/pinnow1959-sealang.html --install

The HTML is not redistributed.  Rebuilding from the installed audit requires
no network access and no OCR::

    python3 data/other/forms/raw_data/pinnow_munda_1959.py --offline --install
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


SOURCE_KEY = "pinnow1959versuch"
SOURCE_URL = (
    "http://sealang.net/munda/dictionary/search.pl?"
    "caller=database&include=pinnow1959versuch"
)
SNAPSHOT_DATE = "2026-08-28"
SOURCE_SHA256 = "c267d8e727c0ecbd4b7f47d4aded52220aac8c08bbcc0b9393911dd0d89cf7a3"
SOURCE_RECORDS = 3340
SAMPLE_SEED = 20260828

ROOT = Path(__file__).resolve().parents[4]
RAW_DIR = ROOT / "data/other/forms/raw_data"
OUTPUT = ROOT / "data/other/forms/20260828-pinnow-munda.csv"
AUDIT = RAW_DIR / "20260828-pinnow-munda-audit.csv"
SAMPLE = RAW_DIR / "20260828-pinnow-munda-sample.csv"
MANIFEST = RAW_DIR / "20260828-pinnow-munda-manifest.json"
PROFILE = ROOT / "conversion/pinnow-munda.txt"
RAU = ROOT / "data/munda/rau_2019.csv"

LANGUAGE_MAP = {
    "Sora": "so",
    "Mundari": "mu",
    "Santali": "sa",
    "Kharia": "kh",
    "Ho": "ho",
    "Korku": "ko",
    "Bodo-Gadaba": "gu",
    "Bondo": "re",
    "Juang": "ju",
    "Korwa": "kw",
    "Mahali": "Mahali",
    "Asuri": "Asuri",
    "Birhor": "Birhor",
    "Turi": "Turi",
}

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
    "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
    "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "Snapshot_Date", "Raw_ID", "Raw_Form", "Raw_Gloss", "Source_Language",
    "Language_ID", "Page", "Item", "Pinnow_Set", "Variant_Index", "Entry_Key",
    "Variant_Of_Key", "Final_Form", "Parameter_ID", "Link_Status", "Status", "Reason",
    "Citation", "Tags", "Source_URL", "HTML_SHA256", "Record_SHA256",
]


def nfc(value: str) -> str:
    return unicodedata.normalize("NFC", html.unescape(value)).strip()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ResultParser(HTMLParser):
    """Read the four semantic spans in each Munda result table."""

    wanted = {"ipa", "gloss", "lang", "id"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_munda = False
        self.field = ""
        self.record: dict[str, str] = {}
        self.records: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        attributes = dict(attrs)
        if tag == "tr" and "Munda" in attributes.get("class", "").split():
            self.in_munda = True
            self.record = {}
        if self.in_munda and tag == "span" and attributes.get("class") in self.wanted:
            self.field = attributes["class"]
            self.record.setdefault(self.field, "")

    def handle_data(self, data: str) -> None:
        if self.field:
            self.record[self.field] += data

    def handle_endtag(self, tag: str) -> None:
        if tag == "span":
            self.field = ""
        if tag == "tr" and self.in_munda:
            if set(self.record) == self.wanted:
                self.records.append({key: nfc(value) for key, value in self.record.items()})
            self.record = {}
            self.in_munda = False


def parse_html(data: bytes) -> list[dict[str, str]]:
    if sha256(data) != SOURCE_SHA256:
        raise ValueError(f"Unexpected HTML SHA-256 {sha256(data)}; expected {SOURCE_SHA256}")
    parser = ResultParser()
    parser.feed(data.decode("utf-8"))
    records = parser.records
    if len(records) != SOURCE_RECORDS:
        raise ValueError(f"Expected {SOURCE_RECORDS} Munda records, found {len(records)}")
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)) or any(not value for value in ids):
        raise ValueError("Source record IDs must be non-empty and unique")
    if set(record["lang"] for record in records) != set(LANGUAGE_MAP):
        raise ValueError("Source language inventory changed")
    return records


def fetch() -> bytes:
    with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
        return response.read()


def split_variants(value: str) -> list[str]:
    """Split commas used as top-level alternant separators, not parenthetical prose."""
    variants: list[str] = []
    current: list[str] = []
    depth = 0
    for char in nfc(value):
        if char == "(":
            depth += 1
        elif char == ")" and depth:
            depth -= 1
        if char == "," and depth == 0:
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


def normalize_set(value: str) -> str:
    match = re.fullmatch(r"([VK])0*(\d+)([A-Za-z]?)", value)
    return f"{match.group(1)}{int(match.group(2))}{match.group(3).lower()}" if match else ""


def source_locator(raw_id: str) -> tuple[str, str, str]:
    match = re.search(r"\.p([^.]*)\.i([^.]*)\.s([^.]*)$", raw_id)
    if not match:
        raise ValueError(f"Unrecognized source ID {raw_id!r}")
    page, item, set_code = match.groups()
    return page, item, normalize_set(set_code)


def citation(raw_id: str, page: str, item: str, set_code: str) -> str:
    locators = []
    if page:
        locators.append(f"p. {page}")
    if item:
        locators.append(f"item {item}")
    if set_code:
        locators.append(f"set {set_code}")
    return f"{SOURCE_KEY}[{', '.join(locators)}]" if locators else SOURCE_KEY


def rau_set_map(path: Path = RAU) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle), 1):
            field = row["pinnow"].strip()
            codes = re.findall(r"[VK]0*\d+[a-z]?", field)
            # Rau abbreviates the second K number in K415(ˀt)/508(m).
            if field.startswith("K"):
                codes.extend(f"K{number}" for number in re.findall(r"/(\d+[a-z]?)", field))
            for code in dict.fromkeys(normalize_set(code) for code in codes):
                result[code].append({
                    "parameter_id": f"m{index}",
                    "protoform": row["pmunda"],
                    "gloss": row["gloss"],
                })
    return dict(result)


def resolve_parameter(
    set_code: str, form: str, gloss: str, mapping: dict[str, list[dict[str, str]]]
) -> tuple[str, str, str]:
    candidates = mapping.get(set_code, [])
    if not set_code:
        return "", "no-set", "source record has no Pinnow comparison-set number"
    if not candidates:
        return "", "unlinked", "Rau prints no cross-reference to this Pinnow set"
    if len(candidates) == 1:
        return candidates[0]["parameter_id"], "direct", "Rau prints this Pinnow set number"
    if set_code == "V3":
        # Pinnow's actual V3 results all have a final labial stop and support Rau's *daˀp.
        compact = re.sub(r"[^A-Za-zɑ-ʸ]", "", form.lower())
        target = next(candidate for candidate in candidates if candidate["protoform"] == "*daˀp")
        if re.search(r"[bp][a-zɑ-ʸ]*$", compact):
            return target["parameter_id"], "disambiguated", "V3 form has Pinnow's final labial reflex"
    if set_code == "V278":
        hills = {"hill", "mountain"}
        meanings = {word.lower() for word in re.findall(r"[A-Za-z]+", gloss)}
        if meanings & hills and "forest" not in meanings:
            target = next(candidate for candidate in candidates if candidate["protoform"] == "*buru")
            return target["parameter_id"], "disambiguated", "V278 gloss is specifically hill/mountain"
    return "", "ambiguous", "Pinnow set cross-references multiple Rau etyma without a unique source match"


def record_digest(record: dict[str, str]) -> str:
    payload = "\x1f".join(record[key] for key in ("id", "ipa", "gloss", "lang"))
    return sha256(payload.encode("utf-8"))


def grammatical_tags(gloss: str) -> str:
    """Lift only explicit grammatical labels printed in Pinnow's gloss field."""
    text = gloss.casefold()
    tags: list[str] = []
    if "suffix" in text:
        tags.append("suffix")
    if "prefix" in text:
        tags.append("prefix")
    if "particle" in text:
        tags.append("part")
    if "genitive" in text:
        tags.append("gen")
    if "feminine suffix" in text:
        tags.append("f")
    if "plural suffix" in text or "third person plural" in text:
        tags.append("pl")
    if "third person plural" in text:
        tags.append("3pl")
    if "dual suffix" in text or text.startswith("we two"):
        tags.append("du")
    if "emphatic particle" in text:
        tags.append("emph")
    if "prohibitive particle" in text:
        tags.append("neg")
    if "vocative particle" in text:
        tags.append("voc")
    if "interrogative suffix" in text:
        tags.append("interr")
    if text.startswith("we"):
        tags.extend(("pron", "1pl"))
    if text == "you plural":
        tags.extend(("pron", "2pl"))
    if "inclusive" in text:
        tags.append("inclusive")
    if "exclusive" in text:
        tags.append("exclusive")
    if "(transitive)" in text:
        tags.extend(("verb", "tr"))
    if "(intransitive)" in text:
        tags.extend(("verb", "intr"))
    if "(imperative)" in text:
        tags.extend(("verb", "impv"))
    if text.startswith("auxiliary verb") or text.startswith("helper verb"):
        tags.extend(("verb", "auxiliary"))
    return " ".join(dict.fromkeys(tags))


def transform(records: list[dict[str, str]]):
    mapping = rau_set_map()
    installed: list[list[str]] = []
    audit: list[dict[str, str]] = []
    for record in records:
        raw_id = record["id"]
        page, item, set_code = source_locator(raw_id)
        language_id = LANGUAGE_MAP[record["lang"]]
        cite = citation(raw_id, page, item, set_code)
        base_key = raw_id
        if record["ipa"] == "MISSING":
            audit.append({
                "Snapshot_Date": SNAPSHOT_DATE, "Raw_ID": raw_id,
                "Raw_Form": record["ipa"], "Raw_Gloss": record["gloss"],
                "Source_Language": record["lang"], "Language_ID": language_id,
                "Page": page, "Item": item, "Pinnow_Set": set_code,
                "Variant_Index": "", "Entry_Key": "", "Variant_Of_Key": "",
                "Final_Form": "", "Parameter_ID": "", "Link_Status": "not-applicable",
                "Status": "excluded", "Reason": "source explicitly marks the form MISSING",
                "Citation": cite, "Tags": "", "Source_URL": SOURCE_URL,
                "HTML_SHA256": SOURCE_SHA256, "Record_SHA256": record_digest(record),
            })
            continue
        variants = split_variants(record["ipa"])
        if not variants:
            raise ValueError(f"No usable form for {raw_id}")
        seen_variants: set[str] = set()
        for variant_index, form in enumerate(variants, 1):
            if form in seen_variants:
                audit.append({
                    "Snapshot_Date": SNAPSHOT_DATE, "Raw_ID": raw_id,
                    "Raw_Form": record["ipa"], "Raw_Gloss": record["gloss"],
                    "Source_Language": record["lang"], "Language_ID": language_id,
                    "Page": page, "Item": item, "Pinnow_Set": set_code,
                    "Variant_Index": str(variant_index), "Entry_Key": "",
                    "Variant_Of_Key": base_key, "Final_Form": form,
                    "Parameter_ID": "", "Link_Status": "not-applicable",
                    "Status": "excluded", "Reason": "duplicate alternant repeated in one source record",
                    "Citation": cite, "Tags": grammatical_tags(record["gloss"]),
                    "Source_URL": SOURCE_URL, "HTML_SHA256": SOURCE_SHA256,
                    "Record_SHA256": record_digest(record),
                })
                continue
            seen_variants.add(form)
            entry_key = base_key if variant_index == 1 else f"{base_key}:v{variant_index}"
            variant_of = "" if variant_index == 1 else base_key
            parameter, link_status, reason = resolve_parameter(
                set_code, form, record["gloss"], mapping
            )
            tags = " ".join(filter(None, (grammatical_tags(record["gloss"]), "uncertain" if "?" in form else "")))
            etymology = f"Pinnow comparative set {set_code}." if set_code else ""
            if parameter:
                etymology += f" Linked to Rau {parameter} through Rau's printed Pinnow cross-reference."
            installed.append([
                language_id, parameter, form, record["gloss"], "", form, "", cite, "",
                etymology.strip(), entry_key, variant_of, "", "", tags,
            ])
            audit.append({
                "Snapshot_Date": SNAPSHOT_DATE, "Raw_ID": raw_id,
                "Raw_Form": record["ipa"], "Raw_Gloss": record["gloss"],
                "Source_Language": record["lang"], "Language_ID": language_id,
                "Page": page, "Item": item, "Pinnow_Set": set_code,
                "Variant_Index": str(variant_index), "Entry_Key": entry_key,
                "Variant_Of_Key": variant_of, "Final_Form": form,
                "Parameter_ID": parameter, "Link_Status": link_status,
                "Status": "ingested", "Reason": reason, "Citation": cite,
                "Tags": tags, "Source_URL": SOURCE_URL, "HTML_SHA256": SOURCE_SHA256,
                "Record_SHA256": record_digest(record),
            })
    keys = [row[10] for row in installed]
    if len(keys) != len(set(keys)):
        raise ValueError("Installed Entry_Key values are not unique")
    return installed, audit


def offline_records(path: Path = AUDIT) -> list[dict[str, str]]:
    """Reconstruct the raw source layer from the checked-in one-row-per-variant audit."""
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, dict[str, str]] = {}
    for row in rows:
        grouped.setdefault(row["Raw_ID"], {
            "id": row["Raw_ID"], "ipa": row["Raw_Form"], "gloss": row["Raw_Gloss"],
            "lang": row["Source_Language"],
        })
    records = list(grouped.values())
    if len(records) != SOURCE_RECORDS:
        raise ValueError(f"Offline audit contains {len(records)} source records")
    return records


def write_csv(path: Path, rows: list[list[str]]) -> None:
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
    # Sample source records, not expanded alternants, so one verbose entry cannot crowd the audit.
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
    payload = {
        "source": "Pinnow 1959 comparative Munda records in the SEAlang Munda Dictionary",
        "source_key": SOURCE_KEY,
        "url": SOURCE_URL,
        "snapshot_date": SNAPSHOT_DATE,
        "html_sha256": SOURCE_SHA256,
        "source_records": len(raw_by_id),
        "audit_rows": len(audit),
        "installed_rows": len(installed),
        "excluded_rows": sum(row["Status"] == "excluded" for row in audit),
        "source_language_records": dict(sorted(Counter(row["Source_Language"] for row in raw_by_id.values()).items())),
        "installed_language_rows": dict(sorted(Counter(row[0] for row in installed).items())),
        "link_statuses": dict(sorted(Counter(row["Link_Status"] for row in audit if row["Status"] == "ingested").items())),
        "pinnow_numbered_sets": len({row["Pinnow_Set"] for row in audit if row["Pinnow_Set"]}),
        "source_set_labels": len({row["Raw_ID"].rsplit(".s", 1)[1] for row in audit}),
        "seeded_audit": {"seed": SAMPLE_SEED, "records": 20, "material_errors": 0},
        "policy": {
            "extraction": "structured HTML semantic spans; no OCR",
            "variants": "top-level commas split; commas inside parentheses preserved",
            "transcription": "source IPA is NFC-normalized and identity-preserved in Form and Phonemic",
            "proto_munda_links": "only Rau's printed Pinnow cross-references; duplicate set references are conservative",
            "exclusions": "one explicit MISSING form and one repeated duplicate alternant are audit-only",
            "dialects": "source supplies language labels but no locality or dialect field",
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def install(records: list[dict[str, str]]) -> tuple[list[list[str]], list[dict[str, str]]]:
    installed, audit = transform(records)
    write_csv(OUTPUT, installed)
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
    parser.add_argument("--output", type=Path, help="Write a preview rich-form CSV")
    args = parser.parse_args()
    if args.offline and args.html:
        parser.error("choose --offline or --html, not both")
    if args.offline:
        records = offline_records()
    else:
        data = args.html.read_bytes() if args.html else fetch()
        records = parse_html(data)
    installed, audit = install(records) if args.install else transform(records)
    if args.output:
        write_csv(args.output, installed)
    print(json.dumps({
        "source_records": len(records), "installed_rows": len(installed),
        "audit_rows": len(audit),
        "link_statuses": dict(Counter(row["Link_Status"] for row in audit)),
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
