"""Prepare reviewed Haijong sections; never install an incomplete replacement.

This module supplies the grammar/table portion of the full recovery importer.
Specimens and native-script alignment remain separate review inputs. Running this
file validates those completed sections without writing canonical data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
SOURCE = "grierson1903haijong"
MYMENSINGH = "dialect:Hajong:lsi1903-haijong-mymensingh:Haijong%20%28Mymensingh%29"
SYLHET = "dialect:Hajong:lsi1903-haijong-sylhet:Haijong%20%28Sylhet%29"
LABEL_TAGS = {
    "Imperative": ["verb", "impv"],
    "Infinitive": ["verb", "inf"],
    "Infinitive of purpose": ["verb", "inf", "finalis"],
    "Present Participle": ["verb", "pres", "participle"],
    "Past Tense": ["verb", "pret"],
    "Verb Substantive": ["copula"],
    "Future": ["verb", "fut"],
    "root": ["verb", "stem"],
    "singular": ["sg"], "plural": ["pl"],
    "nominative": ["nom"], "oblique": ["obl"],
    "person:1": ["first-person"],
    "person:2": ["second-person"],
    "person:3": ["third-person"],
}
SECTION_TAGS = {
    "personal pronouns": ["pron", "personal"],
    "demonstratives": ["pron", "demonstrative"],
    "relatives": ["pron", "relative"],
    "interrogatives": ["interr"],
    "indefinites": ["pron", "indef"],
    "copula": ["verb", "copula", "pres"],
    "past copula": ["verb", "copula", "pret"],
    "verbal root": ["verb", "stem"],
    "present": ["verb", "pres"], "past": ["verb", "pret"],
    "imperative": ["verb", "impv"], "infinitive": ["verb", "inf"],
    "future": ["verb", "fut"],
    "conjunctive participle": ["verb", "conjunctive-participle"],
}


def load_reviewed(name: str, expected: int) -> list[dict]:
    records = [json.loads(line) for line in (PACKAGE / name).read_text().splitlines()]
    if len(records) != expected or len({r["entry_key"] for r in records}) != expected:
        raise ValueError(f"Incomplete or duplicate source units in {name}")
    for record in records:
        if not record.get("review_evidence"):
            raise ValueError(f"Missing independent review: {record['entry_key']}")
        if not isinstance(record["raw_forms"], list):
            raise ValueError(f"Unstructured alternatives: {record['entry_key']}")
    return records


def structured_tags(record: dict, kind: str) -> list[str]:
    tags = []
    for label in record.get("source_grammar_labels", []):
        if label not in LABEL_TAGS:
            raise ValueError(f"Unmapped grammatical label: {label}")
        tags.extend(LABEL_TAGS[label])
    if kind == "grammar":
        tags.extend(SECTION_TAGS.get(record["section"], []))
        # Case headings describe a constituent in the sentence, not the whole
        # phrase; retain their scope in the audit rather than mistagging it.
    else:
        tags.append(MYMENSINGH)
        if record["prompt"].isdigit() and int(record["prompt"]) <= 13:
            tags.append("num")
    if record.get("uncertainty"):
        tags.append("uncertain")
    return list(dict.fromkeys(tags))


def grammar_gloss(record: dict) -> str:
    """Supply lexical glosses only where explicit paradigm labels establish them."""
    if record["gloss"]:
        return record["gloss"]
    labels = record.get("source_grammar_labels", [])
    if record["section"] == "personal pronouns":
        person = next(x for x in labels if x.startswith("person:"))
        plural = "plural" in labels
        oblique = "oblique" in labels
        return {
            ("person:1", False, False): "I", ("person:1", False, True): "me",
            ("person:2", False, False): "thou", ("person:2", False, True): "thee",
            ("person:3", False, False): "he; she; it", ("person:3", False, True): "him; her; it",
            ("person:1", True, False): "we", ("person:1", True, True): "us",
            ("person:2", True, False): "you", ("person:2", True, True): "you",
            ("person:3", True, False): "they", ("person:3", True, True): "them",
        }[person, plural, oblique]
    return {"past copula": "was; were", "infinitive": "to strike", "future": "will strike"}[record["section"]]


def grammar_note(record: dict) -> str:
    section = record["section"]
    notes = {
        "nominative": "The source describes rā as frequent and ā as occasional nominative endings.",
        "accusative": "The source describes rā as optional and gē as the regular accusative ending, usable after a nominative form.",
        "ablative": "The source allows tan after either the noun or its genitive form.",
        "locative": "The source reports standard Bengali locatives alongside the additional endings mi, ni and mini.",
        "future": "The source reports both the standard future māriba and the additional form karaṅga; its proposed historical explanation is a source claim, not an assigned borrowing link.",
        "conjunctive participle": "The source describes locative mi as commonly added to the conjunctive participle in iyā.",
    }
    note = notes.get(section, "")
    if record["printed_page"] == 215 and section in {
        "past copula", "present", "past", "imperative", "infinitive", "future"
    }:
        note = (note + " " if note else "") + (
            "The source says these additional conjugations do not vary by person or number; "
            "the English gloss does not restrict the form to third person."
        )
    return note


def prepare_sections() -> tuple[list[list[str]], list[dict]]:
    """Return a partial staging assembly, explicitly not an installable source."""
    grammar = load_reviewed("grammar-reviewed.jsonl", 70)
    tables = load_reviewed("table-reviewed.jsonl", 245)
    expected_prompts = {str(i) for i in range(1, 242)} | {"51(a)", "52(a)", "60(a)", "61(a)"}
    if {r["prompt"] for r in tables} != expected_prompts:
        raise ValueError("Comparative-table prompt coverage changed")
    if Counter(r["record_kind"] for r in grammar) != {
        "target": 57, "bound_marker": 12, "comparison_control": 1
    }:
        raise ValueError("Grammar inclusion accounting changed")
    rows, audit = [], []
    for kind, records in (("grammar", grammar), ("table", tables)):
        for record in records:
            item = dict(record)
            keys = []
            included = kind == "table" or record["record_kind"] == "target"
            if included:
                for index, form in enumerate(record["raw_forms"], 1):
                    if not form.strip() or "�" in form:
                        raise ValueError(f"Invalid source form: {record['entry_key']}")
                    key = record["entry_key"] + (f":variant:{index}" if index > 1 else "")
                    keys.append(key)
                    locator = (
                        f"p. {record['printed_page']}, Haijong (Mymensingh) column, prompt {record['prompt']}"
                        if kind == "table" else
                        f"p. {record['printed_page']}, {record['section']}, example {record['entry_key'].rsplit(':', 1)[1]}"
                    )
                    rows.append([
                        "Hajong", "", unicodedata.normalize("NFC", form),
                        grammar_gloss(record) if kind == "grammar" else record["gloss"],
                        "", "", grammar_note(record) if kind == "grammar" else "",
                        f"{SOURCE}[{locator}]", "", "", key,
                        record["entry_key"] if index > 1 else "", "", "",
                        " ".join(structured_tags(record, kind)),
                    ])
            item["entry_keys"] = keys
            item["status"] = "staged" if keys else (
                "source_blank" if included else record["record_kind"]
            )
            audit.append(item)
    if len({r[10] for r in rows}) != len(rows):
        raise ValueError("Duplicate generated keys")
    return rows, audit


def prepare_roman_assembly() -> tuple[list[list[str]], list[dict]]:
    """Assemble all reviewed Roman evidence, pending native and final gates."""
    rows, audit = prepare_sections()
    specimens = [json.loads(line) for line in
                 (PACKAGE / "specimens-reconciled-20260926.jsonl").read_text().splitlines()]
    if len(specimens) != 566 or Counter(r["specimen"] for r in specimens) != {"I": 391, "II": 175}:
        raise ValueError("Specimen completeness census changed")
    if len({(r["printed_page"], r["specimen"], r["line"]) for r in specimens}) != 65:
        raise ValueError("Expected all65 interlinear lines")
    for record in specimens:
        if not record.get("independent_review") or len(record["forms"]) != 1:
            raise ValueError(f"Unreviewed specimen atom: {record['source_unit_key']}")
        expected_site = {"I": "Mymensingh", "II": "Sylhet"}[record["specimen"]]
        if record["site"] != expected_site:
            raise ValueError("Specimen/site mismatch")
        key = record["source_unit_key"]
        dialect = MYMENSINGH if record["specimen"] == "I" else SYLHET
        tags = [dialect] + (["uncertain"] if record.get("uncertainty") else [])
        locator = (f"p. {record['printed_page']}, specimen {record['specimen']}, "
                   f"line {record['line']}, word {record['word']}")
        rows.append([
            "Hajong", "", unicodedata.normalize("NFC", record["forms"][0]),
            record["gloss"], "", "", "", f"{SOURCE}[{locator}]", "", "", key,
            "", "", "", " ".join(tags),
        ])
        audit.append(dict(record, entry_keys=[key], status="staged"))
    keys = {r[10] for r in rows}
    if len(keys) != len(rows) or any(r[11] and r[11] not in keys for r in rows):
        raise ValueError("Duplicate keys or unresolved variant target")
    return rows, audit


def prepare_native_assembly() -> tuple[list[list[str]], list[dict]]:
    """Attach exact parallel-script evidence without splitting shared words."""
    rows, audit = prepare_roman_assembly()
    summary = json.loads((PACKAGE / "independent-native-alignment-summary-20260926.json").read_text())
    for name, expected in summary["input_sha256"].items():
        if hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Native review input changed: {name}")
    native = [json.loads(line) for line in
              (PACKAGE / "native-p216-reviewed-20260926.jsonl").read_text().splitlines()]
    lines = {r["native_unit_key"]: r for r in native}
    alignments = [json.loads(line) for line in
                  (PACKAGE / "native-specimen-I-alignments-20260926.jsonl").read_text().splitlines()]
    by_key = {r["source_unit_key"]: r for r in alignments}
    specimen_keys = {r[10] for r in rows if ":specimen-I:" in r[10]}
    if len(lines) != 26 or len(alignments) != 391 or set(by_key) != specimen_keys:
        raise ValueError("Incomplete native alignment coverage")
    audit_by_key = {r.get("source_unit_key", r.get("entry_key")): r for r in audit}
    for row in rows:
        alignment = by_key.get(row[10])
        if alignment is None:
            continue
        if unicodedata.normalize("NFC", alignment["roman"]) != row[2]:
            raise ValueError(f"Native alignment refers to stale Roman reading: {row[10]}")
        for locator in alignment["native_locators"]:
            line = lines[locator["native_unit_key"]]
            start, end = locator["character_start"], locator["character_end_exclusive"]
            if not 0 <= start < end <= len(line["native"]):
                raise ValueError("Invalid native character span")
        row[4] = unicodedata.normalize("NFC", alignment["recommended_native"])
        if row[4]:
            citations = [f"{SOURCE}[p. 216, native line {lines[x['native_unit_key']]['line']}]"
                         for x in alignment["native_locators"]]
            row[7] += ";" + ";".join(dict.fromkeys(citations))
        audit_by_key[row[10]]["native_alignment"] = alignment
    if sum(bool(r[4]) for r in rows) != 385:
        raise ValueError("Unexpected direct native span count")
    # Full line witnesses account for the six shared-word atoms as well as
    # punctuation; no native word is forced into an individual Roman atom.
    audit.extend(dict(r, status="parallel_native_witness") for r in native)
    return rows, audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", action="store_true", help="Write source-local proposed files; never install")
    args = parser.parse_args()
    rows, audit = prepare_native_assembly()
    if args.stage:
        with (PACKAGE / "full-proposed.csv").open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerows(rows)
        (PACKAGE / "full-proposed-audit.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in audit),
            encoding="utf-8",
        )
    print(json.dumps({
        "status": "partial_staging_only_not_installable",
        "source_units": len(audit), "candidate_rows": len(rows),
        "decisions": dict(Counter(r["status"] for r in audit)),
        "native_rows": sum(bool(r[4]) for r in rows),
        "remaining": ["final output audit", "focused tests", "metadata reconciliation"],
    }, indent=2))
