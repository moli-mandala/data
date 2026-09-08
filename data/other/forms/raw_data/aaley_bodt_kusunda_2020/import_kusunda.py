#!/usr/bin/env python3
"""Install the pinned Lexibank Aaley--Bodt Kusunda v2.1 wordlist.

The source has 250 elicitation prompts with three lexical layers: two speaker
attestations and the authors' tentative ground-form reconstruction.  The
released CLDF contains one selected form per non-empty prompt/layer cell.  This
importer maps every released form to the canonical Jambu Kusunda language and
writes a 750-cell audit so missing responses and upstream form selection remain
visible.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
PACKAGE = Path(__file__).resolve().parent
SNAPSHOT = PACKAGE / "snapshot"
MANIFEST = PACKAGE / "manifest.json"
INSTALLED = ROOT / "data/other/forms/20260901-aaley-bodt-kusunda.csv"
AUDIT = PACKAGE / "20260901-aaley-bodt-kusunda-audit.csv"
SOURCE_KEY = "aaley-bodt2020kusunda"
SAMPLE_SEED = 20260901

LECTS = {
    "ProtoKusunda": {
        "raw_value": "Kusunda",
        "raw_comment": "Comments1",
        "provenance_type": "reconstruction",
        "speaker": "",
        "locator": "ground-form reconstruction",
    },
    "KusundaGM": {
        "raw_value": "Gyani Maiya",
        "raw_comment": "Comments2",
        "provenance_type": "speaker",
        "speaker": "Gyani Maiya Sen Kusunda",
        "locator": "speaker Gyani Maiya Sen Kusunda",
    },
    "KusundaK": {
        "raw_value": "Kamala",
        "raw_comment": "Comments3",
        "provenance_type": "speaker",
        "speaker": "Kamala Khatri (Sen Kusunda)",
        "locator": "speaker Kamala Khatri (Sen Kusunda)",
    },
}

INSTALLED_COLUMNS = 15
AUDIT_FIELDS = [
    "Audit_ID",
    "Raw_ID",
    "Concept",
    "Concepticon_ID",
    "Source_Lect",
    "Provenance_Type",
    "Speaker",
    "Raw_Value",
    "Raw_Comment",
    "Upstream_Form_ID",
    "Upstream_Value",
    "Upstream_Form",
    "Upstream_Segments",
    "Status",
    "Reason",
    "Language_ID",
    "Gloss",
    "Tags",
    "Source",
    "Entry_Key",
    "Unresolved",
    "Review_State",
]


def read_dicts(path: Path, *, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(snapshot: Path = SNAPSHOT) -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for filename, expected in manifest["files"].items():
        path = snapshot / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing pinned upstream file: {path}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Checksum mismatch for {path}: {actual} != {expected}")
    return manifest


def clean_comment(value: str) -> str:
    value = unicodedata.normalize("NFC", value.strip())
    if value in {"#ERROR!", "#NAME?"}:
        return ""
    return value


def source_uncertainty(raw_value: str, raw_comment: str) -> str:
    text = f"{raw_value} {raw_comment}".casefold()
    reasons = []
    if raw_value.strip() in {"", "∅"}:
        reasons.append("missing-form")
    if "#error!" in text or "#name?" in text:
        reasons.append("upstream-spreadsheet-error-in-comment")
    if any(
        marker in text
        for marker in (
            "no consensus",
            "not clear",
            "unclear",
            "neologism",
            "newly coined",
            "but told by uday",
            "from gorkha",
            "does not exist",
            "do not remember",
            "don’t remember",
            "seems to",
        )
    ):
        reasons.append("source-uncertainty")
    return ";".join(dict.fromkeys(reasons))


def grammatical_tags(gloss: str, raw_comment: str, unresolved: str) -> list[str]:
    tags: list[str] = []
    folded = gloss.casefold()
    if folded.startswith("to "):
        tags.append("verb")
    elif folded.startswith("the "):
        tags.append("noun")

    if "[intransitive]" in folded:
        tags.append("intr")
    if "[transitive]" in folded:
        tags.append("tr")

    person = re.search(r"\[(first|second|third) person (singular|plural)", folded)
    if person:
        tags.extend(
            [
                "pron",
                {"first": "1", "second": "2", "third": "3"}[person.group(1)]
                + {"singular": "sg", "plural": "pl"}[person.group(2)],
            ]
        )
        if "inclusive" in folded:
            tags.append("inclusive")

    if folded in {
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "twenty",
        "hundred",
    }:
        tags.append("num")
    if folded in {"this", "that"}:
        tags.extend(["determiner", "demonstrative"])
    if folded in {"what", "who"}:
        tags.extend(["pron", "interr"])
    elif folded == "where":
        tags.extend(["adv", "interr"])

    # Only mark the selected row as a Nepali loan when the source comment
    # directly presents that cell's response as deriving from NEP.  Mentions of
    # a different alternate later in a comment do not trigger this rule.
    if raw_comment.lstrip().upper().startswith("< NEP"):
        tags.extend(["loanword", "loan:Nepali"])
    if unresolved and "missing-form" not in unresolved:
        tags.append("uncertain")
    return list(dict.fromkeys(tags))


def note_for(upstream: dict[str, str], raw_comment: str) -> str:
    notes = []
    if upstream["Value"] != upstream["Form"]:
        notes.append(f"Full source value: {upstream['Value']}")
    cleaned = clean_comment(raw_comment)
    if cleaned:
        notes.append(cleaned)
    return "; ".join(notes)


def build(snapshot: Path = SNAPSHOT) -> tuple[list[list[str]], list[dict[str, str]]]:
    verify_snapshot(snapshot)
    raw = read_dicts(snapshot / "Kusunda_2019_250_lexical_items.tsv", delimiter="\t")
    upstream_forms = read_dicts(snapshot / "forms.csv")
    parameters = {row["ID"]: row for row in read_dicts(snapshot / "parameters.csv")}

    assert len(raw) == 250
    assert len(upstream_forms) == 662
    assert len(parameters) == 230
    assert [row["ID"] for row in raw] == [str(i) for i in range(1, 251)]

    forms_by_cell: dict[tuple[str, str], dict[str, str]] = {}
    for row in upstream_forms:
        key = (row["Language_ID"], row["Parameter_ID"])
        if key in forms_by_cell:
            raise ValueError(f"Multiple released forms for one source cell: {key}")
        if row["Source"] != "Bodt2019b" or not row["Form"]:
            raise ValueError(f"Unexpected upstream form row: {row}")
        forms_by_cell[key] = row

    installed: list[list[str]] = []
    audit: list[dict[str, str]] = []
    ingested_audit_ids: list[str] = []

    for raw_row in raw:
        raw_id = raw_row["ID"]
        parameter_id = next(
            (pid for pid in parameters if pid.split("_", 1)[0] == raw_id),
            "",
        )
        concepticon_id = parameters.get(parameter_id, {}).get("Concepticon_ID", "")
        gloss = parameters.get(parameter_id, {}).get("Name", raw_row["ENGLISH"])
        for source_lect, lect in LECTS.items():
            raw_value = unicodedata.normalize("NFC", raw_row[lect["raw_value"]].strip())
            raw_comment = unicodedata.normalize("NFC", raw_row[lect["raw_comment"]].strip())
            upstream = forms_by_cell.get((source_lect, parameter_id))
            audit_id = f"{raw_id}:{source_lect}"
            unresolved = source_uncertainty(raw_value, raw_comment)
            tags = grammatical_tags(gloss, raw_comment, unresolved)

            if upstream is None:
                status = "skipped"
                reason = (
                    "source cell is blank"
                    if not raw_value
                    else "source marks no installable form"
                    if raw_value == "∅"
                    else "upstream CLDF excludes cell after form cleaning"
                )
                entry_key = ""
                source = ""
            else:
                status = "ingested"
                reason = "released CLDF lexeme"
                entry_key = upstream["ID"]
                source = (
                    f"{SOURCE_KEY}[concept {raw_id}, {lect['locator']}, "
                    f"upstream {upstream['ID']}]"
                )
                installed.append(
                    [unicodedata.normalize("NFC", value) for value in [
                        "Kusunda",
                        "",
                        unicodedata.normalize("NFC", upstream["Form"]),
                        gloss,
                        "",
                        unicodedata.normalize("NFC", upstream["Form"]),
                        note_for(upstream, raw_comment),
                        source,
                        "",
                        "",
                        entry_key,
                        "",
                        "",
                        "",
                        " ".join(tags),
                    ]]
                )
                ingested_audit_ids.append(audit_id)

            audit.append(
                {
                    "Audit_ID": audit_id,
                    "Raw_ID": raw_id,
                    "Concept": raw_row["ENGLISH"],
                    "Concepticon_ID": concepticon_id,
                    "Source_Lect": source_lect,
                    "Provenance_Type": lect["provenance_type"],
                    "Speaker": lect["speaker"],
                    "Raw_Value": raw_value,
                    "Raw_Comment": raw_comment,
                    "Upstream_Form_ID": upstream["ID"] if upstream else "",
                    "Upstream_Value": upstream["Value"] if upstream else "",
                    "Upstream_Form": upstream["Form"] if upstream else "",
                    "Upstream_Segments": upstream["Segments"] if upstream else "",
                    "Status": status,
                    "Reason": reason,
                    "Language_ID": "Kusunda" if upstream else "",
                    "Gloss": gloss if upstream else "",
                    "Tags": " ".join(tags) if upstream else "",
                    "Source": source,
                    "Entry_Key": entry_key,
                    "Unresolved": unresolved,
                    "Review_State": "not-sampled",
                }
            )

    sampled = set(random.Random(SAMPLE_SEED).sample(ingested_audit_ids, 20))
    for row in audit:
        if row["Audit_ID"] in sampled:
            row["Review_State"] = "verified-no-material-error"

    if len(installed) != 662 or len(audit) != 750:
        raise AssertionError((len(installed), len(audit)))
    if len({row[10] for row in installed}) != len(installed):
        raise AssertionError("Installed Entry_Key values are not unique")
    if sum(row["Status"] == "skipped" for row in audit) != 88:
        raise AssertionError("Expected exactly 88 non-installable source cells")
    return installed, audit


def write_rows(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerows(rows)


def write_audit(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=AUDIT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()

    installed, audit = build(args.snapshot)
    output = args.output or (INSTALLED if args.install else Path("/tmp/kusunda-proposed.csv"))
    audit_output = args.audit_output or (
        AUDIT if args.install else Path("/tmp/kusunda-proposed-audit.csv")
    )
    write_rows(output, installed)
    write_audit(audit_output, audit)
    print(
        json.dumps(
            {
                "raw_prompts": 250,
                "source_cells": len(audit),
                "installed": len(installed),
                "skipped_cells": sum(row["Status"] == "skipped" for row in audit),
                "unresolved_installed": sum(
                    row["Status"] == "ingested" and bool(row["Unresolved"]) for row in audit
                ),
                "sampled_verified": sum(
                    row["Review_State"] == "verified-no-material-error" for row in audit
                ),
                "output": str(output),
                "audit": str(audit_output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
