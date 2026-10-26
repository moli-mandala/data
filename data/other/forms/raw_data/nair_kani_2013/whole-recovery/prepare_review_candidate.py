"""Make a source-local, uninstalled 15-column Kani review candidate.

The candidate deliberately uses a provisional Kani language ID and must not be
installed until identity, transcription, independent audit, and metadata gates
are accepted. All Malayalam and Tamil Nadu controls remain in the audit.
"""

from __future__ import annotations

import csv
import hashlib
import json
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIRST = HERE / "physical-units-first-reading.jsonl"
CSV = HERE / "review-candidate.csv"
AUDIT = HERE / "review-audit.jsonl"
STATUS = HERE / "review-status.json"
SOURCE_ID = "nair2013kani"
LANGUAGE_ID = "Kanikkaran"  # Proposed base; not yet registered in Jambu.
DIALECT_TAG = "dialect:Kanikkaran:nair2013-kerala-kani:Kerala%20Kani"
TARGET = "Kerala Kani target"


def tags_for(unit: dict, gloss: str) -> list[str]:
    section = unit["source_unit_key"].split(":")[2]
    kind = unit["kind"]
    tags: list[str] = []
    if kind == "morphology":
        tags.append("suffix")
        if section == "present":
            tags.append("pres")
        if section == "negative":
            tags.append("neg")
        return tags
    if section == "noun-increment":
        tags.extend(["acc", "pron" if gloss == "who" else "noun"])
        if gloss == "who":
            tags.append("interr")
        return tags
    if section == "present":
        tags.extend(["verb", "pres"])
        if " " in unit["source_witnesses"][0]["form"]:
            tags.extend(["multiword-expression", "sentential"])
        return tags
    if section == "negative":
        tags.extend(["verb", "neg"])
        if " " in unit["source_witnesses"][0]["form"]:
            tags.append("multiword-expression")
        return tags
    if gloss == "five":
        return ["num"]
    if gloss in {"white"}:
        return ["adj"]
    if gloss in {"sold", "be destroyed"}:
        return ["verb"]
    if gloss in {"if not", "taste"}:
        return []  # The source does not give a precise POS.
    return ["noun"] if gloss else []


def main() -> None:
    units = [json.loads(line) for line in FIRST.read_text(encoding="utf-8").splitlines()]
    assert len(units) == 42
    rows: list[list[str]] = []
    audits: list[dict] = []
    keys: set[str] = set()
    for unit in units:
        emitted: list[str] = []
        witness_audit: list[dict] = []
        for i, entry in enumerate(unit["source_witnesses"], 1):
            entry_key = f'{unit["source_unit_key"]}:w{i}'
            assert entry_key not in keys
            keys.add(entry_key)
            scope = entry["scope"]
            record = {"entry_key": entry_key, "printed_form": entry["form"], "printed_gloss": entry["gloss"], "source_scope": scope, "status": "control_or_documentary", "reason": entry["note"]}
            if scope == TARGET:
                form = unicodedata.normalize("NFC", entry["form"])
                gloss = entry["gloss"]
                taglist = [*tags_for(unit, gloss), DIALECT_TAG]
                if form and "�" not in form:
                    rows.append([
                        LANGUAGE_ID, "", form, gloss, "", "", "",
                        f'{SOURCE_ID}[p. {unit["printed_page"]}]', "", "", entry_key,
                        "", "", "", " ".join(taglist),
                    ])
                    emitted.append(entry_key)
                    record.update(status="proposed_uninstalled", tags=taglist, reason="provisional base-language registration and independent audit pending")
            witness_audit.append(record)
        audits.append({
            "source_unit_key": unit["source_unit_key"],
            "printed_page": unit["printed_page"],
            "kind": unit["kind"],
            "physical_disposition": unit["disposition"],
            "source_claim": unit["source_claim"],
            "note": unit["note"],
            "witnesses": witness_audit,
            "emitted_entry_keys": emitted,
            "review": unit["review"],
        })
    assert len(keys) == 66
    assert len(rows) == 36
    assert len({row[10] for row in rows}) == len(rows)
    assert all(len(row) == 15 for row in rows)
    with CSV.open("w", newline="", encoding="utf-8") as out:
        csv.writer(out).writerows(rows)
    with AUDIT.open("w", encoding="utf-8") as out:
        for record in audits:
            out.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    STATUS.write_text(json.dumps({
        "status": "review candidate only; not accepted or installed",
        "source_input_sha256": hashlib.sha256(FIRST.read_bytes()).hexdigest(),
        "candidate_csv_sha256": hashlib.sha256(CSV.read_bytes()).hexdigest(),
        "audit_sha256": hashlib.sha256(AUDIT.read_bytes()).hexdigest(),
        "physical_units": len(units),
        "witness_slots": len(keys),
        "candidate_rows": len(rows),
        "excluded_control_or_documentary_witnesses": len(keys) - len(rows),
        "language_id": "Kanikkaran (proposed, unregistered; Glottolog kani1275/ISO kev)",
        "dialect_tag": DIALECT_TAG,
        "source_id": SOURCE_ID,
        "sound_profile": "No conversion proposed until full literal symbol review; source doubled vowels and combining diacritics are preserved in Form and future Original.",
        "deferred_gates": ["independent full literal review", "identity/lect registry", "bibliography and YAML", "source-profile coverage", "fresh 20-unit output audit", "actual parser", "canonical source stage", "full CLDF build and tests"],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
