"""Propose the entire archived A–C expression recovery without installing it."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
FROZEN_CANONICAL = "2ce9444da454e32b58f69a3815e6d1c26639330ea3ef0a6c94eb0c92b441855b"


def prepare():
    spec = importlib.util.spec_from_file_location("ho_existing_import", HERE / "import_source.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    baseline = HERE / "legacy-before-expression-recovery.csv"
    data = baseline.read_bytes()
    assert hashlib.sha256(data).hexdigest() == FROZEN_CANONICAL
    with baseline.open() as handle:
        legacy = list(csv.reader(handle))
    assert len(legacy) == 2146
    recovered = [json.loads(line) for line in (HERE / "expression-recovery-first-reading-20260926.jsonl").read_text().splitlines()]
    by_form = {}
    for row in legacy:
        by_form.setdefault(row[2], []).append(row[10])
    rows = [row[:] for row in legacy]
    edited_legacy_keys = set()
    for record in recovered:
        assert record["existing_form_keys"] == by_form.get(record["source_form"], [])
        if record["existing_form_keys"]:
            correction = record.get("existing_row_correction")
            if correction:
                for row in rows:
                    if row[10] in record["existing_form_keys"]:
                        row[3] = correction["gloss"]
                        row[6] = correction["notes"] + " Source uncertainty: " + correction["typed_reason"] + "."
                        row[14] = " ".join(correction["retain_tags"])
                        edited_legacy_keys.add(row[10])
            continue
        note = record["notes"]
        if record["uncertainty_types"]:
            note += " Source uncertainty: " + "; ".join(record["uncertainty_types"]) + "."
        rows.append([
            "ho", "", record["source_form"], record["gloss"], "", "", note,
            "DHED[archive entry " + record["source_id"] + "]", "", "",
            record["entry_key"], "", "", "", " ".join(record["tags"]),
        ])
    keys = {row[10] for row in rows}
    assert len(keys) == len(rows)
    assert [row[10] for row in rows[:2146]] == [row[10] for row in legacy]
    assert all(new == old or new[10] in edited_legacy_keys for new, old in zip(rows[:2146], legacy))
    assert edited_legacy_keys == {"ho-mla2004:01364:supplement:1", "ho-mla2004:01364:supplement:2"}
    assert all(not row[11] or row[11] in keys for row in rows)
    physical = module.records()
    old_audit = {record["source_id"]: record for record in map(json.loads, (HERE / "legacy-before-expression-recovery-audit.jsonl").read_text().splitlines())}
    census = []
    for record in physical:
        sid = record["source_id"]
        additions = [r for r in recovered if r["source_id"] == sid]
        embedded = []
        # Each literal angle span retains its raw source context. Nested/broken
        # delimiters are additionally covered by source-local explicit records.
        first_grammar = record["raw"].find("{")
        for match in re.finditer(r"<([^<>]*)>", record["raw"]):
            if first_grammar >= 0 and match.start() < first_grammar:
                continue
            form = match.group(1)
            recovered_keys = [r["entry_key"] for r in additions if r["source_form"] == form]
            existing_keys = by_form.get(form, [])
            status = "proposed_expression" if recovered_keys else "existing_literal_attestation" if existing_keys else "source_context_reference"
            embedded.append({
                "literal_span": form, "offset": [match.start(), match.end()],
                "status": status, "existing_keys": existing_keys, "recovery_keys": recovered_keys,
                "context": record["raw"][max(0, match.start() - 90):match.end() + 120],
            })
        census.append({
            **record,
            "prior_status": old_audit[sid]["status"],
            "prior_keys": [r["entry_key"] for r in old_audit[sid].get("rows", [])],
            "recovery_keys": [r["entry_key"] for r in additions if not r["existing_form_keys"]],
            "reviewed_existing_scope": [r["entry_key"] for r in additions if r["existing_form_keys"]],
            "embedded_spans": embedded,
            "decision": "Recover all separately translated examples and explicit attested example terms, including unglossed examples. Bare cross-references, donor/control comparisons, and constituent analysis remain attributed source context; no meanings inferred from them.",
        })
    assert len(census) == 1524
    module.separate_archived_references(rows)
    return rows, census, recovered


if __name__ == "__main__":
    rows, census, recovered = prepare()
    with (HERE / "expression-recovery-proposed.csv").open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)
    (HERE / "expression-recovery-census-20260926.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in census))
    print(json.dumps({"rows": len(rows), "legacy_keys_preserved": 2146, "legacy_gloss_scope_repairs": 2, "new_forms": len(rows) - 2146, "physical_units": len(census), "recovery_review_records": len(recovered)}))
