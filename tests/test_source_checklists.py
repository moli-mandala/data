import csv
import gzip
from pathlib import Path

import audit_source_ingestions as source_audits


ROOT = Path(__file__).parents[1]


def test_every_installed_ingestion_unit_has_a_fresh_checklist_copy():
    units, outputs = source_audits.expected_outputs()
    assert units
    assert source_audits.check_outputs(outputs) == []
    assert len({unit.id for unit in units}) == len(units)
    assert all((ROOT / "source_checklists" / f"{unit.id}.md").exists() for unit in units)


def test_retrospective_audit_accounts_for_every_installed_input_row():
    units = source_audits.build_units()
    audit_path = ROOT / "source_checklists/installed-record-audit.csv.gz"
    with gzip.open(audit_path, "rt", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == sum(unit.row_count for unit in units)
    assert all(row["Status"] == "installed" for row in rows)
    assert all(row["Row_SHA256"] for row in rows)


def test_source_checklist_machine_gates_are_clean_before_full_validation():
    for unit in source_audits.build_units():
        assert unit.blank_form_count == 0, unit.id
        assert unit.replacement_character_count == 0, unit.id
        assert unit.unresolved_references == [], unit.id
        assert unit.unregistered_languages == [], unit.id
        assert unit.unregistered_dialect_tags == [], unit.id
        assert unit.profiles, unit.id
        assert unit.audits, unit.id
        assert unit.tests, unit.id


def test_grammar_gate_is_evidence_based_and_only_verified_tagless_units_are_empty():
    units = source_audits.build_units()
    assert all(
        unit.source_grammar_evidence_rows == 0
        or unit.compiled_grammar_tagged_rows > 0
        for unit in units
    )
    assert {
        unit.id for unit in units if unit.compiled_grammar_tagged_rows == 0
    } == {"20220913-dhivehi", "20220913-kvari", "20230403-arora"}
