"""Focused guards for the exhausted Bangladesh survey-program candidate set."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).parents[1]
RAW = ROOT / "data/other/forms/raw_data"
AUDIT = RAW / "sil_bangladesh_remaining_unclassified_audit.json"
PROGRAM = RAW / "sil_bangladesh_survey_program_audit.csv"


def load_audit():
    return json.loads(AUDIT.read_text(encoding="utf-8"))


def program_rows():
    with PROGRAM.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_audit_pins_the_complete_program_census():
    audit = load_audit()
    rows = program_rows()
    program_input = audit["inputs"][0]
    assert hashlib.sha256(PROGRAM.read_bytes()).hexdigest() == program_input["sha256"]
    assert len(rows) == program_input["rows"] == 29
    assert Counter(row["Census_Disposition"] for row in rows) == Counter(
        program_input["dispositions"]
    )


def test_every_unresolved_program_row_is_accounted_for_in_census_order():
    audit = load_audit()
    rows = program_rows()
    unresolved = [
        (index, row)
        for index, row in enumerate(rows, 1)
        if row["Census_Disposition"]
        in {"unclassified_candidate", "missing_lexical_candidate"}
    ]
    candidates = audit["candidate_rows_in_census_order"]
    assert [index for index, _ in unresolved] == [9, 10, 12, 13, 19, 20]
    assert [candidate["program_row"] for candidate in candidates] == [
        index for index, _ in unresolved
    ]
    assert [candidate["community"] for candidate in candidates] == [
        row["Program_Community"] for _, row in unresolved
    ]
    assert all(candidate["excluded_active_audit"] for candidate in candidates)


def test_no_candidate_or_primary_topology_is_invented():
    audit = load_audit()
    selection = audit["selection"]
    candidates = audit["candidate_rows_in_census_order"]
    affirmative = [
        candidate
        for candidate in candidates
        if candidate["affirmative_lexical_or_wordlist_evidence"]
    ]
    assert [(candidate["community"], candidate["excluded_active_audit"]) for candidate in affirmative] == [
        ("Chak", "sil_chak_2007")
    ]
    assert selection["state"] == "no_eligible_candidate"
    assert selection["eligible_unresolved_reports"] == 0
    assert selection["selected_report"] is None
    assert selection["primary_report_acquired"] is False
    assert selection["publisher_pdf_sha256"] is None
    assert selection["publisher_pdf_rendered"] is False
    assert selection["wordlist_pages"] == []
    assert selection["wordlist_lists"] == []
    assert selection["wordlist_cells"] == []
    assert selection["manual_transcription_state"] == (
        "not_started_no_eligible_candidate"
    )
    assert "no report-page, list, or cell coordinates may be asserted" in selection[
        "unresolved_coordinates"
    ]
