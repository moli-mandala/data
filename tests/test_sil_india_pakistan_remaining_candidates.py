"""Focused guards for India/Pakistan SIL survey-census completeness."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).parents[1]
RAW = ROOT / "data/other/forms/raw_data"
AUDIT = RAW / "sil_india_pakistan_remaining_candidate_audit.json"
CENSUS = RAW / "sil_survey_sources.md"
GLOTTOLOG = RAW / "sil_glottolog_candidate_audit.csv"


def load_audit():
    return json.loads(AUDIT.read_text(encoding="utf-8"))


def country_rows(country):
    section = CENSUS.read_text(encoding="utf-8").split(f"### {country}", 1)[1]
    section = section.split("\n### ", 1)[0]
    rows = []
    for line in section.splitlines():
        if not line.startswith("| ") or line.startswith("|---"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells[0] == "Source":
            continue
        rows.append(cells)
    return rows


def classify_state(state):
    if "**shared source integration complete" in state:
        return "shared_integration_complete"
    if "**source-local" in state:
        return "complete_source_local_package"
    if "**installed" in state:
        return "installed"
    if "**inspected; no lexical rows to ingest**" in state:
        return "inspected_no_published_lexical_rows"
    if "**exact-source search exhausted; acquisition request required**" in state:
        return "active_acquisition_package"
    raise AssertionError(f"Unclassified census state: {state}")


def test_pinned_inputs_and_country_table_counts_are_current():
    audit = load_audit()
    census_input, glottolog_input = audit["inputs"]
    country_tables = {
        country: country_rows(country) for country in ("India", "Pakistan")
    }
    country_payload = json.dumps(
        country_tables,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert hashlib.sha256(country_payload).hexdigest() == census_input[
        "country_tables_sha256"
    ]
    assert len(country_payload) == census_input["country_tables_bytes"]
    assert hashlib.sha256(GLOTTOLOG.read_bytes()).hexdigest() == glottolog_input[
        "sha256"
    ]
    assert len(country_rows("India")) == census_input["country_table_rows"]["India"] == 50
    assert len(country_rows("Pakistan")) == census_input["country_table_rows"]["Pakistan"] == 7


def test_every_country_row_has_a_terminal_or_source_local_classification():
    audit = load_audit()
    for country in ("India", "Pakistan"):
        counts = Counter(classify_state(row[2]) for row in country_rows(country))
        expected = {
            key: value
            for key, value in audit["census_classification"][country].items()
            if key != "eligible_unrepresented"
        }
        assert counts == Counter(expected)
        assert audit["census_classification"][country]["eligible_unrepresented"] == 0


def test_all_source_local_and_active_packages_exist():
    audit = load_audit()
    source_local = audit["complete_source_local_packages_in_census_order"]
    assert len(source_local) == 14
    assert all((ROOT / record["path"]).is_dir() for record in source_local)
    active = audit["active_acquisition_packages"]
    assert len(active) == 1
    assert active[0]["source"] == (
        "Chamberlain, Chamberlain & Pavey 1998 Kinnauri manuscript"
    )
    active_path = ROOT / active[0]["path"]
    assert active_path.is_dir()
    assert (active_path / "DISCOVERY.md").is_file()
    assert active[0]["primary_report_acquired"] is False
    assert "All 79 manuscript pages" in active[0]["unresolved_coordinates"]


def test_glottolog_audit_has_no_unclassified_india_pakistan_gap():
    audit = load_audit()
    with GLOTTOLOG.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    glottolog_input = audit["inputs"][1]
    assert len(rows) == glottolog_input["rows"] == 32
    assert Counter(row["Disposition"] for row in rows) == Counter(
        glottolog_input["dispositions"]
    )
    partial = [row["Census_Anchor"] for row in rows if row["Disposition"] == "partial_manual"]
    assert partial == audit["glottolog_reconciliation"]["partial_manual_rows"]
    assert audit["glottolog_reconciliation"][
        "unclassified_or_missing_india_pakistan_candidates"
    ] == 0


def test_no_report_or_primary_topology_is_invented():
    selection = load_audit()["selection"]
    assert selection["state"] == "no_eligible_candidate"
    assert selection["eligible_unrepresented_reports"] == 0
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
