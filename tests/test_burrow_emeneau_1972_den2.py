import csv
import json
from collections import Counter
from pathlib import Path

from data.other.forms.raw_data import burrow_emeneau_1972_den2 as den2


ROOT = Path(__file__).parents[1]
SOURCE = den2.SOURCE_ID


def dict_rows(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def form_rows():
    with den2.FORM_OUTPUT.open(encoding="utf-8", newline="") as handle:
        return [dict(zip(den2.FORM_FIELDS, row)) for row in csv.reader(handle)]


def test_every_article_page_is_isolated_and_indexes_are_nonlexical():
    pages = den2.load_pages()
    assert len(pages) == 17
    assert [page["printed_page"] for page in pages] == list(range(475, 492))
    assert [len(page["records"]) for page in pages[:5]] == [29, 38, 31, 17, 4]
    assert all(page["page_kind"] == "lexical_entries" for page in pages[:5])
    assert all(page["page_kind"] == "bibliography" for page in pages[5:])
    assert all(not page["records"] and page["page_notes"] for page in pages[5:])

    records = [record for page in pages for record in page["records"]]
    assert len(records) == 119
    assert sum(len(record["forms"]) for record in records) == 448
    assert all(record["raw_entry_text"] for record in records)
    assert len({record["unit_id"] for record in records}) == len(records)


def test_manifest_and_audit_counts_reconcile_without_redistributing_pdf():
    manifest = json.loads(den2.MANIFEST_OUTPUT.read_text())
    reconciliation = json.loads(den2.RECONCILIATION_OUTPUT.read_text())
    audit = dict_rows(den2.AUDIT_OUTPUT)
    sample = dict_rows(den2.SAMPLE_OUTPUT)

    assert manifest["pdf_sha256"] == den2.PDF_SHA256
    assert manifest["pdf_pages"] == 18
    assert manifest["article_printed_pages"] == [475, 491]
    assert manifest["lexical_printed_pages"] == [475, 479]
    assert manifest["index_printed_pages"] == [480, 491]
    assert manifest["pdf_redistributed"] is False
    assert manifest["record_count"] == reconciliation["record_count"] == 119
    assert manifest["raw_form_count"] == reconciliation["raw_form_count"] == 448
    assert manifest["installed_form_count"] == len(form_rows()) == 159
    assert manifest["audit_count"] == len(audit) == 567
    assert manifest["entry_text_count"] == 0
    assert len(sample) == manifest["sample_count"] == 20
    assert "S-squared" in reconciliation["policy"]


def test_only_current_dedr_corroborated_den_entries_are_installed():
    forms = {row["Entry_Key"]: row for row in form_rows()}
    assert len(forms) == 159
    assert all(row["Parameter_ID"].startswith("d") for row in forms.values())
    assert all(row["Source"].startswith(f"{SOURCE}[") for row in forms.values())
    assert all("DEN II (1972)" in row["Etymology"] for row in forms.values())
    assert all("�" not in "|".join(row.values()) for row in forms.values())

    assert forms[f"{SOURCE}:p475:u001:f001"]["Parameter_ID"] == "d49"
    assert forms[f"{SOURCE}:p475:u001:f001"]["Form"] == "accu"
    assert forms[f"{SOURCE}:p476:u001:f001"]["Parameter_ID"] == "d2121"
    assert forms[f"{SOURCE}:p476:u011:f001"]["Form"] == "sūri"
    assert forms[f"{SOURCE}:p476:u015:f002"]["Form"] == "jammō"
    assert forms[f"{SOURCE}:p476:u020:f003"]["Form"] == "tōṛa (tōṛi-)"
    # The unmarked page-agent form puḷi is a false exact homonym for 'sour'; source gloss and
    # the fuller current-DEDR transcription correctly place 'mist' in d4375.
    mist = forms[f"{SOURCE}:p477:u001:f003"]
    assert mist["Parameter_ID"] == "d4375"
    assert mist["Form"] == "pu·ḷï"


def test_every_exclusion_class_and_dbia_deferral_is_explicit():
    audit = dict_rows(den2.AUDIT_OUTPUT)
    assert Counter(row["Final_Status"] for row in audit) == {
        "raw_segment_audited": 119,
        "installed_form": 159,
        "unresolved_target": 177,
        "unreconciled_transcription": 46,
        "excluded_nonaccepted": 66,
    }
    assert Counter(
        row["Raw_Status"] for row in audit if row["Final_Status"] == "excluded_nonaccepted"
    ) == {
        "queried": 28,
        "comparison_only": 20,
        "deleted": 14,
        "loan": 3,
        "active": 1,
    }
    dbia_forms = [
        row for row in audit if row["Series"] == "DBIA" and row["Item_Type"] == "form"
    ]
    assert dbia_forms
    assert not any(row["Final_Status"] == "installed_form" for row in dbia_forms)
    assert all(
        "separate loan-entry reconciliation" in row["Resolution"]
        for row in dbia_forms if row["Final_Status"] == "unresolved_target"
    )


def test_page_agent_running_text_stays_audit_only_pending_diplomatic_review():
    assert dict_rows(den2.TEXT_OUTPUT) == []
    assert all(
        row["Material_Error"] == "unreviewed"
        for row in dict_rows(den2.AUDIT_OUTPUT)
        if row["Item_Type"] == "record"
    )


def test_unambiguous_dialects_use_registered_lect_ids():
    forms = form_rows()
    parents, labels = den2.shared.load_dialect_registry()
    assert den2.shared.resolve_output_lect("Koraga", "Mudu dial., LSB 12.8", labels) == "mudu"
    assert den2.shared.resolve_output_lect("Malayalam", "Tiyya", labels) == "tiyya"
    assert den2.shared.resolve_output_lect("Gondi", "A. Su.", labels) == "adil"
    assert sum(row["Language_ID"] in parents for row in forms) == 39
    assert {
        row["Language_ID"] for row in forms if row["Language_ID"] in parents
    } >= {"adil", "mudu", "tiyya"}


def test_bibliography_is_compiled_with_stable_record_and_doi():
    references = {row["ID"]: row for row in dict_rows(ROOT / "cldf/references.csv")}
    assert SOURCE in references
    assert "10\\.2307/599958" in references[SOURCE]["Source"]
    assert references[SOURCE]["Progress"].startswith("Every numbered DEDS and DBIA")


def test_build_carries_the_source_citation_without_replacement_glyphs():
    compiled = [
        row for row in dict_rows(ROOT / "cldf/forms.csv") if SOURCE in row["Source"]
    ]
    assert len(compiled) == 161
    assert sum("dedr" in row["Source"].split(";") for row in compiled) == 121
    assert sum(row["Source"].startswith(SOURCE) for row in compiled) == 40
    assert all("�" not in "|".join(row.values()) for row in compiled)
    assert any(
        row["Original"] == "accu" and "current DEDR d49" in row["Etymology"]
        for row in compiled
    )
    assert any(
        row["Original"] == "pu·ḷï" and "current DEDR d4375" in row["Etymology"]
        for row in compiled
    )
