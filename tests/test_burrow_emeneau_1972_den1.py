import csv
import json
from collections import Counter
from pathlib import Path

from data.other.forms.raw_data import burrow_emeneau_1972_den1 as den1


ROOT = Path(__file__).parents[1]
RAW = ROOT / "data/other/forms/raw_data"
SOURCE = den1.SOURCE_ID


def dict_rows(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def form_rows():
    with den1.FORM_OUTPUT.open(encoding="utf-8", newline="") as handle:
        return [dict(zip(den1.FORM_FIELDS, row)) for row in csv.reader(handle)]


def test_every_printed_page_is_isolated_parseable_and_audited():
    pages = den1.load_pages()
    assert len(pages) == 22
    assert [page["printed_page"] for page in pages] == list(range(397, 419))
    assert [len(page["records"]) for page in pages[:2]] == [0, 0]
    assert len(pages[2]["records"]) == 20  # p. 399 bibliography/lexicon boundary

    records = [record for page in pages for record in page["records"]]
    units = [record["unit_id"] for record in records]
    assert len(units) == len(set(units))
    assert all(record["raw_entry_text"] for record in records)

    audit = dict_rows(den1.AUDIT_OUTPUT)
    raw_forms = sum(len(record.get("forms", [])) for record in records)
    assert len(audit) == len(records) + raw_forms
    assert Counter(row["Item_Type"] for row in audit) == {
        "record": len(records), "form": raw_forms,
    }
    assert {row["Review"] for row in audit} == {
        "source-image/current-DEDR reconciled",
        "page-agent structure reviewed; running text pending diplomatic verification",
        "source structure reviewed; exclusion or unresolved state is explicit",
    }
    assert {
        row["Material_Error"] for row in audit if row["Item_Type"] == "record"
    } == {"unreviewed"}
    assert all(
        row["Material_Error"] == "no" for row in audit if row["Item_Type"] == "form"
    )


def test_manifest_counts_reconcile_and_publisher_pdf_is_not_redistributed():
    manifest = json.loads(den1.MANIFEST_OUTPUT.read_text())
    reconciliation = json.loads(den1.RECONCILIATION_OUTPUT.read_text())
    forms = form_rows()
    audit = dict_rows(den1.AUDIT_OUTPUT)
    sample = dict_rows(den1.SAMPLE_OUTPUT)

    assert manifest["pdf_sha256"] == den1.PDF_SHA256
    assert manifest["pdf_pages"] == 23
    assert manifest["article_printed_pages"] == [397, 418]
    assert manifest["pdf_redistributed"] is False
    assert manifest["record_count"] == reconciliation["record_count"]
    assert manifest["raw_form_count"] == reconciliation["raw_form_count"]
    assert manifest["installed_form_count"] == len(forms)
    assert manifest["audit_count"] == len(audit)
    assert len(sample) == manifest["sample_count"] == 20
    assert manifest["entry_text_count"] == 0


def test_only_corroborated_rank_one_forms_are_installed():
    forms = {row["Entry_Key"]: row for row in form_rows()}
    assert len(forms) == len(form_rows())
    assert all(row["Source"].startswith(f"{SOURCE}[") for row in forms.values())
    assert all("�" not in "|".join(row.values()) for row in forms.values())

    assert forms[f"{SOURCE}:p400:u057:f001"]["Form"] == "iḷusan"
    assert forms[f"{SOURCE}:p400:u057:f001"]["Parameter_ID"] == "d512"
    assert forms[f"{SOURCE}:p400:u057:f001"]["Language_ID"] == "tiyya"
    assert forms[f"{SOURCE}:p401:u063:f001"]["Form"] == "talay-ēru"
    assert forms[f"{SOURCE}:p401:u063:f001"]["Parameter_ID"] == "d811"
    assert forms[f"{SOURCE}:p407:u006:f001"]["Form"] == "jicoṇa"
    assert forms[f"{SOURCE}:p407:u006:f001"]["Parameter_ID"] == "d800"

    # The single page-agent field combines variants that later split across d2621 and d709.
    # It remains audit-only instead of acquiring an arbitrary rank-1 target.
    assert f"{SOURCE}:p407:u013:f001" not in forms
    audit = {row["Unit_ID"]: row for row in dict_rows(den1.AUDIT_OUTPUT)}
    assert audit["p407:u013:f001"]["Final_Status"] == "variant_split_pending"


def test_unambiguous_source_dialects_use_registered_lect_ids():
    forms = form_rows()
    dialect_parents, dialect_labels = den1.load_dialect_registry()
    assert den1.resolve_output_lect("Gondi", "A Su.", dialect_labels) == "adil"
    assert den1.resolve_output_lect("Gondi", "Koya Su.", dialect_labels) == "koya"
    assert den1.resolve_output_lect("Koraga", "Onti dial., LSB 7.14", dialect_labels) == "onti"
    assert den1.resolve_output_lect("Koraga", "Onti, Tappu dialects, LSB 12.8", dialect_labels) == "Koraga"
    assert den1.resolve_output_lect("Tamil", "RS, p. 141, item 190", dialect_labels) == "Tamil"
    assert all(
        row["Language_ID"] not in dialect_parents
        or dialect_parents[row["Language_ID"]] in {
            "Gadaba", "Gondi", "Kannada", "Kolami", "Konda", "Koraga", "Kuwi",
            "Malayalam", "Tamil",
        }
        for row in forms
    )


def test_uncertain_deleted_comparison_and_loan_forms_never_become_reflex_rows():
    audit = dict_rows(den1.AUDIT_OUTPUT)
    excluded = [row for row in audit if row["Final_Status"] == "excluded_nonaccepted"]
    assert excluded
    assert {row["Raw_Status"] for row in excluded} >= {
        "queried", "deleted", "comparison_only", "loan",
    }
    assert all(not row["Emitted_Key"] for row in excluded)


def test_agent_running_text_stays_in_raw_evidence_until_diplomatic_review():
    # Luna is useful for entry/form structure but not reliable enough on dense Dravidianist
    # diacritics to publish its running-text transcript.  The raw segment is fully audited while
    # the app-facing entry-text sidecar intentionally remains empty.
    assert dict_rows(den1.TEXT_OUTPUT) == []
    assert all(
        row["Final_Status"] == "raw_segment_audited"
        for row in dict_rows(den1.AUDIT_OUTPUT)
        if row["Item_Type"] == "record"
    )


def test_bibliography_is_compiled_with_stable_record_and_doi():
    references = {row["ID"]: row for row in dict_rows(ROOT / "cldf/references.csv")}
    assert SOURCE in references
    assert "10\\.2307/600566" in references[SOURCE]["Source"]
    assert references[SOURCE]["Progress"].startswith("Every numbered additions-and-corrections")


def test_build_merges_source_citations_without_duplicate_reflexes():
    compiled = [
        row for row in dict_rows(ROOT / "cldf/forms.csv") if SOURCE in row["Source"]
    ]
    assert len(compiled) == 713
    assert sum("dedr" in row["Source"].split(";") for row in compiled) == 579
    assert sum(row["Source"].startswith(SOURCE) for row in compiled) == 134
    assert all("�" not in "|".join(row.values()) for row in compiled)

    ilusan = [
        row for row in compiled
        if row["Language_ID"] == "Malayalam" and row["Original"] == "iḷusan"
    ]
    assert len(ilusan) == 1
    assert "dedr" in ilusan[0]["Source"]
    assert "dialect:Malayalam:tiyya:Thiyya" in ilusan[0]["Tags"]
