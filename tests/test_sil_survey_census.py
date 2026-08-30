"""Regression checks for the auditable SIL South Asia survey census."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
CENSUS = ROOT / "data/other/forms/raw_data/sil_survey_sources.md"
CANDIDATES = ROOT / "data/other/forms/raw_data/sil_glottolog_candidate_audit.csv"
CATALOG_MANIFEST = (
    ROOT / "data/other/forms/raw_data/sil_glottolog_candidate_audit_manifest.json"
)
BANGLADESH_ARCHIVE = (
    ROOT / "data/other/forms/raw_data/sil_bangladesh_archive_candidate_audit.csv"
)
CHAK_DISCOVERY = ROOT / "data/other/forms/raw_data/sil_chak_2007_discovery.json"
CHITTAGONIAN_DISCOVERY = (
    ROOT / "data/other/forms/raw_data/sil_chittagonian_2007_discovery.json"
)
BANGLADESH_PROGRAM = (
    ROOT / "data/other/forms/raw_data/sil_bangladesh_survey_program_audit.csv"
)
BANGLADESH_UNLISTED = (
    ROOT / "data/other/forms/raw_data/sil_bangladesh_2007_unlisted_reports_discovery.json"
)
JLSR_2025_006_DISCOVERY = (
    ROOT / "data/other/forms/raw_data/sil_jlsr_2025_006_discovery.json"
)
PAKISTAN_SSNP = (
    ROOT / "data/other/forms/raw_data/sil_pakistan_ssnp_series_manifest.json"
)
INDIAN_SIGN_LANGUAGE = (
    ROOT / "data/other/forms/raw_data/sil_indian_sign_language_2008/inspection.json"
)


def test_country_tagged_glottolog_candidates_have_explicit_dispositions():
    with CANDIDATES.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 32
    assert len({row["Glottolog_Key"] for row in rows}) == 32
    status = {row["SIL_Status"] for row in rows}
    assert status == {"sil", "not_sil_false_positive"}
    genuine = [row for row in rows if row["SIL_Status"] == "sil"]
    assert len(genuine) == 30
    assert all(row["Census_Anchor"] for row in genuine)
    assert {row["Disposition"] for row in genuine} <= {
        "installed", "covered_component", "duplicate_reissue", "inspected_no_rows",
        "partial_manual",
    }
    assert sum(row["Disposition"] == "partial_manual" for row in genuine) == 3
    assert all(row["Disposition"] == "outside_scope" for row in rows if row not in genuine)

    manifest = json.loads(CATALOG_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["audit_rows"] == len(rows)
    assert manifest["genuine_sil_rows"] == len(genuine)
    assert manifest["false_positive_rows"] == len(rows) - len(genuine)
    assert manifest["sha256"] == (
        "ac05cd1a2b546e855904f3a53d4c4f01e131c1a69f4e0ec1838a38b6b1fc4ece"
    )
    assert manifest["bytes"] == 46_017_843
    assert "manually classified for genuine SIL provenance" in manifest["selection"]


def test_kundal_shahi_sil_working_paper_is_reconciled_in_pakistan_census():
    installed = ROOT / "data/other/forms/20220913-kundalshahi.csv"
    with installed.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 163
    assert all(len(row) == 15 and row[0] == "Kund" and row[7] == "kund" for row in rows)

    census = CENSUS.read_text(encoding="utf-8")
    assert "SIL EWP 2005-008 Kundal Shahi, Azad Kashmir" in census
    assert "`20220913-kundalshahi.csv`: 163 source rows" in census


def test_all_five_ssnp_volumes_are_explicitly_mapped_to_installed_files():
    manifest = json.loads(PAKISTAN_SSNP.read_text(encoding="utf-8"))
    assert manifest["series_title"] == "Sociolinguistic Survey of Northern Pakistan"
    assert manifest["volume_count"] == 5
    assert [volume["number"] for volume in manifest["volumes"]] == [1, 2, 3, 4, 5]
    assert all(volume["state"] == "installed" for volume in manifest["volumes"])
    assert all((ROOT / volume["installed_file"]).is_file() for volume in manifest["volumes"])

    census = CENSUS.read_text(encoding="utf-8")
    assert "five-volume *Sociolinguistic Survey of Northern Pakistan*" in census
    assert "six-volume *Sociolinguistic Survey of Northern Pakistan*" not in census
    assert all(f"SSNP vol. {number}" in census for number in range(1, 6))


def test_current_jlsr_object_is_explicitly_dispositioned_outside_country_scope():
    census = CENSUS.read_text(encoding="utf-8")
    assert "JLSR 2025-006" in census
    assert "identified outside country scope" in census
    assert "western Sulawesi, Indonesia" in census

    discovery = json.loads(JLSR_2025_006_DISCOVERY.read_text(encoding="utf-8"))
    assert discovery["object_id"] == "JLSR2025-006"
    assert discovery["official_object_url"] == (
        "https://test-silorg.sil.org/system/files/reapdata/other/JLSR2025-006.pdf"
    )
    assert discovery["publisher_file_acquired"] is False
    assert discovery["lexical_scope"] == (
        "not_inspected_after_out_of_scope_geography_was_proven"
    )
    assert discovery["identity"] == {
        "authors": [
            "Renhard Saupia",
            "Stan Anonby",
            "Tiar Simanjuntak",
            "Geraldy Ruwayari",
        ],
        "countries": ["Indonesia"],
        "languages": [
            "Mandar [mdr]",
            "Pannei-Ulumanda / Ulumanda [ulm]",
            "Pannei-Polewali / Pannei [pnc]",
            "Koneq-koneq [cml]",
            "Dakka [dkk]",
        ],
        "title": (
            "A Sociolinguistic Survey of Mandar, Pannei-Ulumanda, "
            "Pannei-Polewali, Koneq-koneq, and Dakka"
        ),
        "year": 2025,
    }
    assert discovery["discovery_state"] == "identified_outside_country_scope"
    assert len(discovery["retrieval_attempts"]) == 3
    assert "explicitly outside" in discovery["policy"]


def test_indian_sign_language_report_is_closed_without_invented_forms():
    inspection = json.loads(INDIAN_SIGN_LANGUAGE.read_text(encoding="utf-8"))
    assert inspection["source"]["series"] == "SIL Electronic Survey Report 2008-006"
    assert inspection["acquisition"]["sha256"] == (
        "00ae89e7fcfee81dd46c6895f338dd989ed9dd7ebe99cf8604c946eaf18a426f"
    )
    assert inspection["acquisition"]["physical_pages"] == 121
    assert inspection["inspection"]["text_layer_extracted_pages"] == 121
    assert inspection["inspection"]["rendered_pages"] == 121
    assert inspection["inspection"]["published_prompt_rows"] == 245
    assert inspection["inspection"]["published_pairwise_judgment_slots"] == 2450
    assert inspection["inspection"]["item_level_sign_forms_published"] is False
    assert inspection["inspection"]["underlying_sign_recordings_published"] is False
    assert inspection["inspection"]["unresolved_published_lexical_readings"] == []
    disposition = inspection["editorial_disposition"]
    assert disposition["state"] == "inspected_no_lexical_forms_to_ingest"
    assert disposition["installed_lexical_rows"] == 0
    assert disposition["excluded_prompt_only_rows"] == 245
    assert disposition["excluded_pairwise_judgment_slots"] == 2450

    census = CENSUS.read_text(encoding="utf-8")
    assert "ESR 2008-006 Indian Sign Language" in census
    assert "**inspected; no lexical rows to ingest**" in census
    assert "all six Appendix G pages 58-63 visually checked" in census
    assert "representation decision and source acquisition pending" not in census


def test_official_bangladesh_survey_list_is_exhaustively_dispositioned():
    with BANGLADESH_ARCHIVE.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 11
    assert len({row["Official_List_Title"] for row in rows}) == 11
    dispositions = {state: 0 for state in (
        "installed", "partial_manual", "missing_lexical_candidate", "unclassified_candidate"
    )}
    for row in rows:
        dispositions[row["Census_Disposition"]] += 1
        assert row["Official_List_URL"] == (
            "https://www.silbangladesh.org/resources/publications/survay_report"
        )
        assert row["Lexical_Evidence"] and row["Census_Anchor"]
    assert dispositions == {
        "installed": 6,
        "partial_manual": 3,
        "missing_lexical_candidate": 1,
        "unclassified_candidate": 1,
    }

    census = CENSUS.read_text(encoding="utf-8")
    assert "SIL Bangladesh 2007 Chak" in census
    assert "missing lexical candidate" in census
    assert "SIL Bangladesh 2007 Chittagonian-speaking Community" in census
    assert "unclassified acquisition gap" in census
    assert "official SIL Bangladesh survey landing list contains eleven named reports" in census

    chak = json.loads(CHAK_DISCOVERY.read_text(encoding="utf-8"))
    assert chak["canonical_candidate_title"] == (
        "The Chak of Bangladesh: A Sociolinguistic Study"
    )
    assert chak["extent"] == "ix + 56 pages"
    assert chak["lexical_content_confidence"] == "high"
    assert chak["publisher_pdf_acquired"] is False
    assert chak["discovery_state"] == (
        "official_locator_confirmed_publisher_pdf_unavailable"
    )
    official = chak["official_locator"]
    assert official["publisher_listing_verified"] is True
    assert official["publisher_listing_file_link_present"] is False
    assert official["publisher_program_status"] == "Done"
    assert official["publisher_archive_record_found"] is False
    primary = chak["primary_source_verification"]
    assert primary["publisher_pdf_acquired"] is False
    assert primary["sha256"] is None
    assert primary["rendered_pages"] is False
    assert primary["wordlist_presence_verified_from_publisher_pages"] is False
    assert primary["wordlist_page_coordinates"] == []
    assert primary["wordlist_topology_verified"] is False
    assert chak["secondary_variety_evidence"]["reported_count"] == 4
    assert "official locator confirmed" in census
    assert "primary wordlist pages, four-way labels and topology remain unverified" in census
    assert "secondary forms are not transcription evidence" in census

    chittagonian = json.loads(
        CHITTAGONIAN_DISCOVERY.read_text(encoding="utf-8")
    )
    assert chittagonian["canonical_title"] == (
        "A Sociolinguistic Survey of the Chittagonian-speaking Community"
    )
    assert chittagonian["responsibility"]["compiled_by"] == [
        "Loren Maggard", "Mridul Sangma", "Sayed Ahmad"
    ]
    assert chittagonian["publication_date"] == "February 2007"
    assert chittagonian["extent"] is None
    assert chittagonian["series_issue"] is None
    assert chittagonian["archive_entry_number"] is None
    assert chittagonian["primary_cover_asset"]["sha256"] == (
        "f5870274f73a2baaed1b7b0f57c81ddbe05e272962e7dcd8eb8c93ea3fe2eb55"
    )
    assert chittagonian["publisher_pdf_acquired"] is False
    topology = chittagonian["primary_source_topology"]
    assert topology["wordlist_presence_verified"] is False
    assert topology["ipa_wordlist_presence_verified"] is False
    assert topology["wordlist_pages"] == []
    assert topology["wordlist_cells"] == []
    assert chittagonian["independent_lexical_evidence"]["state"] == "none_found"
    assert "absence of indexed lexical evidence is not evidence of absence" in census


def test_bangladesh_twenty_nine_community_program_is_reconciled():
    with BANGLADESH_PROGRAM.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 29
    assert len({row["Program_Community"] for row in rows}) == 29
    dispositions = {}
    for row in rows:
        dispositions[row["Census_Disposition"]] = (
            dispositions.get(row["Census_Disposition"], 0) + 1
        )
        assert row["Program_URL"] == (
            "https://www.silbangladesh.org/language_education/sociolinguistics"
        )
        assert row["Cluster"] and row["Source_Candidate"] and row["Census_Anchor"]
    assert dispositions == {
        "installed": 18,
        "partial_manual": 5,
        "missing_lexical_candidate": 1,
        "unclassified_candidate": 5,
    }

    census = CENSUS.read_text(encoding="utf-8")
    assert "official SIL Bangladesh sociolinguistic-program chart" in census
    assert "29 communities" in census
    assert "SIL Bangladesh 2007 Chakma and Tanchangya" in census
    assert "SIL Bangladesh 2007 Marma and Rakhine" in census
    assert "every unclassified report still requires publisher-file acquisition" in census

    unlisted = json.loads(BANGLADESH_UNLISTED.read_text(encoding="utf-8"))
    assert len(unlisted["reports"]) == 2
    assert {report["title"] for report in unlisted["reports"]} == {
        "A Sociolinguistic Survey among the Chakma and Tanchangya Communities",
        "The Marma and Rakhine Communities of Bangladesh: A Sociolinguistic Survey",
    }
    assert all(report["year"] == 2007 for report in unlisted["reports"])
    assert all(report["publisher_file_acquired"] is False for report in unlisted["reports"])
    by_title = {report["title"]: report for report in unlisted["reports"]}
    chakma_tanchangya = by_title[
        "A Sociolinguistic Survey among the Chakma and Tanchangya Communities"
    ]
    assert chakma_tanchangya["lexical_scope"] == (
        "wordlist_comparison_reported_secondarily_published_forms_unverified"
    )
    assert chakma_tanchangya["state"] == (
        "official_program_complete_primary_report_unacquired"
    )
    assert chakma_tanchangya["publisher"] is None
    assert chakma_tanchangya["extent"] is None
    assert chakma_tanchangya["series"] is None
    assert chakma_tanchangya["series_issue"] is None
    assert chakma_tanchangya["archive_entry_number"] is None
    assert chakma_tanchangya["primary_artifact"]["visually_verified"] is False
    assert chakma_tanchangya["official_locator"]["archive_language_counts"] == {
        "Chakma [ccp]": 4,
        "Tangchangya [tnv]": 4,
    }
    lexical_evidence = chakma_tanchangya["independent_lexical_evidence"]
    assert lexical_evidence["published_wordlist_presence_verified"] is False
    assert (
        chakma_tanchangya["primary_source_topology"]["wordlist_pages"] == []
    )
    unresolved = chakma_tanchangya["primary_source_topology"]["unresolved_coordinates"]
    assert "all report pages" in unresolved
    assert "lists" in unresolved
    assert "cells" in unresolved
    marma_rakhine = by_title[
        "The Marma and Rakhine Communities of Bangladesh: A Sociolinguistic Survey"
    ]
    assert marma_rakhine["lexical_scope"] == (
        "unverified_no_independent_evidence_of_published_forms"
    )
    assert marma_rakhine["state"] == (
        "official_program_complete_primary_report_unacquired"
    )
    assert marma_rakhine["extent"] is None
    assert marma_rakhine["series"] is None
    assert marma_rakhine["series_issue"] is None
    assert marma_rakhine["archive_entry_number"] is None
    assert marma_rakhine["primary_artifact"]["visually_verified"] is False
    assert marma_rakhine["official_locator"]["archive_language_counts"] == {
        "Marma [rmz]": 3,
        "Rakhine [rki]": 1,
    }
    related = marma_rakhine["official_locator"]["related_official_archive_record"]
    assert related["entry_number"] == 94638
    assert "distinct work" in related["identity"]
    marma_lexical = marma_rakhine["independent_lexical_evidence"]
    assert marma_lexical["published_wordlist_presence_verified"] is False
    assert marma_lexical["ipa_wordlist_presence_verified"] is False
    assert marma_rakhine["primary_source_topology"]["wordlist_pages"] == []
    assert "all report pages, lists, and cells" in (
        marma_rakhine["primary_source_topology"]["unresolved_coordinates"]
    )
    assert "No lexical form may be transcribed" in unlisted["policy"]
