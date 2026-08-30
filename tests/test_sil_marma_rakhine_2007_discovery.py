"""Focused acquisition guards for the unavailable SIL Marma-Rakhine report."""

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
DISCOVERY = (
    ROOT / "data/other/forms/raw_data/sil_marma_rakhine_2007_discovery.json"
)


def record():
    return json.loads(DISCOVERY.read_text(encoding="utf-8"))


def test_candidate_identity_is_secondary_and_primary_artifact_is_absent():
    source = record()
    assert source["canonical_candidate_title"] == (
        "The Marma and Rakhine Communities of Bangladesh: "
        "A Sociolinguistic Survey"
    )
    assert source["discovery_state"] == (
        "official_program_complete_primary_report_unacquired"
    )
    assert source["bibliographic_identity"]["authors"] == [
        "Loren Maggard", "Sayed Ahmad", "Mridul Sangma"
    ]
    assert "secondary" in source["bibliographic_identity"]["evidence_scope"]
    primary = source["primary_artifact"]
    assert primary["cover_found"] is False
    assert primary["title_page_found"] is False
    assert primary["report_found"] is False
    assert primary["publisher_pdf_acquired"] is False
    assert primary["publisher_pdf_sha256"] is None
    assert primary["publisher_pdf_rendered"] is False
    assert primary["visually_verified"] is False


def test_reproducible_archive_searches_leave_acquisition_blocked():
    audit = record()["acquisition_audit"]
    official = audit["official_domain_pdf_audit"]
    assert official["indexed_pdf_urls"] == 76
    assert official["unique_pdf_digests"] == 72
    assert official["marma_rakhine_filename_matches"] == 0
    assert official["response_sha256"] == (
        "ac42ec77c8e180a55f7ff167b5f1cbfa4f14ee6fa204df457226839d0f63e994"
    )
    assert audit["official_named_asset_audit"] == {
        "marma_successful_matches": 0,
        "rakhine_successful_matches": 0,
        "each_empty_response_bytes": 3,
        "each_empty_response_sha256": (
            "37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570"
        ),
    }
    assert audit["olac_exact_title_audit"]["matches"] == 0
    assert audit["olac_language_code_audit"]["matches"] == 10
    assert audit["olac_language_code_audit"]["sil_archive_matches"] == 0
    assert audit["internet_archive_exact_title_audit"]["catalog_matches"] == 0
    assert audit["direct_publisher_filename_probe"]["result"] == "inconclusive"
    assert "ask staff to search unpublished 2007 survey manuscripts" in (
        audit["next_action"]
    )


def test_no_lexical_topology_or_reading_is_invented():
    source = record()
    topology = source["primary_source_topology"]
    assert topology["report_page_count_verified"] is False
    assert topology["wordlist_presence_verified"] is False
    assert topology["ipa_wordlist_presence_verified"] is False
    assert topology["wordlist_pages"] == []
    assert topology["wordlist_lists"] == []
    assert topology["wordlist_cells"] == []
    assert topology["manual_transcription_state"] == (
        "not_started_acquisition_blocked"
    )
    assert "all report pages, lists, tables, rows, columns, and cells" in (
        topology["unresolved_coordinates"]
    )
    lexical = source["independent_lexical_evidence"]
    assert lexical["published_wordlist_presence_verified"] is False
    assert lexical["ipa_wordlist_presence_verified"] is False
    related = source["distinct_related_source"]
    assert related["archive_entry_number"] == 94638
    assert "distinct 2014 M.A. thesis" in related["identity"]
    assert "neither establish" in related["lexical_limit"]
    assert "No forms may be taken" in source["policy"]
