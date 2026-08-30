"""Focused acquisition guards for the unavailable SIL Bangladesh Chak report."""

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
DISCOVERY = ROOT / "data/other/forms/raw_data/sil_chak_2007_discovery.json"


def record():
    return json.loads(DISCOVERY.read_text(encoding="utf-8"))


def test_primary_cover_metadata_is_pinned_but_not_misclassified_as_report():
    source = record()
    assert source["canonical_candidate_title"] == (
        "The Chak of Bangladesh: A Sociolinguistic Study"
    )
    assert source["discovery_state"] == (
        "official_locator_confirmed_publisher_pdf_unavailable"
    )
    cover = source["primary_cover_asset"]
    assert cover["visually_verified"] is True
    assert cover["visible_title"] == source["canonical_candidate_title"]
    assert cover["visible_compilers"] == [
        "Loren Maggard", "Mridul Sangma", "Sayed Ahmad"
    ]
    assert cover["visible_field_researchers"] == ["Mridul Sangma", "Sayed Ahmad"]
    assert cover["visible_date"] == "June 2007"
    assert cover["visible_publisher"] == "SIL Bangladesh"
    assert cover["sha256"] == (
        "22dee3662876dcb29b467a5f12b9fa35d875233bcd218ea82d6e1d61983116f6"
    )
    assert cover["dimensions"] == "285 x 380 pixels"
    assert "not the report" in cover["scope"]
    assert source["publisher_pdf_acquired"] is False


def test_exhaustive_official_archive_checks_still_leave_acquisition_blocked():
    source = record()
    audit = source["acquisition_audit"]
    assert audit["official_asset_prefix_audit"] == {
        "scope": (
            "Every unique successful application/pdf capture indexed under the "
            "official www.silbangladesh.org/sites/ban/files/ prefix."
        ),
        "unique_pdf_digests": 87,
        "chak_report_matches": 0,
        "cdx_response_sha256": (
            "12c271a36245f970f870f6eb7b5b6410dee49719baee30d41426998537cd157e"
        ),
        "cdx_response_bytes": 15459,
    }
    esr = audit["sil_esr_2007_audit"]
    assert esr["unique_report_files"] == 17
    assert esr["issue_numbers"] == [
        "003", "004", "005", "007", "008", "010", "012", "013", "014",
        "015", "016", "017", "019", "020", "022", "023", "024",
    ]
    assert esr["chak_report_matches"] == 0
    assert audit["internet_archive_exact_title_audit"]["catalog_matches"] == 0
    assert "request a scan" in audit["next_action"]


def test_no_lexical_topology_or_reading_is_claimed_without_the_report():
    source = record()
    primary = source["primary_source_verification"]
    assert primary["publisher_pdf_acquired"] is False
    assert primary["sha256"] is None
    assert primary["rendered_pages"] is False
    assert primary["wordlist_presence_verified_from_publisher_pages"] is False
    assert primary["wordlist_page_coordinates"] == []
    assert primary["wordlist_topology_verified"] is False
    assert primary["manual_transcription_state"] == "not_started_acquisition_blocked"
    assert "all publisher report pages, wordlist lists and cells" in (
        primary["unresolved_coordinates"]
    )
    assert "Secondary works establish" in source["policy"]
    assert "cannot supply or verify Jambu forms" in source["policy"]

