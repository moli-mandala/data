"""Focused acquisition guards for the unavailable Chakma-Tanchangya report."""

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
DISCOVERY = (
    ROOT / "data/other/forms/raw_data/sil_bangladesh_2007_unlisted_reports_discovery.json"
)
TITLE = "A Sociolinguistic Survey among the Chakma and Tanchangya Communities"


def record():
    manifest = json.loads(DISCOVERY.read_text(encoding="utf-8"))
    return next(report for report in manifest["reports"] if report["title"] == TITLE)


def test_candidate_identity_is_not_promoted_beyond_secondary_evidence():
    report = record()
    assert report["authors"] == ["Loren Maggard", "Mridul Sangma", "Sayed Ahmad"]
    assert report["year"] == 2007
    assert report["publisher"] is None
    assert report["extent"] is None
    assert report["series"] is None
    assert report["series_issue"] is None
    assert report["archive_entry_number"] is None
    assert report["primary_artifact"] == {
        "cover_found": False,
        "report_found": False,
        "publisher_pdf_url_found": False,
        "visually_verified": False,
        "verification_scope": (
            "No primary cover, title page, archive record, or report file was retrievable. "
            "Exact title, authorship, year, and Dhaka manuscript note remain "
            "secondary-bibliography metadata."
        ),
    }
    assert report["publisher_file_acquired"] is False
    assert report["publisher_pdf_sha256"] is None


def test_official_catalog_and_asset_searches_are_exhaustively_accounted_for():
    report = record()
    official = report["official_locator"]
    audit = report["acquisition_audit"]
    assert official["archive_language_counts"] == {
        "Chakma [ccp]": 4,
        "Tangchangya [tnv]": 4,
    }
    assert official["archive_language_catalog_wayback_captures"] == {
        "Chakma [ccp]": 0,
        "Tangchangya [tnv]": 0,
    }
    assert audit["official_pdf_asset_audit"]["unique_pdf_digests"] == 87
    assert audit["official_pdf_asset_audit"]["chakma_tanchangya_report_matches"] == 0
    assert audit["official_jpeg_asset_audit"]["unique_jpeg_digests"] == 265
    assert audit["official_jpeg_asset_audit"]["chakma_tanchangya_survey_cover_matches"] == 0
    assert audit["internet_archive_exact_title_audit"]["catalog_matches"] == 0
    assert audit["official_language_result_cdx_audit"]["successful_captures_each"] == 0
    assert "eight archive records" in audit["next_action"]


def test_no_lexical_or_ipa_topology_is_claimed_without_primary_pages():
    report = record()
    evidence = report["independent_lexical_evidence"]
    topology = report["primary_source_topology"]
    assert report["lexical_scope"] == (
        "wordlist_comparison_reported_secondarily_published_forms_unverified"
    )
    assert evidence["published_wordlist_presence_verified"] is False
    assert evidence["ipa_wordlist_presence_verified"] is False
    assert topology["report_page_count_verified"] is False
    assert topology["wordlist_presence_verified"] is False
    assert topology["ipa_wordlist_presence_verified"] is False
    assert topology["wordlist_pages"] == []
    assert topology["wordlist_lists"] == []
    assert topology["wordlist_cells"] == []
    assert topology["manual_transcription_state"] == "not_started_acquisition_blocked"
    assert "all report pages, lexical/IPA lists, tables, and cells" in (
        topology["unresolved_coordinates"]
    )

