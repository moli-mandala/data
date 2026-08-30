"""Focused acquisition guards for the unavailable SIL Chittagonian report."""

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
DISCOVERY = (
    ROOT / "data/other/forms/raw_data/sil_chittagonian_2007_discovery.json"
)


def record():
    return json.loads(DISCOVERY.read_text(encoding="utf-8"))


def test_primary_cover_metadata_is_pinned_without_becoming_report_evidence():
    source = record()
    assert source["canonical_title"] == (
        "A Sociolinguistic Survey of the Chittagonian-speaking Community"
    )
    assert source["discovery_state"] == (
        "primary_cover_identified_publisher_report_unavailable"
    )
    cover = source["primary_cover_asset"]
    assert cover["visually_verified"] is True
    assert cover["visible_title"] == source["canonical_title"]
    assert cover["visible_compilers"] == [
        "Loren Maggard", "Mridul Sangma", "Sayed Ahmad"
    ]
    assert cover["visible_field_researchers"] == [
        "Sayed Ahmad", "Mridul Sangma"
    ]
    assert cover["visible_date"] == "February 2007"
    assert cover["visible_publisher"] == "SIL Bangladesh"
    assert cover["bytes"] == 19_982
    assert cover["dimensions"] == "285 x 380 pixels"
    assert cover["sha256"] == (
        "f5870274f73a2baaed1b7b0f57c81ddbe05e272962e7dcd8eb8c93ea3fe2eb55"
    )
    assert "not the report PDF" in cover["scope"]
    assert source["publisher_pdf_acquired"] is False


def test_reproducible_official_archive_checks_leave_acquisition_blocked():
    audit = record()["acquisition_audit"]
    assert audit["official_listing_capture"] == {
        "url": (
            "https://web.archive.org/web/20241111175402id_/"
            "https://www.silbangladesh.org/resources/publications/survay_report"
        ),
        "bytes": 47_093,
        "sha256": (
            "18abb6dbbc580cf768d49c2137198033cceff0bb3bfc54aef3a45f67654fd84a"
        ),
    }
    official_pdfs = audit["official_domain_pdf_audit"]
    assert official_pdfs["indexed_pdf_urls"] == 76
    assert official_pdfs["unique_pdf_digests"] == 72
    assert official_pdfs["chittagonian_report_matches"] == 0
    assert official_pdfs["response_sha256"] == (
        "ac42ec77c8e180a55f7ff167b5f1cbfa4f14ee6fa204df457226839d0f63e994"
    )
    assert audit["official_chittagonian_asset_prefix_audit"][
        "successful_matches"
    ] == 0
    assert audit["internet_archive_exact_title_audit"]["catalog_matches"] == 0
    assert audit["direct_publisher_filename_probe"]["result"] == "inconclusive"
    assert "request the complete report" in audit["next_action"]


def test_no_topology_or_reading_is_claimed_without_the_primary_report():
    source = record()
    topology = source["primary_source_topology"]
    assert source["publisher_pdf_sha256"] is None
    assert source["publisher_pdf_rendered"] is False
    assert topology["report_page_count_verified"] is False
    assert topology["wordlist_presence_verified"] is False
    assert topology["ipa_wordlist_presence_verified"] is False
    assert topology["wordlist_pages"] == []
    assert topology["wordlist_lists"] == []
    assert topology["wordlist_cells"] == []
    assert topology["manual_transcription_state"] == (
        "not_started_acquisition_blocked"
    )
    assert "all report pages, lists, and cells" in topology["unresolved_coordinates"]
    assert source["independent_lexical_evidence"]["state"] == "none_found"
    assert "may not supply or verify" in (
        source["independent_lexical_evidence"]["policy"]
    )
    assert "manually transcribed and verified cell-by-cell" in source["policy"]
