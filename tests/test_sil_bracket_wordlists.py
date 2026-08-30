"""Regression tests for the SIL Bangladesh bracketed-site-code wordlist ingests."""

import csv
import json
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
COMPILED = ROOT / "cldf/forms.csv"

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
    "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
    "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]

# report -> (installed, audit, source key, sites per language, item count)
REPORTS = {
    "tripura": (
        "20260826-sil-tripura.csv", "20260826-sil-tripura-audit.csv",
        "kim-kim-sangma-ahmad2011tripura",
        {"Kokborok": 20, "Garo": 3, "B": 1},
        306,
    ),
    "kochbd": (
        "20260826-sil-kochbd.csv", "sil_kochbd_2011_manual/staging_audit.tsv",
        "kim-ahmad-kim-sangma2011kochbd",
        {"Koch": 4},
        307,
    ),
    "garobd": (
        "20260826-sil-garobd.csv", "20260826-sil-garobd-audit.csv",
        "kim-kim-sangma2012garo",
        {"Garo": 11, "Koch": 2, "Megam": 2, "Lyngngam": 1, "B": 1},
        307,
    ),
}


def installed(name):
    with (ROOT / "data/other/forms" / REPORTS[name][0]).open(encoding="utf-8", newline="") as s:
        return [dict(zip(FORM_FIELDS, r)) for r in csv.reader(s)]


def audited(name):
    with (ROOT / "data/other/forms/raw_data" / REPORTS[name][1]).open(
            encoding="utf-8", newline="") as s:
        if name == "kochbd":
            return [
                {
                    **row,
                    "Status": (
                        "installed" if row["disposition"] == "staged_target" else "excluded"
                    ),
                    "Item": row["item"],
                }
                for row in csv.DictReader(s, delimiter="\t")
            ]
        return list(csv.DictReader(s))


@pytest.mark.parametrize("name", sorted(REPORTS))
def test_rows_are_survey_lects_with_one_site_tag(name):
    for row in installed(name):
        assert row["Language_ID"] in REPORTS[name][3]
        tags = row["Tags"].split()
        assert len(tags) == 1 and tags[0].startswith(f"dialect:{row['Language_ID']}:")
        assert row["Parameter_ID"] == row["Cognateset"] == row["Etymology"] == ""
        assert row["Form"] and row["Form"] == row["Phonemic"] and row["Native"] == ""


@pytest.mark.parametrize("name", sorted(REPORTS))
def test_site_counts_match_the_printed_key(name):
    sites = {language: set() for language in REPORTS[name][3]}
    for row in installed(name):
        sites[row["Language_ID"]].add(row["Tags"].split(":")[2])
    assert {k: len(v) for k, v in sites.items()} == REPORTS[name][3]


@pytest.mark.parametrize("name", sorted(REPORTS))
def test_every_line_is_accounted_for(name):
    status = Counter(row["Status"] for row in audited(name))
    assert status["unparsed"] == 0
    assert status["unmapped"] == 0
    assert status["installed"] == len(installed(name))
    # the printed "no entry" gaps are excluded, not installed
    assert status["excluded"] > 0


def test_remaining_incomplete_legacy_font_recovery_is_counted_and_queued_for_manual_review():
    """Do not let the still-partial Garo report regress to a false `done` state."""
    expected = {
        "garobd": (4444, 712, 91, 17),
    }
    for name, counts in expected.items():
        dispositions = Counter()
        for row in audited(name):
            if row["Status"] == "installed":
                dispositions["installed"] += 1
            elif "glyph with no verified reading" in row["Reason"]:
                dispositions["unverified_glyph"] += 1
            elif "at this site" in row["Reason"]:
                dispositions["site_gap"] += 1
            elif "at any site" in row["Reason"]:
                dispositions["global_gap"] += 1
            else:
                dispositions["other"] += 1
        assert dispositions["other"] == 0
        assert tuple(dispositions[key] for key in (
            "installed", "unverified_glyph", "site_gap", "global_gap"
        )) == counts

    census = (ROOT / "data/other/forms/raw_data/sil_survey_sources.md").read_text(
        encoding="utf-8"
    )
    queue = (ROOT / "data/other/forms/raw_data/sil_bangladesh_legacy_manual_queue.md").read_text(
        encoding="utf-8"
    )
    acquisition = json.loads(
        (ROOT / "data/other/forms/raw_data/sil_bangladesh_legacy_acquisition_manifest.json")
        .read_text(encoding="utf-8")
    )
    queue_flat = " ".join(queue.split())
    assert "exclude 1,514 attested cells" in census
    assert "Together they expose 1,514 attested records" in queue
    assert "all three page-image sets are now ready" in queue_flat
    assert "not permission to fall back to OCR, raw legacy glyphs" in queue_flat
    assert (
        "2,149 conceptual cells / 2,159 expanded rows = 1,780 attested, "
        "25 printed blanks, 225 unresolved legacy-modifier cells, and 119 "
        "globally unused cells"
    ) in queue_flat
    assert "sil_kochbd_2011_manual/" in queue
    koch_acquisition = acquisition["reports"]["silesr2011_023"]
    assert koch_acquisition["replay_retrieved"] is True
    assert koch_acquisition["canonical_sha256"] == (
        "d1b2d597c16fd0338ad47d2bf031566192c5ff4e26a6651de14a228df681fc10"
    )
    assert len(koch_acquisition["wayback_captures"]) == 2
    assert koch_acquisition["wayback_captures"][1] == {
        "cdx_digest": "OMJBRTPWF6NJBMQ6EG22V3IEOGDFTWB2",
        "cdx_length": 909321,
        "mimetype": "application/pdf",
        "original": "http://www-01.sil.org/silesr/2011/silesr2011-023.pdf",
        "timestamp": "20170809124914",
    }
    assert koch_acquisition["wordlist_render"]["rendered_page_count"] == 20
    assert koch_acquisition["wordlist_render"]["physical_pages"] == [43, 62]
    assert "manual review and shared source integration complete; consolidated build pending" in queue
    kurux_acquisition = acquisition["reports"]["silesr2011_040"]
    assert kurux_acquisition["archive_entry_number"] == 41654
    assert kurux_acquisition["archive_entry_url"] == (
        "https://www.silbangladesh.org/resources/archives/41654"
    )
    assert kurux_acquisition["official_extent"] == "89 pages"
    assert "wordlist comparisons" in kurux_acquisition["official_lexical_evidence"]
    assert kurux_acquisition["publisher_link_result"].startswith("HTTP 403 Forbidden")
    assert kurux_acquisition["replay_retrieved"] is True
    assert kurux_acquisition["canonical_sha256"] == (
        "f2f06c25ac55462d6a40843539d8417e24a647bd1eb0bbe3f24ea3e45f0b9e4b"
    )
    assert kurux_acquisition["wayback_captures"] == [{
        "cdx_digest": "UIINJZKVE4OZCJ4SGUQNU5ZSQUPERV6X",
        "cdx_length": 905546,
        "mimetype": "application/pdf",
        "original": "http://www-01.sil.org/silesr/2011/silesr2011-040.pdf",
        "timestamp": "20170809124903",
    }]
    assert kurux_acquisition["wordlist_render"]["rendered_page_count"] == 19
    assert kurux_acquisition["wordlist_render"]["physical_pages"] == [39, 57]
    assert "archive entry `41654`" in queue
    assert "verified historical replay now provides the required page-image evidence" in queue_flat

    garo_acquisition = acquisition["reports"]["silesr2012_007"]
    assert garo_acquisition["replay_retrieved"] is True
    assert garo_acquisition["canonical_sha256"] == (
        "4248b409d816c153f95c09e50bf51f9e5ff90d456e3c8d9d13dc2eca6f8c4359"
    )
    assert garo_acquisition["wayback_captures"] == [{
        "cdx_digest": "APUVX37LNZW2S56DZXITNJHIPRSBHIUX",
        "cdx_length": 1616418,
        "mimetype": "application/pdf",
        "original": "http://www-01.sil.org/silesr/2012/silesr2012-007.pdf",
        "timestamp": "20170810011246",
    }]
    assert garo_acquisition["wordlist_render"]["rendered_page_count"] == 42
    assert garo_acquisition["wordlist_render"]["physical_pages"] == [52, 93]

    for package, expected_sha, expected_pages in (
        (
            "sil_kurux_2011_manual",
            "f2f06c25ac55462d6a40843539d8417e24a647bd1eb0bbe3f24ea3e45f0b9e4b",
            19,
        ),
        (
            "sil_garobd_2012_manual",
            "4248b409d816c153f95c09e50bf51f9e5ff90d456e3c8d9d13dc2eca6f8c4359",
            42,
        ),
    ):
        source = json.loads(
            (ROOT / "data/other/forms/raw_data" / package / "source_manifest.json")
            .read_text(encoding="utf-8")
        )
        assert source["state"] in {
            "source_acquired_awaiting_manual_review",
            "manual_review_in_progress",
            "partial_manual_review",
            "manual_review_complete",
        } or source["state"].startswith("manual_checkpoint_")
        assert source["source_pdf_sha256"] == expected_sha
        assert source["rendered_page_count"] == expected_pages
        assert source.get("conceptual_cells", source.get("cells_reviewed", 0)) >= 0
    for count in (239, 563, 712):
        assert f"{count} legacy-excluded" in census

    for unit, omitted in {"20260826-sil-garobd": 712}.items():
        checklist = (ROOT / "source_checklists" / f"{unit}.md").read_text(encoding="utf-8")
        assert "Survey wordlists or comparative tables, OCR-heavy source" in checklist
        assert f"{omitted} attested audit records still require independent visual" in checklist
        assert "[ ] 2. Choose the extraction path" in checklist
        assert "[ ] 5. Emit the rich import schema" in checklist
        assert "[ ] 7. Build and verify the sound profile" in checklist
        assert "[ ] 10. Produce a complete audit trail" in checklist

    koch_checklist = (ROOT / "source_checklists/20260826-sil-kochbd.md").read_text(
        encoding="utf-8"
    )
    assert "Installed rows: 1017" in koch_checklist
    assert "225 ambiguity-only conceptual cells" in koch_checklist
    assert "226 coordinates and 226 expanded rows" in koch_checklist
    assert "shared source-specific integration complete" in koch_checklist

    kurux_checklist = (ROOT / "source_checklists/20260826-sil-kurux.md").read_text(
        encoding="utf-8"
    )
    assert "1365 installed records" in kurux_checklist
    assert "the 239 legacy-decoder omissions were independently recovered by hand" in kurux_checklist
    assert "[x] 10. Produce a complete audit trail" in kurux_checklist


@pytest.mark.parametrize("name", sorted(REPORTS))
def test_the_whole_item_list_is_covered_and_keys_are_unique(name):
    rows = installed(name)
    keys = Counter(row["Entry_Key"] for row in rows)
    assert not [key for key, n in keys.items() if n > 1]
    items = {int(row["Item"]) for row in audited(name) if row["Item"]}
    assert items == set(range(1, REPORTS[name][4] + 1))


def test_kurux_manual_install_has_no_legacy_glyph_placeholders():
    """The complete manual install supersedes the partial legacy-font decoder."""
    with (ROOT / "data/other/forms/20260826-sil-kurux.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = [dict(zip(FORM_FIELDS, row)) for row in csv.reader(stream)]
    assert len(rows) == 1365
    assert {row["Language_ID"] for row in rows} == {"Kurux"}
    assert all("\ufffd" not in row["Form"] for row in rows)
    assert all(not any(char in row["Form"] for char in "!%*$") for row in rows)


def test_one_printed_line_becomes_one_row_per_site_code():
    # "2 haphɔŋ [ abjkl ]" is a single printed line standing for five attestations
    rows = [r for r in installed("tripura")
            if r["Gloss"] == "mountain" and r["Form"] == "haphɔŋ"]
    assert len(rows) == 5
    assert len({r["Tags"] for r in rows}) == 5


@pytest.mark.parametrize("name", sorted(REPORTS))
def test_languages_and_sites_are_registered(name):
    with (ROOT / "cldf/languages.csv").open(encoding="utf-8", newline="") as s:
        languages = {row["ID"]: row for row in csv.DictReader(s)}
    with (ROOT / "cldf/dialects.csv").open(encoding="utf-8", newline="") as s:
        dialects = {row["ID"]: row for row in csv.DictReader(s)}
    for language in REPORTS[name][3]:
        assert language in languages and languages[language]["Glottocode"]
    for site in {row["Tags"].split(":")[2] for row in installed(name)}:
        assert site in dialects, site
        # Koch's source-specific integration removes invented legacy centroids.
        if name == "kochbd":
            assert dialects[site]["Latitude"] == dialects[site]["Longitude"] == ""
            assert dialects[site]["Quality"] == ""
        else:
            # Other legacy reports retain explicitly marked approximate points pending review.
            assert dialects[site]["Quality"] == "C"


@pytest.mark.parametrize("name", sorted(REPORTS))
@pytest.mark.skipif(not COMPILED.exists(), reason="cldf/forms.csv has not been built")
def test_every_installed_row_survives_the_full_build(name):
    key = REPORTS[name][2]
    with COMPILED.open(encoding="utf-8", newline="") as s:
        compiled = [
            row for row in csv.DictReader(s)
            if key in {p.split("[", 1)[0].strip() for p in row["Source"].split(";")}
        ]
    if name == "kochbd" and len(compiled) != len(installed(name)):
        pytest.skip(
            "Koch source-specific integration is complete, but consolidated CLDF is explicitly "
            "deferred and still carries the 1,480-row legacy build"
        )
    assert len(compiled) == len(installed(name))
    assert {row["Language_ID"] for row in compiled} == set(REPORTS[name][3])
    assert all(row["Status"] == "unlinked" for row in compiled)
