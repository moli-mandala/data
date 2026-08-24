import csv
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
RAW_DATA = ROOT / "data/other/forms/raw_data"
SCRIPT = RAW_DATA / "hockings_badaga.py"
sys.path.insert(0, str(RAW_DATA))
SPEC = importlib.util.spec_from_file_location("hockings_badaga_extractor", SCRIPT)
badaga = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = badaga
SPEC.loader.exec_module(badaga)


def test_scan_leaf_mapping_excludes_two_inserted_blanks():
    assert badaga.printed_page_for(21) == 1
    assert badaga.printed_page_for(442) == 422
    assert badaga.printed_page_for(445) == 423
    assert badaga.printed_page_for(643) == 621
    assert badaga.parse_page_spec(None)[-1] == 643
    assert 443 not in badaga.parse_page_spec(None)
    assert 444 not in badaga.parse_page_spec(None)


def test_head_parser_prefers_the_first_grammatical_label():
    line = badaga.OCRLine(
        1, 1, 1,
        "betta n. 1. country, surrounding countryside; 2. mountain, sfx in names",
        225, 300, 1200, 340, 95.0,
    )
    assert badaga.head_and_label([line]) == ("betta", "n.")

    line.text = "-a infinitive sfx (vb. stem 0 +a)"
    assert badaga.head_and_label([line]) == ("-a", "sfx.")


def test_dedr_parser_links_only_existing_targets():
    valid = {"7": "7", "2826": "2826", "196a": "196a"}
    links, invalid, uncertain = badaga.dedr_links(
        "? DEDR 7, 2826, 196a; DEDR App. 6", valid
    )
    assert links == ["7", "2826", "196a"]
    assert invalid == ["App. 6"]
    assert uncertain


def test_cf_above_copies_the_named_preceding_definition():
    def entry(top, text):
        line = badaga.OCRLine(1, 1, 1, text, 225, top, 1200, top + 40, 95.0)
        result = badaga.Entry(21, 1, 1, top, [line])
        result.head, result.label = badaga.head_and_label(result.lines)
        return result

    entries = [
        entry(100, "akki n. bird, avifauna"),
        entry(200, "akki ganje n. barley"),
        entry(300, "akkilu/hakkilu/hakki/akki cf. above, akki/akkilu/hakki/hakkilu"),
        entry(400, "me:l ole n. above the hearth"),
    ]
    rows, audit = badaga.build_rows(entries, {})
    by_key = {row[10]: row for row in rows}

    assert by_key[entries[2].key][3] == "bird, avifauna"
    assert audit[2]["Gloss"] == "bird, avifauna"
    assert by_key[entries[3].key][3] == "above the hearth"
    assert audit[3]["Gloss"] == "above the hearth"


def test_checked_in_dictionary_is_fully_accounted_and_review_marked():
    forms_path = ROOT / "data/other/forms/20260818-hockings-badaga.csv"
    audit_path = RAW_DATA / "20260818-hockings-badaga-audit.csv"
    with forms_path.open(encoding="utf-8", newline="") as stream:
        forms = list(csv.reader(stream))
    with audit_path.open(encoding="utf-8", newline="") as stream:
        audit = list(csv.DictReader(stream))

    assert len(audit) == 9993
    assert len(forms) == 16706
    assert all(len(row) == badaga.RICH_COLUMNS for row in forms)
    assert {row[0] for row in forms} == {"Badaga"}
    assert all(row[7].startswith("hockings-pilotraichoor1992[p. ") for row in forms)
    assert len({row["Entry_Key"] for row in audit}) == len(audit)
    assert {row["Status"] for row in audit} == {"ingested"}
    assert {row["Review_State"] for row in audit} == {"needs_transcription_review"}
    corrections_path = RAW_DATA / "20260818-hockings-badaga-corrections.csv"
    with corrections_path.open(encoding="utf-8", newline="") as stream:
        corrections = list(csv.DictReader(stream))
    reviewed_keys = {row["Entry_Key"] for row in corrections}
    assert len(reviewed_keys) == 20
    for row in forms:
        source_key = row[10].split(":link:", 1)[0].split(":variant:", 1)[0]
        if source_key not in reviewed_keys:
            assert "uncertain" in row[14].split()
    assert sum(bool(row["DEDR_IDs"]) for row in audit) == 6421
    assert sum(bool(row["Unresolved_DEDR_IDs"]) for row in audit) == 93
    assert not any("�" in row[2] for row in forms)
    assert not any(tag.startswith("dialect:") for row in forms for tag in row[14].split())

    by_key = {row["Entry_Key"]: row for row in audit}
    assert by_key["hockings-pilotraichoor1992:p5:c2:y0097"]["Form"] == "akkurama"
    # This visually dotted source head is intentionally not guessed from the
    # lossy OCR; it remains in the image-backed correction queue.
    edekadu = by_key["hockings-pilotraichoor1992:p80:c1:y0193"]
    assert edekadu["Form"] == "Edeka:du"
    assert edekadu["Review_State"] == "needs_transcription_review"

    installed_by_key = {row[10]: row for row in forms}
    assert installed_by_key["hockings-pilotraichoor1992:p5:c2:y0229"][2] == "agaṭu madilu"
    assert installed_by_key["hockings-pilotraichoor1992:p5:c2:y0404"][2] == "agaṇḍam"
    assert installed_by_key["hockings-pilotraichoor1992:p5:c1:y0368"][3] == "bird, avifauna DBIA 233b"
    assert by_key["hockings-pilotraichoor1992:p5:c1:y0368"]["Gloss"] == (
        "bird, avifauna DBIA 233b"
    )
    leading_above = [row for row in forms if row[3].casefold().startswith("above,")]
    # This is the inherited lexical definition of osatti/osti, not an
    # unresolved cross-reference marker.
    assert {
        row[10].split(":link:", 1)[0].split(":variant:", 1)[0]
        for row in leading_above
    } == {
        "hockings-pilotraichoor1992:p108:c1:y0267",
        "hockings-pilotraichoor1992:p108:c2:y0492",
        "hockings-pilotraichoor1992:p476:c1:y0071",
        "hockings-pilotraichoor1992:p493:c2:y0257",
    }


def test_scan_backed_calibration_has_twenty_final_passes():
    path = RAW_DATA / "20260818-hockings-badaga-calibration.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 20
    assert {row["Structural_Parse"] for row in rows} == {"PASS"}
    assert {row["Final_Result"] for row in rows} == {"PASS"}
    assert sum(row["Transcription_Decision"] == "corrected" for row in rows) == 4
