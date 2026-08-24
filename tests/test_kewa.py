import csv
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).parents[1]
RAW = ROOT / "data/other/forms/raw_data"
SCRIPT = RAW / "kewa.py"
SPEC = importlib.util.spec_from_file_location("kewa_importer", SCRIPT)
kewa = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = kewa
SPEC.loader.exec_module(kewa)


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_head_matching_is_conservative_about_accent_length_and_segments():
    assert kewa.normalize_head("mā́lā-", preserve_accent=False) == kewa.normalize_head(
        "mālā", preserve_accent=False
    )
    assert kewa.normalize_head("mā́lā-", preserve_accent=True) != kewa.normalize_head(
        "mālā", preserve_accent=True
    )
    assert kewa.normalize_head("mālā", preserve_accent=False) != kewa.normalize_head(
        "mala", preserve_accent=False
    )
    assert kewa.normalize_head("aṅkura", preserve_accent=False) != kewa.normalize_head(
        "ankura", preserve_accent=False
    )
    assert kewa.head_keys(
        "krakacam", preserve_accent=False, source_inflection=True
    ) == {"krakacam", "krakaca"}
    assert kewa.head_keys(
        "araḥ", preserve_accent=False, source_inflection=True
    ) == {"araḥ", "ara"}


def test_cdial_addenda_are_canonicalized_before_matching():
    accented, _, _, valid_ids = kewa.cdial_indexes(kewa.DEFAULT_CDIAL, kewa.DEFAULT_MERGES)
    merges = kewa.load_merges(kewa.DEFAULT_MERGES)
    assert merges["14349"] == "2680"
    assert {entry_id for entry_id, _ in accented[kewa.normalize_head(
        "kaṇṭhá-", preserve_accent=True
    )]} == {"2680"}
    assert {"14349", "2680"} <= valid_ids


def test_checked_in_snapshot_is_complete_auditable_and_reproducible():
    audit = read_rows(RAW / "20260818-kewa-audit.csv")
    blocks = read_rows(ROOT / "data/other/entry_texts/20260818-kewa.csv")
    sample = read_rows(RAW / "20260818-kewa-sample.csv")
    manifest = json.loads((RAW / "20260818-kewa-manifest.json").read_text(encoding="utf-8"))

    assert len(audit) == 9587 == manifest["source_articles"]
    assert Counter(row["Volume"] for row in audit) == {"I": 4108, "II": 2811, "III": 2668}
    assert Counter(row["Status"] for row in audit) == manifest["audit_status_counts"]
    assert Counter(row["Status"] for row in audit) == {
        "ingested": 2400,
        "ambiguous": 886,
        "unmatched": 6301,
    }
    assert len(blocks) == 2432 == manifest["installed_text_blocks"]
    assert len({row["Form_ID"] for row in blocks}) == 2376 == manifest["accepted_target_count"]
    assert sum(int(row["Output_Blocks"]) for row in audit) == len(blocks)
    assert len({row["Upstream_ID"] for row in audit}) == len(audit)
    assert len({row["Entry_Key"] for row in audit}) == len(audit)
    assert len({row["Stable_Anchor"] for row in audit}) == len(audit)
    assert len({row["Image_URL"] for row in audit}) == len(audit)
    assert all(row["Raw_OCR"].strip() and "�" not in row["Raw_OCR"] for row in audit)
    assert all(re.fullmatch(r"[0-9a-f]{64}", row["Image_SHA256"]) for row in audit)
    assert all(int(row["Image_Width"]) >= 100 for row in audit)
    assert all(int(row["Image_Height"]) >= 20 for row in audit)
    assert all(int(row["Image_Bytes"]) > 0 for row in audit)
    assert Counter(row["OCR_Config"] for row in audit) == {
        "-l script/Latin --psm 6 --dpi 300; NFC": 9586,
        "-l script/Latin --psm 11 --dpi 300; NFC": 1,
    } == manifest["ocr_config_counts"]
    assert sum(row["Is_Supplement"] == "yes" for row in audit) == 225
    assert sum(bool(row["Locator_Note"]) for row in audit) == 3

    assert all(row["Kind"] == "etymology" and row["Format"] == "html" for row in blocks)
    assert all(row["Source"].startswith("mayrhofer-kewa[vol. ") for row in blocks)
    assert len({(row["Form_ID"], row["Position"]) for row in blocks}) == len(blocks)
    audit_by_id = {row["Upstream_ID"]: row for row in audit}
    for block in blocks:
        article_id = str(int(block["Position"]) - 300000)
        article = audit_by_id[article_id]
        assert article["Raw_OCR"] not in block["Content"]
        assert article["Image_URL"] in block["Content"]
        assert f"{kewa.INDEX_URL}#{article['Stable_Anchor']}" in block["Content"]
        assert "Unreviewed OCR" not in block["Content"]
    assert manifest["ocr_database_policy"] == (
        "audit only; OCR text is not installed in the database"
    )
    assert manifest["match_evidence"] == (
        "authoritative index heads only; OCR is not used for matching"
    )
    assert not any("semantic overlap" in row["Candidate_Evidence"] for row in audit)

    # This is the exact output of the importer's offline rebuild path.
    assert kewa.text_rows(audit) == blocks
    assert kewa.build_manifest(audit, blocks) == manifest
    assert kewa.sample_rows(audit, RAW / "20260818-kewa-sample.csv") == sample


def test_known_edge_cases_are_conservatively_accounted_for():
    audit = {
        row["Upstream_ID"]: row
        for row in read_rows(RAW / "20260818-kewa-audit.csv")
    }
    assert all(audit[value]["Status"] == "ambiguous" for value in ("1", "2", "3"))
    assert audit["4"]["Status"] == "ambiguous"
    assert audit["44"]["Status"] == "ambiguous"
    assert audit["7158"]["Status"] == "unmatched"
    assert "--psm 11" in audit["7158"]["OCR_Config"]
    assert audit["7158"]["Raw_OCR"] == "rikvå, s. trikvā."
    assert audit["9363"]["Accepted_Targets"] == "1"
    assert {
        row["Upstream_ID"]: row["Printed_Pages"]
        for row in audit.values()
        if row["Locator_Note"]
    } == {"4119": "10-11", "4125": "11-12", "4130": "12-13"}


def test_seeded_twenty_article_visual_review_has_no_structural_or_mapping_errors():
    audit = read_rows(RAW / "20260818-kewa-audit.csv")
    expected = {str(value) for value in kewa.REVIEW_SAMPLE_IDS}
    sample = read_rows(RAW / "20260818-kewa-sample.csv")
    assert len(sample) == 20
    assert {row["Upstream_ID"] for row in sample} == expected
    assert all(
        row["OCR_Compared"] == row["Head_Match_Compared"] == "yes" for row in sample
    )
    assert all(row["OCR_Character_Perfect"] == "no" for row in sample)
    assert all(row["Material_Structural_Error"] == "no" for row in sample)
    assert all(row["Review_Notes"] for row in sample)


def test_kewa_contributes_prose_only_and_all_blocks_survive_the_full_build():
    raw = read_rows(ROOT / "data/other/entry_texts/20260818-kewa.csv")
    compiled = read_rows(ROOT / "cldf/entry-texts.csv")
    compiled_keys = {
        (row["Form_ID"], row["Position"], row["Source"], row["Content"])
        for row in compiled
    }
    assert all(
        (row["Form_ID"], row["Position"], row["Source"], row["Content"]) in compiled_keys
        for row in raw
    )
    assert all(
        "mayrhofer-kewa" not in row["Source"]
        for row in read_rows(ROOT / "cldf/forms.csv")
    )
    assert all(
        "mayrhofer-kewa" not in row["Source"]
        for row in read_rows(ROOT / "cldf/edges.csv")
    )


def test_reference_metadata_records_scope_audit_provenance_and_license_state():
    source = (ROOT / "cldf/sources.bib").read_text(encoding="utf-8")
    entry = source.split("@book{mayrhofer-kewa,", 1)[1].split("\n}", 1)[0]
    assert "9,587" in entry
    assert "2021 version 1.0" in entry
    assert "reuse terms are not stated" in entry
    assert "20260818-kewa-audit.csv" in entry
    assert "ocr" in entry.casefold()

    references = {
        row["ID"]: row for row in read_rows(ROOT / "cldf/references.csv")
    }
    assert references["mayrhofer-kewa"]["OCR"] == "No"
    assert "samskrtam.ru/sanskrit-lexicon/KEWA" in references["mayrhofer-kewa"]["Source"]
