import csv
import gzip
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "data/other/forms/raw_data/berger_cleanup.py"
SPEC = importlib.util.spec_from_file_location("berger_cleanup_test", SCRIPT)
cleanup = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = cleanup
SPEC.loader.exec_module(cleanup)

AUTO = list(csv.reader((ROOT / "data/other/forms/20260726-berger-auto.csv").open(encoding="utf-8")))
GOLD = list(csv.reader((ROOT / "data/other/forms/20220930-berger.csv").open(encoding="utf-8")))
EDITORIAL = list(csv.DictReader(
    (ROOT / "data/other/forms/raw_data/20260828-berger-editorial.csv").open(encoding="utf-8")
))
ENTRY_MAP = list(csv.DictReader(
    (ROOT / "data/other/forms/raw_data/20260828-berger-entry-map.csv").open(encoding="utf-8")
))
with gzip.open(ROOT / "data/other/forms/raw_data/20260828-berger-audit.csv.gz", "rt", encoding="utf-8") as stream:
    AUDIT = list(csv.DictReader(stream))
SAMPLE = list(csv.DictReader(
    (ROOT / "data/other/forms/raw_data/20260828-berger-sample.csv").open(encoding="utf-8")
))
MANIFEST = json.loads(
    (ROOT / "data/other/forms/raw_data/20260828-berger-manifest.json").read_text(encoding="utf-8")
)


def test_scan_scope_restores_page_nine_and_excludes_duplicate_spread():
    assert cleanup.allowed_columns(7) == (2, 3)
    assert cleanup.allowed_columns(247) == (0, 1)
    assert cleanup.printed_page(50, 0) == 94
    assert cleanup.printed_page(50, 1) == 95
    assert cleanup.printed_page(52, 0) == 96
    assert cleanup.printed_page(52, 1) == 97
    reparsed = [row for row in AUDIT if row["Status"] != "installed-preserved"]
    assert {int(row["Printed_Page"]) for row in reparsed} == set(range(9, 487))
    assert "51" not in {row["PDF_Page"] for row in reparsed}
    assert {"7", "247"} <= {row["PDF_Page"] for row in reparsed}


def test_line_reconstruction_and_output_counts_are_exact():
    assert MANIFEST["counts"] == {
        "raw_units": 9700,
        "parsed_entries_including_variants": 11039,
        "audited_records": 11636,
        "installed_auto_rows": 10664,
        "preserved_graph_evidence_rows": 597,
        "gold_rows": 39,
        "excluded_entries": 939,
        "untranslated_entries": 0,
        "direct_turner_links": 431,
    }
    assert len(AUDIT) == MANIFEST["counts"]["audited_records"]
    assert len(GOLD) == 39
    assert len(SAMPLE) == 20


def test_installed_rows_are_rich_english_rows_with_stable_keys():
    rows = [*AUTO, *GOLD]
    assert {len(row) for row in rows} == {15}
    assert all(row[0] in {"Bur", "Werch"} and row[2] for row in rows)
    assert all(row[7].startswith(("berger-auto[p. ", "berger[p. ")) for row in rows)
    assert all(row[10] for row in rows)
    assert len({row[10] for row in rows}) == len(rows)
    assert all("�" not in "".join(row) for row in rows)
    assert all("uncertain" in row[14].split() for row in AUTO)
    # High-frequency German function words must not survive as standalone gloss words.
    german = re.compile(r"\b(?:der|die|das|und|oder|nicht|sich|einer|einen|eine)\b", re.I)
    assert sum(bool(german.search(row[3])) for row in rows if row[3]) <= 35
    assert not any("see das Fg" in row[3] for row in rows)
    assert not any("<0x" in row[3] for row in rows)


def test_editorial_layer_is_complete_and_bound_to_german_source_hashes():
    by_key = {row["Entry_Key"]: row for row in EDITORIAL}
    assert len(by_key) == len(EDITORIAL)
    assert all(row["English_Gloss"] and row["Source_SHA256"] for row in EDITORIAL)
    assert {row["Model"] for row in EDITORIAL} == {"de_en 1.3"}
    assert {row["Review"] for row in EDITORIAL} <= {
        "machine-translated-unreviewed", "editorial-override",
    }
    for row in AUDIT:
        if row["Raw_Gloss_German"] and row["Status"] not in {"excluded"}:
            assert row["Installed_Key"] in by_key


def test_identity_crosswalk_preserves_the_large_majority_of_old_source_keys():
    assert len(ENTRY_MAP) >= 7_000
    assert len({row["Stable_Key"] for row in ENTRY_MAP}) == len(ENTRY_MAP)
    assert len({row["Installed_Key"] for row in ENTRY_MAP}) == len(ENTRY_MAP)
    assert {row["Method"] for row in ENTRY_MAP} <= {
        "page-sequence", "printed-form", "unique-page-form", "exact-variant",
    }
    assert all(float(row["Score"]) >= 0.55 for row in ENTRY_MAP)


def test_turner_links_are_direct_and_hedged_comparisons_remain_unlinked():
    valid = {"644", "11433", "1464", "3315", "14398"}
    assert cleanup.direct_turner_ids("(T 644)", valid) == ["644"]
    assert cleanup.direct_turner_ids("(T 3315; 14398)", valid) == ["3315", "14398"]
    assert cleanup.direct_turner_ids("(vgl. T 11433?)", valid) == []
    assert cleanup.direct_turner_ids("(zu T 1464)", valid) == []
    linked = [row for row in AUDIT if row["Direct_Turner_IDs"]]
    assert linked
    assert all("?" not in row["Direct_Turner_IDs"] for row in linked)


def test_false_comparative_variants_and_damaged_rows_stay_out():
    forms = {row[2].casefold() for row in AUTO}
    assert not forms & {"indoar.", "vgl.", "skt.", "a$", "å-"}
    assert not forms & {"brücke", "mensch", "glas", "fisch", "táś", "séance"}
    assert not any(any(character.isdigit() for character in form) for form in forms)
    by_key = {row[10]: row for row in AUTO}
    assert by_key["berger-entry-10531"][3] == "impure by pollution, menstruation, or if one has not bathed after coitus"
    assert by_key["berger-entry-2690:dialect:1"][3] == "in heat, rutting"
    assert by_key["berger-entry-5209"][2] == "khóośo"
    assert by_key["berger-entry-6046"][2] == "móṭis"
    assert by_key["berger-entry-6054:dialect:2"][2] == "muṣṭí"
    assert by_key["berger-entry-7319"][3] == "concubine"
    assert by_key["berger-entry-997:dialect:1"][2:4] == ["biéeço", "weaver"]
    assert by_key["berger:p395:c2:e008"][2:4] == [
        "śiridáko", "main post of the house, plays a role in rites",
    ]
    statuses = Counter(row["Status"] for row in AUDIT)
    assert statuses["excluded"] == MANIFEST["counts"]["excluded_entries"]
    assert statuses["installed-preserved"] == MANIFEST["counts"]["preserved_graph_evidence_rows"]
    assert all(row["Review"] for row in AUDIT if row["Status"] == "excluded")


def test_seeded_twenty_entry_source_image_audit_is_clean():
    assert all(row["Source_Image_Check"] == "checked" for row in SAMPLE)
    assert all(row["Material_Error"] == "no" for row in SAMPLE)
    assert all(row["Resolution"] for row in SAMPLE)
