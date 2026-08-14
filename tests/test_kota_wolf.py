import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "data/other/forms/raw_data/kota_wolf.py"
SPEC = importlib.util.spec_from_file_location("kota_wolf_extractor", SCRIPT)
kota = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = kota
SPEC.loader.exec_module(kota)


def test_dedr_notation_distinguishes_direct_related_and_checked_unlinked():
    links, invalid, checked = kota._dedr_links("x — (77) direct", {"77", "294"})
    assert links == [("77", "direct")]
    assert invalid == [] and not checked

    links, invalid, checked = kota._dedr_links("x — [294] related", {"77", "294"})
    assert links == [("294", "related")]
    assert invalid == [] and not checked

    links, invalid, checked = kota._dedr_links("x — [-] absent", {"77", "294"})
    assert links == [] and invalid == [] and checked


def test_compound_marker_keeps_positive_dedr_member():
    links, _, checked = kota._dedr_links("aṛg gī- — [- + 1957] put out a fire", {"1957"})
    assert links == [("1957", "related")]
    assert checked


def test_dedr_subentry_suffix_resolves_to_dataset_etymon():
    links, invalid, _ = kota._dedr_links("kār — (1278c) black", {"1278"})
    assert links == [("1278", "direct")]
    assert invalid == []


def test_nonstandard_dataset_id_and_obvious_repeated_digit_resolve():
    valid = {"4896": "4896(a)", "4411": "4411"}
    assert kota._dedr_links("muk- — (4896) strain", valid)[0] == [("4896(a)", "direct")]
    assert kota._dedr_links("peydēr — (44111) youth", valid)[0] == [("4411", "direct")]


def test_head_parser_emits_printed_alternants_and_conjugational_stem():
    form, variants, morphology = kota._head_forms("ac/acl/acār")
    assert form == "ac"
    assert variants == ["acl", "acār"]
    assert not morphology

    form, variants, morphology = kota._head_forms("aḍmug-/aḍmūv- (aḍmuṛt-)")
    assert form == "aḍmug-"
    assert variants == ["aḍmūv-", "aḍmuṛt-"]
    assert "Conjugational" in morphology


def test_embedded_italic_dot_is_repaired_as_an_underdot():
    line = {
        "text": "Tuesday: angl.vārm",
        "top": 100.0,
        "chars": [
            {"text": char, "x0": index * 5.0, "x1": index * 5.0 + 5.0, "top": 100.0}
            for index, char in enumerate("Tuesday: anglvārm")
        ],
    }
    # Insert a displaced dot overlapping the l glyph.
    line["chars"].insert(13, {"text": ".", "x0": 61.0, "x1": 64.0, "top": 102.0})
    assert kota._attach_underdots(line, []) == "Tuesday: angḷvārm"


def test_split_no_dash_source_entry():
    entry = kota.Entry(13, 1, ["im female buffalo [DEDR 816]"])
    assert kota.split_entry(entry) == ("im", "female buffalo [DEDR 816]")


def test_generated_source_has_repaired_retroflexes_and_full_rows():
    source = Path(__file__).parents[1] / "data/other/forms/20260813-wolf-kota.csv"
    if not source.exists():
        return
    import csv

    with source.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) > 1500
    assert all(len(row) == kota.RICH_COLUMNS for row in rows)
    assert any(row[2] == "aḍ" for row in rows)
    assert any(row[1] == "d77" for row in rows)
    assert any(not row[1] for row in rows)
    assert all(row[7].startswith("wolf-kota[p. ") for row in rows)
