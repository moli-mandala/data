import csv
import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "data/other/forms/raw_data/gandhari_org.py"
SPEC = importlib.util.spec_from_file_location("gandhari_org_importer", SCRIPT)
gandhari = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = gandhari
SPEC.loader.exec_module(gandhari)


def test_extracts_only_sanskrit_etyma():
    raw = "(Skt. <i>anvāgamayati</i>, <i>anvāgameti</i>, P <i>anvāgameti</i>)"
    assert gandhari.extract_sanskrit_etyma(raw) == ["anvāgamayati", "anvāgameti"]
    compound = "(Skt. <i>a‐</i>, P <i>a‐</i> + Skt. <i>hāni</i>, P <i>hāni</i>)"
    assert gandhari.extract_sanskrit_etyma(compound) == ["a‐", "hāni"]


def test_cdial_matching_discards_only_accents_and_not_vowel_length(tmp_path):
    params = tmp_path / "params.csv"
    params.write_text(
        '2,áṁśa,,,\n3,aṃśa,,,\n4,aṃśā,,,\n5,"foo-, bā́r",,,\n', encoding="utf-8"
    )
    index = gandhari.cdial_index(params)
    assert gandhari.match_cdial(["bār-"], index) == ("5", ["bā́r"], "matched")
    _, candidates, status = gandhari.match_cdial(["aṃśa"], index)
    assert status == "ambiguous"
    assert candidates == ["2:áṁśa", "3:aṃśa"]
    assert gandhari.match_cdial(["aṃśā"], index) == ("4", ["aṃśā"], "matched")


def test_sample_entry_becomes_rich_manual_row():
    entry = {
        "_id": "6971", "_lem": "aṃvagamedi", "_lemNative": "&#x10A00;",
        "_phonetic": "ʔəṽvaːjəmeːði", "_def": "waits until after.", "_pos": "v.",
        "_morphology": '<span class="morph">pres.</span>',
        "_citations": '<span class="attestedform">anmagamehi</span>.',
        "_etymology": "(Skt. <i>anvāgamayati</i>, P <i>anvāgameti</i>)",
        "_etymologyDisp": "(Skt. anvāgamayati, P anvāgameti)",
    }
    row = gandhari.source_row(entry, "123", ["anvāgamayati"])
    assert len(row) == 15
    assert row[:6] == ["Dhp", "123", "aṃvagamedi", "waits until after.", "𐨀", "ʔəṽvaːjəmeːði"]
    assert row[7] == "gandhari"
    assert row[10] == "gandhari:6971"
    assert "a%E1%B9%83vagamedi" in row[6]


def test_generated_snapshot_is_well_formed_when_present():
    source = Path(__file__).parents[1] / "data/other/forms/20260805-gandhari-org.csv"
    if not source.exists():
        return
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 1512
    assert {len(row) for row in rows} == {15}
    assert {row[0] for row in rows} == {"Dhp"}
    assert {row[7] for row in rows} == {"gandhari"}
    assert all(row[1] and row[2] and row[10].startswith("gandhari:") for row in rows)
    assert len({row[10] for row in rows}) == len(rows)
    assert any(row[2] == "agachadi" and row[1] == "1044" for row in rows)
