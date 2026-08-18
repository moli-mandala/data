import csv
import importlib.util
import sys
from collections import Counter
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
    assert row[6] == ""
    assert row[7] == "gandhari[entry 6971, lemma aṃvagamedi]"
    assert row[10] == "gandhari:6971"
    assert row[14] == "verb"


def test_pos_and_pronominal_subtypes_become_canonical_tags():
    cases = [
        ({"_id": "1", "_pos": "m."}, "noun m"),
        ({"_id": "2", "_pos": "f."}, "noun f"),
        ({"_id": "3", "_pos": "n."}, "noun n"),
        ({"_id": "4", "_pos": "adj."}, "adj"),
        ({"_id": "5", "_pos": "ind."}, "indecl"),
        ({"_id": "6", "_pos": "pron.", "_subpos": "interr."}, "pron interr"),
        ({"_id": "7", "_pos": "pron.", "_subpos": "pers."}, "pron personal"),
        ({"_id": "8", "_pos": "pron.", "_subpos": "rel."}, "pron relative"),
        ({"_id": "9", "_pos": "pron.", "_subpos": "dem."}, "pron demonstrative"),
        ({"_id": "10", "_pos": "ord."}, "num ord"),
        ({"_id": "11", "_pos": "adp."}, "postp"),
    ]
    for entry, expected in cases:
        assert gandhari.grammatical_tags(entry) == expected


def test_unknown_grammar_label_cannot_enter_installed_rows():
    entry = {"_id": "99", "_pos": "mystery."}
    try:
        gandhari.grammatical_tags(entry)
    except ValueError as error:
        assert "mystery." in str(error)
    else:
        raise AssertionError("unknown grammatical label was silently accepted")


def test_source_citation_uses_every_available_locator_without_notes_dump():
    entry = {
        "_id": "42", "_lem": "dharma", "_hom": "2", "_volume": "1",
        "_page": "17", "_column": "b",
    }
    assert gandhari.source_citation(entry) == (
        "gandhari[vol. 1, p. 17, col. b, entry 42, homograph 2, lemma dharma]"
    )


def test_generated_snapshot_is_well_formed_when_present():
    source = Path(__file__).parents[1] / "data/other/forms/20260805-gandhari-org.csv"
    if not source.exists():
        return
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 1512
    assert {len(row) for row in rows} == {15}
    assert {row[0] for row in rows} == {"Dhp"}
    assert {row[7].split("[", 1)[0] for row in rows} == {"gandhari"}
    assert {row[6] for row in rows} == {""}
    assert all(row[7].startswith("gandhari[entry ") and row[7].endswith("]") for row in rows)
    assert all(row[14] for row in rows)
    assert all(row[1] and row[2] and row[10].startswith("gandhari:") for row in rows)
    assert len({row[10] for row in rows}) == len(rows)
    assert any(row[2] == "agachadi" and row[1] == "1044" for row in rows)
    assert Counter(row[14] for row in rows) == {
        "noun m": 723,
        "adj": 408,
        "verb": 176,
        "noun f": 146,
        "adv": 36,
        "indecl": 15,
        "pron interr": 3,
        "num": 2,
        "pron personal": 1,
        "pron relative": 1,
        "noun n": 1,
    }


def test_checked_in_audit_accounts_for_every_cached_source_record():
    audit_path = (
        Path(__file__).parents[1]
        / "data/other/forms/raw_data/20260805-gandhari-org-audit.csv"
    )
    with audit_path.open(encoding="utf-8", newline="") as handle:
        audit = list(csv.DictReader(handle))
    assert len(audit) == 5807
    assert Counter(row["Status"] for row in audit) == {
        "unmatched": 3923,
        "matched": 1512,
        "ambiguous": 371,
        "missing": 1,
    }
    assert len({row["Entry_Key"] for row in audit}) == len(audit)
    assert all(row["Source_Citation"].startswith("gandhari[") for row in audit)
    matched = [row for row in audit if row["Status"] == "matched"]
    assert all(row["Tags"] and not row["Exclusion_Reason"] for row in matched)
    assert sum(not row["Morphology_Raw"] for row in matched) == 2
    assert sum(not row["Attestations_Raw"] for row in matched) == 2


def test_seeded_manual_audit_has_no_material_errors():
    raw_data = Path(__file__).parents[1] / "data/other/forms/raw_data"
    with (raw_data / "20260805-gandhari-org-sample.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        sample = list(csv.DictReader(handle))
    assert len(sample) == 20
    assert {row["Seed"] for row in sample} == {"2002"}
    assert {row["Final_Result"] for row in sample} == {"PASS"}
    assert {row["Notes_Length"] for row in sample} == {"0"}
