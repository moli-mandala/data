import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from infer_extensions import APPLY_TIERS, OUT_COLS, OUT_CSV, detect, skeleton, strip_ending


def test_generic_derivative_sections_are_not_base_candidates_or_witnesses(tmp_path, monkeypatch):
    import infer_extensions as inference

    params = tmp_path / "params.csv"
    source = tmp_path / "cdial.csv"
    with params.open("w", newline="") as stream:
        csv.writer(stream).writerow(["1", "*kicca"])
    def row(form, section):
        return ["H", "1", form, "mud", "", "", "", "", section]
    with source.open("w", newline="") as stream:
        csv.writer(stream).writerows([
            row("kīc", "1:2"),
            row("kicṛā", "2:Deriv"),
            row("kīc", "3:unclassified continuation"),
            row("kicṛā", "4:ext. -<i>ḍ</i>-"),
            row("kicṛā", "5:Deriv. with -<i>aka</i>-"),
            row("kīc", ""),
        ])
    monkeypatch.setattr(inference, "PARAMS_CSV", params)
    monkeypatch.setattr(inference, "CDIAL_CSV", source)
    entries = inference.load_entries()
    assert [home for _, home, _ in entries["1"][1]] == [
        ("form", 2), ("other",), ("base",), ("ext", "-ḍ-"), ("other",), ("base",)
    ]
    # The generic derivative has exactly the shape detect() would otherwise infer.
    # It must not leak through the end-to-end candidate selection as a base row.
    candidates, _ = inference.run(entries, evaluate=False)
    assert candidates == []


def test_skeleton_collapses_geminates_but_not_repeated_classes():
    assert skeleton("kakka") == ["K", "K"]
    assert skeleton("kāka") == ["K", "K"]
    assert skeleton("bhāṭhelɔ") == ["P", "T", "L"]  # aspiration digraphs are one consonant
    assert skeleton("kicṛā") == ["K", "C", "T"]
    assert skeleton("*ḍhalati") == ["T", "L", "t"]


def test_citation_endings_are_stripped():
    assert strip_ending("ḍhalakṇā") == "ḍhalak"
    assert strip_ending("toloiki") == "tol"
    assert strip_ending("nu") == "nu"  # never strip a whole word


def test_detect_needs_an_unextended_witness_and_a_free_class():
    # H. kicṛā beside H. kīc under *kicca-: a -ḍ- extension with a same-language witness
    assert detect("kicṛā", [("H", "kīc")], "*kicca", "H") == ("T", "H", "kīc", "same-language")
    # the witness may come from another language, at a lower tier
    assert detect("kicṛā", [("P", "kīc")], "*kicca", "H")[3] == "other-language"
    # only the OIA head itself matching is the weakest tier
    assert detect("kāṭak", [], "kāṭa", "X") == ("K", "Indo-Aryan", "kāṭa", "etymon")
    # the etymon already has the class after its first consonant: cluster residue, not extension
    assert detect("karāṛⁱ", [("G", "karār")], "káḍāra", "G") is None
    # no witness with the shorter skeleton, and the head does not match either: nothing
    assert detect("kicṛā", [("H", "kīcaṛ")], "*kiccama", "H") is None
    # compounds and forms whose last consonant is not in the extension set are ignored
    assert detect("kīc pāth", [("H", "kīc")], "*kicca", "H") is None
    assert detect("kīcan", [("H", "kīc")], "*kicca", "H") is None


def test_generated_table_is_applied_tier_only():
    with open(OUT_CSV, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows and list(rows[0].keys()) == OUT_COLS
    assert {r["Tier"] for r in rows} <= set(APPLY_TIERS)
    assert all(r["Morpheme"] and r["Row"].isdigit() for r in rows)
