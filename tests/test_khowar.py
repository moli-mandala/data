import csv
import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "data/other/forms/raw_data/khowar.py"
SPEC = importlib.util.spec_from_file_location("khowar_extractor", SCRIPT)
khowar = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = khowar
SPEC.loader.exec_module(khowar)


def test_turner_reference_pattern_handles_dictionary_notation():
    text = "[< Skt. (M:1973) (T9170) (T11334)] and (< Skt. T9360 √BHAJ2)"
    assert khowar.TURNER_REFERENCE.findall(text) == ["9170", "11334", "9360"]


def test_headword_preserves_spaces_and_homonym_number():
    line = {
        "x0": 54.0,
        "text": "bar1 (n) ‘load’",
        "chars": [
            {"fontname": "ABC+Arial-BoldMT", "text": char}
            for char in "bar1"
        ]
        + [{"fontname": "ABC+Arial-ItalicMT", "text": "("}],
    }
    assert khowar._headword(line, 54) == "bar¹"

    line["text"] = "azgará korík (vtr) ‘to clear the throat’"
    line["chars"] = [
        {"fontname": "ABC+Arial-BoldMT", "text": char}
        for char in "azgarákorík"
    ] + [{"fontname": "ABC+Arial-ItalicMT", "text": "("}]
    assert khowar._headword(line, 54) == "azgará korík"


def test_headword_repairs_displaced_retroflex_dot_and_false_spaces():
    assert khowar._normalize_headword("acḥú") == "ac̣hú"
    assert khowar._normalize_headword("cọ cḥ ík") == "c̣oc̣hík"
    assert khowar._normalize_headword("bacạ́ c ̣") == "bac̣ác̣"
    assert khowar._normalize_headword("mucḳ") == "muc̣"
    assert khowar._normalize_headword("c̣enȷ̌ị́k") == "c̣enǰík"
    assert khowar._normalize_headword("ɫup korIk") == "ɫup korík"
    assert khowar._normalize_headword("pracγ̣ár") == "prac̣yár"


def test_gloss_accepts_source_typographical_quote_errors():
    assert khowar._gloss("x (n) ‘large falcon‘ {MAK}", "x") == "large falcon"
    assert khowar._gloss("x (vt) 'to lose something' {ZHD}", "x") == "to lose something"


def test_finish_links_every_valid_cdial_reference():
    entry = khowar.Entry("baɫéik", 23, 10)
    entry.lines = [
        "baɫéik (vtr) ‘to overcome by force’ [< Skt. (T9170) (T11334)]"
    ]
    rows = []
    khowar._finish(entry, rows, {"9170", "11334"})
    assert [row[1] for row in rows] == ["9170", "11334"]
    assert all(row[0] == "Kho" for row in rows)
    assert all(row[3] == "to overcome by force" for row in rows)
    assert all(row[6].startswith("tr verb;") for row in rows)
    assert all(len(row) == khowar.RICH_COLUMNS for row in rows)


def test_direct_donor_requires_a_form_and_unambiguous_language():
    assert khowar._direct_donor("x [< Ur. ghaṛī ‘watch, clock’]") == (
        "H", "ghaṛī", "watch, clock"
    )
    assert khowar._direct_donor("x [< Eng. ‘beam’]") == ("Eng", "beam", "")
    assert khowar._direct_donor("x [< Ar. ‘devil’]") is None
    assert khowar._direct_donor("x [< Ar., Prs., Ur. hukm ‘order’]") is None


def test_rich_rows_create_regional_variant_donor_and_derivation():
    entry = khowar.Entry("aldú", 20, 7, sequence=1)
    entry.lines = [
        "aldú /Other pronunc: aʋdú (in Laspur, IF)/ (adj) ‘taken’ "
        "[< Ur. aldū ‘taken’] {IF}"
    ]
    entry.bold_spans = [("aldú", False), ("aʋdú", True)]
    rows, dialects, audit = khowar.build_rows([entry], set())

    assert {item["Role"] for item in audit} == {"head", "variant"}
    assert "Kho-Bashir-reg-laspur" in dialects
    assert "Kho-Bashir-src-if" in dialects
    assert any(row[0] == "H" and row[14] == "source-form" for row in rows)
    assert any(row[11] and row[13] for row in rows)  # variant + derivation-parent keys
    assert any(row[12] for row in rows if row[2] == "aldú")  # donor key


def test_generated_khowar_source_has_linked_and_unlinked_entries():
    source = Path(__file__).parents[1] / "data/other/forms/20260725-bashir-khowar.csv"
    if not source.exists():
        return
    with source.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) > 3000
    assert sum(bool(row[1]) for row in rows) > 200
    assert any(not row[1] for row in rows)
    assert all(row[7] == "bashir2023" and len(row) == khowar.RICH_COLUMNS for row in rows)
    assert any(row[0].startswith("Kho-Bashir-") for row in rows)
    assert any(row[11] for row in rows)  # alternate pronunciations
    assert any(row[13] for row in rows)  # derivational parents
    assert not any("unresolved Turner" in row[6] for row in rows)
