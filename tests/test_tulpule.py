import csv
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "data/other/forms/raw_data/tulpule.py"
SPEC = importlib.util.spec_from_file_location("tulpule_importer", SCRIPT)
tulpule = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = tulpule
SPEC.loader.exec_module(tulpule)


SAMPLE = """
<div class="px-4"><hw><b>गाणेंवाणें</b> <b>gāṇēṃvāṇēm</b></hw>
n. [Sk. gāyana-varṇana] <d>गायन आणि वर्णन;</d> singing and praise.
<d>ठाकिती तें गाणांवाणां</d> Jñā. 18.1517</div>
<div class="px-4"><hw><b>गाणा</b> <b>gāṇā</b></hw>
m. [Sk. gāyana/cf. gāṇa] <d>गायक;</d> a singer. <d>गाणे रायातें</d> LU. 173</div>
<div class="px-4"><head><d>अ</d></head><hw><b>अ</b> <b>a</b></hw> prefix [Sk.] <b>1</b>
(<d>अभाववाचक</d>); (a negative). <d>अलणवाह</d> SI. 1;
<b>2</b> (<d>आधिक्यवाचक</d>); (an excessive.) <d>अप्राशे</d> SI. 2</div>
<div><hw><b>कुटिल</b> <b>kuṭila</b>, <b>कुटिळ</b> <b>kuṭiḷa</b></hw>
[Sk. kuṭila] <b>1</b> adj. <d>हीन;</d> evil. <d>example</d> SV. 59;
<b>2</b> n. <d>पाप;</d> evil; sin. <d>example</d> JñāGā. 292</div>
"""


def test_parses_headword_grammar_etymology_and_english_definitions():
    rows = tulpule.parse_page(200, SAMPLE)
    assert rows[0].forms == (("गाणेंवाणें", "gāṇēṁvāṇēṁ"),)
    assert rows[0].grammar == "n."
    assert rows[0].tags == "n noun"
    assert rows[0].etymology == "Sk. gāyana-varṇana"
    assert rows[0].gloss == "singing and praise."
    assert rows[2].tags == "prefix"
    assert rows[2].gloss == "a negative.; an excessive."
    assert rows[3].forms == (("कुटिल", "kuṭila"), ("कुटिळ", "kuṭiḷa"))
    assert rows[3].tags == "n adj noun"
    assert rows[3].gloss == "evil.; evil; sin."


def test_primary_sanskrit_match_is_strict_and_cf_is_not_promoted():
    index = {
        tulpule.normalize_match("gāyana"): [("4136", "gāyana")],
        tulpule.normalize_match("gāṇa"): [("999", "gāṇa")],
        tulpule.normalize_match("gāyana-varṇana"): [("x", "gāyana-varṇana")],
    }
    compound, simple, _ = tulpule.parse_page(200, SAMPLE)[:3]
    assert tulpule.match_etymon(simple, index) == ("4136", ["gāyana"], "matched")
    assert tulpule.match_etymon(compound, index) == ("x", ["gāyana-varṇana"], "matched")


def test_additional_source_grammar_labels_are_canonicalized():
    assert tulpule.grammatical_tags("postpos. of dat.") == "postp dat"
    assert tulpule.grammatical_tags("ind. interrog.") == "indecl interr"
    assert tulpule.grammatical_tags("postpost.") == "postp"


def test_ambiguous_jambu_heads_are_not_ingested():
    entry = tulpule.parse_page(200, SAMPLE)[1]
    key = tulpule.normalize_match("gāyana")
    assert tulpule.match_etymon(entry, {key: [("1", "gāyana"), ("2", "gā́yana")]})[2] == "ambiguous"


def test_build_emits_rich_source_keyed_rows(tmp_path):
    entry = tulpule.parse_page(200, SAMPLE)[1]
    output, audit = tmp_path / "forms.csv", tmp_path / "audit.csv"
    index = {tulpule.normalize_match("gāyana"): [("4136", "gāyana")]}
    counts = tulpule.build([entry], output, audit, index=index)
    row = next(csv.reader(output.open(encoding="utf-8")))
    assert counts == {"matched": 1}
    assert len(row) == 15
    assert row[:5] == ["OM", "4136", "gāṇā", "a singer.", "गाणा"]
    assert row[7] == "tulpule1999[p. 200, entry 2]"
    assert row[9] == "Sk. gāyana/cf. gāṇa"
    assert row[10] == "tulpule:p200:e2:v1"
    assert row[14] == "m noun"


def test_generated_snapshot_is_source_tagged_and_linked_when_present():
    source = ROOT / "data/other/forms/20260810-tulpule-old-marathi.csv"
    if not source.exists():
        return
    rows = list(csv.reader(source.open(encoding="utf-8")))
    assert rows
    assert {len(row) for row in rows} == {15}
    assert {row[0] for row in rows} == {"OM"}
    assert all(row[1] and row[2] and row[3] for row in rows)
    assert all(row[7].startswith("tulpule1999[p. ") for row in rows)
    assert all(row[10].startswith("tulpule:p") for row in rows)
    assert len({row[10] for row in rows}) == len(rows)
