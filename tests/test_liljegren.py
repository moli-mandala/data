import csv
import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "data/other/forms/raw_data/liljegren.py"
SPEC = importlib.util.spec_from_file_location("liljegren_extractor", SCRIPT)
liljegren = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = liljegren
SPEC.loader.exec_module(liljegren)


def test_parts_of_speech_and_subclasses_become_structured_tags():
    assert liljegren.grammatical_tags("n.fem:pn") == ["noun", "f", "proper-noun"]
    assert liljegren.grammatical_tags("v.tr:cjt.ninc") == [
        "verb", "tr", "compound", "conjunct-verb", "non-incorporating"
    ]
    assert liljegren.grammatical_tags("pron:ind") == ["pron", "indef", "interr"]
    assert liljegren.grammatical_tags("adv.tm:dem") == [
        "adv", "temporal", "demonstrative"
    ]


def test_palula_nominal_and_verbal_classes_are_filterable():
    assert liljegren.grammatical_tags(
        "n.masc", "a-decl/aan-decl (Obl): -á"
    ) == ["noun", "m", "Palula-noun-class-a", "Palula-noun-class-aan"]
    assert liljegren.grammatical_tags(
        "v.intr", "T/L:cons (Prs): form"
    ) == ["verb", "intr", "Palula-verb-class-L-consonant", "Palula-verb-class-T"]
    assert liljegren.grammatical_tags("v.tr", "Suppl") == [
        "verb", "tr", "Palula-verb-class-suppletive"
    ]


def test_turner_number_parsing_handles_source_variants():
    assert liljegren.turner_parameter("átyēti 'enters' (T: 227)") == "227"
    assert liljegren.turner_parameter("ǰána 'person' (T5098)") == "5098"
    assert liljegren.turner_parameter("putrá- 'son' (T. 8265)") == "8265"
    assert liljegren.turner_parameter("*cucci- 'breast' (4855)") == "4855"
    assert liljegren.turner_parameter("no proposed etymology") == ""
    assert liljegren.turner_parameters("*root (T: 3153, 14343)") == ["3153", "14343"]


def test_variant_qualifier_applies_to_every_form_in_group():
    assert liljegren.split_variants(
        "-áand, -éend (With a closed class of (motion) verbs) ; -éen (Biori)"
    ) == [
        ("-áand", "With a closed class of (motion) verbs"),
        ("-éend", "With a closed class of (motion) verbs"),
        ("-éen", "Biori"),
    ]


def test_origin_languages_become_loan_tags():
    assert liljegren.loan_tags("Urdu (Persian/Arabic): aaraam") == [
        "loanword", "loan:Arabic", "loan:Persian", "loan:Urdu"
    ]
    assert liljegren.loan_tags("") == []


def test_installed_dictionary_contains_unetymologised_rows_and_tags():
    source = Path(__file__).parents[1] / "data/other/forms/20220913-palula.csv"
    with source.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) > 2_700
    assert sum(not row[1] for row in rows) > 2_000
    assert any(row[2] == "aabaád" and not row[1] and "noun" in row[14] for row in rows)
    assert any("Palula-noun-class-a" in row[14] for row in rows)
    assert any("Palula-verb-class-T" in row[14] for row in rows)
    assert {row[1] for row in rows if row[2] == "kháaču"} >= {"3153", "14343"}
    assert all(len(row) == 15 for row in rows)
