import csv
from collections import Counter, defaultdict
from pathlib import Path

from unify_cldf import select_pdr_headwords


ROOT = Path(__file__).resolve().parents[1]


def form(form_id, parameter_id, language_id, value, source):
    return {
        "ID": form_id,
        "Parameter_ID": parameter_id,
        "Language_ID": language_id,
        "Form": value,
        "Source": source,
    }


def test_headword_precedence_and_non_reconstruction_tag():
    params = [
        {"ID": "d1", "Language_ID": "PDr", "Name": "*legacy"},
        {"ID": "d2", "Language_ID": "PDr", "Name": ""},
        {"ID": "d3", "Language_ID": "PDr", "Name": ""},
        {"ID": "d4", "Language_ID": "PDr", "Name": ""},
        {"ID": "d583A", "Language_ID": "PDr", "Name": ""},
        {"ID": "d4896(a)", "Language_ID": "PDr", "Name": ""},
        {"ID": "da1", "Language_ID": "PDr", "Name": ""},
        {"ID": "dbia1", "Language_ID": "PDr", "Name": ""},
    ]
    forms = defaultdict(list, {
        "d1": [
            form("legacy", "d1", "PDr", "*legacy", "krishnamurti"),
            form("m1", "d1", "PDr", "*merriam", "merriam2026dravidiandb[record 1]"),
        ],
        "d2": [
            form("subgroup", "d2", "PSTDr", "*subgroup", "merriam2026dravidiandb[record 2]"),
            form("m2", "d2", "PDr", "*merriam", "merriam2026dravidiandb[record 3]"),
            form("d2-reflex", "d2", "Tamil", "actual", "dedr"),
        ],
        "d3": [
            form("d3-reflex-1", "d3", "Kannada", "first", "dedr"),
            form("d3-reflex-2", "d3", "Tamil", "second", "dedr"),
        ],
        "d583A": [form("supplement", "d583A", "Tamil", "supplement", "dedr")],
        "d4896(a)": [form("subentry", "d4896(a)", "Tamil", "subentry", "dedr")],
        "da1": [form("addendum", "da1", "Tamil", "addendum", "dedr")],
        "dbia1": [form("dbia-form", "dbia1", "Tamil", "loan", "dbia")],
    })

    decisions = select_pdr_headwords(params, forms)

    assert [row["Name"] for row in params] == [
        "*legacy", "*merriam", "first", "", "supplement", "subentry", "", "",
    ]
    assert decisions["d1"]["Strategy"] == "krishnamurti-pfeiffer"
    assert decisions["d1"]["Source_Form_ID"] == ""
    assert decisions["d2"]["Strategy"] == "merriam-pdr"
    assert decisions["d2"]["Source_Form_ID"] == "m2"
    assert decisions["d3"]["Strategy"] == "dedr-reflex"
    assert decisions["d3"]["Tags"] == "not-reconstructed"
    assert decisions["d4"]["Strategy"] == "unresolved"
    assert decisions["d583A"]["Strategy"] == "dedr-reflex"
    assert decisions["d4896(a)"]["Strategy"] == "dedr-reflex"
    assert "da1" not in decisions
    assert "dbia1" not in decisions


def test_compiled_pdr_headword_audit_and_forms_follow_policy():
    audit_path = ROOT / "cldf/pdr-headword-audit.csv"
    forms_path = ROOT / "cldf/forms.csv"
    if not audit_path.exists() or not forms_path.exists():
        return

    with audit_path.open(encoding="utf-8", newline="") as handle:
        audit = list(csv.DictReader(handle))
    with forms_path.open(encoding="utf-8", newline="") as handle:
        compiled = {row["ID"]: row for row in csv.DictReader(handle)}

    assert len(audit) == 5562
    counts = Counter(row["Strategy"] for row in audit)
    assert counts == {
        "krishnamurti-pfeiffer": 817,
        "merriam-pdr": 1166,
        "dedr-reflex": 3575,
        "unresolved": 4,
    }
    assert {row["Parameter_ID"] for row in audit if row["Strategy"] == "unresolved"} == {
        "d82", "d92", "d2035", "d3091",
    }

    for decision in audit:
        entry = compiled[decision["Parameter_ID"]]
        assert entry["Form"] == decision["Headword"]
        if decision["Strategy"] == "dedr-reflex":
            assert "not-reconstructed" in entry["Tags"].split()
            assert not entry["Form"].startswith("*")
        else:
            assert "not-reconstructed" not in entry["Tags"].split()
        if decision["Source_Form_ID"]:
            assert decision["Source_Form_ID"] in compiled
