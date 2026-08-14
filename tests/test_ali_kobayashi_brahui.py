import csv
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "data/other/forms/raw_data/ali_kobayashi_brahui.py"
SPEC = importlib.util.spec_from_file_location("ali_kobayashi_brahui_extractor", SCRIPT)
brahui = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = brahui
SPEC.loader.exec_module(brahui)


def test_loan_code_parser_handles_single_and_compound_labels():
    assert brahui.split_loan_codes("n. human [F]") == ("n. human", ["F"])
    assert brahui.split_loan_codes("n. loan [AF]") == ("n. loan", ["A", "F"])
    assert brahui.split_loan_codes("n. native") == ("n. native", [])


def test_line_joining_repairs_prose_but_preserves_analysed_forms():
    assert brahui._join_parts(["a. pros-", "perous [F]"]) == "a. prosperous [F]"
    assert brahui._join_parts(["vi. (=tūl-", "ing)"]) == "vi. (=tūl-ing)"


def test_source_categories_map_to_canonical_grammatical_tags():
    assert brahui.grammatical_tags("vt. to fulfil") == ["verb", "tr"]
    assert brahui.grammatical_tags("int. adv. why") == ["interr", "adv"]
    assert brahui.grammatical_tags("COP.NEG.PRS.3SG (ann-ing)") == [
        "verb", "copula", "pres", "neg", "3sg",
    ]


def test_generated_brahui_glossary_is_complete_and_traceable():
    source = ROOT / "data/other/forms/20260813-ali-kobayashi-brahui.csv"
    audit_path = ROOT / "data/other/forms/raw_data/20260813-ali-kobayashi-brahui-audit.csv"
    if not source.exists() or not audit_path.exists():
        return

    with source.open(encoding="utf-8", newline="") as stream:
        forms = list(csv.reader(stream))
    with audit_path.open(encoding="utf-8", newline="") as stream:
        audit = list(csv.DictReader(stream))

    assert len(forms) == len(audit) == 3483
    assert all(len(row) == brahui.RICH_COLUMNS for row in forms)
    assert all(row[7].startswith("ali-kobayashi2024[p. ") for row in forms)
    assert len({row[10] for row in forms}) == len(forms)
    assert all(row["Definition"] for row in audit)
    assert all(row["Tags"] for row in audit)
    assert {row["Printed_Page"] for row in audit} == {str(page) for page in range(687, 734)}

    by_key = {row["Entry_Key"]: row for row in audit}
    assert by_key["ali-kobayashi2024:p687:e3"]["Form"] == "aḍḍ"
    assert by_key["ali-kobayashi2024:p690:e53"]["Form"] == "baxā"
    assert any(
        tag.startswith("dialect:Brahui:")
        for tag in by_key["ali-kobayashi2024:p690:e53"]["Tags"].split()
    )
    assert sum(bool(row["Loan_Codes"]) for row in audit) == 1203
    assert sum("loanword" in row["Tags"].split() for row in audit) == 1203
    assert sum(any(tag.startswith("dialect:") for tag in row["Tags"].split()) for row in audit) == 3
