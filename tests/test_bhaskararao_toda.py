import csv
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "data/other/forms/raw_data/bhaskararao_toda.py"
SPEC = importlib.util.spec_from_file_location("bhaskararao_toda_extractor", SCRIPT)
toda = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = toda
SPEC.loader.exec_module(toda)


def test_head_parser_separates_past_stem_and_homophone_number():
    form, variants, sense = toda._head_forms("ark- (arky-)")
    assert form == "ark-"
    assert variants == ["arky-"]
    assert sense == ""

    form, variants, sense = toda._head_forms("ark- (1)")
    assert form == "ark-" and variants == [] and sense == "1"


def test_dedr_parser_retains_multi_entry_citations_and_repairs_leading_zero():
    links, invalid = toda._dedr_links(
        "DEDR 212, 0221, TGT 120", {"212": "212", "221": "221"}
    )
    assert links == ["212", "221"]
    assert invalid == []


def test_generated_toda_dictionary_is_complete_and_unicode_preserving():
    source = ROOT / "data/other/forms/20260813-bhaskararao-toda.csv"
    audit_path = ROOT / "data/other/forms/raw_data/20260813-bhaskararao-toda-audit.csv"
    if not source.exists() or not audit_path.exists():
        return

    with source.open(encoding="utf-8", newline="") as stream:
        forms = list(csv.reader(stream))
    with audit_path.open(encoding="utf-8", newline="") as stream:
        audit = list(csv.DictReader(stream))

    assert len(audit) == 7560  # total stated in the dictionary preface
    assert len(forms) > len(audit)
    assert all(len(row) == toda.RICH_COLUMNS for row in forms)
    assert all(row[7].startswith("bhaskararao-toda2025[p. ") for row in forms)
    assert len({row[10] for row in forms}) == len(forms)
    assert all(row["Definition"] for row in audit)
    assert not any(row["Unresolved_DEDR_IDs"] for row in audit)

    by_form = {}
    for row in audit:
        by_form.setdefault(row["Form"], []).append(row)
    assert "aṛ-koṟ" in by_form
    assert by_form["ačok"][0]["DEDR_IDs"] == "474|2876"

    ark = [row for row in by_form["ark-"] if row["Printed_Page"] == "4"]
    assert [row["Definition"] for row in ark] == [
        "vt. (1) to chip, cut square (end of plank or post). DEDR 212, TGT 120",
        "vt. (2) to file, rub (to be checked). PB 8",
    ]
