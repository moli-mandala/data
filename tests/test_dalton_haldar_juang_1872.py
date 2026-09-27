"""Source-stage checks for Haldar's compiled Juanga column in Dalton 1872."""

import csv
import importlib.util
import io
import json
import sys
from collections import Counter
from pathlib import Path

from segments.tokenizer import Tokenizer


DATA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATA))
PACKAGE = DATA / "data/other/forms/raw_data/dalton_haldar_juang_1872"
CSV = DATA / "data/other/forms/20260925-dalton-haldar-juang.csv"
PROFILE = DATA / "conversion/dalton-haldar-juang-1872.txt"
spec = importlib.util.spec_from_file_location("dalton_haldar_juang", PACKAGE / "import_source.py")
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def installed():
    with CSV.open(encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def test_entire_p236_table_and_reproducibility():
    rows, audit = source.build_p236()
    assert rows == installed()[:31]
    assert len(audit) == 42 and len(rows) == 31
    assert [a["row"] for a in audit] == list(range(1, 43))
    assert len({a["entry_key"] for a in audit}) == 42
    assert Counter(a["status"] for a in audit) == {"selected": 27, "selected_alternates": 2, "held": 10, "blank": 3}
    assert {a["printed_page"] for a in audit} == {236}
    assert {a["pdf_page"] for a in audit} == {251}
    assert all(len(a["control_columns_present"]) == 8 for a in audit)
    assert all(not a["selected_forms"] for a in audit if a["status"] in ("held", "blank"))
    assert [a["row"] for a in audit if a["status"] == "blank"] == [25, 26, 27]
    assert [r[2] for r in rows if r[3] == "brother"] == ["boka", "ká"]
    assert [r[2] for r in rows if r[3] == "dog"] == ["sétag", "sello"]
    assert not any(r[3] == "star" or r[3] == "house" for r in rows)


def test_print_review_controls_provenance_and_overlap():
    _, audit = source.build_p236()
    visual = [json.loads(line) for line in (PACKAGE / "visual-review-20260925.jsonl").read_text().splitlines()]
    assert len(visual) == len({v["entry_key"] for v in visual}) == 42
    assert {v["entry_key"] for v in visual} == {a["entry_key"] for a in audit}
    assert all(v["juanga_cell_reviewed"] and v["eight_control_column_presence_reviewed"] for v in visual)
    assert all(not v["accepted_form_material_error_remaining"] for v in visual)
    sample = json.loads((PACKAGE / "control-sample.json").read_text())
    assert len(sample["control_order"]) == 8
    assert len(sample["sampled_rows"][0]["controls"]) == 8
    assert len(sample["sampled_rows"][1]["controls"]) == 8
    assert sample["sampled_rows"][2]["controls_blank"] == ["Kuri/Muasi", "Talain/Mon"]
    manifest = json.loads((PACKAGE / "manifest.json").read_text())
    assert tuple(manifest[k] for k in ("printed_prompt_rows", "selected_juanga_prompt_rows", "held_juanga_prompt_rows", "blank_juanga_prompt_rows", "installed_forms")) == (42, 29, 10, 3, 31)
    assert manifest["append_order"] == 104
    assert "public domain" in manifest["rights"].lower()
    assert "compiled" in manifest["provenance"] and "not" in manifest["dependence"]
    assert len(manifest["scan_sha256"]) == 64
    overlap = json.loads((PACKAGE / "overlap-review.json").read_text())
    assert overlap["existing_Juang_source_rows_checked"] == 2049
    assert overlap["selected_forms_checked"] == 31
    assert overlap["normalized_exact_form_matches"] == 7
    assert overlap["samuells_1856_exact_matches"] == 0


def test_source_dialect_profile_and_scoped_parser():
    rows = installed()
    assert all(len(r) == 15 and r[0] == "ju" and r[14] == source.DIALECT for r in rows)
    assert len({r[10] for r in rows}) == 196
    assert all(r[7].startswith("dalton1872haldarjuang[p. ") for r in rows)
    assert sum("Source annotation: (s.)" in r[9] for r in rows) >= 7
    dialects = {r[0]: r for r in csv.reader((DATA / "cldf/dialects.csv").open())}
    dialect = dialects["haldar-1872-juanga"]
    assert dialect[1:5] == [source.DIALECT, "ju", "dalton1872haldarjuang:Juanga", "Juanga (Haldar compiled 1872)"]
    assert dialect[5:8] == ["", "", ""]
    bib = (DATA / "cldf/sources.bib").read_text()
    assert bib.count("@book{dalton1872haldarjuang,") == 1
    assert "20260925-dalton-haldar-juang.csv" in bib
    tokenizer = Tokenizer(str(PROFILE))
    assert tokenizer("juanglé", column="IPA").replace(" ", "") == "juangle"
    assert tokenizer("bhagwán", column="IPA").replace(" ", "") == "bʰagvan"
    assert tokenizer("ghorá", column="IPA").replace(" ", "") == "gʰora"
    import make_cldf

    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(CSV), errors, name=CSV.stem)
    assert not errors.getvalue()
    assert len(parsed) == stats["converted"] == 196


def test_complete_comparison_and_separate_list_inventory():
    rows, audit = source.build()
    assert rows == installed()
    assert len(audit) == 361 and len(rows) == 196
    assert Counter(a["status"] for a in audit) == {
        "selected": 186, "selected_alternates": 3, "held": 19, "blank": 153
    }
    assert {a["printed_page"] for a in audit} == set(range(235, 243))
    assert len({a["entry_key"] for a in audit}) == len(audit)
    assert all(a["selected_forms"] for a in audit if a["status"].startswith("selected"))
    assert all(not a["selected_forms"] for a in audit if a["status"] in {"held", "blank"})
    assert rows[:31] == source.build_p236()[0]
    assert all((PACKAGE / "images" / (f"page-{a['pdf_page']}.png" if a["pdf_page"] <= 255 else f"hi-{a['pdf_page']}.png")).exists() for a in audit)


def test_independent_full_extension_and_separate_list_review():
    comparison = json.loads((PACKAGE / "comparative-extension-independent-audit-2026092501.json").read_text())
    assert comparison["final_material_errors"] == 0
    assert comparison["reviewed_accepted_cells"] == 30
    assert [(r["printed_page"], r["row"], r["after"]) for r in comparison["corrections"]] == [(235, 15, "ainyá")]
    separate = json.loads((PACKAGE / "separate-list-sample-audit-2026092501.json").read_text())
    assert separate["prompt_rows"] == 134
    assert separate["sample_size"] == 20 and separate["material_errors"] == 0
    assert len(separate["sample"]) == 20
    assert all(row["result"] == "pass" for row in separate["sample"])
