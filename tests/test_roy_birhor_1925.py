"""Historical pilot provenance and current Roy source installation checks."""

import collections
import csv
import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path

from segments import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "data/other/forms/raw_data/roy_birhor_1925"
STEM = "20260925-roy-birhor-p567-p568"


def importer():
    spec = importlib.util.spec_from_file_location("roy_birhor", PACKAGE / "import_source.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_entire_print_page_and_deterministic_csv():
    rows, audit = importer().build()
    installed = list(csv.reader((PACKAGE / "legacy-pilot.csv").open(newline="")))
    assert rows == installed
    assert len(audit) == 59 and len(rows) == 46
    assert collections.Counter(x["status"] for x in audit) == {"ingested": 46, "excluded": 13}
    assert [(x["column"], x["column_item"]) for x in audit if x["printed_page"] == 567] == (
        [("L", n) for n in range(1, 10)] + [("R", n) for n in range(1, 15)]
    )
    assert [(x["column"], x["column_item"]) for x in audit if x["printed_page"] == 568] == (
        [("L", n) for n in range(1, 21)] + [("R", n) for n in range(1, 17)]
    )
    assert len({row[10] for row in rows}) == 46
    assert all(len(row) == 15 and row[0] == "Birhor" and row[7].startswith("roy1925birhors[") for row in rows)
    assert all(not any(row[i] for i in (1, 4, 5, 6, 9, 11, 12, 13, 14)) for row in rows)
    assert {x["entry_key"].split(":")[-1] for x in audit if x["printed_page"] == 567 and x["status"] == "excluded"} == {
        "L01", "L04", "L05", "L06", "L07", "R08"
    }
    assert {x["entry_key"].split(":")[-1] for x in audit if x["printed_page"] == 568 and x["status"] == "excluded"} == {
        "L05", "L10", "L11", "L13", "L14", "L15", "R16"
    }


def test_parallel_pinnow_citations_and_distinct_senses():
    rows, _ = importer().build()
    by_key = {row[10].split("roy1925birhor:", 1)[-1]: row for row in rows}
    assert by_key["p567:R03"][3] == "Grand-mother"
    assert by_key["p567:R06"][2] == by_key["p567:R07"][2] == "ālāng"
    assert by_key["p567:R06"][3] != by_key["p567:R07"][3]
    assert by_key["p567:R14"][8] == "Roy comparison: H."
    overlap = [json.loads(line) for line in (PACKAGE / "overlap-review.jsonl").read_text().splitlines()]
    assert len(overlap) == 59
    assert sum(bool(x["compiled_birhor_matches"]) for x in overlap) == 19
    assert "underlying Pinnow source dependence not established" in overlap[1]["decision"]
    assert any(x["gloss"] == "elder sister" for x in overlap[11]["compiled_birhor_matches"])


def test_profile_scoped_parser_and_metadata():
    sys.path.insert(0, str(ROOT))
    import make_cldf

    profile = Tokenizer(str(ROOT / "conversion/roy-birhor.txt"))
    rows, _ = importer().build()
    assert all("�" not in profile(row[2], column="IPA") for row in rows)
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(
        str(ROOT / f"data/other/forms/{STEM}.csv"), errors, name=STEM
    )
    assert len(parsed) == stats["converted"] == 1019 and not errors.getvalue()
    assert parsed[0].form == "ā’"
    assert {"dūrbal", "ga’ui", "urū"} <= {row.form for row in parsed}
    yaml = (ROOT / f"data/other/forms/{STEM}.yaml").read_text()
    assert "append_order: 74" in yaml and "profile: roy-birhor" in yaml
    assert (ROOT / "cldf/sources.bib").read_text().count("@book{roy1925birhors,") == 1


def test_manifest_visual_sample_and_public_domain_witness():
    manifest = json.loads((PACKAGE / "manifest.json").read_text())
    assert manifest["audited_heads"] == 59 and manifest["installed_rows"] == 46
    assert manifest["excluded_heads"] == 13 and manifest["pdf_pages_selected"] == [655, 656]
    for name, expected in manifest["assets"].items():
        assert hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest() == expected
    sample = [json.loads(line) for line in (PACKAGE / "visual-sample-20260925.jsonl").read_text().splitlines()]
    assert len(sample) == len({item["entry_key"] for item in sample}) == 20
    assert all(not item["material_error"] for item in sample)
    assert all((PACKAGE / f"printed-p{page}.png").stat().st_size > 100_000 for page in (567, 568))
    assert "pre-1931 publication" in manifest["rights"]
