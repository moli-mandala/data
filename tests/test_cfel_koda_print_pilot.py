"""Focused checks for the seven-page CFEL Koda print pilot."""

import collections
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

from segments import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "data/other/forms/raw_data/cfel_koda_2022"
STEM = "20260925-cfel-koda-print-pilot"


def importer():
    spec = importlib.util.spec_from_file_location("cfel_koda_print", PACKAGE / "import_source.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bounded_audit_and_reproducible_csv():
    rows, audit = importer().build()
    installed = list(csv.reader((PACKAGE / f"{STEM}.csv").open(newline="")))
    assert rows == installed and len(rows) == 17 and len(audit) == 53
    assert collections.Counter(x["status"] for x in audit) == {"ingested": 17, "excluded": 36}
    assert collections.Counter(x["pdf_page"] for x in audit) == {
        10: 6, 25: 6, 76: 9, 120: 7, 180: 10, 260: 6, 345: 9
    }
    assert len({row[10] for row in rows}) == 17
    assert all(len(row) == 15 and row[0] == "Koda" and row[4] and row[7].startswith("pradhan-tripathi2022koda[") for row in rows)
    assert all(not row[i] for row in rows for i in (1, 5, 6, 8, 9, 11, 12, 13))
    assert {row[3] for row in rows} == {x["english_head"] for x in audit if x["status"] == "ingested"}


def test_excluded_uncertainties_and_exact_api_pairing():
    rows, audit = importer().build()
    by_head = {item["english_head"]: item for item in audit}
    assert "visarga" in by_head["Cow"]["reason"]
    assert "question-mark" in by_head["Two"]["reason"]
    assert "different Koda form" in by_head["Die"]["reason"]
    assert all(head not in {row[3] for row in rows} for head in ("Cow", "Two", "Die"))
    evidence = [json.loads(line) for line in (PACKAGE / "api-evidence.jsonl").read_text().splitlines()]
    assert len(evidence) == len({item["api_id"] for item in evidence}) == 17
    assert all(item["api_word"].strip() == by_head[item["query"]]["printed_koda"] for item in evidence)
    assert all(len(item["response_sha256"]) == 64 for item in evidence)


def test_sample_manifest_and_rights_boundary():
    manifest = json.loads((PACKAGE / "manifest.json").read_text())
    assert manifest["installed_rows"] == 17 and manifest["excluded_entries"] == 36
    for name, expected in manifest["assets"].items():
        assert hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest() == expected
    sample = [json.loads(line) for line in (PACKAGE / "visual-sample-20260925.jsonl").read_text().splitlines()]
    assert len(sample) == len({item["entry_key"] for item in sample}) == 20
    assert all(item["printed_page"] == item["pdf_page"] - 3 and item["material_error"] is False for item in sample)
    assert {item["english_head"] for item in sample if item["status"] == "excluded"} == {"Cow", "Two", "Die"}
    assert not list(PACKAGE.glob("*.pdf")) and not list(PACKAGE.glob("*.png"))
    assert "public-release decision" in (PACKAGE / "README.md").read_text()


def test_profile_and_scoped_source_registration():
    rows, _ = importer().build()
    profile = Tokenizer(str(ROOT / "conversion/cfel-koda-print.txt"))
    converted = [profile(row[2], column="IPA").replace(" ", "").replace("#", " ") for row in rows]
    assert len(converted) == 17 and all(value and "�" not in value for value in converted)
    assert converted[0] == "toraʔ" and converted[8] == "boi"
    yaml = (ROOT / f"data/other/forms/{STEM}.yaml").read_text()
    assert "append_order: 70" in yaml and "profile: cfel-koda-print" in yaml
    assert (ROOT / "cldf/sources.bib").read_text().count("@book{pradhan-tripathi2022koda,") == 1
