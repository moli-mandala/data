"""Focused Sikalgari source checks; no full data or browser database build."""

import csv
import importlib.util
import io
import json
import random
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

from segments.tokenizer import Tokenizer


DATA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATA))
PACKAGE = DATA / "data/other/forms/raw_data/grierson_sikalgari_1922"
CSV = DATA / "data/other/forms/20260925-grierson-sikalgari.csv"
PROFILE = DATA / "conversion/grierson-sikalgari-1922.txt"
spec = importlib.util.spec_from_file_location("grierson_sikalgari", PACKAGE / "import_source.py")
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def installed():
    return list(csv.reader(CSV.open(encoding="utf-8", newline="")))


def audited():
    return [json.loads(line) for line in (PACKAGE / "audit.jsonl").read_text().splitlines()]


def test_exact_scope_counts_and_regeneration():
    rows, audit = source.generate()
    assert rows == installed() and audit == audited()
    assert len(rows) == len(audit) == 82
    assert {a["prompt"] for a in audit} == set(source.PROMPTS)
    assert all(a["status"] == "ingested" and a["language_id"] == "Sik" for a in audit)
    assert len({r[10] for r in rows}) == 82
    assert [r[10] for r in rows] == [a["entry_key"] for a in audit]
    assert all((PACKAGE / a["ocr_comparison"]).exists() for a in audit)


def test_page_breaks_and_representative_printed_readings():
    by_prompt = {a["prompt"]: a for a in audited()}
    for prompt, printed, pdf in [(1, 181, 193), (13, 181, 193), (32, 185, 197), (52, 185, 197), (53, 189, 201), (79, 189, 201), (80, 193, 205), (100, 193, 205)]:
        assert (by_prompt[prompt]["printed_page"], by_prompt[prompt]["pdf_page"]) == (printed, pdf)
    assert by_prompt[1]["raw_cell"] == "Ēk"
    assert by_prompt[52]["raw_cell"] == "Bāykō"
    assert by_prompt[53]["raw_cell"] == "Ramban"
    assert by_prompt[79]["raw_cell"] == "Bukhal"
    assert by_prompt[80]["raw_cell"] == "Ākhtal"
    assert by_prompt[100]["raw_cell"] == "Ayyāyō"


def test_canonical_dialect_reference_and_unlinked_forms():
    dialects = {r[0]: r for r in csv.reader((DATA / "cldf/dialects.csv").open())}
    d = dialects["sik_belgaum"]
    assert d[1] == source.DIALECT and d[2] == "Sik" and d[5:8] == ["", "", ""]
    assert "Sampgaon taluka" in d[9]
    languages = {r[0]: r for r in csv.reader((DATA / "cldf/languages.csv").open())}
    assert languages["Sik"][2] == "guja1252" and languages["Sik"][5] == "Gujaratic"
    for row in installed():
        assert len(row) == 15 and row[0] == "Sik" and row[14] == source.DIALECT
        assert row[7].startswith("grierson1922lsi11[p. ") and ", item " in row[7]
        assert row[2] and "�" not in row[2]
        assert row[1] == row[4] == row[5] == row[8] == row[9] == row[11] == row[12] == row[13] == ""
    bib = (DATA / "cldf/sources.bib").read_text()
    assert bib.count("@book{grierson1922lsi11,") == 1
    assert "20260925-grierson-sikalgari.csv" in bib


def test_profile_covers_every_source_form():
    tokenizer = Tokenizer(str(PROFILE))
    for row in installed():
        form = row[2]
        converted = tokenizer(unicodedata.normalize("NFC", form), column="IPA")
        assert "�" not in converted, row[10]
        assert converted.replace(" ", "").replace("#", " ") == form.lower().replace("w", "v").replace("ṅ", "ŋ")
    assert tokenizer("Dēwṭō", column="IPA").replace(" ", "") == "dēvṭō"


def test_parser_keeps_original_and_all_82_rows():
    import make_cldf

    errors = io.StringIO()
    rows, stats = make_cldf.parse_file(str(CSV), errors, name="20260925-grierson-sikalgari")
    original = {r[10]: r for r in installed()}
    assert not errors.getvalue()
    assert len(rows) == stats["converted"] == 82
    assert {r.entry_key for r in rows} == set(original)
    assert all(r.old_form == original[r.entry_key][2] and r.form and "�" not in r.form for r in rows)
    assert all(r.lang == "Sik" and r.param == "" for r in rows)


def test_seeded_20_cell_scan_audit_is_reproducible():
    sample = json.loads((PACKAGE / "sample-audit-2026092601.json").read_text())
    assert sample["seed"] == 20260926 and sample["population"] == 82
    assert sample["sample_size"] == len(sample["rows"]) == 20 and sample["material_errors"] == 0
    assert {r["printed_page"] for r in sample["rows"]} == {181, 185, 189, 193}
    assert all(r["status"] == "matches-printed-source" and not r["material_error"] for r in sample["rows"])
    grouped = defaultdict(list)
    for row in audited():
        grouped[row["printed_page"]].append(row)
    rng = random.Random(sample["seed"])
    expected = []
    for page in sorted(grouped):
        expected += sorted(rng.sample(grouped[page], 5), key=lambda r: r["prompt"])
    assert [r["source_cell_key"] for r in sample["rows"]] == [r["source_cell_key"] for r in expected]
