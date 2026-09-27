"""Full-table checks for Erza's 2015 Pattapu numerals."""

import csv
import importlib.util
import io
import json
import random
import sys
from pathlib import Path

from segments.tokenizer import Tokenizer


DATA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATA))
PACKAGE = DATA / "data/other/forms/raw_data/erza_pattapu_numerals_2015"
CSV = DATA / "data/other/forms/20260925-erza-pattapu-numerals.csv"
PROFILE = DATA / "conversion/erza-pattapu-2015.txt"
spec = importlib.util.spec_from_file_location("erza_pattapu", PACKAGE / "import_source.py")
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def installed():
    return list(csv.reader(CSV.open(encoding="utf-8", newline="")))


def audited():
    return [json.loads(line) for line in (PACKAGE / "audit.jsonl").read_text().splitlines()]


def test_complete_source_inventory_and_regeneration():
    rows, audit = source.generate()
    assert rows == installed() and audit == audited()
    assert len(audit) == 42 and len(rows) == 43
    assert [a["number"] for a in audit] == source.NUMBERS
    assert len({r[10] for r in rows}) == len(rows)
    assert all(a["status"] == "ingested" for a in audit)
    assert all(len(r) == 15 and r[0] == "Pattapu" and r[14] == "num" for r in rows)


def test_double_numbered_cells_and_complete_alternatives():
    cells = {a["number"]: a for a in audited()}
    assert cells[200]["answers"] == ["renɖuru", "yaŋŋuru"]
    assert (cells[200]["table_row"], cells[200]["table_column"]) == (18, 2)
    assert (cells[400]["table_row"], cells[400]["table_column"]) == (18, 2)
    assert (cells[800]["table_row"], cells[800]["table_column"]) == (19, 2)
    assert (cells[1000]["table_row"], cells[1000]["table_column"]) == (19, 2)
    assert cells[5]["answers"] == ["aⁱndʒi"] and cells[25]["answers"] == ["irəvat̪t̪a aⁱndʒi"]
    assert cells[2000]["entry_keys"] == ["erza2015pattapu:number:2000:answer:1"]


def test_reference_language_and_distinct_source():
    languages = {r[0]: r for r in csv.reader((DATA / "cldf/languages.csv").open())}
    assert languages["Pattapu"][2] == "patt1247"
    bib = (DATA / "cldf/sources.bib").read_text()
    assert bib.count("@misc{erza2015pattapu,") == 1 and CSV.name in bib
    for r in installed():
        assert r[2] and "�" not in r[2]
        assert r[2] == r[5]
        assert r[7].startswith("erza2015pattapu[Pattapu table, numeral ")
        assert all(r[i] == "" for i in (1, 4, 6, 8, 9, 11, 12, 13))
    assert "lindgren2023dravidian" not in " ".join(r[7] for r in installed())


def test_profile_and_scoped_conversion():
    import make_cldf

    tokenizer = Tokenizer(str(PROFILE))
    assert tokenizer("muːɖu", column="IPA").replace(" ", "") == "mūḍu"
    assert tokenizer("aⁱndʒi", column="IPA").replace(" ", "") == "aⁱnji"
    assert tokenizer("irəvat̪t̪a", column="IPA").replace(" ", "") == "iravat̪t̪a"
    for row in installed():
        assert "�" not in tokenizer(row[2], column="IPA"), row[10]
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(CSV), errors, name=CSV.stem)
    assert not errors.getvalue()
    assert len(parsed) == stats["converted"] == 43
    originals = {r[10]: r for r in installed()}
    assert {r.entry_key for r in parsed} == set(originals)
    assert all(r.lang == "Pattapu" and r.old_form == originals[r.entry_key][2] for r in parsed)


def test_seeded_review():
    sample = json.loads((PACKAGE / "sample-audit-2026092601.json").read_text())
    assert sample["seed"] == 20260926 and sample["population"] == 42
    assert sample["sample_size"] == len(sample["rows"]) == 20 and sample["material_errors"] == 0
    expected = sorted(random.Random(sample["seed"]).sample(audited(), 20), key=lambda a: a["number"])
    assert [r["source_cell_key"] for r in sample["rows"]] == [a["source_cell_key"] for a in expected]
    assert all(not r["material_error"] for r in sample["rows"])
