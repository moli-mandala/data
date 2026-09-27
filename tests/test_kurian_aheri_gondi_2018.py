"""Focused checks for Kurian's archived Aheri Gondi numerals."""

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
PACKAGE = DATA / "data/other/forms/raw_data/kurian_aheri_gondi_2018"
CSV = DATA / "data/other/forms/20260925-kurian-aheri-gondi-numerals.csv"
PROFILE = DATA / "conversion/kurian-aheri-gondi-2018.txt"
spec = importlib.util.spec_from_file_location("kurian_aheri", PACKAGE / "import_source.py")
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def installed():
    return list(csv.reader(CSV.open(encoding="utf-8", newline="")))


def audited():
    return [json.loads(line) for line in (PACKAGE / "audit.jsonl").read_text().splitlines()]


def test_exact_prompt_inventory_and_regeneration():
    rows, audit = source.generate()
    assert rows == installed() and audit == audited()
    assert len(audit) == len(rows) == 42
    assert [a["number"] for a in audit] == source.NUMBERS
    assert all(a["status"] == "ingested" for a in audit)
    assert len({r[10] for r in rows}) == 42


def test_adjacent_spans_and_explicit_mixed_prompt_cells():
    by_number = {a["number"]: a for a in audited()}
    assert by_number[3]["joined_cell"] == "3. muːɖu"
    assert by_number[23]["joined_cell"] == "23. iruvəjmuːɖu"
    assert "</span><span" in by_number[3]["raw_markup"]
    assert by_number[13]["source_form"] == "pəd̪əmuːɖu"
    assert by_number[100]["joined_cell"] == by_number[200]["joined_cell"]
    assert by_number[100]["source_form"] == "nuːru"
    assert by_number[200]["source_form"] == "rɛɳɖunuhuku"
    assert by_number[400]["joined_cell"] == by_number[800]["joined_cell"]
    assert by_number[400]["source_form"] == "naːluŋnuhuku"
    assert by_number[800]["source_form"] == "ɛnmid̪inuhuku"
    assert (by_number[1]["table_row"], by_number[1]["table_column"]) == (1, 1)
    assert (by_number[2000]["table_row"], by_number[2000]["table_column"]) == (20, 2)


def test_precise_canonical_and_reference():
    languages = {r[0]: r for r in csv.reader((DATA / "cldf/languages.csv").open())}
    language = languages["Aheri Gondi"]
    assert language[2] == "aher1237" and language[3:5] == ["", ""]
    assert language[5] == "S. Dravidian II" and "Maharashtra" in language[6]
    assert languages["Gondi"][2] == "gond1265"
    for row in installed():
        assert len(row) == 15 and row[0] == "Aheri Gondi" and row[14] == "num"
        assert row[2] == row[5] and row[2] and "�" not in row[2]
        assert row[7].startswith("kurian2018aherigondi[Aheri Gondi table, numeral ")
        assert row[1] == row[4] == row[6] == row[8] == row[9] == row[11] == row[12] == row[13] == ""
    bib = (DATA / "cldf/sources.bib").read_text()
    assert bib.count("@misc{kurian2018aherigondi,") == 1
    assert "20260925-kurian-aheri-gondi-numerals.csv" in bib


def test_profile_and_scoped_conversion():
    import make_cldf

    tokenizer = Tokenizer(str(PROFILE))
    assert tokenizer("muːɖu", column="IPA").replace(" ", "") == "mūḍu"
    assert tokenizer("pəd̪ɛɳmid̪i", column="IPA").replace(" ", "") == "pad̪ɛṇmid̪i"
    assert tokenizer("hɛjːuŋ", column="IPA").replace(" ", "") == "hɛyyuŋ"
    assert tokenizer("pənːɛɳɖu", column="IPA").replace(" ", "") == "pannɛṇḍu"
    for row in installed():
        assert "�" not in tokenizer(row[2], column="IPA"), row[10]
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(CSV), errors, name=CSV.stem)
    assert not errors.getvalue()
    assert len(parsed) == stats["converted"] == 42
    originals = {r[10]: r for r in installed()}
    assert {r.entry_key for r in parsed} == set(originals)
    assert all(r.lang == "Aheri Gondi" and r.old_form == originals[r.entry_key][2] for r in parsed)


def test_seeded_audit_and_profile_inventory():
    import profile_policy

    sample = json.loads((PACKAGE / "sample-audit-2026092601.json").read_text())
    assert sample["seed"] == 20260926 and sample["population"] == 42
    assert sample["sample_size"] == len(sample["rows"]) == 20 and sample["material_errors"] == 0
    expected = sorted(random.Random(sample["seed"]).sample(audited(), 20), key=lambda a: a["number"])
    assert [r["source_cell_key"] for r in sample["rows"]] == [a["source_cell_key"] for a in expected]
    assert all(r["status"] == "matches-source-table" and not r["material_error"] for r in sample["rows"])
    assert "kurian-aheri-gondi-2018" not in profile_policy.audit(profile_policy.source_inventory())
