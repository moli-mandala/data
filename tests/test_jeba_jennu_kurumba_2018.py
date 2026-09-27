"""Focused checks for Jeba's archived Jennu Kurumba numeral table."""

import csv
import importlib.util
import io
import json
import random
import sys
from collections import Counter
from pathlib import Path

from segments.tokenizer import Tokenizer


DATA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATA))
PACKAGE = DATA / "data/other/forms/raw_data/jeba_jennu_kurumba_2018"
CSV = DATA / "data/other/forms/20260925-jeba-jennu-kurumba-numerals.csv"
PROFILE = DATA / "conversion/jeba-jennu-kurumba-2018.txt"
spec = importlib.util.spec_from_file_location("jeba_jennu", PACKAGE / "import_source.py")
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def installed():
    return list(csv.reader(CSV.open(encoding="utf-8", newline="")))


def audited():
    return [json.loads(line) for line in (PACKAGE / "audit.jsonl").read_text().splitlines()]


def test_both_contributor_tables_audited_and_regenerate():
    rows, audit = source.generate()
    assert rows == installed() and audit == audited()
    assert len(audit) == 82 and len(rows) == 42
    assert Counter(a["status"] for a in audit) == {"ingested": 42, "excluded_control": 40}
    assert Counter(a["table_index"] for a in audit) == {1: 42, 5: 40}
    assert [a["number"] for a in audit if a["status"] == "ingested"] == source.NUMBERS
    assert len({r[10] for r in rows}) == 42


def test_html_span_boundary_mixed_cells_and_control_exclusion():
    audit = audited()
    target = {a["number"]: a for a in audit if a["status"] == "ingested"}
    controls = [a for a in audit if a["status"] == "excluded_control"]
    assert target[26]["source_form"] == "ippɐt̪t̪ɐːɾɨ"
    assert "</span><span" in target[26]["raw_markup"]
    assert target[100]["joined_cell"] == target[200]["joined_cell"]
    assert target[400]["joined_cell"] == target[800]["joined_cell"]
    assert target[100]["source_form"] == "nuːɾɨ"
    assert target[2000]["source_form"] == "eɾɖɨ sɐːʋɾɐ"
    assert (target[1]["table_row"], target[1]["table_column"]) == (1, 1)
    assert (target[2000]["table_row"], target[2000]["table_column"]) == (20, 2)
    assert len({(a["table_row"], a["table_column"]) for a in controls}) == 40
    assert all(a["entry_key"] == "" and a["control_cell"] for a in controls)


def test_precise_canonical_and_legacy_kannada_rows_untouched():
    languages = {r[0]: r for r in csv.reader((DATA / "cldf/languages.csv").open())}
    target = languages["Jennu Kurumba"]
    assert target[2] == "jenn1240" and target[3:5] == ["", ""]
    assert target[5] == "S. Dravidian I" and "Karnataka" in target[6]
    dialects = {r[0]: r for r in csv.reader((DATA / "cldf/dialects.csv").open())}
    assert dialects["coorg"][2] == "Kannada"
    assert dialects["sil-kurumba-1985-masinagudi-jennu"][2] == "Kannada"
    for row in installed():
        assert len(row) == 15 and row[0] == "Jennu Kurumba" and row[14] == "num"
        assert row[2] == row[5] and row[2] and "�" not in row[2]
        assert row[7].startswith("jeba2018jennukurumba[Jeba Jennu Kurumba table, numeral ")
        assert row[1] == row[4] == row[6] == row[8] == row[9] == row[11] == row[12] == row[13] == ""
    bib = (DATA / "cldf/sources.bib").read_text()
    assert bib.count("@misc{jeba2018jennukurumba,") == 1
    assert "20260925-jeba-jennu-kurumba-numerals.csv" in bib


def test_profile_covers_source_and_scoped_conversion():
    import make_cldf

    tokenizer = Tokenizer(str(PROFILE))
    assert tokenizer("ippɐt̪t̪ɐːɾɨ", column="IPA").replace(" ", "") == "ippat̪t̪ārɨ"
    assert tokenizer("ɜːɭɨ", column="IPA").replace(" ", "") == "ɜ̄ḷɨ"
    assert tokenizer("on̪d̪ɨ", column="IPA").replace(" ", "") == "on̪d̪ɨ"
    for row in installed():
        assert "�" not in tokenizer(row[2], column="IPA"), row[10]
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(CSV), errors, name=CSV.stem)
    assert not errors.getvalue()
    assert len(parsed) == stats["converted"] == 42
    originals = {r[10]: r for r in installed()}
    assert {r.entry_key for r in parsed} == set(originals)
    assert all(r.lang == "Jennu Kurumba" and r.old_form == originals[r.entry_key][2] for r in parsed)


def test_seeded_audit_and_profile_inventory():
    import profile_policy

    sample = json.loads((PACKAGE / "sample-audit-2026092601.json").read_text())
    assert sample["seed"] == 20260926 and sample["population"] == 42
    assert sample["sample_size"] == len(sample["rows"]) == 20 and sample["material_errors"] == 0
    expected = sorted(random.Random(sample["seed"]).sample(
        [a for a in audited() if a["status"] == "ingested"], 20
    ), key=lambda a: a["number"])
    assert [r["source_cell_key"] for r in sample["rows"]] == [a["source_cell_key"] for a in expected]
    assert all(r["status"] == "matches-source-table" and not r["material_error"] for r in sample["rows"])
    assert "jeba-jennu-kurumba-2018" not in profile_policy.audit(profile_policy.source_inventory())
