"""Whole archived Ho expression recovery; source-stage checks only."""
import csv
import importlib.util
import io
import json
from pathlib import Path

from segments import Tokenizer

PACKAGE = Path(__file__).resolve().parents[1] / "data/other/forms/raw_data/ho_mla_2004"


def prepared():
    spec = importlib.util.spec_from_file_location("ho_expression_recovery", PACKAGE / "prepare_expression_recovery.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.prepare()


def test_entire_snapshot_accounting_and_legacy_identity():
    rows, census, records = prepared()
    assert len(rows) == 2227 and len(census) == 1524 and len(records) == 90
    assert [c["source_id"] for c in census if c["source_id"].isdigit()] == [f"{i:05d}" for i in range(1, 1517)]
    legacy = list(csv.reader((PACKAGE / "legacy-before-expression-recovery.csv").open()))
    assert [r[10] for r in rows[:2146]] == [r[10] for r in legacy]
    changed = [new[10] for old, new in zip(legacy, rows) if old != new]
    assert changed == ["ho-mla2004:01364:supplement:1", "ho-mla2004:01364:supplement:2"]
    keys = {r[10] for r in rows}
    assert len(keys) == len(rows)
    assert all(not r[11] or r[11] in keys for r in rows)
    assert rows == list(csv.reader((PACKAGE / "expression-recovery-proposed.csv").open()))


def test_whole_examples_and_explicit_alternative_scope():
    rows, _, records = prepared()
    forms = {r[2] for r in rows}
    assert {"esuko asultana", "esuko asulakana"} <= forms
    assert "alom boroya" in forms and "alom boroya (boro-e-a)" not in forms
    assert "buruken chauli gaRaken mayomeJ punjiyama-hinjiyama" in forms
    assert "mata-bA_r" in forms and "ir-bA_r" in forms
    prayer = next(r for r in records if r["source_id"] == "00305")
    assert " ... " in prayer["source_form"] and "sentential" not in prayer["tags"]
    assert "ellipsis" in prayer["notes"]
    abuse = next(r for r in records if r["source_id"] == "00815")
    assert "verb" not in abuse["tags"] and "pejorative" in abuse["tags"]


def test_untranslated_examples_and_shared_glosses_are_not_invented():
    rows, _, records = prepared()
    bykey = {r[10]: r for r in rows}
    unknown = next(r for r in records if r["source_id"] == "00010")
    assert unknown["gloss"] == "" and "verb" not in unknown["tags"]
    assert "??Gloss for quote" in unknown["raw_source"]
    for record in records:
        if not record["gloss"]:
            assert record["uncertainty_types"] and "uncertain" in record["tags"]
    for suffix in ("1", "2"):
        row = bykey["ho-mla2004:01364:supplement:" + suffix]
        assert row[3] == "" and "joint translation mageya spirits of hill and dale" in row[6]
        assert "uncertain" in row[14]
    assert bykey["ho-mla2004:01003:supplement:1"][3] == "to be attacked by a snake"


def test_full_profile_and_actual_parser_roundtrip(monkeypatch):
    import make_cldf

    rows, _, _ = prepared()
    tokenizer = Tokenizer(str(PACKAGE / "expression-recovery-profile.txt"))
    monkeypatch.setitem(make_cldf.convertors, "ho-mla", tokenizer)
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(PACKAGE / "expression-recovery-proposed.csv"), errors, name="20260925-donegan-stampe-ho")
    assert not errors.getvalue() and len(parsed) == stats["converted"] == 2227
    assert len({r.id for r in parsed}) == 2227
    bykey = {r[10]: r for r in rows}
    for parsed_row in parsed:
        raw = bykey[parsed_row.entry_key]
        assert parsed_row.old_form == raw[2]
        assert parsed_row.source == raw[7]
        assert parsed_row.notes == raw[6]
        assert parsed_row.tags == raw[14]
        normalized = tokenizer(raw[2], column="IPA").replace(" ", "").replace("#", " ")
        assert normalized == " ".join(raw[2].replace("w", "v").split())
        assert parsed_row.form == normalized
