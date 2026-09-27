"""Focused checks for Vaz's archived Hill Madia numeral table."""

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
PACKAGE = DATA / "data/other/forms/raw_data/vaz_hill_maria_2011"
CSV = DATA / "data/other/forms/20260925-vaz-hill-maria-numerals.csv"
PROFILE = DATA / "conversion/vaz-hill-maria-2011.txt"
spec = importlib.util.spec_from_file_location("vaz_hill_maria", PACKAGE / "import_source.py")
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def installed():
    return list(csv.reader(CSV.open(encoding="utf-8", newline="")))


def audited():
    return [json.loads(line) for line in (PACKAGE / "audit.jsonl").read_text().splitlines()]


def test_target_comparator_inventory_and_regeneration():
    rows, audit = source.generate()
    assert rows == installed() and audit == audited()
    assert len(audit) == 80 and len(rows) == 40
    assert Counter(a["status"] for a in audit) == {"ingested": 40, "excluded-comparator": 40}
    assert Counter(a["table_index"] for a in audit) == {1: 40, 5: 40}
    assert sorted(a["number"] for a in audit if a["status"] == "ingested") == source.NUMBERS
    assert len({r[10] for r in rows}) == 40


def test_span_joins_marker_and_comments():
    target = {a["number"]: a for a in audited() if a["status"] == "ingested"}
    assert {n: target[n]["source_form"] for n in (10, 14, 15, 17, 18)} == {
        10: "d̪ɘʔa", 14: "tʃɘvd̪a", 15: "pɘnd̪ɾa", 17: "sɘt̪ɾa", 18: "ɘʈɾa"}
    assert target[7]["raw_cell"] == "7. ʔeːɽʊŋ *"
    assert target[7]["source_form"] == "ʔeːɽʊŋ" and target[7]["unresolved_marker"] == "*"
    assert target[30]["source_comment"] == ["( 20+ 10 )"]
    assert target[50]["source_comment"] == ["( 2 x 20+ 10 )"]
    assert target[1]["table_row"] == 1 and target[1]["table_column"] == 1
    assert target[2000]["table_row"] == 20 and target[2000]["table_column"] == 2
    assert all(not a["entry_key"] for a in audited() if a["status"] == "excluded-comparator")


def test_canonical_and_reference_registration():
    languages = {r[0]: r for r in csv.reader((DATA / "cldf/languages.csv").open())}
    assert languages["Maria (India)"][2] == "mari1414"
    assert not languages["Maria (India)"][3] and not languages["Maria (India)"][4]
    dialects = {r[0]: r for r in csv.reader((DATA / "cldf/dialects.csv").open())}
    assert dialects["vaz2011-hill-madia"][2] == "Maria (India)"
    assert not dialects["vaz2011-hill-madia"][6] and not dialects["vaz2011-hill-madia"][7]
    assert dialects["maria"][2] == "Gondi" and dialects["maria"][5] == "mari1414"
    for dialect in ("beine_gba", "beine_gbh", "beine_get", "beine_gja"):
        assert dialects[dialect][2] == "Gondi"
    for row in installed():
        assert len(row) == 15 and row[0] == "Maria (India)" and row[2] == row[5]
        assert row[2] and "�" not in row[2]
        assert row[7].startswith("vaz2011hillmaria[Hill Madia table, numeral ")
        assert all(row[i] == "" for i in (1, 4, 6, 8, 9, 11, 12, 13))
        assert row[14] == f"num {source.DIALECT}"
    bib = (DATA / "cldf/sources.bib").read_text()
    assert bib.count("@misc{vaz2011hillmaria,") == 1
    assert CSV.name in bib


def test_profile_and_scoped_conversion():
    import make_cldf

    tokenizer = Tokenizer(str(PROFILE))
    assert tokenizer("na:lʊŋ", column="IPA").replace(" ", "") == "nāluŋ"
    assert tokenizer("tʃɘvd̪a", column="IPA").replace(" ", "") == "cɘvd̪a"
    assert tokenizer("ʔɘdʒɘɾ", column="IPA").replace(" ", "") == "ʔɘjɘr"
    for row in installed():
        assert "�" not in tokenizer(row[2], column="IPA"), row[10]
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(CSV), errors, name=CSV.stem)
    assert not errors.getvalue()
    assert len(parsed) == stats["converted"] == 40
    originals = {r[10]: r for r in installed()}
    assert {r.entry_key for r in parsed} == set(originals)
    assert all(r.lang == "Maria (India)" and r.old_form == originals[r.entry_key][2] for r in parsed)


def test_seeded_review_and_profile_inventory():
    import profile_policy

    sample = json.loads((PACKAGE / "sample-audit-2026092601.json").read_text())
    target = [a for a in audited() if a["status"] == "ingested"]
    assert sample["seed"] == 20260926 and sample["population"] == 40
    assert sample["sample_size"] == len(sample["rows"]) == 20 and sample["material_errors"] == 0
    expected = sorted(random.Random(sample["seed"]).sample(target, 20), key=lambda a: a["number"])
    assert [r["source_cell_key"] for r in sample["rows"]] == [a["source_cell_key"] for a in expected]
    assert all(r["status"] == "matches-source-table" and not r["material_error"] for r in sample["rows"])
    assert "vaz-hill-maria-2011" not in profile_policy.audit(profile_policy.source_inventory())


def test_dialect_tag_is_one_registered_whitespace_free_token():
    from urllib.parse import unquote
    assert len(source.DIALECT.split()) == 1
    language = "Northern Hindko" if "kagani" in source.DIALECT else "Maria (India)"
    assert unquote(source.DIALECT.split(":")[1]) == language
    with (DATA / "cldf/dialects.csv").open(encoding="utf-8", newline="") as stream:
        assert any(row["Tag"] == source.DIALECT and row["Language_ID"] == language
                   for row in csv.DictReader(stream))
