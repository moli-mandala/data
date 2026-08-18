import csv
import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "data/other/forms/raw_data/southworth_marathi.py"
SPEC = importlib.util.spec_from_file_location("southworth_marathi_importer", SCRIPT)
southworth = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = southworth
SPEC.loader.exec_module(southworth)


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_checked_transcription_counts_and_stable_keys():
    forms, audit, blocks, sample = southworth.build_outputs()
    assert len(southworth.TABLE1) == 25
    assert len(southworth.TABLE2) == 23
    assert len(forms) == 30
    assert len(audit) == 48
    assert len(blocks) == 25
    assert len(sample) == 20
    assert {len(row) for row in forms} == {15}
    assert len({row[10] for row in forms}) == len(forms)
    assert len({row["Record_Key"] for row in audit}) == len(audit)
    assert {row[0] for row in forms} == {"M", "OM"}
    assert all(row[1].startswith(">d") for row in forms)
    assert all(row[7].startswith("southworth2005m[p. ") for row in forms)
    assert all(row[7].count("southworth2005m[") == 1 for row in forms)
    assert sum(" and p. 10, table 2" in row[7] for row in forms) == 8
    assert all(row[14].split()[0] == "loanword" for row in forms)


def test_source_symbols_old_marathi_and_uncertainty_are_separated():
    forms, audit, _, _ = southworth.build_outputs()
    by_key = {row[10]: row for row in forms}
    assert by_key["southworth2005m:p9:t1:r07:marathi"][2] == "kāḷ@"
    assert by_key["southworth2005m:p9:t1:r07:old-marathi"][2] == "kāḷa-"
    assert by_key["southworth2005m:p9:t1:r19:old-marathi"][2] == "mecu"
    assert by_key["southworth2005m:p9:t1:r22:old-marathi"][2] == "ḍoi"
    assert by_key["southworth2005m:p9:t1:r02:marathi"][14] == "loanword uncertain"
    assert by_key["southworth2005m:p9:t1:r16:marathi"][14] == "loanword verb"
    assert by_key["southworth2005m:p9:t1:r21:marathi"][14] == "loanword adj"
    assert all(row["Review"].startswith("source-image-verified") for row in audit)
    assert {row["Material_Error"] for row in audit} == {"no"}


def test_distribution_grid_preserves_plus_minus_unknown_and_blank():
    by_item = {row.item: row for row in southworth.TABLE2}
    assert len(southworth.DIST_COLUMNS) == 22
    assert by_item[1].marks == "+" * 22
    assert by_item[7].marks == ".....?.+..++++....++.."
    assert by_item[10].marks == ".....?...??.....+.++.."
    assert by_item[11].marks == "-------------------+--"
    assert by_item[23].marks == "-+-----------------+--"
    assert all(len(row.marks) == len(southworth.DIST_COLUMNS) for row in by_item.values())


def test_printed_citation_mismatches_are_conservative_and_audited():
    by_item = {row.item: row for row in southworth.TABLE2}
    dog = by_item[17]
    assert dog.printed_citation == "3276-8"
    assert dog.targets == ("3277", "3278")
    assert dog.match_status == "partial_printed_range"
    assert "rent, lease" in dog.match_reason and "3275" in dog.match_reason

    pool = by_item[20]
    assert pool.printed_citation == "5634"
    assert pool.targets == ("5635",)
    assert pool.match_status == "corrected_unique_headword_gloss"
    assert "form and gloss" in pool.match_reason


def test_installed_outputs_account_for_every_record():
    forms_path = ROOT / "data/other/forms/20260818-southworth-marathi.csv"
    audit_path = ROOT / "data/other/forms/raw_data/20260818-southworth2005m-audit.csv"
    blocks_path = ROOT / "data/other/entry_texts/20260818-southworth2005m.csv"
    sample_path = ROOT / "data/other/forms/raw_data/20260818-southworth2005m-sample.csv"
    if not all(path.exists() for path in (forms_path, audit_path, blocks_path, sample_path)):
        pytest.skip("run the Southworth importer with --install")

    with forms_path.open(encoding="utf-8", newline="") as stream:
        forms = list(csv.reader(stream))
    audit = read_rows(audit_path)
    blocks = read_rows(blocks_path)
    sample = read_rows(sample_path)
    assert len(forms) == 30
    assert len(audit) == 48
    assert len(blocks) == 25
    assert len(sample) == 20
    assert sum(len(row["Emitted_Keys"].split("|")) for row in audit if row["Emitted_Keys"]) == 30
    assert sum(len(row["Entry_Text_Targets"].split("|")) for row in audit if row["Entry_Text_Targets"]) == 25
    assert {row["Status"] for row in audit} == {"ingested", "comparison_ingested", "comparison_on_form"}
    assert {row["Material_Error"] for row in sample} == {"no"}
    assert {row["Seed"] for row in sample} == {str(southworth.SAMPLE_SEED)}


def test_compiled_rows_edges_keys_and_blocks_survive_full_build():
    key_path = ROOT / "cldf/form-source-keys.csv"
    if not key_path.exists():
        pytest.skip("run make all")
    keys = read_rows(key_path)
    source_keys = {
        row["Source_Key"]: row["Legacy_ID"]
        for row in keys
        if row["Source_Key"].startswith("southworth2005m:p9:t1:")
    }
    if not source_keys:
        pytest.skip("run make all after installing the Southworth source")
    assert len(source_keys) == 30

    aliases = {
        row["Legacy_ID"]: row["Form_ID"]
        for row in read_rows(ROOT / "cldf/form-id-aliases.csv")
    }
    forms = read_rows(ROOT / "cldf/forms.csv")
    by_id = {row["ID"]: row for row in forms}
    compiled = [by_id[aliases[legacy_id]] for legacy_id in source_keys.values()]
    assert len(compiled) == 30
    assert {row["Language_ID"] for row in compiled} == {"M", "OM"}
    assert all("southworth2005m[p. 9" in row["Source"] for row in compiled)
    assert "pʰaḷ" in {row["Form"] for row in compiled}
    assert "āī" in {row["Form"] for row in compiled}
    assert "māṇḍī" in {row["Form"] for row in compiled}
    assert "ḍokə̄" in {row["Form"] for row in compiled}

    edges = read_rows(ROOT / "cldf/edges.csv")
    rank1 = {
        row["Child_ID"]: row
        for row in edges
        if row["Rank"] == "1" and row["Kind"] in {"reflex", "borrowed", "variant"}
    }
    assert all(rank1[row["ID"]]["Kind"] == "borrowed" for row in compiled)
    assert all(rank1[row["ID"]]["Parent_ID"].startswith("d") for row in compiled)

    blocks = [
        row for row in read_rows(ROOT / "cldf/entry-texts.csv")
        if row["Source"].startswith("southworth2005m[p. 10")
    ]
    assert len(blocks) == 25
    assert {row["Kind"] for row in blocks} == {"comparison"}
    assert any(
        row["Form_ID"] == "5635" and "printed Turner reference 5634" in row["Content"]
        for row in blocks
    )


def test_reference_metadata_records_partial_scope_and_ocr():
    source = (ROOT / "cldf/sources.bib").read_text(encoding="utf-8")
    entry = source.split("@article{southworth2005m,", 1)[1].split("\n}", 1)[0]
    assert "DravidianElement.pdf" in entry
    assert "Tables 1--2" in entry
    assert "ocr" in entry.casefold()
    assert "licen" in entry.casefold()

    references = {
        row["ID"]: row
        for row in read_rows(ROOT / "cldf/references.csv")
    }
    formatted = references["southworth2005m"]["Source"]
    assert r"\textasciitilde" not in formatted
    assert "https://ccat.sas.upenn.edu/~fsouth/DravidianElement.pdf" in formatted
