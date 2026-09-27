"""Focused source-input checks; no full CLDF or browser DB build."""

import csv
import importlib.util
import io
import json
import random
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from segments.tokenizer import Tokenizer


DATA = Path(__file__).resolve().parents[1]
PACKAGE = DATA / "data/other/forms/raw_data/grierson_suketi_1916"
CSV = DATA / "data/other/forms/20260925-grierson-suketi.csv"
PROFILE = DATA / "conversion/grierson-suketi-1916.txt"
spec = importlib.util.spec_from_file_location("grierson_suketi_1916", PACKAGE / "import_source.py")
source = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = source
spec.loader.exec_module(source)


def installed():
    with CSV.open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


def audited():
    return [json.loads(line) for line in (PACKAGE / "audit.jsonl").read_text().splitlines()]


def test_exact_bounded_scope_counts_and_regeneration():
    rows, audit = source.generate()
    assert rows == installed()
    assert audit == audited()
    assert len(audit) == 61 and len(rows) == 55
    assert Counter(a["status"] for a in audit) == {
        "ingested": 53, "ambiguous": 5, "blank": 2, "skip_complex": 1,
    }
    assert {a["prompt"] for a in audit} == set(source.EXPECTED_PROMPTS)
    assert len({r[10] for r in rows}) == 55


def test_page_item_locators_blanks_and_multiple_answers():
    audit = {a["prompt"]: a for a in audited()}
    rows = {r[10]: r for r in installed()}
    for prompt, page, pdf_page in ((1, 759, 775), (13, 759, 775), (32, 760, 776), (52, 760, 776), (53, 761, 777), (79, 761, 777)):
        a = audit[prompt]
        assert (a["printed_page"], a["pdf_page"], a["citation_locator"]) == (page, pdf_page, f"p. {page}, item {prompt}")
    assert audit[57]["status"] == audit[61]["status"] == "blank"
    assert audit[50]["status"] == "skip_complex" and "oblique" in audit[50]["reason"]
    assert {p for p, a in audit.items() if a["status"] == "ambiguous"} == {35, 36, 48, 72, 76}
    assert rows["grierson1916suketi:prompt:49:answer2"][2] == "bhāyā"
    assert rows["grierson1916suketi:prompt:49:answer2"][11] == ""
    assert rows["grierson1916suketi:prompt:51:answer2"][2] == "mānachh"
    assert rows["grierson1916suketi:prompt:79"][7] == "grierson1916suketi[p. 761, item 79]"


def test_seeded_twenty_cell_page_image_review():
    audit = audited()
    with (PACKAGE / "sample-review-20.tsv").open(encoding="utf-8", newline="") as stream:
        reviewed = list(csv.DictReader(stream, delimiter="\t"))
    sampled = random.Random(1916).sample(audit, 20)
    assert [int(r["prompt"]) for r in reviewed] == [a["prompt"] for a in sampled]
    assert all(r["source_cell"] == a["raw_source_transcription"] for r, a in zip(reviewed, sampled))
    assert all(r["decision"] == a["status"] for r, a in zip(reviewed, sampled))
    assert all(r["manual_comparison"].startswith(("match", "held")) for r in reviewed)
    assert sum(r["manual_comparison"] == "match" for r in reviewed) == 18


def test_canonical_language_references_and_complete_profile():
    import profile_policy
    import source_meta

    rows = installed()
    tokenizer = Tokenizer(str(PROFILE))
    for row in rows:
        assert len(row) == 15 and row[0] == "suk"
        assert row[1] == row[4] == row[5] == row[8] == row[9] == row[11] == row[12] == row[13] == row[14] == ""
        assert row[2] and "�" not in row[2]
        assert row[7].startswith("grierson1916suketi[p. ")
        original = unicodedata.normalize("NFC", row[2])
        converted = tokenizer(original, column="IPA").replace(" ", "")
        assert converted == original.lower().replace("w", "v"), row[10]
    assert source_meta.SourceMeta().transcription("grierson1916suketi", CSV, "suk")[0] == "grierson-suketi-1916"
    assert "grierson-suketi-1916" not in profile_policy.audit({})
    assert "@book{grierson1916suketi," in (DATA / "cldf/sources.bib").read_text()
    languages = {r[0] for r in csv.reader((DATA / "cldf/languages.csv").open())}
    assert "suk" in languages


def test_cldf_parse_preserves_all_source_rows_and_originals():
    import make_cldf

    errors = io.StringIO()
    rows, stats = make_cldf.parse_file(str(CSV), errors, name="20260925-grierson-suketi")
    raw = {r[10]: r for r in installed()}
    assert not errors.getvalue()
    assert len(rows) == stats["converted"] == 55
    assert {r.entry_key for r in rows} == set(raw)
    assert all(r.old_form == raw[r.entry_key][2] for r in rows)
    assert all(r.form == r.old_form.lower().replace("w", "v") for r in rows)
