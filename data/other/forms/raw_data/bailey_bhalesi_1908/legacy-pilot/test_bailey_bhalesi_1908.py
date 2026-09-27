"""Focused source-input checks for Bailey's bounded Bhalesi list."""

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
PACKAGE = DATA / "data/other/forms/raw_data/bailey_bhalesi_1908"
CSV = DATA / "data/other/forms/20260925-bailey-bhalesi.csv"
PROFILE = DATA / "conversion/bailey-bhalesi-1908.txt"
spec = importlib.util.spec_from_file_location("bailey_bhalesi_1908", PACKAGE / "import_source.py")
source = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = source
spec.loader.exec_module(source)


def installed():
    with CSV.open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


def audited():
    return [json.loads(line) for line in (PACKAGE / "audit.jsonl").read_text().splitlines()]


def test_complete_list_and_regeneration():
    rows, audit = source.generate()
    assert rows == installed() and audit == audited()
    assert len(audit) == 34 and len(rows) == 16
    assert Counter(a["status"] for a in audit) == {
        "ingested": 15, "hold_typography": 16, "exclude_same_print": 1,
        "hold_complex": 1, "hold_incomplete": 1,
    }
    assert [a["vocabulary_item"] for a in audit] == list(range(1, 35))
    assert {(a["printed_page"], a["scan_page"]) for a in audit} == {(73, 187), (74, 188)}
    assert len({r[10] for r in rows}) == 16


def test_answers_exclusions_and_mapping():
    audit = {a["vocabulary_item"]: a for a in audited()}
    rows = {r[10]: r for r in installed()}
    assert all(r[0] == "bhal" and r[11] == "" for r in rows.values())
    assert len(audit[1]["entry_keys"]) == 2
    assert rows["bailey1908bhalesi:p73:left:item:1"][2] == "bāb"
    assert rows["bailey1908bhalesi:p73:left:item:1:answer2"][2] == "bājī"
    assert audit[15]["status"] == "exclude_same_print"
    assert audit[21]["status"] == "hold_complex"
    assert audit[22]["status"] == "hold_incomplete"
    assert not any("woman" == r[3] for r in rows.values())


def test_seeded_image_review():
    sampled = random.Random(1908).sample(audited(), 20)
    with (PACKAGE / "sample-review-20.tsv").open(encoding="utf-8", newline="") as stream:
        reviewed = list(csv.DictReader(stream, delimiter="\t"))
    assert [int(r["item"]) for r in reviewed] == [a["vocabulary_item"] for a in sampled]
    assert all(r["source_cell"] == a["printed_form_review"] for r, a in zip(reviewed, sampled))
    assert all(r["decision"] == a["status"] for r, a in zip(reviewed, sampled))
    assert all(r["manual_comparison"].startswith("match" if a["status"] == "ingested" else "held:") for r, a in zip(reviewed, sampled))


def test_profile_metadata_and_parse():
    import make_cldf
    import profile_policy
    import source_meta

    tokenizer = Tokenizer(str(PROFILE))
    for row in installed():
        assert len(row) == 15
        assert row[1] == row[4] == row[5] == row[8] == row[9] == row[11] == row[12] == row[13] == row[14] == ""
        original = unicodedata.normalize("NFC", row[2])
        assert tokenizer(original, column="IPA").replace(" ", "") == original
    assert source_meta.SourceMeta().transcription("bailey1908bhalesi", CSV, "bhal")[0] == "bailey-bhalesi-1908"
    assert "bailey-bhalesi-1908" not in profile_policy.audit({})
    assert "@book{bailey1908bhalesi," in (DATA / "cldf/sources.bib").read_text()
    assert "bhal" in {r[0] for r in csv.reader((DATA / "cldf/languages.csv").open())}
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(CSV), errors, name="20260925-bailey-bhalesi")
    assert not errors.getvalue()
    assert len(parsed) == stats["converted"] == 16
    raw = {r[10]: r for r in installed()}
    assert {r.entry_key for r in parsed} == set(raw)
    assert all(r.old_form == raw[r.entry_key][2] for r in parsed)
