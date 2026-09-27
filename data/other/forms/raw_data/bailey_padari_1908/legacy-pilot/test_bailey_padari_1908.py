"""Focused source-input checks for Bailey's bounded Pādari glossary."""

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
PACKAGE = DATA / "data/other/forms/raw_data/bailey_padari_1908"
CSV = DATA / "data/other/forms/20260925-bailey-padari.csv"
PROFILE = DATA / "conversion/bailey-padari-1908.txt"
spec = importlib.util.spec_from_file_location("bailey_padari_1908", PACKAGE / "import_source.py")
source = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = source
spec.loader.exec_module(source)


def installed():
    with CSV.open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


def audited():
    return [json.loads(line) for line in (PACKAGE / "audit.jsonl").read_text().splitlines()]


def test_complete_column_and_regeneration():
    rows, audit = source.generate()
    assert rows == installed() and audit == audited()
    assert len(audit) == 28 and len(rows) == 13
    assert Counter(a["status"] for a in audit) == {
        "ingested": 13, "hold_typography": 13, "exclude_same_print": 2,
    }
    assert [a["vocabulary_item"] for a in audit] == list(range(1, 29))
    assert {(a["printed_page"], a["scan_page"], a["column"]) for a in audit} == {(82, 196, "right")}
    assert len({r[10] for r in rows}) == 13


def test_source_mapping_and_duplicate_evidence():
    audit = {a["vocabulary_item"]: a for a in audited()}
    rows = {r[10]: r for r in installed()}
    assert all(r[0] == "Padri" and r[11] == "" for r in rows.values())
    assert audit[9]["status"] == audit[18]["status"] == "exclude_same_print"
    assert not any(r[3] in {"fox", "hair"} for r in rows.values())
    assert rows["bailey1908padari:p82:right:item:1"][2:4] == ["sūr", "pig"]


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
    assert source_meta.SourceMeta().transcription("bailey1908padari", CSV, "Padri")[0] == "bailey-padari-1908"
    assert "bailey-padari-1908" not in profile_policy.audit({})
    assert "@book{bailey1908padari," in (DATA / "cldf/sources.bib").read_text()
    assert "Padri" in {r[0] for r in csv.reader((DATA / "cldf/languages.csv").open())}
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(CSV), errors, name="20260925-bailey-padari")
    assert not errors.getvalue()
    assert len(parsed) == stats["converted"] == 13
    raw = {r[10]: r for r in installed()}
    assert {r.entry_key for r in parsed} == set(raw)
    assert all(r.old_form == raw[r.entry_key][2] for r in parsed)
