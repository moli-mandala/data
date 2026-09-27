"""Focused checks for Crooke's independent 1892 Mirzapur Korwa glossary."""

import csv
import importlib.util
import io
import json
import sys
from collections import Counter
from pathlib import Path

from segments.tokenizer import Tokenizer


DATA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATA))
PACKAGE = DATA / "data/other/forms/raw_data/crooke_korwa_1892"
CSV = DATA / "data/other/forms/20260925-crooke-korwa-mirzapur.csv"
PROFILE = DATA / "conversion/crooke-korwa-1892.txt"
spec = importlib.util.spec_from_file_location("crooke_korwa", PACKAGE / "import_source.py")
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def installed():
    with CSV.open(encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def test_complete_scope_and_reproducibility():
    rows, audit = source.build()
    assert rows == installed()
    assert len(rows) == 123 and len(audit) == 123
    assert Counter(x["printed_page"] for x in audit) == {125: 28, 126: 42, 127: 40, 128: 13}
    assert Counter(x["status"] for x in audit) == {"selected": 123}
    assert len({x["entry_key"] for x in audit}) == 123
    assert all(x["reason_or_note"] for x in audit if x["status"] == "held")
    assert all((PACKAGE / f"printed-p{p}.png").exists() for p in range(125, 129))
    assert {x["scan_page"] - x["printed_page"] for x in audit} == {10}
    manifest = json.loads((PACKAGE / "manifest.json").read_text())
    assert (manifest["printed_items"], manifest["selected"], manifest["held"]) == (123, 123, 0)
    assert "W. H. P. Driver" in manifest["scope"]
    # The final printed page switches authors after Crooke's thirteenth line.
    assert [x["item"] for x in audit if x["printed_page"] == 128] == list(range(1, 14))
    assert audit[-1]["entry_key"] == "crooke1892korwa:mirzapur:p128:item13"


def test_seeded_visual_review_matches_audit():
    audit = [json.loads(line) for line in (PACKAGE / "legacy-before-full-recovery/audit.jsonl").read_text().splitlines()]
    by_key = {x["entry_key"]: x for x in audit}
    review = [json.loads(line) for line in (PACKAGE / "visual-sample-20260925.jsonl").read_text().splitlines()]
    assert len(review) == len({x["entry_key"] for x in review}) == 20
    assert Counter(x["printed_page"] for x in review) == {125: 6, 126: 6, 127: 6, 128: 2}
    assert not any(x["material_error"] or not x["inventory_match"] for x in review)
    for sample in review:
        row = by_key[sample["entry_key"]]
        assert row["printed_page"] == sample["printed_page"]
        assert row["source_form"] == sample["printed_form"]
        assert row["status"] == sample["status"]


def test_independent_provenance_and_scope_boundary():
    rows = installed()
    assert all(len(r) == 15 and r[0] == "kw" and source.DIALECT in r[14].split() for r in rows)
    assert all(r[7].startswith("crooke1892korwa[p. ") for r in rows)
    assert all(r[10].startswith("crooke1892korwa:mirzapur:") for r in rows)
    assert not any("driver" in " ".join(r).casefold() for r in rows)
    assert any(r[2:4] == ["lutur", "ear"] for r in rows)
    assert any(r[2] == "kori hopûnu" for r in rows)
    dialects = {r[0]: r for r in csv.reader((DATA / "cldf/dialects.csv").open())}
    d = dialects["crooke-1892-mirzapur"]
    assert d[1:5] == [source.DIALECT, "kw", "crooke1892korwa:Korwa (southern Mirzapur)", "Mirzapur Korwa"]
    assert d[5:8] == ["", "", ""] and d[8] == "Munda"
    bib = (DATA / "cldf/sources.bib").read_text()
    assert bib.count("@article{crooke1892korwa,") == 1
    assert "20260925-crooke-korwa-mirzapur.csv" in bib


def test_profile_and_scoped_parser():
    tokenizer = Tokenizer(str(PROFILE))
    assert tokenizer("ingâ", column="IPA").replace(" ", "") == "ingā"
    assert tokenizer("chirâ", column="IPA").replace(" ", "") == "cirā"
    assert tokenizer("buwâku", column="IPA").replace(" ", "") == "buvāku"
    import make_cldf

    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(CSV), errors, name=CSV.stem)
    assert not errors.getvalue()
    assert len(parsed) == stats["converted"] == 123
    assert len({r.form for r in parsed}) >= 90
