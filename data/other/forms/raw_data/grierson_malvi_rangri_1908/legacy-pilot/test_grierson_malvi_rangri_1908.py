"""Focused checks for LSI IX(II) Mālvī (Rāngrī), printed p. 307."""

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
PACKAGE = DATA / "data/other/forms/raw_data/grierson_malvi_rangri_1908"
CSV = DATA / "data/other/forms/20260925-grierson-malvi-rangri.csv"
PROFILE = DATA / "conversion/grierson-malvi-rangri-1908.txt"
spec = importlib.util.spec_from_file_location("grierson_malvi_rangri_1908", PACKAGE / "import_source.py")
source = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = source
spec.loader.exec_module(source)


def installed():
    with CSV.open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


def audited():
    return [json.loads(line) for line in (PACKAGE / "audit.jsonl").read_text().splitlines()]


def test_bounded_page_regenerates_and_reconciles():
    rows, audit = source.generate()
    assert rows == installed() and audit == audited()
    assert len(audit) == 21 and len(rows) == 5
    assert Counter(a["status"] for a in audit) == {"ingested": 5, "hold_typography": 16}
    assert [a["standard_list_item"] for a in audit] == list(range(32, 53))
    assert {(a["printed_page"], a["scan_page"]) for a in audit} == {(307, 322)}
    assert all(a["canonical_language"] == "Malw" and a["control_excluded"] for a in audit)
    assert {a["standard_list_item"] for a in audit if a["status"] == "ingested"} == {32, 33, 34, 35, 44}
    assert len({r[10] for r in rows}) == len(rows)


def test_controls_held_glyphs_and_seeded_review():
    audit = {a["standard_list_item"]: a for a in audited()}
    assert audit[44]["standard_malvi_control"].startswith("Standard Mālvī has")
    assert audit[47]["status"] == "hold_typography"
    assert audit[52]["status"] == "hold_typography"
    assert all(a["reading_is_provisional"] for a in audit.values() if a["status"] != "ingested")
    with (PACKAGE / "transcription.tsv").open(encoding="utf-8", newline="") as stream:
        source_rows = list(csv.DictReader(stream, delimiter="\t"))
    with (PACKAGE / "sample-review-20.tsv").open(encoding="utf-8", newline="") as stream:
        reviewed = list(csv.DictReader(stream, delimiter="\t"))
    sampled = random.Random(307).sample(source_rows, 20)
    assert [int(r["item"]) for r in reviewed] == [int(r["item"]) for r in sampled]
    assert all(r["source_cell"] == s["printed_form_review"] and r["control_cell"] == s["standard_malvi_control"]
               and r["decision"] == s["decision"] for r, s in zip(reviewed, sampled))
    assert all(r["visual_check"] == "pass" and r["material_error"] == "0" for r in reviewed)


def test_metadata_dialect_profile_and_local_parse():
    import make_cldf
    import profile_policy
    import source_meta

    assert "@book{grierson1908malvirangri," in (DATA / "cldf/sources.bib").read_text()
    languages = {r["ID"]: r for r in csv.DictReader((DATA / "cldf/languages.csv").open())}
    assert languages["Malw"]["Glottocode"] == "malv1243"
    dialects = {r["ID"]: r for r in csv.DictReader((DATA / "cldf/dialects.csv").open())}
    assert dialects["lsi1908-malvi-rangri"]["Language_ID"] == "Malw"
    assert dialects["lsi1908-malvi-rangri"]["Tag"] == source.DIALECT
    assert not dialects["lsi1908-malvi-rangri"]["Latitude"] and not dialects["lsi1908-malvi-rangri"]["Longitude"]
    assert source_meta.SourceMeta().transcription("grierson1908malvirangri", CSV, "Malw")[0] == "grierson-malvi-rangri-1908"
    assert "grierson-malvi-rangri-1908" not in profile_policy.audit({})
    tokenizer = Tokenizer(str(PROFILE))
    for row in installed():
        assert len(row) == 15 and row[0] == "Malw"
        assert row[7].startswith("grierson1908malvirangri[p. 307, Mālvī (Rāngrī) column, standard-list item ")
        assert row[14] == source.DIALECT
        assert row[1] == row[4] == row[5] == row[6] == row[8] == row[9] == row[11] == row[12] == row[13] == ""
        assert "�" not in tokenizer(row[2], column="IPA")
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(CSV), errors, name="20260925-grierson-malvi-rangri")
    assert not errors.getvalue()
    assert len(parsed) == stats["converted"] == 5
    original = {r[10]: r for r in installed()}
    assert {r.entry_key for r in parsed} == set(original)
    assert all(r.old_form == original[r.entry_key][2] for r in parsed)
    assert {r.form for r in parsed} == {"hāt", "pag", "nāk", "ākh", "loh"}
