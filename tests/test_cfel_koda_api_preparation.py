"""Focused source-package checks; no complete data/database build."""

import csv
import hashlib
import importlib.util
import io
import json
import unicodedata
from pathlib import Path

from segments.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "data/other/forms/raw_data/cfel_koda_api_2026"
STEM = "20260925-cfel-koda-adornments"


def importer():
    spec = importlib.util.spec_from_file_location("cfel_koda_importer", PACKAGE / "import_source.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_query_scope_and_source_response_accounting():
    manifest = json.loads((PACKAGE / "manifest.json").read_text())
    for name, field in (("queries.json", "queries_sha256"), ("proposals.jsonl", "proposals_sha256")):
        assert hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest() == manifest[field]
    queries = json.loads((PACKAGE / "queries.json").read_text())["queries"]
    proposals = [json.loads(line) for line in (PACKAGE / "proposals.jsonl").read_text().splitlines()]
    assert len(queries) == len(set(queries)) == len(proposals) == 60
    assert [p["query"] for p in proposals] == queries
    assert all(p["reported_count"] == len(p["candidates"]) for p in proposals)
    assert sum(len(p["candidates"]) for p in proposals) == 62
    assert all(p["response_sha256"] and p["retrieved_utc"] for p in proposals)
    assert manifest["in_domain_count"] == 60 and manifest["installed_count"] == 47
    assert sum(manifest["print_comparator"]["domain_entries_by_xps_page"].values()) == 60


def test_installed_csv_and_audit_reproduce_from_snapshot():
    rows, audit = importer().build()
    assert len(rows) == 47 and len(audit) == 62
    assert {a["status"] for a in audit} == {
        "installed", "withheld_transcription", "excluded_other_domain"
    }
    assert sum(a["status"] == "withheld_transcription" for a in audit) == 13
    assert sum(a["status"] == "excluded_other_domain" for a in audit) == 2
    assert sum(bool(a["raw_native_alternates"]) for a in audit if a["publisher_domain"] == "Adornments and Costumes") == 13
    assert len({row[10] for row in rows}) == 47
    assert all(len(row) == 15 and row[0] == "Koda" and row[4] and row[14] in {"noun", "verb"} for row in rows)
    assert all(cell == unicodedata.normalize("NFC", cell) for row in rows for cell in row)
    by_gloss = {row[3]: row for row in rows}
    for left, right in (("Sandal", "Slipper"), ("Hat", "Cap"), ("Nose Pin", "Nose Ring")):
        assert by_gloss[left][2] == by_gloss[right][2]
        assert by_gloss[left][10] != by_gloss[right][10]
    # Retained pilot snapshot is historical evidence; the canonical file now
    # contains the independently reviewed full source.
    assert rows == list(csv.reader((PACKAGE / f"{STEM}.csv").open()))
    assert {r[10] for r in rows} <= {r[10] for r in csv.reader((ROOT / f"data/other/forms/{STEM}.csv").open())}
    assert audit == [json.loads(line) for line in (PACKAGE / "audit.jsonl").read_text().splitlines()]


def test_visual_sample_and_source_fields_survive_parser():
    import make_cldf

    languages = {row["ID"]: row for row in csv.DictReader((ROOT / "cldf/languages.csv").open())}
    assert languages["Koda"]["Glottocode"] == "koda1236"
    sample = json.loads((PACKAGE / "visual-sample-20260925.json").read_text())
    assert sample["seed"] == 20260925 and len(sample["records"]) == 20
    assert len({r["query"] for r in sample["records"]}) == 20
    assert all(r["xps_page"] == r["printed_page"] + 3 for r in sample["records"])
    path = ROOT / f"data/other/forms/{STEM}.csv"
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(path), errors, name=STEM)
    assert not errors.getvalue()
    assert len(parsed) == stats["converted"] == 3223
    raw = {r[10]: r for r in csv.reader(path.open())}
    for row in parsed:
        assert row.old_form == raw[row.entry_key][2]
        assert row.native == raw[row.entry_key][4]
        assert row.gloss == raw[row.entry_key][3]
        assert row.form and "�" not in row.form


def test_profile_preserves_source_contrasts_and_normalization():
    profile = Tokenizer(str(ROOT / "conversion/cfel-koda-api.txt"))

    def convert(value):
        return unicodedata.normalize("NFC", profile(value, column="IPA").replace(" ", "").replace("#", " "))

    assert convert("aŋʈi") == "aŋṭi"
    assert convert("t̪ʰole") == "tʰole"
    assert convert("nid̪d̪a sɔnɔʔ") == "nidda sɔnɔʔ"
    assert convert("kiʧiʔ siniʔ") == "kiciʔ siniʔ"
    rows, _ = importer().build()
    for row in rows:
        assert convert(unicodedata.normalize("NFD", row[2])) == convert(row[2])
