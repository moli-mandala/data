"""Canonical full-chapter installation checks; detailed checks in full_stage."""
import csv
import importlib.util
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1]
PACKAGE = DATA / 'data/other/forms/raw_data/grierson_turi_1906'
CSV = DATA / 'data/other/forms/20260925-grierson-turi-sites.csv'
spec = importlib.util.spec_from_file_location('turi_install', PACKAGE / 'import_source.py')
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def test_installed_complete_chapter_is_reproducible():
    rows, audit = source.build()
    assert rows == list(csv.reader(CSV.open()))
    assert audit == [json.loads(l) for l in (PACKAGE / 'full-installed-audit.jsonl').read_text().splitlines()]
    manifest = json.loads((PACKAGE / 'manifest.json').read_text())
    assert manifest['installed'] == len(rows) == 291
    assert manifest['candidate_examples_and_controls'] == len(audit) == 352
    assert manifest['typed_uncertain_readings'] == 1


def test_reference_has_full_chapter_coverage():
    bib = (DATA / 'cldf/sources.bib').read_text()
    assert bib.count('@book{grierson1906lsi4,') == 1
    assert 'Complete Turi chapter, printed pp. 128-134' in bib
    assert 'grierson_turi_1906/full-installed-audit.jsonl' in bib
