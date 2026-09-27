"""Installed whole-article source-stage checks; no database build."""
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import yaml
from segments import Tokenizer

DATA = Path(__file__).resolve().parents[1]
PACKAGE = DATA / 'data/other/forms/raw_data/samuells_juang_1856'
CSV = DATA / 'data/other/forms/20260925-samuells-juang.csv'
PROFILE = DATA / 'conversion/samuells-juang-1856.txt'
spec = importlib.util.spec_from_file_location('samuells_installed', PACKAGE / 'import_source.py')
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def test_installed_whole_source_matches_independent_full_audit():
    rows, audit = source.build()
    assert rows == list(csv.reader(CSV.open()))
    assert audit == [json.loads(x) for x in (PACKAGE / 'audit.jsonl').read_text().splitlines()]
    assert len(rows) == 35 and len(audit) == 33
    report = json.loads((PACKAGE / 'independent-full-output-review-20260926.json').read_text())
    assert report['status'] == 'passed' and report['material_errors'] == 0
    assert report['reviewed_units'] == 33 and report['reviewed_forms'] == 35
    for name, actual in [('proposal.csv', CSV), ('proposal-audit.jsonl', PACKAGE/'audit.jsonl'), ('proposal-profile.txt', PROFILE)]:
        assert hashlib.sha256(actual.read_bytes()).hexdigest() == report['hashes'][name]
    manifest = json.loads((PACKAGE / 'manifest.json').read_text())
    assert (manifest['installed_forms'], manifest['held_prompt_cells'], manifest['selected_prompt_cells']) == (35, 0, 31)
    assert 'public domain' in manifest['rights'].lower()
    assert '1856' in manifest['edition_date'] and '1857' in manifest['edition_date']
    assert json.loads((PACKAGE / 'full-overlap-review.json').read_text())['other_juang_rows_checked'] >= 2204


def test_registered_source_dialect_reference_and_profile_policy():
    import pybtex.database
    rows, _ = source.build()
    meta = yaml.safe_load(CSV.with_suffix('.yaml').read_text())
    assert meta['defaults']['identity']['append_order'] == 102
    assert meta['defaults']['transcription']['input'] == 'form'
    assert meta['defaults']['transcription']['preserve_hyphens'] is True
    assert meta['sources']['samuells1856juang']['reference']['ocr'] is False
    assert meta['sources']['samuells1856juang']['forms']['split_alternates'] is False
    dialects = {r[0]: r for r in csv.reader((DATA/'cldf/dialects.csv').open())}
    assert dialects['samuells-1856-juang'][2] == 'ju'
    assert dialects['samuells-1856-juang'][5:8] == ['', '', '']
    bib = pybtex.database.parse_file(DATA/'cldf/sources.bib').entries['samuells1856juang']
    assert bib.fields['year'] == '1856' and bib.fields['ocr'] == 'No'
    assert '35 forms' in bib.fields['included']
    tok = Tokenizer(str(PROFILE))
    for r in rows:
        assert tok(r[2], column='IPA').replace(' ', '').replace('#', ' ') == r[2].lower()


def test_actual_installed_parser_original_keys_aliases_and_legacy_ids():
    import make_cldf
    rows, _ = source.build()
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(CSV), errors, name=CSV.stem)
    assert not errors.getvalue() and len(parsed) == stats['converted'] == 35
    legacy = list(csv.reader((PACKAGE/'legacy-before-full-recovery/20260925-samuells-juang.csv').open()))
    assert [r.entry_key for r in parsed[:21]] == [r[10] for r in legacy]
    assert len({r.id for r in parsed}) == 35
    for raw, result in zip(rows, parsed):
        assert result.old_form == raw[2] and result.form == raw[2].lower()
        assert result.notes == raw[6] and result.source == raw[7]
        assert result.entry_key == raw[10] and result.variant_of_key == raw[11]
        assert result.native == '' and result.etymology == '' and result.borrowed_from_key == ''
        assert set(raw[14].split()) <= set(result.tags.split())
