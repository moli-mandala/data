"""Installed full-source checks for Bailey's complete Kotkhai chapter."""
import csv
import importlib.util
import io
import json
from collections import Counter
from pathlib import Path

from segments import Tokenizer

DATA = Path(__file__).resolve().parents[1]
PACKAGE = DATA / 'data/other/forms/raw_data/bailey_kotkhai_1908'
CSV = DATA / 'data/other/forms/20260925-bailey-kotkhai.csv'
PROFILE = DATA / 'conversion/bailey-kotkhai-1908.txt'
spec = importlib.util.spec_from_file_location('bailey_kotkhai_installed', PACKAGE / 'import_source.py')
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def test_complete_chapter_regeneration_and_legacy_continuity():
    rows, audit = source.generate()
    assert rows == list(csv.reader(CSV.open()))
    assert audit == [json.loads(x) for x in (PACKAGE / 'audit.jsonl').read_text().splitlines()]
    assert len(rows) == 73 and len(audit) == 77
    assert Counter(u['status'] for u in audit) == {'ingested': 65, 'source_blank': 11, 'excluded_control': 1}
    legacy = list(csv.reader((PACKAGE / 'legacy-pilot.csv').open()))
    by_key = {r[10]: r for r in rows}
    assert len(legacy) == 2
    for old in legacy:
        assert by_key[old[10]][2:4] == old[2:4]
    assert len([u for u in audit if u['section'] == 'lexical-difference']) == 5
    assert by_key['bailey1908kotkhai:p24:item:3'][2] == 'pāṭṛī'
    assert by_key['bailey1908kotkhai:p24:item:4'][2] == 'shēḷā'


def test_independent_audit_and_same_print_provenance():
    import hashlib
    verdict = json.loads((PACKAGE / 'independent-full-audit-final.json').read_text())
    assert verdict['status'] in {'passed', 'pass'} and verdict['material_errors'] == 0
    assert verdict['sample_size'] == 20
    for name, actual in [('proposal.csv', CSV), ('proposal-audit.jsonl', PACKAGE / 'audit.jsonl'), ('proposal-profile.txt', PROFILE)]:
        assert hashlib.sha256(actual.read_bytes()).hexdigest() == verdict['hashes'][name]
    rows, audit = source.generate()
    rice = next(u for u in audit if u['source_unit_key'] == 'bailey1908kotkhai:p24:item:2')
    assert rice['same_print_overlap']['entry_key'] == 'zoller2023:18.1:p686:1223:span5:lect1:form1'
    assert rice['status'] == 'ingested' and rice['entry_keys']
    assert all(not r[1] and not any(r[8:10]) and not any(r[11:14]) for r in rows)


def test_registered_metadata_profile_reference_and_actual_parser():
    import make_cldf
    import profile_policy
    import source_meta
    import pybtex
    import pybtex.database
    rows = list(csv.reader(CSV.open()))
    tokenizer = Tokenizer(str(PROFILE))
    meta = source_meta.SourceMeta()
    assert meta.transcription('bailey1908kotkhai', CSV, 'Kotkhai')[0] == 'bailey-kotkhai-1908'
    assert 'bailey-kotkhai-1908' not in profile_policy.audit({})
    dialects = {r['ID']: r for r in csv.DictReader((DATA / 'cldf/dialects.csv').open())}
    d = dialects['bailey1908-kotkhai']
    assert d['Language_ID'] == 'Kotkhai' and not d['Latitude'] and not d['Longitude']
    assert all(d['Tag'] in r[14].split() for r in rows)
    bibliography = pybtex.database.parse_file(str(DATA / 'cldf/sources.bib'))
    entry = bibliography.entries['bailey1908kotkhai']
    formatted = pybtex.PybtexEngine().format_from_string(entry.to_string('bibtex'), 'plain', output_backend='markdown')
    assert '1908' in formatted and 'Bailey' in formatted
    assert '77' in entry.fields['included'] and '73' in entry.fields['included']
    assert 'full-transcription.jsonl' in entry.fields['provenance']
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(CSV), errors, name='20260925-bailey-kotkhai')
    assert not errors.getvalue() and len(parsed) == stats['converted'] == 73
    raw = {r[10]: r for r in rows}
    assert {r.entry_key for r in parsed} == set(raw)
    for r in parsed:
        original = raw[r.entry_key]
        assert r.old_form == original[2] and not r.native and not r.ipa
        assert r.source == original[7]
        assert r.form == tokenizer(original[2], column='IPA').replace(' ', '').replace('#', ' ')
