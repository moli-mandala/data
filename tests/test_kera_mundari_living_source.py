"""Pinned complete web source; no build, installation, or network access."""
import csv
import importlib.util
import io
import json
import unicodedata as ud
from pathlib import Path

import pytest
import tags
from dialects import dialect_tag

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / 'data/other/forms/raw_data/kera_mundari_living_2026'

@pytest.fixture(scope='module')
def prepared():
    spec = importlib.util.spec_from_file_location('kera_proposal', P / 'prepare_source.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m, *m.prepare()

def test_all_entries_senses_notes_and_media_are_accounted(prepared):
    m, rows, audit, summary = prepared
    assert summary['raw_entries'] == summary['raw_senses'] == 446
    assert summary['dictionary_rows'] == 462
    assert summary['notes_reviewed'] == 440 and summary['audio_records'] == 445
    assert all(m['original_url'] == 'https://media.livingdictionaries.app/' + m['storage_path']
               for a in audit for m in a.get('media', []))
    assert len(rows) == 479 and len(audit) == 466
    assert len({r[10] for r in rows}) == 479
    assert {a['raw']['upstream_entry_id'] for a in audit if a['section'] == 'dictionary'} == {
        json.loads(l)['upstream_entry_id'] for l in (P / 'lexical-snapshot.jsonl').read_text().splitlines()}
    assert m.serialize(rows).encode() == (P / 'proposal.csv').read_bytes()
    assert ''.join(json.dumps(a, ensure_ascii=False) + '\n' for a in audit) == (P / 'proposal-audit.jsonl').read_text()

def test_source_adposition_registration_without_false_position(prepared):
    _, rows, audit, _ = prepared
    assert 'adposition' in tags.GRAMMATICAL_TAGS
    extracted, note = tags.extract_tags('adposition; Source position unspecified.', attestations=False)
    assert extracted == 'adposition' and note == 'Source position unspecified.'
    extracted, note = tags.extract_tags('An adposition may precede a noun.', attestations=False)
    assert not extracted and note == 'An adposition may precede a noun.'
    keys = {k for a in audit if a.get('source_pos') == ['adp'] for k in a['entry_keys']}
    assert len(keys) == 30
    assert all('adposition' in r[14].split() and not {'prep','postp'} & set(r[14].split()) for r in rows if r[10] in keys)
    frontend = (ROOT.parent / 'jambu-static/src/lib/tags.ts').read_text()
    assert "'adposition'" in frontend and "adposition: 'adposition'" in frontend

def test_explicit_alternates_no_homonym_merges(prepared):
    m, rows, _, _ = prepared
    assert m.readings('bʰõs(u)ɽi') == ['bʰõsuɽi', 'bʰõsɽi']
    assert m.readings('iɖi ~ idi') == ['iɖi', 'idi']
    twins = [r for r in rows if r[2] == 'maɳɖitihi' and r[3] == 'left or right']
    assert len(twins) == 2 and twins[0][10] != twins[1][10]
    bykey = {r[10]:r for r in rows}
    for r in rows:
        if r[11]:
            parent = bykey[r[11]]
            assert r[10].rsplit(':reading:',1)[0] == parent[10].rsplit(':reading:',1)[0]
            assert r[3] == parent[3]
        assert not r[12] and not r[13]

def test_borrowings_are_claims_without_donor_nodes(prepared):
    _, rows, audit, summary = prepared
    assert summary['borrowing_labels'] == {'Hindi Loan':26,'English Loan':1,'Possible Hindi Loan':1}
    for a in audit:
        for claim in a.get('borrowing_claims', []):
            assert claim['donor_form'] is None and claim['graph_edge'] is None
            out = [r for r in rows if r[10] in a['entry_keys']]
            assert all('loanword' in r[14].split() for r in out)
            if claim['certainty'] == 'possible':
                assert all('uncertain' in r[14].split() and 'tentatively' in r[6] for r in out)

def test_grammar_scope_and_about_controls(prepared):
    _, rows, audit, _ = prepared
    assert all('num' in r[14].split() for r in rows if ', prompt Num ' in r[7])
    assert sum('sentential' in r[14].split() for r in rows) == 1
    assert not any('verb' in r[14].split() for r in rows)
    assert all(not a['entry_keys'] for a in audit if a['section'] == 'about' and a['raw']['disposition'] == 'nonlexical-context')
    hasada = [r for r in rows if ':about:hasada-' in r[10]]
    assert len(hasada) == 2 and all('dialect:mu:hasada_mundari_living:' in r[14] for r in hasada)
    other = next(r for r in rows if ':about:past-keda:' in r[10])
    assert other[2] == '-keda' and 'dialect:' not in other[14]
    plural = next(r for r in rows if ':about:plural:' in r[10])
    assert plural[2] == '-kul-ko' and {'suffix','pl'} <= set(plural[14].split())
    assert 'non-human' in plural[6]
    assert all({t for t in r[14].split() if not t.startswith('dialect:')} <= tags.GRAMMATICAL_TAGS for r in rows)

def test_literal_profile_covers_nfc_and_nfd_without_invented_ipa(prepared):
    _, rows, _, _ = prepared
    rules = dict(line.split('\t') for line in (P / 'proposal-profile.txt').read_text().splitlines()[1:])
    assert rules['g'] == 'g' and all(k == v for k,v in rules.items())
    for r in rows:
        assert not r[5] and ud.is_normalized('NFC',r[2])
        assert all(c == ' ' or c in rules for c in ud.normalize('NFD',r[2]))
    dialect = next(csv.DictReader((P / 'proposed-dialect.csv').open()))
    assert dialect['Tag'] == dialect_tag('mu','kera_mundari_living','KeraɁ Mundari')
    assert not dialect['Latitude'] and not dialect['Longitude']

def test_actual_generic_parser_preserves_expansions_and_literal_source(prepared):
    import make_cldf
    from source_meta import SourceMeta
    from unittest.mock import patch
    _, rows, _, _ = prepared
    errors = io.StringIO()
    with patch('make_cldf.source_meta.load', return_value=SourceMeta([P/'proposal.yaml'])):
        parsed, _ = make_cldf.parse_file(str(P / 'proposal.csv'), errors)
    assert not errors.getvalue() and len(parsed) == len(rows) == 479
    bykey = {r[10]:r for r in rows}
    for r in parsed:
        raw = bykey[r.entry_key]
        assert r.old_form == raw[2] and r.ipa == '' and r.native == ''
        assert r.form == raw[2]
        assert set(raw[14].split()) <= set(r.tags.split())

def test_real_metadata_and_reference_formatter():
    from source_meta import SourceMeta
    from make_refs import ETYMOLOGY_PROVENANCE_VALUES, create_short_ref
    from pybtex.database import parse_file
    import pybtex
    meta = SourceMeta([P / 'proposal.yaml'])
    assert meta.flag('living-kera-mundari2026', 'reference', 'etymology_provenance') == 'source'
    assert meta.flag('living-kera-mundari2026', 'reference', 'etymology_provenance') in ETYMOLOGY_PROVENANCE_VALUES
    assert meta.flag('living-kera-mundari2026', 'forms', 'split_alternates') is False
    rule = meta.transcription_rule('living-kera-mundari2026', P/'proposal.csv', 'mu')
    assert rule['convert'] is False and rule['preserve_hyphens'] is True
    entry = parse_file(str(P / 'proposed-reference.bib')).entries['living-kera-mundari2026']
    text = pybtex.PybtexEngine().format_from_string(entry.to_string('bibtex'), 'plain', output_backend='markdown')
    assert 'Anderson' in text and '2026' in text and 'Kera' in text
    assert create_short_ref(entry)
