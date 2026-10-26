"""Whole pinned dictionary recovery, without installation or a database build."""
import csv
import importlib.util
import io
import json
import sys
import unicodedata
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1] / 'data/other/forms/raw_data/birhor_living_2026'
OUT = ROOT / 'whole-snapshot-recovery'


@pytest.fixture(scope='module')
def proposal():
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location('birhor_whole', ROOT / 'prepare_full_source.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.prepare()


def test_complete_snapshot_notes_and_prior_identity(proposal):
    rows, audit, summary = proposal
    heads = [a for a in audit if a['section'] == 'structured-head-and-senses']
    assert len(heads) == summary['source_entries'] == 2746
    assert all(a['entry_keys'] for a in heads)
    assert summary['legacy_keys_preserved'] == 3100
    assert summary['annotation_fields'] == 98
    expected = {json.loads(l)['candidate_key'] for l in (OUT / 'parenthetical-candidates.jsonl').open()}
    assert expected == {a['source_unit_key'] for a in audit if ':parenthesis:' in a['source_unit_key']}
    by_key = {r[10]: r for r in rows}
    assert len(by_key) == len(rows)
    for a in audit:
        assert set(a['entry_keys']) <= by_key.keys()
        if a['status'] == 'audit-only-context':
            assert not a['entry_keys']


def test_held_heads_recovered_without_invented_alignment(proposal):
    rows, _, _ = proposal
    at = lambda eid: [r for r in rows if f':{eid}:recovered-head:' in r[10]]
    blank = at('Rh3p0zj9xcsBp3YfJbLN')
    assert [(r[2], r[3]) for r in blank] == [('bir saɽu aɽaʔ', 'Colocasia esculenta (L.) Schott')]
    parts = at('boBzg1Wpance1ExpE3p1')
    assert [(r[2], r[3]) for r in parts] == [
        ('haser kapu', 'bulbils of Dioscorea bulbifera L.'),
        ('bawla', 'tubers of Dioscorea bulbifera L.')]
    assert all(not r[4] and not r[5] and not r[11] for r in parts)
    bare = at('lV4SFRDZBI0R4KykWpwJ')[0]
    assert bare[2] == 'babat buna' and bare[4] == 'बबत बुना' and not bare[5]
    assert 'not language-labelled' in bare[6] and 'uncertain' in bare[14].split()


def test_explicit_grammar_and_source_claim_scope(proposal):
    rows, _, _ = proposal
    for eid, tags in [('BQljQyC1C4TMSfO3GcIV', {'verb', '2pl'}),
                      ('kZmdjSWHo52HBu00NyKT', {'verb'}),
                      ('kh7O4atx2XqroLCLdypE', {'loanword'}),
                      ('aR1hCEqiU0nyY7D5WDmR', {'2sg', 'multiword-expression'})]:
        found = [r for r in rows if f':{eid}:sense:' in r[10]]
        assert found and all(tags <= set(r[14].split()) for r in found)
        if eid == 'kZmdjSWHo52HBu00NyKT':
            assert all('inf' not in r[14].split() for r in found)
    for r in rows:
        if ':note:' in r[10]:
            assert not r[4] and not r[5]
            assert all(t.startswith('living-birhor2026[') for t in r[7].split(';'))
    second_plural = next(r for r in rows if ':C7p2ookGsnHyJl0wKgXM:sense:' in r[10])
    assert second_plural[3] == 'you' and {'pron', '2pl'} <= set(second_plural[14].split())
    irregular = [r for r in rows if ':WuKw7IHCzydDhokpXSZZ:sense:' in r[10]]
    assert [r[3] for r in irregular] == ['muskrat, Sorex caerulescens', 'mole rat (Talpidae)']
    assert ':meaning:1:' in irregular[0][10] and ':meaning:2:' in irregular[1][10]


def test_exact_reuse_preserves_each_citation_and_alternative(proposal):
    rows, audit, summary = proposal
    assert summary['exact_note_reuses'] > 0
    by_key = {r[10]: r for r in rows}
    for a in audit:
        if a['section'] != 'source-note-attestation' or not a['entry_keys']:
            continue
        forms = a['decision'].get('forms') or [a['decision']['form']]
        for form, key in zip(forms, a['entry_keys']):
            row = by_key[key]
            assert row[2] == unicodedata.normalize('NFC', form)
            assert row[3] == unicodedata.normalize('NFC', a['decision']['gloss'])
            assert f"entry {a['entry_id']}, note " in row[7]
    assert all(not r[11] or r[11] in by_key and r[11] != r[10] for r in rows)
    documentary = [a for a in audit if a.get('decision', {}).get('disposition') == 'documentary-prose-label-unassigned-language']
    assert {a['decision']['form'] for a in documentary} == {'dael', 'bhang', 'laddu', 'chutney', 'pidha', 'okhli'}
    assert all(not a['entry_keys'] for a in documentary)
    assert any(r[2] == 'loo' and r[3] == 'sunstroke' for r in rows)


def test_profile_nasal_equivalence_house_w_and_unknowns(proposal):
    from segments.tokenizer import Tokenizer
    tok = Tokenizer(str(OUT / 'proposal-profile.txt'))
    def convert(s):
        return unicodedata.normalize('NFC', tok(s, column='IPA').replace(' ', '').replace('#', ' '))
    assert convert('bawla') == 'bavla'
    assert convert('dʒo') == 'jo' and convert('majom') == 'mayom'
    for r in proposal[0]:
        value = r[5] or r[2]
        assert '�' not in convert(value)
        assert convert(value) == convert(unicodedata.normalize('NFD', value))


def test_real_parser_retains_layers_and_durable_keys(proposal, tmp_path):
    import make_cldf
    import source_meta
    from segments.tokenizer import Tokenizer
    rows = proposal[0]
    path = tmp_path / 'other/forms/20260927-birhor-living.csv'
    path.parent.mkdir(parents=True)
    with path.open('w', newline='') as f:
        csv.writer(f).writerows(rows)
    meta = source_meta.SourceMeta([OUT / '20260927-birhor-living.yaml'])
    errors = io.StringIO()
    with patch.object(source_meta, 'load', return_value=meta), patch.dict(
            make_cldf.convertors, {'birhor-living': Tokenizer(str(OUT / 'proposal-profile.txt'))}):
        parsed, stats = make_cldf.parse_file(str(path), errors, name='20260927-birhor-living')
    assert not errors.getvalue()
    assert len(parsed) == len(rows)
    raw = {r[10]: r for r in rows}
    assert {r.entry_key for r in parsed} == set(raw)
    for r in parsed:
        source = raw[r.entry_key]
        assert (r.old_form, r.native, r.ipa, r.notes, r.source, r.gloss) == tuple(
            source[i] for i in (2, 4, 5, 6, 7, 3))
        assert set(source[14].split()) <= set(r.tags.split())


def test_all_citations_resolve_and_recovered_heads_keep_source_roles(proposal):
    from pybtex.database import parse_file
    bib = parse_file(str(OUT / 'proposal-sources.bib'))
    rows, audit, _ = proposal
    used = {token.split('[', 1)[0] for row in rows for token in row[7].split(';')}
    assert used == set(bib.entries)
    assert 'living dictionary is allowed per sarva' in bib.entries['living-birhor2026'].fields['note']
    by_key = {r[10]: r for r in rows}
    for a in audit:
        if a['section'] == 'structured-head-and-senses':
            required = {r['citation'] for r in a['source_roles'] if r['citation']}
            for key in a['entry_keys']:
                assert required <= set(by_key[key][7].split(';'))
