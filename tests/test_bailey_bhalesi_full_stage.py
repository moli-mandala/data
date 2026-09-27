"""Focused whole-chapter Bhalesi staging checks; no compiled build."""
import csv
import importlib.util
import io
import json
import unicodedata
from pathlib import Path
from segments import Tokenizer

DATA = Path(__file__).resolve().parents[1]
P = DATA / 'data/other/forms/raw_data/bailey_bhalesi_1908'
spec = importlib.util.spec_from_file_location('bhalesi_full', P / 'prepare_full.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def rows():
    return list(csv.reader((P / 'proposal.csv').open()))


def test_full_scope_and_legacy_keys():
    actual, audit = module.generate()
    assert actual == rows()
    assert len(actual) == 435 and len(audit) == 436
    assert sum(a['status'] == 'ingested' for a in audit) == 410
    assert sum(a['status'] == 'other-lect-control' for a in audit) == 7
    assert sum(a['status'] == 'bound-morphology-audit-only' for a in audit) == 19
    keys = {r[10] for r in actual}
    assert len(keys) == len(actual)
    assert {k for a in audit for k in a['entry_keys']} == keys
    assert all(not r[11] or r[11] in keys for r in actual)
    old = list(csv.reader((P / 'legacy-pilot/20260925-bailey-bhalesi.csv').open()))
    assert len(old) == 16 and {r[10] for r in old} <= keys
    assert len([a for a in audit if a['section'] == 'sentence']) == 22
    assert len([a for a in audit if a['section'] == 'glossary']) == 34


def test_fine_marks_and_literal_variation():
    r = {x[10]: x for x in rows()}
    prefix = 'bailey1908bhalesi:'
    assert r[prefix + 'p71:become:pres-ind'][2] == 'bhō̃tan'
    assert r[prefix + 'p71:come:pres-subj-5'][2] == 'ēith'
    assert r[prefix + 'p71:come:pres-subj-6'][2] == 'ēīn'
    assert r[prefix + 'p72:beat:pres-ind-1'][2] == 'kuṭtan'
    assert r[prefix + 'p72:beat:pres-ind-3'][2] == 'kuṭtau'
    assert r[prefix + 'p74:right:item:26'][2] == 'tshĕrṛō'
    assert r[prefix + 'p74:left:item:15'][2] == unicodedata.normalize('NFC', 'kue͞ũṅs̲h̲')
    assert 'uncertain' in r[prefix + 'p74:left:item:15'][14].split()
    assert 'whole eu span' in r[prefix + 'p74:left:item:15'][6]
    assert r[prefix + 'p70:adjectival-series:interr-2'][2] == 'kuthur'
    assert r[prefix + 'p74:sentence:17'][2].endswith('bannhath.')
    assert r[prefix + 'p72:beat:participle-active'][2] == 'kuṭtau'
    assert r[prefix + 'p72:beat:participle-past'][2] == 'kuṭṭō'
    assert r[prefix + 'p71:fall:fut-4'][2] == 'khirkkamal'


def test_source_typography_and_grammatical_labels():
    r = {x[10]: x for x in rows()}
    prefix = 'bailey1908bhalesi:'
    assert 'shortened eu' in r[prefix + 'p71:fall:fut-f-1'][6]
    assert 'italic within roman' in r[prefix + 'p73:right:item:12'][6]
    assert 'sentential' in r[prefix + 'p75:sentence:22'][14].split()
    assert r[prefix + 'p75:sentence:22'][2].endswith('.')
    assert 'erg' in r[prefix + 'p69:pronoun-1sg:agent'][14].split()
    assert 'Agent case' in r[prefix + 'p69:pronoun-1sg:agent'][6]
    assert 'ipfv' in r[prefix + 'p71:fall:imperf-f-pl'][14].split()
    assert 'progressive' not in r[prefix + 'p71:fall:imperf-f-pl'][14].split()
    assert 'probably' in r[prefix + 'p73:participle-prose:eat-stative'][6]
    assert not set(r[prefix + 'p71:fall:pres-ind-1'][14].split()) & {'1sg', '2sg', '3sg', '1pl', '2pl', '3pl'}


def test_shared_source_context_and_variation():
    r = {x[10]: x for x in rows()}
    prefix = 'bailey1908bhalesi:'
    assert r[prefix + 'p28:shared-comparison:horse-pl-gen'][2] == 'ghōṛ kēū'
    assert r[prefix + 'p53:shared-introduction:say'][2] == 'dzāṇū'
    assert r[prefix + 'p72:say:inf'][2] == 'dzōṇu'
    assert 'Source explicitly compares' in r[prefix + 'p53:shared-introduction:say'][6]
    assert 'general source claim' in r[prefix + 'p71:fall:fut-1'][6]
    assert 'interr' in r[prefix + 'p54:shared-introduction:where-interrogative'][14].split()
    assert 'relative' in r[prefix + 'p54:shared-introduction:where-relative'][14].split()


def test_profile_tags_and_citation_parser():
    import tags
    from make_refs import source_ids
    tokenizer = Tokenizer(str(P / 'proposal-profile.txt'))
    for r in rows():
        assert len(r) == 15
        converted = tokenizer(r[2], column='IPA').replace(' ', '').replace('#', ' ')
        assert converted == r[2].replace('ṅ', 'ŋ').replace('.', '').replace('?', '')
        assert set(r[14].split()) <= tags.GRAMMATICAL_TAGS | tags.GENDER_TAGS
        assert set(source_ids(r[7])) == {'bailey1908bhalesi'}


def test_scoped_parser(monkeypatch):
    import make_cldf
    monkeypatch.setitem(make_cldf.convertors, 'bailey-bhalesi-1908', Tokenizer(str(P / 'proposal-profile.txt')))
    error = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(P / 'proposal.csv'), error, name='20260925-bailey-bhalesi')
    assert not error.getvalue(), error.getvalue()
    assert len(parsed) == stats['converted'] == 435
    assert {r.entry_key for r in parsed} == {r[10] for r in rows()}
