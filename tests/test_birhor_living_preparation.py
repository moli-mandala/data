"""Focused preparation checks; no source installation or database build."""
import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'data/other/forms/raw_data/birhor_living_2026'

def importer():
    spec=importlib.util.spec_from_file_location('birhor_senses',ROOT/'prepare_senses.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def test_numbered_senses_and_grammar_stay_local():
    m=importer()
    assert m.senses('1. split (tr.vb.) 2. saw (vb.)')==[(1,'split (tr.vb.)'),(2,'saw (vb.)')]
    assert m.clean_grammar('split (tr.vb.)')==('split',['verb','tr'])
    assert m.clean_grammar('saw (vb.)')==('saw',['verb'])
    assert m.clean_grammar('pulp (food)')==('pulp (food)',[])
    assert m.clean_grammar('<i>Mucuna pruriens</i> (L.) DC.')==('Mucuna pruriens (L.) DC.',[])
    assert m.clean_grammar('spatter(ed)')==('spatter(ed)',[])
    import pytest
    with pytest.raises(ValueError):m.senses('1. first 3. third')

def test_every_entry_is_accounted_for_without_invented_variants():
    rows,audit=importer().prepare()
    assert len(rows)==3097 and len(audit)==2746
    assert len({r['entry_key'] for r in rows})==3097
    assert sum(len(r['candidate_keys']) for r in audit)==3097
    assert sum(len(r['candidate_keys'])>1 for r in audit)==259
    assert sum(bool(r['issues']) for r in audit)==29
    assert sum('transcription:headword-IPA-disagreement' in r['issues'] for r in audit)==23
    special=[r for r in rows if r['entry_id']=='lV4SFRDZBI0R4KykWpwJ']
    assert len(special)==1 and special[0]['source_headword']=='babat buna (alkusi)'
    assert special[0]['source_ipa']=='babat buna (alkusi)' and special[0]['review']
    mismatch=next(r for r in rows if r['entry_id']=='0SpouO6c5VXFt3rbWSGY')
    assert mismatch['source_headword']=='hudʒuʔraʔ din'
    assert mismatch['source_ipa']=='hidʒuʔraʔ din'
    assert all(r['native'] and r['source_ipa'] for r in rows)
    missing=[r for r in rows if not r['gloss']]
    assert len(missing)==1 and missing[0]['entry_id']=='Rh3p0zj9xcsBp3YfJbLN'
    assert 'gloss:missing-English-gloss' in missing[0]['review']


def row_importer():
    import sys
    sys.path.insert(0,str(ROOT))
    spec=importlib.util.spec_from_file_location('birhor_rows',ROOT/'import_source.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def test_proposed_rows_preserve_transcription_layers_and_resolve_variant_keys():
    rows,audit,refs=row_importer().build()
    assert len(rows)==3100 and len(audit)==2746
    keys={r[10] for r in rows}
    assert len(keys)==3100
    assert sum(bool(r[11]) for r in rows)==6
    assert all(not r[11] or r[11] in keys for r in rows)
    assert sum(r['status']=='withheld-lexical-scope' for r in audit)==2
    assert sum(r['status']=='withheld-missing-English-gloss' for r in audit)==1
    assert sum('uncertain' in r[14].split() for r in rows)==25
    assert all(len(r)==15 and not any(r[i] for i in (1,8,9,12,13)) for r in rows)
    for r in rows:
        assert 'alkusi' not in r[5] and '(' not in r[5] and ')' not in r[5]
        assert not 'photo-source' in r[7]
    mismatch=next(r for r in rows if ':0SpouO6c5VXFt3rbWSGY:' in r[10])
    assert mismatch[2]=='hudʒuʔraʔ din' and mismatch[5]=='hidʒuʔraʔ din'
    optional=[r for r in rows if ':ST6m46b9cgDXJrKKkdkK:' in r[10]]
    assert {r[5] for r in optional}=={'masari dajl','masri dajl'}
    assert {r[2] for r in optional}=={'mas(a)ri dajl'}
    assert len(refs)==91
    assert sum(r['role']=='botanical-identification' for r in refs.values())==89


def test_profile_preserves_glottal_stop_nasality_and_short_vowels():
    import unicodedata
    from segments.tokenizer import Tokenizer
    tok=Tokenizer(str(ROOT/'birhor-living.txt'))
    def convert(value):return unicodedata.normalize('NFC',tok(value,column='IPA').replace(' ','').replace('#',' '))
    assert convert('oɽaʔ')=='oṛaʔ'
    assert convert('dʒiliŋ')=='jiliŋ'
    assert convert('pʰuɽuʔ roʔem')=='phuṛuʔ roʔem'
    assert convert('iɲaʔ')=='iñ aʔ'.replace(' ','')
    assert convert('a i u')=='a i u'
    rows,_,_=row_importer().build()
    for r in rows:
        assert '�' not in convert(r[5])
        assert convert(r[5])==convert(unicodedata.normalize('NFD',r[5]))


def test_explicit_pronoun_features_and_bibliography_are_resolved():
    rows,_,_=row_importer().build()
    for eid,tags in [('lwXFNuxotWcYelxejTRK','pron 1pl inclusive'),('xu42TGs9OC7UBblxQHld','pron 1pl exclusive'),('Ys9ZdxLkHyGKofQgidbI','pron 2sg'),('gQJpNjnoP88Xx2on9uM1','pron interr pl')]:
        row=next(r for r in rows if ':'+eid+':' in r[10])
        assert row[14]==tags
    from pybtex.database import parse_file
    bib=parse_file(str(ROOT/'sources.bib'))
    used={token.split('[',1)[0] for row in rows for token in row[7].split(';')}
    assert used==set(bib.entries)=={'living-birhor2026','lakra-etal2022birhor-ethnobotany','tropicos'}


def test_actual_parser_preserves_all_source_layers(tmp_path):
    import csv,sys
    sys.path.insert(0,str(ROOT))
    from audit_output import parse_draft
    path=tmp_path/'other/forms/20260922-birhor-living.csv';path.parent.mkdir(parents=True)
    rows,_,_=row_importer().build()
    with path.open('w',newline='') as f:csv.writer(f).writerows(rows)
    parsed,stats=parse_draft(path)
    assert len(parsed)==3100 and stats=={'converted':3100,'for_conversion':3100}
    originals={r[10]:r for r in rows}
    assert {r.entry_key for r in parsed}==set(originals)
    for r in parsed:
        raw=originals[r.entry_key]
        assert (r.old_form,r.native,r.ipa,r.notes,r.source,r.tags,r.gloss)==tuple(raw[i] for i in (2,4,5,6,7,14,3))
        assert r.lang=='Birhor' and r.is_lone and not r.param
