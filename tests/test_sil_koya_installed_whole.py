"""Accepted Koya whole-source installation and exact source-key reconciliation."""
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import unicodedata as ud
import yaml

ROOT=Path(__file__).parents[1]
P=ROOT/'data/other/forms/raw_data/sil_koya_2021'
W=P/'whole-source-recovery'
F=W/'literal-integration-metadata-r2'
STEMS={'wordlist':'20260828-sil-koya','narrative':'20260927-sil-koya-narrative','comparative':'20260927-sil-koya-comparative'}
def rows(path):
    with path.open(newline='') as f:return list(csv.reader(f))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def test_guarded_installation_reproduces_accepted_bytes_and_source_keys():
    spec=importlib.util.spec_from_file_location('koya_guarded',P/'install_whole_source.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.verify_frozen()
    allrows=[]
    for kind,stem in STEMS.items():
        current=ROOT/f'data/other/forms/{stem}.csv'
        assert current.read_bytes()==(F/f'proposed-{kind}.csv').read_bytes()
        allrows+=rows(current)
    assert len(allrows)==len({r[10] for r in allrows})==1665
    active={r[10] for r in allrows}
    historical={r[10] for r in rows(P/'historical-wordlist-installation/20260828-sil-koya.csv')}
    assert historical-active=={'silkoya1985:utn:i198:v2'}
    assert len(historical&active)==1437
    assert {'silkoya1985:utn:i198:v1','silkoya1985:jag:i198:v1'}<=active
    alias=ROOT/'data/other/form_aliases/20260927-koya.csv'
    assert alias.read_bytes()==(W/'durable-alias-review-20260927/proposed-source-key-alias.csv').read_bytes()
    assert (P/'whole-audit.jsonl').read_bytes()==(F/'proposed-whole-audit.jsonl').read_bytes()
    assert len((P/'whole-audit.jsonl').read_text().splitlines())==2119


def test_actual_canonical_parser_preserves_source_fields_graph_and_display():
    sys.path.insert(0,str(ROOT))
    import make_cldf
    meta=make_cldf.source_meta.load();keys=set();edges=[]
    for kind,stem in STEMS.items():
        path=ROOT/f'data/other/forms/{stem}.csv';raw=rows(path)
        errors=io.StringIO();parsed,stats=make_cldf.parse_file(str(path),errors)
        assert not errors.getvalue() and stats['converted']==len(raw)==len(parsed)
        for actual,expected in zip(parsed,raw):
            assert (actual.entry_key,actual.old_form,actual.notes,actual.source,actual.tags,actual.variant_of_key)==(expected[10],expected[2],expected[6],expected[7],expected[14],expected[11])
            assert actual.ipa=='' and make_cldf.source_entry_dedupe_key(actual,meta)==expected[10]
            keys.add(actual.entry_key)
            if actual.variant_of_key:edges.append((actual.entry_key,actual.variant_of_key))
            if kind=='wordlist':assert ud.normalize('NFC',actual.form)==ud.normalize('NFC',expected[2])
            else:
                display=expected[2].strip('1234⁴5⁵67⁷,;.')
                display=display.replace('G','ɣ') if kind=='comparative' else display.replace('[stutter]','')
                assert ' '.join(actual.form.split())==' '.join(display.split())
    assert len(keys)==1665 and len(edges)==36
    assert all(a in keys and b in keys for a,b in edges)


def test_metadata_profiles_reference_and_identity_routing():
    from segments.tokenizer import Tokenizer
    import pybtex.database
    for kind,stem in STEMS.items():
        actual=yaml.safe_load((ROOT/f'data/other/forms/{stem}.yaml').read_text())
        importer=actual['defaults'].pop('importer')
        assert importer['commands']==[['data/other/forms/raw_data/sil_koya_2021/install_whole_source.py','--install']]
        assert actual==yaml.safe_load((F/f'proposed-{kind}.yaml').read_text())
        profile='literal' if kind=='wordlist' else kind
        path=ROOT/f'conversion/sil-koya-{profile}.txt'
        assert path.read_bytes()==(F/f'proposed-sil-koya-{profile}.txt').read_bytes()
        tokenizer=Tokenizer(str(path))
        for row in rows(ROOT/f'data/other/forms/{stem}.csv'):
            for normalization in ['NFC','NFD']:
                assert '�' not in tokenizer(ud.normalize(normalization,row[2]),column='IPA')
    entry=pybtex.database.parse_file(str(ROOT/'cldf/sources.bib')).entries['devagnanavaram-et-al2021koya']
    proposal=json.loads((F/'bibliography-update-proposal.json').read_text())
    assert all(entry.fields[k]==v for k,v in proposal['fields'].items())
    historical=pybtex.database.parse_file(str(P/'historical-wordlist-installation/source-reference.bib')).entries['devagnanavaram-et-al2021koya']
    assert entry.persons==historical.persons
    for field in proposal['unchanged_fields']:
        if field!='author':assert entry.fields[field]==historical.fields[field]
