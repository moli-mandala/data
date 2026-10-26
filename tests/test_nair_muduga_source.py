"""Whole Muduga chapter proposal checks; canonical installation not performed."""
import csv
import importlib.util
import json
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]/'data/other/forms/raw_data/nair_muduga_2013'

def rows():
    return list(csv.reader((BASE/'proposed.csv').open(newline='')))

def test_whole_chapter_and_documentary_accounting():
    audit=[json.loads(x) for x in (BASE/'proposal-audit.jsonl').read_text().splitlines()]
    output=rows()
    assert len(audit)==128 and len(output)==204
    assert sum(len(u['physical_witnesses']) for u in audit)==229
    assert sum(not w['entry_keys'] for u in audit for w in u['physical_witnesses'])==25
    assert {k for u in audit for k in u['entry_keys']}=={r[10] for r in output}
    assert len({r[10] for r in output})==204
    assert sum(r[0]=='Muduga' for r in output)==183
    assert sum(r[0]=='Malayalam' for r in output)==21
    assert all(not r[4] and not r[5] for r in output)

def test_drawn_lowlines_are_not_encoded_underdots():
    bykey={r[10]:r for r in rows()}
    assert [bykey[f'nair2013muduga:p51:l-changes:{n}:w1'][2] for n in range(1,7)]==['mal̲a','vaal̲a','pal̲am','kal̲utti','kol̲uppu','kool̲i']
    assert bykey['nair2013muduga:p53:imperative-examples:w8'][2]=='el̲utiri'
    assert bykey['nair2013muduga:p51:l-changes:5:w2'][2]=='koḷuppu'
    geometry=json.loads((BASE/'scope/font-and-vector-census.json').read_text())
    assert len(geometry['filled_rectangle_lowlines'])==10

def test_affix_person_and_zero_scope():
    bykey={r[10]:r for r in rows()}
    assert all('verb' not in r[14].split() for r in bykey.values() if 'affix' in r[14].split())
    assert not any(r[2] in ('ø','–ø-') for r in bykey.values())
    for n in (2,4):
        tags=set(bykey[f'nair2013muduga:p54:personal-suffixes:w{n}'][14].split())
        assert tags >= {'suffix','pl'} and not {'first-person','1pl','exclusive','inclusive'}&tags
    for n,t in ((1,'1sg'),(3,'1pl'),(7,'2sg'),(9,'2pl'),(11,'3sg'),(15,'3pl')):
        assert t in bykey[f'nair2013muduga:p54:personal-examples:{n}:w1'][14].split()
    assert all(not r[11] and not r[12] and not r[13] for r in bykey.values())

def test_151_earlier_source_keys_remain_exact():
    legacy=[json.loads(x) for x in (BASE/'scope/legacy-baseline.jsonl').read_text().splitlines()]
    assert len(legacy)==151 and len({x['entry_key'] for x in legacy})==151
    forms=BASE.parents[1]
    actual={f:list(csv.reader((forms/f).open(newline=''))) for f in {r['file'] for r in legacy}}
    for r in legacy:
        assert actual[r['file']][r['row']-1]==r['fields']
    assert not {r['entry_key'] for r in legacy}&{r[10] for r in rows()}

def test_actual_parser_and_metadata_formatter():
    spec=importlib.util.spec_from_file_location('muduga_candidate_validation',BASE/'validate_candidate.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.main()
    report=json.loads((BASE/'candidate-validation.json').read_text())
    assert all(r['rows']==204 and r['exact_fields']==15 and r['errors']==0 for r in report['checks'])

def test_complete_printed_subject_metadata_and_tags_only_continuity():
    repairs=json.loads((BASE/'metadata-subject-repair.json').read_text())['decisions']
    assert len(repairs)==37
    keyed={r[10]:r for r in rows()}
    for d in repairs:
        assert keyed[d['entry_key']][14]==d['new_tags']
    for tail in ('p52:transitive-analysis:3:w2','p52:transitive-analysis:4:w1','p53:present-examples:2:w1','p53:present-examples:4:w1'):
        assert {'3sg','f'}<=set(keyed['nair2013muduga:'+tail][14].split())
    for n in range(1,9):
        assert ('2sg' if n<=2 else '2pl') in keyed[f'nair2013muduga:p53:imperative-examples:w{n}'][14].split()
    your=set(keyed['nair2013muduga:p51:genitive-examples:2:w1'][14].split())
    assert {'pron','second-person'}<=your and not {'sg','pl','2sg','2pl'}&your
    old=list(csv.reader((BASE/'revision1/proposed.csv').open(newline='')))
    current=rows()
    assert len(old)==len(current)==204
    assert [r[10] for r in old]==[r[10] for r in current]
    assert all(a[:14]==b[:14] for a,b in zip(old,current))
    assert sum(a[14]!=b[14] for a,b in zip(old,current))==37
    assert (BASE/'proposal-audit.jsonl').read_bytes()==(BASE/'revision1/proposal-audit.jsonl').read_bytes()
