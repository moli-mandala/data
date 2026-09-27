"""Whole-source proposal invariants; no database construction."""
import csv
import json
from pathlib import Path

P=Path(__file__).resolve().parents[1]/'data/other/forms/raw_data/grierson_holiya_1906'

def inventory():
    return list(csv.reader((P/'proposal.csv').open())), [json.loads(x) for x in (P/'proposal-audit.jsonl').read_text().splitlines()]

def test_whole_source_scope_and_lect_boundaries():
    rows,audit=inventory()
    assert len(audit)==1282
    assert len({a['source_unit_key'] for a in audit})==1282
    assert {a['printed_page'] for a in audit}==set(range(386,396))
    assert sum(not a['entry_keys'] for a in audit)==37
    assert all(not a['entry_keys'] for a in audit if a.get('scope','holiya')!='holiya')
    by_key={r[10]:r for r in rows}
    for a in audit:
        for key in a['entry_keys']:
            assert key in by_key
            expected=f"dialect:Holiya:lsi1906-holiya:{a['site']}" if a['site'] else ''
            assert ([x for x in by_key[key][14].split() if x.startswith('dialect:')]==([expected] if expected else []))

def test_continuation_and_editorial_sic_are_not_conflated():
    rows,audit=inventory();a={x['source_unit_key']:x for x in audit};r={x[10]:x for x in rows}
    k='grierson1906holiya:p388:specimen:'
    assert a[k+'line8:word10']['entry_keys']==a[k+'line9:word1']['entry_keys']
    row=r[a[k+'line8:word10']['entry_keys'][0]]
    assert row[2:4]==['Khōlī-dā','room-in']
    assert 'line 8, word 10 and line 9, word 1' in row[7]
    first=a['grierson1906holiya:p392:specimen:line2:word9']
    second=a['grierson1906holiya:p392:specimen:line3:word1']
    assert first['entry_keys']!=second['entry_keys']
    assert r[first['entry_keys'][0]][2:4]==['Nin','His']
    assert r[second['entry_keys'][0]][2:4]==['nani','him']
    assert 'sic' in r[first['entry_keys'][0]][6]

def test_exact_reuse_and_graph_targets_are_closed():
    rows,audit=inventory();by_key={r[10]:r for r in rows}
    assert len(by_key)==len(rows)
    assert all(len(r)==15 for r in rows)
    for r in rows:
        if r[11]:
            assert r[11] in by_key and r[11]!=r[10]
            assert by_key[r[11]][0]==r[0] and by_key[r[11]][3]==r[3]
    for a in audit:
        if a.get('exact_attestation_reuse'):
            assert all(k in by_key for k in a['entry_keys'])
    assert all(not r[4] for r in rows)  # Native explicitly not printed.

def test_literal_profile_and_registered_grammar_coverage():
    from segments import Tokenizer
    from tags import GRAMMATICAL_TAGS,GENDER_TAGS
    rows,_=inventory();tokenizer=Tokenizer(str(P/'proposal-profile.txt'))
    for r in rows:
        assert '�' not in tokenizer(r[2],column='IPA')
        assert set(r[14].split())-{x for x in r[14].split() if x.startswith('dialect:')} <= GRAMMATICAL_TAGS|GENDER_TAGS
        assert 'Source transcription warning' in r[6]
    assert tokenizer('āū̃',column='IPA').replace(' ','')=='āū̃'


def test_scoped_parser_with_proposed_metadata(monkeypatch):
    import io
    import make_cldf,source_meta
    from segments import Tokenizer
    key='grierson-holiya-1906'
    meta=source_meta.SourceMeta([P/'20260926-grierson-holiya.yaml'])
    monkeypatch.setattr(source_meta,'load',lambda:meta)
    monkeypatch.setitem(make_cldf.convertors,key,Tokenizer(str(P/'proposal-profile.txt')))
    errors=io.StringIO()
    parsed,stats=make_cldf.parse_file(str(P/'proposal.csv'),errors,name='20260926-grierson-holiya')
    rows,_=inventory()
    assert len(parsed)==stats['converted']==len(rows)
    assert not errors.getvalue()
    assert {r.entry_key:r.old_form for r in parsed}=={r[10]:r[2]for r in rows}
    assert all(not r.ipa for r in parsed)
    tokenizer=make_cldf.convertors[key]
    assert {r.entry_key:r.form for r in parsed}=={r[10]:tokenizer(r[2],column='IPA').replace(' ','').replace('#',' ') for r in rows}


def test_installed_source_stage_matches_approved_freeze():
    import hashlib
    DATA=P.parents[4]
    report=json.loads((P/'independent-full-audit-20260926-pass1.json').read_text())
    assert report['status']=='passed' and report['material_errors']==0
    targets={'proposal.csv':DATA/'data/other/forms/20260926-grierson-holiya.csv','proposal-audit.jsonl':P/'audit.jsonl','proposal-profile.txt':DATA/'conversion/grierson-holiya-1906.txt'}
    for name,path in targets.items():assert hashlib.sha256(path.read_bytes()).hexdigest()==report['hashes'][name]
    registry={r['Tag']:r for r in csv.DictReader((DATA/'cldf/dialects.csv').open())}
    for site in ['Bhandara','Balaghat','Seoni']:
        d=registry[f'dialect:Holiya:lsi1906-holiya:{site}']
        assert d['Language_ID']=='Holiya' and not d['Latitude'] and not d['Longitude']
    import source_meta
    assert source_meta.SourceMeta().transcription('grierson1906holiya',targets['proposal.csv'],'Holiya')[0]=='grierson-holiya-1906'
