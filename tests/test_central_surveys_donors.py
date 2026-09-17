import csv, copy, importlib.util, json, unicodedata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/other/params/raw_data'
P=ROOT/'curation/etymology-lab/central-surveys-20260911'
def records():
    return json.loads((RAW/'20260911-central-surveys-donors-audit.json').read_text())
def test_source_heads_reproduce_and_have_metadata():
    s=importlib.util.spec_from_file_location('central_donors',RAW/'central_surveys_donors.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    assert m.OUTPUT.read_bytes()==m.render()
    langs={r['ID'] for r in csv.DictReader((ROOT/'cldf/languages.csv').open())}
    bib=(ROOT/'cldf/sources.bib').read_text()
    for r in records():
        assert r['Language_ID'] in langs
        assert r['Form']==unicodedata.normalize('NFC',r['Form'])
        assert all(r[k] for k in ['Entry_Key','Native','Original','SourceExcerpt','Transcription'])
        for citation in r['Source'].split(';'):assert '{'+citation.split('[')[0]+',' in bib
def test_semantics_and_identity():
    from assign_form_ids import assign_ids
    d={r['ID'].removeprefix('central-surveys-donor-'):r for r in records()}
    assert d['badan']['Gloss']=='body'
    assert d['aurat']['Gloss']=='private parts' and 'Urdu' in d['aurat']['Transcription']
    assert d['sabut']['Form']=='s̤ubūt' and 'entire' not in d['sabut']['Gloss']
    assert d['wazni']['Language_ID']=='Ar' and d['wazni']['Form']=='waznī'
    assert d['wazndar']['Language_ID']=='Pers' and d['wazndar']['Form']=='wazn-dār'
    assert d['phulgobi']['Form']=='pʰūlgobī'
    forms=[dict(ID=r['ID'],Language_ID=r['Language_ID'],Form=r['Form'],Original=r['Form'],Gloss=r['Gloss'],Source=r['Source'],Status='entry') for r in records()]
    reg=json.loads((P/'new-donor-identities.json').read_text())
    expected={r['ID']:r['Persistent_ID'] for r in records()}
    assert assign_ids(copy.deepcopy(forms),copy.deepcopy(reg))[0]==expected
    assert assign_ids([dict(r,Gloss=r['Gloss']+'; wording clarified') for r in reversed(forms)],copy.deepcopy(reg))[0]==expected
def test_approved_scope_and_nesting():
    a=json.loads((P/'approved-assignments.json').read_text())
    assert len(a)==3777 and len({r['Form_ID'] for r in a})==3731
    corrections=json.loads((P/'donor-nesting-corrections.json').read_text())
    assert sum(r['rows'] for r in corrections if r['language'] in {'Pers','Ar'})==223
    old={r['oldParent'] for r in corrections}
    assert not any(r['Etymon_ID'] in old for r in a)
    assert all(r['Kind']=='borrowed' for r in a if r['Etymon_ID'] in {x['newParent'] for x in corrections})
