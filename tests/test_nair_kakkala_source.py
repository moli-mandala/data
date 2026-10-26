"""Whole Nair chapter candidate and installed source contracts."""
import csv
import hashlib
import importlib.util
import json
import re
import io
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / 'data/other/forms/raw_data/nair_kakkala_2013'

def records():
    return list(csv.reader((BASE / 'proposed.csv').open(newline='')))

def test_source_literals_keys_and_scope_are_preserved():
    old = list(csv.reader((BASE / 'revision1/proposed.csv').open(newline='')))
    new = records()
    assert len(new) == len(old) == 202
    for a, b in zip(old, new):
        assert [a[i] for i in (0,1,2,4,5,6,7,8,9,10,12,13)] == [b[i] for i in (0,1,2,4,5,6,7,8,9,10,12,13)]
    assert (BASE/'first-reading.jsonl').read_bytes() == (BASE/'revision1/first-reading.jsonl').read_bytes()
    assert (BASE/'proposal-audit.jsonl').read_bytes() == (BASE/'revision1/proposal-audit.jsonl').read_bytes()

def test_relative_verbal_and_negative_participle_scope():
    rows = {r[10]:r for r in records()}
    for key in ('p29:negative-markers:w1','p29:negative-examples:w1','p29:negative-examples:w2'):
        assert {'relative','participle','neg'} <= set(rows['nair2013kakkala:'+key][14].split())
    for key in ('p29:negative-markers:w2','p29:negative-examples:w3','p29:negative-examples:w4','p30:participle-other-examples:w1','p30:participle-other-examples:w2'):
        tags=set(rows['nair2013kakkala:'+key][14].split())
        assert {'neg','participle'} <= tags and 'relative' not in tags
    assert 'relative' not in rows['nair2013kakkala:p30:participle-zero-context:w1'][14].split()

def test_grammatical_gloss_labels_and_person_table_scope():
    rows={r[10]:r for r in records()}
    assert not any(re.search(r'\((sg|pl)\.?\)',r[3],re.I) for r in rows.values())
    for n in range(1,5):
        assert '2sg' in rows[f'nair2013kakkala:p30:imperative-singular-examples:w{n}'][14].split()
        assert '2pl' in rows[f'nair2013kakkala:p30:imperative-plural-examples:w{n}'][14].split()
    for n, tense in ((8,'pret'),(9,'pret'),(10,'pres'),(11,'pres'),(12,'fut'),(13,'fut')):
        assert tense in rows[f'nair2013kakkala:p31:personal-examples:w{n}'][14].split()
    # Temporal conditioning of the suffix itself is not a tense assertion.
    for n in (9,10):
        assert not {'pret','pres','fut'} & set(rows[f'nair2013kakkala:p31:personal-suffixes:w{n}'][14].split())

def test_only_two_reviewed_slash_variants_are_applied():
    rows={r[10]:r for r in records()}
    expected={f'nair2013kakkala:p28:pronoun-paradigm:w{n}:v2':f'nair2013kakkala:p28:pronoun-paradigm:w{n}:v1' for n in (9,11)}
    assert {k:r[11] for k,r in rows.items() if r[11]} == expected
    assert all(parent in rows and parent != child for child,parent in expected.items())
    assert all(not r[12] and not r[13] for r in rows.values())

def test_actual_parser_source_meta_and_formatter():
    spec=importlib.util.spec_from_file_location('kakkala_candidate_validation',BASE/'validate_candidate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.main()
    report=json.loads((BASE/'candidate-validation.json').read_text())
    assert report['csv_sha256']==hashlib.sha256((BASE/'proposed.csv').read_bytes()).hexdigest()
    assert all(c['rows']==202 and c['errors']==0 for c in report['checks'])

def test_installed_registered_parser_and_accounting():
    import make_cldf
    data=Path(__file__).resolve().parents[1]
    path=data/'data/other/forms/20260927-nair-kakkala.csv'
    assert path.read_bytes()==(BASE/'proposed.csv').read_bytes()
    errors=io.StringIO()
    parsed,stats=make_cldf.parse_file(str(path),errors,name='20260927-nair-kakkala')
    assert not errors.getvalue() and stats=={'converted':0,'for_conversion':0}
    fields=('lang','param','old_form','gloss','native','ipa','notes','source','cognateset','etymology','entry_key','variant_of_key','borrowed_from_key','derivation_parent_keys','tags')
    assert len(parsed)==202
    import source_meta
    meta=source_meta.load()
    assert len({make_cldf.source_entry_dedupe_key(p,meta) for p in parsed})==202
    for p,r in zip(parsed,records()):
        assert [getattr(p,f) for f in fields]==r,r[10]
        assert p.form==r[2]
    audit=[json.loads(x) for x in (BASE/'installed-audit.jsonl').read_text().splitlines()]
    assert len(audit)==41 and sum(len(a['physical_witnesses']) for a in audit)==207
    assert all(a['accepted_for_source_stage'] for a in audit)
    assert {r[10] for r in records()}=={k for a in audit for k in a['entry_keys']}

def test_installed_references_dialect_and_source_flags():
    import source_meta
    from pybtex.database import parse_file
    data=Path(__file__).resolve().parents[1]
    meta=source_meta.load()
    for key in ('nair2013kakkala','azad1980kakkala','sheela1980kakkala'):
        assert meta.flag(key,'reference','ocr') is False
        assert meta.flag(key,'reference','editor')=='OpenAI Codex'
        assert meta.flag(key,'reference','etymology_provenance')==('source' if key=='nair2013kakkala' else 'none')
    assert meta.flag('nair2013kakkala','identity','dedupe_by_entry_key') is True
    assert meta.transcription('nair2013kakkala','20260927-nair-kakkala.csv','Kakkala')==(None,False)
    bib=parse_file(data/'cldf/sources.bib')
    assert {'nair2013kakkala','azad1980kakkala','sheela1980kakkala'}<=set(bib.entries)
    rows=[r for r in csv.DictReader((data/'cldf/dialects.csv').open()) if r['ID']=='nair2013_kakkala_kerala']
    assert len(rows)==1 and rows[0]['Language_ID']=='Kakkala'
    assert rows[0]['Glottocode']==rows[0]['Latitude']==rows[0]['Longitude']==''
    langs={r['ID']:r for r in csv.DictReader((data/'cldf/languages.csv').open())}
    assert {r[0] for r in records()}<=langs.keys() and langs['Kakkala']['Glottocode']=='kakk1234'

def test_installed_guarded_repeat_noop():
    spec=importlib.util.spec_from_file_location('kakkala_installed',BASE/'import_source.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    result=mod.main(True)
    assert result['canonical_mutations'] is False and result['status'].startswith('already installed')
