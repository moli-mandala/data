"""Scoped first-reading and proposal checks; no canonical installation required."""
import csv,json,unicodedata
from pathlib import Path
import yaml
from pybtex.database import parse_file
from tags import GRAMMATICAL_TAGS,GENDER_TAGS
from source_meta import SourceMeta
P=Path(__file__).resolve().parent

def data():
    units=[json.loads(s) for s in (P/'physical-units-first-reading.jsonl').read_text().splitlines()]
    audit=[json.loads(s) for s in (P/'proposal-audit.jsonl').read_text().splitlines()]
    rows=list(csv.reader((P/'proposed.csv').open()))
    return units,audit,rows

def test_complete_chapter_census_and_dispositions():
    units,audit,rows=data()
    assert {u['printed_page'] for u in units}=={19,20,21}
    assert len(units)==33 and sum(len(u['source_forms']) for u in units)==116
    assert len(audit)==117 and len(rows)==104
    assert sum(a['disposition']=='audit-only' for a in audit)==12
    assert sum(a['disposition']=='documentary' for a in audit)==1
    assert {a['unit_key'] for a in audit}=={u['unit_key'] for u in units}
    assert len({a['witness_key'] for a in audit if a['witness_key']})==116
    assert len({r[10] for r in rows})==104

def test_source_scoped_grammar_controls():
    _,audit,rows=data();by={r[10]:r for r in rows}
    p='nair2013cholanaickan:p20:'
    assert by[p+'personal-pronouns:form1'][14]=='pron 1sg'
    assert by[p+'interrogatives:form1'][14]=='pron interr'
    assert by[p+'interrogatives:form7'][14]=='interr spatial'
    assert by['nair2013cholanaickan:p19:ablative-examples:form1'][14]=='adv abl'
    assert by['nair2013cholanaickan:p19:ablative-examples:form2'][14]=='noun abl'
    assert by['nair2013cholanaickan:p19:locative-examples:form2'][14]=='pron loc'
    assert by['nair2013cholanaickan:p19:plural-example:form1'][14]=='multiword-expression num'
    assert all(len(r[14].split())==len(set(r[14].split())) for r in rows)
    assert by[p+'interrogatives:form10'][3]=='when /now'
    assert p+'feminine-markers:form8' not in by
    assert p+'feminine-markers:form9' not in by
    assert 'comparative reconstruction' in next(a['reason'] for a in audit if a['witness_key']==p+'feminine-markers:form8')
    assert 'source-word' not in ' '.join(r[6] for r in rows)

def test_literal_profile_and_source_metadata():
    _,_,rows=data()
    assert all(len(r)==15 and r[0]=='Cholanaickan' and r[7].startswith('nair2013cholanaickan[') for r in rows)
    assert all(r[2] and unicodedata.is_normalized('NFC',r[2]) and '\ufffd' not in r[2] for r in rows)
    assert all(not r[4] and not r[5] and not r[12] and not r[13] for r in rows)
    graph=json.loads((P/'graph-decisions-proposal.json').read_text())
    expected={e['child']:e['parent'] for e in graph['proposed_variants']}
    assert len(expected)==11
    assert {r[10]:r[11] for r in rows if r[11]}==expected
    assert set(expected.values())<={r[10] for r in rows}
    assert all(set(r[14].split())<=(GRAMMATICAL_TAGS|GENDER_TAGS) for r in rows)
    y=yaml.safe_load((P/'20260927-nair-cholanaickan.yaml').read_text())
    assert y['defaults']['transcription']=={'convert':False,'preserve_hyphens':True}
    assert y['sources']['nair2013cholanaickan']['forms']['split_alternates'] is False
    assert y['sources']['nair2013cholanaickan']['reference']['ocr']=='No'
    assert 'nair2013cholanaickan' in SourceMeta([P/'20260927-nair-cholanaickan.yaml']).sources
    assert set(parse_file(str(P/'proposed-reference.bib')).entries)=={'nair2013cholanaickan','muralidharan1988cholanaikka'}

def test_source_anomalies_remain_reviewable():
    units,audit,rows=data();by={u['unit_key']:u for u in units}
    assert by['nair2013cholanaickan:p19:instrumental-markers']['source_forms']==['-indu','-indu']
    assert any(a['witness_key']=='nair2013cholanaickan:p19:instrumental-markers:form2' and a['disposition']=='audit-only' for a in audit)
    assert by['nair2013cholanaickan:p21:hortative-examples']['source_glosses'][1]=='left (someone) open'
    assert len([r for r in rows if 'personal-pronouns' in r[10]])==20
    assert len([r for r in rows if 'interrogatives' in r[10]])==12

def test_source_visible_low_lines_preserved_without_phonetic_claim():
    units,_,rows=data();by={u['unit_key']:u for u in units}
    assert by['nair2013cholanaickan:p19:accusative-example']['source_forms']==['kūsin̲a','kūsin̲e']
    assert by['nair2013cholanaickan:p19:genitive-examples']['source_forms']==['en̲n̲u','en̲n̲ule']
    assert by['nair2013cholanaickan:p20:personal-pronouns']['source_forms'][0]=='naan̲u'
    assert by['nair2013cholanaickan:p20:interrogatives']['source_forms'][1]=='een̲n̲ei'
    assert 'n̲' not in by['nair2013cholanaickan:p20:personal-pronouns']['source_forms'][1]
    assert sum(r[2].count('\u0332') for r in rows)==19
    assert all(r[5]=='' for r in rows)

def test_actual_installed_source_stage_and_scoped_parser():
    import hashlib
    import make_cldf
    data_root=next(p for p in P.parents if (p/'tags.py').exists())
    canonical=data_root/'data/other/forms/20260927-nair-cholanaickan.csv'
    assert canonical.read_bytes()==(P/'proposed.csv').read_bytes()
    assert (data_root/'data/other/forms/20260927-nair-cholanaickan.yaml').read_bytes()==(P/'20260927-nair-cholanaickan.yaml').read_bytes()
    receipt=json.loads((P/'installation-receipt.json').read_text())
    assert len(list(csv.reader(canonical.open())))==receipt['rows']==104
    assert len((P/'installed-audit.jsonl').read_text().splitlines())==receipt['audit_units']==117
    assert hashlib.sha256((data_root/'data/form-identities.csv').read_bytes()).hexdigest()==receipt['durable_identity_sha256']
    assert any(r['ID']=='Cholanaickan' and not r['Glottocode'] and not r['Latitude'] and not r['Longitude']
               for r in csv.DictReader((data_root/'cldf/languages.csv').open()))
    bibliography=parse_file(str(data_root/'cldf/sources.bib'))
    assert {'nair2013cholanaickan','muralidharan1988cholanaikka'}<=set(bibliography.entries)
    errors=[];parsed,stats=make_cldf.parse_file(str(canonical),errors)
    assert len(parsed)==104 and not errors and stats['converted']==0
    raw={r[10]:r for r in csv.reader(canonical.open())}
    assert {r.entry_key for r in parsed}==set(raw)
    assert all(r.form==raw[r.entry_key][2] and r.old_form==raw[r.entry_key][2]
               and r.variant_of_key==raw[r.entry_key][11] and not r.ipa for r in parsed)
