"""Source coverage, diplomatic transcription, grammar, and stable graph identity."""
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import random
import unicodedata
from pathlib import Path

import pytest
from assign_form_ids import assign_ids
from make_cldf import parse_file
from segments import Tokenizer
from coordinate_policy import assert_reviewed_or_blank, assert_reviewed_point

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/other/forms/raw_data'

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj)
    return obj

v=module('varli2003',RAW/'dadra_varli_2003.py')
g=module('ghatage_western',RAW/'ghatage_western/import_glossaries.py')
SOURCES=[('dadra-varli',v.SOURCE,816,'dadra-varli'),
         ('ghatage-konkani','ghatage-konkani1963',1389,'ghatage-western'),
         ('ghatage-kudali','ghatage-kudali1965',1902,'ghatage-western')]

def installed(name):
    with (RAW.parent/f'20260911-{name}.csv').open() as f:return list(csv.reader(f))

def test_varli_every_prompt_and_control_is_accounted_for():
    rows,audit=v.build()
    assert rows==installed('dadra-varli')
    assert len(audit)==2484 and len(rows)==816
    assert sum(a['status']=='source-blank' for a in audit)==53
    assert sum(a['status']=='excluded-control' for a in audit)==1656
    assert sum(a['status']=='ingested' for a in audit)==775
    assert {a['item'] for a in audit}==set(range(1,415))
    assert all(a['pdf_page']==a['printed_page']+14 for a in audit)
    assert {a['column'] for a in audit if a['status']=='excluded-control'}=={5,6,7,8}
    assert [r[0] for r in rows].count('Bhili')==430
    assert [r[0] for r in rows].count('Varli')==386
    assert all(not any(r[i] for i in [1,4,5,6,8,9,11,12,13]) for r in rows)
    possessive=next(r for r in rows if ':i403:davar' in r[10])
    assert possessive[2]=='tumco' and {'m','pl'}<=set(possessive[14].split())
    assert all('uncertain' in r[14] for r in rows if any(c in r[2] for c in 'LC'))

@pytest.mark.parametrize('name,count,raw_count', [('konkani',1389,1439),('kudali',1902,2034)])
def test_ghatage_complete_reproducible_pinned_input(name,count,raw_count):
    rows,audit=g.build(name)
    assert rows==installed('ghatage-'+name) and len(rows)==count
    assert len(audit)==raw_count
    assert sum(len(a['emitted']) for a in audit)==count
    assert all(a['raw_record']['raw_words'] for a in audit)
    for a in audit:
        assert a['raw_record']['pdf_page']-a['raw_record']['printed_page']==g.VOLUMES[name]['offset']
        if a['review']==['ocr:unreviewed']:
            assert all({'ocr-review','uncertain'}<=set(e['tags']) for e in a['emitted'])
        if not a['emitted']: assert a['status']=='excluded-unresolved-ocr'
    manifest=json.loads((g.HERE/f'{name}-manifest.json').read_text())
    assert hashlib.sha256((g.HERE/f'{name}-ocr.json.gz').read_bytes()).hexdigest()==manifest['input_sha256']
    assert hashlib.sha256((g.HERE/'corrections.json').read_bytes()).hexdigest()==manifest['corrections_sha256']
    with gzip.open(g.HERE/f'{name}-ocr.json.gz','rt') as f:pages=json.load(f)
    first,last=g.VOLUMES[name]['pages']
    assert sorted(map(int,pages))==list(range(first,last+1))
    assert len({json.dumps(p['words'],sort_keys=True) for p in pages.values()})==len(pages)

def test_pos_scope_and_capital_morphophonemes():
    assert g.split_heads('koyti F. koyto M.','kudali')==[
        ('koyti',['noun','f']),('koyto',['noun','m'])]
    assert g.split_heads('x M. F.','kudali')==[('x',['noun','m','f'])]
    assert g.split_heads('aK F.','kudali')==[('aK',['noun','f'])]
    assert g.split_heads('aC Adv.','kudali')==[('aC',['adv'])]
    assert g.split_heads('foo N. unexplained','kudali')==[]
    assert not g.permissible('sərə]') and not g.permissible('$E š s”')
    assert g.permissible('aK') and g.permissible('čəǰə')

def test_column_boundaries_wraps_and_first_last_pages():
    first=g.parse('konkani',128)
    assert len(first)==51
    atti=next(r for r in first if r['col']==1 and r['ordinal']==23)
    assert atti['gloss']=='cooking pot'  # pot must not spill into second head column
    assert first[-1]['printed_page']==120
    middle=g.parse('konkani',135)
    assert next(r for r in middle if r['col']==1 and r['ordinal']==31)['gloss']=='to decide'
    assert next(r for r in middle if r['col']==1 and r['ordinal']==32)['left']=='thikki'
    assert next(r for r in g.parse('konkani',141) if r['col']==2 and r['ordinal']==33)['gloss']=='to fry (with- out oil)'
    assert len(g.parse('kudali',105))==32
    assert len(g.parse('kudali',160))==20  # exclude library stamp below the vocabulary
    last=installed('ghatage-kudali')[-1]
    assert last[2:4]==['hovri','room']

@pytest.mark.parametrize('name,source,count,profile',SOURCES)
def test_profile_covers_every_form_and_preserves_evidence(name,source,count,profile):
    errors=io.StringIO();rows,stats=parse_file(str(RAW.parent/f'20260911-{name}.csv'),errors)
    assert not errors.getvalue()
    assert len(rows)==count and stats=={'for_conversion':count,'converted':count}
    original={r[10]:r[2] for r in installed(name)}
    assert {r.entry_key:r.old_form for r in rows}==original
    assert all(not r.ipa and not r.native for r in rows)
    tokenizer=Tokenizer(str(ROOT/f'conversion/{profile}.txt'))
    for form in original.values():
        for norm in ['NFC','NFD']:
            assert '�' not in tokenizer(unicodedata.normalize(norm,form),column='IPA')
    assert all(len(r)==15 and r[2] and source+'[p.' in r[7] for r in installed(name))
    assert len({r[10] for r in installed(name)})==count
    assert all(unicodedata.is_normalized('NFC',value) and '�' not in value for r in installed(name) for value in r)

def test_distinct_affricates_and_undefined_codes_are_not_conflated():
    def convert(profile,form):
        t=Tokenizer(str(ROOT/f'conversion/{profile}.txt'))
        return unicodedata.normalize('NFC',t(form,column='IPA').replace(' ','').replace('#',' '))
    assert convert('ghatage-western','čə cə ǰə jə')=='čə cə ǰə jə'
    assert convert('ghatage-western','aK aC laK')=='aK aC laK'
    assert convert('ghatage-western','ka:ḷmi:ri')=='kāḷmīri'
    assert convert('dadra-varli','DhOg')=='ḍhɔg'
    assert convert('dadra-varli','pã:c')=='pā̃c'
    assert convert('dadra-varli','patOL phukoCO')=='patɔL phukoCɔ'

def test_language_dialect_assignment_and_no_invented_map_points():
    dialects={r['Tag']:r for r in csv.DictReader((ROOT/'cldf/dialects.csv').open())}
    languages={r['ID']:r for r in csv.DictReader((ROOT/'cldf/languages.csv').open())}
    assert languages['Varli']['Glottocode']=='varl1238'
    assert languages['Varli']['Clade']=='Marathi-Konkani'
    assert (languages['Varli']['Latitude'],languages['Varli']['Longitude'],languages['Varli']['Quality'])==('20.5635','73.2975','C')
    for name,*_ in SOURCES:
        for row in installed(name):
            tags=[t for t in row[14].split() if t.startswith('dialect:')]
            assert len(tags)==1
            d=dialects[tags[0]]
            assert d['Language_ID']==row[0] and d['Location']
            assert_reviewed_or_blank(d)
    assert dialects[v.dialect_tag('Davar')]['Glottocode']=='dava1244'

def test_durable_ids_survive_corrections_and_reordering():
    rows=[r for name,*_ in SOURCES for r in installed(name)]
    initial=[dict(ID=f'tmp-{i}',Language_ID=r[0],Original=r[2],Form=r[2],Gloss=r[3],
                  Native='',Source=r[7],Status='unlinked') for i,r in enumerate(rows)]
    keys={row['ID']:r[10] for row,r in zip(initial,rows)}
    first,registry=assign_ids(initial,[],keys)
    corrected=[dict(r,ID='new-'+r['ID'],Original='corrected',Form='corrected',Gloss='corrected') for r in reversed(initial)]
    second,_=assign_ids(corrected,registry,{r['ID']:keys[r['ID'][4:]] for r in corrected})
    assert len(set(first.values()))==len(rows)
    assert all(second['new-'+old]==fid for old,fid in first.items())

@pytest.mark.parametrize('name,source,count,profile',SOURCES)
def test_compiled_source_records_references_and_unlinked_graph(name,source,count,profile):
    expected={r[10]:r for r in installed(name)}
    aliases={r['Legacy_ID']:r['Form_ID'] for r in csv.DictReader((ROOT/'cldf/form-id-aliases.csv').open())}
    keys={r['Source_Key']:aliases[r['Legacy_ID']] for r in csv.DictReader((ROOT/'cldf/form-source-keys.csv').open()) if r['Source_Key'].startswith(source+':')}
    assert set(keys)==set(expected)
    forms={r['ID']:r for r in csv.DictReader((ROOT/'cldf/forms.csv').open()) if source in r['Source']}
    assert len(forms)==count
    for key,fid in keys.items():
        r=forms[fid];raw=expected[key]
        assert r['Original']==raw[2] and r['Language_ID']==raw[0]
        assert r['Status']=='unlinked' and source in r['Source']
        assert not r['Phonemic'] and not r['Native']
    assert not any(r['Child_ID'] in forms for r in csv.DictReader((ROOT/'cldf/edges.csv').open()))
    refs=[r for r in csv.DictReader((ROOT/'cldf/references.csv').open()) if r['ID']==source]
    assert len(refs)==1
