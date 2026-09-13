import csv
import gzip
import importlib.util
import io
import json
import re
import unicodedata
from pathlib import Path
from functools import lru_cache
from make_cldf import parse_file, WESTERN_SURVEY_FILES
from segments import Tokenizer
from tags import GENDER_TAGS, GRAMMATICAL_TAGS

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('keed_import',ROOT/'data/other/forms/raw_data/keed_2018.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
@lru_cache(None)
def built():return m.build()

def test_complete_source_accounting_and_reproducibility():
    rows,audit,records=built()
    assert len(records)==28797 and len(audit)==31250 and len(rows)==43120
    assert sum(r['status']=='alphabet-heading' for r in records)==50
    assert {r['key'] for r in records if r['status']=='parse-review'}=={'keed2018:p698:c1:e21'}
    assert list(csv.reader(m.OUT.open()))==rows
    assert len({r[10] for r in rows})==len(rows)
    assert all(len(r)==15 and r[2] for r in rows)
    assert all('�' not in v and unicodedata.is_normalized('NFC',v) for r in rows for v in r)
    assert all(not any(0xe000<=ord(c)<=0xf8ff for c in v) for r in rows for v in r)
    assert all(not re.search('[①②③④⑤⑥⑦⑧⑨⑩{}]',r[2]) for r in rows)
    for r in rows:
        if r[4]:assert re.fullmatch(r'[\u0c80-\u0cff\s,/-]+',r[4])
    keys={r[10] for r in rows}
    assert all(key in keys for r in rows for i in (11,12,13) for key in r[i].split('|') if key)

def test_font_gloss_and_citation_regressions():
    rows,audit,_=built();bykey={r['key']:r for r in audit}
    assert bykey['keed2018:p12:c1:e5']['native']=='ಅಗಲಿಚು'
    assert bykey['keed2018:p133:c1:e8']['gloss']=='to spit'
    assert bykey['keed2018:p133:c1:e8']['extra_ipa']==['ulɪ̆ɡu']
    assert bykey['keed2018:p129:c2:e2']['gloss']==''
    assert 'My' not in bykey['keed2018:p67:c1:e9']['gloss']
    assert 'Hal.' not in bykey['keed2018:p909:c2:e3']['gloss']
    assert bykey['keed2018:p25:c1:e14']['gloss']=='uvula'
    assert 'cokka' not in bykey['keed2018:p370:c2:e3']['gloss']
    assert 'm' in bykey['keed2018:p709:c2:e17']['tags']
    assert 'pl' in bykey['keed2018:p704:c1:e3']['tags']
    assert 'caus' in bykey['keed2018:p661:c1:e10']['tags']
    assert 'dat' not in bykey['keed2018:p24:c1:e15:sub:4']['tags']
    assert bykey['keed2018:p120:c2:e7:sub:1']['ety']==['Sk.']
    assert 'IMP 4.267' in bykey['keed2018:p689:c1:e12']['aux_citations']
    assert 'pee-' not in bykey['keed2018:p871:c1:e14']['gloss']
    assert 'municipality' in bykey['keed2018:p375:c2:e13']['gloss']
    assert 'loanword' in bykey['keed2018:p613:c1:e2']['tags']
    assert 'loanword' in bykey['keed2018:p375:c2:e13']['tags']
    assert 'stem' in bykey['keed2018:p243:c1:e9']['tags']
    assert 'pejorative' in bykey['keed2018:p431:c1:e19']['tags']
    assert '?' not in bykey['keed2018:p431:c1:e19']['gloss']
    assert 'ಮುದ್ರೆ' not in bykey['keed2018:p745:c2:e6']['gloss']
    assert bykey['keed2018:p698:c1:e22']['native'].startswith('ಭೂಮಾಪನ, ಕಂದಾಯವ್ಯವಸ್ಥೆ')
    assert 'bʰūdākʰalegaḷa' in bykey['keed2018:p698:c1:e22']['form']
    assert bykey['keed2018:p589:c2:e5']['native']=='ಪರ್ರ್'
    assert bykey['keed2018:p825:c2:e17']['native']=='ಶರ್ಟ್'
    assert bykey['keed2018:p142:c1:e2']['form']=='r̥̄'
    assert 'ō̃' in bykey['keed2018:p332:c1:e15']['ety'][0]

def test_source_claims_and_relationships():
    rows,audit,_=built();bykey={r[10]:r for r in rows}
    assert bykey['keed2018:p718:c1:e5'][1]=='d4723'
    assert bykey['keed2018:p13:c2:e14'][12]
    donor=bykey[bykey['keed2018:p13:c2:e14'][12]]
    assert donor[0]=='Sk' and donor[1]=='8478' and donor[2]=='pragrahaṇa-'
    assert bykey['keed2018:p511:c2:e13'][12]
    donor=bykey[bykey['keed2018:p511:c2:e13'][12]]
    assert donor[0]=='H' and donor[1]=='6886'
    assert bykey['keed2018:p469:c1:e4:sub:1'][13]=='keed2018:p469:c1:e4'
    assert not bykey['keed2018:p25:c1:e14'][1]
    assert 'uncertain' in bykey['keed2018:p25:c1:e14'][14]
    assert bykey['keed2018:p430:c2:e10'][2]=='taḷar'
    assert bykey['keed2018:p430:c2:e10:head-variant:2'][2]=='taḷaru'
    assert bykey['keed2018:p430:c2:e10:head-variant:2'][5]=='təɭəru'
    assert all(not re.search(r'[0-9{}]',r[2]) for r in rows if ':donor:' in r[10])

def test_profiles_registry_and_boundaries():
    assert "data/other/forms/20260912-keed.csv" in WESTERN_SURVEY_FILES
    assert "data/other/forms/20260912-muduga.csv" in WESTERN_SURVEY_FILES
    errors=io.StringIO();parsed,stats=parse_file(str(m.OUT),errors)
    assert not errors.getvalue()
    assert stats['converted']==stats['for_conversion']==len(built()[0])
    assert len(parsed)==len(built()[0])
    bykey={r.entry_key:r for r in parsed}
    assert bykey['keed2018:p325:c2:e15'].form=='-ge'
    t=Tokenizer(str(ROOT/'conversion/keed.txt'))
    def convert(x):return unicodedata.normalize('NFC',t(x,column='IPA').replace(' ','').replace('#',' '))
    assert convert('ಅಗಲಿಚು')==convert('agalicu')=='agalicu'
    assert convert('ಅಂಕ')==convert('aṃka')
    assert convert('r̤')=='ṛ̆' and convert('r̥̄')=='ṝ'
    assert convert('ಕಲ್ದ-')=='kalda-'
    dialects={r['Tag']:r for r in csv.DictReader((ROOT/'cldf/dialects.csv').open())}
    languages={r['ID'] for r in csv.DictReader((ROOT/'cldf/languages.csv').open())}
    for r in built()[0]:
        assert r[0] in languages
        for tag in r[14].split():
            assert tag in GENDER_TAGS|GRAMMATICAL_TAGS or tag in dialects
            if tag in dialects:assert dialects[tag]['Language_ID']==r[0]
    assert any('dialect:Kannada:havyaka:' in r[14] for r in built()[0] if r[10]=='keed2018:p755:c2:e13')

def test_compiled_source_keys_and_references():
    keys={r['Source_Key']:r for r in csv.DictReader((ROOT/'cldf/form-source-keys.csv').open()) if r['Source_Key'].startswith('keed2018:')}
    assert set(keys)=={r[10] for r in built()[0]}
    refs={r['ID'] for r in csv.DictReader((ROOT/'cldf/references.csv').open())}
    for r in built()[0]:
        assert all(x.split('[',1)[0] in refs for x in r[7].split(';'))
