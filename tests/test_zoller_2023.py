"""Source contracts, including error classes found in the seeded PDF audits."""
import csv,gzip,importlib.util,io,json,sys,unicodedata
from pathlib import Path
import pytest
from make_cldf import parse_file
ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/other/forms/raw_data/zoller_2023'
sys.path.insert(0,str(RAW))
from parse import candidates
from build import build,direct_claim
from cache import load_records
from west_pahari import PROMOTIONS, resolve_language
OUT=ROOT/'data/other/forms/20260913-zoller-linguistic-data.csv'
@pytest.fixture(scope='module')
def rows():return list(csv.reader(OUT.open()))
@pytest.fixture(scope='module')
def records():return {x['key']:x for x in load_records()}

def test_counts_replay_keys_and_locators(rows):
    built,audit,recs,dialects=build()
    assert built==rows
    assert len(rows)==17754 and len(audit)==32470 and len(recs)==4377
    assert len({r[10] for r in rows})==len(rows)
    assert all(len(r)==15 and r[2] and r[7].startswith('zoller2023[p') for r in rows)
    assert all('�' not in v and unicodedata.is_normalized('NFC',v) for r in rows for v in r)
    assert sum(bool(r[1]) for r in rows)==341
    assert sum(bool(r[11]) for r in rows)==70
    assert sum(bool(r[8]) for r in rows)==7
    assert all(not r[11] or r[11] in {x[10] for x in rows} for r in rows)

def test_profile_and_layers(rows):
    err=io.StringIO();parsed,stats=parse_file(str(OUT),err)
    assert not err.getvalue() and stats['converted']==len(rows)==len(parsed)
    assert all(not r[5] and not r[6] for r in rows)
    assert all(not r[4] or (r[0]=='Gk' and r[4]==r[2]) for r in rows)
    assert [r.form for r in parsed]==[r[2] for r in rows]

def test_registered_languages_and_dialects(rows):
    languages={r['ID']:r for r in csv.DictReader((ROOT/'cldf/languages.csv').open())}
    dialects={r['Tag']:r for r in csv.DictReader((ROOT/'cldf/dialects.csv').open())}
    assert len({r[0] for r in rows})==328
    for r in rows:
        assert r[0] in languages
        for t in r[14].split():
            if t.startswith('dialect:'):assert dialects[t]['Language_ID']==r[0]
    mapping=json.loads((RAW/'language-map-proposed.json').read_text())
    assert mapping['West Pahāṛī']['language']=='WPah'
    assert mapping['Chang']['language']=='ChangNaga'
    assert mapping['Gāndhārī']['language']=='Dhp'
    assert mapping['Surin Khmer']['language']=='NorthernKhmer'
    assert not set(mapping)&{'Kol','Kol.','Kor.','Semnan','Pahāṛī','Munda'}

def test_west_pahari_languages_are_separate_and_survive_remapping(rows):
    from collections import Counter
    languages={r['ID']:r for r in csv.DictReader((ROOT/'cldf/languages.csv').open())}
    expected={'Bangani':2186,'Deogari':236,'Himachali':116,'Khashdhari':45,
              'Khashi':26,'Padri':25,'Bauri':19,'Bushahari':8,'Barari':6,
              'OuterSiraji':3,'Shoracholi':3,'Kotguru':2,'ShimlaSiraji':2,
              'Kotkhai':1,'WesternWestPahari':2}
    counts=Counter(r[0] for r in rows)
    assert {ident:counts[ident] for ident in expected}==expected
    assert counts['WPah']==32  # Source did not provide a more specific attribution.
    for name,record in PROMOTIONS.items():
        assert languages[record['ID']]['Clade']=='W. Pahari'
        assert resolve_language('WPah',name)==(record['ID'],'')
        assert resolve_language(record['ID'],name)==(record['ID'],'')
    assert resolve_language('Garh','Bangani')==('Bangani','')
    assert resolve_language('WPah','')==('WPah','')
    assert not any('dialect:WPah:' in r[14] or 'zoller-garh-bangani' in r[14] for r in rows)
    assert not any(r[0] in expected and 'dialect:' in r[14] for r in rows)

def test_printed_boundaries_superscripts_and_lowered_accents(rows):
    forms={r[2] for r in rows}
    assert {'-tai','gʰãːɖ','minᵃkī','tata̤m'}<=forms
    assert any('pạ' in f for f in forms)
    assert all(not unicodedata.combining(f[0]) for f in forms)
    assert any('-̇' in r[2] and 'uncertain' in r[14] for r in rows)

def test_attribution_scopes(records):
    def get(key,form):return next(c for c in candidates(records[key]) if c['form']==form)
    c=get('zoller2023:18.1:p537:130','atī');assert c['language']=='Pr' and c['labels']==['Pr.']
    c=get('zoller2023:18.4:p781:13','huz, həz');assert c['language']!='D'
    c=get('zoller2023:18.1:p654:978','pūryate');assert c['status']!='candidate'
    c=get('zoller2023:18.7.13:p951:14','uśkār') if 'zoller2023:18.7.13:p951:14' in records else None
    rec=next(r for r in records.values() if r['key'].endswith(':14') and r['page']==952)
    assert not any(c['form']=='uśkār' and c['gloss']=='to groan' and c['status']=='candidate' for c in candidates(rec))

def test_keys_do_not_use_spelling_or_gloss():
    def rec(form,gloss):return {'key':'zoller2023:18.1:p519:9999','page':519,'section':'18.1','tokens':[{'style':'r','text':'Bng. '},{'style':'f','text':form},{'style':'r','text':f' ‘{gloss}’'}]}
    assert candidates(rec('a','one'))[0]['key']==candidates(rec('ā','corrected'))[0]['key']

def test_qualified_links_and_components_are_not_inheritance(records):
    for key in ['zoller2023:18.1:p532:88','zoller2023:18.1:p658:1032']:
        claim=direct_claim(records[key],[],{'1225','8898'});assert claim is not None
    for text in ['1. Bng. a ‘a’ < OIA a ‘a’ (1), but this is wrong.', '1. Bng. a ‘a’ with first component < OIA a ‘a’ (1).']:
        rec={'tokens':[{'style':'r','text':text}]};assert direct_claim(rec,[],{'1'}) is None

def test_compiled_all_source_keys_and_references(rows):
    keys={r['Source_Key']:r['Legacy_ID'] for r in csv.DictReader((ROOT/'cldf/form-source-keys.csv').open()) if r['Source_Key'].startswith('zoller2023:')}
    assert set(keys)=={r[10] for r in rows}
    refs={r['ID']:r for r in csv.DictReader((ROOT/'cldf/references.csv').open())}
    assert 'zoller2023' in refs
    assert all(c.split('[',1)[0] in refs for r in rows for c in r[7].split(';'))

def test_compiled_graph_and_source_layers(rows):
    aliases={r['Legacy_ID']:r['Form_ID'] for r in csv.DictReader((ROOT/'cldf/form-id-aliases.csv').open())}
    keys={r['Source_Key']:aliases[r['Legacy_ID']] for r in csv.DictReader((ROOT/'cldf/form-source-keys.csv').open()) if r['Source_Key'].startswith('zoller2023:')}
    edges={(r['Child_ID'],r['Parent_ID'],r['Kind']) for r in csv.DictReader((ROOT/'cldf/edges.csv').open())}
    forms={r['ID']:r for r in csv.DictReader((ROOT/'cldf/forms.csv').open()) if 'zoller2023' in r['Source']}
    for row in rows:
        r=forms[keys[row[10]]]
        assert r['Language_ID']==row[0]
        assert r['Original']==row[2] and r['Form']==row[2]
        if row[11]:assert (keys[row[10]],keys[row[11]],'variant') in edges
        elif row[1]:assert (keys[row[10]],row[1],'borrowed' if row[8] else 'reflex') in edges

def test_audited_mixed_fonts_tone_digits_and_nested_glosses(records, rows):
    forms={r[2] for r in rows}
    assert 'nã̄-kᵛüṭ̚' in forms and 'cuaŋ⁴' in forms
    cs=list(candidates(records['zoller2023:18.1:p641:868']))
    assert not any(c['form']=='üṭ̚' for c in cs)
    c=next(c for c in candidates(records['zoller2023:18.1:p718:1474']) if c['form']=='śɔ̀keṛu')
    assert '‘robbing’ food; a child' in c['gloss']
    c=next(c for c in candidates(records['zoller2023:18.7.13:p960:60']) if c['form']=='-ṇɔ')
    assert c['status']!='candidate' and not c['gloss']
    assert any(c['form']=='tup³¹' for c in candidates(records['zoller2023:18.7.16:p1011:9']))
    assert not {'*m(a)t','*-lɔc'}&forms

def test_historical_stages_and_source_varieties(rows):
    registry={r['ID']:r for r in csv.DictReader((ROOT/'cldf/languages.csv').open())}
    assert any('Old English' in registry[r[0]]['Name'] for r in rows)
    tags=' '.join(r[14] for r in rows)
    assert all(name in tags for name in ['Northern','Lashkhi','Singhbhum'])


def test_new_nihali_attestations_do_not_inherit_an_older_review(rows):
    nihali=[r for r in rows if r[0]=='Ni']
    assert len(nihali)==5
    assert all(not r[1] and not r[11] and not r[12] for r in nihali)
