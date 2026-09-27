"""Source, transcription, grammar and graph regressions for Perder (2013)."""
import csv
import importlib.util
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from segments import Tokenizer
from tags import GENDER_TAGS, GRAMMATICAL_TAGS

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('perder_dameli_2013',ROOT/'data/other/forms/raw_data/perder_dameli_2013.py')
source=importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)
RAW=source.records()
FORMS,AUDIT=source.build()
BY_KEY={r['Entry_Key']:r for r in FORMS}
AUDIT_BY_UNIT={r['Unit_ID']:r for r in AUDIT}


def resolve(unit):
    a=AUDIT_BY_UNIT[unit]
    key=(a['Merged_Into'] or a['Emitted_Key']).split(';')[0]
    return BY_KEY[key]


def test_reproducible_installed_rows_and_total_audit_accounting():
    installed=list(csv.reader(source.FORM_OUTPUT.open()))
    assert installed==[[r[k] for k in source.FIELDS] for r in FORMS]
    assert len(FORMS)==1856
    assert len(RAW)==len(AUDIT)==3398
    assert len({r['unit'] for r in RAW})==len(RAW)
    assert Counter(r['Status'] for r in AUDIT)=={
        'installed':2801,'installed_after_repair':75,'skipped':522}
    assert all(r['Reason'] for r in AUDIT)
    assert all(bool(r['Emitted_Key'])==(r['Status']!='skipped') for r in AUDIT)
    for r in AUDIT:
        assert all(p==BY_KEY[p['Entry_Key']] for p in json.loads(r['Parsed_Records']))


def test_complete_tables_continuations_and_appendix():
    counts=Counter(r['region'] for r in RAW)
    assert counts['table48']==152
    assert counts['table26']==40
    assert counts['table27']==158  # gender-spanning cells counted once
    assert counts['table17']==59   # both continuations, including printed p. 69
    assert counts['table39']==75   # all 25 conjuncts and both component columns
    assert resolve('p207:t48:y90:c1')['Form']=='aċap'
    assert resolve('p208:t48:y605:c2')['Form']=='žup'
    examples={int(r['example']) for r in RAW if r.get('example')}
    assert examples==set(range(1,178))-{38}
    assert resolve('p94:prose:y237:x97')['Form']=='nikʰaa −aw −aai −uma'
    assert {r['page'] for r in RAW if r['region']=='interlinear' and r['page']>=210}==set(range(210,217))
    assert resolve('p126:t39:y153:c3')['Gloss']=='search'
    assert resolve('p69:t17:y155:c1')['Form']=='zaatak'


def test_grammar_does_not_destroy_lexical_slashes_or_word_class():
    assert source.parse_gloss('he/she did (do − PFV.3SG)')[0]=='do'
    lexical,tags=source.parse_gloss('read.CAUS2−FUT.1SG',interlinear=True)
    assert lexical=='read'
    assert set(tags)>={'verb','caus','second-causative','fut','1sg'}
    assert 'pron' not in tags
    lexical,tags=source.parse_gloss('is−Q',interlinear=True)
    assert lexical=='is' and set(tags)>={'verb','copula','interr'}
    assert 'part' not in tags
    assert source.parse_gloss('aunt (MZ)')[0]=='aunt (MZ)'
    assert source.parse_gloss('s.o. arranges a marriage')[0]=='s.o. arranges a marriage'
    assert resolve('p212:extext:y186:w2')['Gloss']=='late'
    assert 'uncertain' in resolve('p212:extext:y186:w2')['Tags'].split()


def test_person_case_and_gender_spans_are_scoped_to_the_correct_cells():
    rows=[r for r in FORMS if r['Form']=='leeṇḍa' and r['Gloss']=='bald']
    assert {r['Tags'] for r in rows}=={'adj f','adj m'}
    # Three feminine rows in this paradigm share the printed masculine cell;
    # the m/f distinction must not be fabricated for the transitive forms.
    trans=resolve('p95:t27:y328:c2')
    assert trans['Form']=='leekʰee'
    assert set(trans['Tags'].split())>={'3sg','pfv','tr','verb'}
    assert not set(trans['Tags'].split())&{'m','f'}
    first=resolve('p72:t18:y493:c2')
    assert first['Form']=='muu'
    assert set(first['Tags'].split())>={'1sg','obl','erg','pron'}


def test_variants_and_semantically_different_parallel_forms():
    alt=BY_KEY['perder2013dameli:p84:t22:y86:c1:v2']
    assert alt['Form']=='krinaa'
    assert BY_KEY[alt['Variant_Of_Key']]['Form']=='krnaa'
    assert resolve('p10:prose:y403:x307:part1')['Gloss']=='grandfather'
    assert resolve('p10:prose:y403:x307:part2')['Gloss']=='grandmother'
    assert resolve('p10:prose:y403:x307:part2')['Variant_Of_Key']==''
    assert resolve('p89:t26:y54:c1')['Gloss']=='one'
    assert BY_KEY['perder2013dameli:p89:t26:y300:c2:v2']['Form']=='sawa'
    assert resolve('p126:t39:y222:c3')['Gloss']=='start'


def test_profile_covers_every_form_and_preserves_distinctions():
    t=Tokenizer(str(ROOT/'conversion/perder-dameli.txt'))
    def convert(s):return unicodedata.normalize('NFC',t(s,column='IPA').replace(' ','').replace('#',' '))
    assert convert('c̣ʰaar')=='ʦ̣ʰār'
    assert convert('čay')=='cay'
    assert convert('ċĩĩt')=='ʦī̃t'
    assert convert('ɡaṭ−aw−a−ee')=='gaṭ-av-a-ē'
    assert convert('ẉi−i')=='ɻi-i'
    assert convert('žǎn žân')=='źǎn źân'
    assert convert('mãã−Ø')=='mā̃-Ø'
    assert convert('ba.loy')=='ba.loy'
    assert convert('uu oo')=='ū ō'
    for r in FORMS:
        assert '�' not in convert(r['Form']),r
        assert unicodedata.is_normalized('NFC',r['Form'])
    assert resolve('p39:prose:y356:x64')['Phonemic']=='ˈbɑːˌʂæː'
    assert resolve('p40:prose:y269:x284')['Form']=='c̣ai'
    assert resolve('p32:prose:y375:x131')['Form']=='žǎn'


def test_explicit_graph_claims_have_unique_resolvable_targets():
    linked=[r for r in FORMS if r['Parameter_ID']]
    assert len(linked)==1 and linked[0]['Parameter_ID']=='13734'
    assert linked[0]['Form']=='ištrii'
    compound=resolve('p50:t14:y236:c1')
    assert {BY_KEY[k]['Form'] for k in compound['Derivation_Parent_Keys'].split('|')}=={'draak','muṭ'}
    for r in FORMS:
        for field in ('Variant_Of_Key','Borrowed_From_Key','Derivation_Parent_Keys'):
            assert all(k in BY_KEY and k!=r['Entry_Key'] for k in r[field].split('|') if k)
    assert not any(r['Borrowed_From_Key'] for r in FORMS)
    assert resolve('p99:t30:y541:c3')['Derivation_Parent_Keys']==''
    assert resolve('p94:ex37:y111:w1')['Tags'].split().count('uncertain')==1


def test_comparanda_rejected_forms_and_translation_tiers_are_excluded():
    for u in ('p34:prose:y615:x221','p90:prose:y412:x64','p90:prose:y412:x313',
              'p173:prose:y384:x293','p35:prose:y236:x102','p58:prose:y82:x64',
              'p192:prose:y305:x91'):
        assert AUDIT_BY_UNIT[u]['Status']=='skipped'
    assert not any(re.search(r'[A-Z]{2,}|[‘’“”<>*?]',r['Form']) for r in FORMS)
    historical=resolve('p90:prose:y244:x321')
    assert 'morgenstierne1942dameli[p. 137' in historical['Source']
    assert 'uncertain' in historical['Tags'].split()
    assert 'did not recognise' in historical['Notes']
    assert resolve('p179:t47:y141:c1')['Gloss']=='what'
    finger=resolve('p30:prose:y502:x242')
    assert (finger['Form'],finger['Gloss'])==('aaŋɡuẉi','finger')
    assert 'p. 30, prose' in finger['Source']
    assert 'jest’ero:' in resolve('p83:prose:y274:x84')['Notes']


def test_language_dialect_reference_and_tag_registration():
    dialects={r['Tag']:r for r in csv.DictReader((ROOT/'cldf/dialects.csv').open())}
    assert {r['Language_ID'] for r in FORMS}=={'Dm'}
    for r in FORMS:
        for tag in r['Tags'].split():
            if tag.startswith('dialect:'):
                assert dialects[tag]['Language_ID']=='Dm'
                assert dialects[tag]['Latitude'] and dialects[tag]['Longitude']
            else:assert tag in GRAMMATICAL_TAGS|GENDER_TAGS
    assert 'dialect:Dm:perder2013-Aspar:Aspar' in resolve('p10:prose:y403:x307:part1')['Tags']
    assert not any(t.startswith('dialect:') for t in resolve('p10:prose:y417:x64:part2')['Tags'].split())
    bib=(ROOT/'cldf/sources.bib').read_text()
    for k in ('perder2013dameli','morgenstierne1942dameli','cacopardo2008dameli'):
        assert re.search(r'@\w+\{'+k+',',bib)


def test_fresh_visual_sample_has_no_material_errors(tmp_path):
    sample=list(csv.DictReader((source.RAW/f'{source.PREFIX}-sample.csv').open()))
    assert len(sample)==20 and len({r['Entry_Key'] for r in sample})==20
    assert {r['Seed'] for r in sample}=={'7698818706101321111'}
    assert {r['Result'] for r in sample}=={'pass'}
    for r in sample:
        assert r['Source_Form']==BY_KEY[r['Entry_Key']]['Form']
        assert r['Gloss']==BY_KEY[r['Entry_Key']]['Gloss']
    output=tmp_path/'sample.csv'
    source.sample_report(7698818706101321111,output)
    regenerated=list(csv.DictReader(output.open()))
    assert [r['Entry_Key'] for r in regenerated]==[r['Entry_Key'] for r in sample]


def test_compiled_source_keeps_original_phonemic_and_resolved_graph():
    rows={r['ID']:r for r in csv.DictReader((ROOT/'cldf/forms.csv').open())}
    aliases={r['Legacy_ID']:r['Form_ID'] for r in csv.DictReader((ROOT/'cldf/form-id-aliases.csv').open())}
    keys={r['Source_Key']:aliases.get(r['Legacy_ID'],r['Legacy_ID']) for r in csv.DictReader((ROOT/'cldf/form-source-keys.csv').open()) if r['Source_Key'].startswith(source.SOURCE_ID+':')}
    assert set(keys)==set(BY_KEY)
    for key,r in BY_KEY.items():
        compiled=rows[keys[key]]
        assert compiled['Language_ID']=='Dm'
        assert compiled['Original']==r['Form']
        assert compiled['Phonemic']==r['Phonemic']
        assert set(r['Tags'].split())<=set(compiled['Tags'].split())
        assert set(r['Source'].split(';'))<=set(compiled['Source'].split(';'))
    assert rows[keys['perder2013dameli:p40:prose:y269:x284']]['Form']=='ʦ̣ai'
    edges=list(csv.DictReader((ROOT/'cldf/edges.csv').open()))
    outgoing={}
    for e in edges:
        outgoing.setdefault(e['Child_ID'],[]).append(e)
    for key,r in BY_KEY.items():
        actual={(e['Parent_ID'],e['Kind'],e['Rank']) for e in outgoing.get(keys[key],[])}
        expected=set()
        if r['Parameter_ID']:
            expected.add((r['Parameter_ID'],'reflex','1'))
        if r['Variant_Of_Key']:
            expected.add((keys[r['Variant_Of_Key']],'variant','1'))
        parents=[p for p in r['Derivation_Parent_Keys'].split('|') if p]
        kind='component' if len(parents)>1 else 'derived'
        for parent in parents:
            expected.add((keys[parent],kind,'1'))
        assert actual==expected,(key,actual,expected)
        if not expected:
            assert rows[keys[key]]['Status']=='unlinked'
