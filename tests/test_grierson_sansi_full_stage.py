"""Full Sansi proposal checks; these do not install data or build a database."""
import csv
import importlib.util
import json
from pathlib import Path

from segments import Tokenizer
from tags import GENDER_TAGS, GRAMMATICAL_TAGS, extract_tags

DATA = Path(__file__).resolve().parents[1]
PACKAGE = DATA/'data/other/forms/raw_data/grierson_sansi_1922'
spec=importlib.util.spec_from_file_location('sansi_full',PACKAGE/'preview_full_source.py')
source=importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)

def test_complete_source_accounting_and_legacy_keys():
    rows,audit=source.generate()
    assert len(rows)==1943 and len(audit)==2562
    assert all(a['english_printed_page']==int(a['page']) for a in audit if a['section']=='table')
    assert rows==list(csv.reader((PACKAGE/'full-preview.csv').open()))
    assert audit==[json.loads(x) for x in (PACKAGE/'full-preview-audit.jsonl').read_text().splitlines()]
    assert len({r[10] for r in rows})==len(rows)
    assert all(len(r)==15 and r[0]=='Sansi' and r[2] for r in rows)
    assert sum(a['status']=='non_target_Hindostani_control' for a in audit)==80
    assert sum(a['status']=='comparison_control' for a in audit)==32
    assert sum(a['status']=='bound_morpheme_audit_only' for a in audit)==26
    assert sum(a['status']=='pronunciation_evidence' for a in audit)==2
    assert sum(bool(a['reuse_entry_keys']) for a in audit)==535
    legacy=set(json.loads((PACKAGE/'legacy-pilot-entry-keys.json').read_text()))
    assert len(legacy)==87 and legacy <= {r[10] for r in rows}

def test_alternative_scope_and_literal_readings():
    rows,_=source.generate()
    by_key={r[10]:r for r in rows}
    assert by_key['grierson1922lsi11:sasi_ordinary:76:answer:1'][3]=='Little bird'
    assert by_key['grierson1922lsi11:sasi_ordinary:76:answer:2'][3]=='Bird'
    assert source.alternatives(241,'argot','Ḍhāmē-(or nādā)-gē bēkkī kūṭiā-wāḷē nāsā')==['Ḍhāmē-gē bēkkī kūṭiā-wāḷē nāsā','nādā-gē bēkkī kūṭiā-wāḷē nāsā']
    assert by_key['grierson1922lsi11:sasi_argot:213:answer:2'][2]=='Buh jastiā'
    assert by_key['grierson1922lsi11:sasi_argot:237'][2].find('chaī̃')>=0
    assert by_key['grierson1922lsi11:sasi_argot:97'][2]=='Jēkar jē'
    assert by_key['grierson1922lsi11:sasi_ordinary:217:answer:1'][14].split()[-3:]==['impv','2sg','sg']
    assert not any('(' in r[2] or ')' in r[2] for r in rows if ':sasi_' in r[10])

def test_register_dialects_and_source_relations():
    rows,audit=source.generate()
    registry={r['Tag']:r for r in csv.DictReader((DATA/'cldf/dialects.csv').open())}
    keys={r[10] for r in rows}
    for r in rows:
        for tag in r[14].split():
            if tag.startswith('dialect:'): assert registry[tag]['Language_ID']=='Sansi'
            else: assert tag in GRAMMATICAL_TAGS|GENDER_TAGS
        assert not r[13] or r[13] in keys
        assert not r[11] and not r[12]
    assert sum(bool(r[13]) for r in rows)==154
    assert all(not a['entry_keys'] for a in audit if a['status']=='comparison_control')
    assert all('argot' in r[14].split() for r in rows if ':sasi_argot:' in r[10])
    assert all('argot' not in r[14].split() for r in rows if ':sasi_ordinary:' in r[10])
    assert extract_tags('argot; ordinary explanatory prose',attestations=False)==('argot','ordinary explanatory prose')
    assert extract_tags('An argot comparison is tentative.',attestations=False)==('', 'An argot comparison is tentative.')
    frontend=(DATA.parent/'jambu-static/src/lib/tags.ts').read_text()
    assert "'argot'" in frontend and "argot: 'argot'" in frontend

def test_literal_sound_profile_and_pronunciation_separation():
    rows,_=source.generate()
    tokenizer=Tokenizer(str(DATA/'conversion/grierson-sansi-1922.txt'))
    for r in rows:
        converted=tokenizer(r[2],column='IPA')
        assert '�' not in converted,r[10]
    for text in ['chaī̃','g̲h̲','á','ā́','ŭ','ḷ','ṇ']:
        assert '�' not in tokenizer(text,column='IPA'),text
    phonemic=[r for r in rows if r[5]]
    assert len(phonemic)==2 and {r[5] for r in phonemic}=={'‘ūwā','g‘ōṛā'}
    assert all(r[2]!=r[5] for r in phonemic)
