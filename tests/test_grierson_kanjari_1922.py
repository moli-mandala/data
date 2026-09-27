"""Focused full-source checks without a database build."""
import csv,importlib.util,io,json,sys,unicodedata
from collections import Counter
from pathlib import Path
from segments import Tokenizer
DATA=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(DATA))
PACKAGE=DATA/'data/other/forms/raw_data/grierson_kanjari_1922'
CSV=DATA/'data/other/forms/20260925-grierson-kanjari.csv'
spec=importlib.util.spec_from_file_location('kanjari_full',PACKAGE/'import_source.py')
source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
def installed():return list(csv.reader(CSV.open()))
def test_complete_scope_and_legacy_identity():
    rows,audit=source.generate()
    assert rows==installed()
    assert len(rows)==1832 and len(audit)==2229
    assert Counter(a['section'] for a in audit)=={'table':482,'prose':239,'specimen':1508}
    table=[a for a in audit if a['section']=='table']
    assert Counter(a['prompt'] for a in table)=={str(i):2 for i in range(1,242)}
    assert sum(a['status']=='source_blank' for a in table)==35
    keys={r[10] for r in rows}
    legacy=json.loads((PACKAGE/'legacy-pilot-keys.json').read_text())
    assert legacy['count']==163 and set(legacy['keys'])<=keys
    assert len(keys)==len(rows)
def test_reuse_keeps_every_source_locator_and_control():
    rows,audit=source.generate();index={r[10]:r for r in rows}
    reused=[a for a in audit if a['reuse_entry_keys']]
    assert len(reused)==391
    for a in reused:
        row=index[a['reuse_entry_keys'][0]]
        assert unicodedata.normalize('NFC',a['form'])==row[2] and a['gloss']==row[3]
        assert f"p. {a['page']}, specimen {a['specimen']}, line {a['line']}, aligned word {a['word']}" in row[7]
    controls=[a for a in audit if a['status']=='non_target_Hindostani_control']
    assert {(a['page'],a['unit']) for a in controls}=={('97','kheri-heart'),('97','kheri-inhabitant'),('101','lex15')}
    assert all(not a['entry_keys'] for a in controls)
    assert next(a for a in controls if a['page']=='101')['forms']=='khamāl'
def test_source_specific_typography_and_variant_separation():
    rows,_=source.generate();r={x[10]:x for x in rows};p='grierson1922lsi11:'
    expected={'specimen:p108-l3-w3':'Thōṛā','specimen:p114-l14-w1':'bharwāṛ-ko','specimen:p114-l15-w5':'khuṭ-ko','belgaum:79':'Chaiṭ','sitapur:230':'Birō pēṛhēlā tar ghuṛārā par chhaiṭhō hai','specimen:p116-l8-w7':'jīdē','specimen:p116-l1-w4':'ītnā','specimen:p116-l9-w8':'byādīk','specimen:p109-l14-w6':'rīs','specimen:p109-l20-w5':'līnē','specimen:p120-l8-w8':'i','specimen:p114-l3-w2':'kīdō','specimen:p104-l2-w2':'kī','specimen:p111-l4-w3':'kī','specimen:p115-l3-w4':'apaṇī','specimen:p116-l9-w6':'khuśī','specimen:p117-l8-w5':'rahāt-bī-nā','prose:p99:lex31':'riūchhis','prose:p101:lex46':'riūchhis','prose:p100:lex31':'jhuraī','prose:p98:lex7':'bihārō̃-mē','belgaum:61':'Saitāne','sitapur:66':'Nimāni','sitapur:88':'Tar-hēlī','belgaum:231':'Urō-kō bhai urō-ki bhayaṇ-dē khuñchī hē','belgaum:233':'Mērō bāpōne wō nunke nandō-mā rahēndō','sitapur:33':'Guṛārā','sitapur:33:answer2':'gurārā','prose:p100:lex27':'ṭipuī','specimen:p103-l10-w8':'ṭipūī','prose:p100:lex13':'hū̃dō','specimen:p104-l7-w4':'hū̃ḍō','sitapur:36':'Khō̃sar','specimen:p115-l2-w1':'byādīk-mā','specimen:p115-l7-w4':'dusārnō-nā'}
    for key,form in expected.items():assert r[p+key][2]==unicodedata.normalize('NFC',form)
def test_source_profile_and_registered_dialects():
    tokenizer=Tokenizer(str(DATA/'conversion/grierson-kanjari-1922.txt'))
    dialects={r[0]:r for r in csv.reader((DATA/'cldf/dialects.csv').open())}
    for r in installed():
        assert len(r)==15 and r[0]=='Kanjari'
        assert all(not r[i] for i in [1,4,5,11,12,13])
        actual=tokenizer(r[2],column='IPA').replace(' ','').replace('#',' ').strip()
        expected=r[2].lower().replace('w','v').replace('ṅ','ŋ').replace('d̲z̲','ʣ')
        for c in '?[]':expected=expected.replace(c,'')
        assert actual==expected,r[10]
        for tag in r[14].split():
            if tag.startswith('dialect:'):
                d=dialects[tag.split(':')[2]]
                assert d[1]==tag and d[2]=='Kanjari' and d[6:8]==['','']
def test_focused_parser_has_all_keys_and_no_errors():
    import make_cldf
    errors=io.StringIO();rows,stats=make_cldf.parse_file(str(CSV),errors,name='20260925-grierson-kanjari')
    assert not errors.getvalue()
    assert len(rows)==stats['converted']==1832
    original={r[10]:r for r in installed()}
    assert {r.entry_key for r in rows}==set(original)
    assert all(r.old_form==original[r.entry_key][2] for r in rows)
