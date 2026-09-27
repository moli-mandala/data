"""Whole-primer recovery gates without canonical installation or a database build."""
import csv,hashlib,importlib.util,json,sys
from pathlib import Path
from segments import Tokenizer
DATA=Path(__file__).resolve().parents[1]
P=DATA/'data/other/forms/raw_data/hahn_asur_1900'
sys.path.insert(0,str(DATA))
spec=importlib.util.spec_from_file_location('hahn_whole',P/'prepare_whole.py');stage=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)
def rows():return list(csv.reader((P/'proposal.csv').open()))
def audit():return [json.loads(x) for x in (P/'proposal-audit.jsonl').read_text().splitlines()]
def test_complete_recovery_and_regeneration():
 r,a,e=stage.build();assert r==rows() and a==audit()
 assert len(r)==835 and len(a)==670 and len(e)==153
 assert len({x[10] for x in r})==len(r)
 legacy,_=stage.legacy();assert len(legacy)==621
 assert {x[10] for x in legacy}<={x[10] for x in r}
 assert not [x for x in a if x['status'] in ['held','same_source_reuse','excluded_nonlexical_context']]
 assert {int(x['printed_page']) for x in a}==set(range(149,173))
def test_all_reviewed_responses_have_outputs():
 a={x['source_unit_key']:x for x in audit()}
 for f in ['expression-recovery-p154-161-reviewed.jsonl','expression-recovery-p162-169-first-reading.jsonl','expression-recovery-song-first-reading.jsonl']:
  for u in stage.read(f):
   k=u.get('source_unit_key',u.get('entry_key'))
   if u['status'] in ['already_inventoried_scope','control_non_target']:continue
   assert a[k]['entry_keys'],k
 assert len(a['hahn1900asur:p166:expression:35-compounds:01']['pre_reuse_entry_keys'])==2
 assert len(a['hahn1900asur:p166:expression:35-compounds:02']['pre_reuse_entry_keys'])==2
def test_independent_glyph_repairs_and_nonvariants():
 r={x[10]:x for x in rows()}
 expected={'hahn1900asur:p165:s35-head:01':'dohóteā','hahn1900asur:p167:s38:02':'Kuniā','hahn1900asur:p167:s38:03':'Kuneā','hahn1900asur:p150:sintro:01':'jhaṛī','hahn1900asur:p159:s16:06:variant:2':'kuniā','hahn1900asur:155:5:expression:03':'meṛhed rā kaṭu','hahn1900asur:161:19:expression:02':'Nihī tuanā'}
 for k,v in expected.items():assert r[k][2]==v
 assert all(not x[11] for x in r.values())
 assert any(x[2]=='pa’eṉ' and not x[3] for x in r.values())
 assert any(x[2]=='paheṉ' and not x[3] for x in r.values())
def test_source_morphology_categories_are_synchronized():
 import tags
 frontend=(DATA.parent/'jambu-static/src/lib/tags.ts').read_text()
 for tag in ['infix','affix','completive']:
  assert tag in tags.GRAMMATICAL_TAGS and f"'{tag}'" in frontend and f"{tag}: '{tag}'" in frontend
 r=rows()
 assert any(x[2]=='gē' and {'infix','caus'}<=set(x[14].split()) for x in r)
 assert any(x[2]=='cabā' and {'affix','completive'}<=set(x[14].split()) for x in r)
 assert any(x[2]=='hōṉ' and 'affix' in x[14] for x in r)
 assert any(x[2]=='hōṉ' and x[3]=='even' and 'affix' not in x[14] for x in r)
def test_all_symbols_tags_and_citations_resolve():
 import tags,profile_policy,make_refs
 from unify_cldf import citation_keys
 t=Tokenizer(str(P/'proposal-profile.txt'));known=tags.GRAMMATICAL_TAGS|tags.GENDER_TAGS
 for r in rows():
  assert len(r)==15 and r[2] and '�' not in t(r[2],column='IPA')
  assert all(x.startswith('dialect:') or x in known for x in r[14].split())
  assert set(make_refs.source_ids(r[7]))==citation_keys(r[7])=={'hahn1900asur'}
 _,body=profile_policy.read_profile(P/'proposal-profile.txt');rules={x[0]:x[1] if len(x)>1 else '' for x in body}
 assert all(profile_policy.house_output(g,o,rules,'hahn-asur-1900')==profile_policy.nfc(o) for g,o in rules.items())
 assert t('ch',column='IPA').replace(' ','')=='cʰ'
def test_morphology_functions_do_not_collapse_and_song_stays_whole():
 a=audit();r={x[10]:x for x in rows()}
 u=next(x for x in a if x['source_unit_key']=='hahn1900asur:160:19:bound-context')
 yana=[r[k] for k in u['entry_keys'] if r[k][2]=='yanā'];assert len(yana)==2 and yana[0][14]!=yana[1][14]
 song=next(x for x in a if x['source_unit_key']=='hahn1900asur:p172:closing-song:stanza01')
 assert len(song['entry_keys'])==1
 assert 'spendid' in r[song['entry_keys'][0]][3] and 'poetic' in r[song['entry_keys'][0]][14]
def test_reviewed_notes_predicates_and_participle_metadata():
 r={x[10]:x for x in rows()}
 for x in r.values():
  if ':comparison:' in x[10]:assert x[6]!=x[2]
  if ':predicate-alternate:' in x[10]:
   assert ' ' not in x[2] and 'multiword-expression' not in x[14].split()
   assert 'shared hāsu' in x[6]
 for i in (15,16,17):
  x=r[f'hahn1900asur:161:19-23:bound-context:function:{i}']
  assert {'pret','perfect','participle'}<=set(x[14].split())
  assert 'section 20,' in x[7]
 assert all('section source context' not in x[7] for x in r.values() if ':bound-context:' in x[10])
 assert all('independent review' not in x[6].lower() and 'morphology audit' not in x[6].lower() for x in r.values())

def test_auxiliary_source_anomaly_and_repeated_witness():
 r={x[10]:x for x in rows()};a={x['source_unit_key']:x for x in audit()}
 x=r['hahn1900asur:p162:whole-recovery:25:quoted-auxiliary']
 assert x[2]=='dohótauā' and x[3]=='' and {'auxiliary','uncertain'}<=set(x[14].split())
 assert 'p. 167, section 37' in r['hahn1900asur:p166:s37:02'][7]
 assert a['hahn1900asur:p167:whole-recovery:37:repeated-auxiliary']['entry_keys']==['hahn1900asur:p166:s37:02']
def test_complete_o_diacritic_class_reconciliation():
 r={x[10]:x for x in rows()}
 assert r['hahn1900asur:p171:s49:04'][2]=='pōtā'
 assert r['hahn1900asur:p149:sintro-totems:06'][2]=='Rōtē'
 assert r['hahn1900asur:p165:s32-cont:04'][2]=='alom rūēmē'
 assert r['hahn1900asur:p164:whole-recovery:31:imperative-3'][2]=='kā'
 assert 'section 21,' in r['hahn1900asur:161:19-23:bound-context:function:20'][7]
 assert r['hahn1900asur:p171:s50-right:06'][2]=='dohō'
 assert r['hahn1900asur:p167:expression:37-auxiliary:03'][2]=='iŋ rū dohōkedā'

def test_adjudicated_kinship_reading_overrides_tentative_correction():
 decision=json.loads((P/'root-kinship-literal-review-20260926.json').read_text())
 r={x[10]:x for x in rows()}
 assert r[decision['entry_key']][2]==decision['current_form']=='iŋā daimiŋ'
 assert r[decision['entry_key']][2]!=decision['proposed_form']
