"""Whole Korava proposal regressions; no canonical mutation or full build."""
import csv,importlib.util,json,sys,unicodedata
from pathlib import Path
from segments import Tokenizer
DATA=Path(__file__).resolve().parents[1]
P=DATA/'data/other/forms/raw_data/grierson_korvi_1906'
sys.path.insert(0,str(P))
spec=importlib.util.spec_from_file_location('korvi_full',P/'prepare_full.py');stage=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)
def rows():return list(csv.reader((P/'proposal.csv').open()))
def audit():return [json.loads(x) for x in (P/'proposal-audit.jsonl').read_text().splitlines()]
def test_every_source_unit_and_legacy_key_accounted():
 r,a=stage.generate();assert r==rows() and a==audit()
 assert len(r)==990 and len(a)==1237
 assert {x['prompt'] for x in a if 'prompt' in x}==set(range(1,242))
 assert sum(x['section'].startswith('specimen') for x in a)==899
 assert sum(x['section']=='grammar' for x in a)==89
 legacy={x[10] for x in csv.reader((P/'historical-column.csv').open())}
 assert len(legacy)==156 and legacy<={x[10] for x in r}
def test_fine_mark_corrections_and_sibling_senses():
 by={x['prompt']:x for x in audit() if 'prompt' in x};out={r[10]:r for r in rows()}
 assert by[94]['forms']==['Yā̃tka']
 assert by[150]['forms']==['Oṇḍē hō̃ta']
 assert 'āmḷ-' in by[129]['forms'][0] and 'āmḷ-' in by[223]['forms'][0]
 assert by[223]['forms'][0].startswith('Ninnāvun ')
 assert [out[k][3] for k in by[49]['entry_keys']]==['elder brother','younger brother']
 assert [out[k][3] for k in by[50]['entry_keys']]==['elder sister','younger sister']
 assert all(not out[k][11] for n in [49,50] for k in by[n]['entry_keys'])
def test_original_and_erratum_are_distinct_witnesses():
 out={r[10]:r for r in rows()};base='grierson1906lsi4:korvi_belgaum:'
 assert out[base+'186'][2]=='Nī adāsā'
 assert out[base+'186:erratum'][2]=='Nī aḍasā'
 assert out[base+'164'][2]=='Ava indū' and out[base+'164:erratum'][2]=='Ãva indū'
 assert 'grierson_addenda_minora_iv_boundin[' in out[base+'164:erratum'][7]
 assert not out[base+'164:erratum'][11]
def test_explicit_optional_letters_expanded_without_erasing_raw():
 out={r[10]:r for r in rows()};units=[x for x in audit() if 'optional_expansion' in x]
 assert len(units)==7
 for u in units:
  assert set(u['optional_expansion']['forms'])=={out[k][2] for k in u['entry_keys']}
  assert '(' in u['optional_expansion']['literal']
 assert not any('(' in r[2] for r in rows())
def test_lects_controls_and_bound_morphology_do_not_leak():
 out={r[10]:r for r in rows()}
 for u in audit():
  if u['status']!='ingested':assert not u['entry_keys']
  if u['section'].startswith('specimen'):
   for k in u['entry_keys']:assert stage.LECTS[u['section']] in out[k][14]
  if u['section'] in {'grammar','source-wide-comparison'}:
   assert all('dialect:' not in out[k][14] for k in u['entry_keys'])
 assert sum(u['status']=='excluded_control' for u in audit())==15
 assert sum(u['status']=='bound_morphology_evidence' for u in audit())==20
def test_all_commentary_and_continuations_survive():
 out={r[10]:r for r in rows()}
 for u in audit():
  if u.get('source_commentary'):
   assert all(u['source_commentary'] in out[k][6] for k in u['entry_keys'])
 continued=[u for u in audit() if u.get('physical_fragments')]
 assert len(continued)==2 and all(len(u['physical_fragments'])==2 for u in continued)
 anomalous=next(u for u in audit() if u.get('gloss')=='time-is')
 assert all('uncertain' in out[k][14] and 'verb' not in out[k][14] for k in anomalous['entry_keys'])
def test_profile_tags_and_graph():
 sys.path.insert(0,str(DATA));import tags
 t=Tokenizer(str(P/'proposal-profile.txt'));out={r[10]:r for r in rows()}
 for r in out.values():
  assert '�' not in t(r[2],column='IPA')
  assert all(x.startswith('dialect:') or x in tags.GRAMMATICAL_TAGS|tags.GENDER_TAGS for x in r[14].split())
  assert not r[11] or r[11] in out
 assert t(unicodedata.normalize('NFC','ṯs̱'),column='IPA').replace(' ','')=='ʦ'
 assert t('Chhalū',column='IPA').replace(' ','')=='cʰalū'
 assert t(unicodedata.normalize('NFC','kharṯs̱'),column='IPA').replace(' ','')=='kʰarʦ'
 assert t('mawn-ka',column='IPA').replace(' ','')=='mavn-ka'
 import profile_policy
 _,body=profile_policy.read_profile(P/'proposal-profile.txt');rules={r[0]:r[1] if len(r)>1 else '' for r in body}
 assert all(profile_policy.house_output(g,o,rules,'grierson-korvi-1906')==profile_policy.nfc(o) for g,o in rules.items())
def test_exact_reuse_keeps_every_evidence_path():
 out={r[10]:r for r in rows()};reused=[u for u in audit() if u.get('exact_attestation_reuse')]
 assert reused
 for u in audit():
  for k in u['entry_keys']:
   assert k in out
   if u.get('citation_locator'):assert '['+u['citation_locator']+']' in out[k][7]

def test_distinct_neuter_and_structured_past_glosses():
 out={r[10]:r for r in rows()}
 r=out['grierson1906lsi4:korava:p414:comparison:one:alternate:2']
 assert r[2]=='oṇḍ' and 'n' in r[14].split() and not r[11]
 for n in range(185,191):
  key=f'grierson1906lsi4:korvi_belgaum:{n}'
  assert '(past tense)' not in out[key][3] and 'pret' in out[key][14].split()
  assert '(past tense)' in next(u for u in audit() if u['source_unit_key']==key)['gloss']

def test_co_listed_table_synonyms_have_no_variant_edges():
 out={r[10]:r for r in rows()}
 for n in [86,87,93]:
  for suffix in [':1',':2']:
   assert not out[f'grierson1906lsi4:korvi_belgaum:{n}{suffix}'][11]
