"""Focused source-stage tests; no compiled-data or database build required."""
import csv,hashlib,io,json,re,unicodedata as ud
from pathlib import Path
import pytest
import make_cldf,source_meta
from segments.tokenizer import Tokenizer
from tags import GRAMMATICAL_TAGS
ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'data/other/forms/raw_data/peterson_turi_2024/whole-source-recovery'
OUT=HERE/'proposal'
def rows():
 return [r for p in sorted(OUT.glob('*.csv')) for r in csv.reader(p.open())]
def audit():return [json.loads(x) for x in (OUT/'whole-audit.jsonl').read_text().splitlines()]
def test_existing_keys_and_complete_numbered_scope():
 # Historical proposal validation pins the immutable pre-recovery baseline;
 # current canonical equality is covered by test_peterson_turi_whole_installed.
 base=ROOT/'data/other/forms/raw_data/peterson_turi_2024/whole-source-recovery/proposal/20260921-peterson-turi.csv'
 assert (OUT/base.name).read_bytes()==base.read_bytes()
 pinned=json.loads((HERE/'baseline-installed.json').read_text())
 assert hashlib.sha256(base.read_bytes()).hexdigest()==pinned['sha256']
 actual=rows();assert len({r[10] for r in actual})==len(actual)
 assert set(pinned['keys']) <= {r[10] for r in actual}
 units=[a for a in audit() if a['section']=='numbered']
 assert {int(re.match(r'\d+',a['evidence']['example'])[0]) for a in units}==set(range(1,63))
 assert len(units)==79
 assert len([a for a in audit() if a['section']=='appendix1'])==275
 assert {a['evidence']['printed_page'] for a in units if int(re.match(r'\d+',a['evidence']['example'])[0])>=44}=={298,299,300}
def test_all_alignment_and_reuse_targets_preserve_literal_forms():
 indexed={r[10]:r for r in rows()}
 for a in audit():
  assert all(k in indexed for k in a['entry_keys'])
  for w in a.get('word_alignment',[]):
   r=indexed[w['entry_key']]
   assert r[2]==ud.normalize('NFC',w['form'])
   assert w['printed_gloss'] in r[6] or w['printed_gloss'] in r[3]
  if a['status'] not in ('emitted','ingested'):
   assert not a['entry_keys']
def test_control_and_lect_boundaries_and_real_variant_links():
 indexed={r[10]:r for r in rows()}
 for a in audit():
  if a['section']=='appendix1':continue
  u=a['evidence']
  if u.get('kind') in ('foreign_control','comparison_unassigned','documentary_name','reconstructed_analysis','phonological_inventory','historical_editorial_evidence'):
   assert a['entry_keys']==[]
  if u.get('lect','').startswith('historical'):
   assert all('turi_odisha' not in indexed[k][14] for k in a['entry_keys'])
  if u.get('lect')=='Jharkhandi':
   assert all('turi_jharkhandi_peterson2024' in indexed[k][14] and 'turi_odisha' not in indexed[k][14] for k in a['entry_keys'])
 for edge in json.loads((OUT/'variant-decisions.json').read_text()):
  assert indexed[edge['child']][11]==edge['parent']
  assert edge['parent'] in indexed and edge['child']!=edge['parent']
 assert all(not r[12] and not r[13] and not r[1] for r in indexed.values())
def test_literal_analysis_conflicts_and_tags_are_visible():
 indexed={r[10]:r for r in rows()}
 for tail in ['p272:example:15b:whole','p273:example:19b-negative:whole','p298:example:46:whole','p276:example:29:whole']:
  assert 'uncertain' in indexed['peterson2024turi:'+tail][14].split()
 for r in rows():
  assert len(r)==15 and r[2] and r[3]
  assert all(ud.is_normalized('NFC',x) and '\ufffd' not in x for x in r)
  assert all(t.startswith('dialect:') or t in GRAMMATICAL_TAGS for t in r[14].split())
  assert all(c.split('[',1)[0]=='peterson2024turi' and c.endswith(']') for c in r[7].split('; '))
 # A source negative sentence's mistaken English gloss is retained, not silently repaired.
 r=indexed['peterson2024turi:p273:example:19b-negative:whole']
 assert r[3]=='s/he is going' and r[2].startswith('kə=ɛ')
def test_actual_parser_with_declared_profiles_preserves_original_and_identity(monkeypatch):
 meta=source_meta.SourceMeta([ROOT/'data/other/forms/20260921-peterson-turi.yaml',*sorted(OUT.glob('*.yaml'))])
 monkeypatch.setattr(source_meta,'load',lambda:meta)
 for group in ['grammar','historical']:
  stem='20260927-peterson-turi-'+group;profile='peterson-turi-'+group
  tok=Tokenizer(str(OUT/(profile+'.txt')))
  monkeypatch.setitem(make_cldf.convertors,profile,tok)
  source=list(csv.reader((OUT/(stem+'.csv')).open()));indexed={r[10]:r for r in source}
  error=io.StringIO()
  actual,stats=make_cldf.parse_file(str(OUT/(stem+'.csv')),errors=error,name=stem)
  assert not error.getvalue(),error.getvalue()
  assert stats=={'converted':len(source),'for_conversion':len(source)}
  assert {r.entry_key for r in actual}==set(indexed)
  for r in actual:
   raw=indexed[r.entry_key]
   assert r.old_form==raw[2] and not r.ipa
   assert set(raw[14].split()) <= set(r.tags.split())
   a=ud.normalize('NFC',tok(ud.normalize('NFC',raw[2]),column='IPA').replace(' ','').replace('#',' '))
   b=ud.normalize('NFC',tok(ud.normalize('NFD',raw[2]),column='IPA').replace(' ','').replace('#',' '))
   assert a==b==ud.normalize('NFC',r.form) and '\ufffd' not in a
