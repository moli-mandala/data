"""Whole public-snapshot recovery: no full data or browser builds."""
import csv,hashlib,importlib.util,io,json,unicodedata as ud
from pathlib import Path
import make_cldf,source_meta,make_refs
from segments.tokenizer import Tokenizer
from tags import GRAMMATICAL_TAGS
ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/other/forms/raw_data'
HERE=RAW/'kharia_living_2026/whole-source-recovery'
OUT=HERE/'proposal'
def rows():return list(csv.reader((OUT/'20260911-kharia-living.csv').open()))
def test_snapshot_completeness_and_stable_legacy_fields():
 snapshot=json.loads((HERE/'lexical-snapshot-20260927.json').read_text())
 assert len(snapshot['entries'])==452 and len(snapshot['sentences'])==3
 compare=json.loads((HERE/'snapshot-lexical-comparison-20260927.json').read_text())
 assert not compare['missing_ids'] and not compare['lexical_field_differences']
 assert all(not v for v in compare['nonempty_other_lexical_fields'].values())
 old=[r for a in map(json.loads,(RAW/'20260911-kharia-living-audit.jsonl').read_text().splitlines()) for r in a['rows']]
 new=rows();indexed={r[10]:r for r in new}
 assert len(new)==len(indexed)==524 and len(old)==521
 for r in old:
  n=indexed[r[10]]
  assert all(n[i]==r[i] for i in [0,1,2,4,5,7,8,9,10,11,12,13])
  assert set(r[14].split())<=set(n[14].split())
 assert len([json.loads(x) for x in (OUT/'whole-audit.jsonl').read_text().splitlines()])==455
def test_whole_native_sentences_are_not_automatic_token_analyses():
 snap=json.loads((HERE/'lexical-snapshot-20260927.json').read_text())
 indexed={r[10]:r for r in rows()}
 for s in snap['sentences']:
  key='living-kharia2026:sentence:'+s['id'];r=indexed[key]
  assert r[2]==r[4]==ud.normalize('NFC',json.loads(s['text'])['default'])
  assert r[3]==json.loads(s['translation'])['en'] and not r[5]
  assert r[14]=='sentential multiword-expression'
  assert not any(k.startswith(key+':') for k in indexed)
  assert not any(r[i] for i in [11,12,13])
def test_source_metadata_scope_and_clean_labels():
 dec=json.loads((OUT/'metadata-decisions.json').read_text());indexed={r[10]:r for r in rows()}
 for d in dec:
  r=indexed[d['entry_key']];g=d['source_gloss'].strip().lower();ts=set(r[14].split())
  if g in {'first','second','third','fourth'}:assert {'num','ord'}<=ts
  if g=='fly (n)':assert r[3]=='fly' and 'noun' in ts
  if g=='her':assert not ts & {'poss','acc','gen'}
  if g=='you':assert 'second-person' in ts and not ts & {'sg','pl','du','2sg','2pl'}
  if g=='we 2 (not you)':assert r[3]=='we' and {'du','exclusive','first-person'}<=ts and not ts & {'pl','1pl'}
  if g.startswith('my '):assert not ts & {'1sg','sg','poss'}
  if g=='waters':assert 'pl' not in ts  # English plural alone is not source inflection evidence.
  if g.startswith('you two '):assert {'du','second-person','sentential'}<=ts
 for r in rows():assert all(t.startswith('dialect:') or t in GRAMMATICAL_TAGS for t in r[14].split())
def test_real_parser_profile_and_citation_contract(monkeypatch):
 meta=source_meta.SourceMeta([OUT/'20260911-kharia-living.yaml'])
 monkeypatch.setattr(source_meta,'load',lambda:meta)
 tok=Tokenizer(str(OUT/'kharia-living.txt'));monkeypatch.setitem(make_cldf.convertors,'kharia-living',tok)
 error=io.StringIO();parsed,stats=make_cldf.parse_file(str(OUT/'20260911-kharia-living.csv'),errors=error,name='20260911-kharia-living')
 assert not error.getvalue(),error.getvalue()
 indexed={r[10]:r for r in rows()}
 assert len(parsed)==524 and {r.entry_key for r in parsed}==set(indexed)
 for r in parsed:
  raw=indexed[r.entry_key];assert r.old_form==raw[2] and r.ipa==raw[5]
  assert r.native==raw[4]
  assert set(raw[14].split())<=set(r.tags.split())
  assert set(make_refs.source_ids(raw[7]))=={'living-kharia2026'}
  assert '\ufffd' not in r.form
  for norm in ('NFC','NFD'):
   converted=ud.normalize('NFC',tok(ud.normalize(norm,raw[2]),column='IPA').replace(' ','').replace('#',' '))
   assert converted==ud.normalize('NFC',r.form)
 keys=set(indexed)
 assert all(not r[11] or r[11] in keys for r in indexed.values())
