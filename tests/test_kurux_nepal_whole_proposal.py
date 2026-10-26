"""Bounded whole-source Kurux proposal checks; no compiled build."""
import csv,io,json,unicodedata as ud
from pathlib import Path
import make_cldf,source_meta,make_refs
from segments.tokenizer import Tokenizer
from tags import GRAMMATICAL_TAGS
ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'data/other/forms/raw_data/kurux_nepal_2022/whole-source-recovery'
OUT=HERE/'proposal'
BUCKETS=('wordlist','grammar','native','roman')
def rows(bucket):return list(csv.reader((OUT/f'{bucket}.csv').open()))
def audit():return [json.loads(s) for s in (OUT/'audit.jsonl').read_text().splitlines()]
def test_complete_physical_wordlist_and_all_legacy_keys():
 source=[json.loads(s) for s in (HERE/'wordlist-reviewed.jsonl').read_text().splitlines()]
 assert len(source)==2028 and len({r['source_unit_key'] for r in source})==2028
 assert {r['prompt'] for r in source}==set(range(1,339))
 assert len({r['lect'] for r in source})==6
 old=list(csv.reader((HERE/'legacy-before-whole-recovery.csv').open()))
 new={r[10]:r for b in BUCKETS for r in rows(b)}
 assert len(old)==1336 and {r[10] for r in old}<=new.keys()
 physical={r['source_unit_key']:r for r in audit() if r.get('prompt')}
 assert len(physical)==2028
 assert sum(not r['entry_keys'] for r in physical.values())==74
 assert all(k in new for r in physical.values() for k in r['entry_keys'])
def test_source_geometry_and_native_mark_preservation():
 word={r[10]:r for r in rows('wordlist')}
 assert word['kurux-nepal:1:kurux_lochani:1'][2]=='d̤ɛ'
 assert 'ᵊ' in word['kurux-nepal:328:kurux_tokla:1'][2]
 assert word['kurux-nepal:181:kurux_pathibari:1'][2]==ud.normalize('NFC','bʰəɖhĩj̃a')
 native={r[10]:r for r in rows('native')}
 assert native['kurux-nepal:p66:narrative:28:word1'][2]==native['kurux-nepal:p66:narrative:28:word1'][4]=='पिंंटकाय'
 assert not native['kurux-nepal:p66:narrative:28:word1'][5]
 a={r.get('source_unit_key',r.get('key')):r for r in audit()}
 shared=[a[f'kurux-nepal:p67:narrative:35:word{i}']['entry_keys'] for i in (2,3,4)]
 assert shared[0]==shared[1]==shared[2]
 assert native[shared[0][0]][2:4]==['आ उल्ला तिम','since that day']
 assert len([r for r in a if ':orthography:line' in r])==11
def test_explicit_grammar_and_closed_reference_keys():
 allrows=[r for b in BUCKETS for r in rows(b)];keys={r[10] for r in allrows}
 assert len(keys)==len(allrows)
 for r in allrows:
  assert all(t.startswith('dialect:') or t in GRAMMATICAL_TAGS for t in r[14].split())
  assert not r[11] or r[11] in keys
  assert set(make_refs.source_ids(r[7]))=={'shackelford-swenson-chaudhary-maggard2022kurux'}
 for r in rows('wordlist'):
  prompt=int(r[10].split(':')[1]);tags=set(r[14].split())
  if 204<=prompt<=217:assert 'num' in tags
  if r[3].startswith('don’t '):assert {'verb','neg','impv','second-person','sg','2sg'}<=tags
 native={r[10]:r for r in rows('native')}
 analyses={'EMPH':'emph','POSS':'poss','POSTP':'postp','PREP':'prep','beat-PT':'pret','killed-2P':'second-person'}
 for a in audit():
  if a.get('gloss') in analyses and ':narrative:' in a.get('source_unit_key',''):
   for key in a['entry_keys']:
    assert analyses[a['gloss']] in native[key][14].split()
    assert native[key][3]!=a['gloss'] and a['gloss'] in native[key][6]
def test_isolated_affix_scope_and_lexical_glosses():
 grammar={r[10]:r for r in rows('grammar')};seen=set()
 for unit in audit():
  if unit.get('kind') not in {'suffix-group','prefix-suffix-cooccurrence-group'}:continue
  for key in unit['entry_keys']:
   seen.add(key);r=grammar[key];tags=set(r[14].split())
   assert 'verb' not in tags and tags & {'prefix','suffix'}
   assert r[3]=='' and 'Source analysis:' in r[6]
   if unit['kind']=='suffix-group':assert {'pret','third-person','sg','3sg'}<=tags
   else:assert {'neg','impv','second-person','sg','2sg'}<=tags
 assert seen
 for r in grammar.values():
  if set(r[14].split()) & {'prefix','suffix','affix'}:
   assert 'verb' not in r[14].split()
   assert r[3] in {'','at'}
 old={r[10]:r for b in BUCKETS for r in csv.reader((HERE/'revision1/proposal'/f'{b}.csv').open())}
 new={r[10]:r for b in BUCKETS for r in rows(b)}
 assert old.keys()==new.keys()
 for key,before in old.items():
  after=new[key]
  assert all(before[i]==after[i] for i in range(15) if i not in {3,6,14})
  if key not in seen:assert before==after
def test_actual_parser_originals_profiles_and_keys(monkeypatch):
 meta=source_meta.SourceMeta([OUT/f'{b}.yaml' for b in BUCKETS]);monkeypatch.setattr(source_meta,'load',lambda:meta)
 for name in ('phonetic','native','roman'):
  profile='kurux-nepal-whole-'+name;monkeypatch.setitem(make_cldf.convertors,profile,Tokenizer(str(OUT/(profile+'.txt'))))
 for bucket in BUCKETS:
  error=io.StringIO();parsed,_=make_cldf.parse_file(str(OUT/f'{bucket}.csv'),errors=error,name=bucket)
  assert not error.getvalue(),error.getvalue()
  raw={r[10]:r for r in rows(bucket)}
  assert len(parsed)==len(raw) and {r.entry_key for r in parsed}==raw.keys()
  for r in parsed:
   original=raw[r.entry_key]
   assert r.old_form==original[2] and r.native==original[4] and r.ipa==original[5]
   assert set(original[14].split())<=set(r.tags.split())
   assert '\ufffd' not in r.form
   if bucket in {'native','roman'}:assert ud.normalize('NFC',r.form)==ud.normalize('NFC',original[2])
   profile='kurux-nepal-whole-'+('phonetic' if bucket in {'wordlist','grammar'} else bucket)
   tok=make_cldf.convertors[profile]
   for norm in ('NFC','NFD'):
    converted=ud.normalize('NFC',tok(ud.normalize(norm,original[2]),column='IPA').replace(' ','').replace('#',' '))
    assert converted==ud.normalize('NFC',r.form)
