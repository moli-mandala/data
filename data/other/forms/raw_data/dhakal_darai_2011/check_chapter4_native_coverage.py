"""Physical span coverage diagnostic; no unmatched line is silently excluded."""
from pathlib import Path
import json,collections
P=Path(__file__).resolve().parent
load=lambda n:[json.loads(x) for x in (P/n).read_text().splitlines()]
refs=collections.defaultdict(set)
for r in load('chapter4-expressions-first-reading.jsonl'):
 p=r['printed_page']
 for li in r['surface_lines']:refs[p,li].add(r['source_unit_key'])
 for x in r['interlinear_evidence']:refs[p,x['line']].add(r['source_unit_key'])
for fn in ['chapter4-tables-first-reading.jsonl','chapter4-prose-first-reading.jsonl']:
 for r in load(fn):
  for li in r['source_lines']:refs[r['printed_page'],li].add(r['source_unit_key'])
rows=[];txt=[]
for p in range(77,125):
 pg=json.loads((P/'chapter4-native'/f'p{p:03}.json').read_text()); missing=[]
 for li in pg['lines']:
  ipa=[s for s in li['spans'] if 'IPA93' in s['font'] and s['decoded'].strip()]
  if ipa and not refs[p,li['line']]:
   missing.append(dict(line=li['line'],spans=ipa,text=''.join(s['decoded'] for s in li['spans'])))
   txt.append(f"p{p} l{li['line']}: "+''.join(s['decoded'] for s in li['spans']))
 rows.append(dict(printed_page=p,unassigned_native_lines=missing,status='pending explicit disposition' if missing else 'all IPA-bearing lines represented; lexical and interlinear correctness review still required'))
(P/'chapter4-native-coverage-diagnostic.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
(P/'chapter4-unassigned-native-lines.txt').write_text('\n'.join(txt)+'\n')
print('\n'.join(txt));print(len(txt),'unassigned IPA-bearing lines')
