"""Verify existing dry-run output and current review manifests, without graph writes."""
import collections,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parents[2]))
from etymology_lab import rows_from_decisions
checkpoint=sys.argv[1]
s=(P/f'check-pass-{checkpoint}.log').read_text();v=json.JSONDecoder().raw_decode(s[s.index('{'):])[0]
assert v['dryRun'] and v['secondApplicationChanges']==0
z=json.load(open(P/'proposed-decisions.json'));assert z['status']=='pending-review';expected=rows_from_decisions(z['accepted'],'');actual=[]
for paths in json.load(open(P/'review-manifest-paths.json')).values():
 for f in paths:
  m=json.load(open(f));assert m['status']=='pending-review'
  for prop in m['proposals']:
   assert prop['status']=='pending-review';actual.extend(prop['assignments'])
key=lambda r:tuple(str(r.get(k,'') or '') for k in ['Form_ID','Etymon_ID','Kind','Rank','Pos'])
assert collections.Counter(map(key,actual))==collections.Counter(map(key,expected))
assert len(actual)==v['assignmentRows']
(P/f'checkpoint-{checkpoint}-validation.json').write_text(json.dumps({'manifestAgreement':True,'assignmentRows':len(actual),'validation':v},indent=2)+'\n')
print(json.dumps({'checkpoint':checkpoint,'manifestAgreement':True,'records':v['affectedRecords'],'rows':len(actual),'secondApplicationChanges':v['secondApplicationChanges']}))
