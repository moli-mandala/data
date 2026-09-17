import json,csv,collections,re,unicodedata
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914')
def norm(s):return unicodedata.normalize('NFC',s.strip())
def gloss(s):return s.strip().lower().rstrip('?.!')
idx=collections.defaultdict(list)
for f in json.loads((P/'pass-ledger.json').read_text())['decisionFiles']:
 for x in json.loads((P/f).read_text())['accepted']:
  r=x['record']
  if x.get('components') or '/' in r['Form']:continue
  idx[r['Language_ID'],norm(r['Form']),gloss(r['Gloss'])].append(x)
cs=[]
for r in csv.DictReader((P/'unresearched-records.csv').open()):
 if '/' not in r['Form']:continue
 ws=r['Form'].split('/')
 if any(not norm(w) for w in ws):continue
 matches=[idx[r['Language_ID'],norm(w),gloss(r['Gloss'])] for w in ws]
 if not all(matches):continue
 ps=[{(x['parent'],x.get('kind','reflex')) for x in ms} for ms in matches];common=set.intersection(*ps)
 if len(common)!=1:continue
 parent,kind=next(iter(common));support=[next(x for x in ms if x['parent']==parent and x.get('kind','reflex')==kind) for ms in matches]
 cs.append(dict(record=r,parts=ws,parent=parent,kind=kind,comparanda=support))
(P/'alternant-candidates.json').write_text(json.dumps(cs,ensure_ascii=False,indent=1));print(len(cs))
for x in cs:print(x['parent'],x['kind'],x['record']['Language_ID'],x['record']['Form'],x['record']['Gloss'])
