import json,csv,collections,unicodedata,re
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914')
verbs=['eat','drink','hear','listen','speak','walk','run','come','go','give','see','look','sleep','sit','die','kill','bite','burn','lie']
def sense(s):
 s=s.lower()
 for v in verbs:
  if re.search(r'\b'+v+r'\b',s):return {'listen':'hear','look':'see'}.get(v,v)
 return None
def norm(s):return unicodedata.normalize('NFC',s.strip())
idx=collections.defaultdict(list)
for f in json.loads((P/'pass-ledger.json').read_text())['decisionFiles']:
 for x in json.loads((P/f).read_text())['accepted']:
  r=x['record'];g=sense(r['Gloss'])
  if not g or x.get('components'):continue
  for w in re.split('[,;/]',r['Form']):
   if ' ' not in w.strip():idx[r['Language_ID'],g,norm(w)].append(x)
cs=[]
for r in csv.DictReader((P/'unresearched-records.csv').open()):
 g=sense(r['Gloss'])
 if not g or ',' not in r['Form']:continue
 ws=r['Form'].split(',');ms=[idx[r['Language_ID'],g,norm(w)] for w in ws]
 if not all(ms):continue
 ps=[{(x['parent'],x.get('kind','reflex')) for x in m} for m in ms];common=set.intersection(*ps)
 if len(common)!=1:continue
 parent,kind=next(iter(common));support=[next(x for x in m if x['parent']==parent and x.get('kind','reflex')==kind) for m in ms]
 cs.append(dict(record=r,parts=ws,parent=parent,kind=kind,comparanda=support,sense=g))
(P/'paired-inflection-candidates.json').write_text(json.dumps(cs,ensure_ascii=False,indent=1));print(len(cs))
for x in cs:print(x['parent'],x['record']['Language_ID'],x['record']['Form'],x['record']['Gloss'])
