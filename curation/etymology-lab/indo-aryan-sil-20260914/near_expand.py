"""One-edit whole-word candidate discovery against reviewed comparative families only."""
import json,csv,collections,re
from pathlib import Path
P=Path(__file__).resolve().parent
exec((P/'semantic_expand.py').read_text().split('idx=collections')[0])
idx=collections.defaultdict(list)
def keys(w):
 yield w
 for i in range(len(w)):yield w[:i]+w[i+1:]
for x in accepted:
 if not x['parent'][0].isdigit():continue
 w=norm(x['record']['Form'])
 if len(w)<3 or re.search(r'[,;/ ()\[\]]',w):continue
 for key in set(keys(w)):idx[key].append(x)
def edit1(a,b):
 if a==b:return True
 if abs(len(a)-len(b))>1:return False
 if len(a)==len(b):return sum(x!=y for x,y in zip(a,b))<=1
 if len(a)>len(b):a,b=b,a
 return any(a==b[:i]+b[i+1:] for i in range(len(b)))
raw={r['ID']:r for r in json.loads((P/'inventory.json').read_text())};held={x['record']['ID'] for f in ledger['decisionFiles'] for x in json.loads((P/f).read_text())['held']};cs=[]
for rr in csv.DictReader((P/'unresearched-records.csv').open()):
 if rr['ID'] in current|held:continue
 r=raw[rr['ID']];w=norm(r['Form']);g=senses(r['Gloss'])
 if len(w)<3 or re.search(r'[,;/ ()\[\]]',w):continue
 hits={}
 for key in set(keys(w)):
  for x in idx[key]:
   cw=norm(x['record']['Form'])
   if g and g<=senses(x['record']['Gloss']) and edit1(w,cw):hits[x['record']['ID']]=x
 if not hits:continue
 by=collections.defaultdict(list)
 for x in hits.values():by[x['parent']].append(x)
 cs.append(dict(record=r,parents=sorted(by),comparanda={k:sorted(v,key=lambda x:(norm(x['record']['Form'])!=w,x['record']['Language_ID']!=r['Language_ID']))[:3] for k,v in by.items()}))
(P/'near-expansion-candidates.json').write_text(json.dumps(cs,ensure_ascii=False,indent=1));grouped=collections.defaultdict(list)
for x in cs:grouped['/'.join(x['parents'])].append(x)
with (P/'near-expansion-review.txt').open('w') as f:
 for k,xs in sorted(grouped.items(),key=lambda kv:-len(kv[1])):f.write(k+' ('+str(len(xs))+'): '+'; '.join(sorted({x['record']['Language_ID']+' '+x['record']['Form']+' «'+x['record']['Gloss']+'»' for x in xs}))+'\n')
print(len(cs),len(grouped))
