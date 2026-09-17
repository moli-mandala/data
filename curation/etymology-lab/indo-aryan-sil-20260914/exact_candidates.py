"""Same-language, same-form and compatible-sense discovery, with full graph provenance."""
import csv,json,collections,unicodedata,re
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
def norm(s):return unicodedata.normalize('NFC',s).replace('ʰ','h').replace('ʱ','h').replace('ɦ','h').replace('ɾ','r').strip().lower()
def senses(s):return {x.strip().lower() for x in re.split(r'; |, ',s) if x.strip()}
inv=json.loads((P/'inventory.json').read_text());targets=[r for r in inv if not r['previously_linked']]
keys={(r['Language_ID'],norm(r['Form'])) for r in targets}
edges=collections.defaultdict(list)
for r in csv.DictReader((ROOT/'cldf/edges.csv').open()):
 if r['Rank']=='1':edges[r['Child_ID']].append(r)
for r in csv.DictReader((ROOT/'data/etymology-assignments.csv').open()):
 if r['Status']=='accepted' and r['Rank']=='1':
  e=dict(Child_ID=r['Form_ID'],Parent_ID=r['Etymon_ID'],Kind=r['Kind'],Rank=r['Rank'],Source=r['Source'],Note=r['Notes'],Pos=r['Pos'])
  if e['Kind'] in ['reflex','borrowed']:edges[r['Form_ID']]=[e]
  elif e not in edges[r['Form_ID']]:edges[r['Form_ID']].append(e)
c=collections.defaultdict(list)
for r in csv.DictReader((ROOT/'cldf/forms.csv').open()):
 if r['Redirect'] or (r['Language_ID'],norm(r['Form'])) not in keys:continue
 es=edges[r['ID']]
 if len(es)!=1 or es[0]['Kind']!='reflex' or not re.fullmatch(r'[0-9]+[a-z]?(?:-\d+x?)?',es[0]['Parent_ID']):continue
 c[(r['Language_ID'],norm(r['Form']))].append((r,es[0]))
out=[]
for r in targets:
 if re.search(r'[,;/ ()]',r['Form']):continue
 rs=senses(r['Gloss']);hits=[(a,e) for a,e in c[(r['Language_ID'],norm(r['Form']))] if rs and rs<=senses(a['Gloss'])]
 if hits:out.append(dict(record=r,comparanda=[dict(record=a,edge=e) for a,e in hits],parents=sorted({e['Parent_ID'] for a,e in hits})))
(P/'exact-candidates.json').write_text(json.dumps(out,ensure_ascii=False,indent=1))
g=collections.defaultdict(list)
for x in out:
 if len(x['parents'])==1:g[x['parents'][0]].append(x)
with (P/'exact-candidate-review.txt').open('w') as f:
 for par,xs in sorted(g.items(),key=lambda x:-len(x[1])):
  f.write(par+f' ({len(xs)}): '+'; '.join(l+': '+', '.join(sorted({x['record']['Form']+' ‘'+x['record']['Gloss']+'’' for x in xs if x['record']['Language_ID']==l})) for l in sorted({x['record']['Language_ID'] for x in xs}))+'\n')
print('matches',len(out),'unique parent',sum(len(x['parents'])==1 for x in out),'parents',len(g))
