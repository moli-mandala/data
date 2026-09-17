"""Broaden typography-aware discovery only; no accepted rows are generated here."""
import json,csv,re,collections
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
s=(P/'sixth_prepare.py').read_text();exec(s.split('qs=[]')[0])
ids={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())}
targets=[r for r in json.loads((P/'inventory.json').read_text()) if r['ID'] in ids]
keys={(r['Language_ID'],norm(r['Form'])) for r in targets if not re.search(r'[,;/ ()]',r['Form'])}
edges={}
for r in csv.DictReader((ROOT/'cldf/edges.csv').open()):
 if r['Rank']=='1' and r['Kind'] in {'reflex','borrowed'}:edges[r['Child_ID']]=dict(parent=r['Parent_ID'],kind=r['Kind'],citation=r['Source'],evidence=r['Note'])
for r in csv.DictReader((ROOT/'data/etymology-assignments.csv').open()):
 if r['Status']=='accepted' and r['Rank']=='1' and r['Kind'] in {'reflex','borrowed'}:edges[r['Form_ID']]=dict(parent=r['Etymon_ID'],kind=r['Kind'],citation=r['Source'],evidence=r['Notes'])
c=collections.defaultdict(list)
for r in csv.DictReader((ROOT/'cldf/forms.csv').open()):
 k=(r['Language_ID'],norm(r['Form']))
 if r['Redirect'] or k not in keys or r['ID'] not in edges:continue
 e=edges[r['ID']]
 if e['parent']==r['ID']:continue
 c[k].append(dict(record={k:r[k] for k in ['ID','Language_ID','Form','Gloss','Source','Description','Tags']},edge=e))
def senses(s):return {x.strip().lower() for x in re.split(r'; |, ',s) if x.strip()}
out=[]
for r in targets:
 if re.search(r'[,;/ ()]',r['Form']):continue
 rs=senses(r['Gloss']);hits=[x for x in c[(r['Language_ID'],norm(r['Form']))] if rs and rs<=senses(x['record']['Gloss'])]
 if hits:out.append(dict(record=r,comparanda=hits,parents=sorted({x['edge']['parent'] for x in hits})))
(P/'expanded-exact-candidates.json').write_text(json.dumps(out,ensure_ascii=False,indent=1))
g=collections.defaultdict(list)
for x in out:g[tuple(x['parents'])].append(x)
with (P/'expanded-exact-review.txt').open('w') as f:
 for ps,xs in sorted(g.items(),key=lambda z:-len(z[1])):
  f.write('/'.join(ps)+f' ({len(xs)}): '+'; '.join(sorted({x['record']['Language_ID']+' '+x['record']['Form']+' «'+x['record']['Gloss']+'»' for x in xs}))+'\n')
print('records',len(out),'parent groups',len(g));print((P/'expanded-exact-review.txt').read_text()[:15000])
