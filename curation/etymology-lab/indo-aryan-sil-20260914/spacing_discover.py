import json,csv,re,collections
from pathlib import Path
P=Path(__file__).resolve().parent
exec((P/'sixth_prepare.py').read_text().split('qs=[]')[0])
acc=[x for f in json.loads((P/'pass-ledger.json').read_text())['decisionFiles'] for x in json.loads((P/f).read_text())['accepted']]
def senses(r):return {x.strip().lower().rstrip('?!.') for x in r['Gloss'].split(';')}
idx=collections.defaultdict(list)
for x in acc:
 w=norm(x['record']['Form'])
 if not x['parent'][0].isdigit() or x.get('components') or re.search(r'[ ,;/()\[\]]',w) or len(w)<3:continue
 idx[w].append(x)
cs=[]
for r in csv.DictReader(open(P/'unresearched-records.csv')):
 w=norm(r['Form'])
 if ' ' not in w or re.search(r'[,;/()\[\]]',w):continue
 w=w.replace(' ','');xs=[x for x in idx[w] if senses(r)<=senses(x['record'])]
 by=collections.defaultdict(list)
 for x in xs:by[x['parent']].append(x)
 if by:cs.append(dict(record=r,parents=sorted(by),comparanda={k:v[:2] for k,v in by.items()}))
(P/'spacing-candidates.json').write_text(json.dumps(cs,ensure_ascii=False,indent=1))
g=collections.defaultdict(list)
for x in cs:g['/'.join(x['parents'])].append(x['record'])
with (P/'spacing-review.txt').open('w') as f:
 for k,rs in sorted(g.items(),key=lambda x:-len(x[1])):f.write(k+' '+str(len(rs))+': '+'; '.join(sorted({r['Language_ID']+' '+r['Form']+' «'+r['Gloss']+'»' for r in rs}))+'\n')
print(len(cs),len(g))
