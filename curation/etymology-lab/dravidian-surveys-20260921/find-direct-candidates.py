import json,collections,sys,re,importlib.util
from pathlib import Path
p=Path(__file__).resolve().parent
from research_common import fold
index=collections.defaultdict(list)
queries=[]
for name in ['reflex-discovery-results.json','additional-discovery-results.json']:
 if (p/name).exists():queries.extend(json.load(open(p/name)))
for q in queries:
 for r in q['rows']:
  if not any(x['id']=='dedr' for x in r.get('references',[])):continue
  if not (r['origin_lemma_id'] or '').startswith('d'):continue
  index[(r['language_id'],fold(r['word']))].append(r)
known=set()  # Keep the full candidate pool: prepare.py handles overlaps reproducibly.
fams=collections.defaultdict(list)
for r in json.load(open(p/'survey-inventory.json')):
 if r['ID'] in known:continue
 g=r['Gloss'].lower().replace('ɡ','g')
 if len(g)<3 or re.search(r'[;/!]',g):continue
 matches={}
 for x in index.get((r['Language_ID'],fold(r['Form'])),[]):
  if re.search(r'(?<!\w)'+re.escape(g)+r'(?!\w)',x['gloss'] or '',re.I):matches[x['origin_lemma_id']]=x
 if len(matches)==1:
  parent,x=next(iter(matches.items()));fams[parent].append({'record':r,'comparison':x})
json.dump(fams,open(p/'direct-candidates.json','w'),ensure_ascii=False,indent=2)
for k,rr in sorted(fams.items(),key=lambda x:-len(x[1]))[:85]:
 c=collections.Counter((r['record']['Language_ID'],r['record']['Form'],r['record']['Gloss']) for r in rr)
 print(k,len(rr),'; '.join(f'{l} {f} “{g}” ({n})' for (l,f,g),n in list(c.items())[:12]))
