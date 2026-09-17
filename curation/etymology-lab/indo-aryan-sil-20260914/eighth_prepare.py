"""Discover previously researched families under explicit survey verb prompt variants."""
import json,csv,re,unicodedata
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
s=(P/'prepare.py').read_text();exec(s.split('inv=json.loads')[0]);oldnorm=norm
def norm(s):return oldnorm(s.replace('ɡ','g').replace('ǰ','j').replace('č','c').replace('ˈ','').replace('ˌ',''))
verbs={'go!, he went':('go','goes','went'),'come!, he came':('come','comes','came'),'give!, he gave':('give','gives','gave'),'eat!, he ate':('eat','eats','ate'),'drink!, he drank':('drink','drinks','drank'),'sleep!, he slept':('sleep','sleeps','slept'),'lie down!, he lay down':('lie down','lies down','lay down'),'walk!, he walked':('walk','walks','walked'),'speak!, he spoke':('speak','speaks','spoke'),'listen!, he heard':('listen','listens','listened','hear','hears','heard')}
def compatible(g,v):
 parts=re.split(r'[,;/]',g.lower().strip())
 for t in parts:
  t=t.strip().strip('!.').strip()
  t=re.sub(r'^(?:\(you\)|\(he\)|you|he|to)\s+','',t)
  t=t.strip().strip('!.')
  if t not in v:return False
 return True
# Only newly covered prompts, only already researched explicit forms, no stem regex.
qs=[]
for i,q in enumerate(families):
 if q['gloss'] in verbs:qs.append((i,q,{norm(w) for w in q['forms'] if not re.search(r'[,;/ ()]',w)}))
cs=[]
for r in csv.DictReader((P/'unresearched-records.csv').open()):
 w=norm(r['Form'])
 if re.search(r'[,;/ ()]',w):continue
 ii=[i for i,q,ws in qs if w in ws and compatible(r['Gloss'],verbs[q['gloss']])]
 if ii:cs.append(dict(record=r,families=ii))
ids={x['record']['ID'] for x in cs};byid={r['ID']:r for r in json.loads((P/'inventory.json').read_text()) if r['ID'] in ids}
for x in cs:x['record']=byid[x['record']['ID']]
(P/'eighth-candidates.json').write_text(json.dumps(cs,ensure_ascii=False,indent=1))
for i,q,ws in qs:
 rs=[x['record'] for x in cs if i in x['families']]
 if rs:print(i,q['parent'],q['gloss'],len(rs));print('; '.join(l+': '+', '.join(sorted({r['Form'] for r in rs if r['Language_ID']==l})) for l in sorted({r['Language_ID'] for r in rs})))
