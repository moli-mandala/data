"""Inspect previously missed glyph-equivalent matches to already studied families."""
import json,csv,re,unicodedata,collections
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
src=(P/'prepare.py').read_text();exec('def norm'+src.split('def norm',1)[1].split('families=json.loads')[0]);oldnorm=norm
def norm(s):
 s=s.replace('ɡ','g').replace('ǰ','j').replace('č','c').replace('ʧ','c').replace('ʤ','j').replace('͡','').replace('ˈ','').replace('ˌ','').replace('ɑ','a')
 s=oldnorm(s)
 # Phonetic colon after a consonant and doubled letters both mark gemination.
 s=re.sub(r'([bcdfghjklmnpqrstvwxyzṭḍṇḷṛśṣ])ː',r'\1\1',s)
 return s
exec('families=json.loads'+src.split('families=json.loads',1)[1].split('inv=json.loads')[0])
linked={r['Child_ID'] for r in csv.DictReader((ROOT/'cldf/edges.csv').open()) if r['Rank']=='1'}
linked.update(r['Form_ID'] for r in csv.DictReader((ROOT/'data/etymology-assignments.csv').open()) if r['Status']=='accepted' and r['Rank']=='1')
examined={x['record']['ID'] for f in ['decisions.json','second-decisions.json','third-decisions.json'] for xs in json.loads((P/f).read_text()).values() for x in xs}
excluded=linked|examined
cs=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] in excluded:continue
 w=norm(r['Form']);ss={s.strip().lower() for s in r['Gloss'].split(';')}
 if re.search(r'[,;/ ()]',w):continue
 ii=[i for i,q in enumerate(families) if w in q['words'] and ss<=q['senses']]
 if ii:cs.append(dict(record=r,families=ii))
(P/'fourth-candidates.json').write_text(json.dumps(cs,ensure_ascii=False,indent=1))
with (P/'fourth-review.txt').open('w') as f:
 for i,q in enumerate(families):
  rs=[x['record'] for x in cs if i in x['families']]
  if not rs:continue
  f.write(f"\n{i}. {q['parent']} {q['gloss']} ({len(rs)})\n")
  f.write('; '.join(l+': '+', '.join(sorted({r['Form'] for r in rs if r['Language_ID']==l})) for l in sorted({r['Language_ID'] for r in rs}))+'\n')
print('Glyph-equivalent candidates',len(cs))
