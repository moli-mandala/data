"""Discovery only: compare final central/high vowels within language and sense."""
import json,re,collections
from pathlib import Path
from research_common import fold,gloss
P=Path(__file__).resolve().parent
z=json.load(open(P/'proposed-decisions.json'));a=json.load(open(P/'unexamined-inventory.json'))
def vf(s):
 s=fold(s)
 return re.sub(r'[uɨɯə]+$','V',s)
index=collections.defaultdict(list)
for x in z['accepted']:
 if x.get('kind')!='reflex' or x.get('components') or ' ' in x['record']['Form']:continue
 r=x['record'];index[(r['Language_ID'],gloss(r['Gloss']),vf(r['Form']))].append(x)
out=[]
for r in a:
 if ' ' in r['Form'] or not re.search('[uɨɯə]$',fold(r['Form'])):continue
 matched=index.get((r['Language_ID'],gloss(r['Gloss']),vf(r['Form'])),[])
 if matched:
  parents={x['parent'] for x in matched}
  out.append({'record':r,'candidate_parents':sorted(parents),'comparators':[{'record_id':x['record']['ID'],'form':x['record']['Form'],'parent':x['parent'],'evidence':x['evidence']} for x in matched[:5]]})
(P/'final-vowel-discovery.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print('candidates',len(out));print(collections.Counter(p for x in out for p in x['candidate_parents']).most_common(30))
