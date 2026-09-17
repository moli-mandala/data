import json,csv,collections,re
from pathlib import Path
P=Path(__file__).resolve().parent
exec((P/'sixth_prepare.py').read_text().split('qs=[]')[0])
import unicodedata
strictnorm=norm
def norm(s):
 s=strictnorm(s)
 # Discovery only: retain retroflex and sibilant letters, discard stress/tone and articulatory modifiers.
 s=unicodedata.normalize('NFD',s)
 s=''.join(c for c in s if c not in '\u0301\u0300\u0302\u0304\u0306\u0308\u030b\u030c\u030f\u0311\u0318\u0319\u031a\u0324\u0325\u0329\u032a\u032f\u0330\u033a\u033b\u033d\u035c')
 return unicodedata.normalize('NFC',s).replace('ʷ','w').replace('ʲ','y').replace('ʸ','y').replace('ⁱ','i').replace('ᵘ','u').replace('ᵃ','a').replace('ᵊ','a').replace('ʰ','h').replace('ː','').replace('ˑ','')
ledger=json.loads((P/'pass-ledger.json').read_text());acc=[x for f in ledger['decisionFiles'] for x in json.loads((P/f).read_text())['accepted']]
idx=collections.defaultdict(list)
def glosses(r):return frozenset(g.strip().lower().rstrip('?!.') for g in r['Gloss'].split(';'))
for x in acc:
 r=x['record'];w=norm(r['Form']);g=glosses(r)
 if not x['parent'][0].isdigit() or len(w)<3 or len(re.findall('[bcdfghjklmnpqrstvwxyzṭḍṇḷṛśṣ]',w))<2 or re.search('[ ,;/()\[\]]',w):continue
 idx[w].append(x)
cs=[]
for r in csv.DictReader((P/'unresearched-records.csv').open()):
 w=norm(r['Form']);g=glosses(r);xs=[x for x in idx[w] if g<=glosses(x['record'])]
 if not xs:continue
 by=collections.defaultdict(list)
 for x in xs:by[x['parent']].append(x)
 cs.append(dict(record=r,parents=sorted(by),comparanda={k:v[:3] for k,v in by.items()}))
(P/'pass108-expansion-candidates.json').write_text(json.dumps(cs,ensure_ascii=False,indent=1))
groups=collections.defaultdict(list)
for x in cs:groups['/'.join(x['parents'])].append(x)
with (P/'pass108-expansion-review.txt').open('w') as f:
 for k,xs in sorted(groups.items(),key=lambda kv:-len(kv[1])):
  f.write(k+' ('+str(len(xs))+'): '+'; '.join(sorted({x['record']['Language_ID']+' '+x['record']['Form']+' «'+x['record']['Gloss']+'»' for x in xs}))+'\n')
print(len(cs),len(groups))
