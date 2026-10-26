import json,glob,unicodedata,re,difflib
from pathlib import Path
P=Path(__file__).resolve().parent
A=json.load(open(P/'survey-inventory.json'));Q=[]
for f in sorted(P.glob('discovery-*-results.json')):Q+=json.load(open(f))
assert len(A)==len(Q)==250

def norm(s):return re.sub('[^a-z]','',unicodedata.normalize('NFD',s.lower()))
def score(a,b):
 if norm(a)==norm(b):return 1
 return difflib.SequenceMatcher(None,norm(a),norm(b)).ratio()
for i,(r,q) in enumerate(zip(A,Q),1):
 cand=[z for z in q['rows'] if z['language_id']=='Kannada' and z.get('etymon_id')]
 cand.sort(key=lambda z:(score(r['Form'],z['word']), bool(norm(r['Gloss']) in norm(z['gloss']))),reverse=True)
 seen=set();out=[]
 for z in cand:
  k=(z['word'],z['etymon_id'])
  if k in seen:continue
  seen.add(k);out.append(f"{z['word']}:{z['gloss'][:23]}→{z['etymon_id']}")
  if len(out)==5:break
 print(f"{i:03} {r['Form'][:20]:20} {r['Gloss'][:20]:20} {q['count']:4} "+' | '.join(out))
