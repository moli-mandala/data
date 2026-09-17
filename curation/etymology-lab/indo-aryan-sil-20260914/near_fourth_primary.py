import json
from pathlib import Path
P=Path(__file__).resolve().parent
heads='3167 3197 3832 5679 6849 3865 10648 9502 12772 9209 14028 13291 6261 3244 6835 1111 9964 9757 142 11165'.split();found={}
for f in P.glob('*primary-articles.json'):
 d=json.loads(f.read_text())
 if isinstance(d,dict):
  for k in heads:
   if k in d and k not in found:found[k]=d[k]
assert set(heads)==set(found),set(heads)-set(found)
(P/'near-fourth-primary-articles.json').write_text(json.dumps(found,ensure_ascii=False,indent=1))
for k in heads:print(k,json.dumps(found[k],ensure_ascii=False))
