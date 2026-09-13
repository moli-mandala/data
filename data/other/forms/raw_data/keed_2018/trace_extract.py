import sys,json,gzip,collections
import os
import pymupdf as f
from pathlib import Path
root=Path(os.environ['KEED_CACHE'])
d=f.open(os.environ['KEED_PDF'])
assert len(d)==971
with gzip.open(root/'keed-traces.jsonl.gz','wt') as out:
 for p in range(29,len(d)):
  spans=[]
  for s in d[p].get_texttrace():
   spans.append(dict(font=s['font'],size=round(s['size'],4),bbox=[round(x,4) for x in s['bbox']],chars=[[c[0],c[1],*[round(x,4) for x in c[2]],*[round(x,4) for x in c[3]]] for c in s['chars']]))
  out.write(json.dumps(dict(page=p+1,spans=spans),ensure_ascii=False)+'\n')
  if p%100==0:print(p,flush=True)
