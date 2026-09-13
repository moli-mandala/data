#!/usr/bin/env python3
"""Write a reproducible source-vs-output review sample; optionally render PDF crops.

uv run --with pymupdf python .../sample_audit.py --seed 2026091213 --output DIR --pdf KEED_2018.pdf
Omit --pdf to emit only the pinned evidence/output sample (stdlib only).
"""
import argparse,gzip,json,random
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=2026091213);p.add_argument('--output',type=Path,required=True);p.add_argument('--pdf',type=Path);a=p.parse_args()
records=[json.loads(l) for l in gzip.open(Path(__file__).parent/'audit.jsonl.gz','rt')]
sample=random.Random(a.seed).sample(records,20);a.output.mkdir(parents=True,exist_ok=True)
(a.output/'sample.json').write_text(json.dumps(sample,ensure_ascii=False,indent=2))
if a.pdf:
 import pymupdf
 doc=pymupdf.open(a.pdf)
 for i,u in enumerate(sample,1):
  page=doc[u['physical_page']+28];col=u['physical_column']-1
  page.get_pixmap(matrix=pymupdf.Matrix(2,2),clip=pymupdf.Rect(60+col*199,55,256+col*199,680)).save(a.output/f'{i:02}.png')
print(json.dumps({'seed':a.seed,'keys':[u['key'] for u in sample]},indent=2))
