"""Reproduce cached page OCR in a scratch directory (requires PyMuPDF, Pillow, numpy, Tesseract 5.5.2)."""
import argparse, shutil, hashlib
import json,subprocess,csv,io,concurrent.futures
from pathlib import Path
from PIL import Image,ImageOps
import numpy as np
ROOT=Path(__file__).parent
TESS=shutil.which('tesseract') or '/opt/homebrew/bin/tesseract'

def run(task):
 name,page=task
 src=ROOT/f'ocr/{name}/{page:03}.png'
 out=ROOT/f'deskew/{name}';out.mkdir(parents=True,exist_ok=True)
 dest=out/f'{page:03}.json'
 if dest.exists():return name,page,'cached'
 im=Image.open(src).convert('L');w,h=im.size
 small=im.crop((int(w*.10),int(h*.13),int(w*.93),int(h*.89)))
 small.thumbnail((650,1100))
 def score(a):
  x=np.array(small.rotate(a,resample=Image.Resampling.BILINEAR,fillcolor=255))<145
  s=x.sum(axis=1);return ((s[1:]-s[:-1])**2).sum()
 angle=max([a/10 for a in range(-25,26)],key=score)
 im=im.rotate(angle,resample=Image.Resampling.BICUBIC,fillcolor=255)
 im.save(out/f'{page:03}.png')
 txt=subprocess.run([TESS,str(out/f'{page:03}.png'),'stdout','-l','script/Latin','--psm','6','tsv'],capture_output=True,text=True,check=True).stdout
 (out/f'{page:03}.tsv').write_text(txt)
 words=[]
 for r in csv.DictReader(io.StringIO(txt),delimiter='\t',quoting=csv.QUOTE_NONE):
  if r['level']=='5' and r.get('text','').strip():words.append(dict(text=r['text'],x=int(r['left']),y=int(r['top']),w=int(r['width']),h=int(r['height']),confidence=float(r['conf'])))
 dest.write_text(json.dumps(dict(width=w,height=h,angle=angle,words=words),ensure_ascii=False));return name,page,angle
if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('scratch',type=Path);args=ap.parse_args();ROOT=args.scratch
 tasks=[('kudali',p) for p in range(105,161)]+[('konkani',p) for p in range(128,149)]
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  for r in pool.map(run,tasks):print(*r,flush=True)
