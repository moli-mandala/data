"""Render a fresh seeded audit: audit_sample.py SEED /path/to/Zoller.pdf (requires PyMuPDF and Pillow)."""
import json,gzip,random,sys,pymupdf as f
from pathlib import Path
from PIL import Image,ImageDraw
R=Path(__file__).parent
a=[r for r in map(json.loads,gzip.open(R/'audit.jsonl.gz','rt')) if r['status']=='ingested']
seed=int(sys.argv[1]) if len(sys.argv)>1 else 919
ss=random.Random(seed).sample(a,20);rr={r['key']:r for r in json.load(gzip.open(R/'records.json.gz','rt'))};pdf=f.open(sys.argv[2])
for j,s in enumerate(ss):
 r=rr[s['record']];form=s['form'];line=next((l for l in r['lines'] if form in l['text']),None)
 if line is None:line=next((l for l in r['lines'] if form.split()[0] in l['text']),r['lines'][0])
 p=line.get('page',r['page']);y=line['y'];page=pdf[p+28]
 px=page.get_pixmap(matrix=f.Matrix(2.5,2.5),clip=f.Rect(48,max(55,y-19),435,min(630,y+25)))
 im=Image.frombytes('RGB',(px.width,px.height),px.samples);out=Image.new('RGB',(1000,170),'white');out.paste(im,(10,30));ImageDraw.Draw(out).text((10,4),f'{j+1}: p.{p} {s["label"]} / {s["key"]}',fill='black');out.save(f'/tmp/zoller-sample-{j+1}.png')
 s['sample_printed_page']=p;s['sample_y']=y
for j in range(4):
 sheet=Image.new('RGB',(1000,850),'white')
 for k in range(5):sheet.paste(Image.open(f'/tmp/zoller-sample-{j*5+k+1}.png'),(0,k*170))
 sheet.save(f'/tmp/zoller-sample-sheet-{j+1}.png')
json.dump(ss,open(f'/tmp/zoller-sample{seed}.json','w'),ensure_ascii=False,indent=2)
for j,s in enumerate(ss):print(j+1,s['label'],s['form'],':',s['gloss'])
