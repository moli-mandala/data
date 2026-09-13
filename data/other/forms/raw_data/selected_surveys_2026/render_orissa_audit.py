"""Render a fresh seeded source-versus-output OCR audit and unusual cells."""
import json,random,sys
from pathlib import Path
from PIL import Image,ImageDraw
from pypdf import PdfReader
import orissa
R=Path(__file__).resolve().parent
seed=int(sys.argv[1]) if len(sys.argv)>1 else 2026091102
rr,_=orissa.records();sample=random.Random(seed).sample(rr,20)
(R/f'orissa-seeded-audit-{seed}.json').write_text(json.dumps({'seed':seed,'records':sample,'parsed':[orissa.parse(r) for r in sample],'result':'pending'},ensure_ascii=False,indent=2)+'\n')
p=PdfReader(R/'orissa-official.pdf');images={};items=[]
for r in sample:
 if r['pdf_page'] not in images:images[r['pdf_page']]=p.pages[r['pdf_page']-1].images[0].image.convert('RGB').rotate(r['rotation_degrees'],resample=Image.Resampling.BICUBIC,fillcolor='white')
 im=images[r['pdf_page']];x0,y0,x1,y1=r['bbox'];crop=im.crop((x0,y0,x1,y1))
 label=f"i{r['item']} c{r['column']} p{r['printed_page']} EN: {r['eng']} | LAT: {r['latin']} | gloss: {r['gloss']}"
 items.append((label,crop))
for start in [0,10]:
 canvas=Image.new('RGB',(1400,1800),'#eeeeee');d=ImageDraw.Draw(canvas)
 for n,(label,im) in enumerate(items[start:start+10]):
  y=n*180;d.text((5,y+2),label,fill='black');canvas.paste(im,(5,y+25))
 canvas.save(f'/tmp/orissa-audit-{seed}-{start//10}.png')
if seed!=2026091102:sys.exit(0)
# English fields needing source collation: blank, noisy, or ambiguous abbreviations.
ids=[25,27,78,82,83,84,87,101,102,122,138,139,140,152,154,157,159,192,193,201,202,208,252,255,260,263,302,308,346,350,351,353,363,365,367,380,389,399,408,409,414,457,467,477,485,499,500,507,508,515,526,531,557,573,585,611,620,626,630,637,679,688,713,764,782,796,817,819,839,847,853,858,864,874,887,896,897,906,907,908,910,914,925,948,953,961,985,986,990,993,1000,1002,1007,1012]
with (R/'orissa-gloss-crop-index.json').open('w') as f:json.dump(ids,f)
for start in range(0,len(ids),20):
 canvas=Image.new('RGB',(1400,1600),'#eeeeee');d=ImageDraw.Draw(canvas)
 for n,i in enumerate(ids[start:start+20]):
  r=next(r for r in rr if r['item']==i);pg=json.loads((R/f'orissa-p{r["pdf_page"]}-cells.json').read_text());c=next(c for c in pg['cells'] if c['row']==r['row'] and c['column']==1)
  if r['pdf_page'] not in images:images[r['pdf_page']]=p.pages[r['pdf_page']-1].images[0].image.convert('RGB').rotate(r['rotation_degrees'],resample=Image.Resampling.BICUBIC,fillcolor='white')
  im=images[r['pdf_page']].crop(c['bbox']);x=n%2*700;y=n//2*160;d.text((x+5,y+2),str(i),fill='black');canvas.paste(im,(x+5,y+20))
 canvas.save(f'/tmp/orissa-gloss-review-{start//20}.png')
