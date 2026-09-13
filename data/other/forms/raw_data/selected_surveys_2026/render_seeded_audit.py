import json,random,sys
from pathlib import Path
from PIL import Image,ImageDraw
import pdfplumber
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R));import koraga
for k in ['angika','majhi','koraga']:
 records=koraga.records() if k=='koraga' else json.loads((R/(k+'-cells.json')).read_text())
 sample=random.Random(2026091101).sample(records,20)
 (R/(k+'-seeded-audit.json')).write_text(json.dumps({'seed':2026091101,'records':sample,'result':'pending'},ensure_ascii=False,indent=2)+'\n')
 with pdfplumber.open(R/(k+'.pdf')) as pdf:
  items=[]
  for r in sample:
   p=pdf.pages[r['pdf_page']-1]
   if k=='koraga':
    import difflib
    o=json.loads((R/'koraga-ocr-lines.json').read_text());a=max([a for a in o if a['printed_page']==r['printed_page']],key=lambda a:difflib.SequenceMatcher(None,a['text'],r['initial_collation']).ratio());b=a['bbox'];bb=(b[0]-2,b[1]-3,min(p.width,b[2]+2),b[3]+3);label=f"p{r['printed_page']} e{r['item']} "+r['text']
   else:
    b=r['bbox'];bb=(max(0,b[0]-1),max(0,b[1]-2),min(p.width,b[2]+1),min(p.height,b[3]+2));label=f"p{r['printed_page']} i{r['item']} c{r['column']} "+r['text']
   im=p.crop(bb).to_image(resolution=230).original
   if im.width>1150:im=im.resize((1150,round(im.height*1150/im.width)))
   items.append((label,im))
  for start in [0,10]:
   canvas=Image.new('RGB',(1200,1700),'#eeeeee');draw=ImageDraw.Draw(canvas)
   for n,(label,im) in enumerate(items[start:start+10]):
    y=n*170;draw.text((5,y+2),label,fill='black');canvas.paste(im,(5,y+25))
   canvas.save(f'/tmp/{k}-audit-{start//10}.png')
