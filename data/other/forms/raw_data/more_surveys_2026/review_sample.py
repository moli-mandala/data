import json,random,sys,pathlib,pdfplumber
from PIL import Image,ImageDraw,ImageFont
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent.parent));import more_surveys as m
# Rendering is optional; seeded JSON samples remain platform-independent.
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--render',action='store_true');ap.add_argument('--font',default='/System/Library/Fonts/Supplemental/Arial Unicode.ttf');args=ap.parse_args()
font=ImageFont.truetype(args.font,15) if args.render else None
for k in m.SOURCES:
 a=[json.loads(l) for l in open(m.RAW/(k+'-audit.jsonl')) if json.loads(l)['readings']]
 sample=random.Random(20260920).sample(a,20);(m.RAW/(k+'-sample.json')).write_text(json.dumps(sample,ensure_ascii=False,indent=2)+'\n')
 if not args.render:continue
 sheet=Image.new('RGB',(1400,2000),'white');d=ImageDraw.Draw(sheet)
 with pdfplumber.open('/tmp/more-'+k+'.pdf') as pdf:
  for i,x in enumerate(sample):
   r=x['raw'];box=r['bbox'];pg=pdf.pages[r['pdf_page']-1]
   crop=pg.crop((max(0,box[0]-1),max(0,box[1]-1),min(pg.width,box[2]+1),min(pg.height,box[3]+1))).to_image(resolution=160).original
   crop.thumbnail((370,75));y=i*100
   d.text((0,y),str(i+1)+' '+str(r['item'])+':'+str(r['column'])+' '+m.space(r['gloss'])[:35],font=font,fill='black')
   sheet.paste(crop,(350,y))
   display=' | '.join(z['row'][2]+' — '+z['row'][3] for z in x['readings']);d.text((740,y),display[:77],font=font,fill='black')
   if len(display)>77:d.text((740,y+23),display[77:154],font=font,fill='black')
   d.line((0,y+97,1400,y+97),fill='gray')
 sheet.save('/tmp/more-audit-'+k+'.png');print(k,flush=True)
