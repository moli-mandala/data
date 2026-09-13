import json,re,unicodedata,difflib,sys
from pathlib import Path
from PIL import Image,ImageDraw
import pdfplumber
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R));import koraga
ocr=json.loads((R/'koraga-ocr-lines.json').read_text())
def norm(s):return re.sub('[^a-z: ,]','',unicodedata.normalize('NFD',s).lower())
out=[];pics=[]
with pdfplumber.open(R/'koraga.pdf') as pdf:
 rendered={}
 for r in koraga.records():
  raw=r['initial_collation'];body=re.split(r': (?![omt](?:,|$))',raw,maxsplit=1)[0]
  wanted=set()
  for mt in koraga.HEAD.finditer(body):
   for j,c in enumerate(mt.group(1)):
    if c=='i' and (j+1==len(mt.group(1)) or mt.group(1)[j+1]!=':'):wanted.add(mt.start(1)+j)
  if not wanted:continue
  cand=[a for a in ocr if a['printed_page']==r['printed_page']]
  a=max(cand,key=lambda a:difflib.SequenceMatcher(None,norm(a['text'])[:80],norm(raw)[:80]).ratio());b=a['bbox'];p=pdf.pages[r['pdf_page']-1]
  cc=sorted([c for c in p.chars if b[1]-1<c['top']<b[3]-.1],key=lambda c:c['x0'])
  # Normalize one codepoint at a time so matching indices retain source boxes.
  sn=[];si=[]
  for ix,c in enumerate(cc):
   for x in norm(c['text']):sn.append(x);si.append(ix)
  mn=[];mi=[]
  for ix,c in enumerate(raw):
   for x in norm(c):mn.append(x);mi.append(ix)
  mapping={}
  for mt in difflib.SequenceMatcher(None,''.join(mn),''.join(sn),autojunk=False).get_matching_blocks():
   for i in range(mt.size):mapping[mi[mt.a+i]]=cc[si[mt.b+i]]
  if r['pdf_page'] not in rendered:rendered[r['pdf_page']]=p.to_image(resolution=400).original
  im=rendered[r['pdf_page']];scale=400/72
  for j in sorted(wanted):
   if j not in mapping:
    out.append({'id':len(out),'page':r['printed_page'],'entry':r['item'],'char':j,'context':raw[:60],'status':'unmatched','ocr_line':a});pics.append(None);continue
   c=mapping[j];bb=[c['x0']-.65,c['top']-1,c['x1']+.65,c['bottom']+1]
   crop=im.crop(tuple(round(v*scale) for v in bb))
   out.append({'id':len(out),'page':r['printed_page'],'entry':r['item'],'char':j,'context':raw[:60],'status':'pending','bbox':bb,'ocr_char':c['text']});pics.append(crop)
 for start in range(0,len(out),100):
  canvas=Image.new('RGB',(1000,1000),'#eeeeee');d=ImageDraw.Draw(canvas)
  for n,im in enumerate(pics[start:start+100]):
   x=(n%10)*100;y=(n//10)*100;d.text((x+2,y+1),str(start+n),fill='black')
   if im:canvas.paste(im,(x+35,y+18))
   else:d.text((x+10,y+40),'UNMATCHED',fill='red')
  canvas.save('/tmp/koraga-glyphs-'+str(start//100)+'.png')
(R/'koraga-glyph-review.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print('glyphs',len(out),'unmatched',sum(a['status']=='unmatched' for a in out))
