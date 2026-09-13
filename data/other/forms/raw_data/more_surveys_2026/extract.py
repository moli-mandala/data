"""Extract bounded lexical tables from pinned PDFs; no OCR or network."""
import json,re,logging,collections,hashlib
from pathlib import Path
import pdfplumber
from pypdf import PdfReader,PdfWriter
logging.disable(logging.WARNING)
R=Path(__file__).resolve().parent
CONF={'jharkhand':(957,1001),'himachal':(1100,1147),'rajasthan':(481,494),'west-bengal':(722,750),'kisan':(105,113)}
def extract(k):
 path=Path('/tmp/more-'+k+'.pdf')
 snap=json.loads((R/'snapshot.json').read_text())
 assert hashlib.sha256((R/(k+'.pdf')).read_bytes()).hexdigest()==snap['sources'][k]['pdf_sha256']
 reader=PdfReader(R/(k+'.pdf'))
 assert len(reader.pages)=={'jharkhand':1002,'himachal':1159,'rajasthan':494,'west-bengal':817,'kisan':118}[k]
 w=PdfWriter();w.append(reader);w.write(path)
 out=[]
 with pdfplumber.open(path) as pdf:
  for pi in range(*CONF[k]):
   page=pdf.pages[pi]
   for tab in page.find_tables():
    for ri,raw in enumerate(tab.extract()):
     boxes=tab.rows[ri].cells
     pairs=[(v,b) for v,b in zip(raw,boxes) if b is not None]
     raw=[v for v,b in pairs];boxes=[b for v,b in pairs]
     if k=='jharkhand':order=([0,1,2,3,4,5,6,7,8] if (pi-957)%2==0 else [7,6,5]);cols=list(range(7)) if (pi-957)%2==0 else [7]
     elif k=='himachal':
      n=len(raw)
      if n==6:
       group=0 if pi<1112 else 4 if pi<1124 else 8
      elif n==4:group=12
      else:continue
      order=list(range(n));cols=list(range(group,group+n-2))
     elif k=='rajasthan':order=list(range(9));cols=list(range(7))
     elif k=='west-bengal':order=list(range(5));cols=list(range(3))
     else:order=[0,1,3,4,5,6,7];cols=list(range(5))
     if len(raw)<=max(order):continue
     vals=[raw[i] or '' for i in order]
     if not re.fullmatch(r'\d+\.?',vals[0].strip()):continue
     item=int(vals[0].strip('. '))
     if not 1<=item<=500:continue
     for j,(ci,v) in enumerate(zip(cols,vals[2:])):
      bb=boxes[order[2+j]]
      if bb is None:raise ValueError((k,pi,raw))
      images=[{'name':im['name'],'bbox':[im['x0'],im['top'],im['x1'],im['bottom']]} for im in page.images if bb[0]<(im['x0']+im['x1'])/2<bb[2] and bb[1]<(im['top']+im['bottom'])/2<bb[3]]
      # Census lexical appendices have constant print offsets, unlike preceding chapters.
      offset={'jharkhand':18,'himachal':24,'west-bengal':20,'rajasthan':18,'kisan':9}[k]
      out.append(dict(item=item,column=ci,pdf_page=pi+1,printed_page=pi+1-offset,gloss=vals[1],text=v,bbox=bb,images=images))
 (R/(k+'-cells.json')).write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
 print(k,len(out),dict(collections.Counter(x['column'] for x in out)),'images',sum(bool(x['images']) for x in out),flush=True)
if __name__=='__main__':
 import sys
 for k in sys.argv[1:] or CONF:extract(k)
