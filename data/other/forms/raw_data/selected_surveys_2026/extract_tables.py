"""Positioned native PDF extraction for the two selected LinSuN reports."""
import collections,json,re,hashlib
from pathlib import Path
import pdfplumber
R=Path(__file__).resolve().parent
CONF={'angika':(94,102,12),'majhi':(80,96,10)}
def extract(k):
 out=[]
 snapshot=json.loads((R/'snapshot.json').read_text())['sources'][k]
 assert hashlib.sha256((R/(k+'.pdf')).read_bytes()).hexdigest()==snapshot['pdf_sha256']
 with pdfplumber.open(R/(k+'.pdf')) as pdf:
  assert len(pdf.pages)==snapshot['pages']
  lo,hi,off=CONF[k]
  for pi in range(lo,hi):
   page=pdf.pages[pi]
   for tab in page.find_tables():
    for cells,raw in zip(tab.rows,tab.extract()):
     pairs=[(b,t or '') for b,t in zip(cells.cells,raw) if b]
     if len(pairs)!=8:continue
     if not re.fullmatch(r'\d+\.?',pairs[0][1].strip()):continue
     item=int(pairs[0][1].strip('. '))
     if not 1<=item<=210:continue
     gloss=' '.join(pairs[1][1].split())
     for col,(bb,txt) in enumerate(pairs[3:]):
      chars=[dict(c) for c in page.crop(bb).chars]
      for c in chars:
       if k=='angika' and c['text']=='\uf04e':c['text']='N'
       if k=='angika' and c['text']=='h' and c['size']<9:c['text']='ʰ'
       if k=='majhi' and c['text']=='(cid:13)' and 'TimesNewRoman' in c['fontname']:c['text']='̃'
       if k=='majhi' and c['text']=='ȹ':c['text']='ʰ'
       if k=='majhi' and c['text'] in {'(cid:1)','(cid:2)'} and 'CambriaMath' in c['fontname']:c['text']='̪'
      decoded=pdfplumber.utils.extract_text(chars,x_tolerance=2,y_tolerance=4)
      out.append({'item':item,'column':col,'pdf_page':pi+1,'printed_page':pi+1-off,'bbox':list(bb),'gloss':gloss,'raw_text':txt,'text':decoded,'chars':[{'text':c['text'],'font':c['fontname'],'size':c['size'],'x0':c['x0'],'top':c['top']} for c in chars]})
 (R/(k+'-cells.json')).write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
 print(k,len(out),collections.Counter(r['column'] for r in out));print('max',max(r['item'] for r in out));print('bad',[r for r in out if '(cid:' in r['text']][:2])
if __name__=='__main__':
 for k in CONF:extract(k)
