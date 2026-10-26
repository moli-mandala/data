"""Recover source-order diacritics and raised glyphs from PDF text spans."""
import fitz,json,unicodedata
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent
PDF=HERE.parents[5]/'tmp/pdfs/kurux/official.pdf'
SUPERSCRIPT={'ə':'ᵊ','ʌ':'ᶺ'}

def extract():
 first=[json.loads(s) for s in (HERE/'wordlist-first-reading.jsonl').read_text().splitlines()]
 doc=fitz.open(PDF); pages={};small=Counter();result=[];unmapped=[]
 for r in first:
  pn=r['pdf_page']
  if pn not in pages:
   pages[pn]=[l for b in doc[pn-1].get_text('dict')['blocks'] for l in b.get('lines',[])]
  left,top,right,bottom=r['source_bbox'];selected=[]
  for line in pages[pn]:
   nonblank=[s for s in line['spans'] if s['text'].strip()]
   if not nonblank:continue
   full=[s for s in nonblank if s['size']>=8]
   anchor=full[0] if full else nonblank[0]
   x,y=anchor['origin']
   if left<=x<right and top<=y<bottom:
    parts=[];evidence=[]
    for s in line['spans']:
     text=s['text']
     raised=bool(text.strip()) and s['size']<8
     if raised:
      small[text]+=1
      unknown=[ch for ch in text if not ch.isspace() and ch not in SUPERSCRIPT and not unicodedata.combining(ch)]
      if unknown:unmapped.append({'key':r['source_unit_key'],'text':text,'unknown':unknown})
      text=''.join(SUPERSCRIPT.get(ch,ch) for ch in text)
     parts.append(text)
     evidence.append({k:s[k] for k in ('text','size','flags','font','origin','bbox')})
    selected.append({'y':y,'x':x,'right':line['bbox'][2],'text':''.join(parts).strip(),'spans':evidence})
  selected.sort(key=lambda l:(l['y'],l['x']))
  value=''
  for i,line in enumerate(selected):
   if i:
    prev=selected[i-1]
    wrap=prev['right']>right-16 and line['x']<left+2
    value+=' ' if not wrap else ''
   value+=line['text']
  value=unicodedata.normalize('NFC',' '.join(value.split()))
  result.append({'source_unit_key':r['source_unit_key'],'printed_page':r['printed_page'],'pdf_page':pn,'prompt':r['prompt'],'lect':r['lect'],'gloss':r['gloss'],'form':value,'previous_draft':r['draft'],'source_bbox':r['source_bbox'],'span_evidence':selected,'status':'geometry first reading; original-page review pending'})
 assert len(result)==2028
 (HERE/'wordlist-geometry-first-reading.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result))
 report={'status':'draft geometry recovery; original comparison pending','small_span_inventory':dict(small),'unmapped':unmapped,'changed_cells':[{'key':r['source_unit_key'],'before':r['previous_draft'],'after':r['form']} for r in result if r['form']!=r['previous_draft']]}
 (HERE/'wordlist-geometry-delta-first-reading.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print('cells',len(result),'changed',len(report['changed_cells']),'small spans',dict(small),'unmapped',unmapped)
if __name__=='__main__':extract()
