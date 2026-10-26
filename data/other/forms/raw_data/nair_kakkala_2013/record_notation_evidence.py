"""Census all source vector lowlines, independent of extracted text glyphs."""
import hashlib
import json
from pathlib import Path
import fitz

HERE=Path(__file__).resolve().parent
PDF=HERE.parents[5]/'tmp/pdfs/tribal-languages-kerala-nair2013.pdf'
doc=fitz.open(PDF);expected={28:5,29:9,30:1,31:5};pages=[]
for printed,count in expected.items():
 page=doc[printed-1];words=page.get_text('words')
 chars=[c for b in page.get_text('rawdict')['blocks'] if 'lines' in b
        for l in b['lines'] for s in l['spans'] for c in s['chars']]
 found=[]
 for d in page.get_drawings():
  r=d['rect']
  if not (.2<r.height<2 and 2<r.width<50 and r.y0<740): continue
  ws=[w[4] for w in words if w[0]-1<=r.x0<=w[2]+1 and w[1]<=r.y0<=w[3]+3]
  if not ws: continue
  covered=[c['c'] for c in chars if c['bbox'][0]+.5>=r.x0 and c['bbox'][2]-.5<=r.x1
           and c['bbox'][1]<=r.y0<=c['bbox'][3]+3]
  found.append(dict(rect_pdf_points=[round(x,3) for x in r],extracted_word=ws[0],
                    covered_glyphs=''.join(covered),encoding_policy='U+0332 on each visibly underlined glyph; no phonemic claim'))
 assert len(found)==count,(printed,found)
 pages.append(dict(printed_page=printed,pdf_page=printed,drawn_underlines=found))
r=dict(status='complete four-page drawn-underline census',source_pdf_sha256=hashlib.sha256(PDF.read_bytes()).hexdigest(),
        total_drawn_underlines=20,pages=pages,
        policy='Dot below, drawn lowline, tilde and literal macrons remain distinct. Doubled underlined nn uses two U+0332 marks. No automatic IPA conversion.')
(HERE/'notation-evidence.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
print('20 lowline drawings across all four original pages')
