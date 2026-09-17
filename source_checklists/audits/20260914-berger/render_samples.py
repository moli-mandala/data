import hashlib,importlib.util,json,sys
from pathlib import Path
import pypdfium2 as pdfium
from PIL import Image,ImageDraw
root=Path(__file__).resolve().parents[3];out=root.parent/'tmp/berger-audit-20260914';(out/'images').mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('berger_render',root/'data/other/forms/raw_data/berger_cleanup.py');b=importlib.util.module_from_spec(spec);sys.modules[spec.name]=b;spec.loader.exec_module(b)
assert b.sha256_path(b.legacy.DEFAULT_PDF)==b.PDF_SHA256
pages={p['pdf_page']:p for p in b.load_pages(b.CACHE_DIR)}
units={u.stable_key:u for u in b.reconstruct_units(pages.values())}
samples=json.loads((out/'sample.json').read_text())
doc=pdfium.PdfDocument(b.legacy.DEFAULT_PDF)
for i,s in enumerate(samples,1):
 a=s['audit'];u=units[a['Stable_Key']];p=pages[u.pdf_page];physical=b.physical_column(u.left,p['width'])
 groups={}
 for line in u.lines:
  matches=[number for number in range(u.pdf_page,min(u.pdf_page+3,248)) if number in pages and any(l['left']==line['left'] and l['top']==line['top'] and b.legacy.canonical(l['text'])==line['text'] for l in pages[number]['lines'])]
  number=matches[0] if matches else u.pdf_page
  groups.setdefault((number,b.physical_column(line['left'],pages[number]['width'])),[]).append(line)
 panels=[]
 for (number,col),lines in groups.items():
  p=pages[number];page=doc[number-1];rendered=page.render(scale=3).to_pil();ratio=rendered.width/p['width']
  left=max(0,min(l['left'] for l in lines)-55);right=min(p['width'],left+1000)
  top=max(0,min(l['top'] for l in lines)-130);bottom=max(l['top'] for l in lines)+180
  crop=rendered.crop(tuple(int(x*ratio) for x in (left,top,right,bottom)))
  panel=Image.new('RGB',(max(crop.width,720),crop.height+40),'white');ImageDraw.Draw(panel).text((10,6),f"{i:02} | {a['Stable_Key']} | {a['Installed_Key']} | PDF {number}",fill='black');panel.paste(crop,(0,35));panels.append(panel);page.close()
 sheet=Image.new('RGB',(max(p.width for p in panels),sum(p.height for p in panels)),'white');yp=0
 for panel in panels:sheet.paste(panel,(0,yp));yp+=panel.height
 sheet.save(out/'images'/f'sample-{i:02}.png')
 s['image_sha256']=hashlib.sha256((out/'images'/f'sample-{i:02}.png').read_bytes()).hexdigest()
 print(i,a['Stable_Key'],flush=True)
(out/'sample.json').write_text(json.dumps(samples,ensure_ascii=False,indent=2)+'\n')
