"""Small sequential native-headword OCR pilot; no installation or accepted readings."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
import pdfplumber
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parent
KEYS=['cfelmahali2024:p10:entry:'+str(n) for n in range(1,6)]+['cfelmahali2024:p17:entry:2','cfelmahali2024:p58:entry:1','cfelmahali2024:p100:entry:8','cfelmahali2024:p104:entry:8','cfelmahali2024:p136:entry:6']
def run(pdf,output):
    assert hashlib.sha256(pdf.read_bytes()).hexdigest()==json.loads((ROOT/'source-manifest.json').read_text())['pdf_sha256']
    rows={r['entry_key']:r for r in map(json.loads,(ROOT/'positioned-candidates.jsonl').read_text().splitlines())}
    output.mkdir(parents=True,exist_ok=True);records=[]
    with pdfplumber.open(pdf) as d:
        for key in KEYS:
            row=rows[key];page=d.pages[row['pdf_page']-1];hit=page.search(r'\(([^()]+)\)\s*-')[row['page_item']-1]
            # Pilot deliberately refuses wrapped native headings, rather than clipping silently.
            assert hit['x0']>120,(key,'native heading may wrap')
            box=(108,max(0,hit['top']-8),hit['x0']-1,hit['top']+13)
            image=page.crop(box).to_image(resolution=400).original.convert('RGB');image=ImageOps.expand(image,border=20,fill='white')
            crop=output/(key.replace(':','-')+'.png');image.save(crop)
            command=['tesseract',str(crop),'stdout','-l','ben','--psm','7']
            result=subprocess.run(command,env={**os.environ,'OMP_THREAD_LIMIT':'1'},capture_output=True,text=True,check=True,timeout=30)
            records.append({'entry_key':key,'pdf_page':row['pdf_page'],'crop_box_points':box,'resolution_dpi':400,'border_pixels':20,'command':command,'text_layer_native':row['raw_native'],'ocr_native':result.stdout.strip(),'stderr':result.stderr,'status':'pilot raw OCR; not accepted Native'})
            page.close()
    (output/'pilot.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    return records
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pdf',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    for row in run(a.pdf,a.output):print(row['entry_key'],repr(row['text_layer_native']),'->',row['ocr_native'])
