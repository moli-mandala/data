"""Reproduce cached page OCR in a scratch directory (requires PyMuPDF, Pillow, numpy, Tesseract 5.5.2)."""
import argparse, shutil, hashlib
import csv, json, subprocess, io, concurrent.futures
from pathlib import Path
import pymupdf as fitz

ROOT=Path(__file__).parent
TESS=shutil.which('tesseract') or '/opt/homebrew/bin/tesseract'
TESS=shutil.which('tesseract') or '/opt/homebrew/bin/tesseract'

def run(task):
    name,page=task
    pdf=fitz.open(ROOT/f'{name}.pdf')
    p=pdf[page-1]
    out=ROOT/'ocr'/name
    out.mkdir(parents=True,exist_ok=True)
    image=out/f'{page:03}.png'
    if not image.exists(): p.get_pixmap(dpi=300).save(str(image))
    dest=out/f'{page:03}.tsv'
    if not dest.exists():
        proc=subprocess.run([TESS,str(image),'stdout','-l','script/Latin','--psm','6','tsv'],capture_output=True,text=True,check=True)
        dest.write_text(proc.stdout)
    words=[]
    for r in csv.DictReader(io.StringIO(dest.read_text()),delimiter='\t',quoting=csv.QUOTE_NONE):
        if r['level']=='5' and r.get('text','').strip():
            words.append(dict(text=r['text'],x=int(r['left']),y=int(r['top']),w=int(r['width']),h=int(r['height']),confidence=float(r['conf'])))
    (out/f'{page:03}.json').write_text(json.dumps(dict(width=round(p.rect.width*300/72),height=round(p.rect.height*300/72),words=words),ensure_ascii=False))
    return name,page,len(words)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('scratch',type=Path);args=ap.parse_args();ROOT=args.scratch
    from import_glossaries import VOLUMES
    for name,v in VOLUMES.items():
        pdf=ROOT/f'{name}.pdf'
        if not pdf.exists(): raise SystemExit(f'Missing pinned scan: {pdf}')
        if hashlib.sha256(pdf.read_bytes()).hexdigest()!=v['sha256']: raise SystemExit(f'Wrong scan hash: {pdf}')
        with fitz.open(pdf) as doc: assert len(doc)==v['total_pages']
    tasks=[('kudali',p) for p in range(105,161)]+[('konkani',p) for p in range(128,149)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for x in pool.map(run,tasks): print(*x,flush=True)
