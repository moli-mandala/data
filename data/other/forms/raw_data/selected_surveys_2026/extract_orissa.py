"""Orissa 2002 original 300dpi scan: grid geometry plus two pinned OCR passes.

Run with bundled Python (numpy, PIL, pypdf) and Tesseract 5.5.1.
Raw page TSVs and cell readings are review material, not automatic installation.
"""
from pathlib import Path
import argparse, csv, hashlib, io, json, subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from pypdf import PdfReader
R=Path(__file__).resolve().parent
C=Path('/private/tmp/jambu-orissa-ocr'); C.mkdir(exist_ok=True)
HASH='bf03f7fd8333064f6653b6cdcf89938cb30fce21f1c2524cdff60f9a2ec170c7'
def groups(a):
    out=[]
    for n in a:
        if not out or n>out[-1][-1]+12: out.append([int(n)])
        else:out[-1].append(int(n))
    return [int(round(sum(g)/len(g))) for g in out]
def page(pi,pdf):
    im=pdf.pages[pi].images[0].image.convert('RGB'); w,h=im.size
    # Estimate scan skew from long horizontal rules, independently of OCR text.
    small=im.convert('L').resize((w//2,h//2))
    scored=[]
    for angle in np.arange(-1.5,1.51,.1):
        rotated=small.rotate(float(angle),resample=Image.Resampling.BILINEAR,fillcolor=255)
        arr=np.asarray(rotated)[int(h*.07):int(h*.46),int(w*.025):int(w*.48)]<170
        counts=arr.sum(axis=1);score=float(np.sort(counts)[-80:].astype(float).dot(np.sort(counts)[-80:].astype(float)))
        scored.append((score,round(float(angle),2)))
    angle=max(scored)[1]
    if angle:im=im.rotate(angle,resample=Image.Resampling.BICUBIC,fillcolor='white')
    thick=np.asarray(im.convert('L').point(lambda x:0 if x<160 else 255).filter(ImageFilter.MinFilter(9)))<128
    ys=groups(np.where(thick[:,int(w*.05):int(w*.95)].sum(axis=1)>w*.62)[0]);ys=[y for y in ys if h*.06<y<h*.97]
    xs=[]
    for threshold in [.75,.70,.65,.60,.55,.50,.45]:
        candidates=groups(np.where(thick[ys[0]:ys[-1]].sum(axis=0)>(ys[-1]-ys[0])*threshold)[0])
        candidates=[x for x in candidates if w*.025<x<w*.98]
        if len(candidates)==8 and min(np.diff(candidates))>70:
            xs=candidates;break
    print(pi,'x',xs,'y',ys,flush=True)
    assert len(xs)==8,(pi,'expected eight vertical borders',xs)
    assert 8<=len(ys)<=32,(pi,'unexpected horizontal borders',ys)
    boxes=[]
    for ri,(y0,y1) in enumerate(zip(ys,ys[1:])):
        for col,(x0,x1) in enumerate(zip(xs,xs[1:])):
            boxes.append((ri,col,x0,y0,x1,y1))
    def ocr_cell(box):
        ri,col,x0,y0,x1,y1=box
        p=C/f'p{pi+1}-r{ri}-c{col}.png'
        # Tight rule erasure damages italic initial letters. Crop a minimal inset
        # and retain raw pixels; OCR disagreements remain explicit review material.
        crop=im.crop((x0+2,y0+4,x1-2,y1-4))
        # Remove only detected long rules at the outer edge, not a fixed strip
        # that would erase italic headword initials. All raw geometry is retained.
        arr=np.asarray(crop).copy(); ink=(arr.mean(axis=2)<175);ch,cw=ink.shape
        for x in list(range(min(12,cw)))+list(range(max(0,cw-12),cw)):
            if ink[:,x].sum()>ch*.62:arr[:,max(0,x-1):min(cw,x+2)]=255
        for y in list(range(min(5,ch)))+list(range(max(0,ch-5),ch)):
            if ink[y].sum()>cw*.65:arr[max(0,y-1):min(ch,y+2),:]=255
        crop=Image.fromarray(arr);padded=Image.new('RGB',(crop.width+24,crop.height+24),'white');padded.paste(crop,(12,12));padded.save(p)
        rr={}
        for lang in ['eng','script/Latin']:
            name=f'orissa-v2-p{pi+1}-r{ri}-c{col}-{x0}-{y0}-{x1}-{y1}-a{angle}-{lang.replace("/","-")}.tsv'; cache=C/name
            if not cache.exists():
                r=subprocess.run(['tesseract',str(p),'stdout','-l',lang,'--psm','6','tsv'],capture_output=True,check=True)
                cache.write_bytes(r.stdout)
            words=list(csv.DictReader(io.StringIO(cache.read_text()),delimiter='\t',quoting=csv.QUOTE_NONE))
            selected=[a for a in words if a.get('text','').strip()]
            rr[lang]={'text':' '.join(a['text'] for a in selected),'words':selected}
        return {'pdf_page':pi+1,'printed_page':pi-24,'row':ri,'column':col,'bbox':[x0,y0,x1,y1],'image_size':[w,h],'rotation_degrees':angle,'ocr':rr}
    with ThreadPoolExecutor(max_workers=8) as pool: cells=list(pool.map(ocr_cell,boxes))
    return {'pdf_page':pi+1,'x_borders':xs,'y_borders':ys,'cells':cells}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--first',type=int,default=216);p.add_argument('--last',type=int,default=258);a=p.parse_args()
    path=R/'orissa-official.pdf';assert hashlib.sha256(path.read_bytes()).hexdigest()==HASH
    pdf=PdfReader(path);assert len(pdf.pages)==615
    # pypdf image caches are read sequentially; OCR itself is the dominant cost.
    out=[]
    for pi in range(a.first,a.last):
        result=page(pi,pdf);out.append(result)
        (R/f'orissa-p{pi+1}-cells.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (R/'orissa-grid.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
