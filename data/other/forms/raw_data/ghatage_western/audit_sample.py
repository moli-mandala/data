"""Select an independent, reproducible raw-OCR versus parsed-record audit.

Example: python audit_sample.py --seed 20260912 --output /tmp/ghatage-sample
With --images SCRATCH, also crop the cached deskewed 300-DPI page images.
Selection never asserts review success: results must be recorded separately.
"""
import argparse
import json
import random
from pathlib import Path
from import_glossaries import build, VOLUMES

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed',type=int,default=20260911)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--images',type=Path)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    for name in VOLUMES:
        _,audit=build(name)
        sample=random.Random(args.seed).sample(audit,20)
        (args.output/f'{name}.json').write_text(json.dumps(dict(seed=args.seed,records=sample),ensure_ascii=False,indent=2)+'\n')
        if args.images:
            from PIL import Image,ImageDraw
            for record in sample:
                r=record['raw_record'];words=r['raw_words']
                path=args.images/f'deskew/{name}/{r["pdf_page"]:03}.png'
                if not path.exists():raise SystemExit(f'Missing cached image: {path}')
                with Image.open(path) as im:
                    box=(max(0,min(w['x'] for w in words)-45),max(0,min(w['y'] for w in words)-22),
                         min(im.width,max(w['x']+w['w'] for w in words)+50),
                         min(im.height,max(w['y']+w['h'] for w in words)+22))
                    im.crop(box).save(args.output/f'{name}-{r["pdf_page"]}-{r["col"]}-{r["ordinal"]}.png')

if __name__=='__main__':main()
