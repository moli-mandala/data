"""Reuse existing original controls for immutable narrow reconciliation evidence."""
import hashlib
import json
from pathlib import Path
from PIL import Image

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
ROOT=next(p for p in HERE.parents if (p/'data/source_meta.py').exists())
DEST=ROOT/'tmp/jambu-next-source/kolami-grammar-original-controls'
REQUESTS=[
    (566,'top','kolami-amd-avnd',[210,780,1550,905]),
    (567,'top','negative-marker',[225,1005,670,1075]),
    (567,'top','conjunctive-last',[200,1150,820,1240]),
    (567,'top','present-singular',[270,565,620,820]),
    (567,'top','future-plural',[1830,565,2240,820]),
    (566,'top','dative-marker',[1870,1080,2180,1150]),
    (562,'top','noun-and-case',[190,885,2240,1590]),
    (562,'bottom','pronouns-copula',[220,420,2200,1560]),
    (563,'top','negative-and-nonfinite',[200,640,2240,1290]),
    (566,'top','dative-and-five',[200,930,2240,1470]),
    (566,'bottom','pronouns-and-copula',[190,0,2240,1540]),
    (567,'top','beat-table',[250,400,2240,840]),
    (567,'top','negative-conjunctive',[200,920,2240,1260]),
]

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    result=[]
    for page,part,label,rect in REQUESTS:
        source=PACKAGE/f'independent-grammar-review/original-controls/printed-{page}-{part}.png'
        output=DEST/f'p{page}-{label}-owner-control.png'
        if not output.exists():
            with Image.open(source) as im:
                crop=im.crop(rect);crop.resize((crop.width*2,crop.height*2)).save(output)
        result.append(dict(page=page,rect=rect,scale=2,source=str(source.relative_to(ROOT)),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),crop=str(output.relative_to(ROOT)),
            crop_sha256=hashlib.sha256(output.read_bytes()).hexdigest()))
    (HERE/'grammar-control-locators.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
