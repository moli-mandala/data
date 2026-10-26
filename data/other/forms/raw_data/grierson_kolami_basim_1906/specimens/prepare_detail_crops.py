"""Immutable locatable narrow crops from existing original page renders."""
import hashlib
import json
from pathlib import Path
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'data/source_meta.py').exists())
BASE=ROOT/'tmp/jambu-next-source/kolami-specimen-originals'
REQUESTS=[dict(page=564,name='p564-line7-going-first-glyph',rect=[1000,1860,1690,2030],reason='ectnā versus actnā; compare actual type, no analogy')]


def main():
    reports=[]
    for request in REQUESTS:
        path=BASE/f"printed-{request['page']}-2800.png"
        output=BASE/(request['name']+'.png')
        if not output.exists():
            with Image.open(path) as im:
                crop=im.crop(request['rect']);crop.resize((crop.width*3,crop.height*3)).save(output)
        reports.append(dict(**request,source=str(path.relative_to(ROOT)),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            crop=str(output.relative_to(ROOT)),crop_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),scale=3))
    (HERE/'detail-crop-locators.json').write_text(json.dumps(reports,indent=2)+'\n')
    print(json.dumps(reports))


if __name__=='__main__':main()
