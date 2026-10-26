from pathlib import Path
from PIL import Image
import hashlib,json
HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'data/make_cldf.py').exists())
SOURCE=ROOT/'tmp/jambu-next-source/ahirwati-native-specimens-20260927'
controls=[]
for page,pdf,bounds in [(233,250,(240,350,1980,1090)),(237,254,(230,480,1980,1300)),(238,255,(250,2530,1990,2820)),(239,256,(300,405,1970,1030))]:
    source=SOURCE/f'printed-{page}-pdf-{pdf}-Im{pdf}.png'
    target=HERE/f'p{page}-documentary-native-control.png'
    Image.open(source).crop(bounds).save(target)
    controls.append(dict(printed_page=page,source=str(source.relative_to(ROOT)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),original_pixel_bounds=bounds,crop=str(target.relative_to(ROOT)),crop_sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
(HERE/'control-provenance.json').write_text(json.dumps(controls,indent=2)+'\n')
