"""Retain bounded native-pixel No38 Latin controls; no glyph interpretation."""
import hashlib,json
from pathlib import Path
from PIL import Image
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[7]
SOURCE=ROOT/'tmp/jambu-next-source/ahirwati-native-specimens-20260927/printed-238-pdf-255-Im255.png'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
im=Image.open(SOURCE)
specs=[('bere-nai',(1550,740,1860,850)),('latin-upper-pronouns',(900,1110,1890,1270)),
       ('kai-tarah-first',(400,1700,820,1830)),('beautiful',(410,2090,700,2210)),
       ('last-quantity',(410,2270,2040,2440))]
rows=[]
for name,box in specs:
    path=HERE/f'narrow-{name}.png'
    crop=im.crop(box)
    if path.exists():assert Image.open(path).tobytes()==crop.tobytes()
    else:crop.save(path)
    rows.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path),'source':str(SOURCE.relative_to(ROOT)),
                 'source_sha256':sha(SOURCE),'rectangle_in_original_pixels':box,'method':'native pixels, no resampling'})
path=HERE/'narrow-control-provenance.json'
text=json.dumps(rows,indent=2)+'\n'
if path.exists():assert path.read_text()==text
else:path.write_text(text)
print(json.dumps({'controls':len(rows)}))
