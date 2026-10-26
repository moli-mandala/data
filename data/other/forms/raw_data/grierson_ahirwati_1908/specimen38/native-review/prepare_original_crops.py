"""Small immutable native-pixel controls for independent No38 reading."""
import hashlib
import json
from pathlib import Path
from PIL import Image
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[7]
PACKAGE=HERE.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((PACKAGE/'scope/native-specimen-image-manifest.json').read_text())
pages={r['printed_page_expected']:r for r in manifest['pages']}
specs=[(238,'roman-top',(240,720,2030,1390)),(238,'roman-middle',(240,1360,2030,2050)),
       (238,'roman-bottom',(240,2020,2030,2490)),
       (237,'persian-top',(230,1720,2030,2010)),(237,'persian-middle',(230,2000,2030,2370)),
       (237,'persian-bottom',(230,2360,2030,2720))]
records=[]
for page,name,box in specs:
    source=ROOT/pages[page]['lossless_png_path']
    assert sha(source)==pages[page]['lossless_png_sha256']
    target=HERE/f'original-{name}.png'
    im=Image.open(source).crop(box)
    if target.exists():assert Image.open(target).tobytes()==im.tobytes()
    else:im.save(target)
    records.append({'path':str(target.relative_to(ROOT)),'sha256':sha(target),
                    'source':str(source.relative_to(ROOT)),'source_sha256':sha(source),
                    'rectangle_in_original_pixels':box,'method':'native pixels; no enhancement or new detail'})
path=HERE/'original-crop-provenance.json'
text=json.dumps(records,indent=2)+'\n'
if path.exists():assert path.read_text()==text
else:path.write_text(text)
print(json.dumps({'crops':len(records)}))
