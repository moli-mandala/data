"""Bounded quantity/nasal controls; preserve all earlier original crops."""
import hashlib,json
from pathlib import Path
from PIL import Image
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[6]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
pages={r['printed_page_expected']:r for r in json.loads((HERE/'native-originals-manifest.json').read_text())['pages']}
rows=[]
for pg,name,box in [(149,'written-genitive',(1340,430,2050,610)),
                    (151,'past-gender-number',(710,1190,1720,1430)),
                    (151,'imperfect-gender-number',(710,2520,1720,2740))]:
    source=ROOT/pages[pg]['lossless_png_path'];assert sha(source)==pages[pg]['lossless_png_sha256']
    path=HERE/f'postreading-p{pg}-{name}.png';im=Image.open(source).crop(box)
    if path.exists():assert Image.open(path).tobytes()==im.tobytes()
    else:im.save(path)
    rows.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path),'source':str(source.relative_to(ROOT)),
                 'source_sha256':sha(source),'rectangle_in_original_pixels':box,'method':'native pixels; no enhancement'})
text=json.dumps(rows,indent=2)+'\n';path=HERE/'postreading-control-provenance.json'
if path.exists():assert path.read_text()==text
else:path.write_text(text)
