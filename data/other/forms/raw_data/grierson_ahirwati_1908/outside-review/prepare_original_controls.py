"""Bounded original-pixel comparison controls, including page headings."""
import hashlib,json
from pathlib import Path
from PIL import Image
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[6]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
pages={r['printed_page_expected']:r for r in json.loads((HERE/'native-originals-manifest.json').read_text())['pages']}
rows=[]
for pg,name,box in [(149,'header',(350,240,2050,420)),(149,'phonology',(310,395,2050,850)),
                    (149,'context',(310,840,2050,1270)),(151,'header',(310,240,2050,445)),
                    (151,'auxiliary',(310,455,2050,1450)),(151,'finite-context',(310,1410,2050,2840))]:
    source=ROOT/pages[pg]['lossless_png_path'];assert sha(source)==pages[pg]['lossless_png_sha256']
    target=HERE/f'p{pg}-{name}-native.png';im=Image.open(source).crop(box)
    if target.exists():assert Image.open(target).tobytes()==im.tobytes()
    else:im.save(target)
    rows.append({'path':str(target.relative_to(ROOT)),'sha256':sha(target),'source':str(source.relative_to(ROOT)),
                 'source_sha256':sha(source),'rectangle_in_original_pixels':box,'method':'native-resolution crop, no enhancement'})
text=json.dumps(rows,indent=2)+'\n';path=HERE/'original-control-provenance.json'
if path.exists():assert path.read_text()==text
else:path.write_text(text)
