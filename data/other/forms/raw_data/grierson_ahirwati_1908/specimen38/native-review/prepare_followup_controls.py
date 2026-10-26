"""Two correctly positioned word controls following broader original crops."""
import hashlib,json
from pathlib import Path
from PIL import Image
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[7]
SOURCE=ROOT/'tmp/jambu-next-source/ahirwati-native-specimens-20260927/printed-238-pdf-255-Im255.png'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=[]
for name,box in [('beautiful-full',(320,2040,730,2160)),('jai-full',(1260,1200,1650,1320)),('kai-tarah-last-full',(290,2390,550,2480))]:
    path=HERE/f'followup-{name}.png';im=Image.open(SOURCE).crop(box)
    if path.exists():assert Image.open(path).tobytes()==im.tobytes()
    else:im.save(path)
    rows.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path),'source':str(SOURCE.relative_to(ROOT)),
                 'source_sha256':sha(SOURCE),'rectangle_in_original_pixels':box,'method':'native pixels; no resampling'})
text=json.dumps(rows,indent=2)+'\n';path=HERE/'followup-control-provenance.json'
if path.exists():assert path.read_text()==text
else:path.write_text(text)
