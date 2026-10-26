"""Extract two bounded cached original witnesses; no OCR or download."""
import hashlib,io,json
from pathlib import Path
from PIL import Image
from pypdf import PdfReader
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[6]
SOURCE=ROOT/'tmp/pdfs/lsi-v9-2/LSI-V9-2-alternate.pdf'
PIN='a9fbdec72119b8a6aab236541782342c62318cb1eb7cfddf2d1f5cf5891cbcb1'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
assert sha(SOURCE)==PIN
reader=PdfReader(SOURCE);records=[]
for printed,physical in [(149,166),(151,168)]:
    images=list(reader.pages[physical-1].images);assert len(images)==1
    original=images[0];path=HERE/f'candidate-printed-{printed}-pdf-{physical}-{original.name}'
    if path.exists():assert path.read_bytes()==original.data
    else:path.write_bytes(original.data)
    im=Image.open(io.BytesIO(original.data));png=path.with_suffix('.png')
    if png.exists():assert Image.open(png).tobytes()==im.tobytes()
    else:im.save(png)
    records.append({'printed_page_expected':printed,'physical_pdf_onebased':physical,
      'path':str(path.relative_to(ROOT)),'sha256':sha(path),'bytes':len(original.data),
      'native_dimensions':list(im.size),'lossless_png_path':str(png.relative_to(ROOT)),
      'lossless_png_sha256':sha(png),'verification':'Individual folio verification required; not a global offset assertion'})
text=json.dumps({'pdf':str(SOURCE.relative_to(ROOT)),'pdf_sha256':PIN,'pages':records},indent=2)+'\n'
manifest=HERE/'native-originals-manifest.json'
if manifest.exists():assert manifest.read_text()==text
else:manifest.write_text(text)
print(json.dumps({'pages':len(records),'embedded_bytes':sum(r['bytes'] for r in records)}))
