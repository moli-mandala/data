from pathlib import Path
from PIL import Image
import json,hashlib
P=Path(__file__).resolve().parent
root=Path.cwd()
source=root/'tmp/jambu-next-source/ahirwati-native-specimens-20260927/printed-237-pdf-254-Im254.png'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='545a263919ffb7a1bf65f861712bbc89056135b3ad0e2fb9a73b1fdbfeae8c52'
image=Image.open(source)
records=[]
for n,y in enumerate([1720,1810,1900,1990,2080,2170,2260,2350,2440,2530,2620],1):
 rect=(230,y,2030,y+100)
 out=P/f'line-{n:02d}-native-pixels-2x-nearest.png'
 crop=image.crop(rect);crop.resize((crop.width*2,crop.height*2),Image.Resampling.NEAREST).save(out)
 records.append({'line':n,'rectangle':rect,'image':str(out.relative_to(root)),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'source':str(source.relative_to(root)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'policy':'2x nearest display of retained native pixels; no enhancement/new source detail'})
(P/'line-control-manifest.json').write_text(json.dumps(records,indent=2)+'\n')
