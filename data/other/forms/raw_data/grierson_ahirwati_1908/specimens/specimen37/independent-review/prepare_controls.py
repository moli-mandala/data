from pathlib import Path
from PIL import Image
import json,hashlib
P=Path(__file__).resolve().parent; ROOT=Path.cwd()
S=ROOT/'tmp/jambu-next-source/ahirwati-native-specimens-20260927'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
specs=[('ham-person','p235-owner-top',(0,430,230,535)),
 ('like-even','p235-owner-top',(880,0,1170,100)),
 ('lost','p235-owner-middle',(1240,165,1610,300)),
 ('give-vowel','p234-owner-top',(1280,110,1510,205))]
manifest=[]
for name,stem,rect in specs:
    source=S/f'specimen37-{stem}.png';out=P/f'{name}-native-control.png'
    with Image.open(source) as im:
        crop=im.crop(rect);crop.resize((crop.width*3,crop.height*3),Image.Resampling.NEAREST).save(out)
    manifest.append({'path':str(out.relative_to(ROOT)),'sha256':sha(out),'source':str(source.relative_to(ROOT)),
      'source_sha256':sha(source),'rectangle':rect,'method':'native crop, 3x nearest display; no new source detail'})
(P/'control-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(len(manifest))
