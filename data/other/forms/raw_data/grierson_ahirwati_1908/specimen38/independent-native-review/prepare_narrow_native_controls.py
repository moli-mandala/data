from pathlib import Path
from PIL import Image
import json,hashlib
P=Path(__file__).resolve().parent;R=Path.cwd();src=R/'tmp/jambu-next-source/ahirwati-native-specimens-20260927/printed-237-pdf-254-Im254.png';im=Image.open(src);out=[]
controls={'line4-second-house-final':(410,1990,735,2100),'line3-initial-pronoun':(1790,1900,2030,2010),'line7-mouth':(965,2260,1260,2380),'line9-relative-tail':(390,2395,660,2510),'line10-personal-name':(955,2495,1320,2600),'line11-relative-tail':(1400,2590,1770,2710),'line11-give-final':(260,2590,515,2710),'line11-addressee':(560,2590,850,2710)}
for name,rect in controls.items():
 f=P/(name+'-native3x.png');c=im.crop(rect);c.resize((c.width*3,c.height*3),Image.Resampling.NEAREST).save(f);out.append({'name':name,'rectangle':rect,'path':str(f.relative_to(R)),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'source':str(src.relative_to(R)),'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'policy':'Nearest display of native pixels, not enhancement; declared name is a search lead, actual contents checked independently.'})
(P/'narrow-native-control-manifest.json').write_text(json.dumps(out,indent=2)+'\n')
