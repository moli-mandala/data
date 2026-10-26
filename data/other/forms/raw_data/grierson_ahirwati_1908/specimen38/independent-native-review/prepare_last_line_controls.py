from PIL import Image
from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent;root=Path.cwd();source=root/'tmp/jambu-next-source/ahirwati-native-specimens-20260927/printed-237-pdf-254-Im254.png';im=Image.open(source);rows=[]
for n,y,hi in [(9,2395,2510),(10,2495,2600),(11,2590,2710)]:
 rect=(230,y,2030,hi);out=P/f'line-{n:02d}-complete-native-v2.png';c=im.crop(rect);c.resize((c.width*2,c.height*2),Image.Resampling.NEAREST).save(out)
 rows.append({'line':n,'rectangle':rect,'image':str(out.relative_to(root)),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'source':str(source.relative_to(root)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'history':'Initial line-control window clipped top/nearby letter portions; original full Roman/Persian crops never changed. This version extends margin for complete current native line.'})
(P/'last-three-line-complete-control-manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
