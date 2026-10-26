from pathlib import Path
import fitz,json,hashlib
root=Path(__file__).resolve().parents[7]
p=root/'tmp/pdfs/tribal-languages-kerala-nair2013.pdf';d=fitz.open(p);out=Path(__file__).parent
regions=[('p9-intro-nasal',8,(80,100,535,295)),('p9-comparisons',8,(80,340,520,716)),('p10-upper',9,(85,75,530,480)),('p10-paradigm',9,(85,485,465,682)),('p11-vocab',10,(85,64,535,110))]
a=[]
for name,i,b in regions:
 dst=out/(name+'.png');d[i].get_pixmap(matrix=fitz.Matrix(3,3),clip=fitz.Rect(b)).save(dst);a.append(dict(image=dst.name,pdf_page=i+1,bbox=b,scale=3,sha256=hashlib.sha256(dst.read_bytes()).hexdigest()))
(out/'render-manifest.json').write_text(json.dumps({'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'images':a},indent=2)+'\n')
