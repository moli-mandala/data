from pathlib import Path
import fitz,hashlib,json
H=Path(__file__).resolve().parent
W=next(p for p in H.parents if (p/'data/tags.py').exists())
p=W/'tmp/pdfs/tribal-languages-kerala-nair2013.pdf';d=fitz.open(p)
items=[('preface-page01.png',0,None,1.4),('nasal-lower-marks-page09.png',8,(80,244,510,287),4),('lower-r-page09.png',8,(80,685,520,720),4),('pronouns-verbs-page10.png',9,(70,75,540,415),2.5)]
manifest=[]
for name,page,clip,scale in items:
 out=H/name
 if not out.exists():d[page].get_pixmap(matrix=fitz.Matrix(scale,scale),clip=fitz.Rect(clip) if clip else None).save(out)
 manifest.append(dict(path=name,pdf_page=page+1,clip=clip,scale=scale,sha256=hashlib.sha256(out.read_bytes()).hexdigest()))
(H/'notation-render-manifest.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),images=manifest,font45=d.xref_object(45),to_unicode=d.xref_stream(46).decode('latin1')),indent=2)+'\n')
