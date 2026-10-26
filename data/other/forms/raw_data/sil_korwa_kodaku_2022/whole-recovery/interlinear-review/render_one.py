from pathlib import Path
import fitz,sys,json,hashlib
H=Path(__file__).resolve().parent;W=next(p for p in H.parents if (p/'data/tags.py').exists());n=int(sys.argv[1]);assert 93<=n<=108
p=W/'tmp/pdfs/korwa-kodaku-2022/source.pdf';d=fitz.open(p);out=H/'active-original.png';d[n-1].get_pixmap(matrix=fitz.Matrix(2,2)).save(out)
digest=hashlib.sha256(out.read_bytes()).hexdigest()
old=H/f'render-{n}.json'
if old.exists(): assert json.loads(old.read_text())['render_sha256']==digest
if not old.exists():
 old.write_text(json.dumps(dict(pdf_page=n,printed_page=n-10,source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),scale=2,render_sha256=digest,path=out.name,lifecycle='Single overwritten review render; original PDF retained'))+'\n')
