from pathlib import Path
import fitz,sys,json,hashlib
H=Path(__file__).resolve().parent;W=next(p for p in H.parents if (p/'data/tags.py').exists())
n=int(sys.argv[1]);assert 93<=n<=105
p=W/'tmp/pdfs/korwa-kodaku-2022/source.pdf';doc=fitz.open(p)
out=H/'active-original.png';doc[n-1].get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(out)
(H/f'page-{n}-evidence.json').write_text(json.dumps(dict(pdf_page=n,printed_page=n-10,source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),render_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),scale=1.5,lifecycle='Single overwritten render; original retained'))+'\n')
