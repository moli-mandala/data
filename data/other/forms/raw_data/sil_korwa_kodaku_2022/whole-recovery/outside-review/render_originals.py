from pathlib import Path
import fitz
H=Path(__file__).resolve().parent
W=next(p for p in H.parents if (p/'data/tags.py').exists())
d=fitz.open(W/'tmp/pdfs/korwa-kodaku-2022/source.pdf')
for p in (11,12,13,14,28,57,59,114):
    out=H/f'original-pdf{p:03d}-150dpi.png'
    if not out.exists(): d[p-1].get_pixmap(dpi=150).save(out)
print('Rendered six original pages without overwriting evidence.')
