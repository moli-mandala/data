from pathlib import Path
import fitz
H=Path(__file__).resolve().parent
W=next(p for p in H.parents if (p/'data/tags.py').exists())
d=fitz.open(W/'tmp/pdfs/korwa-kodaku-2022/source.pdf')
for p in (27,59,93,97,99,105,108):
    out=H/f'original-boundary-pdf{p:03d}-120dpi.png'
    if not out.exists(): d[p-1].get_pixmap(dpi=120).save(out)
print('Seven immutable scope images retained; no accepted glyph transcription.')
