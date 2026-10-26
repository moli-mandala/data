"""Bounded sequential contents/header originals; no OCR and no book accumulation."""
from pathlib import Path
import subprocess, hashlib, json
from PIL import Image
here=Path(__file__).resolve().parent
root=next(p for p in here.parents if (p/'data/source_meta.py').exists())
records=[]
for physical in range(8,16):
    image=here/f'frontmatter-physical-{physical:03}-1400.jpg'
    if not image.exists():
        scratch=here/'frontmatter-unreviewed.ppm'
        subprocess.run(['/opt/homebrew/bin/ddjvu','-format=ppm',f'-page={physical}','-size=1400x2100',str(root/'tmp/pdfs/LSI-V4.djvu'),str(scratch)],check=True)
        with Image.open(scratch) as im: im.convert('RGB').save(image,quality=88)
        scratch.unlink()
    records.append({'physical_page':physical,'image':str(image.relative_to(root)),'sha256':hashlib.sha256(image.read_bytes()).hexdigest()})
(here/'frontmatter-original-manifest.json').write_text(json.dumps(records,indent=2)+'\n')
print('Rendered or reused 8 small original contents pages; not accepted readings.')
