"""Small original views for scoped lead adjudication; no OCR or transcription."""
from pathlib import Path
import subprocess, io
from PIL import Image

ROOT=Path(__file__).resolve().parents[7]
OUT=ROOT/'tmp/jambu-next-source/ahirwati-scope-20260927'
src=ROOT/'tmp/pdfs/lsi-v9-2/LSI-V9-2.djvu'
for printed in [3,43,44,45,53,54,147,148,149,151,477]:
    physical=printed+15
    p=OUT/f'outside-printed-{printed}-physical-{physical}.png'
    if not p.exists():
        raw=subprocess.check_output(['/opt/homebrew/bin/ddjvu','-format=pnm',f'-page={physical}','-size=1600x2200',str(src)])
        Image.open(io.BytesIO(raw)).save(p)
for printed in range(304,322):
    physical=printed+15
    p=OUT/f'table-footer-printed-{printed}-physical-{physical}.png'
    if not p.exists():
        raw=subprocess.check_output(['/opt/homebrew/bin/ddjvu','-format=pnm',f'-page={physical}','-size=1600x2200',str(src)])
        im=Image.open(io.BytesIO(raw));im.crop((0,int(im.height*.78),im.width,im.height)).save(p)
print('controls ready')
