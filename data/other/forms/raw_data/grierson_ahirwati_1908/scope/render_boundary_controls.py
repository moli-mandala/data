"""Bounded title/end and homonymous-language controls; no OCR."""
from pathlib import Path
import io, subprocess
from PIL import Image

ROOT=Path(__file__).resolve().parents[7]
OUT=ROOT/'tmp/jambu-next-source/ahirwati-scope-20260927'
SRC=ROOT/'tmp/pdfs/lsi-v9-2/LSI-V9-2.djvu'
for physical in [1,2,3,4,438,439,440,493,494]:
    target=OUT/f'boundary-physical-{physical}.png'
    if target.exists(): continue
    raw=subprocess.check_output(['/opt/homebrew/bin/ddjvu','-format=pnm',f'-page={physical}','-size=1300x1800',str(SRC)])
    Image.open(io.BytesIO(raw)).save(target)
print('boundary controls ready')
