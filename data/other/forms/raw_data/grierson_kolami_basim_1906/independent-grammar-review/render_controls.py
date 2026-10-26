from pathlib import Path
import subprocess
from PIL import Image

ROOT=Path(__file__).resolve().parents[7]
OUT=Path(__file__).parent/'original-controls'
OUT.mkdir(exist_ok=True)
for page in (561,562,563,565,566,567):
    raw=OUT/f'printed-{page}.ppm'
    subprocess.run(['/opt/homebrew/bin/ddjvu','-format=ppm',f'-page={page+20}','-size=2800x3962',str(ROOT/'tmp/pdfs/LSI-V4.djvu'),str(raw)],check=True)
    im=Image.open(raw)
    for label,box in [('top',(200,350,2450,1950)),('bottom',(200,1900,2450,3500))]:
        im.crop(box).save(OUT/f'printed-{page}-{label}.png')
    raw.unlink()
    print(page,flush=True)
