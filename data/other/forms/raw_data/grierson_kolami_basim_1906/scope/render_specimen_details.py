"""Sequential immutable original crops; no OCR or source transcription."""
from pathlib import Path
import subprocess
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'data/source_meta.py').exists())
DEST=ROOT/'tmp/jambu-next-source/kolami-specimen-originals'


def main():
    DEST.mkdir(exist_ok=True)
    for page in (564,565,568,569):
        full=DEST/f'printed-{page}-2800.png'
        if not full.exists():
            ppm=DEST/f'printed-{page}.ppm'
            subprocess.run(['/opt/homebrew/bin/ddjvu','-format=ppm',f'-page={page+20}','-size=2800x3962',str(ROOT/'tmp/pdfs/LSI-V4.djvu'),str(ppm)],check=True)
            with Image.open(ppm) as image:image.save(full)
            ppm.unlink()
        with Image.open(full) as image:
            width,height=image.size
            # Overlapping thirds preserve line context; no reviewed image overwritten.
            for label,top,bottom in [('top',.115,.405),('middle',.365,.66),('bottom',.62,.89)]:
                path=DEST/f'printed-{page}-{label}.png'
                if not path.exists():image.crop((int(.12*width),int(top*height),int(.885*width),int(bottom*height))).save(path)
        print(page,flush=True)


if __name__=='__main__':main()
