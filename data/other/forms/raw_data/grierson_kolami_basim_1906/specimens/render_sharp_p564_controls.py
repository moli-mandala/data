"""Retain exact native rectangles for newly resolved p564 glyphs."""
import hashlib
import json
from pathlib import Path
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[6]
SOURCE = ROOT / 'tmp/jambu-next-source/kolami-specimen-originals/alternate-leaf585-printed564.jpg'
RECTS = {
    'share': (1900, 1700, 2520, 1970),
    'went': (2070, 2270, 2540, 2560),
    'going': (2000, 3050, 2530, 3320),
    'husks': (3460, 3320, 3910, 3550),
    'running': (280, 5370, 720, 5680),
    'appeared': (2790, 3520, 3250, 3780),
    'hunger': (1660, 4080, 2160, 4370),
    'country': (400, 3050, 980, 3320),
    'day': (750, 4630, 1430, 4900),
    'will-go': (200, 4350, 780, 4650),
    'will-say': (1170, 4350, 1720, 4650),
    'presence-comma': (3440, 4350, 4000, 4650),
    'feed': (200, 3330, 720, 3580),
    'gave-not': (200, 3860, 710, 4130),
}

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    assert sha(SOURCE) == '46268f190aac6db45ddc1e4195d033e75edf04b87351ef5aa4a309819046f8a4'
    im = Image.open(SOURCE)
    rows = []
    for name, rect in RECTS.items():
        path = SOURCE.parent / f'owner-native-p564-{name}.png'
        if not path.exists(): im.crop(rect).save(path)
        rows.append(dict(name=name, source=str(SOURCE.relative_to(ROOT)), source_sha256=sha(SOURCE),
                         rectangle=list(rect), crop=str(path.relative_to(ROOT)), crop_sha256=sha(path)))
    (HERE / 'sharp-p564-glyph-control-locators.json').write_text(json.dumps(rows, indent=2)+'\n')

if __name__ == '__main__': main()
