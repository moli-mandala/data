"""Unique immutable crops for occurrence-specific underdot adjudications."""
import hashlib
import json
from pathlib import Path
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'data/source_meta.py').exists())
BASE = ROOT / 'tmp/jambu-next-source/kolami-specimen-originals'
REQUESTS = [
    (569, 'p569-line12-dead-man', [1050, 2030, 1435, 2150]),
    (569, 'p569-line1-dead-man', [970, 455, 1270, 570]),
    (564, 'p564-line15-came-correct-position', [1920, 2980, 2260, 3110]),
    (569, 'p569-line7-both-ini', [1600, 1320, 2220, 1435]),
    (565, 'p565-line2-ring', [1400, 640, 1650, 760]),
    (565, 'p565-line9-came', [620, 1580, 1100, 1730]),
    (565, 'p565-line15-came', [390, 2470, 830, 2600]),
    (564, 'p564-line15-came', [150, 3270, 1100, 3430]),
]

def main():
    rows = []
    for page, name, rect in REQUESTS:
        source = BASE / f'printed-{page}-2800.png'
        output = BASE / f'{name}.png'
        if not output.exists():
            with Image.open(source) as im:
                crop = im.crop(rect)
                crop.resize((crop.width * 4, crop.height * 4)).save(output)
        rows.append(dict(page=page, name=name, rect=rect, scale=4,
            source=str(source.relative_to(ROOT)), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            crop=str(output.relative_to(ROOT)), crop_sha256=hashlib.sha256(output.read_bytes()).hexdigest()))
    (HERE / 'underdot-control-locators.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps(rows))

if __name__ == '__main__':
    main()
