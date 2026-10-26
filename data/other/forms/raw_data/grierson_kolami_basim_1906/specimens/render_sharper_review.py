"""Unique native-resolution strips for the owner whole-specimen reread."""
import hashlib
import json
from pathlib import Path
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[6]
MANIFEST = HERE.parent / 'specimen-conservation-review/sharper-selected-originals-manifest.json'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    records = []
    for page in json.loads(MANIFEST.read_text())['pages']:
        source = ROOT / page['path']
        assert sha(source) == page['sha256']
        im = Image.open(source)
        for index, (top, bottom) in enumerate(((250, 1400), (1200, 3000), (2800, 4600), (4400, 6250))):
            rect = (180, top, 4320, bottom)
            path = source.parent / f'owner-sharp-p{page["printed_page"]}-body{index}.png'
            crop = im.crop(rect)
            if not path.exists():
                crop.save(path)
            records.append(dict(printed_page=page['printed_page'], source=str(source.relative_to(ROOT)),
                                source_sha256=page['sha256'], rectangle=list(rect),
                                crop=str(path.relative_to(ROOT)), crop_sha256=sha(path)))
    (HERE / 'sharper-owner-crop-locators.json').write_text(json.dumps(records, indent=2) + '\n')

if __name__ == '__main__':
    main()
