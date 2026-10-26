"""Retain uniquely named native controls, without changing source readings."""
import hashlib
import json
from pathlib import Path
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[6]
BASE = ROOT / 'tmp/jambu-next-source/kolami-specimen-originals'
RECTS = {
    565: {
        'servants-said': (2600, 370, 3530, 660),
        'lost-was': (1440, 1210, 2200, 1480),
        'big-son': (1540, 2030, 3060, 2310),
        'house-came-v2': (260, 2020, 1060, 2310),
        'went-v2': (2330, 2020, 3080, 2320),
        'said8-v2': (3440, 2300, 4040, 2610),
        'father-happy-v2': (2640, 2580, 3540, 2910),
        'big10-v2': (690, 2840, 1300, 3190),
        'big15-v2': (1610, 4230, 2150, 4590),
        'my16-v2': (3270, 4530, 3680, 4880),
        'thine-make-v2': (250, 4800, 2080, 5180),
        'son-enclosure': (1660, 1740, 2930, 2040),
        'father9': (2230, 2570, 2850, 2890),
        'big10-v3': (480, 2840, 940, 3180),
    },
    568: {
        'village-far': (1840, 1980, 2680, 2250),
        'squandering': (2980, 1980, 3610, 2250),
        'servants': (1160, 2530, 3650, 2820),
        'he7': (1090, 3090, 1640, 3390),
        'servants8': (1940, 3370, 3550, 3670),
        'say9': (2950, 3650, 3950, 3950),
        'fault10': (1970, 3920, 2880, 4220),
        'servant11': (2390, 4190, 3690, 4520),
        'went-saw': (640, 4470, 1930, 4800),
        'him12': (2710, 4470, 3470, 4800),
        'kiss13': (3150, 4750, 3970, 5080),
        'servants15': (3300, 5310, 4200, 5640),
        'clothes16': (1160, 5580, 4210, 5910),
        'hands-shoes': (250, 5860, 2770, 6220),
        'shoes-isolated': (2040, 5840, 2540, 6150),
        'say9-v2': (3670, 3650, 4300, 3950),
        'saw12-v2': (1750, 4470, 2230, 4800),
        'him12-v2': (3350, 4470, 3900, 4800),
    },
    569: {
        'all2': (280, 820, 780, 1110),
        'servant-said4': (340, 1370, 2530, 1680),
        'came3': (2700, 1090, 3270, 1390),
        'anger6': (770, 1920, 1760, 2230),
        'years7': (2160, 2200, 2700, 2520),
        'women9': (2980, 2740, 3690, 3080),
        'giving10': (340, 3030, 1070, 3350),
        'only11': (2270, 3300, 2890, 3630),
        'years7-v2': (2920, 2130, 3470, 2430),
        'women9-v2': (3170, 2620, 3770, 2910),
        'giving10-v2': (280, 2890, 740, 3200),
        'only11-v2': (2260, 3130, 2850, 3440),
        'anger6-v2': (660, 1870, 1050, 2190),
    },
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    rows = []
    for page, controls in RECTS.items():
        source = BASE / f'alternate-leaf{page + 21}-printed{page}.jpg'
        im = Image.open(source)
        for name, rect in controls.items():
            path = BASE / f'owner-native-p{page}-{name}-control.png'
            if not path.exists():
                im.crop(rect).save(path)
            rows.append(dict(printed_page=page, name=name, rectangle=list(rect),
                             source=str(source.relative_to(ROOT)), source_sha256=sha(source),
                             crop=str(path.relative_to(ROOT)), crop_sha256=sha(path)))
    (HERE / 'sharp-remaining-glyph-control-locators.json').write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + '\n')

if __name__ == '__main__':
    main()
