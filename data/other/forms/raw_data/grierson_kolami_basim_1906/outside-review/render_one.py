"""One bounded immutable discovery original at a time, no OCR."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'data/source_meta.py').exists())
printed = int(sys.argv[1])
physical = int(sys.argv[2]) if len(sys.argv) > 2 else printed + 20
image = HERE / f'original-printed-{printed:03}-1600.jpg'
manifest = image.with_suffix('.json')
if image.exists():
    assert manifest.exists()
    assert hashlib.sha256(image.read_bytes()).hexdigest() == json.loads(manifest.read_text())['sha256']
else:
    scratch = HERE / 'unreviewed-render.ppm'
    subprocess.run(['/opt/homebrew/bin/ddjvu', '-format=ppm', f'-page={physical}', '-size=1600x2300', str(ROOT / 'tmp/pdfs/LSI-V4.djvu'), str(scratch)], check=True)
    with Image.open(scratch) as source:
        source.convert('RGB').save(image, quality=90)
    scratch.unlink()
    manifest.write_text(json.dumps({'requested_printed_page': printed, 'djvu_physical_page': physical, 'printed_folio_visual_check_required': True, 'source_sha256': '33e9aaa220db22fcde712705581a69f7edcfc7e20b3e7a09347dc969768020f1', 'sha256': hashlib.sha256(image.read_bytes()).hexdigest(), 'original_render_not_ocr': True}, indent=2) + '\n')
print(image, flush=True)
