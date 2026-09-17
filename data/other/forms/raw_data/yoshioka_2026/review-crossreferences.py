"""Render the complete reviewed cross-reference inventory from the pinned PDF."""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageDraw

script = Path(__file__).resolve().parent.parent / 'yoshioka_cleanup.py'
spec = importlib.util.spec_from_file_location('yoshioka_xref_review', script)
y = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = y
spec.loader.exec_module(y)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--pdf', type=Path, default=y.legacy.DEFAULT_PDF)
args = parser.parse_args()
with args.pdf.open('rb') as stream:
    if hashlib.file_digest(stream, 'sha256').hexdigest() != y.PDF_SHA:
        raise ValueError('Image review requires the pinned Yoshioka PDF')
out = args.output
out.mkdir(parents=True, exist_ok=True)
groups = json.loads((y.PACKAGE / 'crossreference-decisions.json').read_text())['image_review']
records = {r['key']: r for r in y.entry_records(y.load_snapshot())}
pdf = pdfium.PdfDocument(args.pdf)
for number, group in enumerate(groups, 1):
    panels = []
    for kind, keys in [('Index', group['source_keys']), ('Target group', group['target_keys'])]:
        spans = {}
        for key in keys:
            for line in records[key]['lines']:
                spans.setdefault(line['pdf_page'], []).append(line)
        for page_number, lines in sorted(spans.items()):
            # Target groups retain intervening root/stem context. Index rows
            # are shown separately to avoid long stretches of irrelevant index.
            runs = [lines] if kind == 'Target group' else [[line] for line in lines]
            page = pdf[page_number-1]
            rendered = page.render(scale=2).to_pil()
            for run in runs:
                top, bottom = min(l['top'] for l in run)-4, max(l['top'] for l in run)+18
                crop = rendered.crop((170, int(top*2), 1050, int(bottom*2)))
                panel = Image.new('RGB', (900, crop.height+28), 'white')
                ImageDraw.Draw(panel).text((8, 3), f'{number:02} {kind} | PDF {page_number}', fill='black')
                panel.paste(crop, (10, 24))
                panels.append(panel)
            page.close()
    # Bound each image to keep the source text legible in review.
    chunks, chunk, height = [], [], 0
    for panel in panels:
        if chunk and height + panel.height > 1800:
            chunks.append(chunk)
            chunk, height = [], 0
        chunk.append(panel)
        height += panel.height + 10
    if chunk:
        chunks.append(chunk)
    for part, chunk in enumerate(chunks, 1):
        sheet = Image.new('RGB', (900, sum(p.height+10 for p in chunk)), '#dddddd')
        top = 0
        for panel in chunk:
            sheet.paste(panel, (0, top))
            top += panel.height + 10
        sheet.save(out/f'group-{number:02}-{part}.png')
    print(number, group['printed_target'], len(chunks), flush=True)
