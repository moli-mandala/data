"""Render reproducible source-image samples against the current offline parser."""
import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / 'yoshioka_cleanup.py'
spec = importlib.util.spec_from_file_location('yoshioka_review_importer', SCRIPT)
source = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = source
spec.loader.exec_module(source)


def main():
    import pypdfium2 as pdfium
    from PIL import Image, ImageDraw
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=20260914)
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--keys', nargs='*')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = source.entry_records(source.load_snapshot())
    rows, audits = source.compile_records(records)
    accepted = {a['Entry_Key']: a for a in audits if a['Status'] == 'installed'}
    keys = args.keys or random.Random(args.seed).sample(sorted(accepted), args.count)
    by_key = {r['key']: r for r in records}
    args.output.mkdir(parents=True, exist_ok=True)
    doc = pdfium.PdfDocument(source.legacy.DEFAULT_PDF)
    panels, samples = [], []
    for key in keys:
        record = by_key[key]
        sample = {'Entry_Key': key, 'seed': args.seed, 'audit': accepted.get(key),
                  'rows': [r for r in rows if r[10] == key or r[10].startswith(key+':')]}
        samples.append(sample)
        for number in sorted({l['pdf_page'] for l in record['lines']}):
            lines = [l for l in record['lines'] if l['pdf_page'] == number]
            top, bottom = min(l['top'] for l in lines)-5, max(l['top'] for l in lines)+20
            page = doc[number-1]
            rendered = page.render(scale=2).to_pil()
            crop = rendered.crop((170, int(top*2), 1050, int(bottom*2)))
            panel = Image.new('RGB', (900, crop.height+34), 'white')
            ImageDraw.Draw(panel).text((8, 4), f'{key} | PDF {number}', fill='black')
            panel.paste(crop, (10, 28))
            panels.append(panel)
            page.close()
    for offset in range(0, len(panels), 5):
        group = panels[offset:offset+5]
        sheet = Image.new('RGB', (900, sum(x.height+12 for x in group)), '#dddddd')
        y = 0
        for panel in group:
            sheet.paste(panel, (0, y))
            y += panel.height+12
        sheet.save(args.output / f'sheet-{offset//5+1}.png')
    (args.output / 'sample.json').write_text(json.dumps(samples, ensure_ascii=False, indent=2)+'\n')
    print('\n'.join(keys))


if __name__ == '__main__':
    main()
