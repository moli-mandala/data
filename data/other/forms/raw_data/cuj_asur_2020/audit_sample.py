"""Render a seeded complete-entry sample alongside its rich output proposal."""
import argparse
import importlib.util
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent


def render(pdf, output, seed, count=20):
    import pdfplumber
    from PIL import Image, ImageDraw
    spec = importlib.util.spec_from_file_location('asur_importer', HERE / 'import_source.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidates = list(map(json.loads, (HERE / 'candidates.jsonl').open()))
    rows, audit = module.build(candidates, json.loads((HERE / 'audit.json').read_text()))
    entries = list(map(json.loads, (HERE / 'entries.jsonl').open()))
    sample = random.Random(seed).sample(entries, count)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    with pdfplumber.open(pdf) as doc:
        for n, record in enumerate(sample):
            groups = []
            for token in record['tokens']:
                if not groups or (token['pdf_page'], token['column']) != (groups[-1][-1]['pdf_page'], groups[-1][-1]['column']) or token['top'] < groups[-1][-1]['top'] - 20:
                    groups.append([])
                groups[-1].append(token)
            images = []
            for group in groups:
                page = doc.pages[group[0]['pdf_page'] - 1]
                left = 43 if group[0]['column'] == 1 else 202
                bbox = (left, max(65, min(t['top'] for t in group) - 3), left + 164,
                        min(567, max(t['bottom'] for t in group) + 3))
                images.append(page.crop(bbox).to_image(resolution=240).original)
            canvas = Image.new('RGB', (550, sum(i.height for i in images) + 25), 'white')
            ImageDraw.Draw(canvas).text((0, 0), record['entry_key'], fill='black')
            y = 25
            for im in images:
                canvas.paste(im, (0, y)); y += im.height
            canvas.save(output / f'{n:02d}.png')
            records.append({'entry_key': record['entry_key'], 'image': f'{n:02d}.png',
                            'rows': [r for r in rows if r[10].startswith(record['entry_key'] + ':')],
                            'candidate': next(r for r in candidates if r['entry_key'] == record['entry_key'])})
    (output / 'sample.json').write_text(json.dumps({'seed': seed, 'records': records}, ensure_ascii=False, indent=2) + '\n')
    for start in range(0, len(sample), 5):
        ims = [Image.open(output / f'{i:02d}.png') for i in range(start, min(start + 5, len(sample)))]
        canvas = Image.new('RGB', (550, sum(i.height for i in ims) + 25 * len(ims)), 'white')
        y = 0
        for im in ims:
            canvas.paste(im, (0, y)); y += im.height + 25
        canvas.save(output / f'sheet-{start // 5}.png')
    print(json.dumps([{'key': r['entry_key'], 'rows': r['rows']} for r in records], ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pdf', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    args = p.parse_args()
    render(args.pdf, args.output, args.seed)
