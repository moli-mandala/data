"""Pin a fresh physical-record sample to the combined lexical/prose preview."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent


def sample(seed, count=20):
    lexical = sorted(HERE.glob('reviewed-p*.json'))
    records = [r for f in lexical for r in json.loads(f.read_text())]
    selected = random.Random(seed).sample(records, count)
    preview = HERE / 'integration-preview'
    rows = [json.loads(line) for line in (preview / 'rich-row-audit.jsonl').read_text().splitlines()]
    physical = {r['physical_entry_key']: r for r in map(json.loads, (preview / 'physical-record-audit.jsonl').read_text().splitlines())}
    with (preview / '20260921-bhattacharya-ollari-entry-texts.csv').open(newline='') as stream:
        prose = list(csv.DictReader(stream))
    inputs = lexical + sorted(HERE.glob('comparison-reviewed-p*.json')) + [
        HERE / 'reviewed-relations.json', HERE / 'morphology-dispositions.json',
        HERE / 'explicit-reference-resolution.json',
        preview / 'rich-row-audit.jsonl', preview / 'physical-record-audit.jsonl',
        preview / '20260921-bhattacharya-ollari.csv',
        preview / '20260921-bhattacharya-ollari-entry-texts.csv']
    return dict(seed=seed, count=count, population=len(records),
                scope='Combined preview source-to-output audit; compiled IDs and graph survival remain a separate gate.',
                sampling='random.Random(seed).sample over page-ordered physical records',
                input_sha256={str(f.relative_to(HERE)): hashlib.sha256(f.read_bytes()).hexdigest() for f in inputs},
                records=[dict(sample_number=i+1, source_record=r,
                              preview_rows=[a for a in rows if a['physical_entry_key'] == r['entry_key']],
                              prose_rows=[p for p in prose if p['Entry_Key'] == r['entry_key']],
                              morphology_dispositions=physical[r['entry_key']]['morphology_dispositions'])
                         for i, r in enumerate(selected)])


def render(report, pdf, output):
    import pdfplumber
    from PIL import Image, ImageDraw
    with pdf.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != json.loads((HERE / 'manifest.json').read_text())['sha256']:
            raise ValueError('PDF differs from pinned source')
    output.mkdir(parents=True, exist_ok=True)
    records = [r for f in sorted(HERE.glob('reviewed-p*.json')) for r in json.loads(f.read_text())]
    with pdfplumber.open(pdf) as document:
        for item in report['records']:
            r = item['source_record']
            page = document.pages[r['pdf_page']-1]
            im = page.to_image(resolution=200).original
            scale = 200/300
            next_y = min([n['top'] for n in records if n['printed_page'] == r['printed_page'] and n['column'] == r['column'] and n['top'] > r['top']], default=im.height/scale)
            midpoint = im.width//2
            x0, x1 = (0, midpoint+15) if r['column'] == 1 else (midpoint-15, im.width)
            crop = im.crop((x0, max(0, int(r['top']*scale)-10), x1, min(im.height, int(next_y*scale)+5)))
            labelled = Image.new('RGB', (crop.width, crop.height+30), 'white')
            labelled.paste(crop, (0,30))
            ImageDraw.Draw(labelled).text((8,8), f"Sample {item['sample_number']} | p. {r['printed_page']} col. {r['column']}", fill='black')
            labelled.save(output / f"sample-{item['sample_number']:02d}.png")
            page.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pdf', type=Path)
    parser.add_argument('--render-dir', type=Path)
    args = parser.parse_args()
    if bool(args.pdf) != bool(args.render_dir):
        parser.error('--pdf and --render-dir must be used together')
    report = sample(args.seed)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    if args.pdf:
        render(report, args.pdf, args.render_dir)
