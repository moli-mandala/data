"""Render bounded unreviewed publisher correspondences for visual review only."""
import argparse, hashlib, json
from pathlib import Path
import pdfplumber
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent

def render(pdf, output, limit=30):
    assert 1 <= limit <= 50
    with pdf.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert digest == json.loads((ROOT/'source-manifest.json').read_text())['pdf_sha256']
    read = lambda name: [json.loads(s) for s in (ROOT/name).read_text().splitlines()]
    reviewed = {r['entry_key'] for r in read('native-recovery-review.jsonl')}
    raw = {r['entry_key']: r for r in read('positioned-candidates.jsonl')}
    proposals = [r for r in read('publisher-proposals.jsonl') if r['entry_key'] not in reviewed][:limit]
    output.mkdir(parents=True, exist_ok=True)
    evidence = []
    with pdfplumber.open(pdf) as document:
        sheet = None
        for i, proposal in enumerate(proposals):
            row = raw[proposal['entry_key']]
            page = document.pages[row['pdf_page']-1]
            hit = page.search(r'\(([^()]+)\)\s*-')[row['page_item']-1]
            # A multiline grammar label has a union box reaching the left
            # margin. The opening parenthesis still follows the headword.
            opening = hit['chars'][0]
            assert opening['text'] == '('
            # Page 149 visually checked: these are one-letter headwords, not wraps.
            short_headwords = {'cfelmahali2024:p149:entry:1', 'cfelmahali2024:p149:entry:2'}
            assert opening['x0'] > 120 or (row['entry_key'] in short_headwords and opening['x0'] > 118), (row['entry_key'], 'possible wrapped headword')
            box = (108, max(0, opening['top']-8), opening['x0']-1, opening['top']+13)
            crop = page.crop(box).to_image(resolution=300).original.convert('RGB')
            crop.thumbnail((1100, 95))
            if i % 7 == 0:
                sheet = Image.new('RGB', (1150, 7*135), 'white')
            y = (i % 7)*135
            ImageDraw.Draw(sheet).text((5, y+4), row['entry_key']+' '+proposal['query'], fill='black')
            sheet.paste(crop, (5, y+28))
            name = f'native-review-{i//7+1}.png'
            evidence.append({'entry_key': row['entry_key'], 'sheet': name, 'panel': i%7+1,
                             'crop_box_points': box, 'dpi': 300,
                             'status': 'rendered for review; not accepted'})
            if i % 7 == 6 or i == len(proposals)-1:
                sheet.save(output/name)
            page.close()
    (output/'render-evidence.json').write_text(json.dumps(evidence, indent=2)+'\n')
    return evidence

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=30)
    args = parser.parse_args()
    print(f'{len(render(args.pdf, args.output, args.limit))} headwords rendered; no readings accepted')
