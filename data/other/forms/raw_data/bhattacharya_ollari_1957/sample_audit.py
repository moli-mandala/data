"""Reproduce a seeded source-to-preview audit sample and optional scan crops."""
import argparse
import hashlib
import json
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent


def sample(seed, count):
    files = sorted(HERE.glob('reviewed-p*.json'))
    records = [r for f in files for r in json.loads(f.read_text())]
    selected = random.Random(seed).sample(records, count)
    preview = HERE / 'preview/rich-row-audit.jsonl'
    rows = [json.loads(line) for line in preview.read_text().splitlines()]
    return {'seed':seed, 'count':count, 'population':len(records),
            'sampling':'random.Random(seed).sample over page-ordered physical records',
            'input_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
            'preview_sha256':hashlib.sha256(preview.read_bytes()).hexdigest(),
            'records':[{'sample_number':i+1, 'source_record':r,
                        'preview_rows':[a for a in rows if a['physical_entry_key']==r['entry_key']]}
                       for i,r in enumerate(selected)],
            'scope':'Lexical headwords, glosses, POS/gender, printed inflections and lexical display conversion. Excludes comparative prose and compiled graph.'}


def render(report, pdf, output):
    import pdfplumber
    expected = json.loads((HERE/'manifest.json').read_text())['sha256']
    with pdf.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest() != expected:
            raise ValueError('PDF differs from pinned source')
    output.mkdir(parents=True,exist_ok=True)
    with pdfplumber.open(pdf) as document:
        for item in report['records']:
            record=item['source_record']
            page=document.pages[record['pdf_page']-1]
            im=page.to_image(resolution=180).original
            midpoint=im.width//2
            x0,x1=(0,midpoint) if record['column']==1 else (midpoint,im.width)
            y=int(record['top']*.6)
            im.crop((x0,max(0,y-12),x1,min(im.height,y+230))).save(output/f"sample-{item['sample_number']:02d}.png")
            page.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--count',type=int,default=20)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pdf',type=Path)
    parser.add_argument('--render-dir',type=Path)
    args=parser.parse_args()
    if bool(args.pdf) != bool(args.render_dir):parser.error('--pdf and --render-dir must be used together')
    report=sample(args.seed,args.count)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    if args.pdf:render(report,args.pdf,args.render_dir)
