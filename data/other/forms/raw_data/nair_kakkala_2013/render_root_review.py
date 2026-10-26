"""Independent original-page renders and reproducible actual-output sample."""
import csv
import hashlib
import json
from pathlib import Path
import fitz

HERE = Path(__file__).resolve().parent
DEST = HERE / 'independent-review'
DEST.mkdir(exist_ok=True)
ROOT = next(p for p in HERE.parents if (p / 'tmp/pdfs/tribal-languages-kerala-nair2013.pdf').exists())
PDF = ROOT / 'tmp/pdfs/tribal-languages-kerala-nair2013.pdf'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(PDF) == '4d2588987dab5d15c62ecc95330ccf7e925a9bce80d08068b91ca3f48c8cc5af'
assert sha(HERE / 'proposed.csv') == '1f2cc08fe786e25c3618aafdbf6164149c18da2854b5cc88bcbdf813c0b8a4bd'
assert sha(HERE / 'proposal-audit.jsonl') == 'b92f5954a6878ca6d546388d68c98798e641dd96513e2cce78c34b6bbe5cdd44'
doc = fitz.open(PDF)
images = {}
rows = list(csv.reader((HERE / 'proposed.csv').open()))
sample = []
for page in range(28, 32):
    path = DEST / f'pdf{page}-original.png'
    doc[page - 1].get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False).save(path)
    images[str(page)] = sha(path)
    pool = [r for r in rows if f':p{page}:' in r[10]]
    selected = sorted(pool, key=lambda r: hashlib.sha256(('20260927461:' + r[10]).encode()).hexdigest())[:5]
    sample.extend({'page': page, 'entry_key': r[10], 'row': r} for r in selected)
assert len(sample) == 20
report = {'status': 'selected only; independent manual original/output review pending', 'seed': '20260927461',
          'sampling': 'SHA256(seed:Entry_Key), five per p28–31 stratum',
          'csv_sha256': sha(HERE / 'proposed.csv'), 'audit_sha256': sha(HERE / 'proposal-audit.jsonl'),
          'source_pdf_sha256': sha(PDF), 'image_sha256': images, 'sample': sample}
(DEST / 'root-fresh20-selection-20260927461.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
for item in sample:
    r = item['row']
    print(json.dumps({'page': item['page'], 'entry_key': r[10], 'form': r[2], 'gloss': r[3], 'notes': r[6], 'tags': r[14]}, ensure_ascii=False))
