"""Reproduce the image-checked 1911 Vedda vocabulary (printed pp. 424–450).

The pinned PDF OCR is an audit scaffold, not a source of automatically accepted
headwords. verified.tsv records the image transcription. No network is needed.
Use --install after inspecting the preview; --sample SEED selects audit articles.
"""
import argparse
import csv
import hashlib
import json
import random
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE / 'seligmann_vedda_1911'
ROOT = HERE.parents[3]
STEM = '20260911-seligmann-vedda'
SOURCE = 'seligmann1911vedda'
LECTS = {
    'B': 'Bandaraduwa', 'Bl': 'Bulugahaladena', 'D': 'Dambani',
    'G': 'Godatalawa', 'K': 'Kovil Vanamai', 'L': 'Lindegala',
    'N': 'Nilgala', 'R': 'Rerenkadi', 'Tk': 'Tamankaduwa',
    'U': 'Unuwatura Bubula', 'W': 'Sitala Wanniya',
}

def dialect_id(label):
    return 'vedda_' + LECTS[label].lower().replace(' ', '_')

def dialect_tag(label):
    return f'dialect:Vedda:{dialect_id(label)}:{LECTS[label].replace(" ", "_")}'

def load():
    manifest = json.loads((PACKAGE / 'manifest.json').read_text())
    for name, expected in manifest['input_hashes'].items():
        if hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Checked source input changed: {name}; review and repin before installation')
    with (PACKAGE / 'verified.tsv').open() as f:
        return list(csv.reader(f, delimiter='\t'))

def emit(records=None):
    records = load() if records is None else records
    pages = json.loads((PACKAGE / 'pages.json').read_text())
    output, audit = [], []
    for page, item, gloss, groups in records:
        assert len([page, item, gloss, groups]) == 4
        tags = ['verb'] if '(v.)' in gloss else []
        gloss = gloss.replace(' (v.)', '')
        if item == '116.m': tags = ['m']
        if item == '116.f': tags = ['f']
        for gi, group in enumerate(groups.split(';'), 1):
            forms, labels = group.split('@')
            labels = labels.split()
            assert set(labels) <= set(LECTS) | {'O', 'T'}
            for fi, form in enumerate(forms.split(','), 1):
                key = f'{SOURCE}:{item}:g{gi}:f{fi}'
                row_tags = tags + [dialect_tag(x) for x in labels if x in LECTS]
                issues = []
                if 'T' in labels:
                    issues.append('dialect-mapping:T is not silently equated with Tk; p. 423 defines T as Tamil')
                if item == '17.i' and form == 'kanda arini':
                    issues.append('gloss:source questions identification as bambara on p. 427')
                if item == '3' and form == 'adane':
                    issues.append('transcription:source explicitly questions this reading; retained as printed')
                if issues: row_tags.append('uncertain')
                if (item, form) in {('87', 'den'), ('150', 'indepa'), ('149', 'gikiapan')}:
                    row_tags.extend(['verb', 'impv'])
                row_tags = list(dict.fromkeys(row_tags))
                notes = ''
                if 'O' in labels:
                    notes = 'Attested by Wannaku of Uniche.'
                if item == '2':
                    notes = 'At Bulugahaladena the authors report this word for a betel pouch.'
                if item == '38' and gi == 1:
                    notes = 'The authors report that this expression also means “to sell”.'
                row = ['Vedda', '', form, gloss, '', '', notes,
                       f'{SOURCE}[p. {page}, entry {item}]', '', '', key,
                       '', '', '', ' '.join(row_tags)]
                row = [unicodedata.normalize('NFC', x) for x in row]
                output.append(row)
                audit.append({
                    'entry_key': key, 'article': int(item.split('.')[0]),
                    'item': item, 'printed_page': int(page), 'pdf_page': int(page)+168,
                    'source_group': group, 'source_labels': labels,
                    'raw_ocr_page': f'seligmann_vedda_1911/pages.json#{page}',
                    'raw_ocr': pages[page], 'form': form, 'gloss': gloss,
                    'language': 'Vedda', 'tags': row_tags, 'source': row[7],
                    'status': 'ingested', 'review': 'headword, gloss and labels checked against page image',
                    'issues': issues,
                    'etymology_status': 'unlinked: lexical attestation scope; commentary retained in raw OCR, no inferred ancestry',
                    'variant_status': 'co-listed forms kept distinct; punctuation alone is not a variant claim',
                })
    assert {a['article'] for a in audit} == set(range(1,186))
    assert len({r[10] for r in output}) == len(output)
    return output, audit

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true')
    parser.add_argument('--sample', type=int)
    args = parser.parse_args()
    rows, audit = emit()
    if args.sample is not None:
        chosen = sorted(random.Random(args.sample).sample(range(1,186),20))
        print(json.dumps([a for a in audit if a['article'] in chosen], ensure_ascii=False, indent=2))
        return
    target = HERE.parent if args.install else ROOT / 'tmp/pdfs/seligmann-vedda/preview'
    target.mkdir(parents=True, exist_ok=True)
    with (target / f'{STEM}.csv').open('w', newline='') as f:
        csv.writer(f).writerows(rows)
    audit_path = HERE / f'{STEM}-audit.jsonl' if args.install else target / f'{STEM}-audit.jsonl'
    audit_path.write_text(''.join(json.dumps(a, ensure_ascii=False)+'\n' for a in audit))
    print(json.dumps({'articles':185, 'sense_records':len(load()), 'forms':len(rows),
                      'uncertain':sum(bool(a['issues']) for a in audit), 'output':str(target)}))

if __name__ == '__main__': main()
