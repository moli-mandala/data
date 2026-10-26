"""Resumable, single-worker raw OCR of every page outside the prior wordlist.

This is evidence recovery, not lexical acceptance or canonical installation.
Only a single temporary page render is retained at a time.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import pypdfium2 as pdfium

PDF_SHA = 'f022a2a448445cdc9840891478bd5864e2ded1ae98985332b51f0938c075842d'
PAGES = [p for p in range(1, 111) if not 21 <= p <= 48]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tessdata', type=Path, required=True)
    args = parser.parse_args()
    assert sha(args.pdf) == PDF_SHA
    args.output.mkdir(parents=True, exist_ok=True)
    model_hashes = {name: sha(args.tessdata / f'{name}.traineddata')
                    for name in ('ben', 'eng')}
    recipe = dict(pdf_sha256=PDF_SHA, pdf_pages=110, selected_pdf_pages=PAGES,
                  prior_wordlist_pdf_pages=[21,48], render_scale=3,
                  languages='ben+eng', psm=3, workers=1, model_sha256=model_hashes,
                  tesseract_version=subprocess.check_output(['tesseract','--version'], text=True).splitlines()[0],
                  status='Raw unreviewed OCR evidence; no lexical acceptance or installation')
    manifest = args.output / 'recipe.json'
    if manifest.exists():
        assert json.loads(manifest.read_text()) == recipe, 'Recipe differs; choose a new output directory.'
    else:
        manifest.write_text(json.dumps(recipe, indent=2)+'\n')
    document = pdfium.PdfDocument(args.pdf)
    assert len(document) == 110
    env = dict(os.environ, OMP_THREAD_LIMIT='1')
    ledger = []
    for number in PAGES:
        target = args.output / f'p{number:03}.tsv'
        if not target.exists():
            page = document[number-1]
            bitmap = page.render(scale=3)
            image_path = args.output / 'current-page.png'
            bitmap.to_pil().save(image_path)
            bitmap.close(); page.close()
            temporary = target.with_suffix('.tmp')
            with temporary.open('w') as stream:
                subprocess.run(['tesseract', str(image_path), 'stdout', '--tessdata-dir',
                                str(args.tessdata), '-l', 'ben+eng', '--psm', '3',
                                '-c', 'tessedit_create_tsv=1'],
                               stdout=stream, check=True, env=env)
            assert temporary.stat().st_size > 0
            temporary.replace(target)
            image_path.unlink()
        ledger.append(dict(pdf_page=number, path=target.name, sha256=sha(target),
                           review_status='unreviewed OCR; structural and glyph review pending'))
        (args.output/'progress.json').write_text(json.dumps(ledger, indent=2)+'\n')
        print(f'page {number}/110: {len(ledger)}/{len(PAGES)} remaining-scope pages recovered', flush=True)
    assert len(ledger) == 82
    (args.output/'complete.json').write_text(json.dumps(dict(recipe=recipe, pages=ledger), indent=2)+'\n')


if __name__ == '__main__':
    main()
