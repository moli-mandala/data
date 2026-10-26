"""Record complete recovered page evidence without accepting OCR as forms."""
import argparse
import hashlib
import json
from pathlib import Path
from extract import read_lines
from recover_remaining_ocr import PDF_SHA, PAGES

HERE=Path(__file__).resolve().parent


def scope(page):
    for start,end,label in [
        (1,6,'front-matter'),(7,20,'grammar'),(21,48,'prior-main-wordlist'),
        (49,61,'bengali-chakma'),(62,70,'homophones'),(71,76,'named-variety-comparison'),
        (77,78,'language-abbreviations'),(79,89,'cross-language-comparisons'),
        (90,92,'kokborok-chakma'),(93,95,'brok-skad-chakma'),
        (96,99,'santali-chakma'),(100,103,'further-comparisons'),
        (104,107,'language-demographics'),(108,109,'basic-wordlist'),(110,110,'back-cover')]:
        if start<=page<=end:return label
    raise AssertionError(page)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    args=parser.parse_args()
    completion=json.loads((args.cache/'complete.json').read_text())
    assert completion['recipe']['pdf_sha256']==PDF_SHA
    assert [r['pdf_page'] for r in completion['pages']]==PAGES
    target=HERE/'whole-remaining-raw-lines.jsonl'
    total=0
    with target.open('w') as stream:
        for record in completion['pages']:
            path=args.cache/record['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']
            lines=read_lines(path.read_text(),0)
            total+=len(lines)
            stream.write(json.dumps(dict(pdf_page=record['pdf_page'],
                scope_proposal=scope(record['pdf_page']),raw_tsv_sha256=record['sha256'],
                status='unreviewed OCR; no lexical acceptance',lines=lines),ensure_ascii=False)+'\n')
    manifest=dict(stage='All remaining-page OCR recovered; structure/transcription review pending',
        pdf_sha256=PDF_SHA,pages=82,raw_ocr_lines=total,
        recipe=completion['recipe'],raw_lines_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        page_caches=completion['pages'],canonical_rows_changed=0,
        next='Review every page and lexical-bearing unit; preserve prior742 keys and exact overlapping attestations.')
    (HERE/'whole-remaining-ocr-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(pages=82,raw_ocr_lines=total,canonical_rows_changed=0)))


if __name__=='__main__':main()
