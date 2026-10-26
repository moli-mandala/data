"""Pin North Jubbal whole-chapter scope and retain the PDF text for review.

Text-layer output is finding evidence, never accepted literal transcription.
"""
import hashlib
import json
from pathlib import Path

import fitz

P = Path(__file__).resolve().parent
D = P.parents[4]
W = D.parent
OUT = P / 'whole-source-recovery'
PDF = W / 'tmp/pdfs/bailey1920/bailey1920.pdf'
SHA = '7a1577a2d2a24eca8b95e9b1ed700ef01f716d1069e4468f36a673611d093b39'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    assert sha(PDF) == SHA
    OUT.mkdir(exist_ok=True)
    pages = list(range(113, 119)) + [143] + list(range(172, 189))
    records = []
    with fitz.open(PDF) as doc:
        assert len(doc) == 310
        for printed in pages:
            page = doc[printed + 25]
            lines = []
            for block in page.get_text('dict')['blocks']:
                for line in block.get('lines', []):
                    lines.append({'bbox': list(line['bbox']),
                                  'spans': [{'text': s['text'], 'font': s['font'], 'size': s['size'], 'bbox': list(s['bbox'])}
                                            for s in line['spans']]})
            records.append({'printed_page': printed, 'pdf_page': printed + 26,
                            'text': page.get_text(), 'lines': lines,
                            'status': 'native finding layer; original image review required'})
    target = OUT / 'native-page-snapshot-20260927.jsonl'
    target.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    scope = {'status': 'whole-source recovery in progress', 'owner': 'root',
             'source_pdf': str(PDF.relative_to(W)), 'source_sha256': SHA,
             'printed_pages_cached': pages, 'snapshot_sha256': sha(target),
             'scope': {'shared_notation_and_attribution': '113–118; distinguish Koci controls from explicit Jubbal-applicable notation',
                       'cross_chapter_explicit_morphology': '143 explicitly assigns -ēṛu to North and South Jubbal; separate from other-lect comparisons',
                       'introduction': '172–175; separate North Jubbal from Bishshau and other comparisons',
                       'grammar': '175–182', 'complete_translated_responses': '183–184',
                       'existing_glossary': '185–188'},
             'canonical_before_sha256': sha(D / 'data/other/forms/20260925-bailey-north-jubbal.csv'),
             'existing_glossary_transcription_sha256': sha(P / 'full-transcription.tsv'),
             'policy': 'Recover every target form, paradigm and complete response. Preserve existing stable keys and literal marks; do not install OCR as a literal reading or attribute Bishshau controls to Barari.',
             'pending': ['complete original-page reading and source-unit inventory', 'reproducible whole-source proposal', 'profile/metadata/reference/graph checks', 'independent output audit', 'focused validation and source-stage installation'],
             'deferred': ['full CLDF/build/full suite', 'database/browser QA'],
             'canonical_changes': False}
    (OUT / 'scope-and-ownership-20260927.json').write_text(json.dumps(scope, indent=2) + '\n')
    print(json.dumps({'pages': len(records), 'lines': sum(len(r['lines']) for r in records), 'status': scope['status']}))


if __name__ == '__main__':
    main()
