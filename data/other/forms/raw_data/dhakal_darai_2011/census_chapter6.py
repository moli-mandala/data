"""Inventory every Chapter 6 page; candidate counts are not acceptance counts."""
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build():
    source = HERE/'whole-native-pages.jsonl'
    pages = []
    with source.open() as handle:
        for line in handle:
            page = json.loads(line)
            number = page.get('printed_page')
            if number is None or not 146 <= number <= 205:
                continue
            headings, numbered, target_lines, private = [], [], [], []
            for item in page['lines']:
                text = ''.join(span['decoded'] for span in item['spans']).strip()
                if re.match(r'^6\.\d', text):
                    headings.append({'line':item['line'],'text':text})
                if re.match(r'^\(\d+\)', text):
                    numbered.append({'line':item['line'],'text':text})
                if any('SILDoulos' in span['font'] for span in item['spans']):
                    target_lines.append(item['line'])
                if '⟦' in text:
                    private.append({'line':item['line'],'text':text})
            pages.append({'printed_page':number,'pdf_page':page['pdf_page'],
                          'line_count':len(page['lines']),'headings':headings,
                          'numbered_example_starts':numbered,
                          'source_font_candidate_lines':target_lines,
                          'unresolved_decoding_lines':private,
                          'image_blocks':page['image_blocks'],
                          'status':'Full extraction and original review pending'})
    assert [page['printed_page'] for page in pages] == list(range(146,206))
    return {'status':'Complete 60-page structural census, not lexical recovery',
            'snapshot_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'scope':'All Chapter 6 prose, paradigm examples, morphemes, full expressions and aligned words; Appendix 3 paradigms remain a separate mandatory whole-thesis scope.',
            'caution':'Special-font lines include controls and glosses; ordinary-font target words may also occur. Counts cannot define lexical coverage.',
            'pages':pages,'canonical_changed':False}


if __name__ == '__main__':
    report=build()
    (HERE/'chapter6-scope-census-20260927.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(len(report['pages']),'pages accounted structurally; extraction and visual review pending')
