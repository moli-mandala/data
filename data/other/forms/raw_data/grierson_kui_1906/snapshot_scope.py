"""Inventory cached Nordic LSI text for whole-source planning; no lexical installation."""
import bz2
import hashlib
import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
SOURCE = ROOT / 'tmp/pdfs/lsi-sprakbanken.xml.bz2'
PAGES = set(range(457, 472))

def main():
    stack, pages, mentions = [], [], []
    with bz2.open(SOURCE, 'rb') as stream:
        for event, element in ET.iterparse(stream, events=('start', 'end')):
            if event == 'start':
                stack.append(element)
                continue
            if element.tag == 'page':
                volume = next((a.get('vol') for a in reversed(stack)
                               if a.tag == 'corpus' and a.get('vol')), None)
                if volume == '4':
                    sentences = [' '.join(w.text or '' for w in s.findall('w'))
                                 for s in element.findall('sentence')]
                    row = {'source_page': element.get('pno'),
                           'attributes': dict(element.attrib), 'sentences': sentences}
                    if element.get('pno', '').isdigit() and int(element.get('pno')) in PAGES:
                        pages.append(row)
                    hits = [s for s in sentences if re.search(
                        r'\b(?:kui|kūi|kuvi|kuwi|kandh\w*|khond\w*|kalahandi)\b', s, re.I)]
                    if hits:
                        mentions.append({**row, 'sentences': hits})
                stack[-2].remove(element)
                element.clear()
            stack.pop()
    artifacts = {'nordic-scope-snapshot.json': pages,
                 'nordic-volume-kui-mentions.json': mentions}
    for name, value in artifacts.items():
        (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    status = {
        'status': 'source scope discovery only; no lexical extraction or installation',
        'cached_source': str(SOURCE.relative_to(ROOT)),
        'cached_source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'requested_printed_pages': sorted(PAGES),
        'keyed_pages_found': [p['source_page'] for p in pages],
        'volume_mention_pages': [p['source_page'] for p in mentions],
        'limits': 'Keyed text omits tables and interlinear matter. Absence here is not absence from the original volume.',
        'required_whole_scope': [
            'Original chapter printed 457–471, all grammar, examples, specimens and notes',
            'Entire Kui comparative-table column, with prompt census verified from original headers',
            'Volume introduction, cross-references and errata for explicitly attributed evidence',
            'Resolve Kui versus Kuvi/Kuwi and Kalahandi specimen identity before canonical routing'],
        'canonical_changed': False,
    }
    (HERE / 'scope-snapshot-status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(status, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
