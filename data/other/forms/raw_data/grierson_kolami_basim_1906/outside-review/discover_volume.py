"""Streaming discovery only; source images control acceptance of every lead."""
import bz2
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'data/source_meta.py').exists())
SOURCE = ROOT / 'tmp/pdfs/lsi-sprakbanken.xml.bz2'

def fold(text):
    return ''.join(c for c in unicodedata.normalize('NFD', text.casefold()) if not unicodedata.combining(c))

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()

def main():
    stack, records, pages = [], [], []
    pattern = re.compile(r'\b(?:kolam\w*|colam\w*|basim|pusad|wun|bhil\w*)\b')
    with bz2.open(SOURCE, 'rb') as stream:
        for event, element in ET.iterparse(stream, events=('start', 'end')):
            if event == 'start':
                stack.append(element)
                continue
            if element.tag == 'page':
                volume = next((a.get('vol') for a in reversed(stack) if a.tag == 'corpus' and a.get('vol')), None)
                if volume == '4':
                    sentences = [' '.join(w.text or '' for w in s.findall('w')) for s in element.findall('sentence')]
                    hits = [{'sentence_index': i + 1, 'raw': s, 'matched_labels': pattern.findall(fold(s))} for i, s in enumerate(sentences) if pattern.search(fold(s))]
                    pages.append({'printed_page': element.get('pno'), 'sentences': len(sentences)})
                    if hits:
                        records.append({'printed_page': element.get('pno'), 'attributes': dict(element.attrib), 'hits': hits, 'complete_page_sentences': sentences})
                stack[-2].remove(element)
                element.clear()
            stack.pop()
    output = {'status': 'Unaccepted discovery leads; no source literal acceptance', 'source_sha256': sha(SOURCE), 'method': 'NFD accent-insensitive label detection, exact raw keyed context retained; geographic leads separately adjudicated', 'all_volume_keyed_pages': pages, 'lead_pages': records, 'limits': 'Keyed prose omits images, tables and some front/back matter. Original index, errata and table-header closure must be separately checked.'}
    (HERE / 'normalized-volume-leads.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'keyed_pages': len(pages), 'lead_pages': [r['printed_page'] for r in records], 'lead_sentences': sum(len(r['hits']) for r in records)}))

if __name__ == '__main__':
    main()
