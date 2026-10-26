"""Bounded whole-section discovery from cached sources; no installed outputs."""
import bz2
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
SOURCE = ROOT / 'tmp/pdfs/lsi-sprakbanken.xml.bz2'
PAGES = {561, 562, 563, *range(570, 576)}

def snapshot():
    stack, rows, mentions = [], [], []
    with bz2.open(SOURCE, 'rb') as stream:
        for event, el in ET.iterparse(stream, events=('start', 'end')):
            if event == 'start':
                stack.append(el)
                continue
            if el.tag == 'page':
                volume = next((a.get('vol') for a in reversed(stack) if a.tag == 'corpus' and a.get('vol')), None)
                if volume == '4':
                    sentences = [' '.join(w.text or '' for w in s.findall('w')) for s in el.findall('sentence')]
                    hits = [s for s in sentences if 'naik' in s.lower()]
                    if hits:
                        mentions.append({'source_page': el.get('pno'), 'attributes': dict(el.attrib), 'matching_sentences': hits})
                if volume == '4' and el.get('pno', '').isdigit() and int(el.get('pno')) in PAGES:
                    rows.append({'source_page': int(el.get('pno')), 'attributes': dict(el.attrib),
                                 'sentences': [' '.join(w.text or '' for w in s.findall('w')) for s in el.findall('sentence')]})
                stack[-2].remove(el)
                el.clear()
            stack.pop()
    payload = json.dumps(rows, ensure_ascii=False, indent=2)+'\n'
    (HERE/'nordic-scope-snapshot.json').write_text(payload)
    (HERE/'nordic-volume-naiki-mentions.json').write_text(json.dumps(mentions, ensure_ascii=False, indent=2)+'\n')
    (HERE/'scope-snapshot-status.json').write_text(json.dumps({
        'status': 'whole-section scope inspection; no lexical proposal or installation',
        'source': 'local cached tmp/pdfs/lsi-sprakbanken.xml.bz2',
        'requested_printed_pages': sorted(PAGES),
        'keyed_pages_found': [r['source_page'] for r in rows],
        'snapshot_sha256': hashlib.sha256(payload.encode()).hexdigest(),
        'keyed_text_limits': 'Missing tables and interlinear matter require original-image recovery; absence from this snapshot is not absence from the volume.',
        'canonical_language_candidate': 'Naiki / Naiki (Chanda) / naik1250, distinct from Naikri / sout1549',
        'pending': ['Original whole-chapter scope review', 'Shared introduction attribution review', 'Comparative table and errata applicability', 'Complete per-record census before parsing'],
        'canonical_changed': False,
    }, ensure_ascii=False, indent=2)+'\n')
    print(f'Snapshotted {len(rows)} selected keyed pages; no canonical changes')

if __name__ == '__main__':
    snapshot()
