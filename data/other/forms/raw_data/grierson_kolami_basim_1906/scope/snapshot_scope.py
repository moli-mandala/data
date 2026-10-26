"""Scope-only cache of keyed Nordic prose; original pages control every reading."""
import bz2
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/'data/source_meta.py').exists())
SOURCE = ROOT/'tmp/pdfs/lsi-sprakbanken.xml.bz2'


def main():
    stack, pages, mentions = [], [], []
    with bz2.open(SOURCE, 'rb') as stream:
        for event, element in ET.iterparse(stream, events=('start','end')):
            if event == 'start':
                stack.append(element); continue
            if element.tag == 'page':
                volume = next((a.get('vol') for a in reversed(stack) if a.tag=='corpus' and a.get('vol')),None)
                if volume == '4':
                    sentences = [' '.join(w.text or '' for w in s.findall('w')) for s in element.findall('sentence')]
                    number = element.get('pno','')
                    if number.isdigit() and 561 <= int(number) <= 569:
                        pages.append(dict(printed_page=int(number),attributes=dict(element.attrib),sentences=sentences))
                    hits = [s for s in sentences if any(x in s.casefold() for x in ('kōlā','kolam','basim','bāsim'))]
                    if hits: mentions.append(dict(printed_page=number,attributes=dict(element.attrib),matching_sentences=hits))
                stack[-2].remove(element);element.clear()
            stack.pop()
    for name, rows in [('nordic-prose-snapshot.json',pages),('nordic-volume-mention-leads.json',mentions)]:
        (HERE/name).write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    digest=hashlib.sha256()
    with SOURCE.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    report=dict(status='Unaccepted keyed discovery prose, not an original transcription',source_sha256=digest.hexdigest(),
        requested_printed_pages=list(range(561,570)),keyed_pages_found=[r['printed_page'] for r in pages],
        source_wide_mention_leads=len(mentions),canonical_changes=False,
        limits='Missing interlinear/table data is recovered from originals; mentions are leads, not completed outside closure.')
    (HERE/'snapshot-status.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
