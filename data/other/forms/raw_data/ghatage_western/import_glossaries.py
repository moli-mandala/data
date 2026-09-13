"""Reproduce Ghatage (1963, 1965) vocabulary imports from pinned OCR evidence.

Run from any directory. Dry runs go to data/tmp; --install writes canonical
CSVs and per-record audits. The checked OCR contains word boxes and confidence;
manual corrections never overwrite that evidence. No linguistic links inferred.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from layout import parse

RAW = HERE.parent
ROOT = RAW.parents[3]
VOLUMES = {
    'konkani': dict(source='ghatage-konkani1963', language='Ko',
                    dialect='South Kanara Chitrapur Sarasvat Konkani',
                    pages=(128,148), offset=8, total_pages=149,
                    sha256='72ddf107fdeef7d427a24d15addda1311c6731c6fc64f44d297a488baca5c4c5'),
    'kudali': dict(source='ghatage-kudali1965', language='M', dialect='Kudali (Vengurla)',
                   pages=(105,160), offset=10, total_pages=161,
                   sha256='f94ccc31234e1037feb5be0525f04340ed733ca28074b8569f854631674f8ea3'),
}
POS_TAGS = {'M': ['noun','m'], 'F': ['noun','f'], 'N': ['noun','n'],
            'V': ['verb'], 'Adj': ['adj'], 'Adv': ['adv'], 'Nu': ['num'],
            'Pro': ['pron'], 'Ind': ['indeclinable'], 'Conj': ['conj'],
            'Interj': ['interj'], 'Postp': ['postp'], 'Part': ['part']}
POS = re.compile(r'(?<!\w)(Adj|Adv|Interj|Postp|Conj|Part|Pro|Ind|Nu|M|F|N|V)[.,:]*?(?=\s|$)')
# The capitals can denote the source's morphophonemes and must not be lowercased.
# Unusual Latin-script OCR diacritics are retained under ocr-review; punctuation,
# digits, non-Latin noise, and replacement characters are not lexical heads.
ALLOWED = set('abcdefghijklmnopqrstuvwxyzBDJGKCTPəɛɔɨŋɲǰčšśṭḍṇḷñāēīōūãẽĩõũ: -()\u0303')

def tag(v):
    return f'dialect:{v["language"]}:{v["source"]}:{quote(v["dialect"],safe="")} '

def clean(s):
    return unicodedata.normalize('NFC', re.sub(r'\s+', ' ', s).strip())

def split_heads(left, name):
    """Scope trailing labels to each preceding head; combine adjacent genders."""
    left = clean(left)
    if name == 'konkani': return [(left, [])]
    matches=list(POS.finditer(left))
    if not matches: return [(left, [])]
    heads=[];end=0
    for m in matches:
        form=left[end:m.start()].strip(' ,.;:')
        tags=POS_TAGS[m[1]]
        if form: heads.append((form,list(tags)))
        elif heads: heads[-1][1].extend(t for t in tags if t not in heads[-1][1])
        else: return []
        end=m.end()
    if left[end:].strip(' .,;:'): return []  # unresolved material after grammatical label
    return heads

def permissible(form):
    for ch in form:
        if ch in ALLOWED: continue
        # Preserve recognizably Latin OCR letter substitutions under a typed flag.
        if ch.islower() and 'LATIN' in unicodedata.name(ch,''): continue
        if unicodedata.category(ch).startswith('M'): continue
        return False
    return bool(form) and any(ch.isalpha() for ch in form)

def build(name):
    v=VOLUMES[name]
    corrections=json.loads((HERE/'corrections.json').read_text()).get(name,{})
    output=[];audit=[];seen=set()
    for page in range(v['pages'][0],v['pages'][1]+1):
        for record in parse(name,page):
            key=f'{v["source"]}:p{record["printed_page"]}:c{record["col"]}:e{record["ordinal"]}'
            short=f'{page}:{record["col"]}:{record["ordinal"]}'
            seen.add(short)
            correction=corrections.get(short)
            left=record['left'];gloss=record['gloss'];review=['ocr:unreviewed']
            if correction:
                left=correction.get('left',left);gloss=correction.get('gloss',gloss)
                review=['image-reviewed']
            heads=split_heads(left,name)
            emitted=[];rejected=[]
            if correction and correction.get('exclude'): heads=[]
            for i,(form,grammar) in enumerate(heads,1):
                form=clean(re.sub(r'\s*:\s*',':',form)).strip(' ,.;')
                if not permissible(form) or not gloss.strip():
                    rejected.append(form);continue
                child=key if i==1 else f'{key}:response:{i}'
                tags=list(grammar)+[tag(v).strip()]
                lexical_gloss=clean(gloss)
                for label in re.findall(r'\((pl|sg)\.\)',lexical_gloss):
                    if label not in tags: tags.append(label)
                lexical_gloss=re.sub(r'\s*\((?:pl|sg)\.\)', '', lexical_gloss)
                if review==['ocr:unreviewed']: tags+=['ocr-review','uncertain']
                row=[v['language'],'',form,lexical_gloss,'','','',
                     f'{v["source"]}[p. {record["printed_page"]}, col. {record["col"]}, entry {record["ordinal"]}]',
                     '','',child,'','','',' '.join(tags)]
                output.append(row);emitted.append(dict(key=child,form=form,gloss=row[3],tags=tags))
            audit.append(dict(key=key,status='ingested' if emitted else 'excluded-unresolved-ocr',
                reason=(correction or {}).get('reason','OCR transcription; raw boxes retained; no etymology inferred'),
                source=v['source'],language=v['language'],dialect=v['dialect'],review=review,
                raw_record=record,parsed_left=left,parsed_gloss=gloss,emitted=emitted,rejected=rejected))
    assert not set(corrections)-seen, set(corrections)-seen
    assert len({r[10] for r in output})==len(output)
    return output,audit

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--install',action='store_true');args=ap.parse_args()
    for name in VOLUMES:
        rows,audit=build(name);stem=f'20260911-ghatage-{name}'
        out=ROOT/'tmp'/stem;out.mkdir(parents=True,exist_ok=True)
        csvpath=RAW.parent/f'{stem}.csv' if args.install else out/f'{stem}.csv'
        auditpath=RAW/f'{stem}-audit.jsonl' if args.install else out/f'{stem}-audit.jsonl'
        with csvpath.open('w',newline='') as f: csv.writer(f).writerows(rows)
        auditpath.write_text(''.join(json.dumps(a,ensure_ascii=False,sort_keys=True)+'\n' for a in audit))
        report=dict(VOLUMES[name],raw_records=len(audit),installed_rows=len(rows),
            statuses=dict(Counter(a['status'] for a in audit)),
            reviews=dict(Counter(a['review'][0] for a in audit)),
            input_sha256=hashlib.sha256((HERE/f'{name}-ocr.json.gz').read_bytes()).hexdigest(),
            corrections_sha256=hashlib.sha256((HERE/'corrections.json').read_bytes()).hexdigest())
        manifest=HERE/f'{name}-manifest.json' if args.install else out/'manifest.json'
        manifest.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(name,json.dumps({k:report[k] for k in ['raw_records','installed_rows','statuses','reviews']}))

if __name__=='__main__': main()
