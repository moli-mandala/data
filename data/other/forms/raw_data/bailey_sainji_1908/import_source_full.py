"""Import Bailey's complete main Sainji chapter, printed pp. 52–56 (1908)."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
DATA = PACKAGE.parents[4]
INPUT = PACKAGE / 'chapter-transcription-work.tsv'
OUTPUT = DATA / 'data/other/forms/20260925-bailey-sainji.csv'
AUDIT = PACKAGE / 'audit.jsonl'
PDF = DATA.parent / 'tmp/pdfs/bailey-sainji/bailey1908.pdf'
PDF_SHA256 = '953f5da5ee9bc341cdf7eb558920e824dc6f15f7473a8c0d5c8134c401ad60b5'
SOURCE = 'bailey1908sainji'
LECT_TAG = ''
DECISIONS = {'ingest','hold_morphology','exclude_pattern','exclude_other_lect','exclude_other_language','exclude_blank','exclude_rejected_example'}

def read_source(input_path: Path = INPUT):
    rows = list(csv.DictReader(input_path.open(encoding='utf-8',newline=''),delimiter='\t'))
    if len(rows) != 241 or {int(r['page']) for r in rows} != set(range(52,57)):
        raise ValueError('Expected full 241-unit main chapter, printed pp. 52–56')
    glossary = [r for r in rows if r['section']=='glossary']
    if [(r['column'],int(r['item'])) for r in glossary] != [(c,i) for c in ('1','2') for i in range(1,24)]:
        raise ValueError('Incomplete two-column 46-line glossary')
    if [int(r['item']) for r in rows if r['section']=='sentences'] != list(range(1,23)):
        raise ValueError('Incomplete 22 sentences')
    if [int(r['item']) for r in rows if r['section']=='numerals'] != list(range(1,21)):
        raise ValueError('Incomplete20 cardinals')
    for r in rows:
        if r['decision'] not in DECISIONS or (not r['gloss'] and r['section']!='noun-untranslated') or not r['note']:
            raise ValueError(f'Invalid source row: {r}')
        if r['decision']=='ingest' and (not r['printed_form'] or '[see image]' in r['printed_form'] or '(?)' in r['printed_form']):
            raise ValueError(f'Unresolved accepted reading: {r}')
        if r['decision']!='ingest' and not r['source_pattern'] and r['decision']!='exclude_blank':
            raise ValueError(f'Unaccounted exclusion: {r}')
    return rows

def generate(input_path: Path = INPUT):
    out, audit = [], []
    for r in read_source(input_path):
        p,sec,col,item=int(r['page']),r['section'],r['column'],r['item']
        base=f'{SOURCE}:{sec}:{p}:{col}:{item}' if sec in ('glossary','numerals') else f'{SOURCE}:p{p}:{sec}:item:{item}'
        loc=f'p. {p}, {col} column, item {item}' if sec=='glossary' else f'p. {p}, {sec}, item {item}'
        forms=[unicodedata.normalize('NFC',x) for x in r['printed_form'].split(';')] if r['decision']=='ingest' else []
        tags=r['tags'].split(';')
        if len(tags)==1: tags*=len(forms)
        notes=r['answer_notes']
        notes=json.loads(notes) if notes.startswith('[') else [notes]*len(forms)
        if forms and (not all(forms) or len(tags)!=len(forms) or len(notes)!=len(forms)):
            raise ValueError(f'Answer/tag/note mismatch: {base}')
        keys=[]
        for n,(form,tag,note) in enumerate(zip(forms,tags,notes),1):
            key=base if n==1 else (f'{base}:answer{n}' if sec in ('glossary','numerals') else f'{base}:answer:{n}'); keys.append(key)
            if r['source_pattern'] and r['source_pattern']!=r['printed_form']:
                pat=r['source_pattern']
                note=('Source pattern: '+pat+('' if pat.endswith(('.', '?', '!')) else '.')+' '+note).strip()
            if ' ' in form and 'multiword-expression' not in tag: tag=(tag+' multiword-expression').strip()
            tag = (LECT_TAG+' '+tag).strip()
            gloss=('she' if 'f' in tag.split() else 'he') if r['gloss']=='he; she' else r['gloss']
            out.append(['sai','',form,gloss,'','',note,f'{SOURCE}[{loc}]','','',key,'','','',tag])
        audit.append(dict(source_cell_key=base,status='ingested' if forms else r['decision'],reason=r['note'],printed_page=p,scan_page=p+22,section=sec,column=col,item=item,english_headword=r['gloss'],printed_form_review=r['printed_form'] or r['source_pattern'],source_pattern=r['source_pattern'],source_notes=r['answer_notes'],entry_keys=keys,language_id='sai',citation_locator=loc))
    if len(out)!=270 or len({r[10] for r in out})!=len(out):
        raise ValueError('Unexpected forms or duplicate identity keys')
    return out,audit

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=INPUT)
    parser.add_argument('--install',action='store_true')
    parser.add_argument('--check-pdf',action='store_true')
    args=parser.parse_args()
    if args.check_pdf:
        h=hashlib.sha256()
        with PDF.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
        if h.hexdigest()!=PDF_SHA256: raise SystemExit('Original PDF hash mismatch')
    rows,audit=generate(args.input)
    print(f'{len(audit)} units, {len(rows)} forms; {dict(Counter(a["status"] for a in audit))}')
    staged = OUTPUT if args.install else PACKAGE/'full-staged.csv'
    audit_path = AUDIT if args.install else PACKAGE/'full-staged-audit.jsonl'
    if True:
        with staged.open('w',encoding='utf-8',newline='') as f: csv.writer(f).writerows(rows)
        audit_path.write_text(''.join(json.dumps(a,ensure_ascii=False,sort_keys=True)+'\n' for a in audit),encoding='utf-8')
if __name__=='__main__': main()
