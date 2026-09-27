"""Reproduce the complete reviewed LSI Handuri chapter and standard-list source stage."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
DATA = PACKAGE.parents[4]
OUTPUT = DATA / 'data/other/forms/20260925-grierson-handuri.csv'
AUDIT = PACKAGE / 'audit.jsonl'
PDF = DATA.parent / 'tmp/pdfs/lsi-v9-4/LSI-V9-4.pdf'
PDF_SHA256 = 'ef007663270b3a0ef5ba26804404e4db1281fb5eb239614cadf69b20b3d5395f'
SOURCE = 'grierson1916handuri'
_spec = importlib.util.spec_from_file_location('handuri_full_inventory', PACKAGE / 'preview_full_source.py')
_preview = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_preview)
DIALECT = _preview.DIALECT

def explicit_table_tags(item):
    tags = _preview.grammar(item)
    if 14 <= item <= 31:
        tags += ['pl' if item in {17,18,19,23,24,25,29,30,31} else 'sg']
    if 138 <= item <= 155:
        if item in {138,142,146,150,153}: tags += ['m','sg']
        if item in {139,143,147,151,154}: tags += ['f','sg']
        if item in {140,141,144,145,148,149,152,155}: tags += ['pl']
        if item in {141,145,149}: tags += ['f']
    for start in (156,162,179,185,195,205,211):
        if start <= item < start+6:
            offset=item-start
            tags += [['1sg','2sg','3sg','1pl','2pl','3pl'][offset], 'sg' if offset<3 else 'pl']
    if item in {172,173,174,191,192,193,194,201,202,203,204}: tags += ['1sg','sg']
    if 162<=item<=167 or 211<=item<=216: tags += ['pret']
    if item in {191,192}: tags += ['progressive']
    if item in {192,193,203}: tags += ['pret']
    if item == 219: tags += ['participle']
    return list(dict.fromkeys(tags))

def generate():
    rows,audit = _preview.generate()
    by_key={r[10]:r for r in rows}
    aliases={'postposition':'postp','masc':'m','fem':'f','dem':'demonstrative','imp':'impv'}
    for cell in audit:
        section=cell.get('section','table')
        for key in cell['entry_keys']:
            row=by_key[key]
            if section=='table':
                item=int(cell['item'])
                tags=explicit_table_tags(item)+[DIALECT]
                notes=[]
                if cell['decision']=='reviewed_mark_uncertain':
                    tags.append('uncertain')
                    notes.append('Exact diacritic shape is uncertain in the scanned source; the literal reviewed reading is retained.')
                if item==173: notes.append('The source prints adjacent future forms without a separating mark; the literal sequence is retained.')
                if item==231: notes.append('The source adds “than him” although the English prompt says “than his sister”.')
                row[6]=' '.join(notes)
                row[14]=' '.join(dict.fromkeys(tags))
            elif section=='grammar':
                row[6]='Source grammatical example; the printed form and grammatical interpretation are retained.'
                row[14]=' '.join(dict.fromkeys(aliases.get(t,t) for t in row[14].split()))
            else:
                row[6]='Printed interlinear surface form with its aligned translation; repeated identical form-and-gloss attestations share this entry and retain every source locator.'
            row[13]=''
        cell['source_review_status']=cell['status']
        if cell['entry_keys']: cell['status']='ingested_uncertain' if cell.get('decision')=='reviewed_mark_uncertain' else 'ingested'
        if section=='table': cell['grammar_basis']='Explicit source prompt and numbered paradigm contrast; person, number, case and verbal categories transcribed without reconstructing a lemma.'
    assert len(rows)==587 and len(audit)==670
    assert sum(bool(a.get('reuse_entry_keys')) for a in audit)==100
    assert all(len(r)==15 and not r[13] and DIALECT in r[14] for r in rows)
    return rows,audit

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install',action='store_true')
    parser.add_argument('--check-pdf',action='store_true')
    args=parser.parse_args()
    if args.check_pdf:
        digest=hashlib.sha256()
        with PDF.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
        if digest.hexdigest()!=PDF_SHA256:raise SystemExit('Original source PDF hash mismatch')
    rows,audit=generate()
    if args.install:
        with OUTPUT.open('w',newline='') as f:csv.writer(f,lineterminator='\n').writerows(rows)
        with AUDIT.open('w') as f:
            for record in audit:f.write(json.dumps(record,ensure_ascii=False,sort_keys=True)+'\n')
    print(f'{len(audit)} source units; {len(rows)} forms; 100 reused specimen attestations; full database build deferred')
if __name__=='__main__':main()
