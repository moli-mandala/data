"""Reproduce the manually transcribed target columns of the 2003 Varli survey.

Run from any directory, adding --install after inspecting the dry-run files.
The pipe-delimited transcription is read from the checked source layer, never OCR.
The PDF is optional for reproduction; --verify-source checks its pinned bytes.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import quote

RAW = Path(__file__).resolve().parent
ROOT = RAW.parents[3]
STEM = '20260911-dadra-varli'
SOURCE = 'pattanaik-koul2003varli'
PDF_SHA256 = '23024c757475badedfa8f84807701ad7adf7dbf32e991504e248ce74c9864d53'
PDF = ROOT.parent/'tmp/pdfs/ahirwal_2003/ahirwal-dadra-nagar-haveli-2003.pdf'
PAGE_ENDS = [(51,80),(102,81),(157,82),(210,83),(264,84),(319,85),(373,86),(414,87)]
LECTS = {'Davar':('Bhili','davar-varli-2003','Davar Varli'),
         'Dungar':('Varli','dungar-varli-2003','Dungar Varli')}
CONTROLS = ['Dhodia','Kokni/Kokna/Kukna','Gujarati','Marathi']
# The printed column boundary truncates these responses. Preserve what is visible,
# never supply an unprinted ending, and expose the uncertainty on the installed row.
QUALIFIED = {(162,'Davar',2): 'source-truncation: alternative ends at column boundary',
             (296,'Davar',2): 'source-truncation: alternative ends at column boundary',
             (346,'Davar',2): 'source-truncation: alternative ends at column boundary'}

def dialect_tag(lect):
    lang,alias,name=LECTS[lect]
    return f'dialect:{lang}:{alias}:{quote(name,safe="")}'

def read_transcription():
    with (RAW/f'{STEM}-transcription.tsv').open(newline='') as f:
        cells=list(csv.DictReader(f,delimiter='|'))
    assert [int(c['Item']) for c in cells] == list(range(1,415))
    assert all(set(c)=={'Item','Gloss','Davar','Dungar'} for c in cells)
    return cells

def build():
    output=[];audit=[]
    for cell in read_transcription():
        item=int(cell['Item']);page=next(p for end,p in PAGE_ENDS if item<=end)
        for lect in list(LECTS)+CONTROLS:
            key=f'{SOURCE}:p{page}:i{item:03}:{lect.lower().replace("/","-")}'
            base=dict(key=key,item=item,printed_page=page,pdf_page=page+14,
                      column=3+list(LECTS).index(lect) if lect in LECTS else 5+CONTROLS.index(lect),
                      lect=lect,gloss=cell['Gloss'],source=SOURCE,review='manual-page-image',
                      raw=cell.get(lect,''),forms=[])
            if lect in CONTROLS:
                base.update(status='excluded-control',reason='Comparison column; not transcribed or installed',review='scope-only')
            elif not cell[lect]:
                base.update(status='source-blank',reason='Printed empty target cell')
            else:
                base.update(status='ingested',reason='Direct transcription of printed response')
                for n,form in enumerate(re.split(r'\s*/\s*',cell[lect]),1):
                    tags=[dialect_tag(lect)]
                    # This is the sole explicit grammatical label inside the target table.
                    if '(pl. masc.)' in form:
                        form=form.replace(' (pl. masc.)','');tags+=['pl','m']
                    reason=QUALIFIED.get((item,lect,n),'')
                    if any(c in form for c in 'LC'):
                        reason='; '.join(filter(None,[reason,'transcription: undefined capital L/C retained without phonological reinterpretation']))
                    if reason: tags.append('uncertain')
                    form=unicodedata.normalize('NFC',form)
                    child=key if n==1 else f'{key}:response:{n}'
                    row=[LECTS[lect][0],'',form,cell['Gloss'],'','','',
                         f'{SOURCE}[p. {page}, item {item}, {lect} Varli]','','',child,'','','',' '.join(tags)]
                    output.append(row)
                    base['forms'].append(dict(key=child,form=form,tags=tags,uncertainty=reason))
            audit.append(base)
    assert len(audit)==414*6
    assert len({r[10] for r in output})==len(output)
    return output,audit

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--install',action='store_true');ap.add_argument('--verify-source',action='store_true')
    args=ap.parse_args()
    if args.verify_source:
        if not PDF.exists():raise SystemExit(f'Required PDF is absent: {PDF}')
        assert hashlib.sha256(PDF.read_bytes()).hexdigest()==PDF_SHA256
    rows,audit=build()
    out=ROOT/'tmp'/STEM;out.mkdir(parents=True,exist_ok=True)
    formpath=RAW.parent/f'{STEM}.csv' if args.install else out/f'{STEM}.csv'
    auditpath=RAW/f'{STEM}-audit.jsonl' if args.install else out/f'{STEM}-audit.jsonl'
    with formpath.open('w',newline='') as f:csv.writer(f).writerows(rows)
    auditpath.write_text(''.join(json.dumps(a,ensure_ascii=False,sort_keys=True)+'\n' for a in audit))
    report=dict(source=SOURCE,raw_concept_cells=len(audit),target_cells=828,control_cells=1656,
                audit_statuses=dict(Counter(a['status'] for a in audit)),installed_rows=len(rows),
                rows_by_language=dict(Counter(r[0] for r in rows)),qualified_responses=len(QUALIFIED),
                uncertain_output_rows=sum("uncertain" in r[14].split() for r in rows),
                linked=0,borrowed=0,variant_edges=0,pdf_sha256=PDF_SHA256,
                transcription_sha256=hashlib.sha256((RAW/f'{STEM}-transcription.tsv').read_bytes()).hexdigest(),
                source_url='https://censusindia.gov.in/nada/index.php/catalog/34827/download/38515/LSI_DADAR_NAGAR_HAVELI.pdf',
                licence='No explicit open licence verified; public government scan; extracted lexical facts only; scan not redistributed',
                source_pages=101,scope='Comparative Vocabulary List, printed pp. 80–87, Davar and Dungar columns',
                exclusions='Four comparison columns, grammar examples, texts, maps, front matter and bibliography',
                acquisition='Previously cached official PDF, verified 2026-09-11',
                transcription='Direct manual page-image transcription; source uppercase phonetic codes preserved in Original',
                sample_seed=20260911,sample_keys=[r[10] for r in random.Random(20260911).sample(rows,20)])
    manifest=RAW/f'{STEM}-manifest.json' if args.install else out/f'{STEM}-manifest.json'
    manifest.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['raw_concept_cells','audit_statuses','installed_rows','rows_by_language']},indent=2))

if __name__=='__main__':main()
