"""Reproduce full reviewed LSI Sodochi chapter, specimen and comparative-table source stage."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
PACKAGE=Path(__file__).resolve().parent
DATA=PACKAGE.parents[4]
SOURCE='grierson1916sodochi'
OUTPUT=DATA/'data/other/forms/20260925-grierson-sodochi.csv'
AUDIT=PACKAGE/'audit.jsonl'
PDF=DATA.parent/'tmp/pdfs/lsi-v9-4/LSI-V9-4.pdf'
PDF_SHA256='ef007663270b3a0ef5ba26804404e4db1281fb5eb239614cadf69b20b3d5395f'
_spec=importlib.util.spec_from_file_location('sodochi_full_inventory',PACKAGE/'preview_full_source.py')
_module=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

def generate():return _module.generate()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--install',action='store_true');p.add_argument('--check-pdf',action='store_true');a=p.parse_args()
    if a.check_pdf:
        h=hashlib.sha256()
        with PDF.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        if h.hexdigest()!=PDF_SHA256:raise SystemExit('Original source PDF hash mismatch')
    rows,audit=generate()
    if a.install:
        with OUTPUT.open('w',newline='') as f:csv.writer(f,lineterminator='\n').writerows(rows)
        with AUDIT.open('w') as f:
            for record in audit:f.write(json.dumps(record,ensure_ascii=False,sort_keys=True)+'\n')
    print(f'{len(audit)} full-scope units; {len(rows)} forms; database build deferred')
if __name__=='__main__':main()
