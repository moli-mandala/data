"""Reproduce complete Sansi source stage from reviewed chapter and table inventories."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
PACKAGE=Path(__file__).resolve().parent
DATA=PACKAGE.parents[4]
SOURCE='grierson1922lsi11'
OUTPUT=DATA/'data/other/forms/20260925-grierson-sansi.csv'
AUDIT=PACKAGE/'audit.jsonl'
PDF=DATA.parent/'tmp/pdfs/lsi-v11/LSI-V11.pdf'
PDF_SHA256='50df2b41e31420139148e2b16b321336881f227e917ca55cfedf225315cf4574'
_spec=importlib.util.spec_from_file_location('sansi_full_inventory',PACKAGE/'preview_full_source.py')
_module=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
def generate():return _module.generate()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--install',action='store_true');p.add_argument('--check-pdf',action='store_true');a=p.parse_args()
    if a.check_pdf:
        h=hashlib.sha256()
        with PDF.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        if h.hexdigest()!=PDF_SHA256:raise SystemExit('Original PDF hash mismatch')
    rows,audit=generate()
    if a.install:
        with OUTPUT.open('w',newline='') as f:csv.writer(f,lineterminator='\n').writerows(rows)
        with AUDIT.open('w') as f:
            for record in audit:f.write(json.dumps(record,ensure_ascii=False,sort_keys=True)+'\n')
    print(f'{len(audit)} full-scope units; {len(rows)} forms; database build deferred')
if __name__=='__main__':main()
