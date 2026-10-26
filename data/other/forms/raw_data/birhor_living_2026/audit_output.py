"""Run one draft through the real parser with temporary in-memory settings."""
import argparse
import csv
import io
import json
import random
import sys
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[4]))
import make_cldf
import source_meta
from segments.tokenizer import Tokenizer
from import_source import build


def parse_draft(path):
    meta=source_meta.SourceMeta([ROOT/'20260922-birhor-living.yaml'])
    converter=Tokenizer(str(ROOT/'birhor-living.txt'))
    errors=io.StringIO()
    with patch.object(source_meta,'load',return_value=meta),patch.dict(make_cldf.convertors,{'birhor-living':converter}):
        parsed,stats=make_cldf.parse_file(str(path),errors,name='20260922-birhor-living')
    assert not errors.getvalue(),errors.getvalue()
    return parsed,stats

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,required=True);args=p.parse_args()
    directory=args.output/'other/forms';directory.mkdir(parents=True,exist_ok=True);path=directory/'20260922-birhor-living.csv'
    rows,audit,_=build()
    with path.open('w',newline='') as f:csv.writer(f).writerows(rows)
    parsed,stats=parse_draft(path)
    assert len(parsed)==len(rows)==3100
    assert {r.entry_key for r in parsed}=={r[10] for r in rows}
    origins={r[10]:record for record in audit for r in record['emitted_rows']}
    raw_rows={r[10]:r for r in rows}
    sample=random.Random(args.seed).sample(sorted(parsed,key=lambda r:r.entry_key),20)
    output=[]
    for r in sample:
        raw=origins[r.entry_key]['raw']
        result={'entry_key':r.entry_key,'source_entry':raw,'raw_row':raw_rows[r.entry_key],'parsed':{'form':r.form,'original':r.old_form,'phonemic':r.ipa,'native':r.native,'gloss':r.gloss,'tags':r.tags,'notes':r.notes,'source':r.source}}
        output.append(result)
        print(r.entry_key,repr(r.form),repr(r.old_form),repr(r.ipa),repr(r.gloss),r.tags)
    (args.output/f'output-sample-{args.seed}.json').write_text(json.dumps({'seed':args.seed,'stats':stats,'sample':output},ensure_ascii=False,indent=2)+'\n')
