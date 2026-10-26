"""Exact source-stage YAML amendment and whole parsed-field continuity."""
import csv
import hashlib
import json
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent
DATA=next(p for p in HERE.parents if (p/'tags.py').exists())
sys.path.insert(0,str(DATA))
import make_cldf
from source_meta import SourceMeta
p=DATA/'data/other/forms/20260927-nair-betta-kurumba.csv'
y=p.with_suffix('.yaml')
assert y.read_bytes()==(HERE/y.name).read_bytes()
meta=SourceMeta([y])
assert set(meta.sources)=={'nair2013bettakurumba','jayapal1978kurumba'}
assert meta.sources['jayapal1978kurumba']['reference']=={'editor':'OpenAI Codex','ocr':False,'etymology_provenance':'none'}
errors=[]
current,stats=make_cldf.parse_file(str(p),errors)
assert not errors and len(current)==206 and stats['converted']==0
# Historical settings have identical parsing policy; compare every emitted
# field except the process-local generated Row.id and input filename against historical parse.
old,oldstats=make_cldf.parse_file(str(SOURCE/'proposed.csv'),errors)
assert not errors and len(old)==206 and oldstats['converted']==0
for l,r in zip(old,current):
    assert {k:v for k,v in vars(l).items() if k not in ('id','input_file')}=={k:v for k,v in vars(r).items() if k not in ('id','input_file')}, l.entry_key
raw=list(csv.reader(p.open()))
assert p.read_bytes()==(SOURCE/'proposed.csv').read_bytes()
assert {r.entry_key for r in current}=={r[10] for r in raw}
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
receipt=json.loads((SOURCE/'installation-receipt.json').read_text())
assert sha(SOURCE/'installed-audit.jsonl')==receipt['outputs']['audit']
assert sha(DATA/'data/form-identities.csv')==receipt['durable_identity_sha256']
report={'status':'passed','parser_rows':206,'all_parsed_output_fields_equal_historical':True,'excluded_parser_bookkeeping':['id','input_file'],'declared_references':sorted(meta.sources),
        'canonical_yaml_sha256':sha(y),'csv_sha256':sha(p),'audit_sha256':sha(SOURCE/'installed-audit.jsonl'),
        'historical_receipt_sha256':sha(SOURCE/'installation-receipt.json'),'durable_identity_sha256':sha(DATA/'data/form-identities.csv')}
(HERE/'installed-validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
