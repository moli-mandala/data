"""Validate literal fields and actual graph fields with the unmodified parser."""
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import unicodedata as ud
from pathlib import Path
P=Path(__file__).resolve().parent;D=P.parents[4]
R=P/'graph-reconciliation-r3'
S=D.parent/'tmp/kholosi-recovery/parser-sandbox-r3'
F=S/'data/other/forms';F.mkdir(parents=True,exist_ok=True)
(S/'conversion').mkdir(exist_ok=True)
code={}
for name in ('make_cldf.py','source_meta.py'):
 shutil.copyfile(D/name,S/name);code[name]=hashlib.sha256((D/name).read_bytes()).hexdigest()
shutil.copyfile(P/'whole-proposal-r2/whole-metadata-proposed.yaml',F/'20220913-kholosi.yaml')
shutil.copyfile(P/'whole-proposal-r2/whole-profile-proposed.txt',S/'conversion/kholosi-source.txt')
rows=list(csv.reader((R/'whole-proposed.csv').open()))
os.chdir(S);sys.path[:0]=[str(S),str(D)]
import make_cldf
from tags import GRAMMATICAL_TAGS,GENDER_TAGS
unknown={t for r in rows for t in r[14].split()}-(GRAMMATICAL_TAGS|GENDER_TAGS)
assert not unknown,unknown
reports={};displays=None
for normalization in ('NFC','NFD'):
 normalized=[[ud.normalize(normalization,v) if i==2 else v for i,v in enumerate(r)] for r in rows]
 path=F/'20220913-kholosi.csv'
 with path.open('w',newline='') as f:csv.writer(f).writerows(normalized)
 errors=io.StringIO();parsed,stats=make_cldf.parse_file(str(path),errors,name='20220913-kholosi')
 assert not errors.getvalue(),errors.getvalue()
 assert len(parsed)==len(rows)==stats['converted']
 for p,r in zip(parsed,normalized):
  assert (p.entry_key,p.param,p.variant_of_key,p.borrowed_from_key,p.derivation_parent_keys)==tuple(r[i] for i in (10,1,11,12,13))
  assert (p.old_form,p.ipa,p.notes,p.source,p.etymology)==tuple(r[i] for i in (2,5,6,7,9))
  assert set(r[14].split())<=set(p.tags.split())
 result=[ud.normalize('NFC',p.form) for p in parsed]
 if displays is None:displays=result
 else:assert displays==result
 reports[normalization]=dict(rows=len(parsed),parser_errors=0,exact_graph_fields=True)
keys={r[10] for r in rows};assert len(keys)==len(rows)==1135
assert all(not r[11] or r[11] in keys for r in rows)
assert all(k in keys and k!=r[10] for r in rows for k in r[13].split())
report=dict(status='passed actual parser; independent acceptance pending',checks=reports,closed_graph=True,
 unknown_tags=[],code_sha256=code,proposal_sha256=hashlib.sha256((R/'whole-proposed.csv').read_bytes()).hexdigest(),
 profile_sha256=hashlib.sha256((P/'whole-proposal-r2/whole-profile-proposed.txt').read_bytes()).hexdigest(),
 method='Unmodified parser in bounded source-local sandbox; exact ancestry/variant/borrowing/derivation fields checked. No CLDF or database build.')
(R/'actual-parser-validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
