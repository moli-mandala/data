"""Check the one-field reference correction and actual parser with its YAML."""
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import unicodedata as ud
from pathlib import Path
import yaml
P=Path(__file__).resolve().parent;D=P.parents[4]
A=P/'source-stage-metadata-addendum';F=P/'whole-proposal-r3'
before=yaml.safe_load((F/'whole-metadata-proposed.yaml').read_text())
after=yaml.safe_load((A/'whole-metadata-proposed.yaml').read_text())
expected=json.loads(json.dumps(before));expected['sources']['kholosi']['reference']['etymology_provenance']='mixed'
assert after==expected
S=D.parent/'tmp/kholosi-recovery/parser-metadata-addendum';T=S/'data/other/forms'
T.mkdir(parents=True,exist_ok=True);(S/'conversion').mkdir(exist_ok=True)
for name in ('make_cldf.py','source_meta.py'):shutil.copyfile(D/name,S/name)
shutil.copyfile(A/'whole-metadata-proposed.yaml',T/'20220913-kholosi.yaml')
shutil.copyfile(F/'whole-profile-proposed.txt',S/'conversion/kholosi-source.txt')
rows=list(csv.reader((F/'whole-proposed.csv').open()))
os.chdir(S);sys.path[:0]=[str(S),str(D)]
import make_cldf
checks={}
for normalization in ('NFC','NFD'):
 candidate=[[ud.normalize(normalization,v) if i==2 else v for i,v in enumerate(r)] for r in rows]
 path=T/'20220913-kholosi.csv'
 with path.open('w',newline='') as f:csv.writer(f).writerows(candidate)
 errors=io.StringIO();parsed,stats=make_cldf.parse_file(str(path),errors,name='20220913-kholosi')
 assert not errors.getvalue() and len(parsed)==1135==stats['converted']
 for p,r in zip(parsed,candidate):
  assert (p.entry_key,p.param,p.old_form,p.ipa,p.notes,p.source,p.etymology,p.variant_of_key,p.borrowed_from_key,p.derivation_parent_keys)==tuple(r[i] for i in (10,1,2,5,6,7,9,11,12,13))
  assert set(r[14].split())<=set(p.tags.split())
 checks[normalization]=dict(rows=1135,parser_errors=0,exact_literal_and_graph_fields=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report=dict(status='validated metadata-only proposal; independent approval pending',
 changes=[dict(path='sources.kholosi.reference.etymology_provenance',before='jambu',after='mixed')],
 reason='The same citation contains historical Jambu alternatives plus validated direct/qualified source ancestry and components; neither source-only nor Jambu-only describes the full graph provenance.',
 frozen_manifest_sha256=sha(F/'review-manifest.json'),csv_sha256=sha(F/'whole-proposed.csv'),
 profile_sha256=sha(F/'whole-profile-proposed.txt'),yaml_sha256=sha(A/'whole-metadata-proposed.yaml'),
 actual_parser_checks=checks,canonical_writes=False)
(A/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
