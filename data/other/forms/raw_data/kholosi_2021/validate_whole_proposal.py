"""Exercise real parser code in a small source-local sandbox, no CLDF build."""
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

P=Path(__file__).resolve().parent
D=P.parents[4]
S=D.parent/'tmp/kholosi-recovery/parser-sandbox'
STEM='20220913-kholosi'

def main():
 target=S/'data/other/forms';target.mkdir(parents=True,exist_ok=True)
 (S/'conversion').mkdir(exist_ok=True)
 hashes={}
 for name in ('make_cldf.py','source_meta.py'):
  shutil.copyfile(D/name,S/name);hashes[name]=hashlib.sha256((D/name).read_bytes()).hexdigest()
 meta=dict(file=STEM+'.csv',defaults=dict(transcription=dict(profile='kholosi-source',input='form',preserve_hyphens=True),
             gloss=dict(grammar_tags=True)),sources=dict(kholosi=dict(identity=dict(dedupe_by_entry_key=True),
             forms=dict(split_alternates=False),reference=dict(etymology_provenance='jambu',ocr=False))))
 (target/(STEM+'.yaml')).write_text(yaml.safe_dump(meta,sort_keys=False))
 (P/'whole-metadata-proposed.yaml').write_text(yaml.safe_dump(meta,sort_keys=False))
 shutil.copyfile(P/'whole-profile-proposed.txt',S/'conversion/kholosi-source.txt')
 rows=list(csv.reader((P/'whole-proposed.csv').open()))
 baseline=list(csv.reader((P/'legacy-baseline/20220913-kholosi.csv').open()))
 os.chdir(S);sys.path[:0]=[str(S),str(D)]
 import make_cldf
 from tags import GRAMMATICAL_TAGS,GENDER_TAGS
 unknown=sorted({t for r in rows for t in r[14].split()}-(GRAMMATICAL_TAGS|GENDER_TAGS))
 assert not unknown,unknown
 reports={};normal_displays=None
 for normalization in ('NFC','NFD'):
  normalized=[[ud.normalize(normalization,v) if i==2 else v for i,v in enumerate(r)] for r in rows]
  path=target/(STEM+'.csv')
  with path.open('w',newline='') as f:csv.writer(f).writerows(normalized)
  errors=io.StringIO();parsed,stats=make_cldf.parse_file(str(path),errors,name=STEM)
  assert not errors.getvalue(),errors.getvalue()
  assert len(parsed)==len(rows)==stats['converted']
  assert [p.entry_key for p in parsed]==[r[10] for r in rows]
  for p,r in zip(parsed,normalized):
   assert (p.old_form,p.ipa,p.notes,p.source)==(r[2],r[5],r[6],r[7]),p.entry_key
   assert set(r[14].split())<=set(p.tags.split()),p.entry_key
  displays=[ud.normalize('NFC',p.form) for p in parsed]
  if normal_displays is None:normal_displays=displays
  else:assert displays==normal_displays
  reports[normalization]=dict(rows=len(parsed),parser_errors=0,original_phonemic_notes_sources_preserved=True,keys_order_preserved=True)
 assert [r[10] for r in rows[:198]]==[r[10] for r in baseline]
 assert [r[1] for r in rows[:198]]==[r[1] for r in baseline]
 keys={r[10] for r in rows}
 assert len(keys)==len(rows)
 assert all(not r[11] or r[11] in keys for r in rows)
 assert all(k in keys for r in rows for k in r[13].split())
 report=dict(status='passed actual source parser; independent editorial acceptance still pending',checks=reports,
  preserved_legacy_keys=198,preserved_legacy_assignments=198,unknown_tags=unknown,closed_graph=True,
  code_sha256=hashes,proposal_sha256=hashlib.sha256((P/'whole-proposed.csv').read_bytes()).hexdigest(),
  profile_sha256=hashlib.sha256((P/'whole-profile-proposed.txt').read_bytes()).hexdigest(),
  method='Unmodified parser and metadata loader copied into bounded source tree; no monkeypatch, no canonical writes or database generation.',
  deferred=['Independent source/output audit','Canonical installation','Compiled identity/graph','Full suite','Browser QA'])
 (P/'whole-parser-validation.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report))

if __name__=='__main__':main()
