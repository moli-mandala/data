"""Validate/install only the exact independently approved Turi source-stage proposal."""
import argparse,csv,hashlib,json,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent
DATA=HERE.parents[6]
FORMS=HERE.parent.parent.parent.parent
OUT=HERE/'proposal'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def validate(require_approval=False):
 manifest=json.loads((HERE/'frozen-manifest.json').read_text())
 for name,expected in manifest['files'].items():
  p=HERE/name
  assert p.is_file() and sha(p)==expected,f'Stale frozen artifact: {name}'
 baseline=FORMS/'20260921-peterson-turi.csv'
 assert sha(baseline)==manifest['baseline_csv_sha256'],'Installed baseline changed'
 assert (OUT/baseline.name).read_bytes()==baseline.read_bytes()
 allrows=[r for p in OUT.glob('*.csv') for r in csv.reader(p.open())]
 keys={r[10] for r in allrows};assert len(keys)==len(allrows)==manifest['rows']
 assert all(len(r)==15 and r[2] and r[3] for r in allrows)
 audit=[json.loads(x) for x in (OUT/'whole-audit.jsonl').read_text().splitlines()]
 assert len(audit)==manifest['audit_units']
 assert all(set(a['entry_keys'])<=keys for a in audit)
 assert all(not r[11] or r[11] in keys for r in allrows)
 if require_approval:
  approved=json.loads((HERE/'root-installation-approval.json').read_text())
  assert approved['status']=='approved'
  assert approved['frozen_manifest_sha256']==sha(HERE/'frozen-manifest.json')
 return manifest
if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--install',action='store_true');a=ap.parse_args()
 m=validate(a.install)
 if a.install:
  # Shared registry/Bib changes belong to the coordinated installer; no writes to those here.
  for group in ('grammar','historical'):
   stem='20260927-peterson-turi-'+group
   for suffix in ('.csv','.yaml'):shutil.copyfile(OUT/(stem+suffix),FORMS/(stem+suffix))
   profile='peterson-turi-'+group+'.txt'
   shutil.copyfile(OUT/profile,DATA/'conversion'/profile)
 print(json.dumps(dict(status='installed' if a.install else 'validated_not_installed',rows=m['rows'],audit_units=m['audit_units'])))
