"""Idempotent source-stage coordinator for the independently approved revision 4.

The historical frozen proposal is immutable; installation metadata is a separate
overlay. This command does not compile CLDF or build a database.
"""
import argparse,csv,hashlib,json,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent
REV=HERE/'revision4'
OUT=REV/'proposal'
FORMS=HERE.parents[2]
DATA=HERE.parents[5]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def validate():
 m=json.loads((REV/'frozen-manifest.json').read_text())
 for name,digest in m['files'].items():
  assert sha(REV/name)==digest,('Changed frozen artifact',name)
 a=json.loads((REV/'root-installation-approval.json').read_text())
 assert a['status']=='approved' and a['frozen_manifest_sha256']==sha(REV/'frozen-manifest.json')
 overlay=json.loads((HERE/'installation-overlay/manifest.json').read_text())
 assert overlay['frozen_manifest_sha256']==sha(REV/'frozen-manifest.json')
 for name,digest in overlay['files'].items():
  assert sha(DATA.parent/name)==digest,('Changed installation overlay',name)
 baseline=FORMS/'20260921-peterson-turi.csv'
 assert sha(baseline) in {m['baseline_csv_sha256'],sha(OUT/baseline.name)},'Unexpected canonical baseline drift'
 historical=list(csv.reader((HERE/'proposal'/baseline.name).open()))
 revised=list(csv.reader((OUT/baseline.name).open()))
 assert len(historical)==len(revised)==246
 assert all(a[:14]==b[:14] and set(a[14].split())<=set(b[14].split()) for a,b in zip(historical,revised))
 rows=[r for p in OUT.glob('*.csv') for r in csv.reader(p.open())]
 keys={r[10] for r in rows}
 assert len(rows)==len(keys)==648 and all(len(r)==15 for r in rows)
 assert all(not r[11] or r[11] in keys for r in rows)
 for p in OUT.glob('*.csv'):
  target=FORMS/p.name
  if p.name!=baseline.name and target.exists():assert sha(target)==sha(p),('Unexpected canonical drift',target)
 return m
def install():
 m=validate()
 for p in OUT.glob('*.csv'):shutil.copyfile(p,FORMS/p.name)
 for p in OUT.glob('peterson-turi-*.txt'):shutil.copyfile(p,DATA/'conversion'/p.name)
 for p in (HERE/'installation-overlay').glob('*.yaml'):shutil.copyfile(p,FORMS/p.name)
 return m
if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--install',action='store_true');a=ap.parse_args()
 m=install() if a.install else validate()
 print(json.dumps({'status':'installed' if a.install else 'validated','rows':m['rows'],'audit_units':m['audit_units']}))
