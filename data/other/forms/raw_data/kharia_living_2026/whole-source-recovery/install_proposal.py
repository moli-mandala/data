"""Guarded idempotent local source-stage installation; no compiled build."""
import argparse,csv,hashlib,json,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent
DATA=HERE.parents[5];OUT=HERE/'proposal';FORMS=DATA/'data/other/forms'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def validate(approval=False):
 m=json.loads((HERE/'frozen-manifest.json').read_text())
 for n,digest in m['files'].items():assert sha(HERE/n)==digest,('Frozen artifact drift',n)
 target=FORMS/'20260911-kharia-living.csv';profile=DATA/'conversion/kharia-living.txt'
 assert sha(target) in {m['baseline_csv_sha256'],sha(OUT/target.name)}
 assert sha(profile) in {m['baseline_profile_sha256'],sha(OUT/profile.name)}
 rows=list(csv.reader((OUT/target.name).open()));assert len(rows)==len({r[10] for r in rows})==524
 old=[r for a in map(json.loads,(HERE.parents[1]/'20260911-kharia-living-audit.jsonl').read_text().splitlines()) for r in a['rows']]
 indexed={r[10]:r for r in rows}
 for r in old:assert all(indexed[r[10]][i]==r[i] for i in [0,1,2,4,5,7,8,9,10,11,12,13])
 if approval:
  a=json.loads((HERE/'root-installation-approval.json').read_text())
  assert a['status']=='approved' and a['frozen_manifest_sha256']==sha(HERE/'frozen-manifest.json')
 return m
def install():
 m=validate(True)
 for ext in ['csv','yaml']:shutil.copyfile(OUT/f'20260911-kharia-living.{ext}',FORMS/f'20260911-kharia-living.{ext}')
 shutil.copyfile(OUT/'kharia-living.txt',DATA/'conversion/kharia-living.txt')
 return m
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--install',action='store_true');a=p.parse_args()
 m=install() if a.install else validate()
 print(json.dumps({'status':'installed' if a.install else 'validated_not_installed','rows':m['rows']}))
