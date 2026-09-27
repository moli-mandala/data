"""Install the complete independently audited Rangri source stage; no database build."""
import argparse,csv,hashlib,importlib.util,json,shutil
from pathlib import Path
PACKAGE=Path(__file__).resolve().parent
DATA=PACKAGE.parents[4]
SOURCE='grierson1908malvirangri'
DIALECT='dialect:Malw:lsi1908-malvi-rangri:Rangri'
spec=importlib.util.spec_from_file_location('rangri_full_import',PACKAGE/'prepare_full.py')
full=importlib.util.module_from_spec(spec);spec.loader.exec_module(full)

def generate():
 rows,audit,_=full.generate()
 return rows,audit

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--install',action='store_true');parser.add_argument('--check-scan',action='store_true');args=parser.parse_args()
 if args.check_scan:
  scan=DATA.parent/'tmp/pdfs/lsi-v9-2/LSI-V9-2.djvu'
  with scan.open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()=='d6796bad8d267b776d2aec40dabb0b3b74d4f4fc87eb118f81db1b88e9dc6049'
 rows,audit=generate();assert len(rows)==1170 and len(audit)==1346
 assert rows==list(csv.reader((PACKAGE/'proposal.csv').open()))
 assert audit==[json.loads(x) for x in (PACKAGE/'proposal-audit.jsonl').read_text().splitlines()]
 if args.install:
  report=json.loads((PACKAGE/'independent-full-audit-20260926-pass2.json').read_text())
  assert report['status']=='pass' and report['sample_size']>=20 and report['material_errors']==0
  for name,h in report['hashes_before_and_after'].items():assert hashlib.sha256((PACKAGE/name).read_bytes()).hexdigest()==h,name
  for src,dst in [('proposal.csv',DATA/'data/other/forms/20260925-grierson-malvi-rangri.csv'),('proposal-audit.jsonl',PACKAGE/'audit.jsonl'),('proposal-profile.txt',DATA/'conversion/grierson-malvi-rangri-1908.txt')]:shutil.copyfile(PACKAGE/src,dst)
 print(f'{len(rows)} rows; {len(audit)} source audit units; source-stage only')
if __name__=='__main__':main()
