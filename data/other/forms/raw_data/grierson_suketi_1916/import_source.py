"""Regenerate and install the complete independently reviewed Grierson Suketi source stage."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, shutil
P=Path(__file__).resolve().parent
DATA=P.parents[4]
spec=importlib.util.spec_from_file_location('grierson_suketi_full',P/'prepare_full.py')
full=importlib.util.module_from_spec(spec);spec.loader.exec_module(full)
generate=full.generate

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--install',action='store_true')
 parser.add_argument('--check-pdf',action='store_true')
 args=parser.parse_args()
 if args.check_pdf:
  original=DATA.parent/'tmp/pdfs/lsi-v9-4/LSI-V9-4.pdf'
  with original.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
  assert digest=='ef007663270b3a0ef5ba26804404e4db1281fb5eb239614cadf69b20b3d5395f'
 rows,audit=generate()
 assert len(rows)==353 and len(audit)==440
 if args.install:
  report=json.loads((P/'independent-full-audit-20260926-pass1.json').read_text())
  assert report['state']=='passed_independent_source_sample'
  for name,h in report['hashes'].items():
   assert hashlib.sha256((P/name).read_bytes()).hexdigest()==h, name
  import csv
  assert rows==list(csv.reader((P/'proposal.csv').open()))
  assert audit==[json.loads(x) for x in (P/'proposal-audit.jsonl').read_text().splitlines()]
  for src,dst in [('proposal.csv',DATA/'data/other/forms/20260925-grierson-suketi.csv'),('proposal-audit.jsonl',P/'audit.jsonl'),('proposal-profile.txt',DATA/'conversion/grierson-suketi-1916.txt')]:
   shutil.copyfile(P/src,dst)
 print(f'{len(rows)} rows; {len(audit)} physical units; source-stage only')
if __name__=='__main__':main()
