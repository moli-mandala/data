"""Regenerate and install the complete independently reviewed Bailey Padari source stage."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, shutil
P=Path(__file__).resolve().parent
DATA=P.parents[4]
spec=importlib.util.spec_from_file_location('bailey_padari_full',P/'prepare_full.py')
full=importlib.util.module_from_spec(spec);spec.loader.exec_module(full)
generate=full.generate

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--install',action='store_true')
 parser.add_argument('--check-pdf',action='store_true')
 args=parser.parse_args()
 if args.check_pdf:
  original=DATA.parent/'tmp/pdfs/bailey-sainji/bailey1908.pdf'
  with original.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
  assert digest=='953f5da5ee9bc341cdf7eb558920e824dc6f15f7473a8c0d5c8134c401ad60b5'
 rows,audit=generate()
 assert len(rows)==768 and len(audit)==753
 if args.install:
  report=json.loads((P/'independent-full-audit-20260926-pass3.json').read_text())
  assert report['status']=='passed_independent_source_sample'
  for name,h in report['hashes'].items():
   assert hashlib.sha256((P/name).read_bytes()).hexdigest()==h, name
  import csv
  assert rows==list(csv.reader((P/'proposal.csv').open()))
  assert audit==[json.loads(x) for x in (P/'proposal-audit.jsonl').read_text().splitlines()]
  for src,dst in [('proposal.csv',DATA/'data/other/forms/20260925-bailey-padari.csv'),('proposal-audit.jsonl',P/'audit.jsonl'),('proposal-profile.txt',DATA/'conversion/bailey-padari-1908.txt')]:
   shutil.copyfile(P/src,dst)
 print(f'{len(rows)} rows; {len(audit)} physical units; source-stage only')
if __name__=='__main__':main()
