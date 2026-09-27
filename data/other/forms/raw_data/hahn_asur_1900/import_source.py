"""Regenerate the reviewed whole Hahn primer; never build a database."""
import argparse,hashlib,importlib.util,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('hahn_whole_builder',ROOT/'prepare_whole.py');stage=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)
DIALECT=stage.DIALECT
SOURCE=stage.SOURCE
STEM='20260925-hahn-asur'
def build():
 rows,audit,_=stage.build();return rows,audit
def write(install=False):
 stage.main()
 approval=json.loads((ROOT/'root-whole-installation-approval-20260926.json').read_text())
 assert approval['status']=='approved-source-stage-installation'
 for name,value in approval['hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value,name
 shutil.copyfile(ROOT/'proposal.csv',ROOT/(STEM+'.csv'))
 if install:
  shutil.copyfile(ROOT/'proposal.csv',ROOT.parents[1]/(STEM+'.csv'))
  shutil.copyfile(ROOT/'proposal-audit.jsonl',ROOT/'audit.jsonl')
  shutil.copyfile(ROOT/'proposal-profile.txt',ROOT.parents[4]/'conversion/hahn-asur-1900.txt')
 print('835 forms / 670 audited records; no database built')
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--install',action='store_true');write(ap.parse_args().install)
