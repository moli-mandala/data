"""Canonical whole-source guard and repeatability, without compiled scans."""
import csv,hashlib,json,subprocess,sys
from pathlib import Path
from pybtex.database import parse_file
ROOT=Path(__file__).resolve().parents[1]
H=ROOT/'data/other/forms/raw_data/kharia_living_2026/whole-source-recovery'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def test_frozen_review_matches_installed():
 m=json.loads((H/'frozen-manifest.json').read_text())
 for n,d in m['files'].items():assert sha(H/n)==d
 a=json.loads((H/'root-installation-approval.json').read_text())
 assert a['status']=='approved' and a['frozen_manifest_sha256']==sha(H/'frozen-manifest.json')
 for ext in ['csv','yaml']:
  assert (ROOT/f'data/other/forms/20260911-kharia-living.{ext}').read_bytes()==(H/f'proposal/20260911-kharia-living.{ext}').read_bytes()
 assert (ROOT/'conversion/kharia-living.txt').read_bytes()==(H/'proposal/kharia-living.txt').read_bytes()
 bib=parse_file(str(ROOT/'cldf/sources.bib'));proposal=parse_file(str(H/'proposal/source.bib'))
 assert bib.entries['living-kharia2026']==proposal.entries['living-kharia2026']
def test_both_regeneration_entrypoints_preserve_whole_source():
 paths=[ROOT/'data/other/forms/20260911-kharia-living.csv',ROOT/'data/other/forms/20260911-kharia-living.yaml',ROOT/'conversion/kharia-living.txt']
 before={str(p):sha(p) for p in paths}
 for script in [H/'install_proposal.py',ROOT/'data/other/forms/raw_data/kharia_living.py']:
  subprocess.run([sys.executable,str(script),'--install'],cwd=ROOT,check=True,capture_output=True)
 assert before=={str(p):sha(p) for p in paths}
