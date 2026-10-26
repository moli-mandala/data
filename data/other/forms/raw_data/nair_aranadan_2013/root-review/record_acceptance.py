"""Root-reviewed acceptance gates; source acceptance does not install files."""
import argparse,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent.parent
D=next(p for p in P.parents if (p/'tags.py').exists())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
def read(name):return json.loads((P/name).read_text())
def save(name,value):
 text=json.dumps(value,indent=2)+'\n';p=P/name
 if p.exists() and p.read_text()!=text:
  assert '--refresh-uninstalled' in __import__('sys').argv
  assert not (P/'installation-receipt.json').exists()
  assert not (P/'integration/accepted-audit.jsonl').exists()
  backup=p.with_name(p.stem+'-superseded-'+sha(p)[:12]+'.json')
  assert not backup.exists() or backup.read_bytes()==p.read_bytes()
  backup.write_bytes(p.read_bytes())
 p.write_text(text)
def source():
 csv='integration/review-candidate.csv';audit='integration/review-audit.jsonl'
 assert sha(P/csv)=='d7479dff8846d69095766bae90b10abf837bbe8f5d989d38a635e2e52a90de42'
 assert sha(P/audit)=='0b8d183e825ed416bb5345a3cfc5829a0b0c7e137abc0030c1085ef527b6e517'
 full=read('review/integrated-verification.json')
 assert full['status']=='passed' and full['rows']==102 and full['units']==97
 assert full['physical_witnesses']==105 and full['variant_edges']==2
 assert full['hashes']['review-candidate.csv']==sha(P/csv)
 assert full['hashes']['review-audit.jsonl']==sha(P/audit)
 fresh=read('review/fresh-original-audit.json')
 assert fresh['status']=='passed' and fresh['sampled_count']==20 and fresh['material_error_count']==0
 assert fresh['csv_sha256']==sha(P/csv) and fresh['audit_sha256']==sha(P/audit)
 parser=read('independent-reading/actual-parser-review.json')
 for name,expected in parser['pins'].items():assert sha(P/name)==expected,name
 assert sha(D/'make_cldf.py')==parser['parser_sha256']
 for c in parser['checks'].values():
  assert c['rows']==102 and c['errors']==0 and c['all15fields_exact'] and c['ordered_keys']
 reports=['review/integrated-verification.json','review/fresh-original-audit.json',
  'independent-reading/actual-parser-review.json','independent-reading/metadata-formatter-review.json',
  'independent-reading/complete-independent-original-review.json','review/metadata-graph-proposal.json']
 names=[csv,audit,'root-review/approved-reconciliation.json','root-review/proposed-settings.yaml',
  'root-review/proposed-references.bib','root-review/ethnonym-disposition.json',
  'root-review/transcription-and-provenance.json','first-reading/complete-first-reading.jsonl',
  'integration/prepare_accepted_audit.py',*reports]
 files={name:sha(P/name) for name in names};assert all(files.values())
 save('root-review/source-acceptance.json',{'approved':True,'result':'pass','rows':102,'audit_units':97,
  'files':files,'review_reports':reports,'scope':'Whole chapter source-stage acceptance; full compiled/database/browser gates deferred'})
 print('Root source acceptance frozen; no installation.')
if __name__=='__main__':
 source()
