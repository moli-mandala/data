"""Guarded source-stage coordinator. Default is read-only preflight, never a build."""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import yaml
P=Path(__file__).resolve().parent
D=P.parents[4]
F=P/'whole-proposal-r3'
A=P/'source-stage-metadata-addendum'
I=P/'source-stage-importer-routing-addendum'
G=P/'source-stage-parent-delimiter-addendum'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def verify():
 manifest=json.loads((F/'review-manifest.json').read_text())
 for name,digest in manifest['input_sha256'].items():assert sha(P/name)==digest,name
 for name,digest in manifest['output_sha256'].items():assert sha(F/name)==digest,name
 # Validate original witnesses and every previous stage, not just manifest files.
 for version in ('whole-proposal-r1','whole-proposal-r2'):
  old=json.loads((P/version/'review-manifest.json').read_text())
  for section in ('input_sha256','script_sha256'):
   for name,digest in old.get(section,{}).items():assert sha(P/name)==digest,name
  for name,digest in old['output_sha256'].items():assert sha(P/version/name)==digest,name
 rows=list(csv.reader((F/'whole-proposed.csv').open()))
 legacy=list(csv.reader((P/'legacy-baseline/20220913-kholosi.csv').open()))
 assert len(rows)==1135 and len({r[10] for r in rows})==1135
 assert [r[10] for r in rows[:198]]==[r[10] for r in legacy]
 corrections=json.loads((P/'graph-reconciliation-r3/historical-assignment-correction-proposal.json').read_text())['corrections']
 approved={c['entry_key']:c['proposed_parameter'] for c in corrections}
 assert len(approved)==5
 assert [r[1] for r in rows[:198]]==[approved.get(r[10],r[1]) for r in legacy]
 metadata=json.loads((A/'validation.json').read_text())
 assert metadata['frozen_manifest_sha256']==sha(F/'review-manifest.json')
 assert metadata['yaml_sha256']==sha(A/'whole-metadata-proposed.yaml')
 oldyaml=yaml.safe_load((F/'whole-metadata-proposed.yaml').read_text())
 oldyaml['sources']['kholosi']['reference']['etymology_provenance']='mixed'
 assert yaml.safe_load((A/'whole-metadata-proposed.yaml').read_text())==oldyaml
 assert all(c['rows']==1135 and c['parser_errors']==0 for c in metadata['actual_parser_checks'].values())
 routed=yaml.safe_load((I/'whole-metadata-proposed.yaml').read_text())
 importer=routed['defaults'].pop('importer')
 assert routed==oldyaml
 assert importer['commands']==[['data/other/forms/raw_data/kholosi_2021/install_reviewed_source.py','--install']]
 assert set(importer)=={'commands','note'} and importer['note']
 correction=json.loads((G/'review.json').read_text())
 assert correction['old_csv_sha256']==sha(F/'whole-proposed.csv')
 assert correction['new_csv_sha256']==sha(G/'whole-proposed.csv')
 corrected=list(csv.reader((G/'whole-proposed.csv').open()))
 assert len(corrected)==len(rows)
 assert sum(a!=b for a,b in zip(rows,corrected))==6
 for a,b in zip(rows,corrected):
  assert a[:13]+a[14:]==b[:13]+b[14:]
  assert b[13]=='|'.join(a[13].split())
 keys={r[10] for r in corrected}
 assert all(k in keys for r in corrected for k in r[13].split('|') if k)
 return manifest

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--regenerate',action='store_true',help='Rerun only source-local graph generators, then reverify every freeze hash.')
 parser.add_argument('--install',action='store_true',help='Requires separately recorded accepted independent audit and root approval.')
 args=parser.parse_args()
 manifest=verify()
 if args.regenerate or args.install:
  for name in ('prepare_graph_claim_review_r2.py','review_numeric_claims_r2.py','prepare_local_components_r2.py',
               'review_local_components_r2.py','complete_claim_census_r2.py','prepare_graph_output_draft_r2.py',
               'prepare_historical_assignment_review_r3.py','prepare_donor_classification_r3.py','prepare_output_revision_r3.py'):
   subprocess.run([sys.executable,str(P/name)],check=True)
  subprocess.run([sys.executable,str(P/'prepare_parent_delimiter_addendum.py')],check=True)
  verify()
  assert sha(P/'graph-reconciliation-r3/whole-proposed.csv')==manifest['output_sha256']['whole-proposed.csv']
 destinations={
  D/'data/other/forms/20220913-kholosi.csv':G/'whole-proposed.csv',
  D/'data/other/forms/20220913-kholosi.yaml':I/'whole-metadata-proposed.yaml',
  D/'conversion/kholosi-source.txt':F/'whole-profile-proposed.txt',
 }
 for dst,src in destinations.items():
  if dst.exists():
   permitted={sha(src)}
   baseline=P/'legacy-baseline'/dst.name
   if baseline.exists():permitted.add(sha(baseline))
   if dst.suffix=='.csv':permitted.add(sha(F/'whole-proposed.csv'))
   if dst.suffix=='.yaml':permitted.add(sha(A/'whole-metadata-proposed.yaml'))
   assert sha(dst) in permitted,('Unexpected canonical edit; refuse overwrite',str(dst))
 if args.install:
  approval=json.loads((P/'source-stage-installation-approval.json').read_text())
  assert approval['status']=='accepted'
  assert approval['manifest_sha256']==sha(F/'review-manifest.json')
  audit=P/approval['independent_report']
  assert sha(audit)==approval['independent_report_sha256']
  accepted=json.loads(audit.read_text())
  errors=[accepted[k] for k in ('material_errors','failed_units','material_error_classes') if k in accepted]
  assert errors and all(n==0 for n in errors),'Independent report must explicitly record zero failures.'
  assert accepted.get('manifest_sha256',accepted.get('frozen_manifest_sha256'))==sha(F/'review-manifest.json')
  assert approval['approved_output_sha256']==manifest['output_sha256']
  assert approval['metadata_addendum_yaml_sha256']==sha(A/'whole-metadata-proposed.yaml')
  routing=json.loads((P/'source-stage-importer-routing-approval.json').read_text())
  assert routing['status']=='accepted' and routing['yaml_sha256']==sha(I/'whole-metadata-proposed.yaml')
  graph_approval=json.loads((G/'root-approval.json').read_text())
  assert graph_approval['status']=='accepted'
  assert graph_approval['approved_csv_sha256']==sha(G/'whole-proposed.csv')
  assert graph_approval['installation_approved'] is True
  assert set(graph_approval['changed_keys'])=={c['entry_key'] for c in json.loads((G/'review.json').read_text())['changes']}
  for dst,src in destinations.items():
   if dst.exists() and sha(dst)==sha(src):continue
   temporary=dst.with_name(dst.name+'.kholosi-install.tmp')
   assert not temporary.exists(),temporary
   temporary.write_bytes(src.read_bytes());os.replace(temporary,dst)
  result=dict(status='source-stage files installed; final metadata/docs/parser reconciliation required',
              rows=1135,legacy_keys=198,manifest_sha256=sha(F/'review-manifest.json'),
              installed_sha256={str(p.relative_to(D)):sha(p) for p in destinations},
              deferred=['Bibliography/checklist/ledger reconciliation','Actual canonical parser and focused tests',
                        'Full CLDF/database build','Compiled identity/graph','Full suite','Browser QA'])
  (P/'source-stage-installation-result.json').write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps(result))
 else:print(json.dumps(dict(status='preflight passed; no canonical writes',manifest_sha256=sha(F/'review-manifest.json'),regeneration=args.regenerate)))

if __name__=='__main__':main()
