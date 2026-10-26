"""Guarded source-stage installer, inactive without independent owner approval.

Default prepares the Notes addendum only. --install verifies a separate root
approval pinning every installation input, and never builds compiled data.
"""
import argparse,csv,hashlib,json,re,shutil,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
OUT=HERE/'metadata-addendum'
STEM='20260927-grierson-naiki'
SOURCE='grierson1906naiki'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def guard_canonical_target(target,approved_hash,baseline_hash=None):
 allowed={approved_hash}
 if baseline_hash:allowed.add(baseline_hash)
 assert not target.exists() or sha(target) in allowed,'Refusing to overwrite unexpected canonical bytes; review user edits first.'
def prepare():subprocess.run([sys.executable,str(HERE/'prepare_notes_addendum.py')],check=True)
def install():
 from pybtex.database import parse_file
 approval_path=OUT/'root-installation-approval.json'
 assert approval_path.exists(),'Independent owner review and source-stage installation approval pending.'
 approval=json.loads(approval_path.read_text());assert approval['result']=='pass' and approval['installation_approved'] is True
 prepare()
 required=['metadata-addendum/whole-review-proposal.csv','metadata-addendum/whole-review-proposal-audit.jsonl','metadata-addendum/whole-review-proposal.yaml','proposal-profile.txt','metadata-addendum/proposed-reference.bib','proposed-dialect.csv','metadata-addendum/canonical-baseline.json']
 assert set(required)<=approval['files'].keys()
 for n,h in approval['files'].items():assert sha(HERE/n)==h,n
 rows=list(csv.reader((OUT/'whole-review-proposal.csv').open()));assert len(rows)==694 and len({r[10] for r in rows})==694
 bib=ROOT/'cldf/sources.bib';old={k:v for k,v in parse_file(str(bib)).entries.items() if k!=SOURCE}
 proposed=parse_file(str(OUT/'proposed-reference.bib'));assert set(proposed.entries)=={SOURCE}
 blocks=re.split(r'(?=^@)',bib.read_text(),flags=re.M);indices=[i for i,b in enumerate(blocks) if re.match(r'@\w+\{'+SOURCE+r'\s*,',b)]
 assert len(indices)<=1
 replacement=(OUT/'proposed-reference.bib').read_text().rstrip()+'\n\n'
 if indices:blocks[indices[0]]=replacement
 else:blocks.append(replacement)
 dp=ROOT/'cldf/dialects.csv'
 with dp.open(newline='') as f:
  reader=csv.DictReader(f);fields=reader.fieldnames;existing={r['ID']:r for r in reader}
 additions=list(csv.DictReader((HERE/'proposed-dialect.csv').open()))
 for r in additions:assert set(r)==set(fields) and (r['ID'] not in existing or existing[r['ID']]==r)
 identity=ROOT/'data/form-identities.csv';before=sha(identity)
 target=ROOT/f'data/other/forms/{STEM}.csv'
 baseline=json.loads((OUT/'canonical-baseline.json').read_text())
 assert str(target.relative_to(ROOT))==baseline['path']
 guard_canonical_target(target,approval['files']['metadata-addendum/whole-review-proposal.csv'],baseline['sha256'])
 shutil.copyfile(OUT/'whole-review-proposal.csv',target)
 yaml=(OUT/'whole-review-proposal.yaml').read_text().replace('file: whole-review-proposal.csv',f'file: {STEM}.csv').replace('defaults:\n','defaults:\n  importer:\n    commands:\n      - [data/other/forms/raw_data/grierson_naiki_1906/import_source.py, --install]\n    note: Complete reviewed chapter and explicit additional witness; approved r2 with Notes-only addendum.\n',1)
 (ROOT/f'data/other/forms/{STEM}.yaml').write_text(yaml)
 profile=ROOT/'conversion/grierson-naiki-1906.txt';shutil.copyfile(HERE/'proposal-profile.txt',profile)
 shutil.copyfile(OUT/'whole-review-proposal-audit.jsonl',HERE/'installed-audit.jsonl')
 bib.write_text(''.join(blocks));assert {k:v for k,v in parse_file(str(bib)).entries.items() if k!=SOURCE}==old
 with dp.open('a',newline='') as f:csv.DictWriter(f,fieldnames=fields).writerows(r for r in additions if r['ID'] not in existing)
 assert sha(identity)==before
 report={'status':'Whole reviewed source stage installed; compiled/build/browser gates deferred.','rows':694,'audit_units':699,'canonical_sha256':sha(target),'profile_sha256':sha(profile),'audit_sha256':sha(HERE/'installed-audit.jsonl'),'durable_identity_sha256_unchanged':before,'unrelated_bibliography_entries_unchanged':len(old),'independent_review':'independent-r2-original-review.json','notes_approval':'metadata-addendum/root-installation-approval.json','graph_policy':'Explicit incomplete affix-only components retained as typed audit and source Notes; generic derived edges not fabricated.','deferred':['Full CLDF/database build','Compiled identity/graph/reference checks','Full test suite','Browser QA']}
 (HERE/'installation-report-20260927.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--install',action='store_true');args=parser.parse_args();install() if args.install else prepare()
