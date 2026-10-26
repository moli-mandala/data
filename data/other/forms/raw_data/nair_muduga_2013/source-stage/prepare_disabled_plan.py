"""Stage a disabled concrete plan after root lexical review. Never install or approve."""
from pathlib import Path
import csv,hashlib,json,shutil
HERE=Path(__file__).resolve().parents[1]
DATA=next(p for p in HERE.parents if (p/'source_meta.py').exists())
def sha(p):
 if not p.exists():return None
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 freeze=json.loads((HERE/'review-freeze.json').read_text())
 assert sha(HERE/'review-freeze.json')=='f60b146688ce20e0d922f23caedefa320b733336a8779c294cff131af01433fd'
 assert all(sha(HERE/n)==h for n,h in freeze['files_sha256'].items())
 report_name='root-original-review-final-20260927502.json'
 assert sha(HERE/report_name)=='7f2b81b86d094a533b3abb6ee3ceafc7bff185f887febd2bb3ec5507ea365e80'
 assert not (HERE/'installation-receipt.json').exists() and not (HERE/'installation-transaction.json').exists()
 s=HERE/'source-stage';archive=s/'disabled-scaffold-v1';archive.mkdir(exist_ok=True)
 for source in (HERE/'source-stage-authorization.json',s/'accepted-audit.jsonl'):
  target=archive/source.name
  assert not target.exists();shutil.copy2(source,target)
 raw=[json.loads(x) for x in (HERE/'proposal-audit.jsonl').read_text().splitlines()]
 for u in raw:
  u['accepted_for_source_stage']=True
  u['acceptance_evidence']={'review_report':report_name,'review_report_sha256':sha(HERE/report_name),'source_stage_only':True,'full_compiled_gates':'deferred'}
 (s/'accepted-audit.jsonl').write_text(''.join(json.dumps(u,ensure_ascii=False)+'\n' for u in raw))
 for name in ('test_nair_muduga_source.py','test_nair_muduga_installer.py'):
  shutil.copy2(DATA/'tests'/name,s/name)
 files=dict(freeze['files_sha256'])
 for n in ['review-freeze.json',report_name,'import_source.py','source-stage/accepted-audit.jsonl','source-stage/20260927-nair-muduga.yaml','source-stage/references.bib','source-stage/installer-tests.log','source-stage/test_nair_muduga_source.py','source-stage/test_nair_muduga_installer.py']:
  files[n]=sha(HERE/n)
 targets=[DATA/'data/other/forms/20260927-nair-muduga.csv',DATA/'data/other/forms/20260927-nair-muduga.yaml',HERE/'installed-audit.jsonl',DATA/'cldf/dialects.csv',DATA/'cldf/sources.bib']
 approval={'installation_approved':False,'result':'pass','status':'Root source review accepted; concrete transaction draft disabled for root installation-plan approval. No self-approval.','rows':204,'audit_units':128,'witness_slots':229,'files':files,'source_pdf_sha256':freeze['source_pdf_sha256'],'durable_identity_sha256':sha(DATA/'data/form-identities.csv'),'canonical_baselines':{str(p.relative_to(DATA)):sha(p) for p in targets},'root_lexical_review':{'path':report_name,'sha256':sha(HERE/report_name)},'deferred':['Full CLDF/database/full suite','Compiled graph/reference/identity checks','Browser/publication']}
 (HERE/'source-stage-authorization.json').write_text(json.dumps(approval,ensure_ascii=False,indent=2)+'\n')
 print('disabled draft',sha(HERE/'source-stage-authorization.json'))
 print('frozen revision unchanged',sha(HERE/'review-freeze.json'))
if __name__=='__main__':main()
