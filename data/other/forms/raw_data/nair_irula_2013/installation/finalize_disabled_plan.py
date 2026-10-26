"""Pin completed plan-validation evidence without enabling installation."""
import hashlib
import importlib.util
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
SHA=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
path=HERE/'root-acceptance-draft.json'
draft=json.loads(path.read_text())
assert draft['installation_approved'] is False
assert '8 passed' in (HERE/'installer-tests.log').read_text()
assert 'failed' not in (HERE/'installer-tests.log').read_text()
validation=json.loads((HERE/'metadata-audit-parser-validation.json').read_text())
assert validation['rows']==256 and validation['errors']==[]
assert validation['accepted_audit_sha256']==draft['assets']['audit']['output_sha256']
for name in ['installation/test_installer.py','installation/prepare_fixture_tests.py','installation/verify_plan.py','installation/metadata-audit-parser-validation.json','installation/installer-tests.log','installation/finalize_disabled_plan.py','independent-second-metadata-review/final-r2-scope-resolution.json']:
    draft['files'][name]=SHA(PACKAGE/name)
draft['validation_evidence']={'transaction_fixtures':'8 passed; clean install, repeat, unrelated shared/identity advance, baseline/input/identity drift, process interruption, exact rollback and disabled approval','actual_parser':'256 exact source-preserved fields, zero errors/converted','metadata_delta':validation['metadata_delta'],'accepted_audit':'261 exact source units; 16 audit-only decisions accepted without lexical emission; source_record and historical disposition retained'}
path.write_text(json.dumps(draft,ensure_ascii=False,indent=2)+'\n')
spec=importlib.util.spec_from_file_location('irula_final_disabled_plan',HERE/'install_source.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.main(check_draft=True)
print(json.dumps({'disabled_draft_sha256':SHA(path),'installer_sha256':SHA(HERE/'install_source.py'),'canonical_mutations':False},indent=2))
