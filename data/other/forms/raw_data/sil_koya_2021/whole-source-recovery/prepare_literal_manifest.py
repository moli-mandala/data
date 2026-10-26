"""Pin review-ready inputs and outputs without claiming installation approval."""
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent
OUT=HERE/'literal-integration'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=json.loads((OUT/'parser-validation.json').read_text())
    metadata=json.loads((OUT/'metadata-validation.json').read_text())
    assert parser['status']=='passed_focused_actual_parser_checks'
    assert metadata['status']=='passed_proposal_metadata_checks'
    for name,digest in metadata['sha256'].items():assert sha(OUT/name)==digest,name
    historical=json.loads((HERE/'whole-output-freeze-20260927.json').read_text())
    for name,digest in historical['sha256'].items():assert sha(HERE/name)==digest,('historical freeze changed',name)
    files=[p for p in OUT.iterdir() if p.is_file() and p.name!='review-manifest.json']
    inputs=['notation-keyed-proposal-20260927.jsonl','notation-literal-pages-20260927.json','notation-narrow-adjudications-20260927.json','notation-iota-encoding-review-20260927.json','notation-question-shaped-sign-census-20260927.json','notation-validation-20260927.json','report-census-first-reading-20260927.json','narrative-scope-review-20260927.json','original-page-review-20260927.json','whole-output-freeze-20260927.json','independent-whole-output-audit-20260927.json']
    scripts=['prepare_notation_review.py','validate_notation_review.py','prepare_literal_integration.py','validate_literal_integration.py','validate_literal_metadata.py','prepare_literal_manifest.py']
    report={'status':'ready_for_independent_review_not_installation_approved','rows':{'wordlist':1437,'narrative':210,'comparative':18,'total':1665},'audit_units':2119,'legacy_key_reconciliation':{'active':1437,'proposed_alias':1,'total_accounted':1438},'outputs_sha256':{p.name:sha(p) for p in sorted(files)},'inputs_sha256':{name:sha(HERE/name) for name in inputs},'scripts_sha256':{name:sha(HERE/name) for name in scripts},'canonical_modified':False,'historical_failed_freeze_unchanged':True,'required_next':'Root independently reviews actual emitted rows, source notation and proposed alias; installation remains unauthorized pending that review.'}
    (OUT/'review-manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'manifest_sha256':sha(OUT/'review-manifest.json'),'wordlist_sha256':sha(OUT/'proposed-wordlist.csv'),'whole_audit_sha256':sha(OUT/'proposed-whole-audit.jsonl'),'rows':report['rows']},indent=2))

if __name__=='__main__':main()
