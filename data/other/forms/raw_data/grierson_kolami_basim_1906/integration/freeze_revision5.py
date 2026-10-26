"""Freeze current candidate while retaining root's exact pending twenty-unit sample."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
OUT=HERE/'revision5'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    historical=HERE/'revision4/review-freeze.json';old=json.loads(historical.read_text())
    for name,pin in old['files'].items():assert sha(PACKAGE/name)==pin,('historical r4 drift',name)
    parser=json.loads((OUT/'actual-parser-review.json').read_text());assert parser['candidate_sha256']==sha(OUT/'combined-review-candidate.csv')
    current=json.loads((OUT/'fresh-output-selection-20260927.json').read_text());previous=json.loads((HERE/'revision4/fresh-output-selection-20260927.json').read_text())
    assert [r['source_unit_key'] for r in current['sample']]==[r['source_unit_key'] for r in previous['sample']]
    assert current['candidate_sha256']==sha(OUT/'combined-review-candidate.csv') and current['audit_sha256']==sha(OUT/'combined-review-audit.jsonl')
    pins={str(p.relative_to(PACKAGE)):sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='review-freeze.json'}
    paths=[historical,PACKAGE/'root-independent-review/grammar-object-scope-root-adjudication.json',
      HERE/'apply_whole_pronoun_consistency.py',HERE/'test_pronoun_consistency.py',Path(__file__)]
    grammar=PACKAGE/'grammar-review/whole-metadata-consistency-revision'
    paths.extend(grammar/n for n in ['reviewed-child-witnesses.jsonl','grammar-tag-proposal.jsonl','grammar-graph-proposal.jsonl'])
    pins.update({str(p.relative_to(PACKAGE)):sha(p) for p in paths})
    manifest={'status':'Frozen r5; source-local exact9Tags repair, same20 original sample pending root acceptance','files':pins,
      'historical_revision4_freeze_sha256':sha(historical),'counts':old['counts'],'canonical_changes':False,
      'deferred':['Root final twenty-row original-output acceptance','Exact source-stage installer root approval/canonical checks','DB/CLDF/full suite/browser/publication'],
      'reuse_policy':old['reuse_policy']}
    (OUT/'review-freeze.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'csv':sha(OUT/'combined-review-candidate.csv'),'audit':sha(OUT/'combined-review-audit.jsonl'),'freeze':sha(OUT/'review-freeze.json'),'sample':sha(OUT/'fresh-output-selection-20260927.json'),'same_source_units':20}))
if __name__=='__main__':main()
