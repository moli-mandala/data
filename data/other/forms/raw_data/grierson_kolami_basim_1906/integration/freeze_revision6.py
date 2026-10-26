"""Pin exact Notes-only continuity after fresh focused parser checks."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
OUT=HERE/'revision6'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    historical=HERE/'revision5/review-freeze.json';old=json.loads(historical.read_text())
    for name,pin in old['files'].items():assert sha(PACKAGE/name)==pin,name
    report=json.loads((OUT/'actual-parser-review.json').read_text())
    assert report['candidate_sha256']==sha(OUT/'combined-review-candidate.csv')
    current=json.loads((OUT/'fresh-output-selection-20260927.json').read_text())
    prior=json.loads((HERE/'revision5/fresh-output-selection-20260927.json').read_text())
    assert [r['source_unit_key'] for r in current['sample']]==[r['source_unit_key'] for r in prior['sample']]
    pins={str(p.relative_to(PACKAGE)):sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='review-freeze.json'}
    for p in [historical,HERE/'apply_scholarly_notes.py',HERE/'test_scholarly_notes.py',Path(__file__)]:pins[str(p.relative_to(PACKAGE))]=sha(p)
    manifest={'status':'Frozen r6; exact two scholarly Notes replacements, root acceptance addendum pending',
      'files':pins,'historical_revision5_freeze_sha256':sha(historical),'counts':old['counts'],
      'canonical_changes':False,'reuse_policy':old['reuse_policy'],
      'deferred':['Root exact Notes-only acceptance addendum','Exact root installer activation/canonical checks','DB/CLDF/full suite/browser/publication']}
    (OUT/'review-freeze.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'csv':sha(OUT/'combined-review-candidate.csv'),'audit':sha(OUT/'combined-review-audit.jsonl'),
      'freeze':sha(OUT/'review-freeze.json'),'sample':sha(OUT/'fresh-output-selection-20260927.json')}))
if __name__=='__main__':main()
