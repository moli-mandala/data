"""Record root's completed visual comparison of the seeded r1 output sample."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent/'whole-proposal-r1'
sample=json.loads((R/'root-output-sample-20260927.json').read_text())
manifest=json.loads((R/'review-manifest.json').read_text())
assert hashlib.sha256((R/'review-manifest.json').read_bytes()).hexdigest()==sample['manifest_sha256']
for name,digest in manifest['output_sha256'].items():
    assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest
findings=json.loads((R/'root-output-review-findings-20260927.json').read_text())['findings']
failed={f['key']:f for f in findings}
reviews=[]
for unit in sample['sample']:
    keys=[r[10] for r in unit['actual_rows']]
    errors=[failed[k] for k in keys if k in failed]
    reviews.append({'printed_page':unit['printed_page'],'entry_keys':keys,
                    'status':'failed' if errors else 'passed', 'findings':errors,
                    'evidence':'Original source page image compared to frozen Form, Gloss, Notes, Tags and citation. Shared-heading and abbreviation scope checked where applicable.'})
report={'status':'failed','sample_size':20,'sample_rows':sample['rows'],
        'material_errors':sum(x['status']=='failed' for x in reviews),
        'passed_units':sum(x['status']=='passed' for x in reviews),
        'seed':sample['seed'],'frozen_manifest_sha256':sample['manifest_sha256'],
        'frozen_output_sha256':manifest['output_sha256'],'reviews':reviews,
        'editorial_cleanup':'Move implementation instructions such as apply multiword-expression per emitted answer from Notes into audit; preserve real source abbreviation/context notes.',
        'decision':'Do not install r1. Correct systematic metadata classes, add regression checks and run a fresh independent output sample on a new immutable revision.',
        'limits':'Source-stage audit only; no full CLDF or browser/database gates run.'}
assert report['material_errors']==3 and report['passed_units']==17
(R/'root-output-review-20260927.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(f"Completed {report['sample_size']} units/{report['sample_rows']} rows: 17 passed, 3 failed; installation not approved.")
