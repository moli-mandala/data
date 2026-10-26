"""Record parent/root's explicit source-stage approval, preserving hash scope."""
import hashlib
import json
from pathlib import Path
P=Path(__file__).resolve().parent;F=P/'whole-proposal-r3'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((F/'review-manifest.json').read_text())
report=P/'independent-r3-original-review.json'
review=json.loads(report.read_text())
assert review['status']=='passed' and review['material_errors']==0
assert review['frozen_manifest_sha256']==sha(F/'review-manifest.json')
approval=dict(status='accepted',authorized_by='parent/root explicit source-stage instruction,2026-09-27',
 authorization='Approve source-stage installation CSV97ebfb4f… profile1fcdabf0… with mixed YAML106fc765…, all1135rows198keys and reviewed5assignment fixes.',
 manifest_sha256=sha(F/'review-manifest.json'),independent_report=report.name,
 independent_report_sha256=sha(report),approved_output_sha256=manifest['output_sha256'],
 metadata_addendum_yaml_sha256=sha(P/'source-stage-metadata-addendum/whole-metadata-proposed.yaml'),
 metadata_independent_review_sha256=sha(P/'independent-r3-metadata-addendum-review.json'),
 scope='Source CSV/profile/settings, targeted bibliography, docs/ledger and focused canonical validation only; no DB, full CLDF build, full suite, remote or publication.')
(P/'source-stage-installation-approval.json').write_text(json.dumps(approval,ensure_ascii=False,indent=2)+'\n')
print('Recorded exact approved source-stage artifacts.')
