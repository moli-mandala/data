import hashlib
import json
from pathlib import Path
P=Path(__file__).resolve().parent
path=P/'source-stage-importer-routing-addendum/whole-metadata-proposed.yaml'
report=dict(status='accepted',authorized_by='parent/root explicit instruction,2026-09-27',
 authorization='Approved narrow importer.commands routing addition with guarded coordinator --install and provenance note; record separate metadata hash and prove source_meta registry + repeatinstallation byteparity.',
 yaml_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
 baseline_yaml_sha256=hashlib.sha256((P/'source-stage-metadata-addendum/whole-metadata-proposed.yaml').read_bytes()).hexdigest(),
 scope='Only importer command/note added; lexical rows, profile and all previous metadata unchanged.')
(P/'source-stage-importer-routing-approval.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
