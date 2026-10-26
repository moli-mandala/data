"""Reproduce and snapshot the review proposal; never installs canonical data."""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
outputs=['whole-proposed.csv','whole-proposed-audit.jsonl','whole-profile-proposed.txt','whole-metadata-proposed.yaml']
before={n:sha(P/n) for n in outputs}
scripts=['prepare_witnesses.py','resolve_source_relations.py','expand_explicit_alternates.py',
 'apply_structured_metadata.py','prepare_etymology_claims.py','prepare_legacy_identity_review.py',
 'prepare_whole_proposal.py','prepare_profile.py','validate_whole_proposal.py']
for name in scripts:
 subprocess.run([sys.executable,str(P/name)],check=True)
after={n:sha(P/n) for n in outputs}
assert before==after,{'before':before,'after':after}
dest=P/'whole-proposal-r1'
assert not dest.exists(),'Refuse to overwrite an existing freeze.'
dest.mkdir()
evidence=['whole-proposed-summary.json','whole-parser-validation.json','whole-profile-validation.json',
 'legacy-identity-review.json','source-etymology-claims.jsonl','source-etymology-claim-validation.json',
 'owner-second-reading-decisions.json','source-metadata-plan-20260927.json']
for name in outputs+evidence:
 shutil.copyfile(P/name,dest/name)
inputs=['pdf-first-reading.jsonl','structured-first-reading.jsonl','pdf-metadata-decisions-20260927.json',
 'structured-metadata-decisions-20260927.json','supplemental-annotation-occurrences.jsonl',
 'bound-morphology-review-20260927.json','owner-second-reading-decisions.json',
 'legacy-concordance-candidates.json','legacy-baseline/20220913-kholosi.csv',
 'legacy-baseline/20220913-kholosi.yaml','source-metadata-plan-20260927.json']
inputs += [str(p.relative_to(P)) for p in sorted((P/'upstream').iterdir()) if p.is_file()]
manifest=dict(status='frozen for independent source/output audit; not approved for installation',
 source_commit='18b43f982efad92f35e4054561f2e0fc519f4eb7',
 output_sha256={n:sha(dest/n) for n in outputs+evidence},
 input_sha256={n:sha(P/n) for n in inputs},script_sha256={n:sha(P/n) for n in scripts},
 reproducible_byte_identical=True,canonical_writes=False,
 audit_request='Fresh >=20 stratified source units, actual emitted rows, all source layers including PDF lexical/senses/examples, CSV paradigms/sentences/alternates, historical identity and source-claim conflicts, bound morphology, unglossed text and controls.',
 deferred=['Independent audit acceptance','Guarded installation','Full CLDF/DB build','Full suite','Compiled identity/graph','Browser QA'])
(dest/'review-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(P/'whole-regeneration-validation.json').write_text(json.dumps(dict(status='passed',before=before,after=after,scripts=scripts),indent=2)+'\n')
print('FROZEN',dest,after)
