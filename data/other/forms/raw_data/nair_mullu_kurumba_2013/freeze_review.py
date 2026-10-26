"""Pin the complete review proposal after lightweight checks; no acceptance."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'data/SOURCE_INGESTION_CHECKLIST.md').exists())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
files = ['README.md', 'physical-units-first-reading.jsonl', 'first-reading-status.json', 'prepare_first_reading.py',
    'prepare_candidate.py', 'proposed.csv', 'proposal-audit.jsonl', 'candidate-status.json', 'metadata-proposal.json',
    'graph-proposal.json', 'proposed-settings.yaml', 'proposed-references.bib', 'prepare_metadata_review.py',
    'metadata-profile-review.json', 'validate_actual_parser.py', 'actual-parser-review.json',
    'record_typography_review.py', 'bound-typography-review.json', 'focused-tests.log',
    'scope/source-position-inventory.jsonl', 'scope/detail-font-evidence.json', 'scope/page-native-font-census.json']
files += [f'scope/originals/page-{p}-150dpi.png' for p in (55, 56, 57, 58)]
files += ['apply_reviewed_revision.py', 'review-repair-continuity.json', 'prepare_slash_adjudication.py',
    'slash-alternative-adjudication-proposal.json', 'independent-review/complete-original-review-pass1.json',
    'check_reference_format.py', 'reference-format-review.json']
report = json.loads((HERE/'actual-parser-review.json').read_text())
assert report['inputs']['proposed.csv'] == sha(HERE/'proposed.csv')
assert report['inputs']['proposal-audit.jsonl'] == sha(HERE/'proposal-audit.jsonl')
assert '5 passed' in (HERE/'focused-tests.log').read_text()
manifest = dict(status='Frozen whole-source review proposal; not accepted or installed',
    rows=135, target_rows=110, explicit_comparison_rows=25, audit_groups=65, scoped_witness_records=164,
    exact_commentary_reuses=6, explicit_excluded_or_documentary_witness_records=23, proposed_variant_edges=11,
    files={str((HERE/n).relative_to(ROOT)):sha(HERE/n) for n in files},
    source_pdf_sha256='4d2588987dab5d15c62ecc95330ccf7e925a9bce80d08068b91ca3f48c8cc5af',
    test_file_sha256=sha(ROOT/'data/tests/test_nair_mullu_kurumba_proposal.py'),
    acceptance_pending=['independent original reading/output sample', 'root acceptance', 'guarded source-stage installation'],
    deferred=['full build/compiled survival checks', 'full suite', 'browser QA'], canonical_changes=False)
(HERE/'review-freeze.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'manifest_sha256':sha(HERE/'review-freeze.json'), 'csv_sha256':sha(HERE/'proposed.csv'),
    'audit_sha256':sha(HERE/'proposal-audit.jsonl'), 'canonical_changes':False}))
