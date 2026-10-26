"""Record the isolated installer review without invoking installation."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = next(p for p in HERE.parents if (p / 'data/tags.py').exists())
log = WORKSPACE / 'tmp/kui-specimens/installer-review-tests.log'
assert '18 passed' in log.read_text()
paths = [HERE.parent / 'import_source.py', HERE / 'test_installer_preflight.py', log]
report = {
    'status': 'pass: isolated dry-run guard and recovery review',
    'reviewer': 'rohru_finish, independent of installer author',
    'sha256': {str(p.relative_to(WORKSPACE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    'tests': 18,
    'installation_executed': False,
    'canonical_changes': False,
    'coverage': [
        'Approval/input and durable-identity drift rejected before writes.',
        'Unrelated bibliography bytes and unique dialect header preserved; conflicting IDs rejected.',
        'Missing terminal dialect newline repaired in staged output.',
        'All six partial-write states recover in read-only dry-run, including after bibliography append.',
        'Exact six-target receipt/journal coverage; corrupt staged bytes and unexpected canonical edits rejected.',
        'Complete receipt idempotent; interrupted temporary receipt does not mask validated journal.',
    ],
    'resolved_findings': [
        'Durable staged-assets and transaction journal added before canonical writes.',
        'Dialect append now inserts absent newline.',
        'Receipt requires exact approved target set.',
        'Final receipt uses temporary file and atomic replacement.',
    ],
    'limits': [
        'Tests simulate partial transaction states inside Jambu scratch; no actual --install run.',
        'Source/editorial acceptance and final approved input pins remain parent responsibilities.',
        'No full CLDF/database build, compiled graph, browser QA, or publication performed.',
    ],
    'remaining_material_findings': [],
}
(HERE / 'independent-installer-review.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'status': report['status'], 'tests': report['tests']}))
