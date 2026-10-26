"""Preserve the independently reviewed revision before source-local repairs."""
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEST = HERE / 'revision2-pre-independent-repairs'
assert hashlib.sha256((HERE / 'proposed.csv').read_bytes()).hexdigest() == '38574c8a5365c956852a13cfe75632cfb2d900f2546750f69184d58a2c84bee5'
assert hashlib.sha256((HERE / 'proposal-audit.jsonl').read_bytes()).hexdigest() == '99cebc1204a278c7cf75e4ccbb8cc8a2d9195a3b1d5ed25565d3753bdacebc15'
paths = ['proposed.csv', 'proposal-audit.jsonl', 'proposal-status.json', 'review-freeze.json',
         'prepare_candidate.py', 'prepare_metadata.py', 'validate_candidate.py',
         'graph-proposal.json', 'candidate-validation.json', 'metadata/proposed-references.bib',
         'metadata/proposed-settings.yaml', 'metadata/proposed-language.csv',
         'metadata/proposed-dialect.csv', 'metadata/identity-decision.json', 'metadata/formatter-review.json']
manifest = {}
for name in paths:
    src, dst = HERE / name, DEST / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        assert dst.read_bytes() == src.read_bytes(), name
    else:
        shutil.copyfile(src, dst)
    manifest[name] = hashlib.sha256(dst.read_bytes()).hexdigest()
(DEST / 'archive-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('Reviewed revision archived; no canonical writes.')
