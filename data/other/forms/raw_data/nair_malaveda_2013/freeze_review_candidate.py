"""Pin all source-local inputs needed by the independent reviewer."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
files = [p for p in HERE.iterdir() if p.is_file() and p.name != 'review-freeze.json']
files += [p for p in (HERE / 'metadata').iterdir() if p.is_file()]
manifest = {'status': 'complete source-local candidate frozen; independent audit and relation adjudication pending',
            'source_pdf_sha256': '4d2588987dab5d15c62ecc95330ccf7e925a9bce80d08068b91ca3f48c8cc5af',
            'scope': json.loads((HERE / 'proposal-status.json').read_text()),
            'files_sha256': {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)},
            'canonical_changes': False,
            'review_priorities': ['Fresh20 across all five original pages', 'Fourteen drawn lowlines and literal legacy lower-dot/tilde repairs',
                                  'Personal-termination cell scope, source II (Mas.) typo and blank feminine plural',
                                  'Twelve unapplied variant proposals and held morphological arrows', 'Six reported thesis records and source-mapping uncertainty'],
            'deferred_gates': ['independent final output audit and targeted repair reread', 'guarded source-stage installation',
                               'canonical focused checks', 'full CLDF pipeline', 'compiled graph/identity checks', 'full suite', 'browser refresh/QA']}
(HERE / 'review-freeze.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
print(hashlib.sha256((HERE / 'review-freeze.json').read_bytes()).hexdigest())
