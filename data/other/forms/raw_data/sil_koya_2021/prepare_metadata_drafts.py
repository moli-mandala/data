"""Stage source metadata and old-wordlist note cleanup for review only."""
import csv
import json
from pathlib import Path

from import_koya import HERE, INSTALLED, SOURCE_KEY

OUT = HERE / 'whole-source-recovery'
with INSTALLED.open(newline='') as handle:
    original = list(csv.reader(handle))
cleaned = []
note_audit = []
for old in original:
    row = old.copy()
    note_audit.append({'entry_key': row[10], 'previous_notes': row[6],
                       'disposition': 'retained in audit; previously hidden by source audit_only policy'})
    row[6] = ''
    assert all(a == b for i, (a, b) in enumerate(zip(old, row)) if i != 6)
    cleaned.append(row)
with (OUT / 'proposed-wordlist.csv').open('w', newline='') as handle:
    csv.writer(handle).writerows(cleaned)
(OUT / 'wordlist-note-relocation.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in note_audit))

(OUT / 'proposed-wordlist.yaml').write_text('''file: 20260828-sil-koya.csv
defaults:
  transcription:
    profile: sil-koya
sources:
  devagnanavaram-et-al2021koya:
    forms:
      split_alternates: false
    reference:
      editor: OpenAI Codex
      ocr: true
      etymology_provenance: none
''')
for kind in ['narrative', 'comparative']:
    (OUT / f'proposed-{kind}.yaml').write_text(f'''file: 20260927-sil-koya-{kind}.csv
defaults:
  transcription:
    profile: sil-koya-{kind}
    preserve_hyphens: true
''')
report = {
    'status': 'draft only; canonical unchanged',
    'wordlist_rows': len(cleaned), 'preserved_keys': len({r[10] for r in cleaned}),
    'only_changed_wordlist_field': 'Notes; previously hidden provenance preserved verbatim in wordlist-note-relocation.jsonl',
    'rationale': 'Remove source-global audit_only policy so new residual linguistic notes remain visible, while keeping old extraction metadata out of public notes.',
    'transcription': 'Each of three files selects its own profile via defaults; one shared source declaration. No source-specific shared parser branch.',
    'pending': 'Focused parser/profile/notes checks; independent whole-output audit; bibliography coverage update; guarded installation.',
}
(OUT / 'metadata-draft-summary.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
