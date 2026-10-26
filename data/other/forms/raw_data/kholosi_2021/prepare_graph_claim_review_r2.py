"""Build a review queue only; never changes the frozen proposal or assignments."""
import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path

P = Path(__file__).resolve().parent
R = P / 'graph-reconciliation-r2'
R.mkdir(exist_ok=True)
claims = [json.loads(s) for s in (P/'source-etymology-claims.jsonl').open()]
targets = {r[0]: r[1] for r in csv.reader((P.parents[4]/'data/cdial/params.csv').open())}
audit = {u['source_unit_key']: u for u in map(json.loads, (P/'whole-proposal-r1/whole-proposed-audit.jsonl').open())}

def norm(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s.casefold()) if unicodedata.category(c)[0] == 'L')

records = []
for c in claims:
    text = c['literal_claim']
    cited = []
    extracted = list(c['cited_target_ids'])
    for ident in re.findall(r'CDIAL\s+(\d{1,5})', text, re.I):
        if ident not in {t['cdial_id'] for t in extracted}:
            assert ident in targets
            extracted.append(dict(cdial_id=ident, exists=True, target_head=targets[ident],
                                  extraction_addendum='Case-insensitive citation label; immutable r1 parser missed CDIAl.'))
    for target in extracted:
        ident = target['cdial_id']
        cited.append(dict(target, normalized_head_present=norm(targets[ident]) in norm(text)))
    records.append(dict(c, emitted_entry_keys=audit[c['source_unit_key']]['emitted_entry_keys'],
                        cited_target_ids=cited, review_status='pending explicit claim-scope review'))
(R/'claim-review-queue.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records))
(R/'numeric-head-screen.txt').write_text('\n'.join(
    f"{r['source_unit_key']} | {r['source_form']} | {r['literal_claim']} | " +
    '; '.join(f"{t['cdial_id']}={t['target_head']}" for t in r['cited_target_ids'])
    for r in records if any(not t['normalized_head_present'] for t in r['cited_target_ids']))+'\n')
(R/'review-inputs.json').write_text(json.dumps(dict(
    status='review queue, no proposed links approved', claims=len(records),
    input_sha256={str(p.relative_to(P)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                  (P/'source-etymology-claims.jsonl', P/'whole-proposal-r1/whole-proposed.csv',
                   P/'whole-proposal-r1/whole-proposed-audit.jsonl')},
    screen_method='Accent-insensitive literal head containment is triage only, never authority to change a cited number.'
), indent=2)+'\n')
print('Queued', len(records), 'attributed claims; frozen r1 unchanged.')
