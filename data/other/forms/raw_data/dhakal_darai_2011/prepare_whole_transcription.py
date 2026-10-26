"""Explicit review-only transcription routing; never edit canonical profiles."""
import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path

from whole_legacy_baseline import legacy_rows, legacy_profile

P = Path(__file__).resolve().parent


def build():
    old = {r[10]: r for r in legacy_rows()}
    with (P / 'chapters12-roman-preservation.csv').open(newline='') as stream:
        roman = {r[10] for r in csv.reader(stream)}
    output = []
    for line in (P / 'whole-expanded-metadata-review.jsonl').open():
        row = json.loads(line)
        key, form = row['source_unit_key'], row['form']
        if row['disposition'].startswith('audit-only'):
            continue
        notes = []
        route = 'author-transcription'
        phonemic = form
        if key in old:
            phonemic = old[key][5]
            notes.append('Existing reviewed Phonemic value preserved exactly.')
        elif key in roman:
            route, phonemic = 'literal-roman', ''
            notes.append('Conventional Roman spelling, not an asserted phonemic transcription.')
        elif row.get('phonemic_policy', '').startswith('blank;') or any(x in form for x in ['⟦', 'í', '\u0301', '\u033d', 'ạ', '‘']):
            route, phonemic = 'literal-unresolved', ''
            notes.append('Literal source marks or compressed notation retained in Original; phonemic interpretation withheld.')
        else:
            if '[' in phonemic or ']' in phonemic:
                phonemic = phonemic.replace('[', '').replace(']', '')
                notes.append('Source clause-constituent brackets are structural and retained only in Original.')
            if 'ø' in phonemic or 'Ø' in phonemic:
                phonemic = phonemic.replace('ø', '').replace('Ø', '')
                notes.append('Source zero or omitted-material symbol is structural, not a pronounced segment; Original retains it.')
            phonemic = re.sub(r'\s+', ' ', phonemic).strip()
        output.append({'source_unit_key': key, 'route': route, 'proposed_phonemic': phonemic,
                       'transcription_decisions': notes})
    assert set(old) <= {r['source_unit_key'] for r in output}
    return output


if __name__ == '__main__':
    rows = build()
    out = P / 'whole-transcription-proposal.jsonl'
    out.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    profile = legacy_profile()
    assert '\ng\tg\n' not in profile
    profile += 'g\tg\n'
    rules=list(csv.reader(profile.splitlines(),delimiter='\t'))
    known={r[0] for r in rules}
    for grapheme,output in rules[1:]:
        decomposed=unicodedata.normalize('NFD',grapheme)
        if decomposed not in known:
            profile += decomposed+'\t'+output+'\n'
            known.add(decomposed)
    (P / 'whole-profile-proposed.txt').write_text(profile)
    report = {'status': 'Review-only transcription proposal; profile and actual parser validation pending',
              'records': len(rows), 'metadata_sha256': hashlib.sha256((P / 'whole-expanded-metadata-review.jsonl').read_bytes()).hexdigest(),
              'routes': {route: sum(r['route'] == route for r in rows) for route in sorted({r['route'] for r in rows})},
              'transcription_sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
              'profile_sha256': hashlib.sha256((P / 'whole-profile-proposed.txt').read_bytes()).hexdigest(),
              'canonical_changed': False}
    (P / 'whole-transcription-proposal-summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
