"""Validate a complete first-reading alignment, without source installation."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

P = Path(__file__).resolve().parent / 'whole-source-recovery'


def main():
    source = P / 'parable-aligned-first-reading.tsv'
    lines = list(csv.DictReader(source.open(), delimiter='\t'))
    counts = Counter(int(row['page']) for row in lines)
    assert counts == {101: 10, 102: 15, 103: 15, 104: 15}, counts
    units, keys = [], set()
    for row in lines:
        assert None not in row and all(value is not None for value in row.values())
        forms, glosses = row['forms'].split('|'), row['glosses'].split('|')
        assert len(forms) == len(glosses), (row['page'], row['line'])
        for position, (form, gloss) in enumerate(zip(forms, glosses), 1):
            assert form and gloss and '\ufffd' not in form
            key = f"bailey1920kagani:whole:p{row['page']}:parable:line{row['line']}:word{position}"
            assert key not in keys, key
            keys.add(key)
            units.append(dict(source_unit_key=key, printed_page=int(row['page']),
                pdf_page=int(row['page'])+26, line=int(row['line']), token=position,
                raw_form=form, raw_alignment=gloss, line_comment=row['comment'],
                status='draft_pending_independent_original_review'))
    (P / 'parable-aligned-first-reading.jsonl').write_text(''.join(json.dumps(unit, ensure_ascii=False)+'\n' for unit in units))
    report = dict(status='complete printed narrative represented in first-reading draft; not installed',
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        printed_pages=[101, 102, 103, 104], source_lines=len(lines),
        primary_aligned_tokens=len(units), page_lines=dict(counts),
        checks='Line and token cardinality, nonempty aligned cells and unique page/line/token keys only.',
        deferred=['Independent glyph/alignment review and source-wide corrections',
                  'Separate parenthetical alternatives currently preserved in line comments',
                  'Full-expression boundary reconciliation across pages',
                  'Grammatical tags, references, profiles, complete-source integration and fresh output audit',
                  'Full compiled and browser gates'])
    (P / 'parable-aligned-first-reading-summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
