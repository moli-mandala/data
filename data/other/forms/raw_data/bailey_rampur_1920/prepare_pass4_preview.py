"""Apply the independently confirmed sentence fix without changing failed freezes."""
import copy
import csv
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    frozen = json.loads((P/'whole-paired-full-reread-freeze-20260927.json').read_text())
    review_path = P/'root-independent-whole-output-audit-20260927-pass3.json'
    review = json.loads(review_path.read_text())
    assert review['material_errors'] == 1
    fix, = review['corrections']
    for name, hashes in frozen['packages'].items():
        package = P.parent/f'bailey_{name}_1920'
        for filename, digest in hashes.items():
            assert sha(package/filename) == digest, filename
        with (package/'full-reread-preview.csv').open(newline='') as stream:
            old = list(csv.reader(stream))
        rows = copy.deepcopy(old)
        changed = []
        for row in rows:
            if row[10] == fix['entry_key']:
                assert row[2] == fix['old_form']
                row[2] = fix['form']
                changed.append(row[10])
        assert len(changed) == (1 if name == 'baghi' else 0)
        audit = [json.loads(line) for line in (package/'full-reread-preview-audit.jsonl').read_text().splitlines()]
        for unit in audit:
            if fix['entry_key'] not in unit.get('exported_entry_keys', []):
                continue
            unit['prior_pass3_source_review'] = copy.deepcopy(unit['source_review'])
            unit['source_review']['form'] = fix['form']
            unit['source_review']['review_note'] = unit['source_review']['review_note'].replace('ōrah', 'ōruh')
            unit['source_review']['review_corrections'].append(fix)
            for item in unit['candidates'] + unit['exported_rows']:
                assert item['form'] == fix['old_form']
                item['form'] = fix['form']
                if 'raw_cell' in item:
                    item['raw_cell'] = fix['form']
            unit['pass4_correction'] = dict(evidence=fix, review_sha256=sha(review_path))
        target = package/'pass4-preview.csv'
        with target.open('w', newline='') as stream:
            csv.writer(stream).writerows(rows)
        audit_target = package/'pass4-preview-audit.jsonl'
        audit_target.write_text(''.join(json.dumps(unit, ensure_ascii=False)+'\n' for unit in audit))
        report = dict(status='corrected review proposal; not installed', rows=len(rows),
                      changed_keys=changed, ordered_keys_unchanged=True,
                      baseline_sha256=sha(package/'full-reread-preview.csv'),
                      preview_sha256=sha(target), audit_sha256=sha(audit_target),
                      review_sha256=sha(review_path), canonical_changes=False,
                      related_occurrences=[dict(entry_key=row[10], form=row[2]) for row in rows if 'ōr' in row[2]],
                      next_gate='Recheck related source occurrences and fresh independent frozen audit')
        (package/'pass4-preview-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        print(name, len(rows), changed)
    validator = (P/'validate_full_reread_preview.py').read_text()
    validator = validator.replace('full-reread', 'pass4')
    (P/'validate_pass4_preview.py').write_text(validator)


if __name__ == '__main__':
    main()
