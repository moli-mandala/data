"""Reconcile literal readings with evidence; no canonical export or DB build."""
import csv
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent
FILES = {
    'whole-grammar-first-reading-20260927.tsv': 'grammar',
    'whole-sentences-first-reading-20260927.tsv': 'sentences',
    'whole-context-controls-first-reading-20260927.tsv': 'context',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(row, kind):
    parts = [row['page'], row['group'], row['item']] if kind == 'grammar' else [row['page'], row['item'] if kind == 'sentences' else row['unit']]
    return ':'.join(parts)


def main():
    independent_name = 'independent-whole-second-reading-20260927.json'
    root_name = 'root-targeted-rereading-20260927.json'
    independent = json.loads((P / independent_name).read_text())
    targeted = json.loads((P / root_name).read_text())
    for name, sha in independent['input_sha256'].items():
        assert digest(P / name) == sha, name
    review_pages = {str(c['page']): c for c in independent['coverage']}
    audit = []
    output_hashes = {}
    seen_corrections = set()
    for name, kind in FILES.items():
        with (P / name).open() as f:
            reader = csv.DictReader(f, delimiter='\t')
            fields = reader.fieldnames
            rows = list(reader)
        by_key = {identity(row, kind): row for row in rows}
        assert len(by_key) == len(rows), name
        for page, coverage in review_pages.items():
            assert sum(r['page'] == page for r in rows) == coverage[kind], (name, page)
        for key, row in by_key.items():
            original = dict(row)
            evidence = []
            pending = []
            for index, fix in enumerate(independent['corrections']):
                if fix['file'] != name or identity(fix, kind) != key:
                    continue
                assert row[fix['field']] == fix['old'], (name, key, fix)
                row[fix['field']] = fix['new']
                evidence.append({'reviewer': independent['reviewer'], **fix})
                seen_corrections.add(index)
            if kind == 'grammar':
                for fix in targeted['corrections']:
                    if fix['unit'] != key:
                        continue
                    if row[fix['field']] == fix['old']:
                        row[fix['field']] = fix['new']
                        pending.append(fix)
                    elif row[fix['field']] != fix['new']:
                        raise ValueError(('Readers disagree', key, fix, row[fix['field']]))
            covered = row['page'] in review_pages
            status = 'independent_second_reading' if covered and not pending else 'independent_review_pending'
            row['review_status'] = status
            audit.append(dict(source_unit_key=f'bailey1920bilaspuri:{kind}:{key}', input_file=name, original=original,
                              reconciled=dict(row), independent_corrections=evidence,
                              root_corrections_pending_independent_confirmation=pending,
                              independent_page_coverage=review_pages.get(row['page']), status=status))
        target = P / f'reviewed-{kind}-draft-20260927.tsv'
        with target.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields, delimiter='\t', lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
        output_hashes[target.name] = digest(target)
    assert seen_corrections == set(range(len(independent['corrections'])))
    (P / 'reviewed-draft-audit-20260927.jsonl').write_text(''.join(json.dumps(a, ensure_ascii=False)+'\n' for a in audit))
    report = dict(status='review derivatives only; no canonical export', input_sha256={n:digest(P/n) for n in [*FILES, independent_name, root_name]},
                  output_sha256=output_hashes, groups=len(audit), independently_reviewed_groups=sum(a['status']=='independent_second_reading' for a in audit),
                  pending_groups=sum(a['status']=='independent_review_pending' for a in audit), applied_independent_field_corrections=len(seen_corrections),
                  limitation='Coverage is per source group, not emitted rows. Remaining original review, paradigm expansion, metadata, independent output audit and integration gates are pending.')
    (P / 'reviewed-draft-manifest-20260927.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
