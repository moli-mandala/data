"""Apply full-glossary rereading to separate previews, preserving failed freezes."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REVIEW = HERE/'whole-legacy-full-rereading-20260927.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    review=json.loads(REVIEW.read_text())
    summary={}
    for name in ['rampur','baghi']:
        package=HERE.parent/f'bailey_{name}_1920'
        source=package/'whole-proposed.csv'
        assert sha(source)==review['frozen_csv_sha256'][name], 'Frozen baseline changed'
        with source.open(newline='') as stream: old=list(csv.reader(stream))
        rows=[r[:] for r in old]
        by_key={r[10]:r for r in rows}
        changes={}
        def mark(key, evidence):
            changes.setdefault(key,[]).append(evidence)
        for fix in review['corrections']:
            if fix['package']!=name: continue
            row=by_key[fix['entry_key']]
            assert row[2]==fix['old_form'],fix
            row[2]=fix['form']
            mark(fix['entry_key'],dict(kind='literal',evidence=fix))
        for fix in review['metadata_findings']:
            key=fix['entry_key']
            if key not in by_key: continue
            row=by_key[key]
            additions=['multiword-expression' if t=='multiword' else t for t in fix.get('add_tags',[])]
            row[14]=' '.join(dict.fromkeys(row[14].split()+additions))
            if key.endswith('p147:item:51:answer2'):
                assert row[3]=='with (instrumental)' and 'instr' in row[14].split(),row
                row[3]='with'
            elif not fix['source_claim'].startswith('Explicit printed expression boundaries'):
                row[6]=' '.join(filter(None,[row[6],fix['source_claim']]))
            mark(key,dict(kind='metadata',evidence=fix))
        for fix in review['uncertainties']:
            key=fix['entry_key']
            if key not in by_key: continue
            row=by_key[key]
            assert row[2]==fix['form'],fix
            row[14]=' '.join(dict.fromkeys(row[14].split()+['uncertain']))
            mark(key,dict(kind='typed-uncertainty',evidence=fix))
        assert [r[10] for r in rows]==[r[10] for r in old]
        audit_source=package/'whole-proposed-audit.jsonl'
        audit=[json.loads(line) for line in audit_source.read_text().splitlines()]
        accounted=set()
        for unit in audit:
            keys=unit.get('entry_keys',[])
            modified=[key for key in keys if key in changes]
            if modified:
                unit['full_rereading_changes']={key:changes[key] for key in modified}
                unit['full_rereading_proposed_rows']={key:by_key[key] for key in modified}
                accounted.update(modified)
        assert accounted==set(changes),set(changes)-accounted
        target=package/'full-reread-preview.csv'
        with target.open('w',newline='') as stream: csv.writer(stream).writerows(rows)
        (package/'full-reread-preview-audit.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in audit))
        deltas=[dict(entry_key=a[10],before=a,proposed=b,evidence=changes[a[10]]) for a,b in zip(old,rows) if a!=b]
        report=dict(status='full-reread preview; not frozen or installed',rows=len(rows),
                    changed_rows=len(deltas),ordered_keys_unchanged=True,changes=deltas,
                    baseline_sha256=sha(source),baseline_audit_sha256=sha(audit_source),
                    review_sha256=sha(REVIEW),preview_sha256=sha(target),
                    pending=['Profile coverage and actual parser checks','Final metadata reconciliation',
                             'Fresh frozen output audit','Canonical installation'],canonical_changes=False)
        (package/'full-reread-preview-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        assert sha(source)==review['frozen_csv_sha256'][name]
        summary[name]=dict(rows=len(rows),changed_rows=len(deltas),old_freeze_unchanged=True)
    print(json.dumps(summary))


if __name__=='__main__':
    main()
