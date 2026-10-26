"""Apply adjudicated chapter-four readings without rewriting first-reading evidence."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reconcile():
    owner = json.loads((HERE / 'chapter4-owner-original-review.json').read_text())
    nasal_path = HERE / 'chapter4-nasal-mark-review.json'
    nasal = json.loads(nasal_path.read_text())
    independent = json.loads((HERE / 'root-chapter4-nasal-review-20260926.json').read_text())
    assert independent['reviewed_report_sha256'] == sha(nasal_path)
    corrections = {r['source_unit_key']: r for r in owner['corrections'] + nasal['corrections']}
    outputs = []
    applied = []
    evidence_hashes = {}
    for kind in ['expressions', 'tables', 'prose']:
        path = HERE / f'chapter4-{kind}-first-reading.jsonl'
        assert sha(path) == owner['first_reading_hashes'][path.name]
        evidence_hashes[path.name] = sha(path)
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = row['source_unit_key']
            row['record_type'] = kind
            row['first_reading_file'] = path.name
            row['review_evidence'] = ['chapter4-owner-original-review.json']
            if key in corrections:
                repair = corrections.pop(key)
                assert row['form'] == repair['old_form'], key
                row['form_before_adjudication'] = row['form']
                row['form'] = repair['form']
                row['adjudication'] = repair.get('decision', repair.get('note'))
                row['review_evidence'].append('chapter4-nasal-mark-review.json' if 'input_file' in repair else 'chapter4-owner-original-review.json')
                if 'input_file' in repair:
                    row['review_evidence'].append('root-chapter4-nasal-review-20260926.json')
                applied.append(key)
            row['unresolved_glyphs'] = re.findall(r'⟦U\+[0-9A-F]+⟧', row['form'])
            row['review'] = 'Owner whole-page original second reading recorded; independent full chapter audit and interlinear token review remain pending.'
            outputs.append(row)
    assert not corrections, corrections
    keys = [r['source_unit_key'] for r in outputs]
    assert len(keys) == len(set(keys)) == 624
    return outputs, {'status': 'Adjudications applied to derivative evidence only; full chapter integration pending',
                     'source_counts': dict(Counter(r['record_type'] for r in outputs)),
                     'record_count': len(outputs), 'corrections_applied': applied,
                     'first_reading_sha256': evidence_hashes,
                     'canonical_changed': False,
                     'pending': ['715 geometric interlinear candidates need alignment review',
                                 'Scoped grammar and semantic metadata on full expressions',
                                 'Independent complete chapter audit', 'Whole thesis integration']}


if __name__ == '__main__':
    rows, report = reconcile()
    output = HERE / 'chapter4-adjudicated-records.jsonl'
    output.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    report['output_sha256'] = sha(output)
    (HERE / 'chapter4-adjudication-reconciliation-20260927.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'records': len(rows), 'corrected': len(report['corrections_applied']), 'sha256': report['output_sha256']}))
