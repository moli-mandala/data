"""Apply independently reviewed deltas to historical preview; never install."""
import copy
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECOVERY = HERE / 'whole-source-recovery'
STEM = 'whole-reviewed-preview-20260927'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    review_path = RECOVERY/'independent-sentences-context-review-20260927.json'
    review = json.loads(review_path.read_text())
    for name, digest in review['input_sha256'].items():
        assert sha(RECOVERY/name) == digest, name
    historical_report = json.loads((HERE/'whole-unfrozen-preview-report.json').read_text())
    for name, digest in historical_report['input_sha256'].items():
        assert sha(HERE/name) == digest, name
    assert sha(HERE/'whole-unfrozen-preview.csv') == historical_report['preview_sha256']
    assert sha(HERE/'whole-unfrozen-preview-audit.jsonl') == historical_report['audit_sha256']
    with (HERE/'whole-unfrozen-preview.csv').open(newline='') as stream:
        before = list(csv.reader(stream))
    rows = copy.deepcopy(before)
    by_key = {r[10]: r for r in rows}
    audits = [json.loads(line) for line in (HERE/'whole-unfrozen-preview-audit.jsonl').read_text().splitlines()]
    audit_by_key = {key: a for a in audits for key in a['entry_keys']}
    changes = []
    for change in review['corrections']:
        key = change['source_unit_key']
        assert change['field'] == 'forms' and len(change['before']) == len(change['after']) == 1
        assert by_key[key][2] == change['before'][0]
        by_key[key][2] = change['after'][0]
        a = audit_by_key[key]
        assert a['source_review']['forms'] == change['before']
        a['source_review']['forms'] = change['after']
        a.setdefault('independent_revision', []).append(change)
        changes.append(dict(entry_key=key, kind='literal', delta=change))
    for change in review['grammar_recommendations']:
        key = change['source_unit_key']
        tags = by_key[key][14].split()
        assert 'impv' not in tags
        by_key[key][14] = ' '.join(tags+change['add_tags'])
        a = audit_by_key[key]
        a['normalized_tags'] = by_key[key][14].split()
        a['source_review']['tags'] = list(dict.fromkeys(a['source_review']['tags']+change['add_tags']))
        a.setdefault('independent_revision', []).append(change)
        changes.append(dict(entry_key=key, kind='whole-command-metadata', delta=change))
    legacy = json.loads((RECOVERY/'legacy-metadata-review-20260927.json').read_text())
    assert len(legacy['changes']) == 43
    for change in legacy['changes']:
        key = change['entry_key']
        row = by_key[key]
        for field, index in [('gloss',3),('notes',6),('tags',14)]:
            assert row[index] == change['before'][field], (key,field,row[index],change['before'][field])
            row[index] = change['proposed'][field]
        audit_by_key[key].setdefault('independent_revision', []).append(change)
        changes.append(dict(entry_key=key, kind='legacy-metadata', delta=change))
    for control in legacy['audit_only']:
        a = next(a for a in audits if a['source_cell_key'] == control['source_cell_key'])
        a['resolved_source_cross_references'] = control
    assert len(rows) == 837 and len(audits) == 688
    assert [r[10] for r in rows] == [r[10] for r in before]
    for old,new in zip(before,rows):
        assert all(old[i] == new[i] for i in range(15) if i not in {2,3,6,14})
    assert all(old[2] == new[2] for old,new in zip(before[:330],rows[:330]))
    return rows,audits,changes,review_path


def main():
    rows,audits,changes,review_path = prepare()
    inputs = [HERE/'whole-unfrozen-preview.csv', HERE/'whole-unfrozen-preview-audit.jsonl',
              HERE/'whole-unfrozen-preview-report.json', HERE/'whole-preview-profile.txt',
              review_path, RECOVERY/'legacy-metadata-review-20260927.json', Path(__file__).resolve()]
    hashes = {str(p.relative_to(HERE)): sha(p) for p in inputs}
    with (HERE/f'{STEM}.csv').open('w',newline='') as stream:
        csv.writer(stream).writerows(rows)
    (HERE/f'{STEM}-audit.jsonl').write_text(''.join(json.dumps(a,ensure_ascii=False)+'\n' for a in audits))
    (HERE/f'{STEM}-profile.txt').write_bytes((HERE/'whole-preview-profile.txt').read_bytes())
    report = dict(status='reviewed revision; independent final-output audit pending; not installed',
                  rows=837,audit_units=688,legacy_keys_preserved=330,new_rows=507,
                  literal_corrections=2,command_metadata_corrections=6,legacy_metadata_corrections=43,
                  input_sha256=hashes,changes=changes,
                  output_sha256={f'{STEM}{suffix}':sha(HERE/f'{STEM}{suffix}') for suffix in ['.csv','-audit.jsonl','-profile.txt']},
                  pending=['Independent final-output audit','Guarded installer and canonical metadata integration','Focused installed validation','User-deferred full build and browser QA'])
    (HERE/f'{STEM}-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    assert hashes == {str(p.relative_to(HERE)):sha(p) for p in inputs}
    print(json.dumps({k:report[k] for k in ['status','rows','audit_units','output_sha256']}))


if __name__ == '__main__':
    main()
