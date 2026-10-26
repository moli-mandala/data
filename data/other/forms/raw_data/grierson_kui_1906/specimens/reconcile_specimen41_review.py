"""Apply the complete independent specimen41 review to a separate derivative."""
import json
from reconcile_specimen40_review import HERE, read, write, sha

def main():
    source = HERE/'interlinear-specimens40_42_43-reviewed.jsonl'
    assert sha(source) == '97230e4f43cfe65165be9102cdd686ce44d61b275989d736eee73fa369c999a4'
    report_path = HERE/'independent-specimen41-full-review-20260927.json'
    assert sha(report_path) == '9ed667f404c959146096ce0e54ca7400299cfa7814109bb71965e2f20ab9f281'
    report = json.loads(report_path.read_text())
    assert report['counts']['atoms'] == 328
    assert all(report['frozen_file_checks'].values())
    rows = read(source.name)
    keyed = {r['source_unit_key']:r for r in rows}
    for fix in report['findings']:
        row = keyed[fix['source_unit_key']]
        assert row[fix['field']] == fix['before']
        row[fix['field']] = fix['proposed_after']
    for row in rows:
        if row['specimen'] == 41:
            row['review_status'] = 'independent full page review reconciled; source typography caveats retained'
    assert len(rows) == len(keyed) == 890
    original_keys = [r['source_unit_key'] for r in read('interlinear-first-reading.jsonl')]
    assert [r['source_unit_key'] for r in rows] == original_keys
    outputs = ['interlinear-reviewed.jsonl']
    write(outputs[0],rows)
    for src,dest in [('physical-lines-first-reading.jsonl','physical-lines-reviewed.jsonl'),('expression-spans-first-reading.jsonl','expression-spans-reviewed.jsonl')]:
        groups = read(src)
        for g in groups:
            g['raw_source_form'] = ' '.join(keyed[k]['raw_source_form'] for k in g['atom_keys'])
            g['raw_interlinear_gloss'] = ' '.join(keyed[k]['raw_gloss'] for k in g['atom_keys'])
        write(dest,groups)
        outputs.append(dest)
    result = dict(status='all890 atoms independently reviewed and reconciled; final output/metadata and expression audit pending',
                  inputs={source.name:sha(source),report_path.name:sha(report_path)},
                  corrections=report['findings'], original_key_equality=True, original_key_count=len(original_keys),
                  outputs={p:sha(HERE/p) for p in outputs},canonical_changes=False)
    (HERE/'specimen41-review-reconciliation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result['outputs'],indent=2))

if __name__ == '__main__':
    main()
