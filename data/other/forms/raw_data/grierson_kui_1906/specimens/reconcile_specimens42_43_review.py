"""Apply independently checked occurrence-specific corrections to a new derivative."""
import copy
import json
from reconcile_specimen40_review import HERE, read, sha, write


def main():
    inp = HERE / 'interlinear-specimen40-reviewed.jsonl'
    assert sha(inp) == '311d2af9aefde0f06c9717ef2a856b10a71e387103d4719435160f268243d4ec'
    rows = read(inp.name)
    before = copy.deepcopy(rows)
    keyed = {r['source_unit_key']:r for r in rows}
    reports = {}
    changes = []
    for specimen in (42,43):
        name = f'root-specimen{specimen}-independent-review-20260927.json'
        report = json.loads((HERE/name).read_text())
        reports[name] = sha(HERE/name)
        hashes = report.get('input_hashes') or {report['input']:report['input_sha256']}
        for path, expected in hashes.items():
            assert sha(HERE/path) == expected
        for fix in report['corrections']:
            changes.append(dict(fix, accepted_value=fix['proposed'], owner_resolution='Agreed after original image comparison; narrow crops resolve provisional questions.'))
    def add(page, specimen, line, word, old, new, evidence):
        changes.append(dict(source_unit_key=f'grierson1906kui:p{page}:specimen{specimen}:line{line}:word{word}',
                            field='raw_source_form', old=old, accepted_value=new, evidence=evidence))
    add(468,42,7,2,'dehānē','dēhānē','Alternate n489 crop fractions .07,.55,.33,.62: detached macron above first e is visible.')
    add(470,43,2,2,'‘āhō','‘āhē','Alternate n491 crop .05,.32,.80,.61: final e open right with internal stroke, distinct from closed o.')
    add(470,43,7,8,'mēttēñju','mētēñju','Same original crop: exactly one tall t stem between vowels.')
    add(470,43,5,9,'bēṭaṭi','bētaṭi','Same original crop: only final t carries an underdot; first t baseline unmarked.')
    report43 = json.loads((HERE/'root-specimen43-independent-review-20260927.json').read_text())
    for page in (470,471):
        for loc in report43['class_question'][f'page{page}_locations']:
            key = f'grierson1906kui:p{page}:specimen43:{loc}'
            old = keyed[key]['raw_source_form']
            assert old.startswith('Ē'), (key,old)
            changes.append(dict(source_unit_key=key,field='raw_source_form',old=old,accepted_value='E'+old[1:],
                                evidence=f'Each keyed occurrence separately inspected on alternate n{page+21}; capital E has no detached upper macron. Lowercase macrons retained.'))
    assert len({(d['source_unit_key'],d['field']) for d in changes}) == len(changes)
    for fix in changes:
        row = keyed[fix['source_unit_key']]
        assert row[fix['field']] == fix['old'], (fix,row[fix['field']])
        row[fix['field']] = fix['accepted_value']
    uncertain = keyed['grierson1906kui:p468:specimen42:line11:word8']
    assert uncertain['raw_source_form'] == 'ṛai'
    uncertain['typography_uncertainty'].append(dict(type='lower-r-mark-shape',
        reason='Alternate n489 crop .86,.72,.99,.78: lower mark is broad/fused with r baseline; dot versus short bar cannot be certified. Provisional ṛ retained literally, no phonemic claim.',
        alternatives=['ṟai']))
    for row in rows:
        if row['specimen'] in (42,43):
            row['review_status'] = 'independent full page review reconciled; source typography caveats retained'
    outputs = ['interlinear-specimens40_42_43-reviewed.jsonl']
    write(outputs[0],rows)
    for src,dest in [('physical-lines-first-reading.jsonl','physical-lines-specimens40_42_43-reviewed.jsonl'),
                     ('expression-spans-first-reading.jsonl','expression-spans-specimens40_42_43-reviewed.jsonl')]:
        groups = read(src)
        for g in groups:
            g['raw_source_form'] = ' '.join(keyed[k]['raw_source_form'] for k in g['atom_keys'])
            g['raw_interlinear_gloss'] = ' '.join(keyed[k]['raw_gloss'] for k in g['atom_keys'])
        write(dest,groups)
        outputs.append(dest)
    assert len(rows) == len(keyed) == 890
    assert [r['source_unit_key'] for r in rows] == [r['source_unit_key'] for r in before]
    assert all(a==b for a,b in zip(rows,before) if a['specimen'] not in (42,43))
    result = dict(status='specimens40/42/43 reconciled; specimen41 pending independent review',
                  input_sha256=sha(inp), independent_reports=reports, literal_field_corrections=len(changes),
                  corrections=changes, uncertainty_added=uncertain['source_unit_key'],
                  reviewed_atoms=562, pending_atoms=328, all_atom_keys=890,
                  outputs={n:sha(HERE/n) for n in outputs}, canonical_changes=False)
    (HERE/'specimens42_43-review-reconciliation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result['outputs'],indent=2))

if __name__ == '__main__':
    main()
