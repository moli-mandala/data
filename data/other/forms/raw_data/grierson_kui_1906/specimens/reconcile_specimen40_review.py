"""Reconcile independent specimen 40 review without rewriting frozen readings."""
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(name):
    return [json.loads(line) for line in (HERE / name).read_text().splitlines()]

def write(name, rows):
    (HERE / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))

def main():
    original = HERE / 'interlinear-first-reading.jsonl'
    assert sha(original) == '18fe69f031259534f32172aa4355924c9b54ab9823298b82e6ac6304b6f5b8ca'
    report_path = HERE / 'root-specimen40-independent-review-20260927.json'
    report = json.loads(report_path.read_text())
    assert sha(HERE / report['input']) == report['input_sha256']
    rows = read(original.name)
    baseline = copy.deepcopy(rows)
    keyed = {r['source_unit_key']: r for r in rows}
    assert len(rows) == len(keyed) == 890
    # Coordinates are fractions of the pinned alternate-n485 original image.
    changes = [
        (9, 8, 'raw_source_form', 'vivānē', 'kogāñjū', [.65,.43,1,.58], 'Root full review: the-boy is kogāñjū, not the preceding throwing-from predicate.'),
        (9, 7, 'raw_source_form', 'vīvānē', 'vivānē', [.65,.43,1,.58], 'Owner narrow reread: initial i has a dot, not a macron. Earlier self-review applied this correction to the wrong adjacent atom.'),
        (7, 7, 'raw_source_form', 'ārā', 'ārū', [.65,.43,1,.58], 'Root original review: open-top final u with upper bar.'),
        (14, 1, 'raw_source_form', 'deri', 'dēri', [.05,.63,.36,.78], 'Owner narrow reread corroborates root: upper stroke on first e, as on adjacent repeated word.'),
        (12, 8, 'raw_source_form', 'āā-ātē;', 'ūā-ātē;', [.76,.61,.96,.69], 'Owner narrow reread resolves root question: first vowel is open-top u with macron; second a has a closed bowl.'),
        (7, 4, 'raw_gloss', '‘ever', 'ever', [.15,.39,.50,.48], 'No opening quote before printed ever.'),
        (6, 3, 'raw_gloss', 'what-ever', 'whatever', [.15,.39,.50,.48], 'Owner narrow reread resolves root question: printed whatever is unhyphenated.'),
    ]
    evidence = []
    for line, word, field, old, new, crop, reason in changes:
        key = f'grierson1906kui:p464:specimen40:line{line}:word{word}'
        row = keyed[key]
        assert row[field] == old, (key, row[field], old)
        row[field] = new
        evidence.append(dict(source_unit_key=key, field=field, old=old, accepted_value=new,
                             crop_fraction=crop, evidence=reason))
    restored = keyed['grierson1906kui:p464:specimen40:line9:word8']
    nasal = copy.deepcopy(next(r['typography_uncertainty'] for r in rows
                               if r['specimen'] == 40 and 'ñ' in r['raw_source_form']
                               and r['typography_uncertainty']))
    assert not restored['typography_uncertainty']
    restored['typography_uncertainty'] = nasal
    for row in rows:
        if row['specimen'] == 40:
            row['review_status'] = 'independent_full_page_review_reconciled; source typography caveats retained'
    write('interlinear-specimen40-reviewed.jsonl', rows)
    outputs = ['interlinear-specimen40-reviewed.jsonl']
    for src, dest in [
        ('physical-lines-first-reading.jsonl', 'physical-lines-specimen40-reviewed.jsonl'),
        ('expression-spans-first-reading.jsonl', 'expression-spans-specimen40-reviewed.jsonl'),
    ]:
        grouped = read(src)
        for group in grouped:
            group['raw_source_form'] = ' '.join(keyed[k]['raw_source_form'] for k in group['atom_keys'])
            group['raw_interlinear_gloss'] = ' '.join(keyed[k]['raw_gloss'] for k in group['atom_keys'])
        write(dest, grouped)
        outputs.append(dest)
    assert [r['source_unit_key'] for r in rows] == [r['source_unit_key'] for r in baseline]
    assert all(a == b for a,b in zip(rows,baseline) if a['specimen'] != 40)
    assert sum(r['specimen'] == 40 for r in rows) == 124
    result = dict(status='specimen40 reviewed derivative; other specimens remain pending independent reconciliation',
                  input_sha256=sha(original), independent_review_sha256=sha(report_path),
                  total_atoms=890, reviewed_atoms=124, changed_literal_fields=7,
                  corrections=evidence, additional_uncertainty_propagation=restored['source_unit_key'],
                  source_image_sha256=report['source_image_sha256'],
                  outputs={n:sha(HERE/n) for n in outputs},
                  preserved=['all 890 atom keys and ordering','original frozen inputs','77 expression boundaries','two explicit physical continuation joins'],
                  canonical_changes=False)
    (HERE / 'specimen40-review-reconciliation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result['outputs'], indent=2))

if __name__ == '__main__':
    main()
