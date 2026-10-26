"""Apply independently reviewed metadata to a separate, hash-bound draft ledger."""
import ast
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'whole-source-recovery'
REVIEW = OUT / 'independent-draft-review-p172-182-20260927.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    review = json.loads(REVIEW.read_text())
    rows, controls, inputs, applied, contexts = [], [], {}, [], []
    for page, decision in review['pages'].items():
        path = OUT / decision['input_file']
        assert digest(path) == decision['input_sha256'], f'Stale review: {path.name}'
        inputs[path.name] = digest(path)
        draft = json.loads(path.read_text())
        page_rows = copy.deepcopy(draft.get('source_units', draft.get('target_units', draft.get('cells'))))
        assert page_rows is not None
        by_key = {r['source_unit_key']: r for r in page_rows}
        assert set(by_key) == set(decision['reviewed_source_keys'])
        for correction in decision['metadata_corrections']:
            row = by_key[correction['source_unit_key']]
            assert all(row.get(k) == v for k, v in correction['before'].items()), correction
            assert set(correction['proposed']) <= {'gloss', 'tags'}, correction
            old_forms = copy.deepcopy(row.get('forms', row.get('form')))
            old_gloss = row.get('gloss', '')
            row.update(correction['proposed'])
            if old_gloss and not row.get('gloss'):
                row['source_analysis'] = (row.get('source_analysis', '') + ' Source grammatical description: ' + old_gloss + '.').strip()
            assert row.get('forms', row.get('form')) == old_forms
            applied.append(correction)
        for row in page_rows:
            if 'form' in row:
                row['forms'] = [row.pop('form')]
            row.setdefault('tags', [])
            row.setdefault('printed_page', int(page))
            row.setdefault('pdf_page', int(page) + 26)
            row['review'] = 'owner first reading and independent whole-page second reading; metadata reconciled; not installed'
        rows.extend(page_rows)
        controls.extend(dict(raw_control=c, review_input=path.name) for c in draft.get('controls', []))
        contexts.append(dict(printed_page=int(page), source_context=decision['source_context'],
                             context_retention=decision['context_retention']))
    assert len(rows) == 414 and len(applied) == 160
    assert len({r['source_unit_key'] for r in rows}) == len(rows)
    tree = ast.parse((HERE.parents[4] / 'tags.py').read_text())
    registered = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {'GENDER_TAGS', 'GRAMMATICAL_TAGS'} for t in node.targets):
            registered.update(ast.literal_eval(node.value))
    unknown = sorted({tag for row in rows for tag in row['tags']} - registered)
    assert not unknown, unknown
    payload = dict(status='reviewed grammar draft only; whole-source output not frozen or installed',
                   input_sha256=inputs, independent_review_sha256=digest(REVIEW),
                   source_units=rows, controls=controls, source_contexts=contexts,
                   metadata_corrections=applied, registered_tag_check='passed',
                   pending=['Shared introduction and cross-chapter scope reconciliation',
                            'Sentence-page independent review', 'Legacy glossary reconciliation',
                            'Whole proposal, profile, seeded output audit and installation gates'])
    target = OUT / 'reviewed-grammar-p172-182-20260927.json'
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    print(f'{len(rows)} source units; {len(applied)} metadata corrections; registered tags pass; SHA256 {digest(target)}')


if __name__ == '__main__':
    main()
