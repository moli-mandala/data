"""Assemble lexical review candidates. Does not install or normalize forms."""
import json
from pathlib import Path
from grammar import parse as parse_grammar

HERE = Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE / name).read_text())


def build():
    context = {r['key']: r for r in read('context-dispositions.json')}
    loans = {r['key']: r for r in read('borrowing-dispositions.json')}
    candidates, excluded = [], []
    for source in read('inventory.json'):
        decision = context.get(source['key'], {})
        if decision.get('status', '').startswith('excluded'):
            excluded.append({'key': source['key'], **decision})
            continue
        form = source['form_candidate'] or decision.get('form', '')
        if not form:
            raise ValueError(f"Unresolved inventory disposition: {source['key']}")
        candidates.append({'key': source['key'], 'printed_page': source['printed_page'],
                           'form': form, 'source_gloss': source['gloss_candidate'],
                           'gloss': decision.get('gloss', source['gloss_candidate']),
                           'tags': decision.get('tags', []),
                           'status': 'pending-rich-import-audit'})
    for source in read('inventory-exceptions.json'):
        candidates.append({'key': source['key'], 'printed_page': source['printed_page'],
                           'form': source['form'], 'source_gloss': source['source_gloss'],
                           'gloss': source['source_gloss'], 'tags': [],
                           'status': 'pending-rich-import-audit'})
    index = {r['key']: r for r in candidates}
    assert len(index) == len(candidates)
    for correction in read('form-corrections.json'):
        row = index[correction['key']]
        assert row['form'] == correction['before']
        row['extracted_form'] = row['form']
        row['form'] = correction['after']
    for row in candidates:
        row['gloss'], grammar_tags = parse_grammar(row['gloss'], row['form'])
        row['tags'] = list(dict.fromkeys(row['tags'] + grammar_tags))
    for key, loan in loans.items():
        index[key]['tags'].append('loanword')
        index[key]['source_analysis'] = loan['evidence']
    for relation in read('reviewed-relations.json'):
        assert relation['child'] in index and relation['parent'] in index
        index[relation['child']]['variant_of_key'] = relation['parent']
    for uncertainty in read('uncertainties.json'):
        row = index[uncertainty['key']]
        row['tags'].append('uncertain')
        row['uncertainty'] = uncertainty
    return {'status': 'review proposal, not installed', 'candidates': candidates,
            'excluded': excluded,
            'pending': ['metadata installation and auxiliary reference accounting',
                        'rich CSV and fresh seeded acceptance audit',
                        'full build and compiled validation']}


if __name__ == '__main__':
    result = build()
    (HERE / 'proposal.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(f"{len(result['candidates'])} candidates, {len(result['excluded'])} explicit exclusions; not installed")
