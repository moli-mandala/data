"""Combine lexical and prose previews with individually reviewed relationships."""
import argparse
import csv
import json
from pathlib import Path

import preview_import
import prose_preview

HERE = Path(__file__).resolve().parent


def review_morphology(audits, decisions):
    pending = {(a['physical_entry_key'], p['annotation_index']): p
               for a in audits for p in a['pending_morphology']}
    reviewed = {}
    for decision in decisions:
        key = (decision['entry_key'], decision['annotation_index'])
        if key in reviewed or key not in pending:
            raise ValueError('Duplicate or missing morphology annotation')
        original = pending[key]
        if any(decision[field] != original[field] for field in ('annotation', 'candidate_bases')):
            raise ValueError('Morphology evidence changed')
        if decision['disposition'] != 'retain-unexpanded' or not decision['evidence']:
            raise ValueError('Unsupported morphology disposition')
        reviewed[key] = decision
    if reviewed.keys() != pending.keys():
        raise ValueError('Unreviewed morphology annotation')
    for audit in audits:
        audit['morphology_dispositions'] = [reviewed[(audit['physical_entry_key'], p['annotation_index'])]
                                           for p in audit['pending_morphology']]
    return len(reviewed)


def apply_relations(rows, relations, evidence):
    by_key = {r[10]: r for r in rows}
    if len(by_key) != len(rows):
        raise ValueError('Duplicate lexical entry key')
    seen = set()
    for relation in relations:
        child_key, parent_key = relation['child_key'], relation['parent_key']
        if child_key in seen or child_key == parent_key:
            raise ValueError('Duplicate or self-directed relation')
        seen.add(child_key)
        if child_key not in by_key or parent_key not in by_key:
            raise ValueError('Missing relation endpoint')
        child, parent = by_key[child_key], by_key[parent_key]
        if relation['kind'] not in {'variant', 'derived'} or child[11] or child[12] or child[13]:
            raise ValueError('Unsupported or conflicting reviewed relation')
        if (child[2], parent[2], child[3], parent[3]) != (
            relation['expected_child_form'], relation['expected_parent_form'],
            relation['expected_gloss'], relation.get('expected_parent_gloss', relation['expected_gloss'])):
            raise ValueError('Relation endpoint evidence changed')
        if evidence[(child_key, relation['evidence_position'])] != relation['evidence_text']:
            raise ValueError('Relation source evidence changed')
        child[11 if relation['kind'] == 'variant' else 13] = parent_key
        extra_tags = relation.get('add_tags', [])
        if not set(extra_tags) <= {'pl'}:
            raise ValueError('Unreviewed relation grammar tag')
        child[14] = ' '.join(dict.fromkeys(child[14].split() + extra_tags))
    for key in by_key:
        visited = set()
        while key and key in by_key:
            if key in visited:
                raise ValueError('Variant cycle')
            visited.add(key)
            key = by_key[key][11]
    return rows


def build(output):
    # Keep the old lexical-only sample reproducible; write to a separate directory.
    if output.resolve() == (HERE / 'preview').resolve():
        raise ValueError('Use a separate integration-preview directory')
    summary = preview_import.build(output)
    physical_path = output / 'physical-record-audit.jsonl'
    physical = [json.loads(line) for line in physical_path.read_text().splitlines()]
    dispositions = json.loads((HERE / 'morphology-dispositions.json').read_text())['decisions']
    summary['reviewed_unexpanded_suffix_scopes'] = review_morphology(physical, dispositions)
    physical_path.write_text(''.join(json.dumps(a, ensure_ascii=False, sort_keys=True)+'\n' for a in physical))
    path = output / '20260921-bhattacharya-ollari.csv'
    with path.open(newline='') as stream:
        rows = list(csv.reader(stream))
    relations = json.loads((HERE / 'reviewed-relations.json').read_text())['relations']
    evidence = {(r['entry_key'], p['position']): p['text']
                for file in sorted(HERE.glob('comparison-reviewed-p*.json'))
                for r in json.loads(file.read_text())['records'] for p in r['passages']}
    apply_relations(rows, relations, evidence)
    with path.open('w', newline='') as stream:
        csv.writer(stream).writerows(rows)
    audit_path = output / 'rich-row-audit.jsonl'
    audits = [json.loads(line) for line in audit_path.read_text().splitlines()]
    decisions = {r['child_key']: r for r in relations}
    for row, audit in zip(rows, audits):
        assert row[10] == audit['entry_key']
        audit['proposed_row'] = row
        if row[10] in decisions:
            audit['reviewed_relationship'] = decisions[row[10]]
    audit_path.write_text(''.join(json.dumps(a, ensure_ascii=False, sort_keys=True)+'\n' for a in audits))
    summary['variants'] = sum(bool(r[11]) for r in rows)
    summary['derivations'] = sum(bool(r[13]) for r in rows)
    summary['reviewed_cross_entry_variants'] = sum(r['kind'] == 'variant' for r in relations)
    summary['reviewed_cross_entry_derivations'] = sum(r['kind'] == 'derived' for r in relations)
    (output / 'rich-preview-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    summary['prose'] = prose_preview.build(output)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output), ensure_ascii=False, indent=2))
