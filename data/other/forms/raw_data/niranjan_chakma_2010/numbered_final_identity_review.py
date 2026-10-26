"""Merge reviewed identity evidence without changing any earlier proposal."""
import copy
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

P = Path(__file__).resolve().parent
DATA = P.parents[4]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    files = ['numbered-language-mapping-proposal.json',
             'numbered-target-identity-research-addendum.json',
             'numbered-target-identity-research-addendum.md',
             'numbered-reference-mapping-20260927.json',
             'numbered-structured-review.jsonl']
    old = json.loads((P / files[0]).read_text())
    add = json.loads((P / files[1]).read_text())
    refs = json.loads((P / files[3]).read_text())
    groups = [json.loads(x) for x in (P / files[4]).read_text().splitlines()]
    for name, expected in old['inputs'].items():
        assert sha(P / name) == expected, name
    for name, expected in add['inputs'].items():
        assert sha(P / name) == expected, name
    assert sha(P / files[4]) == next(v for k, v in refs['input_sha256'].items() if k.endswith(files[4]))
    registry = {x['ID']: x for x in csv.DictReader((DATA / 'cldf/languages.csv').open())}
    assert sha(DATA / 'cldf/languages.csv') == old['registry_sha256']
    raw = {c['key']: c for g in groups for c in g['comparisons']}
    overrides = {x['comparison_key']: x for x in add['decisions']}
    assert len(raw) == len(set(raw)) == 395
    assert len(overrides) == 40 and set(overrides) <= raw.keys()
    rows = copy.deepcopy(old['comparators'])
    for r in rows:
        key = r['comparison_key']
        assert r['raw_label'] == raw[key]['source_language_label']
        assert r['literal_form'] == raw[key]['literal_form']
        r['label_role'] = 'source language or variety attribution'
        r['work_provenance'] = []
        r['required_source_note'] = None
        r['new_registry_proposal'] = None
        if key in overrides:
            d = overrides[key]
            r['prior_identity_proposal'] = {k: r[k] for k in ('proposed_base_id', 'identity_status', 'decision_basis')}
            r['proposed_base_id'] = d['proposed_base_id']
            r['identity_status'] = d['decision_status']
            r['decision_basis'] = d['source_scope_note']
            r['required_source_note'] = d['source_scope_note']
            r['candidate_only_ids'] = d['candidate_only_ids']
            r['new_registry_proposal'] = d['proposed_registry_record']
            reg = registry.get(d['proposed_base_id'])
            r['registry_record'] = {k: reg[k] for k in ('ID', 'Name', 'Glottocode', 'Clade')} if reg else None
            if d['proposed_base_id']:
                r['scope_disposition'] = 'target-family identity proposal'
            r['evidence'].append('numbered-target-identity-research-addendum.md')
            if d['proposed_base_id'] == 'Sk':
                r['source_stage_or_variety'] = 'Vedic; no early/late subdivision or named text inferred'
            elif d['proposed_base_id'] == 'Pk':
                r['source_stage_or_variety'] = 'Prakrit, variety unspecified by source'
            elif r['lookup_label'] == 'মু':
                r['scope_disposition'] = 'typed group-versus-lect hold; no lexical emission approved'
                r['identity_status'] = 'closed source-scope uncertainty: Munda group versus unnamed lect'
                r['decision_basis'] += ' Existing Mu is a group record; no explicit ingestion policy authorizing these as lect-level lexical rows was found in the checklist, parser or metadata tests.'
    work = next(w for w in refs['works'] if w['key'] == 'caryapada_work')
    carya_keys = []
    for occurrence in work['source_occurrences']:
        for line in occurrence['lines']:
            if 'চর্য' not in line['literal']:
                continue
            for key in line['record_keys']:
                if key not in raw:
                    continue
                r = next(r for r in rows if r['comparison_key'] == key)
                r['work_provenance'].append({
                    'proposed_auxiliary_work_key': work['key'], 'title': work['title'],
                    'source_group': occurrence['group'], 'source_line_key': line['source_line_key'],
                    'source_literal': line['literal'], 'primary_lexical_citation': 'niranjan2010chakma',
                    'citation_status': 'secondary quotation through Niranjan; no independently consulted passage',
                    'edition': None, 'passage': None, 'date': None,
                    'identity_authority': work['evidence_url'],
                })
                r['evidence'].append('numbered-reference-mapping-20260927.json')
                if r['lookup_label'] == 'চর্যা':
                    assert r['proposed_base_id'] is None
                    r['label_role'] = 'work/corpus provenance, not a language'
                    r['identity_status'] = 'identified work; language not independently assigned'
                    r['decision_basis'] = 'Caryapada is identified as the cited work. Do not turn a work key into Language_ID or infer a language solely from the corpus title.'
                    r['source_stage_or_variety'] = 'Caryapada secondary witness; linguistic stage unresolved'
                else:
                    assert occurrence['group'] == 52 and r['proposed_base_id'] == 'OB'
                    r['label_role'] = 'explicit Old Bengali attribution plus work provenance'
                carya_keys.append(key)
    assert len(carya_keys) == len(set(carya_keys)) == 3
    assert {r['comparison_key'] for r in rows} == raw.keys()
    for r in rows:
        original = raw[r['comparison_key']]
        assert (r['raw_label'], r['raw_value'], r['literal_form'], r['original_disposition']) == (
            original['source_language_label'], original['raw_source_value'], original['literal_form'], original['disposition'])
        assert r['proposed_base_id'] is None or r['proposed_base_id'] in registry or r['new_registry_proposal']
        assert not r['accepted_for_installation']
    counts = {
        'groups': len(groups), 'non_null_heads': sum(g['head'] is not None for g in groups),
        'comparator_slots': len(rows), 'declarations': len(old['declaration_inventory']),
        'research_decisions_applied': len(overrides), 'caryapada_work_occurrences': len(carya_keys),
        'existing_base_proposals': sum(r['proposed_base_id'] in registry for r in rows),
        'new_Chenchu_base_proposals': sum(bool(r['new_registry_proposal']) for r in rows),
        'no_unique_base_proposal': sum(r['proposed_base_id'] is None for r in rows),
        'physical_dispositions': dict(Counter(r['original_disposition'] for r in rows)),
        'identity_statuses': dict(Counter(r['identity_status'] for r in rows)),
    }
    result = {
        'date': '2026-09-27', 'reviewer': 'asur_finish',
        'status': 'integrated final identity review proposal; no canonical installation approval',
        'inputs': {n: sha(P / n) for n in files}, 'registry_sha256': sha(DATA / 'cldf/languages.csv'),
        'coverage': counts, 'comparators': rows,
        'declaration_inventory': old['declaration_inventory'],
        'proposed_auxiliary_works': [work],
        'Mu_policy_review': {'files_checked': ['SOURCE_INGESTION_CHECKLIST.md', 'make_cldf.py', 'unify_cldf.py', 'data/cdial/parse.py', 'tests/test_cdial_metadata.py', 'data/cdial/abbrevs.py', 'cldf/languages.csv'],
                             'decision': 'Mu exists as a group record, but no explicit policy found establishing these source-unspecified Munda words as lexical records of a particular lect. Retain all four literals under a typed scope hold, with Mu only a candidate group reference.'},
        'validation': ['Exact 395 source keys, labels, values and dispositions preserved.',
                       'All source, prior proposal, research and work-census hashes checked.',
                       'All 40 research decisions keyed and applied exactly once.',
                       'Caryapada mapped to work provenance in groups 4, 52 and 53, never to Language_ID.',
                       'Two explicit blank cells and one segmentation hold preserved.',
                       'No accepted-for-installation flags; no canonical writes.'],
    }
    (P / 'numbered-final-identity-review-proposal.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == '__main__':
    main()
