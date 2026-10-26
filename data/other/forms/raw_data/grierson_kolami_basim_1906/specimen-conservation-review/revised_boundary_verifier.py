"""Independent contributor and boundary guards, with explicit old/new anchor history.

Call validate_partition with reviewed atom records and the owner expression records.
This module neither runs the owner generator nor accepts unreviewed corrections.
"""
from collections import Counter
import copy
import hashlib
import json

JOIN_FIRST = 'grierson1906kolamibasim:p568:specimen76-line:11:word:8'
JOIN_TAIL = 'grierson1906kolamibasim:p568:specimen76-line:12:word:1'


def apply_pinned_approved_corrections(physical_atoms, ledger_path, expected_ledger_sha256, approved_cases, metadata_field='literal_correction_proposal'):
    """Construct reviewed source atoms independently from immutable raw records.

    approved_cases must come from a separately pinned independent/root decision,
    never from the owner's generated output. Every literal change needs an exact
    key/field/old/new approval. Extra prose metadata is not a literal approval.
    """
    data = ledger_path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected_ledger_sha256, 'Correction ledger drift'
    ledger = json.loads(data)
    allowed = {(c.get('source_unit_key', c.get('key')), c['field'], c['old'], c['new']) for c in approved_cases}
    corrected = copy.deepcopy(physical_atoms)
    by_key = {a['source_unit_key']: a for a in corrected}
    used = set()
    corrections = ledger if isinstance(ledger, list) else ledger['corrections']
    assert metadata_field in {'literal_correction_proposal', 'native_review_changes'}
    for c in corrections:
        signature = (c['source_unit_key'], c['field'], c['old'], c['new'])
        assert signature in allowed, ('Literal change lacks independent approval', signature)
        assert signature not in used, ('Duplicate correction', signature)
        assert c['field'] in {'raw_source_form', 'raw_interlinear_gloss'}, 'Not a literal correction field'
        atom = by_key[c['source_unit_key']]
        assert atom[c['field']] == c['old'], ('Correction before-value drift', signature)
        atom[c['field']] = c['new']
        if metadata_field == 'native_review_changes':
            atom.setdefault(metadata_field, []).append(c)
        else:
            atom[metadata_field] = c
        used.add(signature)
    return corrected


def validate_partition(atoms, expressions, historical_partitions):
    keys = [a['source_unit_key'] for a in atoms]
    assert len(keys) == len(set(keys)), 'Duplicate physical source key'
    by_key = {a['source_unit_key']: a for a in atoms}
    flat = [k for e in expressions for k in e['physical_source_keys']]
    assert flat == keys and Counter(flat) == Counter(keys), 'Lost, duplicated or reordered contributor'
    assert len({e['source_unit_key'] for e in expressions}) == len(expressions), 'Duplicate expression anchor'
    details = []
    for specimen in [75, 76]:
        selected = [e for e in expressions if e['specimen'] == specimen]
        physical = [a for a in atoms if f':specimen{specimen}-line:' in a['source_unit_key']]
        expected_ends, depth = [], 0
        for index, a in enumerate(physical):
            form = a['raw_source_form']
            closed = form.count('’') + form.count('”')
            depth += form.count('‘') + form.count('“') - closed
            assert depth >= 0, ('Unbalanced source quotation', a['source_unit_key'])
            if depth == 0 and (closed or form.endswith(('.', '?', '!'))):
                expected_ends.append(a['source_unit_key'])
            elif index == len(physical) - 1:
                assert depth == 0, 'Unclosed quotation at source specimen end'
                expected_ends.append(a['source_unit_key'])
        assert depth == 0
        assert [e['last_source_key'] for e in selected] == expected_ends, 'Source terminal/whole-quotation boundary mismatch'
        for index, e in enumerate(selected, 1):
            ks = e['physical_source_keys']
            assert ks and e['source_unit_key'] == ks[0] + ':expression', 'Identity not anchored to first physical atom'
            if 'display_sequence' in e:
                assert e['display_sequence'] == index, 'Display sequence drift (not durable identity)'
            assert e['first_source_key'] == ks[0] and e['last_source_key'] == ks[-1]
            source = [by_key[k] for k in ks]
            assert e['raw_physical_atoms'] == source, 'Raw contributing source records changed'
            assert e['printed_pages'] == sorted({a['printed_page'] for a in source})
            forms = [a['raw_source_form'] for a in source]
            glosses = [a['raw_interlinear_gloss'] for a in source]
            if JOIN_FIRST in ks:
                j = ks.index(JOIN_FIRST)
                assert ks[j + 1] == JOIN_TAIL, 'Continuation keys separated'
                assert forms[j:j + 2] == ['Bāp-', 'phōkān']
                assert glosses[j:j + 2] == ['Father-', 'near']
                forms[j:j + 2] = ['Bāp-phōkān']
                glosses[j:j + 2] = ['Father-near']
            assert e['raw_source_form'] == ' '.join(forms), 'Target text not exact source contributor join'
            assert e['raw_aligned_interlinear_gloss'] == ' '.join(glosses), 'English paraphrase or contributor drift'
            details.append({'key': e['source_unit_key'], 'first_key': ks[0], 'last_key': ks[-1], 'physical_keys': ks, 'pages': e['printed_pages']})

    old = historical_partitions
    assert [k for e in old for k in e['physical_keys']] == keys, 'Historical partition does not cover same physical source'
    old_map = {k: e['key'] for e in old for k in e['physical_keys']}
    new_map = {k: e['key'] for e in details for k in e['physical_keys']}
    old_to_new = []
    for e in old:
        targets = list(dict.fromkeys(new_map[k] for k in e['physical_keys']))
        old_to_new.append({'historical_key': e['key'], 'current_keys': targets, 'relation': 'split' if len(targets) > 1 else 'same-or-merged-span'})
    new_to_old = []
    for e in details:
        sources = list(dict.fromkeys(old_map[k] for k in e['physical_keys']))
        new_to_old.append({'current_key': e['key'], 'historical_keys': sources, 'relation': 'merge' if len(sources) > 1 else 'same-or-split-span'})
    return {'status': 'passed_exact_revised_partition_and_anchor_history', 'counts': {'physical_atoms': len(keys), 'historical_expressions': len(old), 'current_expressions': len(expressions)}, 'expression_partitions': details, 'historical_to_current': old_to_new, 'current_to_historical': new_to_old, 'identity_history_policy': 'These are review-span histories, not automatic Variant or Form-ID aliases. Retired anchors remain historical; new split spans anchor their actual first physical atom.'}
