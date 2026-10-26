"""Guard physical atom accounting independently of expression generation.

Inputs must already be constructed from pinned immutable readings and separately
approved literal corrections. This module does not read an owner generator.
"""
from collections import Counter

from revised_boundary_verifier import JOIN_FIRST, JOIN_TAIL


def validate_atomic_accounting(immutable_atoms, reviewed_atoms, emissions, audit):
    raw = {a['source_unit_key']: a for a in immutable_atoms}
    expected = {a['source_unit_key']: a for a in reviewed_atoms}
    keys = [a['source_unit_key'] for a in immutable_atoms]
    assert len(raw) == len(keys), 'Duplicate immutable physical atom'
    assert list(expected) == keys, 'Reviewed source key order/coverage drift'
    assert [a['source_unit_key'] for a in audit] == keys, 'Physical audit loses/reorders an atom'
    output = {a['source_unit_key']: a for a in emissions}
    assert len(output) == len(emissions), 'Duplicate emitted identity'
    contributors = [k for e in emissions for k in e['physical_source_keys']]
    assert contributors == keys and Counter(contributors) == Counter(keys), 'Atomic contributor loss/duplication/reordering'
    mapping = {}
    for e in emissions:
        parts = e['physical_source_keys']
        assert e['source_unit_key'] == parts[0], 'Atomic identity no longer physical first anchor'
        assert e['raw_source_record'] == raw[parts[0]], 'Immutable raw source record changed'
        if len(parts) == 1:
            for field in ('raw_source_form', 'raw_interlinear_gloss'):
                assert e[field] == expected[parts[0]][field], ('Unapproved emitted literal', parts[0], field)
        else:
            assert parts == [JOIN_FIRST, JOIN_TAIL], 'Unapproved compound join or implicit repeated-token collapse'
            assert [expected[k]['raw_source_form'] for k in parts] == ['Bāp-', 'phōkān']
            assert [expected[k]['raw_interlinear_gloss'] for k in parts] == ['Father-', 'near']
            assert e['raw_source_form'] == 'Bāp-phōkān'
            assert e['raw_interlinear_gloss'] == 'Father-near'
            assert e['source_parts'] == [expected[k] for k in parts], 'Join source parts changed'
        for k in parts:
            mapping[k] = e['source_unit_key']
    for a in audit:
        k = a['source_unit_key']
        assert a['raw_source_record'] == raw[k], 'Physical audit overwrites historical source'
        assert a['proposed_entry_keys'] == [mapping[k]], 'Audit target does not match exact physical contributor'
        assert mapping[k] in output, 'Alias/reuse target missing'
    repeated = Counter((a['raw_source_form'], a['raw_interlinear_gloss'], a.get('source_scope')) for a in reviewed_atoms)
    return {
        'status': 'passed_exact_physical_atomic_accounting',
        'physical_atoms': len(keys),
        'emitted_atomic_records': len(emissions),
        'explicit_join_keys': [JOIN_FIRST, JOIN_TAIL],
        'physical_to_emitted': mapping,
        'repeated_literal_groups_retained_as_occurrences': sum(n > 1 for n in repeated.values()),
        'reuse_policy': 'No unstated repeated-token deduplication. The sole non-emitting physical tail is the explicitly printed Bāp-/phōkān continuation.',
    }
