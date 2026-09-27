"""Focused recovery checks; canonical source remains the five-row pilot until audit."""
import csv
import importlib.util
import json
import unicodedata
from pathlib import Path
from segments import Tokenizer

DATA = Path(__file__).resolve().parents[1]
P = DATA / 'data/other/forms/raw_data/grierson_malvi_rangri_1908'
spec = importlib.util.spec_from_file_location('rangri_full_stage', P / 'prepare_full.py')
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def test_complete_table_expansion_and_stable_pilot_keys():
    rows, audit, expanded = source.generate()
    assert len(expanded) == 241
    assert sum(len(u['forms']) for u in expanded) == 317
    assert [u['prompt_number'] for u in expanded if not u['forms']] == [172, 174, 201]
    units = {u['prompt_number']: u for u in expanded}
    assert units[27]['forms'] == ['Waṇī-kō', 'Waṇī-rō', 'uṇī-kō', 'uṇī-rō', 'vī-kō', 'vī-rō']
    assert units[109]['forms'] == ['Bāpā̃-sū̃', 'Bāpā̃-sē', 'Bāpā̃-ū̃']
    assert units[157]['forms'] == ['Tū̃ hai', 'Tū̃ hē']
    assert units[210]['forms'] == ['Vī jāvē', 'Vī jāy']
    assert 'sē' in units[109]['raw_source_cell']
    pilot = list(csv.reader((DATA / 'data/other/forms/20260925-grierson-malvi-rangri.csv').open()))
    assert {r[10] for r in pilot} <= {r[10] for r in rows}
    assert len({r[10] for r in rows}) == len(rows)
    assert {k for u in audit for k in u['entry_keys']} == {r[10] for r in rows}


def test_independent_literal_corrections_and_source_variation():
    rows, audit, expanded = source.generate()
    units = {u['prompt_number']: u for u in expanded}
    for n, form in [(43, 'Pīṭh'), (100, 'Arē-arē'), (171, 'Waī-nē'), (178, 'Māri-nē'), (188, 'Mhā̃-ē maryō')]:
        assert units[n]['forms'][0] == form
    assert 'ṭēkᵃrī-kā' in units[229]['forms'][0]
    assert 'ḍhā̃ḍhā' in units[229]['forms'][0]
    grammar = {u.get('section'): u for u in audit if 'section' in u}
    assert grammar['irregular-give-past']['forms'] == ['diyō', 'dīdhō', 'dīdō']
    assert grammar['pronoun-1sg-gen']['forms'] == ['mhārō', 'mārō']
    assert grammar['agent-unmarked']['forms'] == ['vō sarᵃdār ārī karī']
    assert grammar['3pl-nom']['forms'] == ['vī']
    assert grammar['strike-conj']['forms'][0] == 'mārī-nē'
    assert grammar['aux-pres-3pl']['forms'] == ['hē', 'hai']
    assert grammar['who-obl']['forms'] == ['kaṇī']
    assert grammar['no-one']['forms'] == ['kaṇī-ē̃ nahĩ diyā']
    assert grammar['rel-sg-obl']['forms'][0] == 'jaṇi'
    assert grammar['there-where']['forms'] == ['jathē']


def test_scope_and_structured_metadata():
    rows, audit, expanded = source.generate()
    for unit in audit:
        if unit['status'] in {'excluded_control', 'source_blank', 'bound_morphology_evidence'}:
            assert not unit['entry_keys']
    grammar = {u.get('section'): u for u in audit if 'section' in u}
    assert grammar['future-suffix']['status'] == 'bound_morphology_evidence'
    assert grammar['future-1sg']['status'] == 'ingested'
    by_key = {r[10]: r for r in rows}
    assert {'adj', 'degree'} <= set(by_key['grierson1908malvirangri:p315:rangri:item:134'][14].split())
    assert {'verb', 'copula', '3pl', 'pres'} <= set(by_key['grierson1908malvirangri:p317:rangri:item:161'][14].split())
    assert {'verb', 'fut', '2pl'} <= set(by_key['grierson1908malvirangri:p319:rangri:item:199'][14].split())
    assert not any('independently confirmed' in r[6] or 'second reading' in r[6] for r in rows)
    present = grammar['simple-pres-1sg']
    assert len(present['entry_keys']) == 3
    senses = [by_key[k] for k in present['entry_keys']]
    assert [r[3] for r in senses] == ['strike', 'may strike', 'shall strike']
    assert all(tag in row[14].split() for tag, row in zip(['pres', 'subjunctive', 'fut'], senses))
    passive = by_key[grammar['past-struck']['entry_keys'][0]]
    assert 'pass' in passive[14].split() and 'Source describes' in passive[6]



def test_complete_profile_symbol_coverage_and_registered_tags():
    import tags
    rows, audit, expanded = source.generate()
    tokenizer = Tokenizer(str(P / 'proposal-profile.txt'))
    for row in rows:
        assert len(row) == 15 and row[0] == 'Malw'
        assert row[2] and row[2] == unicodedata.normalize('NFC', row[2])
        assert tokenizer(row[2], column='IPA').replace(' ', '').replace('#', ' ') == row[2].lower().replace("ṅ", "ŋ")
        assert set(row[14].split()) <= tags.GRAMMATICAL_TAGS | tags.GENDER_TAGS | {source.DIALECT}


def test_specimen_modal_native_and_exact_reuse():
    rows, audit, _ = source.generate()
    by_key = {r[10]: r for r in rows}
    modal = [r for r in rows if r[3].lower() == 'would-have-lived']
    assert modal and all({'verb', 'modal', 'perfect'} <= set(r[14].split()) for r in modal)
    assert all('conditional' not in r[14].split() for r in modal)
    assert len(rows) == 1170 and len(audit) == 1346
    assert sum(u['status'] == 'reused_exact_attestation' for u in audit) == 223
    composite = [r for r in rows if r[2] == 'bāhar āvī-nē']
    assert len(composite) == 1 and composite[0][3] == 'having come out'
    assert composite[0][4] and 'conjunctive-participle' in composite[0][14].split()
    assert sum(u['status'] == 'native_alignment_hold' for u in audit) == 2
    for u in audit:
        if u.get('source_commentary') and u['entry_keys']:
            assert all(u['source_commentary'] in by_key[k][6] for k in u['entry_keys'])


def test_scoped_parser_preserves_source_keys_and_transcription():
    import io
    import make_cldf
    key = 'grierson-malvi-rangri-1908'
    previous = make_cldf.convertors.get(key)
    try:
        make_cldf.convertors[key] = Tokenizer(str(P / 'proposal-profile.txt'))
        errors = io.StringIO()
        parsed, stats = make_cldf.parse_file(str(P / 'proposal.csv'), errors, name='20260925-grierson-malvi-rangri')
        assert not errors.getvalue()
        assert len(parsed) == stats['converted'] == 1170
        rows = list(csv.reader((P / 'proposal.csv').open()))
        assert {r.entry_key: r.old_form for r in parsed} == {r[10]: r[2] for r in rows}
        assert all(not r.ipa for r in parsed)
    finally:
        if previous is None:
            make_cldf.convertors.pop(key, None)
        else:
            make_cldf.convertors[key] = previous


def test_citations_survive_actual_reference_parsers():
    import make_refs
    from unify_cldf import citation_keys
    import re
    rows, audit, _ = source.generate()
    for row in rows:
        assert not re.search(r'\[[^\]]*;', row[7])
        assert set(make_refs.source_ids(row[7])) == {'grierson1908malvirangri'}
        assert citation_keys(row[7]) == {'grierson1908malvirangri'}
    # Raw native locator punctuation is evidence, distinct from exported citations.
    assert any(';' in str(u.get('native_alignment', '')) for u in audit)
