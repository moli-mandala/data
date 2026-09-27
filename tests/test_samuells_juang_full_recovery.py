"""Whole article recovery without building a database or changing installed source files."""
import csv
import hashlib
import importlib.util
import io
import json
from collections import Counter
from pathlib import Path

from segments import Tokenizer

DATA = Path(__file__).resolve().parents[1]
PACKAGE = DATA / 'data/other/forms/raw_data/samuells_juang_1856'
spec = importlib.util.spec_from_file_location('samuells_full', PACKAGE / 'prepare_full.py')
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def test_whole_scope_and_deterministic_accounting():
    rows, audit = source.build()
    assert len(rows) == 35 and len(audit) == 33
    assert Counter(u['section'] for u in audit) == {'vocabulary': 31, 'prose': 2}
    assert all(u['status'] == 'ingested' for u in audit)
    assert sorted(u['item'] for u in audit if u['section'] == 'vocabulary') == list(range(1, 32))
    assert len({r[10] for r in rows}) == 35
    assert {k for u in audit for k in u['entry_keys']} == {r[10] for r in rows}
    for name, encoded in source.encoded(rows, audit).items():
        assert (PACKAGE / name).read_bytes() == encoded.encode()
    scope = json.loads((PACKAGE / 'full-source-scope.json').read_text())
    assert [p['printed_page'] for p in scope['pages']] == list(range(295, 304))
    assert [p['pdf_page'] for p in scope['pages']] == [321, 322, 325, 326, 331, 332, 333, 334, 335]
    assert all(p['full_page_visually_read'] for p in scope['pages'])
    assert len(scope['plates']) == 3 and scope['held_units'] == 0


def test_all_reopened_responses_and_original_separation():
    rows, _ = source.build()
    old = list(csv.DictReader((PACKAGE / 'legacy-before-full-recovery/reviewed_inventory.tsv').open(), delimiter='\t'))
    recovered = {int(x['item']) for x in old if x['status'] == 'held'}
    assert recovered == {6, 7, 11, 17, 20, 24, 25, 26, 27, 28, 29}
    by_gloss = {}
    for r in rows:
        by_gloss.setdefault(r[3], []).append(r)
        assert r[6] != r[2] and not r[9]  # spelling is Original; alternatives are not etymology.
    assert [r[2] for r in by_gloss['woman']] == ['Khemé chélo', 'Juangurrakee']
    assert by_gloss['horse'][0][2] == 'Ghorardendite'
    assert by_gloss['ten men'][0][2] == 'Dench dik'
    assert by_gloss['to give'][0][2] == 'Dinkee mintuk'
    assert by_gloss['to come'][0][2] == 'Mendeldul koa'
    assert by_gloss['to go'][0][2] == 'Heena daee'
    assert by_gloss['we are'][0][2] == by_gloss['i am'][0][2] == 'Aynde asike'
    assert by_gloss['we are'][0][10] != by_gloss['i am'][0][10]
    assert '1pl' in by_gloss['we are'][0][14].split()
    assert '1sg' in by_gloss['i am'][0][14].split()
    assert 'second-person' in by_gloss['you are'][0][14].split()
    assert not {'2sg', '2pl'} & set(by_gloss['you are'][0][14].split())


def test_identity_variant_closure_and_target_attribution():
    rows, _ = source.build()
    legacy = list(csv.reader((PACKAGE / 'legacy-before-full-recovery/20260925-samuells-juang.csv').open()))
    assert [r[10] for r in rows[:21]] == [r[10] for r in legacy]
    for new, old in zip(rows[:21], legacy):
        assert new[2].lower() == old[2] and new[3] == old[3] and new[7] == old[7]
    by_key = {r[10]: r for r in rows}
    variants = [r for r in rows if r[11]]
    assert len(variants) == 2
    for r in variants:
        assert by_key[r[11]][3] == r[3] and by_key[r[11]][0] == r[0]
        assert r[11] != r[10] and not by_key[r[11]][11]
    assert all(not r[8] and not r[9] and not r[12] and not r[13] for r in rows)
    title = next(r for r in rows if r[2] == 'Pudhan')
    assert 'Ooriyas' in title[6] and 'title' in title[3]
    assert not {'Puttooa', 'toonga', 'kurka', 'panee aloo', 'Moolee Pudhan'} & {r[2] for r in rows}


def test_literal_profile_all_forms_and_shared_policy():
    import profile_policy
    rows, _ = source.build()
    profile = PACKAGE / 'proposal-profile.txt'
    tok = Tokenizer(str(profile))
    rules = dict(list(csv.reader(profile.open(), delimiter='\t'))[1:])
    for g, out in rules.items():
        assert profile_policy.house_output(g, out, rules, 'samuells-juang-1856') == out
    for row in rows:
        out = tok(row[2], column='IPA').replace(' ', '').replace('#', ' ')
        assert out == row[2].lower() and '\ufffd' not in out
    # No source key supports reinterpreting acute accents or digraphs as phonemes.
    assert tok('Báa', column='IPA').replace(' ', '') == 'báa'
    assert tok('Chalooko', column='IPA').replace(' ', '') == 'chalooko'


def test_actual_scoped_parser_with_proposed_profile(monkeypatch):
    import make_cldf
    from tags import GRAMMATICAL_TAGS, GENDER_TAGS
    rows, _ = source.build()
    monkeypatch.setitem(make_cldf.convertors, 'samuells-juang-1856', Tokenizer(str(PACKAGE / 'proposal-profile.txt')))
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(PACKAGE / 'proposal.csv'), errors, name=source.STEM)
    assert not errors.getvalue() and stats['converted'] == len(parsed) == 35
    for raw, parsed_row in zip(rows, parsed):
        assert parsed_row.old_form == raw[2]
        assert parsed_row.form == raw[2].lower()
        assert parsed_row.entry_key == raw[10] and parsed_row.variant_of_key == raw[11]
        assert parsed_row.source == raw[7] and parsed_row.notes == raw[6]
        assert parsed_row.native == '' and parsed_row.etymology == ''
        assert source.DIALECT in parsed_row.tags.split()
        assert set(parsed_row.tags.split()) - {source.DIALECT} <= GRAMMATICAL_TAGS | GENDER_TAGS
