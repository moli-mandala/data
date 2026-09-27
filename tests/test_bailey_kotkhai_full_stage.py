"""Focused staging checks for the complete two-page Kotkhai chapter."""
import csv
import importlib.util
import io
from pathlib import Path

from segments import Tokenizer

DATA = Path(__file__).resolve().parents[1]
P = DATA / 'data/other/forms/raw_data/bailey_kotkhai_1908'
spec = importlib.util.spec_from_file_location('kotkhai_full', P / 'prepare_full.py')
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def generated():
    return source.generate(source.INPUT if source.INPUT.exists() else source.DRAFT)


def test_whole_chapter_census_and_existing_keys():
    rows, audit = generated()
    assert len(rows) == 73 and len(audit) == 77
    assert sum(u['status'] == 'source_blank' for u in audit) == 11
    assert sum(u['status'] == 'excluded_control' for u in audit) == 1
    assert len({r[10] for r in rows}) == 73
    assert {k for u in audit for k in u['entry_keys']} == {r[10] for r in rows}
    old = list(csv.reader(source.OUTPUT.open()))
    assert {r[10] for r in old} <= {r[10] for r in rows}
    assert {u['section'] for u in audit} == {
        'noun', 'noun-comment', 'control', 'pronoun', 'adverb',
        'auxiliary', 'verb', 'future', 'imperfect', 'pluperfect', 'lexical-difference'}


def test_alternative_analysis_and_homographs():
    rows, audit = generated()
    by_key = {r[10]: r for r in rows}
    masculine = by_key['bailey1908kotkhai:p23:pronoun:sg-3-dat-acc']
    feminine = by_key['bailey1908kotkhai:p23:pronoun:sg-3-dat-acc:answer2']
    assert by_key['bailey1908kotkhai:p23:pronoun:pl-3-erg'][2] == 'tīnē'
    assert masculine[3] == 'him' and 'm' in masculine[14].split()
    assert feminine[3] == 'her' and 'f' in feminine[14].split()
    tomorrow = by_key['bailey1908kotkhai:p23:adverb:2']
    yesterday = by_key['bailey1908kotkhai:p23:adverb:3']
    assert tomorrow[2] == yesterday[2] and tomorrow[3] != yesterday[3]
    assert 's̲h̲' in tomorrow[2]
    anomalous = by_key['bailey1908kotkhai:p23:adverb:5']
    assert anomalous[3] == 'these' and 'uncertain' in anomalous[14].split()
    assert next(u for u in audit if u['source_unit_key'] == anomalous[10])['uncertainty']['type'] == 'source_gloss'
    rice = by_key['bailey1908kotkhai:p24:item:2']
    assert rice[3] == 'rice' and 'not an independent elicitation' in rice[6]


def test_expansion_is_bounded_and_graphs_are_unasserted():
    rows, audit = generated()
    assert len([u for u in audit if u['section'] == 'future']) == 6
    assert len([u for u in audit if u['section'] == 'imperfect']) == 6
    assert len([u for u in audit if u['section'] == 'pluperfect']) == 1
    assert all(not r[1] and not any(r[8:10]) and not any(r[11:14]) for r in rows)
    assert all(not r[4] and not r[5] for r in rows)


def test_profile_tags_references_and_scoped_parser():
    import make_cldf
    import make_refs
    import tags
    from unify_cldf import citation_keys
    rows, _ = generated()
    profile = Tokenizer(str(P / 'proposal-profile.txt'))
    for r in rows:
        assert len(r) == 15 and r[0] == 'Kotkhai'
        assert profile(r[2], column='IPA').replace(' ', '').replace('#', ' ') == r[2].lower()
        assert set(r[14].split()) <= tags.GRAMMATICAL_TAGS | tags.GENDER_TAGS | {source.DIALECT}
        assert set(make_refs.source_ids(r[7])) == citation_keys(r[7]) == {source.SOURCE}
    route = 'bailey-kotkhai-1908'
    previous = make_cldf.convertors.get(route)
    try:
        make_cldf.convertors[route] = profile
        errors = io.StringIO()
        parsed, stats = make_cldf.parse_file(str(P / 'proposal.csv'), errors, name='20260925-bailey-kotkhai')
        assert not errors.getvalue() and len(parsed) == stats['converted'] == 73
        assert {r.entry_key: r.old_form for r in parsed} == {r[10]: r[2] for r in rows}
        assert all(not r.native and not r.ipa for r in parsed)
    finally:
        if previous is None:
            make_cldf.convertors.pop(route, None)
        else:
            make_cldf.convertors[route] = previous
