"""Whole chapter recovery, deliberately without database generation."""
import csv
import importlib.util
import io
import json
from pathlib import Path

from segments import Tokenizer

PACKAGE = Path(__file__).resolve().parents[1] / 'data/other/forms/raw_data/bailey_surkhuli_1920'


def prepared():
    spec = importlib.util.spec_from_file_location('surkhuli_whole', PACKAGE / 'prepare_whole_source.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.prepare()


def test_complete_chapter_census_and_legacy_identity():
    rows, audit, records = prepared()
    assert len(rows) == 605 and len(audit) == 529 and len(records) == 305
    assert len(audit[:224]) == 224
    assert sum(r['status'] == 'target' for r in records) == 301
    assert sum(r['status'] == 'bound_context' for r in records) == 3
    assert sum(r['status'] == 'control' for r in records) == 1
    assert rows[:267] == list(csv.reader((PACKAGE / 'legacy-before-whole-recovery.csv').open()))
    assert rows == list(csv.reader((PACKAGE / 'whole-proposed.csv').open()))
    assert len({r[10] for r in rows}) == len(rows)
    assert {r['printed_page'] for r in records} == {117,148,149,150,151,152,153,154}
    keys = {r[10] for r in rows}
    assert all(not r[11] or r[11] in keys for r in rows)
    assert sum(bool(r[11]) for r in rows) == 27


def test_whole_responses_and_alternate_boundaries():
    rows, audit, records = prepared()
    sentences = [r for r in records if r['section'] == 'sentences']
    assert len(sentences) == 22
    assert [int(r['item']) for r in sentences] == list(range(1,23))
    bykey = {r[10]:r for r in rows}
    last = bykey['bailey1920surkhuli:154:sentences:22']
    assert last[2] == 'Gāŭ̃ā re baṇīē ku' and last[3] == 'Village of shopkeeper from.'
    assert 'sentential' not in last[14].split()
    assert 'multiword-expression' in last[14].split()
    assert bykey['bailey1920surkhuli:153:sentence-alternative:3a'][2:4] == ['zāŭ̃','up to']
    assert bykey['bailey1920surkhuli:153:sentence-alternative:3b'][2:4] == ['kētti','how much']
    assert len([r for r in rows if r[10].startswith('bailey1920surkhuli:153:sentences:3:answer')]) == 0


def test_explicit_paradigm_expansion_and_source_marks():
    rows, audit, _ = prepared()
    bykey = {r[10]:r for r in rows}
    byunit = {a.get('entry_key'):a for a in audit}
    assert bykey['bailey1920surkhuli:148:noun-girl:sg-4'][2] == 'tsheoṛī kũ'
    assert byunit['bailey1920surkhuli:148:noun-girl:sg-4']['source_forms'] == ['-ī kũ']
    assert bykey['bailey1920surkhuli:148:noun-horse:sg-5'][2] == 'gōhṛe'
    assert bykey['bailey1920surkhuli:148:noun-horse:pl-5'][2] == 'gōhṛĕūe'
    assert bykey['bailey1920surkhuli:152:verbs:drink'][2] == 'pīṇo'
    assert bykey['bailey1920surkhuli:152:eat-present:4'][2] == 'khāī ī'
    assert bykey['bailey1920surkhuli:153:sentences:2'][2].startswith('Es ')
    assert bykey['bailey1920surkhuli:153:sentences:13'][2].startswith('Ehro ')
    assert bykey['bailey1920surkhuli:153:sentences:12'][2].startswith('Ĕsro ')


def test_grammar_labels_and_uncertainty_separation():
    import tags
    rows, _, _ = prepared()
    assert not ({t for r in rows for t in r[14].split()} - tags.GRAMMATICAL_TAGS - tags.GENDER_TAGS)
    bykey = {r[10]:r for r in rows}
    assert 'participle' in bykey['bailey1920surkhuli:154:notes:11'][14].split()
    assert 'part' not in bykey['bailey1920surkhuli:154:notes:11'][14].split()
    assert 'conjunctive-participle' in bykey['bailey1920surkhuli:151:beat-conjunctive:1'][14].split()
    assert 'conditional' in bykey['bailey1920surkhuli:151:beat-past-conditional:sg-1-m'][14].split()
    for key in ['bailey1920surkhuli:154:notes:6','bailey1920surkhuli:117:introduction:9']:
        assert bykey[key][3] == '' and 'uncertain' in bykey[key][14].split()
    for suffix in ['149:pronouns:this-f-agent','149:pronouns:that-f-agent','154:sentences:21']:
        assert 'uncertain' in bykey['bailey1920surkhuli:'+suffix][14].split()
        assert 'roof-shaped' in bykey['bailey1920surkhuli:'+suffix][6]


def test_literal_profile_and_actual_scoped_parser():
    import make_cldf
    rows, _, _ = prepared()
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(PACKAGE / 'whole-proposed.csv'), errors, name='20260925-bailey-surkhuli')
    assert errors.getvalue() == '' and stats['converted'] == len(parsed) == 605
    bykey = {r[10]:r for r in rows}
    for row in parsed:
        raw = bykey[row.entry_key]
        assert (row.old_form,row.notes,row.source,row.tags) == (raw[2],raw[6],raw[7],raw[14])
        expected = ' '.join(raw[2].replace('w','v').replace('ṅ','ŋ').replace('.','').split())
        assert row.form == expected
    old, _ = make_cldf.parse_file(str(PACKAGE / 'legacy-before-whole-recovery.csv'), io.StringIO(), name='20260925-bailey-surkhuli')
    ids = {r.entry_key:r.id for r in parsed}
    assert all(ids[r.entry_key] == r.id for r in old)


def test_identifiable_morphology_and_audit_continuity():
    rows, audit, _ = prepared()
    bykey = {r[10]:r for r in rows}
    selected = json.loads((PACKAGE / 'root-whole-independent-selection-20260926.json').read_text())['rows']
    assert all(bykey[r[10]] == r for r in selected)
    morphology = [a for a in audit if a.get('status') == 'morphology_with_context']
    assert len(morphology) == 3
    emitted = [bykey[k] for a in morphology for k in a['exported_entry_keys']]
    assert len(emitted) == 10
    assert {r[2] for r in emitted} == {'-e','-ā','-ī','ŏndau','-ērōā','n','-dau'}
    assert all(r[3] == '' and {'affix','suffix'} & set(r[14].split()) for r in emitted)
    assert bykey['bailey1920surkhuli:150:time-adverbs:7'][2] == 'pōrs̲h̲ī'
