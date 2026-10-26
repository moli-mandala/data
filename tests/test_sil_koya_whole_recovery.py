"""Whole-source recovery boundaries; canonical installation remains separate."""
import csv
import importlib.util
import json
import sys
from pathlib import Path

from segments.tokenizer import Tokenizer

ROOT = Path(__file__).parents[1]
P = ROOT / 'data/other/forms/raw_data/sil_koya_2021/whole-source-recovery'


def records(name):
    return [json.loads(line) for line in (P / name).read_text().splitlines()]


def rows(kind):
    return list(csv.reader((P / f'proposed-{kind}.csv').open()))


def test_repaired_scope_covers_all_units_without_changing_frozen_proposal():
    sys.path.insert(0, str(P.parent))
    from prepare_whole_additions import prepare
    narrative, comparative, audit = prepare()
    keyed = {r[10]: r for r in narrative}
    prefix = 'devagnanavaram2021:whole:'
    assert 'multiword-expression' in keyed[prefix + 'B.1:p17:7'][14].split()
    assert 'sentential' in keyed[prefix + 'B.7:p30:5'][14].split()
    assert {'sentential', 'interr'} <= set(keyed[prefix + 'B.6:p29:6'][14].split())
    assert sum('sentential' in r[14].split() for r in narrative) == 165
    assert sum('multiword-expression' in r[14].split() for r in narrative) == 40
    assert sum('interr' in r[14].split() for r in narrative) == 74
    for r in comparative:
        assert ('multiword-expression' in r[14].split()) == (' ' in r[2])
    old = {r[10]: r for r in rows('narrative') + rows('comparative')}
    assert all(r[:14] == old[r[10]][:14] for r in narrative + comparative)
    assert sum('scope_review' in u for u in audit) == 205


def test_complete_page_census_has_no_gap_or_overlap():
    census = json.loads((P / 'report-census-first-reading-20260927.json').read_text())
    ranges = census['reviewed_ranges'] + census['appendix_e_reconciliation']
    pages = [page for item in ranges for page in range(item['pdf_pages'][0], item['pdf_pages'][1] + 1)]
    assert sorted(pages) == list(range(1, 125))
    assert not census['remaining_pdf_ranges']


def test_all_questions_separate_answers_without_stripping_narrative_parentheses():
    units = records('narrative-reviewed-analysis.jsonl')
    assert len(units) == 205
    questions = [u for u in units if u['kind'] == 'test-question']
    assert len(questions) == 74
    assert sum(u['expected_answer_notation'] == 'parentheses' for u in questions) == 5
    assert all(u['gloss'].endswith('?') and u['expected_test_answer'] for u in questions)
    assert any('(to sleep)' in u['gloss'] for u in units if u['kind'] != 'test-question')
    assert any('(of day)' in u['gloss'] for u in questions)


def test_proposal_preserves_all_old_keys_and_moves_only_notes():
    old = list(csv.reader((ROOT / 'data/other/forms/raw_data/sil_koya_2021/historical-wordlist-installation/20260828-sil-koya.csv').open()))
    proposed = rows('wordlist')
    note_audit = records('wordlist-note-relocation.jsonl')
    assert len(old) == len(proposed) == len(note_audit) == 1438
    for before, after, evidence in zip(old, proposed, note_audit):
        assert all(a == b for i, (a, b) in enumerate(zip(before, after)) if i != 6)
        assert after[6] == ''
        assert evidence['entry_key'] == before[10]
        assert evidence['previous_notes'] == before[6]


def test_recovery_counts_attribution_and_no_invented_phonemic():
    narrative, comparative = rows('narrative'), rows('comparative')
    assert len(narrative) == 210 and len(comparative) == 18
    assert len({r[10] for r in narrative + comparative}) == 228
    assert all(len(r) == 15 and r[0] == 'Gondi' and not r[5] for r in narrative + comparative)
    audit = records('proposed-additions-audit.jsonl')
    assert len(audit) == 229
    controls = [u for u in audit if not u['proposed_entry_keys']]
    assert len(controls) == 1 and controls[0]['form'] == 'Vija'
    assert {r[2] for r in narrative if not r[3]} == {'lanton', 'jaw', 'Zoma', 'perrama'}
    assert all(not any(r[i] for i in [1, 8, 9, 11, 12, 13]) for r in narrative + comparative)


def test_profiles_preserve_narrative_notation_and_explicit_comparative_G():
    for kind in ['narrative', 'comparative']:
        tokenizer = Tokenizer(str(P / f'proposed-sil-koya-{kind}.txt'))
        for row in rows(kind):
            actual = tokenizer(row[2], column='IPA').replace(' ', '').replace('#', ' ')
            expected = row[2].replace('G', 'ɣ') if kind == 'comparative' else row[2].replace('[stutter]', '')
            assert '�' not in actual
            assert actual.split() == expected.split()
    narrative = rows('narrative')
    assert any('we!la' in r[2] for r in narrative)
    assert any('pooTku koTTaan' in r[2] for r in narrative)
    assert sum('[stutter]' in r[2] for r in narrative) == 1
