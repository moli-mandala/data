"""Recovery-preview invariants; passing these does not approve source installation."""
import csv
import hashlib
import json
import unicodedata
from pathlib import Path

import pytest
from segments.tokenizer import Tokenizer
from tags import GENDER_TAGS, GRAMMATICAL_TAGS

DATA = Path(__file__).resolve().parents[1]
PACKAGE = DATA / 'data/other/forms/raw_data/bailey_north_jubbal_1920'


def read_rows(path):
    with path.open(newline='') as stream:
        return list(csv.reader(stream))


def test_preview_preserves_legacy_and_accounts_for_every_added_unit():
    rows = read_rows(PACKAGE/'whole-unfrozen-preview.csv')
    old = read_rows(PACKAGE/'historical-glossary-installation/20260925-bailey-north-jubbal.csv')
    audit = [json.loads(line) for line in (PACKAGE/'whole-unfrozen-preview-audit.jsonl').read_text().splitlines()]
    assert rows[:len(old)] == old
    assert len(rows) == 837 and len(old) == 330 and len(audit) == 688
    keys = [row[10] for row in rows]
    audit_keys = [key for record in audit for key in record['entry_keys']]
    assert len(set(keys)) == len(keys) and set(audit_keys) == set(keys)
    assert len(audit_keys) == len(keys)
    for row in rows:
        assert len(row) == 15 and row[0] == 'Barari' and row[2]
        assert set(row[14].split()) <= GENDER_TAGS | GRAMMATICAL_TAGS
        assert not any(row[index] for index in [1,4,5,8,9,11,12,13])


@pytest.mark.parametrize('normalization', ['NFC','NFD'])
def test_entire_preview_profile_preserves_unicode_marks(normalization):
    tokenizer = Tokenizer(str(PACKAGE/'whole-preview-profile.txt'))
    for row in read_rows(PACKAGE/'whole-unfrozen-preview.csv'):
        original = unicodedata.normalize(normalization, row[2])
        actual = tokenizer(original, column='IPA').replace(' ', '').replace('#', ' ')
        expected = unicodedata.normalize('NFC', row[2]).lower().replace('w','v').replace('ṅ','ŋ')
        assert unicodedata.normalize('NFC', actual) == expected, row[10]


def test_sentence_alternates_are_complete_and_source_uncertainty_survives():
    rows = {row[10]: row for row in read_rows(PACKAGE/'whole-unfrozen-preview.csv')}
    prefix='bailey1920northjubbal:'
    assert rows[prefix+'p183:sentence:1'][2] == 'Tēro nāũ kā sŏ'
    assert rows[prefix+'p183:sentence:1:answer2'][2] == 'Tēro nāũ kā ŏsso'
    assert rows[prefix+'p184:sentence:22:answer2'][2] == 'Gaũŏ re dŭkāndāro du gīnū'
    assert rows[prefix+'p143:comparison:stative-ending'][2] == '-ēṛu'
    for key in ['p183:sentence:13', 'p184:note:13']:
        row = rows[prefix+key]
        assert 'uncertain' in row[14].split()
        assert 'p. 167, note to sentence 13' in row[7]
        assert 'tentatively' in row[6]
        assert row[11:14] == ['', '', '']


def test_actual_parser_report_matches_current_preview_and_profile():
    report=json.loads((PACKAGE/'whole-preview-parser-validation.json').read_text())
    for field, name in [('preview_sha256','whole-unfrozen-preview.csv'),('profile_sha256','whole-preview-profile.txt')]:
        assert report[field] == hashlib.sha256((PACKAGE/name).read_bytes()).hexdigest()
    for normalization in ['NFC','NFD']:
        assert report['checks'][normalization]['rows'] == 837
        assert report['checks'][normalization]['errors'] == 0
    for name, digest in report['code_sha256'].items():
        assert hashlib.sha256((DATA/name).read_bytes()).hexdigest() == digest


def test_preview_provenance_pins_all_used_inputs_and_audit():
    report=json.loads((PACKAGE/'whole-unfrozen-preview-report.json').read_text())
    for name,digest in report['input_sha256'].items():
        assert hashlib.sha256((PACKAGE/name).read_bytes()).hexdigest() == digest
    assert 'whole-source-recovery/currency-cross-reference-20260927.json' in report['input_sha256']
    assert 'full-transcription.tsv' in report['input_sha256']
    assert 'prepare_whole_preview.py' in report['input_sha256']
    assert report['audit_sha256'] == hashlib.sha256((PACKAGE/'whole-unfrozen-preview-audit.jsonl').read_bytes()).hexdigest()
