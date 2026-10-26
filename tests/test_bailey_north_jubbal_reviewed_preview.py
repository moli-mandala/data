"""Reviewed source deltas, without approving installation or a full build."""
import csv
import hashlib
import importlib.util
import json
import unicodedata
from pathlib import Path

import pytest
from segments.tokenizer import Tokenizer
from tags import GENDER_TAGS, GRAMMATICAL_TAGS

DATA = Path(__file__).resolve().parents[1]
PACKAGE = DATA/'data/other/forms/raw_data/bailey_north_jubbal_1920'
STEM = 'whole-reviewed-preview-20260927'


def rows(name):
    with (PACKAGE/name).open(newline='') as stream:
        return list(csv.reader(stream))


def test_reviewed_revision_reproduces_only_approved_deltas():
    spec = importlib.util.spec_from_file_location('north_reviewed_preview', PACKAGE/'prepare_reviewed_preview.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    generated,audit,changes,_ = module.prepare()
    current = rows(f'{STEM}.csv')
    assert generated == current
    assert len(current) == 837 and len(audit) == 688 and len(changes) == 51
    old = rows('whole-unfrozen-preview.csv')
    assert [r[10] for r in current] == [r[10] for r in old]
    assert len({r[10] for r in current}) == 837
    assert set(k for a in audit for k in a['entry_keys']) == {r[10] for r in current}
    assert all(r[0] == 'Barari' and set(r[14].split()) <= GENDER_TAGS|GRAMMATICAL_TAGS for r in current)
    assert [r[2] for r in current[:330]] == [r[2] for r in old[:330]]
    changed = {a[10] for a,b in zip(old,current) if a != b}
    assert changed == {c['entry_key'] for c in changes}
    assert all(a[i] == b[i] for a,b in zip(old,current) for i in [0,1,4,5,7,8,9,10,11,12,13])


def test_original_mark_corrections_commands_and_legacy_qualifications():
    by = {r[10]:r for r in rows(f'{STEM}.csv')}
    prefix = 'bailey1920northjubbal:'
    assert by[prefix+'p183:sentence:12'][2] == 'Ēh rā bā’ī ăpṇī bauhṇī du jēṭhā'
    assert by[prefix+'p183:sentence:17'][2] == 'Ĕs khūb pīṭĕau rŏs̲h̲īo dā baunho'
    review = json.loads((PACKAGE/'whole-source-recovery/independent-sentences-context-review-20260927.json').read_text())
    for c in review['grammar_recommendations']:
        assert {'impv','sentential','multiword-expression'} <= set(by[c['source_unit_key']][14].split())
    legacy = json.loads((PACKAGE/'whole-source-recovery/legacy-metadata-review-20260927.json').read_text())
    for c in legacy['changes']:
        row = by[c['entry_key']]
        assert (row[3],row[6],row[14]) == tuple(c['proposed'][f] for f in ['gloss','notes','tags'])
    for key in ['p183:sentence:13','p184:note:13']:
        assert 'uncertain' in by[prefix+key][14].split()
        assert 'p. 167, note to sentence 13' in by[prefix+key][7]
        assert by[prefix+key][11:14] == ['','','']


@pytest.mark.parametrize('normalization',['NFC','NFD'])
def test_reviewed_literals_have_complete_profile_coverage(normalization):
    tokenizer = Tokenizer(str(PACKAGE/f'{STEM}-profile.txt'))
    for row in rows(f'{STEM}.csv'):
        actual = tokenizer(unicodedata.normalize(normalization,row[2]),column='IPA').replace(' ','').replace('#',' ')
        assert unicodedata.normalize('NFC',actual) == unicodedata.normalize('NFC',row[2]).lower().replace('w','v').replace('ṅ','ŋ'), row[10]


def test_revision_evidence_and_actual_parser_are_current():
    report = json.loads((PACKAGE/f'{STEM}-report.json').read_text())
    for name,digest in {**report['input_sha256'],**report['output_sha256']}.items():
        assert hashlib.sha256((PACKAGE/name).read_bytes()).hexdigest() == digest, name
    parser = json.loads((PACKAGE/f'{STEM}-parser-validation.json').read_text())
    assert parser['preview_sha256'] == report['output_sha256'][f'{STEM}.csv']
    assert parser['profile_sha256'] == report['output_sha256'][f'{STEM}-profile.txt']
    for norm in ['NFC','NFD']:
        assert parser['checks'][norm]['rows'] == 837
        assert parser['checks'][norm]['errors'] == 0
    for path,digest in parser['canonical_unchanged'].items():
        # This preinstallation check remains historical evidence, now archived.
        archived=PACKAGE/'historical-glossary-installation'/Path(path).name
        assert hashlib.sha256(archived.read_bytes()).hexdigest() == digest
