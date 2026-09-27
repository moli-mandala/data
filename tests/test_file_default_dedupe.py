from types import SimpleNamespace

import pytest

from make_cldf import source_entry_dedupe_key
from source_meta import SourceMeta


def record(key, file='new.csv', source='new[p. 1]', tags='dialect:ko:norton noun'):
    return SimpleNamespace(entry_key=key, input_file=file, source=source,
                           lang='ko', param='', form='hop', gloss='ashes', tags=tags)


def test_installed_file_defaults_prevent_cross_source_identity_and_tag_loss():
    meta = SourceMeta()
    norton = record('cust1884korku:english-kor:p165:left:item18',
                    '20260925-cust-norton-korku.csv', 'cust1884korku[p. 165]')
    survey = record('older:hair', 'older.csv', 'older[p. 1]', 'dialect:ko:older noun')
    survey.gloss = 'hair'
    # This is the immutable-key component consumed by main's cleanup tuple.
    assert source_entry_dedupe_key(norton, meta) == norton.entry_key
    assert source_entry_dedupe_key(survey, meta) == ''
    assert (norton.lang, norton.param, norton.form, source_entry_dedupe_key(norton, meta)) != (
        survey.lang, survey.param, survey.form, source_entry_dedupe_key(survey, meta))
    assert norton.tags == 'dialect:ko:norton noun'
    assert survey.tags == 'dialect:ko:older noun'
    hair = record('cust1884korku:english-kor:p168:left:item02', norton.input_file, norton.source,
                  'dialect:ko:other verb')
    # Even identical literal form and gloss must not erase distinct dialect/POS.
    assert hair.form == norton.form and hair.gloss == norton.gloss
    assert source_entry_dedupe_key(hair, meta) != source_entry_dedupe_key(norton, meta)
    assert {hair.tags, norton.tags} == {'dialect:ko:other verb', 'dialect:ko:norton noun'}


@pytest.mark.parametrize('default,override,expected', [
    (True, None, 'new:one'), (True, False, ''),
    (False, True, 'new:one'), (False, None, ''),
])
def test_explicit_source_override_including_false_precedes_file_default(tmp_path, default, override, expected):
    path = tmp_path / 'new.yaml'
    source = '' if override is None else f'sources:\n  new:\n    identity:\n      dedupe_by_entry_key: {str(override).lower()}\n'
    path.write_text(f'defaults:\n  identity:\n    dedupe_by_entry_key: {str(default).lower()}\n' + source)
    meta = SourceMeta([path])
    row = record('new:one', source='new[p. 1];second[p. 2]')
    assert source_entry_dedupe_key(row, meta) == expected
    assert row.tags == 'dialect:ko:norton noun'
