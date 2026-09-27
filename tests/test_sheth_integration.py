"""Source boundary, registry and compiler tests for the resolved Sheth subset."""
import csv
import gzip
import importlib.util
import io
import json
import shutil
import unicodedata
from collections import Counter
from pathlib import Path

import pytest
from segments.tokenizer import Tokenizer
import make_cldf
import unify_cldf
import burushaski_comparisons
from assign_form_ids import assign_ids

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/other/forms/raw_data'
SPEC = importlib.util.spec_from_file_location('sheth_integration', RAW / 'sheth_integrate.py')
S = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S)
PACKAGE = RAW / 'sheth_2026'


def rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def test_layout_brace_and_reference_encoded_language_are_not_glosses():
    record = S.prepare('<div><hw><b>एवं</b><b>ēvaṃ</b>, <b>एव</b><b>ēva</b></hw> } <reference>(अप)</reference><category>अ</category>इस प्रकार<reference>(हे ४, ३)</reference>।</div>', 1, 1)
    assert len(record['rows']) == 2
    assert all(r[0] == 'Ap' and r[3] == 'इस प्रकार' for r in record['rows'])
    assert record['rows'][1][11] == record['rows'][0][10]
    assert record['printed_references'] == ['(हे ४, ३)']


def test_lect_reference_requires_exact_whole_label_and_preserves_sense_scope():
    raw = '<div><hw><b>अ</b><b>a</b></hw><category>वि</category><definition>१ पहला<reference>(अप १२)</reference></definition><definition>२ <reference>(शौ)</reference>दूसरा</definition></div>'
    result = S.prepare(raw, 1, 1)
    assert [r[0] for r in result['rows']] == ['Pk', 'Pk']
    assert result['rows'][0][14] == 'adj'
    assert result['rows'][1][14] == 'adj dialect:Pk:s:Shauraseni'
    assert '(अप १२)' in result['printed_references']


def test_crossreference_is_confined_to_its_own_sense():
    raw = '<div><hw><b>अ</b><b>a</b></hw><definition>१ पहला</definition><definition>२ <var>देखो इ</var></definition></div>'
    result = S.prepare(raw, 1, 1)
    assert result['rows'][0][3] == 'पहला' and not result['rows'][0][6]
    assert result['rows'][1][3] == ''
    assert result['rows'][1][6] == 'Source cross-reference: देखो इ'


def test_untagged_see_instruction_is_not_a_definition():
    record = S.prepare('<div><hw><b>अ</b><b>a</b></hw>ऊपर देखो<reference>(पउम २, ६३)</reference>।</div>', 1, 1)
    assert record['rows'][0][3] == ''
    assert record['rows'][0][6] == 'Source cross-reference: ऊपर देखो'


@pytest.mark.parametrize('body', [
    'तदासक्त विपा १, २; राज)',
    'आँख मींचना। णिमिल्लइ',
    'नट। °खाइया स्त्री दीक्षा-विशेष',
    '<category>अज्ञात</category>meaning',
    '<var>देखो इ</var><definition>२ दूसरा</definition>',
])
def test_unresolved_structure_is_audit_only(body):
    markup = '<div><hw><b>अ</b><b>a</b></hw>' + body + '</div>'
    record = S.prepare(markup, 1, 1)
    assert not record['rows'] and record['status'] == 'audit-only'
    assert record['selection_reasons'] and record['raw_markup'] == markup


def test_all_installed_rows_have_audited_origin_registered_metadata_and_profiles():
    report = json.loads((PACKAGE / 'report.json').read_text())
    assert report['counts']['installed_rows'] == 42118
    assert report['counts']['installed-input'] == 31501
    assert report['counts']['audit-only'] == 10137
    assert report['counts']['variants'] == 2268
    with (RAW.parent / S.FILENAME).open(newline='') as stream:
        installed = list(csv.reader(stream))
    by_key = {r[10]: r for r in installed}
    assert len(by_key) == len(installed) == report['counts']['installed_rows']
    assert Counter(r[0] for r in installed) == report['languages']
    assert sum(bool(r[11]) for r in installed) == report['counts']['variants']
    languages = {r['ID'] for r in rows(ROOT / 'cldf/languages.csv')}
    dialects = {r['Tag']: r['Language_ID'] for r in rows(ROOT / 'cldf/dialects.csv')}
    assert all(r[0] in languages for r in installed)
    assert all(dialects[t] == r[0] for r in installed for t in r[14].split() if t.startswith('dialect:'))
    assert all(not r[11] or r[11] in by_key for r in installed)
    assert all(not r[1] and not r[8] and not r[12] and not r[13] for r in installed)
    audit_counts = Counter()
    seen = set()
    with gzip.open(PACKAGE / 'audit.jsonl.gz', 'rt') as stream:
        for line in stream:
            record = json.loads(line)
            audit_counts[record['status']] += 1
            for row in record['rows']:
                assert by_key[row[10]] == row
                seen.add(row[10])
            if record['status'] == 'audit-only':
                assert record['selection_reasons'] and not record['rows']
    assert sum(audit_counts.values()) == 41638
    assert seen == set(by_key)
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(RAW.parent / S.FILENAME), errors, file_num='20260914-sheth')
    assert not errors.getvalue() and len(parsed) == len(installed)
    assert stats['converted'] == len(installed)
    tokenizer = Tokenizer(str(ROOT / 'conversion/sheth-ddsa.txt'))
    house = lambda v: unicodedata.normalize('NFC', tokenizer(unicodedata.normalize('NFC', v), column='IPA').replace(' ', '').replace('#', ' '))
    assert house('aaṅkha') == 'aaŋkʰa' and house('amha') == 'amʰa' and house('jīmṛa') == 'jīmr̩a'
    for row in parsed:
        source = by_key[row.entry_key]
        assert source[2] == row.old_form
        assert row.form == house(source[2])
        assert row.ipa == '' and row.source == source[7]
        assert '�' not in row.form
    reference = next(r for r in rows(ROOT / 'cldf/references.csv') if r['ID'] == S.SOURCE)
    assert S.FILENAME in reference['Provenance'] and 'DDSA' in reference['Progress']
    assert reference['Editor'] and reference['Source']


def test_source_only_build_preserves_homographs_variant_edges_and_stable_ids(tmp_path, monkeypatch):
    report = json.loads((PACKAGE / 'report.json').read_text())
    count = report['counts']['installed_rows']
    empty = ['data/cdial/cdial.csv', 'data/munda/forms.csv', 'data/dedr/dedr_new.csv', 'data/dedr/pdr.csv',
             make_cldf.MERRIAM_DRAVIDIAN_DB_FILE, 'data/dbia/forms.csv',
             'data/cdial/params.csv', 'data/munda/params.csv', 'data/dedr/params.csv', 'data/dbia/params.csv',
             'data/etymologies.csv', *make_cldf.WESTERN_SURVEY_FILES, *make_cldf.MANUAL_SURVEY_FILES]
    for name in empty:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (tmp_path / 'cldf').mkdir()
    for name in ['cldf/languages.csv', 'cldf/dialects.csv', make_cldf.SHETH_FILE]:
        shutil.copyfile(ROOT / name, tmp_path / name)
    (tmp_path / 'data/nuristani_cognates.csv').write_text('Ancestor_ID\n')
    for name in ['data/cross-family-comparisons.csv', 'data/manual-cross-family-comparisons.csv', 'data/dbia/comparisons.csv']:
        (tmp_path / name).write_text(','.join(make_cldf.CROSS_FAMILY_COLUMNS) + '\n')
    monkeypatch.chdir(tmp_path)
    for name in ['lang_set', 'param_set', 'included_params']:
        monkeypatch.setattr(make_cldf, name, set())
    make_cldf.main()
    compiled = rows(tmp_path / 'cldf/forms.csv')
    assert len(compiled) == count and not (tmp_path / 'errors.txt').read_text()
    assert len({r['Entry_Key'] for r in compiled}) == count
    by_key = {r['Entry_Key']: r for r in compiled}
    expected_edges = {(r['ID'], by_key[r['Variant_Of_Key']]['ID']) for r in compiled if r['Variant_Of_Key']}
    (tmp_path / 'data/strand_oia_redirects.csv').write_text('Strand_ID,CDIAL_ID\n')
    monkeypatch.setattr(unify_cldf, 'load_burushaski_catalog', lambda: [])
    monkeypatch.setattr(unify_cldf, 'append_burushaski_comparisons', lambda r: burushaski_comparisons.append_comparisons(r, tmp_path / 'cldf/comparisons.csv'))
    monkeypatch.setattr(unify_cldf, 'write_burushaski_comparison_audit', lambda r: burushaski_comparisons.write_audit(r, tmp_path / 'data/burushaski-audit.csv'))
    unify_cldf.main()
    unified = rows(tmp_path / 'cldf/forms.csv')
    edges = rows(tmp_path / 'cldf/edges.csv')
    assert len(unified) == count
    assert len(edges) == report['counts']['variants']
    assert {(e['Child_ID'], e['Parent_ID']) for e in edges} == expected_edges
    assert all(e['Kind'] == 'variant' for e in edges)
    keys = {r['ID']: r['Entry_Key'] for r in compiled}
    ids, registry = assign_ids(unified, [], keys)
    changed = [dict(r, Form=r['Form'] + 'x', Original=r['Original'] + 'x', Gloss=r['Gloss'] + ' corrected') for r in reversed(unified)]
    updated, _ = assign_ids(changed, registry, keys)
    assert ids == updated
