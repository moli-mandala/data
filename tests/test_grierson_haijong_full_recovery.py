"""Full Haijong source-stage regressions; no database or canonical writes."""
import csv
import importlib.util
import json
import unicodedata
from collections import Counter
from pathlib import Path

import pytest
from segments import Tokenizer

DATA = Path(__file__).resolve().parents[1]
PACKAGE = DATA / 'data/other/forms/raw_data/grierson_haijong_1903'
SPEC = importlib.util.spec_from_file_location('haijong_full_preparation', PACKAGE / 'prepare_full_source.py')
source = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(source)
PREFIX = 'grierson1903haijong:'


def records(name):
    return [json.loads(s) for s in (PACKAGE / name).read_text().splitlines()]


@pytest.fixture(scope='module')
def assembly():
    return source.prepare_native_assembly()


def test_complete_source_accounting_and_variants(assembly):
    rows, audit = assembly
    assert len(rows) == 896 and len(audit) == 881 + 26
    assert Counter(a['status'] for a in audit) == {
        'staged': 857, 'parallel_native_witness': 26,
        'bound_marker': 12, 'source_blank': 11, 'comparison_control': 1,
    }
    keys = {r[10] for r in rows}
    assert len(keys) == 896
    assert {k for a in audit for k in a.get('entry_keys', [])} == keys
    assert all(not r[11] or (r[11] in keys and r[11] != r[10]) for r in rows)
    assert all(len(r) == 15 and r[0] == 'Hajong' and not r[5] for r in rows)
    assert {str(i) for i in range(1, 242)} | {'51(a)', '52(a)', '60(a)', '61(a)'} == {
        a['prompt'] for a in audit if 'prompt' in a
    }
    legacy = {PREFIX + f'p354:haijong:prompt:{i}' for i in [1,2,3,4,6,7,8,9,10,11,13,14,20,23]}
    assert len(legacy) == 14 and legacy <= keys


def test_native_coverage_boundaries_and_literal_differences(assembly):
    rows, audit = assembly
    indexed = {r[10]: r for r in rows}
    aligned = [a for a in audit if 'native_alignment' in a]
    assert len(aligned) == 391
    assert sum(bool(r[4]) for r in rows) == 385
    groups = set()
    for a in aligned:
        n = a['native_alignment']; row = indexed[a['source_unit_key']]
        assert row[4] == unicodedata.normalize('NFC', n['recommended_native'])
        if len(n['roman_group_keys']) > 1:
            groups.add(tuple(n['roman_group_keys']))
            assert row[4] == ''
        else:
            assert row[4] and 'grierson1903haijong[p. 216, native line ' in row[7]
        assert n['roman'] == row[2]
    assert len(groups) == 3 and sum(map(len, groups)) == 6
    witnesses = {a['line']: a['native'] for a in audit if a['status'] == 'parallel_native_witness'}
    assert set(witnesses) == set(range(1, 27))
    for line, reading in [(15,'করঙ্গ'), (16,'জিঙ্গিয়াছে'), (19,'যবর্'), (20,'হোলে')]:
        assert reading in witnesses[line]
    assert indexed[PREFIX+'p217:specimen-I:line2:word7'][2:5:2] == ['bhāgrā', 'আগরা']
    assert indexed[PREFIX+'p217:specimen-I:line14:word4'][2:5:2] == ['āpnā', 'আপনর']
    assert all(not r[4] for r in rows if ':specimen-II:' in r[10])


def test_native_original_spans_cover_all_letters_without_forced_splitting():
    native = {r['native_unit_key']: r for r in records('native-p216-reviewed-20260926.jsonl')}
    used = set()
    for a in records('native-specimen-I-alignments-20260926.jsonl'):
        for loc in a['native_locators']:
            text = native[loc['native_unit_key']]['native']
            start, end = loc['character_start'], loc['character_end_exclusive']
            assert 0 <= start < end <= len(text)
            used.update((loc['native_unit_key'], i) for i in range(start, end))
    expected = {(k, i) for k,r in native.items() for i,c in enumerate(r['native'])
                if not c.isspace() and not unicodedata.category(c).startswith('P')}
    assert expected <= used


def test_eight_independent_roman_corrections(assembly):
    rows = {r[10]: r[2] for r in assembly[0]}
    expected = {
        'p217:specimen-I:line6:word3': 'uriyā-phĕlālē',
        'p217:specimen-I:line13:word1': 't̲s̲ākar',
        'p218:specimen-I:line5:word4': 't̲s̲ākar',
        'p218:specimen-I:line7:word1': 'hāta-nī',
        'p218:specimen-I:line14:word6': 'ẓabar',
        'p219:specimen-I:line1:word6': 'phĕlāsē',
        'p219:specimen-II:line3:word6': 'diba',
        'p219:specimen-II:line7:word2': 'bihānte',
    }
    for key, form in expected.items():
        assert rows[PREFIX+key] == form


def test_table_quantity_underdot_and_glyph_classes(assembly):
    rows = {r[10]: r for r in assembly[0]}
    expected = {(354,1):'Ăk', (354,18):'Āmālāk', (354,19):'Āmālāk',
                (358,44):'Lōā', (362,72):'Chaṛă', (366,89):'Bākhādur',
                (382,194):'May kōbābāk pāy', (382,201):'Mage kōbābāk lāgiba'}
    for (page,prompt), form in expected.items():
        assert rows[PREFIX+f'p{page}:haijong:prompt:{prompt}'][2] == form
    table = {a['prompt']:a['raw_forms'] for a in assembly[1] if 'prompt' in a}
    assert all('năthā' in f for k in ['129','131'] for f in table[k])
    assert table['146'][1] == 'Ăkrā kurtā'
    assert table['154'][0].startswith('Ăkra ')
    assert 'lāgibār' in table['174'][0]


def test_four_typed_uncertainties_survive(assembly):
    rows, audit = assembly
    expected = {'sāikkhˢāt','shāikkhˢāt','khˢēttra-ni','haurī'}
    uncertain = {r[10]:r[2] for r in rows if 'uncertain' in r[14].split()}
    assert set(uncertain.values()) == expected and len(uncertain) == 4
    for a in audit:
        if a.get('source_unit_key') in uncertain:
            assert a['uncertainty']


def test_dialect_registration_and_precise_citations(assembly):
    with (DATA/'cldf/dialects.csv').open() as handle:
        registry = {r['Tag']:r for r in csv.DictReader(handle)}
    for tag in [source.MYMENSINGH, source.SYLHET]:
        assert registry[tag]['Language_ID'] == 'Hajong'
        assert registry[tag]['Latitude'] == registry[tag]['Longitude'] == ''
    for r in assembly[0]:
        assert r[7].startswith('grierson1903haijong[p. ')
        if ':specimen-' in r[10]:
            assert ', line ' in r[7] and ', word ' in r[7]
            assert (source.SYLHET if ':specimen-II:' in r[10] else source.MYMENSINGH) in r[14]


def test_full_profile_coverage_and_consequential_mapping(assembly):
    tok = Tokenizer(str(PACKAGE/'full-profile-proposed.txt'))
    def convert(s):
        return unicodedata.normalize('NFC',tok(unicodedata.normalize('NFC',s), column='IPA').replace(' ','').replace('#',' '))
    for r in assembly[0]:
        assert not any(c in r[2][:-1] for c in '.?')
        expected = unicodedata.normalize('NFC',r[2].lower().replace('ṅ','ŋ').replace('w','v').rstrip('.?'))
        assert convert(r[2]) == expected and '�' not in convert(r[2])
    for s in ['sāikkhˢāt','shāikkhˢāt','khˢēttra-ni','thākkʸā','t̲s̲ārābāk','un̲g̲kāni','gaïnyai','uriyā-phĕlālē']:
        assert convert(s) == s
    assert convert('Īshᵛar-ṭhāi') == 'īshᵛar-ṭhāi'
    assert convert('Hē̃') == 'hē̃'
    assert convert('Ăk') != convert('Āk')
    assert convert('jiṅgiyāsē') == 'jiŋgiyāsē'
    assert convert('khāwālē-dāwālē') == 'khāvālē-dāvālē'
    assert convert('Talāk ki nām?') == 'talāk ki nām'
