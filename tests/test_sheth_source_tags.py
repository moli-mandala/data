import importlib.util
import json
from pathlib import Path

import tags
import sheth_sources as S

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('sheth_tag_integration', ROOT / 'data/other/forms/raw_data/sheth_integrate.py')
I = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(I)


def test_known_sources_and_exact_locators():
    names, claims = S.reference_tags(['(दे ४, ३४; हे १, १०८)'])
    assert names == ['Sheth:दे', 'Sheth:हे']
    assert [c['locator'] for c in claims] == ['४, ३४', '१, १०८']
    assert all(c['reference'] == '(दे ४, ३४; हे १, १०८)' for c in claims)


def test_no_prefix_fuzzy_or_language_label_matching():
    for ref in ['(हेम १)', '(अप)', '(शौ)', '(मा)', '(टी)', '(UNKNOWN 2)']:
        names, claims = S.reference_tags([ref])
        assert not names and claims[0]['status'] == 'unresolved'


def test_commentaries_and_variant_readings_are_distinct():
    names, claims = S.reference_tags(['(उप १०३१ टी)', '(गा ९७१ टि)'])
    assert names == ['Sheth:उप:commentary', 'Sheth:गा']
    assert [c['modifier'] for c in claims] == ['commentary', 'variant-reading']
    assert claims[1]['locator'] == '९७१ टि'


def test_nested_nat_abbreviations_cannot_become_main_catalogue_works():
    names, _ = S.reference_tags(['(विक्र १२)', '(नाट — विक्र ८८; चैत ९)'])
    assert names == ['Sheth:विक्र', 'Sheth:नाट', 'Sheth:नाट:विक्र', 'Sheth:नाट:चैत']
    assert S.LABELS['Sheth:विक्र'] == 'विक्रान्तकौरव'
    assert S.LABELS['Sheth:नाट:विक्र'] == 'विक्रमोर्वशी (नाट)'
    assert S.reference_tags(['(नाट — हे 12)'])[0] == ['Sheth:नाट']


def test_numbered_work_prefix_and_numeric_continuation():
    names, claims = S.reference_tags(['(कम्म 1, 12; 13; UNKNOWN; 14)'])
    assert names == ['Sheth:कम्म-१']
    assert claims[1]['continuation'] and claims[1]['locator'] == '13'
    assert claims[-1]['status'] == 'unresolved'
    names, claims = S.reference_tags(['(उप ५८७; ५९७ टी; ६००)'])
    assert names == ['Sheth:उप', 'Sheth:उप:commentary']
    assert [c['modifier'] for c in claims] == ['', 'commentary', '']


def test_source_tags_follow_senses_and_not_etymologies_or_grammar():
    raw = '<div><hw><b>अ</b><b>a</b></hw><reference>(अप)</reference><category>वि</category><etymology>दे</etymology><definition>१ पहला<reference>(हे १)</reference></definition><definition>२ <category>पुं</category>दूसरा<reference>(गा २)</reference></definition><reference>(सुपा ३)</reference></div>'
    result = I.prepare(raw, 1, 1)
    a, b = result['rows']
    assert a[14] == 'adj Sheth:हे'
    assert b[14] == 'noun m Sheth:गा'
    assert a[0] == b[0] == 'Ap'
    assert result['unscoped_references'] == ['(सुपा ३)']
    assert 'Sheth:दे' not in a[14]
    assert a[9] == b[9] == 'दे'


def test_registry_recognizes_work_tags_without_assigning_sanskrit_eras():
    extracted, notes = tags.extract_tags('Sheth:दे; Sheth:हे; source prose')
    assert set(extracted.split()) == {'Sheth:दे', 'Sheth:हे'}
    assert notes == 'source prose'
    assert not set(extracted.split()) & tags.ERA_TAGS


def test_frontend_labels_are_in_sync_with_verified_catalogue():
    labels = json.loads((ROOT.parent / 'jambu-static/src/lib/shethSourceLabels.json').read_text())
    assert labels == S.LABELS
    assert set(labels) <= tags.SOURCE_TAGS
    assert len({(r['Scope'], r['Abbreviation']) for r in S.WORKS}) == len(S.WORKS)
    assert all(r['PDF_Page'] in {'3', '4', '5', '6', '7', '8', '9', '12'} for r in S.WORKS)
