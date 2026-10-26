"""Focused tests for the ESIJa lemma layer (Rastorgueva & Edel'man 2000-2007)."""
import csv
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pytest

import _package_modules

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / 'data/other/forms/raw_data/esija_2000'
PARAMS = ROOT / 'data/other/params/20261025-esija.csv'
AUDIT = PKG / '20261025-esija-audit.csv'
SAMPLE = PKG / '20261025-esija-sample.csv'
MANIFEST = PKG / '20261025-esija-manifest.json'
_PACKAGE, (E, I) = _package_modules.load(PKG, "esija_ocr", "import_source")


@pytest.fixture(autouse=True, scope="module")
def _package_modules_active():
    with _PACKAGE.active():
        yield

KEY = 'rastorgueva2000esija'
STATUSES = {'installed', 'cross_reference', 'layout_false_positive', 'excluded_ambiguous'}
PRINTED_RANGE = {1: (63, 327), 2: (40, 502), 3: (44, 493)}


def rows(path):
    with open(path, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope='module')
def params():
    with open(PARAMS, encoding='utf-8', newline='') as f:
        return list(csv.reader(f))


@pytest.fixture(scope='module')
def audit():
    return rows(AUDIT)


def test_counts_reconcile(audit, params):
    man = json.load(open(MANIFEST, encoding='utf-8'))
    c = Counter(r['Status'] for r in audit)
    assert len(audit) == man['counts']['articles'] == 760
    assert set(c) == STATUSES
    assert c['installed'] == len(params) == man["counts"]["params"] == 587
    assert c['cross_reference'] == 168
    assert c['layout_false_positive'] == 3 and c['excluded_ambiguous'] == 2
    src = Counter(r['Lemma_Source'] for r in audit if r['Status'] == 'installed')
    assert src == {'ocr': 346, 'manual': 241}
    by_vol = Counter((r['Volume'], r['Status']) for r in audit if r['Status'] == 'installed')
    assert (by_vol[('1', 'installed')], by_vol[('2', 'installed')], by_vol[('3', 'installed')]) == (190, 205, 192)


def test_params_schema(params):
    ids = [r[0] for r in params]
    assert len(ids) == len(set(ids))
    for pid, lang, form, gloss, src in params:
        m = re.fullmatch(r'esija-([123])-(\d+)-[a-z0-9]+(-\d{1,2})?(-[b-z])?', pid)
        assert m, pid
        vol, page = int(m.group(1)), int(m.group(2))
        assert lang == 'PIr'
        assert src == f'{KEY}[vol. {vol}, p. {page}]'
        lo, hi = PRINTED_RANGE[vol]
        assert lo <= page <= hi
        assert form.startswith('*') and len(form) > 1
        assert unicodedata.normalize('NFC', form) == form
        assert '�' not in form + gloss
        assert ',' not in form  # make_cldf keeps only the text before a comma
        assert all(ch in I.MANUAL_ALPHABET or ch == '*' for ch in form), form


def test_audit_matches_params(audit, params):
    p = {r[0]: r for r in params}
    keys = [r['Entry_Key'] for r in audit]
    assert len(keys) == len(set(keys))
    for r in audit:
        assert r['Reflex_Stage'] == 'deferred' and r['Article_Pages']
        assert r['Source'].startswith(f'{KEY}[vol. {r["Volume"]}, p. {r["Printed_Page"]}]')
        assert re.fullmatch(rf'{KEY}:v[123]:p\d+:h\d+', r['Entry_Key'])
        if r['Status'] == 'installed':
            assert p[r['ID']][2] == r['Form'] and p[r['ID']][3] == r['Gloss_EN']
            if r['Lemma_Source'] == 'ocr':
                assert 'unreviewed-ocr' in r['Review'].split()
                assert r['OCR_Status'].startswith('ocr_consensus')
            else:
                assert r['Lemma_Source'] == 'manual' and 'manual-collation' in r['Review'].split()
                assert 'unreviewed-ocr' not in r['Review'].split()
        else:
            assert r['ID'] not in p


def test_ids_are_persisted(params):
    ids = {r['Entry_Key']: r['ID'] for r in rows(PKG / 'ids.csv')}
    assert set(ids.values()) >= {r[0] for r in params}


def test_glosses_come_from_translation_table(audit):
    table = {r['Russian']: r['English'] for r in rows(PKG / 'glosses_en.csv')}
    for r in audit:
        if r['Status'] == 'installed' and r['Gloss_RU']:
            assert r['Gloss_EN'] == table.get(r['Gloss_RU'], '')


def test_seeded_sample_zero_material_errors():
    s = rows(SAMPLE)
    assert Counter((r['Round'], r['Seed']) for r in s) == {('final', str(I.SEED)): 20, ('manual', str(I.SEED_MANUAL)): 20}
    assert all(r['Verdict'] == 'ok' and r['Material_Error'] == 'no' for r in s)


@pytest.mark.parametrize('form,gloss,hom', [
    ('*hapta', 'seven', ''), ('*dugdar-', 'daughter', ''), ('*brātar-', None, ''),
    ('*barźa-', 'birch', ''), ('*h(u)u̯ahar-', 'sister', ''), ('*au-', 'care for, help; promote', '1'),
    ('*gada-', 'thief, robber, brigand', '1'), ('*frušā-', 'flea', ''),
])
def test_representative_rows(audit, form, gloss, hom):
    hits = [r for r in audit if r['Status'] == 'installed' and r['Form'] == form and r['Homonym'] == hom]
    assert hits, form
    if gloss:
        assert any(r['Gloss_EN'] == gloss for r in hits)


def test_homonym_in_id_not_form(audit):
    r = next(r for r in audit if r['Entry_Key'] == f'{KEY}:v1:p253:h2')
    assert r['Form'] == '*au-' and r['Homonym'] == '1' and r['ID'].endswith('-1')


def test_cross_references_not_installed(audit):
    xr = [r for r in audit if r['Status'] == 'cross_reference']
    assert all((r['Gloss_Kind'] == 'cross_reference' or r['Lemma_Source'] == 'manual') and not r['ID'] for r in xr)


def test_page_mapping():
    assert I.printed(62) == (1, 63) and I.printed(326) == (1, 327)
    assert I.printed(366) == (2, 40) and I.printed(600) == (2, 274)
    assert I.printed(873) == (3, 44) and I.printed(1127) == (3, 298)


def test_first_lemma_and_cleaning():
    assert I.first_lemma('das-:dās-') == 'das-'
    assert I.first_lemma('gazna-,*ganźa-,*ganǰa-') == 'gazna-'
    assert I.first_lemma('dāda-/*dada-') == 'dāda-'
    assert I.first_lemma('gau-:gū-/gu-') == 'gau-'
    assert I.first_lemma('ap/fšah-') == 'ap/fšah-'  # word-internal slash is notation
    assert I.first_lemma('gar-1)') == 'gar-'
    lem, flags, ok = I.clean_form('afś-man-(?)', 'afs-man-(?)')
    assert lem == 'afś-man-' and 'source-query' in flags and ok
    assert I.clean_form('t', '#1 5')[2] is False  # no vowel: not a lemma
    assert I.clean_form('čid', '»-Cid')[0] == '-čid'
    assert I.clean_form('au', '* au- : ü-')[0] == 'au-'
    assert I.slug('h(u)u̯ahar-') == 'huuahar' and I.slug('xᵛar-') == 'xvar'


def test_gloss_parser():
    head = "*grand-"
    p = E.parse("*grand- 'кусать, жалить; царапать' — из арийск. *ghrand-", head)
    assert p['kind'] == 'gloss' and p['ru'] == 'кусать, жалить; царапать'
    p = E.parse("*dauk- : duk- 'шить, сшивать', предположительно ... 'веду'; 'тяну'", '*dauk- : duk-')
    assert p['ru'] == 'шить, сшивать'  # later quotes belong to comparanda
    p = E.parse("*gaur- см. * 9 gar-.", '*gaur-')
    assert p['kind'] == 'cross_reference'
    p = E.parse("*arša- 'медведь' см. *r̥ša-.", '*arša-')
    assert p['kind'] == 'cross_reference' and p['ru'] == 'медведь'
    p = E.parse("ср. др.-инд. javate '(он) спешит'", '* 8 gau- : gü-')
    assert p['kind'] == 'unanchored' and p['ru'] == ''


def test_abbyy_superscripts():
    words = [['*', 'TimesN', 5.2, 14.5], ['J', 'TimesN', 3.4, 17.5], ['au-', 'TimesN', 5.2, 19.6]]
    assert E.abbyy_homonym(words) == '1' and E.abbyy_clean(words) == '* au-'
    words = [['*', 'TimesN', 5.2, 14.5], ['4', 'TimesN', 3.4, 17.6], ['ah-', 'TimesN', 5.2, 19.8]]
    assert E.abbyy_homonym(words) == '4'


def test_skeleton_maps_abbyy_lookalikes():
    assert E.skel('*£anga-', True) == E.skel('čanga-') == 'canga'
    assert E.skel('*frau5-', True) == E.skel('frauš-')
    assert E.skel('•ansu-', True) == E.skel('anśu-')


def test_pir_registered():
    langs = ROOT / 'cldf/languages.csv'
    if not langs.exists():
        pytest.skip('registry not built')
    assert any(r['ID'] == 'PIr' for r in rows(langs))


def test_bib_key_exists():
    bib = ROOT / 'cldf/sources.bib'
    if not bib.exists():
        pytest.skip('bibliography not available')
    assert f'@book{{{KEY},' in bib.read_text(encoding='utf-8')


def test_manual_layer(audit):
    man = {r['Entry_Key']: r for r in rows(PKG / 'manual-lemmas.csv')}
    assert len(man) == 247 and {r['Reviewer'] for r in man.values()} == {'manual'}
    assert Counter(r['Decision'] for r in man.values()) == {'lemma': 241, 'not_headword': 3, 'ambiguous': 2, 'cross_reference': 1}
    by_key = {r['Entry_Key']: r for r in audit}
    for k, m in man.items():
        r = by_key[k]
        assert r['Lemma_Source'] == 'manual'
        if m['Decision'] == 'lemma':
            assert r['Status'] == 'installed' and r['Form'] == m['Form'] and r['Homonym'] == m['Homonym']
            if m['Gloss_RU']:
                assert r['Gloss_RU'] == m['Gloss_RU']
        else:
            assert r['Status'] in {'layout_false_positive', 'excluded_ambiguous', 'cross_reference'} and r['Reason']


@pytest.mark.parametrize('key,form,hom', [
    ('v1:p63:h1', '*a-', '1'), ('v1:p71:h1', '*a', '5'), ('v2:p242:h1', '*čatu̯ar-', ''),
    ('v3:p271:h3', '*gāi̯-', ''), ('v2:p253:h1', '*čā̆stra-', ''), ('v1:p187:h1', '*ap/fšah-', ''),
    ('v3:p485:h1', '*hu̯aśrū-', ''), ('v1:p270:h1', '*aušta-', ''),
])
def test_manual_examples(audit, key, form, hom):
    r = next(r for r in audit if r['Entry_Key'] == f'{KEY}:{key}')
    assert (r['Status'], r['Form'], r['Homonym']) == ('installed', form, hom)


def test_params_yaml():
    import source_meta
    y = ROOT / 'data/other/params/20261025-esija.yaml'
    assert y.exists()
    text = y.read_text(encoding='utf-8')
    assert 'rastorgueva2000esija:' in text and 'ocr: true' in text and 'etymology_provenance: source' in text
