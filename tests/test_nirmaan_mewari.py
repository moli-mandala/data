"""Focused data-only checks: never invoke a CLDF/database build."""
import collections
import copy
import csv
import importlib.util
import io
import json
from pathlib import Path
import random
import unicodedata

import pybtex.database
from segments.tokenizer import Tokenizer
from make_cldf import parse_file
from assign_form_ids import assign_ids
from dialects import normalize_dialect, load_dialect_aliases

ROOT = Path(__file__).parents[1]
RAW = ROOT / 'data/other/forms/raw_data'
STEM = '20260911-nirmaan-mewari'
FORMS = ROOT / f'data/other/forms/{STEM}.csv'
spec = importlib.util.spec_from_file_location('nirmaan_mewari', RAW / 'nirmaan_mewari.py')
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def articles():
    return [json.loads(line) for line in (RAW / f'{STEM}-audit.jsonl').read_text().splitlines()]


def forms():
    with FORMS.open(newline='') as f:
        return list(csv.reader(f))


def test_complete_source_reconciliation_and_reproducible_emission():
    audit = articles()
    rows = forms()
    assert len(audit) == 6406
    assert len(rows) == 6535 == 6406 + 128 + 1
    assert sum(len(a['senses']) > 1 for a in audit) == 121
    assert sum(bool(a['homograph']) for a in audit) == 376
    assert len({a['key'] for a in audit}) == 6406
    assert len({r[10] for r in rows}) == 6535
    assert all(a['status'] == 'installed-raw' and a['raw_extracted_text'] for a in audit)
    assert source.emit(copy.deepcopy(audit)) == rows
    assert all(len(r) == 15 and r[2] and r[4] and r[2] == r[5] for r in rows)
    assert all('�' not in field for r in rows for field in r)
    assert all(unicodedata.is_normalized('NFC', field) for r in rows for field in r)
    assert audit[0]['key'] == 'nirmaan2018mewari:p003:c1:e01'
    assert audit[-1]['key'] == 'nirmaan2018mewari:p373:c2:e07'


def test_profile_corpus_and_source_layers():
    errors = io.StringIO()
    parsed, stats = parse_file(str(FORMS), errors)
    assert len(parsed) == 6535
    assert stats == {'converted': 6535, 'for_conversion': 6535}
    assert not errors.getvalue()
    assert all(r.old_form == r.ipa for r in parsed)
    by_key = {r.entry_key: r for r in parsed}
    assert by_key['nirmaan2018mewari:p004:c2:e05'].form == 'akkar'
    assert by_key['nirmaan2018mewari:p009:c2:e02'].form == 'andārī'
    assert by_key['nirmaan2018mewari:p163:c1:e08'].form == 't̟āmeṛī'
    assert by_key['nirmaan2018mewari:p251:c1:e04:v2'].form == 'ɓārī'
    converter = Tokenizer(str(ROOT / 'conversion/nirmaan-mewari.txt'))
    for row in forms():
        for normalization in ('NFC', 'NFD'):
            assert '�' not in converter(unicodedata.normalize(normalization, row[2]), column='IPA')


def test_variants_are_explicit_resolved_acyclic_and_do_not_assert_etymologies():
    rows = forms()
    by_key = {r[10]: r for r in rows}
    parents = {r[10]: r[11] for r in rows if r[11]}
    assert len(parents) == 484
    assert all(parent in by_key for parent in parents.values())
    assert all(not r[1] and not r[12] and not r[13] for r in rows)
    for child in parents:
        seen = set()
        while child in parents:
            assert child not in seen
            seen.add(child)
            child = parents[child]
    audit = articles()
    assert sum(bool(a['problems']) for a in audit) == 27
    assert collections.Counter(p for a in audit for p in a['problems']) == {
        'variant:target-or-sense-scope': 24, 'compound:parent-or-sense-scope': 3}
    assert all(v['candidates'] for a in audit for v in a['variants'] if v['resolved'])
    assert parents['nirmaan2018mewari:p251:c1:e04:v2'] == 'nirmaan2018mewari:p251:c1:e04'


def test_parser_regressions_against_positioned_font_fixtures():
    fixtures = json.loads((RAW / f'{STEM}-fixtures.json').read_text())
    parsed = {r['key']: source.parse_article(r) for r in fixtures}
    get = lambda key: parsed[f'nirmaan2018mewari:{key}']
    assert get('p004:c2:e05')['native'] == 'अक्कर'
    assert get('p004:c1:e03')['native'] == 'अइन्दे'
    assert get('p017:c1:e01')['native'] == 'आछ्यो'
    assert get('p137:c2:e01')['native'] == 'टड्डो'
    assert get('p032:c2:e02')['native'] == 'उस्यो'  # adjacent alphabet heading excluded
    assert get('p373:c2:e01')['homograph'] == '1'
    assert get('p272:c1:e05')['senses'][0]['gloss'] == ''
    for key, pos in [('p081:c2:e03','noun'), ('p127:c1:e01','verb'), ('p250:c1:e09','noun'), ('p017:c1:e05','num')]:
        assert pos in get(key)['senses'][0]['tags']
    assert get('p081:c2:e03')['senses'][0]['gloss'] == 'dog'
    assert [s['tags'] for s in get('p105:c2:e03')['senses']] == [['noun'], ['adj']]
    assert source.span_text({'font':'TimesNewRomanPSMT','text':'Dd'}) == 'Dd'
    assert source.span_text({'font':'Krishna','text':'Dd'}) == 'क्क'
    assert source.clean_native('अन्यायय अन्नीी') == 'अन्याय अन्नी'
    # Nearly identical baselines on opposite sides of a rounding boundary.
    spans = [{'x':80,'y':410.998}, {'x':42,'y':411.002}]
    assert [s['x'] for s in source.ordered_spans(spans)] == [42, 80]


def test_dialect_and_reference_are_registered_with_qualified_provenance():
    with (ROOT / 'cldf/dialects.csv').open() as f:
        dialect = next(r for r in csv.DictReader(f) if r['ID'] == 'mewari_kapasan')
    assert dialect['Language_ID'] == 'mewari_dholpura'
    assert dialect['Tag'] == source.DIALECT
    assert dialect['Quality'] == 'C' and 'not an entry-level' in dialect['Location']
    assert (float(dialect['Latitude']),float(dialect['Longitude'])) == (24.88775,74.31232)
    aliases = load_dialect_aliases()
    for r in forms():
        assert r[0] == 'mewari_dholpura' and source.DIALECT in r[14].split()
        assert normalize_dialect(r[0], r[14], aliases)[1] == r[14]
        assert r[7].startswith(source.SOURCE + '[p. ')
    entry = pybtex.database.parse_file(str(ROOT / 'cldf/sources.bib')).entries[source.SOURCE]
    assert entry.fields['year'] == '2018'
    assert len(entry.persons['editor']) == 5
    assert entry.fields['ocr'] == 'No'


def test_stable_keys_protect_all_ids_against_reordering_and_corrections_in_memory():
    rows = forms()
    initial = [dict(ID=f'tmp-{i}',Language_ID=r[0],Original=r[2],Form=r[2],Gloss=r[3],Native=r[4],Source=r[7],Status='unlinked') for i,r in enumerate(rows)]
    source_keys = {row['ID']: r[10] for row,r in zip(initial, rows)}
    first, registry = assign_ids(initial, [], source_keys)
    corrected = [dict(row, ID='new-'+row['ID'], Gloss='corrected', Form='corrected') for row in reversed(initial)]
    corrected_keys = {row['ID']: source_keys[row['ID'][4:]] for row in corrected}
    second, _ = assign_ids(corrected, registry, corrected_keys)
    assert len(set(first.values())) == 6535
    assert all(second['new-'+old] == opaque for old,opaque in first.items())


def test_final_seeded_visual_audit():
    sample = json.loads((RAW / f'{STEM}-sample.json').read_text())
    assert len(sample['entries']) == 20
    assert sample['material_errors'] == 0
    assert [r['key'] for r in random.Random(sample['seed']).sample(articles(),20)] == [r['key'] for r in sample['entries']]
    assert all(r['result'] == 'source-image-verified' for r in sample['entries'])


def test_explicit_parenthetical_grammar_and_register_do_not_remain_in_gloss():
    rows = {r[10]:r for r in forms()}
    for key,gloss,tag in [('p169:c1:e09','yours','f'), ('p169:c2:e01','yours','m'), ('p250:c1:e10','father','colloquial')]:
        row = rows['nirmaan2018mewari:'+key]
        assert row[3] == gloss and tag in row[14].split()
    # A definition of a slang expression is lexical prose, not a trailing label.
    assert 'a kind of slang' in rows['nirmaan2018mewari:p156:c2:e03'][3]
