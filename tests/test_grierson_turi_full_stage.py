"""Full-chapter Turi checks, runnable before canonical installation."""
import csv
import importlib.util
import io
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATA))
PACKAGE = DATA / 'data/other/forms/raw_data/grierson_turi_1906'
spec = importlib.util.spec_from_file_location('turi_complete', PACKAGE / 'import_source.py')
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def test_full_scope_accounting_and_reproducibility():
    rows, audit = source.build()
    assert rows == list(csv.reader((PACKAGE / 'full-review.csv').open()))
    assert Counter(a['preview_disposition'] for a in audit) == {
        'candidate_row': 291, 'attestation_reuse': 57, 'context_only': 4}
    assert Counter(a['printed_page'] for a in audit) == {128: 10, 129: 14, 130: 117, 131: 102, 133: 109}
    assert len({a['entry_key'] for a in audit}) == 352
    assert len({r[10] for r in rows}) == 291
    legacy = [json.loads(l) for l in (PACKAGE / 'audit.jsonl').read_text().splitlines()]
    assert {a['entry_key'] for a in legacy if a['status'] == 'ingested'} <= {r[10] for r in rows}


def test_dialect_and_citation_preservation():
    rows, audit = source.build()
    registry = {r['Tag']: r for r in csv.DictReader((DATA / 'cldf/dialects.csv').open())}
    bykey = {r[10]: r for r in rows}
    for tag in source.DIALECTS.values():
        assert registry[tag]['Language_ID'] == 'Turi'
        assert registry[tag]['Latitude'] == registry[tag]['Longitude'] == ''
    for a in audit:
        if not a['output_entry_key']:
            continue
        row = bykey[a['output_entry_key']]
        assert f"p. {a['printed_page']}, {a['site']}," in row[7]
        assert row[0] == 'Turi'
        assert a['site'] == 'unspecified' or source.DIALECTS[a['site']] in row[14].split()


def test_source_distinctions_and_uncertainty():
    rows, audit = source.build()
    assert {'is', 'his'} <= {r[3] for r in rows if r[2].casefold() == 'apan'}
    assert {'ọâṛ-re', 'kān-iñ-ā', 'lājet’-lid-i-ā', 'bākūnī'} <= {r[2] for r in rows}
    uncertain = [a for a in audit if a.get('typed_uncertainty')]
    assert len(uncertain) == 1
    row = next(r for r in rows if r[10] == uncertain[0]['output_entry_key'])
    assert 'uncertain' in row[14].split()
    assert all(r[1] == r[5] == r[8] == r[9] == '' for r in rows)
    assert all(r[2] == unicodedata.normalize('NFC', r[2]) for r in rows)
    assert not any('independent audit should' in r[6] for r in rows)


def test_full_profile_and_scoped_parser():
    from segments.tokenizer import Tokenizer
    import make_cldf
    rows, _ = source.build()
    tokenizer = Tokenizer(str(DATA / 'conversion/grierson-turi-1906.txt'))
    assert tokenizer('w', column='IPA') == 'v'
    assert tokenizer('ṅ', column='IPA') == 'ŋ'
    for row in rows:
        assert '�' not in tokenizer(row[2], column='IPA')
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(PACKAGE / 'full-review.csv'), errors,
                                        name='20260925-grierson-turi-sites')
    assert not errors.getvalue()
    assert len(parsed) == stats['converted'] == 291
    assert {r.old_form for r in parsed} == {r[2] for r in rows}
