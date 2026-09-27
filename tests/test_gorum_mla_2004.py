"""Full-snapshot Gorum MLA accounting and source-structure regressions."""
import collections
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from segments import Tokenizer

DATA = Path(__file__).resolve().parents[1]
ROOT = DATA / 'data/other/forms/raw_data/gorum_mla_2004'

def importer():
    spec = importlib.util.spec_from_file_location('gorum_import', ROOT / 'import_source.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_whole_snapshot_and_legacy_identities():
    module = importer()
    rows, audit = module.prepare()
    assert rows == list(csv.reader(module.CSV.open()))
    assert audit == [json.loads(s) for s in module.AUDIT.read_text().splitlines()]
    assert len(module.raw_records(module.SOURCE.read_bytes())) == 5824
    assert len(audit) == 6562 and len(rows) == 7493
    assert collections.Counter(a['status'] for a in audit) == {
        'ingested': 5694, 'same_source_reuse': 75, 'held': 46, 'excluded_context': 747}
    keys = {r[10] for r in rows}
    assert len(keys) == len(rows) and len(module.LEGACY) == 559
    assert set(module.LEGACY) <= keys
    assert all(len(r) == 15 and r[0] == 'go' and r[7].startswith('DSGO[') for r in rows)
    assert all(not r[11] or r[11] in keys for r in rows)
    assert all(not r[1] and not r[4] and not r[8] and not r[12] and not r[13] for r in rows)
    assert sum(a['line_end'] - a['line'] + 1 for a in audit) == 6564


def test_witness_senses_and_unquoted_context_are_not_cartesian_products():
    module = importer()
    rows, audit = module.prepare()
    by_key = {r[10]: r for r in rows}
    assert by_key['gorum-mla2004:2671'][3] == 'to ferry across'
    assert by_key['gorum-mla2004:2671:sense:1:head:2'][3] == 'to cross (a river)'
    assert by_key['gorum-mla2004:2701:sense:2:head:1'][3] == 'to open (of flowers)'
    assert 'gorum-mla2004:2701:sense:2:head:2' not in by_key
    tali = [r for r in rows if r[2] == 'tali' and r[10].startswith('gorum-mla2004:33670')]
    assert {r[3] for r in tali} == {'eave of the roof', 'rafter'}
    assert any(r[2] == 'suG' and r[3] == 'house' and ':unnumbered:' in r[10] for r in rows)
    assert any(r[2] == 'tam' and r[3] == 'to sneeze' and ':unnumbered:' in r[10] for r in rows)
    assert by_key['gorum-mla2004:22960'][2:4] == ['mulgi', 'nine']
    assert by_key['gorum-mla2004:6680:source-remark'][2:4] == ['bunD', 'base of a creeper']
    assert sum(len(a.get('unassigned_witness_senses', [])) for a in audit) == 6


def test_duplicate_reconciliation_preserves_collisions_and_raw_occurrences():
    module = importer()
    rows, audit = module.prepare()
    assert len(json.loads((ROOT/'duplicate-reconciliation.json').read_text())['entries']) == 81
    by_id = collections.defaultdict(list)
    for a in audit: by_id[a['source_id']].append(a)
    assert [a['status'] for a in by_id['36280']] == ['ingested', 'same_source_reuse']
    collisions = [r for r in rows if r[10].split(':')[:2] == ['gorum-mla2004', '15410']]
    assert {'bar=kaD', 'kaDaman'} <= {r[2] for r in collisions}
    assert all('uncertain' in r[14].split() for r in collisions)
    assert by_id['30120'][0]['line_end'] == by_id['30120'][0]['line'] + 1
    assert by_id['30121'][0]['line_end'] == by_id['30121'][0]['line'] + 1


def test_grammar_and_source_transcription_regressions():
    module = importer()
    rows, audit = module.prepare()
    by_key = {r[10]: r for r in rows}
    assert 'intr' in by_key['gorum-mla2004:32960'][14].split()
    assert '(intr.)' not in by_key['gorum-mla2004:32960'][3]
    assert 'suffix' in by_key['gorum-mla2004:24240'][14].split()
    assert 'verb' not in by_key['gorum-mla2004:24240'][14].split()
    assert 'loanword' in by_key['gorum-mla2004:4270'][14].split()
    assert 'loanword' in by_key['gorum-mla2004:7230'][14].split()
    assert {'verb', 'multiword-expression'} <= set(by_key['gorum-mla2004:10910'][14].split())
    assert {'2pl', 'subj'} <= set(by_key['gorum-mla2004:21310'][14].split())
    assert not {'2pl', '2sg'}.intersection(by_key['gorum-mla2004:24240'][14].split())
    assert not by_key['gorum-mla2004:4172'][5] and '[goRoG]' in by_key['gorum-mla2004:4172'][6]
    assert by_key['gorum-mla2004:4140'][3] == 'enchanted'
    assert by_key['gorum-mla2004:4140:witness:A'][3] == 'mad, insane'
    assert by_key['gorum-mla2004:8810'][5] == "DuRub'"
    assert by_key['gorum-mla2004:24670'][2:4] == ['ol', 'to lift (something light)']
    assert by_key['gorum-mla2004:3650'][3] == 'to be covered'
    assert {'tr', 'intr'} <= set(by_key['gorum-mla2004:31890'][14].split())
    assert 'tr' not in by_key['gorum-mla2004:31890:sense:1:head:2'][14].split()
    assert (ROOT/'gorum-mla.txt').read_bytes() == (DATA/'conversion/gorum-mla.txt').read_bytes()
    tokenizer = Tokenizer(str(DATA/'conversion/gorum-mla.txt'))
    for row in rows:
        result = tokenizer(row[2], column='IPA').replace(' ', '').replace('#', ' ')
        assert result == row[2] and '�' not in result


def test_scoped_parser_preserves_complete_source():
    import make_cldf
    module = importer()
    errors = io.StringIO()
    rows, stats = make_cldf.parse_file(str(module.CSV), errors, name=module.CSV.stem)
    assert len(rows) == stats['converted'] == 7493
    assert not errors.getvalue()
    assert 'same conditions' in (ROOT/'LICENSE').read_text()
    assert hashlib.sha256(module.SOURCE.read_bytes()).hexdigest() == module.SOURCE_SHA256
