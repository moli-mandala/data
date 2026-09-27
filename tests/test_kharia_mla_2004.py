"""Complete Kharia MLA snapshot: scope, semantics, identity and transcription."""
import collections
import csv
import importlib.util
import io
import json
from pathlib import Path
from segments import Tokenizer

DATA = Path(__file__).resolve().parents[1]
ROOT = DATA/'data/other/forms/raw_data/kharia_mla_2004'


def importer():
    spec = importlib.util.spec_from_file_location('kharia_full', ROOT/'import_source.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_complete_snapshot_and_canonical_identity():
    module = importer()
    rows, audit = module.prepare()
    assert rows == list(csv.reader(module.CSV.open()))
    assert audit == [json.loads(s) for s in module.AUDIT.read_text().splitlines()]
    assert len(audit) == 3631 and len(rows) == 5440
    assert collections.Counter(a['status'] for a in audit) == {'ingested':3623, 'excluded_context':8}
    assert len({r[10] for r in rows}) == len(rows)
    assert len(module.LEGACY) == 829 and set(module.LEGACY) <= {r[10] for r in rows}
    assert sum(a['line_end']-a['line']+1 for a in audit) == 3632
    assert len({sid for a in audit for sid in a['source_ids']}) == 3621
    assert all(len(r)==15 and r[0]=='kh' and r[7].startswith('DSKH[') for r in rows)
    assert all(not r[1] and not r[4] and not r[5] and not r[12] for r in rows)
    assert sum(not r[3] for r in rows) == 91
    assert all('uncertain' in r[14].split() for r in rows if not r[3])


def test_witness_scopes_and_malformed_source_recovery():
    rows, audit = importer().prepare()
    bykey = {r[10]:r for r in rows}
    byid = {a['source_id']:a for a in audit if a['source_id']}
    assert bykey['kharia-mla2004:1741'][3] == 'bull'
    assert bykey['kharia-mla2004:1741:witness:B'][3] == 'male calf'
    assert bykey['kharia-mla2004:4181'][3] == 'brother-in-law'
    assert bykey['kharia-mla2004:4181:witness:D'][3] == "brother-in-law (wife's)"
    assert bykey['kharia-mla2004:9321'][2:4] == ['o-eG-goD','to turn, to return, to give back']
    assert byid['7970']['source_ids'] == ['7970','26602']
    assert bykey['kharia-mla2004:3921'][2] == 'jaRi-bo-te'
    assert byid['3921']['grouping_context'] == '<bo>'
    assert bykey['kharia-mla2004:29651:sense:1:head:3'][2] == 'su?trom'
    assert bykey['kharia-mla2004:101:sense:2:head:1'][3] == 'inside'
    assert bykey['kharia-mla2004:15761:sense:2:head:1'][3] == 'to cut, to cut the hair'
    assert bykey['kharia-mla2004:12652'][3] == "yesterday's"
    assert bykey['kharia-mla2004:8042'][3] == 'the Sabbath, day of rest'
    assert bykey['kharia-mla2004:28271'][3] == 'to straighten'
    assert 'interj' in bykey['kharia-mla2004:26841'][14].split()
    assert 'interr' in bykey['kharia-mla2004:12492'][14].split()
    assert bykey['kharia-mla2004:3213'][9].endswith("`time'.")
    assert bykey['kharia-mla2004:5521'][9].endswith("`id.'.")
    assert all('intr' in r[14].split() for r in rows if r[2] == 'bhule')
    assert 'bound form' in bykey['kharia-mla2004:14521'][8]
    assert 'echo' in bykey['kharia-mla2004:24591'][14].split()
    assert 'proper-noun' not in bykey['kharia-mla2004:670'][14].split()
    assert 'NP' in bykey['kharia-mla2004:670'][9]
    assert 'proper-noun' in bykey['kharia-mla2004:4811'][14].split()
    assert 'pejorative' in bykey['kharia-mla2004:11761'][14].split()
    assert bykey['kharia-mla2004:11761'][3] == 'get!, begone, away'
    assert 'addressed to dogs' in bykey['kharia-mla2004:11761'][8]
    assert bykey['kharia-mla2004:22531'][3] == 'future tense'
    assert 'fut' in bykey['kharia-mla2004:22531'][14].split()
    assert 'pret' in bykey['kharia-mla2004:16781'][14].split()
    assert 'impv' not in bykey['kharia-mla2004:16781'][14].split()
    assert 'reciprocal' in bykey['kharia-mla2004:17101'][14].split()
    assert 'auxiliary' in bykey['kharia-mla2004:18901'][14].split()
    assert {'nom','acc','emph','interr'} <= set(bykey['kharia-mla2004:401'][14].split())
    assert 'postp' in bykey['kharia-mla2004:23390'][14].split()
    assert 'kinship' in bykey['kharia-mla2004:20731'][14].split()
    assert 'pejorative' not in bykey['kharia-mla2004:26781'][14].split()
    assert 'pejorative' not in bykey['kharia-mla2004:26781:witness:B'][14].split()
    assert 'pejorative' in bykey['kharia-mla2004:26781:witness:D'][14].split()
    assert 'used only in songs' in bykey['kharia-mla2004:21731'][8]
    assert 'neg' not in bykey['kharia-mla2004:12470'][14].split()
    assert not any('|' in r[3] or '``' in r[3] for r in rows)


def test_source_claims_and_existing_attestation_reuse():
    rows, audit = importer().prepare()
    bykey = {r[10]:r for r in rows}
    keys = set(bykey)
    assert all(not r[11] or r[11] in keys for r in rows)
    assert all(not r[13] or r[13] in keys for r in rows)
    assert sum(bool(r[13]) for r in rows) == 74
    assert bykey['kharia-mla2004:27061:causative:1'][2:4] == ["ro[b]jo'D", 'to make sour']
    assert sum(len(a.get('causative_mentions', [])) for a in audit) == 38
    assert sum(len(a.get('supplementary_rows', [])) for a in audit) == 25
    assert any(r[2:4] == ['jhalob', 'to be stretched'] for r in rows)
    assert any(r[2:4] == ['gam Oy~EG', 'to answer'] for r in rows)
    assert not any(r[2] == 'abha:ga:' for r in rows)
    assert not bykey['kharia-mla2004:1281'][13]  # source wide vs candidate small conflict
    assert not bykey['kharia-mla2004:1312'][13]  # blank parent gloss cannot verify semantics
    assert bykey['kharia-mla2004:12711:sense:2:head:1'][3] == 'to defecate'
    assert 'excrement' in bykey['kharia-mla2004:12711'][3]
    assert bykey['kharia-mla2004:10041:sense:1:head:2'][2] == 'ghOl'
    assert 'shave' in bykey['kharia-mla2004:32441:sense:2:head:1'][3]
    align = json.loads((ROOT/'same-source-reuse-alignment.json').read_text())
    assert len(align['alignments']) == 6 and len({x['source_id'] for x in align['alignments']}) == 5
    assert sum(bool(a.get('prior_same_source_attestations')) for a in audit) == 5
    assert 'emph' in bykey['kharia-mla2004:22010'][14].split()
    assert 'pl' in bykey['kharia-mla2004:242'][14].split()
    assert 'inclusive' in bykey['kharia-mla2004:721'][14].split()
    assert {'adj','verb'} <= set(bykey['kharia-mla2004:431'][14].split())


def test_complete_source_profile_and_scoped_parser():
    module = importer()
    rows, _ = module.prepare()
    assert (ROOT/'kharia-mla.txt').read_bytes() == (DATA/'conversion/kharia-mla.txt').read_bytes()
    tokenizer = Tokenizer(str(DATA/'conversion/kharia-mla.txt'))
    for row in rows:
        converted = tokenizer(row[2],column='IPA').replace(' ','').replace('#',' ')
        assert converted == row[2].replace('w','v') and '�' not in converted
    import make_cldf
    errors=io.StringIO()
    parsed,stats=make_cldf.parse_file(str(module.CSV),errors,name=module.CSV.stem)
    assert len(parsed) == stats['converted'] == 5440 and not errors.getvalue()
    assert 'same conditions' in (ROOT/'LICENSE').read_text()
