"""Bounded full-chapter conservation and proposed registered-parser checks."""
import csv
import json
from pathlib import Path
import unicodedata
from pybtex.database import parse_file
from source_meta import SourceMeta
from tags import GRAMMATICAL_TAGS,GENDER_TAGS
HERE=Path(__file__).resolve().parent

def corpus():
    return ([json.loads(x) for x in (HERE/'physical-units-first-reading.jsonl').read_text().splitlines()],
            [json.loads(x) for x in (HERE/'proposal-audit.jsonl').read_text().splitlines()],
            list(csv.reader((HERE/'proposed.csv').open())))

def test_every_physical_unit_accounted_once_with_all_pages():
    units,audit,rows=corpus()
    assert len(units)==len(audit)==261
    assert {u['printed_page'] for u in units}==set(range(22,28))
    assert len({u['unit_key'] for u in units})==261
    assert {u['unit_key'] for u in units}=={a['unit_key'] for a in audit}
    assert {k for a in audit for k in a['proposed_entry_keys']}=={r[10] for r in rows}
    assert sum(a['disposition']=='audit-only' for a in audit)==16
    assert len(rows)==256 and len({r[10] for r in rows})==256
    assert sum(u['section']=='personal-ending-table' for u in units)==10
    assert sum(u['section']=='personal-pronoun-table' for u in units)==6


def test_source_notation_and_controls_not_normalized():
    u,a,r=corpus();by={x['unit_key']:x for x in u};rb={x[10]:x for x in r}
    p='nair2013irula:'
    assert by[p+'p22:plural-pronominal-marker:01']['form']=='-mn'
    assert rb[p+'p24:adjective-categories:02'][2]=='pal̲aya'
    assert rb[p+'p24:adjectival-examples:01'][2]=='al̲agaana'
    assert rb[p+'p25:negative-whole-expressions:02'][2].endswith(' all')
    assert rb[p+'p25:personal-ending-table:10'][2].endswith('\\')
    assert p+'p26:unlicensed-noun-compound:01' not in rb
    assert all(x['role']!='comparison-control' or x['unit_key'] not in rb for x in u)
    assert all(unicodedata.is_normalized('NFC',x[2]) and not x[4] and not x[5] for x in r)


def test_all_declared_grammar_tags_registered_and_scoped():
    _,_,rows=corpus();by={r[10]:r for r in rows};p='nair2013irula:'
    assert all(set(r[14].split())<=GRAMMATICAL_TAGS|GENDER_TAGS for r in rows)
    assert {'suffix','permissive'}<=set(by[p+'p26:permissive-markers:01'][14].split())
    assert {'suffix','injunctive'}<=set(by[p+'p26:injunctive-markers:01'][14].split())
    assert {'suffix','concessive'}<=set(by[p+'p27:concessive-marker:01'][14].split())
    assert {'impv','polite'}<=set(by[p+'p26:polite-imperative-examples:01'][14].split())
    assert 'honorific' not in by[p+'p26:polite-imperative-examples:01'][14].split()
    assert 'conditional' not in by[p+'p27:concessive-marker:01'][14].split()
    assert all(not r[3] for r in rows if 'suffix' in r[14].split())
    # Unlabeled columns keep explicit person, without guessed singular/plural.
    assert 'first-person' in by[p+'p25:personal-ending-table:01'][14].split()
    assert {'suffix','opt','weak'}<=set(by[p+'p26:optative-markers:01'][14].split())
    assert {'suffix','opt','strong'}<=set(by[p+'p26:optative-markers:02'][14].split())
    assert 'weak' in by[p+'p26:optative-examples:01'][14].split()
    assert 'strong' in by[p+'p26:optative-examples:02'][14].split()
    for r in rows:
        if 'zvelebil-temporal-comparison' in r[10]:
            assert set(r[14].split())=={'suffix','temporal'}
            assert 'conditioning' in r[6] and 'Zvelebil' in r[6]
    assert 'verb' not in by[p+'p23:purposive-example:01'][14].split()
    assert '2sg' not in by[p+'p26:imperative-examples:01'][14].split()
    assert all('multiword-expression' not in by[p+'p23:universal-examples:'+i][14].split() for i in ('01','03'))
    assert not {'1sg','1pl','sg','pl'} & set(by[p+'p25:personal-ending-table:01'][14].split())


def test_graph_endpoints_and_source_free_variation():
    _,_,rows=corpus();by={r[10]:r for r in rows}
    edges=[r for r in rows if r[11]]
    assert len(edges)==12
    assert by['nair2013irula:p27:conditional-markers:02'][11]=='nair2013irula:p27:conditional-markers:01'
    groups=json.loads((HERE/'same-cell-slash-graph-census.json').read_text())['groups']
    for group in groups:
        assert by[group['keys'][1]][11]==group['keys'][0]
        assert 'Source-listed' in by[group['keys'][1]][6]
    assert not by['nair2013irula:p23:proximate-demonstratives:02'][11]
    assert all(not r[11] or r[11] in by for r in rows)
    assert all(not r[12] for r in rows)
    expected=json.loads((HERE/'graph-decisions-proposal.json').read_text())['relations']
    assert {(r[10],r[13]) for r in rows if r[13]}=={(e['child'],e['parent']) for e in expected}
    assert all(not r[13] or r[13] in by for r in rows)


def test_proposed_metadata_and_actual_generic_parser(monkeypatch):
    import make_cldf
    meta=SourceMeta([HERE/'20260927-nair-irula.yaml'])
    assert set(meta.sources)=={'nair2013irula','zvelebil1982irulapart3','perialwar1978irula'}
    assert set(parse_file(str(HERE/'proposed-reference.bib')).entries)==set(meta.sources)
    monkeypatch.setattr(make_cldf.source_meta,'load',lambda:meta)
    errors=[]
    parsed,stats=make_cldf.parse_file(str(HERE/'proposed.csv'),errors,name='20260927-nair-irula')
    _,_,raw=corpus();by={r[10]:r for r in raw}
    assert len(parsed)==256 and not errors and stats['converted']==0
    assert {r.entry_key for r in parsed}==set(by)
    assert all(r.form==r.old_form==by[r.entry_key][2] and r.gloss==by[r.entry_key][3]
               and r.tags==by[r.entry_key][14] and r.source==by[r.entry_key][7]
               and r.variant_of_key==by[r.entry_key][11] and r.derivation_parent_keys==by[r.entry_key][13] and not r.ipa and not r.native
               for r in parsed)
