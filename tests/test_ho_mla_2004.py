"""Focused source-stage checks for the bounded Ho MLA/Deeney slice."""
import collections
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

from segments import Tokenizer

ROOT = Path(__file__).resolve().parents[1] / "data/other/forms/raw_data/ho_mla_2004"
DATA = ROOT.parents[4]


def importer():
    spec = importlib.util.spec_from_file_location("ho_mla_import", ROOT / "import_source.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_physical_scope_and_stable_keys():
    rows, audit = importer().prepare()
    assert len(rows) == 2227 and len(audit) == 1524
    assert sum(a['status'] != 'ingested' for a in audit) == 8
    assert [a['source_id'] for a in audit if a['source_id'].isdigit()] == [f'{n:05d}' for n in range(1,1517)]
    assert rows == list(csv.reader((ROOT/'proposal.csv').open()))
    assert len({r[10] for r in rows}) == len(rows)
    keys = {r[10] for r in rows}
    assert all(not r[11] or r[11] in keys for r in rows)
    assert rows == list(csv.reader((DATA/'data/other/forms/20260925-donegan-stampe-ho.csv').open()))
    assert audit == [json.loads(l) for l in (ROOT/'audit.jsonl').read_text().splitlines()]
    final = json.loads((ROOT/'root-expression-final-audit-20260926.json').read_text())
    assert final['material_errors'] == 0 and final['sample_count'] == 20
    repair = json.loads((ROOT/'release-citation-serialization-repair-20260926.json').read_text())
    assert final['sha256']['expression-recovery-proposed.csv'] == repair['before_sha256']['data/data/other/forms/raw_data/ho_mla_2004/proposal.csv']
    assert repair['after_sha256']['data/data/other/forms/raw_data/ho_mla_2004/proposal.csv'] == hashlib.sha256((ROOT/'proposal.csv').read_bytes()).hexdigest()
    byid = {a['source_id']:a for a in audit}
    assert byid['00914']['line_end'] == byid['00914']['body_line'] + 1
    assert all(byid[k]['status'] == 'ingested' for k in ('00757','01035','01362'))
    assert all(byid[k]['status'] == 'excluded_control' for k in ('00870','01254'))


def test_alias_locality_grammar_and_crossref_boundaries():
    rows, _ = importer().prepare()
    bykey = {r[10]:r for r in rows}
    get = lambda suffix: bykey['ho-mla2004:'+suffix]
    assert get('01231:alias:2')[2:4] == ['bukur-liyaG','a bird, the Tree Pie']
    assert 'dialect:ho:mla-noamundi' not in get('00678')[14]
    assert 'dialect:ho:mla-noamundi' in get('00678:sense:2')[14]
    assert 'dialect:ho:mla-south-singhbhum' not in get('01410')[14]
    assert 'dialect:ho:mla-south-singhbhum' in get('01410:sense:2')[14]
    assert 'refl' in get('01416')[14].split()
    assert get('00746:supplement:1')[3] == 'it no longer exists'
    assert get('00969')[3].startswith('to move across water')
    assert 'a straw rope' in get('01240')[3]
    assert 'Greyheaded Flycatcher' in get('01473')[3]
    assert 'reciprocal' in get('00750')[14].split()
    assert get('00750')[13] == 'ho-mla2004:00534:sense:2'
    assert get('00280')[3] == 'father' and 'voc' in get('00280')[14].split()
    assert get('00612')[3] == 'garden'
    assert 'pejorative' in get('01338')[14].split()
    assert 'pejorative' not in get('01100')[14].split()
    assert 'pejorative' in get('01100:sense:2')[14].split()
    assert 'proper-noun' in get('00675')[14].split()
    assert len(get('00675:supplement:1')[14].split()) == len(set(get('00675:supplement:1')[14].split()))
    assert 'part' in get('00742')[14].split()
    assert get('01243:supplement:1')[2] == 'Dol-Dol'
    assert get('01243:supplement:1')[11] == ''
    assert 'loanword' in get('00595')[14].split()
    assert 'poetic' not in get('00811')[14].split()
    assert 'poetic' in get('00811:sense:2')[14].split()
    assert get('00619')[3] == 'to burn a hole into e.g. a flute'
    assert 'poetic' not in get('00619')[14].split()
    assert get('00619:sense:2')[3] == 'to burn' and 'poetic' in get('00619:sense:2')[14].split()
    assert 'collective' in get('00703')[14].split()
    assert 'refl' not in get('00087')[14].split()
    assert 'refl' in get('00087:sense:2')[14].split()
    assert get('00279:supplement:1')[2:4] == ['honyar','father-in-law']
    assert get('00730:supplement:1')[2:4] == ['gaya','a eunuch']
    assert 'voc' in get('01301')[14].split()
    assert 'pl' in get('01056:supplement:1')[14].split()
    assert get('01046')[3] == 'the barking deer (muntjak)'
    assert get('01040')[3] == '' and 'uncertain' in get('01040')[14].split()
    assert get('00913')[3] == '' and 'uncertain' in get('00913')[14].split()
    assert get('00775')[3] == 'a flower' and 'poetic' in get('00775')[14].split()
    assert get('00346')[3] == 'tar'
    assert get('00912')[3] == 'unconscious'
    assert get('00912:supplement:1')[2] == 'goe:-anjeD'
    assert 'pl' not in get('00323:supplement:2')[14].split()
    assert 'refl' in get('01193')[14].split() and 'refl.' not in get('01193')[3]
    assert 'reciprocal' not in get('00750:sense:2')[14].split()
    assert get('00308:supplement:1')[2:4] == ['asul-maraG','to raise children']
    assert len([r for r in rows if r[10].startswith('ho-mla2004:01390:supplement:')]) == 5


def test_literal_full_source_profile():
    rows, _ = importer().prepare()
    assert (ROOT/'ho-mla.txt').read_bytes() == (DATA/'conversion/ho-mla.txt').read_bytes()
    tokenizer = Tokenizer(str(DATA/'conversion/ho-mla.txt'))
    for row in rows:
        converted = tokenizer(row[2], column='IPA').replace(' ', '').replace('#', ' ')
        assert converted == ' '.join(row[2].replace('w','v').split())


def test_source_notice_and_uncertainty():
    rows, audit = importer().prepare()
    assert 'same conditions' in (ROOT/'LICENSE').read_text()
    assert (DATA/'cldf/sources.bib').read_text().count('@book{DHED,') == 1
    assert any(r[2].startswith('ch') for r in rows)
    assert all(x.get('review_reasons') for a in audit for x in a['rows'] if 'uncertain' in x['tags'].split())


def test_archived_reference_prose_cannot_create_phantom_sources():
    import re
    rows, audit = importer().prepare()
    assert all(re.fullmatch(r"DHED\[[^;\[\]]+\]", row[7]) for row in rows)
    old = {r[10]: r for r in csv.reader((ROOT/'legacy-before-expression-recovery.csv').open())}
    moved = 0
    for row in rows:
        original = old.get(row[10])
        if original and ', archive refs ' in original[7]:
            locator, prose = original[7].split(', archive refs ', 1)
            assert row[7] == locator + ']'
            assert 'Archived reference prose: ' + prose[:-1] in row[6]
            moved += 1
    assert moved == 93
    row = next(r for r in rows if r[10] == 'ho-mla2004:00185')
    assert row[7] == 'DHED[archive entry 00185]'
    assert "B. 51/6; H. 51/8/1.  Cf. <ambaRa> `hog plum fruit'" in row[6]
    assert next(a for a in audit if a['source_id'] == '00185')['rows'][0]['citation'] == row[7]


def test_multihead_archived_references_do_not_accumulate():
    module = importer()
    covered = 0
    for record in module.records():
        rows, _ = module.parse(record)
        annotated = [row for row in rows if 'Archived reference prose: ' in row[6]]
        for row in annotated:
            assert row[6].count('Archived reference prose: ') == 1
        if len(annotated) > 1:
            covered += 1
    assert covered > 0
