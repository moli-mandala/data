"""Whole original-unit accounting and repairs to hidden forward omissions."""
import csv
import io
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1]
P = DATA / 'data/other/forms/raw_data/norton_korku_1884'


def inventory():
    rows = list(csv.reader((P/'full-recovery-proposal.csv').open()))
    audit = [json.loads(s) for s in (P/'full-recovery-proposal-audit.jsonl').read_text().splitlines()]
    return rows, audit


def test_whole_source_mapping_and_legacy_keys():
    rows, audit = inventory()
    rr = {r[10]: r for r in rows}
    assert len(rows) == len(rr) == 1076
    assert len(audit) == len({a['entry_key'] for a in audit}) == 958
    old = list(csv.reader((P/'expression-proposal.csv').open()))
    assert {r[10] for r in old} <= rr.keys()
    assert all(len(r) == 15 and r[2] for r in rows)
    for a in audit:
        assert set(a['entry_keys']) <= rr.keys()
        assert bool(a['entry_keys']) == (a['status'] not in {'excluded_control','audit_only_metalinguistic'})
    assert sum(a.get('full_cell_reviewed', False) for a in audit) == 477


def test_literal_reverse_reuse_and_original_blanks():
    rows, audit = inventory()
    rr = {r[10]: r for r in rows}
    for a in audit:
        if ':kor-english:' not in a['entry_key']:
            continue
        actual = sorted(rr[k][2].casefold() for k in a['entry_keys'])
        expected = sorted(s.strip().casefold() for s in a['printed_response'].split('|'))
        assert actual == expected, a['entry_key']
    assert rr['cust1884korku:english-kor:p170:right:item10'][2] == 'bīn'
    assert rr['cust1884korku:kor-english:p172:right:item23'][2] == 'bin'
    assert rr['cust1884korku:kor-english:p174:right:item24'][3] == ''
    assert rr['cust1884korku:kor-english:p174:right:item35'][3] == 'he'
    assert rr['cust1884korku:kor-english:p175:left:item03'][3] == 'grain'


def test_all_printed_alternatives_and_scoped_qualifiers():
    rows, audit = inventory()
    byunit = {a['entry_key']: a for a in audit}
    rr = {r[10]: r for r in rows}
    def forms(key):
        return {rr[k][2]: rr[k] for k in byunit[key]['entry_keys']}
    sing = forms('cust1884korku:english-kor:p170:left:item29')
    assert set(sing) == {'sirī,ē','sisīringba','sīsīringba','sisīringken','sīrīen'}
    assert 'uncertain' in sing['sīrīen'][14].split()
    assert 'uncertain' not in sing['sīsīringba'][14].split()
    assert 'sūsūrū' in forms('cust1884korku:english-kor:p165:right:item14')
    assert 'bīan, daien' in forms('cust1884korku:english-kor:p169:left:item03')
    assert {'tūlē','bīdē'} == set(forms('cust1884korku:english-kor:p169:right:item27'))
    assert 'tamāsha' in forms('cust1884korku:english-kor:p170:left:item26')
    careful = forms('cust1884korku:english-kor:p166:left:item12')['khabardār']
    assert 'loanword' not in careful[14].split()
    colour = forms('cust1884korku:english-kor:p166:left:item27')['rango']
    assert 'loanword' in colour[14].split()
    assert 'kētkībā' in forms('cust1884korku:english-kor:p167:left:item17')
    assert 'kētkība' not in forms('cust1884korku:english-kor:p167:left:item17')
    for row in rows:
        assert len(row[14].split()) == len(set(row[14].split()))
        if ':english-kor:' in row[10] and 'impv' in row[14].split():
            assert 'pres' in row[14].split()


def test_tag_completion_keeps_all_lexical_fields_identical():
    rows, audit = inventory()
    old = list(csv.reader((P/'before-tag-completion-full-recovery-proposal.csv').open()))
    assert len(rows) == len(old)
    assert all(a[:14] == b[:14] for a,b in zip(rows,old))
    assert sum(a[14] != b[14] for a,b in zip(rows,old)) == 110
    before = [json.loads(s) for s in (P/'before-tag-completion-full-recovery-proposal-audit.jsonl').read_text().splitlines()]
    assert [{k:v for k,v in a.items() if k != 'grammatical_tag_completion'} for a in audit] == before


def test_actual_parser_preserves_all_rows_including_blank_gloss(monkeypatch):
    import make_cldf
    from segments import Tokenizer
    from tags import GRAMMATICAL_TAGS, GENDER_TAGS
    rows, _ = inventory()
    tokenizer = Tokenizer(str(P/'full-recovery-profile.txt'))
    monkeypatch.setitem(make_cldf.convertors, 'cust-norton-korku-1884', tokenizer)
    for row in rows:
        assert '�' not in tokenizer(row[2], column='IPA')
        assert {t for t in row[14].split() if not t.startswith('dialect:')} <= GRAMMATICAL_TAGS | GENDER_TAGS
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(P/'full-recovery-proposal.csv'), errors, name='20260925-cust-norton-korku')
    assert len(parsed) == stats['converted'] == 1076
    assert not errors.getvalue()
    assert {r.entry_key:r.old_form for r in parsed} == {r[10]:r[2] for r in rows}
