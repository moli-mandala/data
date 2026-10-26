"""Bounded whole-source proposal checks; no compiled database required."""
import csv,json,unicodedata
from pathlib import Path
from segments import Tokenizer
from tags import GRAMMATICAL_TAGS,GENDER_TAGS
P=Path(__file__).resolve().parents[1]/'data/other/forms/raw_data/bailey_kagani_1920/whole-source-recovery'
O=P/'integration-working'
def rows():return list(csv.reader((O/'proposal.csv').open()))
def audit():return [json.loads(s) for s in (O/'proposal-audit.jsonl').read_text().splitlines()]

def test_source_accounting_and_legacy_identity():
    data=rows();units=audit();bykey={r[10]:r for r in data}
    base=json.loads((P/'legacy-baseline-20260927.json').read_text())
    assert len(data)==1306 and len(bykey)==len(data)
    assert [r[10] for r in data[:329]]==base['entry_keys']
    emitted=[k for u in units for k in u.get('entry_keys',[])]
    assert set(emitted)==set(bykey)
    for u in units:
        if u.get('status')=='source_context_control':assert not u['entry_keys']
    target='bailey1920kagani:p90:compound-verbs:leave-1'
    assert target not in bykey
    control=next(u for u in units if u.get('source_unit_key')==target)
    assert control['input']['forms']==['chăḍḍṇā'] and control['input']['source_language']=='Panjabi'

def test_narrative_scope_is_contiguous_and_parentheticals_are_distinct():
    source=[json.loads(s) for s in(P/'parable-aligned-reviewed.jsonl').read_text().splitlines()]
    spans=json.loads((P/'parable-expression-spans-reviewed.json').read_text())
    assert len(source)==362 and len(spans)==16
    assert [k for span in spans for k in span['primary_token_keys']]==[r['source_unit_key'] for r in source]
    bykey={r[10]:r for r in rows()}
    extras=json.loads((P/'parable-parentheticals-reviewed.json').read_text())
    assert len(extras)==6
    for extra in extras:
        out=bykey[extra['source_unit_key']]
        assert out[2]==extra['raw_form'] and 'parenthetical after word' in out[7]
        if extra['raw_alignment'] in {'pigs','prophet'}:assert not out[11]
    for span in spans:
        out=bykey[span['source_unit_key']]
        assert ' through ' in out[7]
        for page in span['printed_pages']:
            assert f'p. {page}' in out[7] or len(span['printed_pages'])>2


def test_interlinear_grammar_and_tone_marks_are_preserved():
    units=audit();bykey={r[10]:r for r in rows()}
    future=[u for u in units if u.get('input',{}).get('raw_alignment')=='will-go']
    assert future
    for unit in future:
        out=bykey[unit['entry_keys'][0]]
        assert out[3]=='go' and {'verb','fut'}<=set(out[14].split())
        assert 'will-go' in out[6]
    assert bykey['bailey1920kagani:p89:tone:be-found-pronounced'][2]=='thē̩ṇā'
    pronounced=[r for r in bykey.values() if r[2]=='thē̩nā']
    assert pronounced, 'The plain-n occurrence on p99 must not be harmonized with p89'
    assert bykey['bailey1920kagani:p90:causatives:do'][2]=='kărna'
    assert bykey['bailey1920kagani:p90:habit:do'][2]=='kărnā'


def test_profiles_registered_tags_and_graph_closure():
    converter=Tokenizer(str(O/'profile-proposal.txt'));data=rows();keys={r[10] for r in data}
    for row in data:
        assert len(row)==15
        assert {t for t in row[14].split() if not t.startswith('dialect:')}<=GRAMMATICAL_TAGS|GENDER_TAGS
        a=converter(unicodedata.normalize('NFC',row[2]),column='IPA')
        b=converter(unicodedata.normalize('NFD',row[2]),column='IPA')
        assert '\ufffd' not in a+b
        assert unicodedata.normalize('NFC',a)==unicodedata.normalize('NFC',b)
        assert not row[11] or row[11] in keys
        assert not row[12] and not row[13]
