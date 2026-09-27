"""Whole-source proposal invariants; no database construction."""
import csv
import json
from pathlib import Path

P=Path(__file__).resolve().parents[1]/'data/other/forms/raw_data/grierson_kaikadi_1906'

def inventory():
    return list(csv.reader((P/'proposal.csv').open())), [json.loads(x) for x in (P/'proposal-audit.jsonl').read_text().splitlines()]

def test_whole_source_scope_and_legacy_keys():
    rows,audit=inventory()
    assert len(audit)==1087 and len({a['source_unit_key'] for a in audit})==1087
    table=[a for a in audit if a['section'].startswith('standard')]
    assert {a['prompt'] for a in table}==set(range(1,242))
    assert sum(not a['entry_keys'] for a in table)==17
    assert all(a['entry_keys'] for a in table if a['prompt']>=220)
    assert sum(a['section'].startswith('specimen') for a in audit)==706
    assert all(not a['entry_keys'] for a in audit if a['scope']!='target')
    legacy=set(json.loads((P/'legacy164-preservation-manifest.json').read_text())['entry_keys'])
    assert len(legacy)==164 and legacy<={r[10] for r in rows}

def test_source_prompt_and_literal_regressions():
    rows,audit=inventory();r={x[10]:x for x in rows}
    t={a['prompt']:a for a in audit if a['section'].startswith('standard')}
    assert t[100]['gloss']=='Alas'
    assert 'rupee' in t[234]['gloss'].lower() and 'those' in t[235]['gloss'].lower()
    assert t[90]['forms']==['Païlī']
    assert t[195]['entry_keys'][0]=='grierson1906lsi4:kaikadi_sholapur:195'
    assert len(t[195]['entry_keys'])==2
    for n in range(185,191):assert 'pret' in r[t[n]['entry_keys'][0]][14].split()
    assert 'ātuṅgrik' in {x[2] for x in rows} and 'ātungrik' in {x[2] for x in rows}

def test_exact_reuse_and_graph_targets_are_closed():
    rows,audit=inventory();by_key={r[10]:r for r in rows}
    assert len(by_key)==len(rows)
    assert all(len(r)==15 for r in rows)
    for r in rows:
        if r[11]:
            assert r[11] in by_key and r[11]!=r[10]
            assert by_key[r[11]][0]==r[0] and by_key[r[11]][3]==r[3]
    for a in audit:
        if a.get('exact_attestation_reuse'):
            assert all(k in by_key for k in a['entry_keys'])
    assert all(not r[4] for r in rows)  # Native explicitly not printed.

def test_literal_profile_and_registered_grammar_coverage():
    from segments import Tokenizer
    from tags import GRAMMATICAL_TAGS,GENDER_TAGS
    rows,_=inventory();tokenizer=Tokenizer(str(P/'proposal-profile.txt'))
    for r in rows:
        assert '�' not in tokenizer(r[2],column='IPA')
        assert set(r[14].split())-{x for x in r[14].split() if x.startswith('dialect:')} <= GRAMMATICAL_TAGS|GENDER_TAGS
    import unicodedata
    literal=unicodedata.normalize('NFC','ṯs̱')
    assert tokenizer(literal,column='IPA').replace(' ','')==literal
    assert tokenizer('Païlī',column='IPA').replace(' ','')=='païlī'


def test_scoped_parser_with_proposed_metadata(monkeypatch):
    import io
    import make_cldf,source_meta
    from segments import Tokenizer
    key='grierson-kaikadi-1906'
    meta=source_meta.SourceMeta([P/'20260925-grierson-kaikadi.yaml'])
    monkeypatch.setattr(source_meta,'load',lambda:meta)
    monkeypatch.setitem(make_cldf.convertors,key,Tokenizer(str(P/'proposal-profile.txt')))
    errors=io.StringIO()
    parsed,stats=make_cldf.parse_file(str(P/'proposal.csv'),errors,name='20260925-grierson-kaikadi')
    rows,_=inventory()
    assert len(parsed)==stats['converted']==len(rows)
    assert not errors.getvalue()
    assert {r.entry_key:r.old_form for r in parsed}=={r[10]:r[2]for r in rows}
    assert all(not r.ipa for r in parsed)
    tokenizer=make_cldf.convertors[key]
    assert {r.entry_key:r.form for r in parsed}=={r[10]:tokenizer(r[2],column='IPA').replace(' ','').replace('#',' ') for r in rows}



def test_explicit_grammar_labels_and_qualified_gender():
    rows,audit=inventory();by_key={r[10]:r for r in rows}
    grammar={a['source_unit_key'].rsplit(':',1)[-1]:a for a in audit if a['section']=='grammar'}
    for label in ['who','what','who-neuter','whose']:
        assert 'interr' in by_key[grammar[label]['entry_keys'][0]][14].split()
    assert 'gen' in by_key[grammar['whose']['entry_keys'][0]][14].split()
    assert 'n' in by_key[grammar['who-neuter']['entry_keys'][0]][14].split()
    good=by_key[grammar['good-woman']['entry_keys'][0]]
    assert 'f' not in good[14].split() and 'tentatively' in good[6]


def test_interlinear_negative_verbs_and_passive_scope():
    rows,_=inventory()
    by_gloss={r[3].lower():set(r[14].split())for r in rows}
    for gloss in ['telling-not','allowed-not']:
        assert {'verb','neg'}<=by_gloss[gloss]
    assert {'verb','pret'}<=by_gloss['anger-came']
    for gloss in ['is-found','was-met','were-let','had-been-lost']:
        assert 'pass' in by_gloss[gloss]
    assert 'pl' not in by_gloss['sons']  # English lexical plural is not a source suffix analysis.


def test_later_addendum_preserves_original_witnesses_and_uncertainty():
    rows,audit=inventory();by_key={r[10]:r for r in rows}
    late=[a for a in audit if a['section']=='later-addendum']
    assert len(late)==8 and sum(bool(a['entry_keys']) for a in late)==4
    for a in late:
        original=by_key[a['linked_original_entry_key']]
        assert original[2] in a['original_edition_forms']
        assert 'Later bound-in' in original[6]
        if a['target_prompt']==95:
            assert not a['entry_keys'] and 'ā/ă' in a['uncertainty']
        if a['entry_keys']:
            corrected=by_key[a['entry_keys'][0]]
            assert corrected[2]==('nāy' if a['target_prompt'] in [146,147] else 'nāyāṅg')
            assert not corrected[11]  # A correction is not an asserted lexical variant.
            assert corrected[7].startswith('grierson_addenda_minora_iv_boundin[')
