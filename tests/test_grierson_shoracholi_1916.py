"""Full-source Shoracholi coverage, source identity and transcription regressions."""
import csv
import importlib.util
import io
import json
from collections import Counter
from pathlib import Path
from segments.tokenizer import Tokenizer
DATA=Path(__file__).resolve().parents[1]
PACKAGE=DATA/'data/other/forms/raw_data/grierson_shoracholi_1916'
CSV=DATA/'data/other/forms/20260925-grierson-shoracholi.csv'
PROFILE=DATA/'conversion/grierson-shoracholi-1916.txt'
spec=importlib.util.spec_from_file_location('shoracholi_full',PACKAGE/'import_source.py')
source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
def installed():return list(csv.reader(CSV.open()))
def audited():return [json.loads(x) for x in (PACKAGE/'audit.jsonl').read_text().splitlines()]
def test_full_scope_and_reproduction():
    rows,audit=source.generate()
    assert rows==installed() and audit==audited()
    assert len(rows)==728 and len(audit)==683
    assert Counter(a['section'] for a in audit)=={'unusual_words':27,'table':241,'grammar':108,'specimen':307}
    assert Counter(a['status'] for a in audit)=={'ingested':598,'ingested_uncertain':3,'source_blank':3,'inventory_only_control_or_ending':8,'reused_same_source_attestation':70,'represented_by_explicit_construction':1}
    keys={r[10] for r in rows};assert len(keys)==728
    old=list(csv.DictReader((PACKAGE/'transcription.tsv').open(),delimiter='\t'))
    for c in old:
        if c['decision']=='ingest':assert f"grierson1916shoracholi:p602:item:{c['item']}" in keys
    assert 'grierson1916shoracholi:p602:item:11:answer2' in keys

def test_blanks_references_and_controls():
    rows={r[10]:r for r in installed()};audit=audited()
    assert {int(a['item']) for a in audit if a['status']=='source_blank'}=={22,174,201}
    for a in audit:
        assert all(k in rows for k in a['entry_keys']+a['reuse_entry_keys'])
        if a['status']=='reused_same_source_attestation':
            target=rows[a['reuse_entry_keys'][0]]
            assert f"p.{a['page']}, specimen7, line {a['line']}, aligned word {a['word']}" in target[7]
            assert target[2:4]==[a['printed_form_review'],a['printed_gloss']]
    head=next(a for a in audit if a['section']=='unusual_words' and a['item']=='8')
    assert not head['entry_keys'] and rows[head['reuse_entry_keys'][0]][2]=='khāyŏ chhĕkṇū'
    assert all(not a['entry_keys'] for a in audit if a['status']=='inventory_only_control_or_ending')

def test_source_literal_corrections_and_variants():
    rows={r[10]:r for r in installed()};s='grierson1916shoracholi:'
    assert rows[s+'p602:item:11:answer2'][2]=='gŏhr'
    assert rows[s+'p602:item:19'][2]=='ōr-dēṇū'
    assert rows[s+'p633:item:53'][2]=='Boṭī'
    assert rows[s+'p641:item:169'][2]=='Ōṇā'
    assert rows[s+'p641:item:177'][2]=='Piṭda'
    assert rows[s+'p645:item:223'][2]=='Tērē bābū-rē kētṭē chhaṅgṭū āsā?'
    assert rows[s+'p639:item:157:variant:2'][2]=='Tū sŏ'
    assert rows[s+'p639:item:156:variant:2'][2]=='Aū̃ āsū sū'
    assert 'uncertain' in rows[s+'p633:item:75'][14].split()
    assert 'uncertain' in rows[s+'p637:item:130'][14].split()
    assert {'verb','2pl','pret'}<=set(rows[s+'p645:item:215'][14].split())
    assert rows[s+'p603:grammar:dem-prox-sg-agent:variant:2'][2]=='ēṇe'

def test_profile_and_scoped_parse():
    import make_cldf,profile_policy,source_meta,tags
    assert source_meta.SourceMeta().transcription(source.SOURCE,CSV,'Shoracholi')[0]=='grierson-shoracholi-1916'
    assert 'grierson-shoracholi-1916' not in profile_policy.audit({})
    tok=Tokenizer(str(PROFILE))
    for r in installed():
        assert len(r)==15 and not r[13]
        assert '�' not in tok(r[2],column='IPA')
        assert set(r[14].split())<=tags.GRAMMATICAL_TAGS|tags.GENDER_TAGS
    errors=io.StringIO();parsed,stats=make_cldf.parse_file(str(CSV),errors,name='20260925-grierson-shoracholi')
    assert not errors.getvalue()
    assert len(parsed)==stats['converted']==728
    raw={r[10]:r for r in installed()}
    assert all(r.old_form==raw[r.entry_key][2] for r in parsed)
