"""Complete Sodochi source scope, lect attribution, source forms and profile."""
import csv,importlib.util,io,json
from collections import Counter
from pathlib import Path
from segments.tokenizer import Tokenizer
DATA=Path(__file__).resolve().parents[1]
PACKAGE=DATA/'data/other/forms/raw_data/grierson_sodochi_1916'
CSV=DATA/'data/other/forms/20260925-grierson-sodochi.csv'
PROFILE=DATA/'conversion/grierson-sodochi-1916.txt'
spec=importlib.util.spec_from_file_location('sodochi_source',PACKAGE/'import_source.py');source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
def test_full_scope_and_reproduction():
    rows,audit=source.generate()
    assert rows==list(csv.reader(CSV.open()))
    assert audit==[json.loads(x) for x in (PACKAGE/'audit.jsonl').read_text().splitlines()]
    assert len(rows)==1146 and len(audit)==1034
    assert Counter(a['section'] for a in audit)=={'table':241,'glossary':187,'grammar':275,'specimen':331}
    keys={r[10] for r in rows};assert len(keys)==len(rows)
    for c in csv.DictReader((PACKAGE/'transcription.tsv').open(),delimiter='\t'):
        if c['decision']=='ingest':assert f"grierson1916sodochi:p663:item:{c['item']}" in keys
    assert 'grierson1916sodochi:p663:item:6:answer2' in keys
    assert {a['item'] for a in audit if a['status']=='source_blank'}=={'174','201'}
    assert all(k in keys for a in audit for k in a['entry_keys']+a['reuse_entry_keys'])
def test_lect_scope_crossreferences_and_reuse():
    output,audit=source.generate();rows={r[10]:r for r in output}
    for a in audit:
        for k in a['entry_keys']:
            r=rows[k]
            if a['status']=='ingested_lect_uncertain':
                assert r[0]=='Kotguru' and 'uncertain' in r[14].split() and 'dialect:' not in r[14]
            if a['status']=='ingested_explicit_OS' or a.get('disposition')=='outer_siraji_control':assert r[0]=='OuterSiraji' and 'dialect:' not in r[14]
            if a.get('disposition')=='inner_siraji_control':assert r[0]=='insir'
        for k in a['reuse_entry_keys']:
            assert rows[k][2:4]==[a['printed_form_review'],a['printed_gloss']]
            assert f"line {a['line']}, aligned word {a['word']}" in rows[k][7]
    refs=[a for a in audit if a.get('resolved_reference')];assert len(refs)==6
    futures=[r for r in output if r[10].endswith('-OS-future') or '-OS-future:answer' in r[10]]
    assert len(futures)==21 and all(r[0]=='OuterSiraji' and 'fut' in r[14].split() for r in futures)
    assert all('mū' not in r[2] and 'mē' not in r[2] for r in futures)
def test_literal_forms_and_scoped_senses():
    rows={r[10]:r for r in source.generate()[0]};s='grierson1916sodochi:'
    assert rows[s+'p666:item:171'][2]=='Auĕŏ'
    assert rows[s+'p666:item:172'][2]=='Mū̃ auū'
    assert rows[s+'p663:item:50'][3]=='elder sister'
    assert rows[s+'p663:item:50:answer2'][3]=='younger sister'
    assert rows[s+'p664:item:82'][2]=='Khŏṛō, au'
    assert 'uncertain' in rows[s+'p664:item:82'][14].split()
    assert rows[s+'p653:grammar:noun-table-elephant-gen'][2]=='bāthīau'
    assert 'uncertain' in rows[s+'p653:grammar:noun-table-elephant-gen'][14].split()
    assert rows[s+'p657:grammar:continuative-Bailey'][3]=='I continue to fall'
def test_profile_and_parser():
    import tags,profile_policy,source_meta,make_cldf
    tok=Tokenizer(str(PROFILE));rows=source.generate()[0]
    assert 'grierson-sodochi-1916' not in profile_policy.audit({})
    for r in rows:
        assert len(r)==15 and not r[13]
        assert '�' not in tok(r[2],column='IPA')
        assert {t for t in r[14].split() if not t.startswith('dialect:')}<=tags.GRAMMATICAL_TAGS|tags.GENDER_TAGS
    assert source_meta.SourceMeta().transcription(source.SOURCE,CSV,'OuterSiraji')[0]=='grierson-sodochi-1916'
    errors=io.StringIO();parsed,stats=make_cldf.parse_file(str(CSV),errors,name='20260925-grierson-sodochi')
    assert not errors.getvalue() and len(parsed)==1146 and stats['converted']==1146
