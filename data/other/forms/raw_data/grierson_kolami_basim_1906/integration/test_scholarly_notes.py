import csv,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
def rows(revision):
    with (HERE/revision/'combined-review-candidate.csv').open() as f:return list(csv.reader(f))
def test_scholarly_suffix_notes_leave_literals_and_other_fields_exact():
    old=rows('revision5');new=rows('revision6')
    delta={b[10]:(a,b) for a,b in zip(old,new) if a!=b}
    assert len(old)==len(new)==827 and len(delta)==2
    expected={'noun-plural':('l','Grierson identifies l as the plural suffix.'),
              'accusative-dative':('n','Grierson identifies n as the accusative and dative suffix.')}
    for fragment,(form,note) in expected.items():
        key=f'grierson1906kolamibasim:p562:{fragment}:1:independent:w1'
        a,b=delta[key];assert a[2]==b[2]==form
        assert a[6]=='No invented leading hyphen' and b[6]==note
        assert a[:6]+a[7:]==b[:6]+b[7:]
def test_same_independent_sample_keys_and_audit_history():
    a=json.loads((HERE/'revision5/fresh-output-selection-20260927.json').read_text())
    b=json.loads((HERE/'revision6/fresh-output-selection-20260927.json').read_text())
    assert [x['source_unit_key'] for x in a['sample']]==[x['source_unit_key'] for x in b['sample']]
    before=[json.loads(s) for s in (HERE/'revision5/combined-review-audit.jsonl').read_text().splitlines()]
    after=[json.loads(s) for s in (HERE/'revision6/combined-review-audit.jsonl').read_text().splitlines()]
    assert len(before)==len(after)==1570
    for x,y in zip(before,after):
        assert x['source_unit_key']==y['source_unit_key'] and x['entry_keys']==y['entry_keys']
        if x!=y:
            assert y['historical_pre_scholarly_notes_row']==x['actual_row']
            assert y['source_record']==x['source_record']
