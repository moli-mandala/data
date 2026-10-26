"""Source-grounded regressions from the failed independent r1 output audit."""
import csv,json
from pathlib import Path
from tags import GRAMMATICAL_TAGS,GENDER_TAGS
P=Path(__file__).resolve().parents[1]/'data/other/forms/raw_data/bailey_kagani_1920/whole-source-recovery'
def rows():return list(csv.reader((P/'revision2-working/proposal.csv').open()))
def test_revision_preserves_every_key_and_source_form():
    old=list(csv.reader((P/'whole-proposal-r1/proposal.csv').open()));new=rows()
    assert len(old)==len(new)==1306
    assert [(r[10],r[2]) for r in old]==[(r[10],r[2]) for r in new]
    allowed={3,6,14}
    assert all(i in allowed for a,b in zip(old,new) for i,(x,y) in enumerate(zip(a,b)) if x!=y)

def test_ing_translation_does_not_imply_present_participle():
    data={r[10]:r for r in rows()};base='bailey1920kagani:whole:'
    for key,form in [('p103:parable:line11:word7','gēṇā'),('p103:parable:line12:word1','băjēṇā')]:
        r=data[base+key];tags=set(r[14].split())
        assert r[2]==form and {'verb','inf'}<=tags
        assert not {'participle','pres'} & tags
    for key in ['p103:parable:line11:word5','p103:parable:line14:word3']:
        r=data[base+key]
        assert r[2]=='ā' and 'participle' not in r[14].split()
    walking=[r for r in data.values() if ':story:' in r[10] and r[2]=='tŭrdā']
    assert walking and {'participle','pres'}<=set(walking[0][14].split())

def test_pronoun_scope_and_user_facing_notes():
    data={r[10]:r for r in rows()};base='bailey1920kagani:whole:'
    he=data[base+'p102:parable:line1:word4'];thine=data[base+'p104:parable:line12:word2']
    assert he[2]=='oh' and {'pron','third-person','sg'}<=set(he[14].split())
    assert thine[2]=='tŭṛai' and {'pron','second-person','sg','gen'}<=set(thine[14].split())
    for r in data.values():
        assert 'Apply multiword-expression' not in r[6] and 'Some proposed answers' not in r[6]
        assert {t for t in r[14].split() if not t.startswith('dialect:')}<=GRAMMATICAL_TAGS|GENDER_TAGS
