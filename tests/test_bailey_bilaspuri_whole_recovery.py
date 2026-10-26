"""Source-stage whole Bilaspuri checks, independent of compiled database builds."""
import csv,json,unicodedata
from pathlib import Path
from segments import Tokenizer
from tags import GRAMMATICAL_TAGS,GENDER_TAGS

P=Path(__file__).resolve().parents[1]/'data/other/forms/raw_data/bailey_bilaspuri_1920'


def rows():return list(csv.reader((P/'whole-proposed.csv').open()))
def audit():return [json.loads(s)for s in(P/'whole-proposed-audit.jsonl').read_text().splitlines()]


def test_whole_scope_and_exact_legacy_preservation():
    data=rows();units=audit()
    assert len(data)==678 and len(units)==546
    assert data[:288]==list(csv.reader((P/'legacy-before-whole-recovery.csv').open()))
    assert len({r[10]for r in data})==678
    new=units[237:]
    assert len(new)==309
    assert sum(':sentences:'in u['source_unit_key']for u in new)==22
    assert sum(':context:'in u['source_unit_key']for u in new)==30
    assert all(not u['exported_entry_keys']for u in new if ':context:'in u['source_unit_key'])
    assert sum(u['disposition']=='printed-continuation-control-no-invented-cells'for u in new)==3


def test_explicit_paradigm_expansion_and_no_etc_forms():
    keyed={r[10]:r for r in rows()}
    base='bailey1920bilaspuri:grammar:'
    assert keyed[base+'235:horse:2:singular_or_form:1'][2]=='ghōṛe dā'
    assert keyed[base+'236:elephant:2:plural:1'][2]=='hāthīā̃ dā'
    assert keyed[base+'239:fall:4:singular_or_form:1'][2]=='rŭṛhe'
    assert not any(r[2]=='etc.'for r in keyed.values())
    # Whole example cannot be misclassified as a suffix or lose its translation.
    example=keyed[base+'233:intro-ability:2:singular_or_form:1']
    assert example[3]=='I cannot eat rice'
    assert 'sentential'in example[14].split() and 'suffix'not in example[14].split()


def test_profile_normalization_and_registered_tags():
    converter=Tokenizer(str(P/'whole-proposed-profile.txt'))
    for r in rows():
        assert len(r)==15 and r[0]=='bil'
        assert set(r[14].split())<=GRAMMATICAL_TAGS|GENDER_TAGS
        a=converter(unicodedata.normalize('NFC',r[2]),column='IPA')
        b=converter(unicodedata.normalize('NFD',r[2]),column='IPA')
        assert '\ufffd'not in a+b
        assert unicodedata.normalize('NFC',a)==unicodedata.normalize('NFC',b)


def test_variants_are_closed_and_not_gender_paradigms():
    keyed={r[10]:r for r in rows()}
    for key,r in keyed.items():
        assert not r[11]or(r[11]in keyed and r[11]!=key)
        assert not r[12]and not r[13]
        if ':intro-future:'in key or ':substantive:'in key:
            assert not r[11], 'Gender paradigm is not a spelling variant'
    cardinal=keyed['bailey1920bilaspuri:grammar:238:cardinal:47:singular_or_form:1']
    assert 'uncertain'in cardinal[14].split() and 'mark'in cardinal[6]


def test_printed_indeclinability_and_relative_labels_survive_cleanly():
    """Regressions from root's original-page audits of pp233,237,239."""
    keyed={r[10]:r for r in rows()}
    base='bailey1920bilaspuri:grammar:'
    future=keyed[base+'233:intro-future:4:singular_or_form:1']
    assert {'verb','fut','indecl'}<=set(future[14].split())
    assert 'Bilaspuri and Sasi' in future[6]
    for key,definition in [('237:pronominal-adjectives:4:singular_or_form:1','as much or many'),
                           ('239:place:4:singular_or_form:1','where')]:
        row=keyed[base+key]
        assert row[3]==definition
        assert 'relative' in row[14].split()
    # New rows must not retain successfully parsed printed grammar labels.
    for row in rows()[288:]:
        assert '(rel.)' not in row[3] and '(correl.)' not in row[3]
