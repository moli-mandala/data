import importlib.util
from pathlib import Path
import pytest

RAW=Path(__file__).parents[1]/'data/other/forms/raw_data'
def module(name):
    spec=importlib.util.spec_from_file_location(name,RAW/f'{name}.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
L=module('lalasa_parse');A=module('lalasa')

def test_special_description_is_not_two_adjective_senses():
    raw='<div><hw><b>हाडवड़ियौ</b><b>hāḍavaṛiyau</b></hw>सं.पु. कृषि कार्य करने वाला। वि.वि.--परस्पर सहयोग की दृष्टि से कार्य करता है।</div>'
    rows=L.parse_article(raw,6009,10)['rows']
    assert len(rows)==1
    assert rows[0]['row'][3]=='कृषि कार्य करने वाला'
    assert rows[0]['row'][6]=='परस्पर सहयोग की दृष्टि से कार्य करता है।'
    assert rows[0]['row'][14]=='noun m'

def test_parenthetical_see_grammar_does_not_create_blank_section():
    rows=L.parse_article("<div><hw><b>अ</b><b>a</b></hw>देखो 'आ' (वि.)</div>",1,1)['rows']
    assert len(rows)==1
    assert rows[0]['row'][3]==''
    assert rows[0]['row'][14]==''

def test_numbering_continues_across_pos_and_preserves_duplicate_source_number():
    raw='<div><hw><b>रोसधर</b><b>rōsadhara</b></hw>वि. 1.कोप करने वाला। सं.पु.-- 2.इन्द्र। 2.वह मकान जिसमें रोस लगे हों।</div>'
    rows=L.parse_article(raw,4258,3)['rows']
    assert [r['row'][3] for r in rows]==['कोप करने वाला','इन्द्र','वह मकान जिसमें रोस लगे हों']
    assert [r['printed_sense'] for r in rows]==['1','2','2']
    assert 'source:duplicate-sense-number' in rows[1]['review']

def test_pos_and_numbered_senses_are_scoped():
    r=L.parse_article('<div><hw><b>अं</b><b>am</b></hw>सानुस्वार अ। सं.पु.-- 1.कमल. 2.पूर्ण ब्रह्म. वि.-- 1.विरक्त. 2.श्रेष्ठ</div>',1,2)
    assert [x['row'][3] for x in r['rows']]==['कमल','पूर्ण ब्रह्म','विरक्त','श्रेष्ठ']
    assert [x['row'][14] for x in r['rows']]==['noun m','noun m','adj','adj']
    assert all(x['row'][6]=='सानुस्वार अ।' for x in r['rows'])

def test_star_is_poetic_and_see_is_not_a_definition():
    r=L.parse_article("<div><hw><b>अंकआड*</b><b>aṅkaāḍa*</b></hw>वि. देखो 'आडे अंक'।</div>",1,4)
    assert r['rows'][0]['row'][2]=='aṅkaāḍa'
    assert r['rows'][0]['row'][3]==''
    assert r['rows'][0]['row'][14]=='adj poetic'
    assert r['crossreferences'][0]['target']=='आडे अंक'

def test_citations_are_usage_and_keep_source_reference():
    r=L.parse_article('<div><hw><b>अंक</b><b>aṅka</b></hw>सं. सं.पु. 1.भाग्य। <cit><quote><b>अंक</b> करे जोधांण</quote><ref>वं.भा.</ref></cit> 2.गोद।</div>',1,3)
    assert [x['row'][3] for x in r['rows']]==['भाग्य','गोद']
    assert r['rows'][0]['row'][6]=='अंक करे जोधांण'
    assert r['rows'][1]['row'][6]==''
    assert r['citations'][0]['reference']=='वं.भा.'
    assert not r['subentry_regions']

def test_verbal_usage_is_not_causative_and_is_not_gloss():
    r=L.parse_article('<div><hw><b>अंकाई</b><b>aṅkāī</b></hw>सं.स्त्री. अनुमान। क्रि.प्र.--करणी-होणी।</div>',1,24)
    assert r['rows'][0]['row'][3]=='अनुमान'
    assert 'caus' not in r['rows'][0]['row'][14]
    assert r['subentry_regions']==['क्रि.प्र.--करणी-होणी।']

def test_inline_family_is_lossless_and_not_parent_gloss():
    r=L.parse_article('<div><hw><b>अंकाणौ</b><b>aṅkāṇau</b></hw>क्रि.स. अंकाना। <b>अंकायोड़ौ</b>--भू.का.कृ.--अंकित कराया हुआ।</div>',1,26)
    assert r['rows'][0]['row'][3]=='अंकाना'
    assert 'अंकित कराया हुआ' in r['subentry_regions'][0]
    assert 'structure:derived-family-pending' in r['review']

def test_acquisition_continuation_is_explicit_and_sequential():
    assert A.inspect_page(A.CONTINUATION+' lalasa-2nd_query.py?page=167',166)==[167]
    with pytest.raises(ValueError):A.inspect_page('unavailable',166)
    with pytest.raises(ValueError):A.inspect_page(A.CONTINUATION,166)
    with pytest.raises(ValueError):A.inspect_page('<hw>word</hw> lalasa-2nd_query.py?page=168',166)


def test_feminine_prefix_does_not_swallow_main_definition():
    r=L.parse_article('<div><hw><b>निपगौ</b><b>nipagau</b></hw>स्त्रीलिंग--निपगी सं.नि+पद वि. अयोग्य, निकम्मा।</div>',2111,19)
    assert r['rows'][0]['row'][3]=='अयोग्य, निकम्मा'
    assert r['rows'][0]['row'][9]=='सं.नि+पद'
    assert r['subentry_regions']==['स्त्रीलिंग--निपगी']


def test_numeral_in_definition_is_not_a_sense_number():
    r=L.parse_article('<div><hw><b>पैंतीसौ</b><b>paintīsau</b></hw>राज.पैंतीस+प्र.औ. सं.पु. 1.पैतीस का वर्ष। 2.तीन हजार पांच सौ, 3500. रू.भे. पैंत्रीसौ।</div>',2589,8)
    assert len(r['rows'])==2
    assert r['rows'][1]['row'][3]=='तीन हजार पांच सौ, 3500'
    assert r['rows'][1]['row'][9]=='राज.पैंतीस+प्र.औ.'
