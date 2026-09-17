"""Small regression cases for the CDIAL grouping policy (no corpus-sized fixtures)."""
from copy import deepcopy
import pytest
from edges_build import validate_edge_dicts
from nuristani_grouping import GROUP_TAG, group_graph


def form(i, lang, word='', status='entry'):
    return dict(ID=i, Language_ID=lang, Form=word, Original=word, Tags='', Status=status, Redirect='')


def edge(child, parent, kind='reflex', rank='1'):
    return dict(Child_ID=child, Parent_ID=parent, Kind=kind, Rank=rank, Pos='', Source='strand', Note='')


def run(forms, edges, groups=None, redirects=None, aliases=None):
    audit = group_graph(forms, edges, groups or {'n1':'10'}, redirects or {}, aliases or {}, {'PNur','Kt'})
    validate_edge_dicts(edges, {r['ID']:r['Status'] for r in forms})
    return audit


@pytest.mark.parametrize('kind', ['reflex', 'borrowed'])
def test_inherited_and_borrowed_branches_become_siblings(kind):
    forms=[form('10','Indo-Aryan','head'), form('n1','PNur','*pn'), form('a','Kt','word','')]
    edges=[edge('n1','10',kind), edge('a','n1')]
    run(forms,edges)
    assert {(e['Child_ID'],e['Parent_ID'],e['Kind']) for e in edges}=={('n1','10','reflex'),('a','10','reflex')}
    assert all(GROUP_TAG in r['Tags'] for r in forms[1:])
    assert all('inheritance versus borrowing unresolved' in e['Note'] for e in edges)
    before=deepcopy((forms,edges))
    assert run(forms,edges)==[]
    assert (forms,edges)==before


def test_blank_pii_redirects_preserve_ids_and_remove_ia_self_edge():
    forms=[form('10','Indo-Aryan','head',''),form('pii','Indo-ir'),form('n1','PNur','*pn',''),form('a','Kt','word','')]
    edges=[edge('10','pii'),edge('n1','pii'),edge('a','n1')]
    run(forms,edges,redirects={'pii':'10'})
    assert forms[1]['Redirect']=='10' and forms[0]['Status']=='entry'
    assert all(e['Parent_ID']!='pii' and e['Child_ID']!='pii' for e in edges)
    assert all(e['Parent_ID']=='10' for e in edges)
    assert run(forms,edges,redirects={'pii':'10'})==[]


def test_real_pii_reconstruction_is_retained_as_sibling():
    forms=[form('10','Indo-Aryan','head'),form('p','Indo-ir','*pii'),form('a','Kt','word','')]
    edges=[edge('a','p')]
    run(forms,edges,groups={'p':'10'})
    assert forms[1]['Form']=='*pii' and forms[1]['Redirect']==''
    assert {e['Parent_ID'] for e in edges}=={'10'}


def test_variant_keeps_lexical_target_but_pnur_is_not_an_intermediate():
    forms=[form('10','Indo-Aryan','head'),form('n1','PNur','*pn'),form('a','Kt','word',''),form('v','Kt','variant','')]
    edges=[edge('a','n1'),edge('v','a','variant')]
    run(forms,edges)
    assert next(e for e in edges if e['Child_ID']=='v')['Parent_ID']=='a'
    assert next(e for e in edges if e['Child_ID']=='a')['Parent_ID']=='10'
    assert GROUP_TAG not in forms[-1]['Tags']


def test_all_sources_and_subsections_group_without_flattening_ia_sections():
    forms=[form('10','Indo-Aryan','head'),form('10-2','Indo-Aryan','section',''),form('n1','PNur','*pn'),form('a','Kt','word','')]
    edges=[edge('10-2','10'),edge('a','10-2')]
    edges[-1]['Source']='buddruss'
    run(forms,edges)
    assert next(e for e in edges if e['Child_ID']=='a')['Parent_ID']=='10-2'
    assert next(e for e in edges if e['Child_ID']=='10-2')['Parent_ID']=='10'
    assert GROUP_TAG in forms[-1]['Tags']


def test_explicit_nested_mapping_wins_and_unmatched_branches_remain():
    forms=[form('10','Indo-Aryan','head'),form('20','Indo-Aryan','other'),form('n1','PNur','*a'),form('n2','PNur','*b',''),form('a','Kt','word',''),form('u','PNur','*u'),form('v','Kt','unmapped','')]
    edges=[edge('n2','n1'),edge('a','n2'),edge('v','u')]
    run(forms,edges,groups={'n1':'10','n2':'20'})
    assert next(e for e in edges if e['Child_ID']=='a')['Parent_ID']=='20'
    assert next(e for e in edges if e['Child_ID']=='v')['Parent_ID']=='u'
    assert forms[-1]['Tags']==''


def test_alias_resolution_and_unrelated_iranian_ancestry_survive():
    forms=[form('10','Indo-Aryan','head',''),form('real-pii','Indo-ir','*pii'),form('durable','PNur','*pn')]
    edges=[edge('10','real-pii')]
    run(forms,edges,aliases={'n1':'durable'})
    assert next(e for e in edges if e['Child_ID']=='10')['Parent_ID']=='real-pii'
    assert next(e for e in edges if e['Child_ID']=='durable')['Parent_ID']=='10'


def test_alternate_same_group_edge_is_removed():
    forms=[form('10','Indo-Aryan','head'),form('n1','PNur','*pn'),form('a','Kt','word','')]
    edges=[edge('a','n1'),edge('a','10',rank='2')]
    run(forms,edges)
    assert len([e for e in edges if e['Child_ID']=='a'])==1


def test_refuses_nonblank_redirect_and_unknown_target():
    forms=[form('10','Indo-Aryan','head'),form('n1','PNur','*pn'),form('p','Indo-ir','*real')]
    with pytest.raises(ValueError,match='nonblank'):run(forms,[],redirects={'p':'10'})
    with pytest.raises(ValueError,match='Missing'):run(forms,[],groups={'n1':'99'})


def test_native_cdial_subsection_can_have_a_durable_id():
    forms=[form('f_section','Indo-Aryan','section'),form('n1','PNur','*pn'),form('a','Kt','word','')]
    forms[0]['Source']='CDIAL'
    edges=[edge('a','f_section')]
    run(forms,edges,groups={'n1':'10-2'},aliases={'10-2':'f_section'})
    assert {e['Parent_ID'] for e in edges}=={'f_section'}


def test_non_nuristani_article_descendant_moves_but_unrelated_ia_reflex_does_not():
    forms=[form('10','Indo-Aryan','head'),form('n1','PNur','*pn'),form('a','Dm','word',''),form('other','H','word','')]
    edges=[edge('a','n1'),edge('other','10')]
    run(forms,edges)
    assert next(e for e in edges if e['Child_ID']=='a')['Parent_ID']=='10'
    assert GROUP_TAG in forms[2]['Tags']
    assert forms[3]['Tags']==''
