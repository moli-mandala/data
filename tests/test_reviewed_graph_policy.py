"""Negative fixtures prove reviewed-source graph checks reject destructive drift."""
import csv
import pytest
import reviewed_graph_policy as policy


@pytest.mark.parametrize('mutation', ['unapproved','missing','rank','position','status','duplicate',None])
def test_graph_reconciliation_rejects_unreviewed_or_lost_evidence(tmp_path, monkeypatch, mutation):
    (tmp_path/'cldf').mkdir()
    fields=['Child_ID','Parent_ID','Kind','Rank','Pos']
    edge=dict(Child_ID='child',Parent_ID='parent',Kind='component',Rank='1',Pos='2')
    rows=[dict(edge)]
    forms=[dict(ID='child',Status='',Redirect='')]
    if mutation=='unapproved':rows.append(dict(edge,Parent_ID='other'))
    elif mutation=='missing':rows=[]
    elif mutation=='rank':rows[0]['Rank']='2'
    elif mutation=='position':rows[0]['Pos']='1'
    elif mutation=='status':forms[0]['Status']='unlinked'
    elif mutation=='duplicate':rows.append(dict(edge))
    with (tmp_path/'cldf/edges.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    monkeypatch.setattr(policy,'ROOT',tmp_path)
    monkeypatch.setattr(policy,'read_assignments',lambda:[dict(Form_ID='child',Etymon_ID='parent',Kind='component',Rank='1',Pos='2',Status='accepted')])
    if mutation is None:
        policy.assert_reviewed_source_graph(forms)
    else:
        with pytest.raises(AssertionError):policy.assert_reviewed_source_graph(forms)
