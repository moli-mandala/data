"""Exercise compound saves against a small scratch graph, never the shared CLDF."""
import csv
import json

import pytest

import etymology_lab as lab


def decision(**kwargs):
    return {"record": {"ID": "f_child", "Language_ID": "test", "Form": "ab"},
            "citation": "CDIAL[1,2]", "evidence": "Both members accounted for.", **kwargs}


def write_csv(path, fields, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def graph(tmp_path, monkeypatch):
    forms, edges, registry = (tmp_path / name for name in ("forms.csv", "edges.csv", "ids.csv"))
    write_csv(forms, ["ID", "Form", "Language_ID", "Status", "Redirect"], [
        {"ID": "f_child", "Form": "ab", "Language_ID": "test", "Status": "unlinked"},
        {"ID": "1", "Form": "a", "Status": "entry"},
        {"ID": "2", "Form": "b", "Status": "entry"},
        {"ID": "other", "Form": "c"},
    ])
    write_csv(edges, ["Child_ID", "Parent_ID", "Kind", "Rank", "Pos", "Source", "Note"], [
        {"Child_ID": "other", "Parent_ID": "1", "Kind": "reflex", "Rank": "1"},
    ])
    write_csv(registry, ["Form_ID", "Status"], [{"Form_ID": "f_child", "Status": "active"}])
    for key, value in {"ROOT": tmp_path, "LAB": tmp_path / "lab", "FORMS": forms,
                       "EDGES": edges, "REGISTRY": registry}.items():
        monkeypatch.setattr(lab, key, value)
    monkeypatch.setattr(lab.overlay, "read_assignments", lambda: [])
    monkeypatch.setattr(lab.overlay, "assignment_files", lambda: [])
    return forms, edges, registry


def test_single_parent_rows_remain_compatible():
    assert lab.rows_from_decisions([decision(parent="1", kind="borrowed")], "Reviewed.") == [{
        "Form_ID": "f_child", "Etymon_ID": "1", "Kind": "borrowed", "Rank": "1",
        "Status": "accepted", "Source": "CDIAL[1,2]",
        "Notes": "Both members accounted for. Reviewed.", "Pos": "",
    }]


@pytest.mark.parametrize("historical", [False, True])
def test_compounds_preserve_order_and_validate_idempotently(graph, historical):
    item = decision(components=["2", "1"])
    if historical:
        item.update(parent="2", kind="component")
    before = [p.read_bytes() for p in graph]
    rows = lab.rows_from_decisions([item], "")
    assert [(r["Etymon_ID"], r["Kind"], r["Pos"]) for r in rows] == [
        ("2", "component", "1"), ("1", "component", "2")]
    result = lab.validate_against_graph([item], rows)
    assert result["firstApplicationChanges"] == 3  # two edges plus child status in scratch
    assert result["secondApplicationChanges"] == 0
    assert [p.read_bytes() for p in graph] == before


@pytest.mark.parametrize("kwargs", [
    {"components": []}, {"components": ["1"]}, {"components": "12"},
    {"components": ["1", ""]}, {"components": ["1", " 2"]},
    {"components": ["1", "2"], "kind": "reflex"},
    {"components": ["1", "2"], "parent": "2"},
    {"components": ["1", "2"], "pos": "1"},
])
def test_malformed_compounds_are_not_silently_saved(kwargs):
    with pytest.raises(ValueError):
        lab.rows_from_decisions([decision(**kwargs)], "")


@pytest.mark.parametrize("problem", ["missing", "unlinked", "redirect"])
def test_every_component_parent_is_checked(graph, problem):
    fields, forms = lab.read_rows(graph[0])
    if problem == "missing":
        forms = [r for r in forms if r["ID"] != "2"]
    else:
        forms[2]["Status" if problem == "unlinked" else "Redirect"] = (
            "unlinked" if problem == "unlinked" else "1")
    write_csv(graph[0], fields, forms)
    item = decision(components=["1", "2"])
    with pytest.raises(ValueError, match="absent|unlinked|redirect|missing etymon"):
        lab.validate_against_graph([item], lab.rows_from_decisions([item], ""))


def test_dry_run_does_not_write_compounds(graph, tmp_path):
    path = tmp_path / "decisions.json"
    path.write_text(json.dumps({"accepted": [decision(components=["1", "2"])]}))
    report = lab.save(path, pass_name="compound", note="", authorization="test", dry_run=True)
    assert report["assignmentRows"] == 2
    assert report["affectedRecords"] == 1
    assert not list(tmp_path.glob("compound-*"))
    assert not lab.LAB.exists()


def test_manifest_keeps_every_component_in_order(graph, tmp_path):
    item = decision(components=["2", "1"])
    rows = lab.rows_from_decisions([item], "")
    selected = lab.validate_against_graph([item], rows)["forms"]
    paths = lab.write_language_manifests([item], rows, selected, pass_dir=tmp_path,
                                       validation_path=tmp_path / "audit.json",
                                       authorization="test", saved_at="test")
    proposal = json.loads(lab.Path(paths[0]).read_text())["proposals"][0]
    assert proposal["componentIds"] == ["2", "1"]
    assert proposal["componentForms"] == ["b", "a"]
    assert proposal["assignments"] == rows
    assert proposal["kind"] == "component"


def test_repeated_lexical_member_keeps_both_positions(graph, tmp_path):
    item = decision(components=["1", "1"])
    rows = lab.rows_from_decisions([item], "")
    result = lab.validate_against_graph([item], rows)
    assert result["firstApplicationChanges"] == 3
    assert result["secondApplicationChanges"] == 0
    paths = lab.write_language_manifests([item], rows, result["forms"], pass_dir=tmp_path,
                                       validation_path=tmp_path / "audit.json",
                                       authorization="test", saved_at="test")
    proposal = json.loads(lab.Path(paths[0]).read_text())["proposals"][0]
    assert proposal["componentIds"] == ["1", "1"]
    assert [r["Pos"] for r in proposal["assignments"]] == ["1", "2"]


@pytest.mark.parametrize("dry_run", [False, True])
def test_retraction_preserves_unrelated_rows_and_compiled_inputs(graph, tmp_path, monkeypatch, dry_run):
    rows = lab.rows_from_decisions([decision(components=["1", "2"])], "")
    other = dict(rows[0], Form_ID="f_unrelated")
    current = rows + [other]
    monkeypatch.setattr(lab.overlay, "read_assignments", lambda: list(current))
    monkeypatch.setattr(lab.overlay, "write_assignments", lambda retained: current.__setitem__(slice(None), retained))
    ledger = tmp_path / "saved.json"
    ledger.write_text(json.dumps(rows))
    output = tmp_path / "retraction.json"
    before = [p.read_bytes() for p in graph]
    lab.retract(ledger, form_ids=["f_child"], reason="Ambiguous suffix.",
                authorization="test", output=output, dry_run=dry_run)
    assert current == (rows + [other] if dry_run else [other])
    assert output.exists() is not dry_run
    assert [p.read_bytes() for p in graph] == before


@pytest.mark.parametrize("change", ["missing", "extra", "edited"])
def test_retraction_refuses_changed_target_analysis(graph, tmp_path, monkeypatch, change):
    rows = lab.rows_from_decisions([decision(components=["1", "2"])], "")
    ledger = tmp_path / "saved.json"
    ledger.write_text(json.dumps(rows))
    current = rows[:1] if change == "missing" else rows + [dict(rows[0], Pos="3")] if change == "extra" else [dict(r, Notes="changed") for r in rows]
    monkeypatch.setattr(lab.overlay, "read_assignments", lambda: current)
    with pytest.raises(ValueError, match="differ"):
        lab.retract(ledger, form_ids=["f_child"], reason="Correction", authorization="test",
                    output=tmp_path / "retraction.json")


def add_base(graph, status="unlinked"):
    fields, forms = lab.read_rows(graph[0])
    forms.append({"ID": "f_base", "Form": "base", "Status": status})
    write_csv(graph[0], fields, forms)


def base_assignment(**kwargs):
    return dict(Form_ID="f_base", Etymon_ID="1", Kind="reflex", Rank="1",
                Status="accepted", Source="test", Notes="", Pos="", **kwargs)


@pytest.mark.parametrize("kind", ["derived", "variant", "borrowed"])
def test_previously_saved_base_is_usable_without_rebuild(graph, monkeypatch, kind):
    add_base(graph)
    monkeypatch.setattr(lab.overlay, "read_assignments", lambda: [base_assignment()])
    item = decision(parent="f_base", kind=kind)
    before = [p.read_bytes() for p in graph]
    report = lab.validate_against_graph([item], lab.rows_from_decisions([item], ""))
    assert report["secondApplicationChanges"] == 0
    assert report["firstApplicationChanges"] == 2
    assert [p.read_bytes() for p in graph] == before


@pytest.mark.parametrize("change", [{"Rank": "2"}, {"Status": "rejected"}])
def test_hypothesis_or_rejection_does_not_link_parent(graph, monkeypatch, change):
    add_base(graph)
    prior = base_assignment()
    prior.update(change)
    monkeypatch.setattr(lab.overlay, "read_assignments", lambda: [prior])
    item = decision(parent="f_base", kind="derived")
    with pytest.raises(ValueError, match="missing etymon|unlinked"):
        lab.validate_against_graph([item], lab.rows_from_decisions([item], ""))


def test_cycle_through_compiled_and_overlay_ancestry_is_rejected(graph, monkeypatch):
    add_base(graph, status="")
    fields, edges = lab.read_rows(graph[1])
    edges.append(dict(Child_ID="f_base", Parent_ID="f_child", Kind="reflex", Rank="1"))
    write_csv(graph[1], fields, edges)
    item = decision(parent="f_base", kind="derived")
    before = [p.read_bytes() for p in graph]
    with pytest.raises(ValueError, match="cycle"):
        lab.validate_against_graph([item], lab.rows_from_decisions([item], ""))
    assert [p.read_bytes() for p in graph] == before


def test_cycle_between_entries_is_allowed(graph):
    # CDIAL cross-references can compile as mutual `derived` edges between two entries
    # (e.g. 4248 <-> 2727); a child of either entry must still be savable.
    fields, edges = lab.read_rows(graph[1])
    edges += [dict(Child_ID="1", Parent_ID="2", Kind="derived", Rank="1"),
              dict(Child_ID="2", Parent_ID="1", Kind="derived", Rank="1")]
    write_csv(graph[1], fields, edges)
    item = decision(parent="other", kind="borrowed")
    result = lab.validate_against_graph([item], lab.rows_from_decisions([item], ""))
    assert result["secondApplicationChanges"] == 0


def test_overlay_cycle_is_rejected_even_if_one_node_was_compiled_linked(graph, monkeypatch):
    add_base(graph, status="")
    prior = base_assignment()
    prior["Etymon_ID"] = "other"
    reverse = dict(prior, Form_ID="other", Etymon_ID="f_base")
    monkeypatch.setattr(lab.overlay, "read_assignments", lambda: [prior, reverse])
    item = decision(parent="f_base", kind="derived")
    with pytest.raises(ValueError, match="cycle"):
        lab.validate_against_graph([item], lab.rows_from_decisions([item], ""))


def test_new_base_and_derivative_validate_independent_of_decision_order(graph):
    add_base(graph)
    fields, registry = lab.read_rows(graph[2])
    registry.append(dict(Form_ID="f_base", Status="active"))
    write_csv(graph[2], fields, registry)
    base = dict(record={"ID": "f_base"}, parent="1", kind="reflex")
    child = decision(parent="f_base", kind="derived")
    items = [child, base]
    result = lab.validate_against_graph(items, lab.rows_from_decisions(items, ""))
    assert result["secondApplicationChanges"] == 0


def test_unrelated_saved_overlay_is_only_applied_in_scratch(graph, monkeypatch):
    prior = dict(base_assignment(), Form_ID="other", Etymon_ID="2")
    monkeypatch.setattr(lab.overlay, "read_assignments", lambda: [prior])
    item = decision(parent="1", kind="reflex")
    before = [p.read_bytes() for p in graph]
    assert lab.validate_against_graph([item], lab.rows_from_decisions([item], ""))["secondApplicationChanges"] == 0
    assert [p.read_bytes() for p in graph] == before


@pytest.mark.parametrize("problem", ["missing", "redirect", "unlinked"])
def test_indirect_compiled_ancestor_must_resolve(graph, problem):
    add_base(graph, status="")
    fields, forms = lab.read_rows(graph[0])
    forms.append(dict(ID="f_terminal", Form="terminal", Status="unlinked" if problem == "unlinked" else "entry", Redirect="1" if problem == "redirect" else ""))
    if problem != "missing":
        write_csv(graph[0], fields, forms)
    fields, edges = lab.read_rows(graph[1])
    edges.append(dict(Child_ID="f_base", Parent_ID="f_terminal", Kind="reflex", Rank="1"))
    write_csv(graph[1], fields, edges)
    item = decision(parent="f_base", kind="derived")
    with pytest.raises(ValueError, match="absent|redirect|unlinked|parent not a node"):
        lab.validate_against_graph([item], lab.rows_from_decisions([item], ""))
