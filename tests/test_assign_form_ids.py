from assign_form_ids import (
    apply_assignments,
    assign_ids,
    drop_stale_subentry_assignments,
    fingerprint,
    has_dictionary_entry_id,
    is_retired_subentry,
)


def form(legacy_id, original, *, rendered=None, gloss="water", relation="local"):
    # relation kept as a fixture parameter; the edge-model row carries only Status
    status = {"local": "unlinked", "": "entry"}.get(relation, "")
    return {
        "ID": legacy_id,
        "Language_ID": "x",
        "Form": rendered or original,
        "Gloss": gloss,
        "Native": "",
        "Original": original,
        "Source": "example-source",
        "Redirect": "",
        "Status": status,
    }


def test_overlay_variant_replaces_ancestry_and_is_idempotent(tmp_path):
    import csv
    from assign_form_ids import EDGES_FIELDS, validate_assignments, write_rows

    forms = [form("head", "phulgobi", relation=""),
             form("old", "gobi", relation=""), form("survey", "fulgobi")]
    assignments = [dict(Form_ID="survey", Etymon_ID="head", Kind="variant",
                        Rank="1", Status="accepted", Source="example-source")]
    validate_assignments(forms, assignments)
    path = tmp_path / "edges.csv"
    write_rows(path, EDGES_FIELDS, [dict(Child_ID="survey", Parent_ID="old",
                                       Kind="reflex", Rank="1", Pos="", Source="", Note="")])
    assert apply_assignments(path, forms, assignments) == 2
    with path.open() as handle:
        edges = list(csv.DictReader(handle))
    assert [(e["Child_ID"], e["Parent_ID"], e["Kind"]) for e in edges] == [
        ("survey", "head", "variant")]
    assert forms[-1]["Status"] == ""
    assert apply_assignments(path, forms, assignments) == 0


def test_overlay_derivative_can_target_base_assigned_in_same_batch(tmp_path):
    import csv
    from assign_form_ids import EDGES_FIELDS, validate_assignments
    from edges_build import validate_edge_dicts

    forms = [form("root", "cut", relation=""), form("base", "chhin"),
             form("passive", "chhinjil")]
    assignments = [
        dict(Form_ID="passive", Etymon_ID="base", Kind="derived", Rank="1", Status="accepted"),
        dict(Form_ID="base", Etymon_ID="root", Kind="reflex", Rank="1", Status="accepted"),
    ]
    validate_assignments(forms, assignments)
    path = tmp_path / "edges.csv"
    path.write_text(",".join(EDGES_FIELDS) + "\n")
    apply_assignments(path, forms, assignments)
    with path.open() as handle:
        edges = list(csv.DictReader(handle))
    validate_edge_dicts(edges, {f["ID"]: f["Status"] for f in forms})
    assert {(e["Child_ID"], e["Parent_ID"], e["Kind"]) for e in edges} == {
        ("passive", "base", "derived"), ("base", "root", "reflex")}
    assert apply_assignments(path, forms, assignments) == 0


def test_overlay_derivative_rejects_unestablished_base():
    import pytest
    from assign_form_ids import validate_assignments

    forms = [form("root", "cut", relation=""), form("base", "chhin"),
             form("passive", "chhinjil")]
    derivative = dict(Form_ID="passive", Etymon_ID="base", Kind="derived", Rank="1", Status="accepted")
    for base_assignment in [
        [],
        [dict(Form_ID="base", Etymon_ID="root", Kind="reflex", Rank="1", Status="rejected")],
        [dict(Form_ID="base", Etymon_ID="root", Kind="reflex", Rank="2", Status="accepted")],
        [dict(Form_ID="base", Etymon_ID="passive", Kind="derived", Rank="1", Status="accepted")],
    ]:
        with pytest.raises(ValueError, match="missing etymon"):
            validate_assignments(forms, [derivative] + base_assignment)


def test_registry_survives_reordering_and_profile_changes():
    first = [form("8-1", "pani", rendered="pāni"), form("8-2", "ag", gloss="fire")]
    initial_mapping, registry = assign_ids(first, [])

    rebuilt = [
        form("11-40", "ag", rendered="aɡ", gloss="fire"),
        form("11-41", "pani", rendered="pɑːni"),
    ]
    rebuilt_mapping, next_registry = assign_ids(rebuilt, registry)

    assert rebuilt_mapping["11-40"] == initial_mapping["8-2"]
    assert rebuilt_mapping["11-41"] == initial_mapping["8-1"]
    assert {row["Status"] for row in next_registry} == {"active"}


def test_active_identity_wins_when_retired_tombstone_reuses_legacy_id():
    original = form("8-1", "pani")
    initial_mapping, registry = assign_ids([original], [])
    active_id = initial_mapping["8-1"]
    retired = dict(registry[0])
    retired.update({
        "Form_ID": "f_retired",
        "Fingerprint": "different",
        "Original": "other",
        "Status": "retired",
    })

    corrected = form("8-1", "pani", gloss="drinking water")
    mapping, _ = assign_ids([corrected], [registry[0], retired])

    assert mapping["8-1"] == active_id


def test_graph_assignment_patches_edges(tmp_path):
    import csv

    from assign_form_ids import EDGES_FIELDS, migrate_assignment_schema, validate_assignments

    local = form("f_example", "pani")
    etymon = form("123", "*paniya", relation="")
    forms = [local, etymon]
    edges_path = tmp_path / "edges.csv"
    with edges_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EDGES_FIELDS)
        writer.writeheader()

    assignments = [
        {"Form_ID": "f_example", "Etymon_ID": "123", "Relation": "reflex", "Status": "accepted"}
    ]
    migrate_assignment_schema(assignments)
    validate_assignments(forms, assignments)
    changed = apply_assignments(edges_path, forms, assignments)

    assert changed == 2  # new rank-1 edge + Status cleared
    assert local["Status"] == ""
    with edges_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "Child_ID": "f_example", "Parent_ID": "123", "Kind": "reflex", "Rank": "1",
            "Pos": "", "Source": "", "Note": "",
        }
    ]


def test_derived_assignment_survives_reapplication(tmp_path):
    import csv
    from assign_form_ids import EDGES_FIELDS, validate_assignments

    local = form("f_needle", "sīlan")
    parent = form("13444", "sīvyati", relation="")
    forms = [local, parent]
    path = tmp_path / "edges.csv"
    with path.open("w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=EDGES_FIELDS).writeheader()
    assignments = [dict(Form_ID="f_needle", Etymon_ID="13444", Kind="derived",
                        Rank="1", Status="accepted", Source="CDIAL",
                        Notes="Deverbal noun; suffix uncertain.")]
    validate_assignments(forms, assignments)
    assert apply_assignments(path, forms, assignments) == 2
    assert apply_assignments(path, forms, assignments) == 0
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 1
    assert rows[0]["Kind"] == "derived"
    assert rows[0]["Note"] == assignments[0]["Notes"]
    assert local["Status"] == ""


def test_rejected_assignment_deletes_generated_alternate(tmp_path):
    import csv

    from assign_form_ids import EDGES_FIELDS, migrate_assignment_schema, validate_assignments

    # already attested (it carries the rank-1 edge below), so Status is empty rather than
    # `unlinked` — a parentless Status alongside an accepted edge is a state the pipeline
    # never produces, and apply_assignments now rejects it.
    local = form("f_example", "pani", relation="reflex")
    forms = [local, form("123", "*paniya", relation=""), form("456", "*panya", relation="")]
    edges_path = tmp_path / "edges.csv"
    with edges_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EDGES_FIELDS)
        writer.writeheader()
        writer.writerow({
            "Child_ID": "f_example", "Parent_ID": "123", "Kind": "reflex", "Rank": "1",
            "Pos": "", "Source": "", "Note": "",
        })
        writer.writerow({
            "Child_ID": "f_example", "Parent_ID": "456", "Kind": "reflex", "Rank": "2",
            "Pos": "", "Source": "", "Note": "review:auto-alternate",
        })

    assignments = [
        {"Form_ID": "f_example", "Etymon_ID": "456", "Kind": "reflex", "Rank": "2",
         "Status": "rejected"}
    ]
    migrate_assignment_schema(assignments)
    validate_assignments(forms, assignments)
    changed = apply_assignments(edges_path, forms, assignments)

    assert changed == 1
    with edges_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["Parent_ID"] for r in rows] == ["123"]


def test_fingerprint_ignores_generated_transcription_and_graph_assignment():
    before = form("1-2", "kaal", rendered="kāl")
    after = form("99-8", "kaal", rendered="kɑːl", relation="reflex")

    assert fingerprint(before) == fingerprint(after)


def test_fingerprint_ignores_reference_locators_and_reference_order():
    before = form("1-2", "kaal")
    before["Source"] = "dictionary-a;dictionary-b"
    after = form("99-8", "kaal")
    after["Source"] = "dictionary-b[p. 42, col. 2]; dictionary-a[entry 17]"

    assert fingerprint(before) == fingerprint(after)


def test_immutable_source_key_survives_source_text_correction_and_reordering():
    initial = [form("4-1", "mistakn", gloss="old gloss")]
    first_mapping, registry = assign_ids(initial, [], {"4-1": "dictionary-entry-72"})
    corrected = [form("18-90", "mistaken", gloss="corrected gloss")]

    next_mapping, _ = assign_ids(corrected, registry, {"18-90": "dictionary-entry-72"})

    assert next_mapping["18-90"] == first_mapping["4-1"]


def test_only_cdial_and_dedr_entries_keep_source_ids():
    cdial = form("3643", "*kakka", relation="reflex")
    cdial.update(Language_ID="Indo-Aryan", Source="CDIAL")
    dedr = form("d142", "*kāk-", relation="reflex")
    dedr.update(Language_ID="PDr", Source="krishnamurti")
    nuristani = form("n42", "*kaka", relation="")
    nuristani.update(Language_ID="PNur", Source="")
    proto_ii = form("pii-42", "*kaka", relation="")
    proto_ii.update(Language_ID="Indo-ir", Source="")
    cdial_reflex = form("3643-2-reflex", "kakka", relation="reflex")
    cdial_reflex.update(Language_ID="Indo-Aryan", Source="CDIAL")

    mapping, _ = assign_ids([cdial, dedr, nuristani, proto_ii, cdial_reflex], [])

    assert has_dictionary_entry_id(cdial)
    assert has_dictionary_entry_id(dedr)
    assert "3643" not in mapping
    assert "d142" not in mapping
    assert mapping["n42"].startswith("f_")
    assert mapping["pii-42"].startswith("f_")
    assert mapping["3643-2-reflex"].startswith("f_")


def test_an_assignment_on_a_retired_dictionary_subentry_is_dropped_not_redirected():
    """A recycled positional pre-ID must not capture a curated etymology.

    make_cldf mints pre-IDs as ``<file>-<row>``, which is the same shape as a
    dictionary sub-entry id such as CDIAL ``103-2`` under etymon ``103``.
    Inserting one source file re-issues every later pre-ID, so a retired
    sub-entry id can be handed to an unrelated row and then aliased to it.
    Zargari ``pani`` 'water' acquired CDIAL 103 ``aŋkapāli`` 'embrace' that way.
    """
    live = {"103", "103-2x", "10005", "10005-2", "f_zargari"}
    retired = {"Form_ID": "103-2", "Etymon_ID": "103", "Kind": "reflex", "Rank": "1"}
    still_there = {"Form_ID": "10005-2", "Etymon_ID": "10005", "Kind": "reflex", "Rank": "1"}
    # A curated assignment on an ordinary source row happens to share the shape
    # but does not name a sub-entry of its own etymon, so it is left alone.
    positional = {"Form_ID": "22-13", "Etymon_ID": "5678", "Kind": "reflex", "Rank": "1"}
    opaque = {"Form_ID": "f_zargari", "Etymon_ID": "103", "Kind": "reflex", "Rank": "1"}

    assert is_retired_subentry(retired, live)
    assert not is_retired_subentry(still_there, live)
    assert not is_retired_subentry(positional, live)
    assert not is_retired_subentry(opaque, live)

    kept, stale = drop_stale_subentry_assignments(
        [retired, still_there, positional, opaque], live
    )
    assert stale == [retired]
    assert kept == [still_there, positional, opaque]


def test_the_dictionary_self_reference_assignments_all_point_at_live_sub_entries():
    """Guards the corpus against the same capture recurring."""
    import csv
    import re
    from pathlib import Path

    from etymology_assignments import read_assignments

    root = Path(__file__).parents[1]
    forms_path = root / "cldf/forms.csv"
    if not forms_path.exists():
        return
    with forms_path.open(encoding="utf-8", newline="") as stream:
        live = {row["ID"] for row in csv.DictReader(stream)}
    assignments = read_assignments()
    captured = [row for row in assignments if is_retired_subentry(row, live)]
    assert captured == []
    # Every surviving self-reference names a sub-entry that still exists.
    self_refs = [
        row for row in assignments
        if row["Etymon_ID"]
        and re.fullmatch(re.escape(row["Etymon_ID"]) + r"-\d+[a-z]*", row["Form_ID"])
    ]
    assert len(self_refs) == 2604
    assert all(row["Form_ID"] in live for row in self_refs)


def test_self_referential_assignment_is_rejected():
    """The regression that dropped 14,506 CDIAL headwords: legacy stored a headword as an entry
    row plus an attested row beneath it, both of which collapse onto one node here, so the
    importer resolved the pair into a link from the node to itself."""
    import pytest

    from assign_form_ids import migrate_assignment_schema, validate_assignments

    etymon = form("9011", "prōñchati", relation="")
    assignments = [{"Form_ID": "9011", "Etymon_ID": "9011", "Kind": "reflex", "Rank": "1",
                    "Status": "accepted"}]
    migrate_assignment_schema(assignments)
    with pytest.raises(ValueError, match="points at itself"):
        validate_assignments([etymon], assignments)


def test_patched_edge_table_is_revalidated(tmp_path):
    """apply_assignments is the last writer of cldf/edges.csv, so overlay-only breakage must not
    slip past the invariants `edges_build` enforces when it first derives the graph."""
    import csv

    import pytest

    from assign_form_ids import EDGES_FIELDS

    etymon = form("9011", "prōñchati", relation="")
    edges_path = tmp_path / "edges.csv"
    with edges_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EDGES_FIELDS)
        writer.writeheader()
        writer.writerow({"Child_ID": "9011", "Parent_ID": "9011", "Kind": "reflex",
                         "Rank": "1", "Pos": "", "Source": "", "Note": ""})

    with pytest.raises(ValueError, match="self-edge"):
        apply_assignments(edges_path, [etymon], [])


def test_rank1_assignment_clears_a_sub_entry_headword_status(tmp_path):
    """A CDIAL sub-entry re-homed onto its article head stops being a parentless headword, so
    forms.csv must stop calling it one — otherwise it says `entry` while edges.csv says reflex."""
    import csv

    from assign_form_ids import EDGES_FIELDS, migrate_assignment_schema, validate_assignments

    head = form("9017", "prṓṣati", relation="")
    section = form("9017-2", "plṓṣati", relation="")
    assert section["Status"] == "entry"
    edges_path = tmp_path / "edges.csv"
    with edges_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=EDGES_FIELDS).writeheader()

    assignments = [{"Form_ID": "9017-2", "Etymon_ID": "9017", "Kind": "reflex", "Rank": "1",
                    "Status": "accepted"}]
    migrate_assignment_schema(assignments)
    validate_assignments([head, section], assignments)
    apply_assignments(edges_path, [head, section], assignments)

    assert section["Status"] == ""
    assert head["Status"] == "entry"


def test_overlay_components_roundtrip_and_require_all_bases(tmp_path):
    import csv
    import pytest
    from assign_form_ids import ASSIGNMENT_FIELDS, EDGES_FIELDS, validate_assignments, write_rows, read_rows

    forms = [form('root', 'twenty', relation=''), form('two', 'two', relation=''),
             form('forty', 'two-twenty'), form('forty-two', 'two-and-forty')]
    assignments = [dict(Form_ID=child, Etymon_ID=parent, Kind='component', Rank='1',
                        Pos=str(pos), Status='accepted', Source='example', Notes='composition')
                   for child, parents in [('forty-two', ['two', 'forty']), ('forty', ['two', 'root'])]
                   for pos, parent in enumerate(parents, 1)]
    overlay = tmp_path / 'assignments.csv'
    write_rows(overlay, ASSIGNMENT_FIELDS, assignments)
    _, readback = read_rows(overlay)
    validate_assignments(forms, readback)
    edges = tmp_path / 'edges.csv'
    edges.write_text(','.join(EDGES_FIELDS) + '\n')
    apply_assignments(edges, forms, readback)
    rows = list(csv.DictReader(edges.open()))
    assert [(r['Parent_ID'], r['Pos']) for r in rows if r['Child_ID']=='forty-two'] == [('two', '1'), ('forty', '2')]
    assert apply_assignments(edges, forms, readback) == 0
    for f in forms:
        if f['ID'] in ['forty', 'forty-two', 'root']: f['Status'] = 'unlinked'
    with pytest.raises(ValueError, match='missing etymon'):
        validate_assignments(forms, assignments)
    forms[0]['Status'] = 'entry'
    bad = [dict(a) for a in assignments]
    bad[1]['Pos'] = '1'
    with pytest.raises(ValueError, match='not contiguous'):
        validate_assignments(forms, bad)
