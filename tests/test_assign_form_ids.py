from assign_form_ids import apply_assignments, assign_ids, fingerprint, has_dictionary_entry_id


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


def test_rejected_assignment_deletes_generated_alternate(tmp_path):
    import csv

    from assign_form_ids import EDGES_FIELDS, migrate_assignment_schema, validate_assignments

    local = form("f_example", "pani")
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
