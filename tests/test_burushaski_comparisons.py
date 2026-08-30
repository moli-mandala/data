import csv
from collections import Counter
from pathlib import Path

from burushaski_comparisons import (
    COMPARISON_FIELDS,
    append_comparisons,
    project_claims,
    reviewed_sample_candidates,
)


ROOT = Path(__file__).parents[1]


def unified_row(
    form_id,
    language,
    form,
    gloss="",
    origin="",
    source="",
    relation="",
    variant_of="",
    etymology="",
):
    return [
        form_id, language, form, gloss, "", "", form, "", "", "", source,
        origin, etymology, relation, "", variant_of, "",
    ]


def test_projection_keeps_distinct_lexemes_and_preserves_internal_variants():
    cdial = unified_row("644", "Indo-Aryan", "ardʰá", "half")
    main = unified_row(
        "bur-1", "Bur", "áḍa", "remaining", "644", "berger", "reflex"
    )
    dialect = unified_row(
        "bur-2", "Bur", "áḍe", "", "644", "berger", "variant", "bur-1"
    )
    homonym = unified_row(
        "bur-3", "Bur", "qalt", "different", "644", "berger", "reflex"
    )
    rows = [cdial, main, dialect, homonym]

    proto, keys, comparisons, audit = project_claims(
        rows,
        {"Bur": "Burushaski", "PBr": "Burushaski", "Indo-Aryan": "OIA"},
        {"berger-entry-1": "bur-1", "berger-entry-1-dialect": "bur-2"},
        {"OIA"},
    )

    assert len(proto) == 2
    assert all(row[2] == row[6] == "" for row in proto)
    assert len(keys) == 2
    assert len(comparisons) == 2
    assert {row["Compared_Entry_ID"] for row in comparisons} == {"644"}
    assert {row["Relation"] for row in comparisons} == {"related"}
    assert {row["Direction"] for row in comparisons} == {"undetermined"}
    assert len({main[11], dialect[11], homonym[11]}) == 2
    assert main[13] == "reflex"
    assert (dialect[13], dialect[15]) == ("variant", "bur-1")
    assert homonym[13] == "reflex"
    assert {row["Status"] for row in audit} == {"converted"}


def test_projection_traces_indirect_cdial_path_without_retaining_cross_family_sibling():
    cdial = unified_row("4683", "Indo-Aryan", "cará", "moving")
    shina = unified_row("sh-1", "Sh", "ʦəri", "moving", "4683", "CDIAL", "reflex")
    bur = unified_row(
        "bur-1", "Bur", "ʦər", "moving", "sh-1", "CDIAL", "variant", "sh-1"
    )
    proto, _keys, comparisons, audit = project_claims(
        [cdial, shina, bur],
        {"Bur": "Burushaski", "PBr": "Burushaski", "Sh": "Shinaic", "Indo-Aryan": "OIA"},
        {},
        {"OIA", "Shinaic"},
        {"bur-1": "14406"},
    )

    assert len(proto) == len(comparisons) == len(audit) == 1
    assert comparisons[0]["Compared_Entry_ID"] == "4683"
    assert comparisons[0]["Source"] == "CDIAL[entry 14406]"
    assert "cross-reference chain resolves to CDIAL 4683" in comparisons[0]["Evidence"]
    assert audit[0]["Claim_Source_Entry_ID"] == "14406"
    assert (bur[11], bur[13], bur[15], bur[16]) == (proto[0][0], "reflex", "", "")


def test_append_comparisons_replaces_only_prior_burushaski_projection(tmp_path):
    path = tmp_path / "comparisons.csv"
    retained = {
        "ID": "dedr:1:cdial:2", "Entry_ID": "1", "Compared_Entry_ID": "2",
        "Relation": "related", "Direction": "undetermined", "Confidence": "medium",
        "Source": "dedr", "Evidence": "retained",
    }
    stale = {
        "ID": "burushaski:old", "Entry_ID": "old", "Compared_Entry_ID": "2",
        "Relation": "related", "Direction": "undetermined", "Confidence": "low",
        "Source": "berger", "Evidence": "stale",
    }
    replacement = stale | {"ID": "burushaski:new", "Entry_ID": "new", "Evidence": "fresh"}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows([retained, stale])

    append_comparisons([replacement], path)
    append_comparisons([replacement], path)

    assert dict_rows(path) == [replacement, retained]


def dict_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_compiled_burushaski_cdial_claims_are_comparisons_not_borrowings():
    forms = {row["ID"]: row for row in dict_rows(ROOT / "cldf/forms.csv")}
    edges = dict_rows(ROOT / "cldf/edges.csv")
    comparisons = [
        row for row in dict_rows(ROOT / "cldf/comparisons.csv")
        if row["ID"].startswith("burushaski:")
    ]
    audit = dict_rows(ROOT / "data/burushaski-indo-aryan-comparisons-audit.csv")

    assert comparisons
    projected = {
        form_id for form_id, row in forms.items()
        if row["Language_ID"] == "PBr"
        and "No reconstruction is proposed" in row["Etymology"]
    }
    assert len(projected) == len({row["Proto_Burushaski_ID"] for row in audit})
    assert all(not forms[form_id]["Form"] and not forms[form_id]["Original"] for form_id in projected)
    rank1 = {
        edge["Child_ID"]: edge["Parent_ID"]
        for edge in edges
        if edge["Rank"] == "1" and edge["Kind"] in {"reflex", "variant", "borrowed"}
    }

    def reaches_projected(form_id):
        seen = set()
        while form_id in rank1 and form_id not in seen:
            seen.add(form_id)
            form_id = rank1[form_id]
            if form_id in projected:
                return True
        return False

    compiled_descendants = sum(
        forms[form_id]["Language_ID"] == "Bur" and reaches_projected(form_id)
        for form_id in forms
    )
    # Three cleaned dialect variants are nested beneath source-linked Berger heads. They inherit
    # the grouping edge but do not independently assert the cross-family comparison.
    assert compiled_descendants == len(audit) + 3
    assert not any(
        edge["Kind"] == "borrowed"
        and forms[edge["Child_ID"]]["Language_ID"] == "Bur"
        and forms[edge["Parent_ID"]]["Language_ID"] == "Indo-Aryan"
        for edge in edges
    )
    for row in comparisons:
        assert forms[row["Entry_ID"]]["Language_ID"] == "PBr"
        assert forms[row["Compared_Entry_ID"]]["Language_ID"] == "Indo-Aryan"
        assert (row["Relation"], row["Direction"]) == ("related", "undetermined")
        assert row["Evidence"].strip()

    assert Counter(row["Status"] for row in audit) == {"converted": len(audit)}


def test_relationship_audit_and_review_sample_are_complete():
    audit = dict_rows(ROOT / "data/burushaski-indo-aryan-comparisons-audit.csv")
    sample = dict_rows(ROOT / "data/burushaski-indo-aryan-comparisons-sample.csv")

    assert len(audit) == 654
    assert len({row["Proto_Burushaski_ID"] for row in audit}) == 418
    assert len({item for row in audit for item in row["Comparison_IDs"].split("|")}) == 426
    assert sum(row["Claim_Source_Entry_ID"] != row["CDIAL_ID"] for row in audit) == 15
    assert len(sample) == 20
    assert {row["Review"] for row in sample} == {"ok"}
    def review_anchor(row):
        return row["Source_Key"] or row["Legacy_Form_ID"]

    assert [review_anchor(row) for row in sample] == [
        review_anchor(row) for row in reviewed_sample_candidates(audit)
    ]
