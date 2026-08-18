import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from unify_cldf import (
    derivation_morpheme,
    is_derivation_section,
    section_flags,
    section_kind,
)


def test_generic_derivation_sections_are_tags_not_promoted_entries():
    assert is_derivation_section("Deriv. vbs")
    assert derivation_morpheme("Deriv. vbs") is None
    assert section_kind("Deriv. vbs") == (None, None, None)
    assert section_flags("Deriv. vbs") == ["derived"]


def test_derivation_sections_with_explicit_morphemes_remain_branches():
    label = "Deriv. adj. with -<i>la</i>-"
    assert derivation_morpheme(label) == "-la-"
    assert section_kind(label) == ("deriv-morph", "-la-", "ext:la")
    assert section_flags(label) == ["derived"]

    historical = "Deriv. with -<i>er</i>- &lt; -<i>a-tara</i>-"
    assert derivation_morpheme(historical) == "-er-"
    assert section_kind(historical) == ("deriv-morph", "-er-", "ext:er")


def test_compiled_generic_derivatives_are_flattened_and_explicit_morphemes_are_not():
    with open("cldf/forms.csv", newline="", encoding="utf-8") as handle:
        forms = list(csv.DictReader(handle))
    by_id = {row["ID"]: row for row in forms}
    with open("cldf/edges.csv", newline="", encoding="utf-8") as handle:
        edges = list(csv.DictReader(handle))
    rank1 = {
        edge["Child_ID"]: edge
        for edge in edges
        if edge["Rank"] == "1" and edge["Kind"] in {"reflex", "borrowed", "variant"}
    }

    assert not any(row["Form"].endswith(" (deriv.)") for row in forms)

    section_rows = [
        row for row in forms
        if ":" in row["Cognateset"]
        and is_derivation_section(row["Cognateset"].split(":", 1)[1])
    ]
    explicit = [
        row for row in section_rows
        if derivation_morpheme(row["Cognateset"].split(":", 1)[1])
    ]
    generic = [row for row in section_rows if row not in explicit]

    assert len(explicit) == 8
    assert all("derived" in row["Tags"].split() for row in section_rows)
    assert all("CDIAL section:" in row["Etymology"] for row in generic)
    parent_languages = [
        by_id[rank1[row["ID"]]["Parent_ID"]]["Language_ID"]
        for row in generic
    ]
    assert set(parent_languages) == {"Indo-Aryan", "PNur"}
    assert parent_languages.count("PNur") == 3

    branch_ids = {rank1[row["ID"]]["Parent_ID"] for row in explicit}
    assert len(branch_ids) == 2
    assert all("derived" in by_id[branch]["Tags"].split() for branch in branch_ids)
    component_parents = {
        branch: {edge["Parent_ID"] for edge in edges if edge["Child_ID"] == branch and edge["Kind"] == "component"}
        for branch in branch_ids
    }
    assert all(len(parents) == 2 for parents in component_parents.values())
