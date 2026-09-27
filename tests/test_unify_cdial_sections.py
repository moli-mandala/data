import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from unify_cldf import (
    derivation_morpheme,
    ext_morpheme,
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


def test_with_morpheme_headers_are_extensions_only_for_pleonastic_suffixes():
    # Turner's "with -X-" phrasing of an extended stem, alongside "ext. -X-" and bare "-X-"
    assert ext_morpheme("ext. -<i>kk</i>-") == "-kk-"
    assert ext_morpheme("-<i>kk</i>-") == "-kk-"
    assert ext_morpheme("with -<i>ḍa</i>-") == "-ḍa-"
    assert ext_morpheme("With -<i>ll</i>-") == "-ll-"
    assert ext_morpheme("with anal. -<i>kk</i>-") == "-kk-"
    assert ext_morpheme("With unexpl. -<i>r</i>- (&lt; *pragāḍa- ?)") == "-r-"
    assert ext_morpheme("-<i>kk</i>- (?)") == "-kk-"
    assert section_kind("-<i>l</i>- or -<i>ll</i>-") == ("ext", "-l-", "ext:l")
    # sound substitutions, compounds, negations and non-pleonastic suffixes are not extensions
    assert ext_morpheme("With -<i>kk</i>- for -<i>tt</i>- (after MIA. type muccaï)") is None
    assert ext_morpheme("With -<i>r</i>- in place of -<i>ḍ</i>-") is None
    assert ext_morpheme("-<i>uḍa</i>- (&lt; *<smallcaps>kuḍa</smallcaps>-¹?") is None
    assert ext_morpheme("Without -<i>kka</i>-") is None
    assert ext_morpheme("with caus. suffix -<i>l</i>-") is None
    assert ext_morpheme("onom. with -<i>k</i>-, -<i>g</i>-") is None
    assert ext_morpheme("with -<i>ima</i>- after <smallcaps>paścimá</smallcaps>-") is None


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
    assert set(parent_languages) == {"Indo-Aryan"}

    branch_ids = {rank1[row["ID"]]["Parent_ID"] for row in explicit}
    assert len(branch_ids) == 2
    assert all("derived" in by_id[branch]["Tags"].split() for branch in branch_ids)
    component_parents = {
        branch: {edge["Parent_ID"] for edge in edges if edge["Child_ID"] == branch and edge["Kind"] == "component"}
        for branch in branch_ids
    }
    assert all(len(parents) == 2 for parents in component_parents.values())
