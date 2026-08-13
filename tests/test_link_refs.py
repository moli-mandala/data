import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1]))

from link_refs import extract_derivations, extract_roots


def test_cf_bracket_is_not_parsed_as_an_etymology():
    params = [{
        "ID": "5550",
        "Description": (
            "<number>5550</number> <b>ḍimba</b> [Cf. <i>ḍimbikā</i>- — perh. belongs "
            'to group of *<smallcaps><a data-entry="5549a">ḍibba</a></smallcaps>-]'
        ),
    }]

    assert extract_derivations(params) == []


def test_non_cf_ancestry_after_commentary_is_still_parsed():
    params = [{
        "ID": "child",
        "Description": (
            '[possibly related — <smallcaps><a data-entry="parent">parent</a></smallcaps>-]'
        ),
    }]

    assert extract_derivations(params) == [("child", "parent")]


def test_cf_structured_dash_and_colon_suffixes_are_parsed():
    params = [
        {
            "ID": "compound-dash",
            "Description": (
                '[Cf. comparison. — <smallcaps><a data-entry="left">left</a></smallcaps>-, '
                '<smallcaps><a data-entry="right">right</a></smallcaps>-]'
            ),
        },
        {
            "ID": "compound-colon",
            "Description": (
                '[Cf. comparison: <smallcaps><a data-entry="left">left</a></smallcaps>-, '
                '<smallcaps><a data-entry="right">right</a></smallcaps>-]'
            ),
        },
        {
            "ID": "shared-smallcaps",
            "Description": (
                '[Cf. <smallcaps>comparison — <a data-entry="left">left</a>, '
                '<a data-entry="right">right</a></smallcaps>-]'
            ),
        },
        {
            "ID": "spaced-hyphen",
            "Description": (
                '[Cf. comparison - <smallcaps><a data-entry="left">left</a>, '
                '<a data-entry="right">right</a></smallcaps>-]'
            ),
        },
        {
            "ID": "bound-component-hyphen",
            "Description": (
                '[Cf. <smallcaps>comparison. -<a data-entry="left">left</a>, '
                '<a data-entry="right">right</a></smallcaps>]'
            ),
        },
    ]

    assert extract_derivations(params) == [
        ("compound-dash", "left"),
        ("compound-dash", "right"),
        ("compound-colon", "left"),
        ("compound-colon", "right"),
        ("shared-smallcaps", "left"),
        ("shared-smallcaps", "right"),
        ("spaced-hyphen", "left"),
        ("spaced-hyphen", "right"),
        ("bound-component-hyphen", "left"),
        ("bound-component-hyphen", "right"),
    ]


def test_cf_prose_suffix_is_not_parsed_even_when_it_contains_a_link():
    params = [{
        "ID": "child",
        "Description": (
            '[Cf. comparison. — See list s.v. '
            '<smallcaps><a data-entry="parent">parent</a></smallcaps>-]'
        ),
    }]

    assert extract_derivations(params) == []


def test_cf_root_requires_a_structured_suffix():
    params = [
        {
            "ID": "structured",
            "Description": "[Cf. comparison. — √<smallcaps>gam</smallcaps>]",
        },
        {
            "ID": "comparison-only",
            "Description": "[Cf. √<smallcaps>car</smallcaps>]",
        },
        {
            "ID": "root-before-prose",
            "Description": (
                "[Cf. <smallcaps>comparison: √gam</smallcaps> — Or possibly something else]"
            ),
        },
        {
            "ID": "semicolon-root",
            "Description": "[Cf. comparison; √<smallcaps>gam</smallcaps>]",
        },
    ]

    root_params, edges, _root_map = extract_roots(params)

    # The comparison-only root remains a link target, but creates no ancestry edge. Keeping it in
    # the root catalog also prevents later r… IDs from being renumbered.
    assert [row["Name"] for row in root_params] == ["√car", "√gam"]
    assert edges == [
        ("structured", "r2"),
        ("root-before-prose", "r2"),
        ("semicolon-root", "r2"),
    ]
