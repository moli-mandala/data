"""Slash-listed alternates become separate forms (YAML `forms.slash_alternates`).

SSNP survey cells such as ``ser / sɪr`` or ``kʌpɑl / sor`` give two responses for one prompt, and
nured writes word-level alternates (``vr̆ẽč ku-/bu-``). make_cldf.py expands them into one row
per alternate; Original keeps the whole source cell, the first alternate keeps the cell's
positional id, and later alternates take non-consuming ``<id>.<n>`` ids (plus a stable
``:variant:<n>`` child key when the source has record keys).
"""
from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SSNP = "data/other/forms/20260725-ssnp.csv"
NURED = "data/other/forms/20260818-nured-org.csv"


@pytest.fixture(scope="module")
def make_cldf():
    import make_cldf as module

    return module


def test_expansion_rules(make_cldf):
    expand = make_cldf.expand_slash_alternates
    assert expand("ser / sɪr") == ["ser", "sɪr"]
    assert expand("ʌŋgui k / ɑŋguɽ") == ["ʌŋgui k", "ɑŋguɽ"]
    assert expand("a / b / c") == ["a", "b", "c"]
    assert expand("tʌpeo /gʌrʌm") == ["tʌpeo", "gʌrʌm"]
    # an unspaced slash alternates the one word that carries it
    assert expand("jʌɳɑ/gʌriɑlɑ") == ["jʌɳɑ", "gʌriɑlɑ"]
    assert expand("vr̆ẽč ku-/bu-") == ["vr̆ẽč ku-", "vr̆ẽč bu-"]
    assert expand("púċën̆ tẽ/tavë̃") == ["púċën̆ tẽ", "púċën̆ tavë̃"]
    # an ending-only alternative (nured's feminine -i) cannot be spelled out: no extra row
    assert expand("malín̆/-i") == ["malín̆"]
    assert expand("puṇḍrë́/-í") == ["puṇḍrë́"]
    # a dangling separator is dropped; slash-free forms are untouched
    assert expand("ke kʊder/") == ["ke kʊder"]
    assert expand("kʌpɑl") == ["kʌpɑl"]


def test_only_declared_sources_split_slashes():
    import source_meta

    meta = source_meta.load()
    assert meta.file_flag(SSNP, "forms", "slash_alternates") is True
    assert meta.flag("nured", "forms", "slash_alternates") is True
    declared = {stem for stem, block in meta.files.items()
                if block.get("forms", {}).get("slash_alternates")}
    declared |= {key for key, block in meta.sources.items()
                 if block.get("forms", {}).get("slash_alternates")}
    assert declared == {"20260725-ssnp", "nured"}


def test_ssnp_alternates_are_rows_and_keep_positional_ids(make_cldf):
    with (ROOT / SSNP).open(encoding="utf-8", newline="") as stream:
        raw = list(csv.reader(stream))
    rows, _ = make_cldf.parse_file(SSNP, errors=io.StringIO(), file_num="F", param_counter={})
    expected_extra = sum(len(make_cldf.expand_slash_alternates(r[2])) - 1 for r in raw)
    assert sum("/" in r[2] for r in raw) == 925
    assert expected_extra == 961
    assert len(rows) == len(raw) + expected_extra
    assert len({row.id for row in rows}) == len(rows)
    # every source record keeps the positional id it had before alternates were split
    mains = [row for row in rows if "." not in row.id]
    assert [row.id for row in mains] == [f"F-{n}" for n in range(len(raw))]
    assert [row.old_form for row in mains] == [r[2] for r in raw]
    # later alternates come after the file's ordinary rows and point at their first alternate
    assert all("." not in row.id for row in rows[: len(raw)])
    by_id = {row.id: row for row in rows}
    alternates = rows[len(raw):]
    for row in alternates:
        main = by_id[row.id.rsplit(".", 1)[0]]
        assert row.variant_of == main.id
        assert row.old_form == main.old_form and row.source == main.source
        assert row.gloss == main.gloss and row.lang == main.lang
    # the cell from analysis.md: Goj 'ser / sɪr' at TSW, item 2
    cell = [row for row in rows if row.old_form == "ser / sɪr" and "location TSW, item 2]" in row.source]
    assert [(row.form, row.ipa) for row in cell] == [("ser", "ser"), ("sir", "sɪr")]
    kho = [row for row in rows if row.old_form == "kʌpɑl / sor" and "location KTR" in row.source]
    assert [row.form for row in kho] == ["kapāl", "sor"]


def test_nured_word_alternates_get_stable_child_keys(make_cldf):
    rows, _ = make_cldf.parse_file(NURED, errors=io.StringIO(), file_num="N", param_counter={})
    by_key = {row.entry_key: row for row in rows}
    assert len(by_key) == len(rows) == 265
    sell = by_key["nured:1318:form:2"]
    sold = by_key["nured:1318:form:2:variant:2"]
    assert (sell.form, sold.form) == ("vr̆ẽc ku-", "vr̆ẽc bu-")
    assert sold.variant_of_key == "nured:1318:form:2" and sold.variant_of == sell.id
    assert sold.id == f"{sell.id}.2" and sold.old_form == sell.old_form == "vr̆ẽč ku-/bu-"
    assert by_key["nured:1403:form:4:variant:2"].form == "púċën̆ tavë̃"
    # ending-only alternatives: the masculine citation form only, Original intact
    white_faced = by_key["nured:263:form:11"]
    assert (white_faced.form, white_faced.old_form) == ("malín̆", "malín̆/-i")
    assert not any("/" in row.form for row in rows)


def test_compiled_alternates_keep_the_cell_identity():
    """After the build the first alternate keeps the cell's durable id (and its curated
    etymology); the later alternate is a new node with the same Original and site tag."""
    csv.field_size_limit(2**31 - 1)
    with (ROOT / "cldf/forms.csv").open(encoding="utf-8", newline="") as stream:
        cell = [row for row in csv.DictReader(stream)
                if row["Original"] == "ser / sɪr" and "location TSW, item 2]" in row["Source"]]
    assert [(row["ID"], row["Form"]) for row in cell] == [
        ("f_5ms6sn2vha7tg", "ser"), (cell[1]["ID"], "sir")]
    assert cell[0]["Status"] == "" and cell[1]["Status"] == "unlinked"
    assert cell[0]["Tags"] == cell[1]["Tags"] == "dialect:Goj:SSNP-gojri-TSW:Ragushu"
    with (ROOT / "cldf/edges.csv").open(encoding="utf-8", newline="") as stream:
        parents = [row for row in csv.DictReader(stream) if row["Child_ID"] == "f_5ms6sn2vha7tg"]
    assert [(row["Kind"], row["Rank"]) for row in parents] == [("reflex", "1")]
