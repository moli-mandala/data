import json
from pathlib import Path
import prepare_tag_proposal as proposal

HERE=Path(__file__).resolve().parent


def rows():
    return {row["witness_key"]:row for row in map(json.loads,(HERE/"independent-grammatical-tag-proposal.jsonl").read_text().splitlines())}


def key(short):
    return "grierson1906kolamibasim:p"+short


def test_exact_key_and_literal_guards():
    report=json.loads((HERE/"independent-grammatical-tag-review.json").read_text())
    source=proposal.ROOT/report["input_path"]
    assert proposal.sha(source)==report["input_sha256"]
    original={row["witness_key"]:row for row in map(json.loads,source.read_text().splitlines())}
    reviewed=rows()
    assert set(original)==set(reviewed) and len(reviewed)==224
    for k,row in reviewed.items():
        assert row["exact_before_guard"]=={name:original[k][name] for name in ("form","gloss","language_label","printed_label")}
        assert row["source_gloss_unchanged"]
        if row["original_literal_uncertainties_retained"] or row["typed_metadata_uncertainties"]:
            assert "uncertain" in row["proposed_tags"]


def test_conflict_and_marker_scope():
    reviewed=rows()
    disputed=reviewed[key("566:present-copula:3:independent:w1")]
    assert {"verb","copula","pres","sg","m","uncertain"}==set(disputed["proposed_tags"])
    assert disputed["typed_metadata_uncertainties"][0]["type"]=="source-numbering"
    for row in reviewed.values():
        if row["interpretation_class"] in ("source-morphological-element","documentary-symbol","documentary-hypothesis","quoted-embedded-analysis-component"):
            assert not {"verb","noun"}&set(row["proposed_tags"])
    assert "3sg" not in reviewed[key("563:negative:1:independent:w3")]["proposed_tags"]
    assert "2sg" not in reviewed[key("567:negative:1:independent:w5")]["proposed_tags"]
    assert reviewed[key("566:pronunciation-n-nd-d:1:independent:w11")]["interpretation_class"]=="documentary-hypothesis"


def test_explicit_categories_do_not_transfer_context():
    reviewed=rows()
    assert "m" not in reviewed[key("562:numerals:1:independent:w1")]["proposed_tags"]
    assert "m" in reviewed[key("562:numerals:1:independent:w2")]["proposed_tags"]
    assert "n" not in reviewed[key("566:possessive-agreement:1:independent:w1")]["proposed_tags"]
    assert "n" in reviewed[key("566:possessive-agreement:1:independent:w3")]["proposed_tags"]
    assert "inf" not in reviewed[key("567:finite-verbs:1:independent:w3")]["proposed_tags"]
    assert "noun" in reviewed[key("563:verbal-nouns:1:independent:w4")]["proposed_tags"]
    assert "impv" in reviewed[key("563:imperative:1:independent:w2")]["proposed_tags"]
    assert "2sg" not in reviewed[key("563:imperative:1:independent:w2")]["proposed_tags"]
