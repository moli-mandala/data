import json
from pathlib import Path
import prepare_graph_proposal as graph

HERE=Path(__file__).resolve().parent


def test_eligible_coverage_and_exact_endpoint_guards():
    source=graph.PACKAGE/"grammar-review/reviewed-child-witnesses.jsonl"
    owner={r["witness_key"]:r for r in map(json.loads,source.read_text().splitlines())}
    proposals={r["witness_key"]:r for r in map(json.loads,(HERE/"independent-grammar-graph-proposal.jsonl").read_text().splitlines())}
    eligible={k for k,r in owner.items() if r["proposed_disposition"]=="eligible_review_candidate"}
    assert len(proposals)==213 and set(proposals)==eligible
    for key,row in proposals.items():
        assert row["exact_before_guard"]=={field:owner[key][field] for field in ("form","gloss","language_label","printed_label")}
        for endpoint in row["endpoint_guards"]:
            target=owner[endpoint["witness_key"]]
            assert endpoint["witness_key"] in eligible and endpoint["witness_key"]!=key
            assert endpoint["form"]==target["form"]!="avnd"
            assert endpoint["language_label"]==target["language_label"]==owner[key]["language_label"]


def test_explicit_variant_direction_and_component_order():
    rows={r["witness_key"]:r for r in map(json.loads,(HERE/"independent-grammar-graph-proposal.jsonl").read_text().splitlines())}
    replacement=rows[graph.key("566:pronunciation-l-n:1:independent:w3")]
    assert replacement["variant_of_witness_key"]==graph.key("566:pronunciation-l-n:1:independent:w4")
    assert [r["form"] for r in replacement["endpoint_guards"]]==["kōsten"]
    genitive=rows[graph.key("562:other-pronouns:1:independent:w4")]
    assert [r["form"] for r in genitive["endpoint_guards"]]==["ad","net"]
    assert rows[graph.key("566:present-copula-comparison:1:independent:w1")]["variant_of_witness_key"]==graph.key("566:present-copula:4:independent:w1")
    for row in rows.values():
        if row["proposed_status"] in ("unlinked_cross_language_comparison","unlinked_hypothetical_ancestry","unlinked_qualified_component_claim"):
            assert not row["variant_of_witness_key"] and not row["ordered_parents"]
