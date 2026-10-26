"""Record independent original reading; guarded against stale review inputs."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
SELECTION = HERE / "selection-20260927.json"


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    selection = json.loads(SELECTION.read_text())
    for name,digest in selection["inputs"].items(): assert sha(PACKAGE / name)==digest,name
    actual = {r[10]:r for r in csv.reader((PACKAGE / "proposed.csv").open(newline=""))}
    findings = []
    for entry in selection["selection"]:
        assert actual[entry["witness_key"]]==entry["actual_csv_row"]
        failed = entry["sample_index"]==9
        findings.append(dict(sample_index=entry["sample_index"],source_unit_key=entry["source_unit_key"],
            witness_key=entry["witness_key"],actual_csv_row=entry["actual_csv_row"],
            original_evidence=entry["source_locator"],original_image=entry["original_image"],
            literal_gloss_scope="pass", grammatical_metadata="material missing source-explicit person/number/possessive" if failed else "pass",
            material_error=failed, source_evidence="Original p41 genitive-example avariṭe is explicitly glossed ‘their’; third-person plural possessive, gender unspecified." if failed else "Actual row compared against original printed form/definition and its grammatical heading; source anomaly flags retained."))
    row = actual["nair2013malaveda:p41:genitive-examples:w1"]
    proposed = row[14] + " 3pl poss"
    assert row[3]=="their" and set(row[14].split()) & {"1sg","2sg","3sg","1pl","2pl","3pl","poss"}==set()
    graph = json.loads((PACKAGE / "graph-proposal.json").read_text())
    edges = []
    for edge in graph["proposals"]:
        assert edge["child_key"] in actual and edge["variant_of_key"] in actual
        assert actual[edge["child_key"]][2]==edge["child_form"]
        assert actual[edge["variant_of_key"]][2]==edge["head_form"]
        edges.append(dict(edge,independent_recommendation="supported as source alternate/free-variant pair; not a derivation or borrowing edge",
            original_evidence="p41 explicit free variation for dative/paired locatives" if edge["printed_page"]==41 else "p42 shared table cell alternatives or explicit imperative suffix -mi- or -miiru and paired examples; ambiguous II(Mas.) label remains unresolved"))
    lowlines = json.loads((PACKAGE / "underline-vector-census.json").read_text())
    assert len(lowlines["records"])==14
    lowline_review = [dict(printed_page=r["pdf_page"],original_pdf_rect=r["rect"],
        raw_original_line=r["candidate_characters"][0][2],independent_result="visible drawn lowline preserved on the corresponding n/l, distinct from dotbelow or tilde") for r in lowlines["records"]]
    report = dict(status="failed metadata class; extraction/gloss pass; frozen outputs unchanged",seed=selection["seed"],
        selection_sha256=sha(SELECTION),inputs_sha256=selection["inputs"],sample_units=20,sample_rows=20,
        literal_gloss_scope_passes=20,overall_passes=19,material_errors=1,findings=findings,
        exact_proposed_corrections=[dict(entry_key=row[10],field="Tags",old=row[14],proposed=proposed,
            source_evidence="Original p41 explicit English possessive ‘their’: add 3pl and poss, no gender inference.")],
        class_recommendation="Review all source-explicit case possessives/person glosses and the overt sister/kinship case examples; retain ambiguous English you number until exact source paradigm evidence supports it.",
        graph_review=dict(proposals=12,source_supported=12,applied_edges=0,findings=edges,
            held_non_edges="Existing morphological arrows/negative same-gloss examples/cross-language comparisons appropriately not promoted to variants or ancestry."),
        lowline_review=dict(count=14,all_originals_inspected=True,records=lowline_review),
        bibliography_review=dict(original_page=44,auxiliary_references=6,identities_years_institutions="pass",uncollated_status="correctly reported only; no auxiliary lexical citations",note="Two titles silently regularize printed Recounstruction and Morophology. Non-blocking scholarly identity agreement, but record raw title and normalization rather than imply an independently checked thesis title."),
        identity_review="Source-named base with unresolved catalogue/coordinates and source-qualified Veda variety is defensible; no invented local elicitation point.",
        canonical_edits=False,full_build_browser_claim=False)
    (HERE / "independent-original-output-review-pass1.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print("20/20 literal-gloss-scope pass; 19/20 overall,1metadata class failure;12graph proposals+14lowlines+6authorities reviewed")


if __name__=="__main__": main()
