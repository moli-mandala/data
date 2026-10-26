"""Source-local graph recommendations; no accepted graph or row mutations."""
from collections import Counter
import json
from pathlib import Path
from prepare_tag_proposal import HERE, PACKAGE, ROOT, sha


def key(short):
    return "grierson1906kolamibasim:p"+short


def main():
    owner=PACKAGE/"grammar-review/reviewed-child-witnesses.jsonl"
    tag_path=HERE/"independent-grammatical-tag-proposal.jsonl"
    all_rows={r["witness_key"]:r for r in map(json.loads,owner.read_text().splitlines())}
    tags={r["witness_key"]:r for r in map(json.loads,tag_path.read_text().splitlines())}
    eligible={k:r for k,r in all_rows.items() if r["proposed_disposition"]=="eligible_review_candidate"}
    assert len(all_rows)==len(tags)==224 and len(eligible)==213
    links={}
    def variant(child,parent,basis,caveat):
        child,parent=key(child),key(parent)
        assert child in eligible and parent in eligible and child!=parent
        assert eligible[child]["language_label"]==eligible[parent]["language_label"]
        links[child]=dict(proposed_status="propose_source_explicit_variant", variant_of_witness_key=parent, source_claim=basis, scope_caveat=caveat, source_field="Variant_Of_Key", ordered_parents=[])
    def components(child,parents,basis,caveat):
        child=key(child); parents=list(map(key,parents))
        assert child in eligible and all(p in eligible and p!=child for p in parents)
        assert all(eligible[p]["language_label"]==eligible[child]["language_label"] for p in parents)
        links[child]=dict(proposed_status="propose_source_explicit_components",variant_of_witness_key="",ordered_parents=parents,source_claim=basis,scope_caveat=caveat,source_field="Derivation_Parent_Keys")
    same_cell=[
        ("562:personal-pronouns:5:independent:w2","562:personal-pronouns:5:independent:w1"),
        ("562:personal-pronouns:10:independent:w2","562:personal-pronouns:10:independent:w1"),
        ("562:personal-pronouns:10:independent:w3","562:personal-pronouns:10:independent:w1"),
        ("566:personal-pronouns:3:independent:w2","566:personal-pronouns:3:independent:w1"),
        ("566:personal-pronouns:8:independent:w2","566:personal-pronouns:8:independent:w1"),
        ("566:personal-pronouns:12:independent:w2","566:personal-pronouns:12:independent:w1"),
        ("566:personal-pronouns:12:independent:w3","566:personal-pronouns:12:independent:w1"),
        ("566:personal-pronouns:13:independent:w2","566:personal-pronouns:13:independent:w1")]
    for child,parent in same_cell:
        assert eligible[key(child)]["gloss"]==eligible[key(parent)]["gloss"]
        variant(child,parent,"Source explicitly comma-lists alternative personal-pronoun realizations under one printed gloss group.","Qualified source alternatives, not an assertion of universal free interchangeability; keep each exact optional-parenthesis form intact and preserve possessive/agreement qualifications.")
    variant("566:pronunciation-l-n:1:independent:w3","566:pronunciation-l-n:1:independent:w4","Source explicitly prints kōstel instead of kōsten, he sent, for its sometimes-L-for-n correspondence.","Direction follows printed replacement versus comparison form, not first-row order; source-specific pronunciation relation only, never a global n/l substitution.")
    variant("566:pronunciation-n-nd-d:1:independent:w4","566:pronunciation-n-nd-d:1:independent:w3","Source explicitly gives avan and avand, he, as examples of its n-to-nd correspondence.","Specific printed realization pair; no invented intermediate, no cross-language derivation.")
    for rank in (8,9):
        variant(f"566:pronunciation-n-nd-d:1:independent:w{rank}","566:pronunciation-n-nd-d:1:independent:w7","Source gives avanuṅg, avandun and avadun together, to him, illustrating nd and d realizations.","Retain source quantity and grammatical-use differences; pointer records this printed comparison set, not a general inflection rule.")
    variant("563:finite-verbs:1:independent:w7","563:finite-verbs:1:independent:w6","Source explicitly compares āṇḍā and āṇḍāt together, it is, within the third-person neuter discussion.","Exact same interpreted present copula use; do not apply to other person/TAM rows or expand optional letters elsewhere.")
    variant("566:present-copula-comparison:1:independent:w1","566:present-copula:4:independent:w1","Immediately after the present table, source says in third-person singular neuter we also find andīn, it is.","Pointer targets the actual third-person neuter row āṇḍād, never disputed2.m. āṇḍān; retain source spelling and quantity inconsistency.")
    variant("567:conjunctive-participle:1:independent:w4","567:conjunctive-participle:1:independent:w3","Source prints vāttān and vaddan together with the shared having-gone interpretation in its conjunctive-participle paragraph.","Printed corresponding forms, not inferred modern phonological normalization; no relation to following having-said example.")
    components("562:other-pronouns:1:independent:w4",["562:other-pronouns:1:independent:w3","562:genitive-locative:1:independent:w3"],"Source explicitly gives ad, that (neuter), genitive adnet; preceding case paragraph explicitly names net as genitive suffix.","Complete eligible ad and net endpoints exist in the same Kolami source scope. Ordered source analysis ad+net, no invented spelling or ancestor.")
    components("562:genitive-locative:1:independent:w9",["562:genitive-locative:1:independent:w10","562:genitive-locative:1:independent:w4"],"Source explicitly names locative t and prints vegāṭ, in the field, followed by parenthetical vegād, field.","Source-supported base-plus-locative analysis; surface vegāṭ is not a literal concatenation vegād+t. Preserve that junction qualification; no phonological conversion rule or generated form is proposed.")
    # The remaining eligible witnesses all receive a finite scope disposition.
    records=[]
    for k,row in eligible.items():
        section=row["physical_parent_key"].split(":")[2]
        rank=int(k.rsplit(":w",1)[1])
        proposed=dict(proposed_status="unlinked_no_explicit_relationship",variant_of_witness_key="",ordered_parents=[],source_field="",source_claim="Printed attestation or grammatical category; no additional source-local graph relationship is stated for this witness.",scope_caveat="Keep its exact source citation, analysis, tags and uncertainty; do not infer graph edges from matching meanings or forms.")
        if row["language_label"] not in ("Kolami","Basim / so-called Bhili"):
            proposed.update(proposed_status="unlinked_cross_language_comparison",source_claim="Explicitly language-labelled comparator; source comparison is not inheritance, borrowing or a source-local variant.",scope_caveat="No cross-language similarity edges or guessed common ancestor. Preserve author comparison text independently of graph.")
        elif tags[k]["interpretation_class"]=="source-morphological-element":
            proposed.update(proposed_status="unlinked_morphological_function",source_claim="Explicit grammatical ending/function is retained; function identity alone does not make distinct markers variants or derivations.",scope_caveat="Isolated marker is not a standalone lexical stem. An edge requires the full source-supported base and element endpoints, not a lone suffix.")
        elif section in ("noun-plural","nouns","accusative-dative","genitive-locative","negative","verbal-nouns","imperative","conjunctive-participle"):
            proposed.update(proposed_status="unlinked_incomplete_formation_endpoints",source_claim="Source supplies inflection/formation context, but an exact complete base/element pair for this individual form is not separately eligible in the grammar inventory.",scope_caveat="Do not peel an unattested base out of a word or whole phrase, use a similar other-language stem, or link only the available affix.")
        elif section in ("beat-paradigm","present-copula","past-copula","finite-verbs"):
            proposed.update(proposed_status="unlinked_paradigm_or_analysis",source_claim="Source explicitly gives finite/paradigm forms; person/TAM rows are not automatically lexical variants or derived from an arbitrarily selected finite cell.",scope_caveat="No conjectural copula stem, no past/agency-noun derivation, no forceful2.m. correction, and no speculative borrowing.")
        if section=="possessive-agreement":
            proposed.update(proposed_status="unlinked_qualified_component_claim",source_claim="Source says possessive forms contain neuter ad and compares agreement with Gondi, without printing an exact complete detachable base for each possessive.",scope_caveat="Retain contained-demonstrative/agreement claim in source analysis; no invented anne/inn base or unquoted Gondi node.")
        if section=="pronunciation-n-nd-d" and row["form"]=="amd":
            proposed.update(proposed_status="unlinked_hypothetical_ancestry",source_claim="Source says Kolami amd must be derived from avnd, but avnd is its explicitly hypothesized unstarred intermediate.",scope_caveat="No eligible avnd node, no graph edge to a hypothetical spelling or Basim analogue. Keep this source derivation assertion as free-text provenance.")
        if k in links: proposed=links[k]
        endpoints=([proposed["variant_of_witness_key"]] if proposed["variant_of_witness_key"] else [])+proposed["ordered_parents"]
        records.append(dict(witness_key=k, exact_before_guard={f:row[f] for f in ("form","gloss","language_label","printed_label")}, proposed_tags_guard=tags[k]["proposed_tags"], **proposed, endpoint_guards=[dict(witness_key=p,form=eligible[p]["form"],gloss=eligible[p]["gloss"],language_label=eligible[p]["language_label"]) for p in endpoints], original_evidence=dict(page=row["printed_page"],image=row["original_image"],image_sha256=row["original_image_sha256"]), source_provenance_retained=True, accepted_for_installation=False))
    assert len(records)==213 and {r["witness_key"] for r in records}==set(eligible)
    assert not any("avnd"==guard["form"] for r in records for guard in r["endpoint_guards"])
    output=HERE/"independent-grammar-graph-proposal.jsonl"
    output.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in records))
    report=dict(status="complete conservative graph proposal; independent root adjudication and final emitted-key closure required",input_sha256=sha(owner),tag_proposal_sha256=sha(tag_path),eligible_witnesses=213,accounted_unemitted=[dict(witness_key=k,form=r["form"],reason=r["disposition_reason"]) for k,r in all_rows.items() if k not in eligible],dispositions=dict(Counter(r["proposed_status"] for r in records)),proposed_variant_pointers=sum(bool(r["variant_of_witness_key"]) for r in records),proposed_component_groups=sum(bool(r["ordered_parents"]) for r in records),proposed_component_edges=sum(len(r["ordered_parents"]) for r in records),all_endpoint_keys_eligible=True,proposals_sha256=sha(output),canonical_changes=False,profile_refreshed=False,limitations=["These are witness-key endpoints, not minted durable form IDs; final exporter must map exact approved emitted keys and verify type/direction/order.","Source-specific alternatives retain their grammatical/contextual qualifications; variant graph does not claim unrestricted free interchangeability.","Two source formation proposals require root review of complete base/suffix scope, especially the nonconcatenative vegād+t→vegāṭ junction.","Final independent output audit/build/graph/browser gates are not performed."])
    (HERE/"independent-grammar-graph-review.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:report[k] for k in ("eligible_witnesses","dispositions","proposed_variant_pointers","proposed_component_groups","proposed_component_edges","proposals_sha256")},indent=2))


if __name__=="__main__":main()
