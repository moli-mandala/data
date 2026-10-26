"""Record bounded independent routing/configuration and source graph review."""
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent
W = P.parents[5]
rows = [json.loads(x) for x in (P / 'grammar-metadata-candidate.jsonl').read_text().splitlines()]
by_key = {r['source_unit_key']: r for r in rows}
prefix = 'grierson1906kui:'
def key(s):
    result = prefix + s
    assert result in by_key, result
    return result
def graph(child, parents, status, reason):
    return {'child_key': key(child), 'parent_keys': [key(x) for x in parents],
            'disposition': status, 'source_claim_and_decision': reason,
            'child_form': by_key[key(child)]['form'],
            'parent_forms': [by_key[key(x)]['form'] for x in parents]}

decisions = [
 graph('p463:negative-conjunctive-example:1', ['p462:principal-conjunctive-past-c1:1', 'p463:negative-conjunctive-morphology:1'], 'accept_ordered_components',
       'Original p463 explicitly forms the negative conjunctive compound by adding ār-aṅgā to the past conjunctive participle. The independently attested past conjunctive pāga is the correct first endpoint, not the homographic verbal noun. Keep source analysis and its literal fused boundary; this is not a variant edge.'),
 graph('p463:compound-example:4', ['p462:principal-conjunctive-past-c5:1', 'p463:copula-1-excl-c1:1'], 'accept_ordered_components',
       'Original p463 explicitly forms compound tenses from conjunctive participles and manba; gia-maĩ contains independently attested gia and maĩ, with compatible standard lect and person. Preserve the perfect construction rather than treating it as ordinary spelling variation.'),
 graph('p463:passive-example:1', ['p462:principal-verbal-noun-c5:1', 'p462:principal-relative-past-c4:1'], 'conditional_on_reviewed_alternate_endpoint',
       'Original p463 explicitly adds āwa to the verbal noun and gives giwa-āti. āti is independently present as the past relative form of āwa. First parent is currently the literal grouped cell giwa, gipka: select its reviewed giwa emission key, never point to the whole comma group as a component. Then two ordered component edges are supported. Preserve the source qualification that the passive is not in common use.'),
 graph('p463:compound-example:1', ['p462:principal-conjunctive-present-c5:1', 'p463:copula-1-excl-c1:1'], 'conditional_on_reviewed_optional_endpoint',
       'Explicit compound-tense construction. First parent is gi(pk)i; select a reviewed gipki optional-form emission before linking gipki-maĩ. Do not point to the notation group or invent an unattested new base.'),
 graph('p463:compound-example:2', ['p462:principal-conjunctive-present-c5:1', 'p463:copula-1-excl-c3:1'], 'conditional_on_reviewed_optional_endpoint',
       'Same explicit construction; use reviewed gipki and the first-person singular past massē endpoint, not homographic third-person massē.'),
 graph('p462:adjective-example:1', ['p462:adjective-morphology:1'], 'incomplete_component_set_hold',
       'Original p462 explicitly derives the adjective by adding gaṭṭa to a noun. Suffix endpoint exists; no independently attested dāṭā noun endpoint was found in the reviewed grammar. Do not serialize a single suffix as a complete derivational base or invent a dāṭā row. Retain the exact claim and incomplete component candidate.'),
 graph('p463:verbal-noun-example:1', [], 'unmatched_relative_participle_base',
       'Original p463 explicitly derives tinañju from a relative participle plus third-person pronoun ending. No independently attested matching tin(a) relative-participle endpoint occurs in the reviewed grammar. tinba to eat is not the stated immediate base; eñju has its own quantity uncertainty. Retain the derivation claim without fabricated endpoints.'),
 graph('p463:verbal-noun-example:2', ['p462:noun-prose:14'], 'incomplete_component_set_hold',
       'Explicit verbal-noun derivation; āri is attested but the matching relative-participle base is absent. Ordinary third-person pronoun āri is not automatically the isolated suffix analysis. Keep incomplete-component evidence, not a lone derived edge to the suffix.'),
]
decisions[2]['disposition'] = 'accept_after_approved_alternate_emission'
decisions[2]['approved_emission_parent_keys'] = [
 prefix+'p462:principal-verbal-noun-c5:1:v1',
 prefix+'p462:principal-relative-past-c4:1']
decisions[2]['parent_approval'] = 'Root message confirms root-grammar-alternative-emission-review-20260927.json approves giwa as :v1. Exporter must assert this key has literal giwa before installing the two-component relationship.'

inputs = ['grammar-metadata-candidate.jsonl', 'source-lect-mapping-approved-20260927.json',
          'outside-comparands-reviewed-candidate-20260927.jsonl',
          'outside-comparands-routing-reuse-review-20260927.json',
          'metadata-profile-proposal/metadata-proposal.json',
          'integration/prepare_scaffold.py', 'integration/scaffold-status.json']
report = {
 'reviewer': '/root/asur_finish', 'date': '2026-09-27',
 'status': 'Source interpretation/configuration proposal review; final CSV/YAML not yet available. Not installation approval.',
 'input_sha256': {s: hashlib.sha256((P/s).read_bytes()).hexdigest() for s in inputs},
 'routes': {
   'historical_Kalahandi': 'Kui + grierson1906_kui_kalahandi; retain conflicting modern Kuvi identification as source-specific uncertainty, never a global mapping.',
   'standard_table': 'Kui + grierson1906_kui_khondmals, from the explicit table header.',
   'literary Telugu': {'language_id': 'Telugu', 'dialect': None, 'note': 'Preserve source literary qualifier; no inferred chronological variety.'},
   'Malayalam': {'language_id': 'Malayalam', 'dialect': None},
   'Tamil dialects': {'language_id': 'Tamil', 'dialect': None, 'note': 'Unspecified source varieties; preserve grapheme holds and do not use a geographic dialect by guess.'},
   'Kaikadi of Berar': {'language_id': 'Kaikadi', 'dialect_id': 'lsi1906_kaikadi_berar', 'tag': 'dialect:Kaikadi:lsi1906-kaikadi:Berar'},
   'Tulu': {'language_id': 'Tulu', 'dialect': None},
   'Kōi': 'Never Kui. Proposed historical Gondi/koya route is explicitly historical and must not alter global modern Koya identity.',
   'Bhili of Basim': 'Hold historical identity; neither Parji nor northern Kolami/Naiki nor modern Bhili follows from a similarity claim.',
   'Burgandi': 'Hold identity; source closely related to Kaikadi does not establish identity.',
   'Korava unlocalized': 'Hold identity; do not apply the separately reviewed locality-specific Yerukula route to generic labels.',
   'p570 Kolami same forms': 'Documentary unresolved reference; three formerly expanded keys must not emit guessed Kolami attestations.',
   'generalized comparisons': 'Documentary hypotheses rather than reconstructed language records or graph edges.'},
 'configuration_findings': [
   {'item': 'etymology_provenance', 'required': 'Serialize source, an allowed enum, with descriptive scope explanation separate. Current metadata-proposal descriptive value is not directly valid.'},
   {'item': 'sound profile', 'required': 'Regenerate coverage from final emitted forms including accepted supplements/alternates. Existing first-reading profile inventory is not final coverage evidence.'},
   {'item': 'scope isolation', 'finding': 'Current scaffold writes only integration files; no broad language pronunciation/default mutation found. Actual final YAML is pending.', 'required': 'Route literal preservation only in this source YAML/profile; no global Kui/Kuwi/Koya profile change.'},
   {'item': 'parser flags', 'required': 'Verify actual generic parser with source-level split_alternates false, exact Original punctuation/case/hyphens and chosen explicit alternate keys. No invented Phonemic from uncertain graphemes.'},
   {'item': 'references', 'required': 'Resolve explicit keys for auxiliary quoted authorities when their claims are cited: Letchmajee, Smith, Frye, Friend-Pereira. Mark reported through Grierson, not independently collated. Check each emitted citation and SourceMeta/Bib formatting.'},
   {'item': 'reuse', 'finding': 'Earlier independent routing report verifies 12 Naiki same-source canonical rows exactly; preserve their existing keys, full analyses and citations.'},
   {'item': 'holds', 'required': 'Keep literal typography holds distinct from unresolved language identity and documentary scope. No blank source cell becomes a guessed word.'}],
 'graph_decisions': decisions,
 'graph_serialization': 'Use ordered Derivation_Parent_Keys only for complete accepted two-parent component sets. Existing unify_cldf feeds ordered section edges to edges_build, which classifies two non-root parents as components. Verify source-local edge output and endpoint existence before installation. No full build performed.',
 'ordinary_inflection': 'No blanket derived/variant edges between all paradigm cells. Gender/number/case/tense contrasts are not free variants. Source comparisons and conjectural historical analyses remain attributed prose unless a specific relation with unambiguous endpoints is reviewed.',
 'limits': ['This bounded graph review covers the named derivation/compound classes on original pp462–463; it is not a claim to have accepted every potential graph in the whole book.', 'Final emitted keys may differ after reviewed expansion/reuse; map physical keys explicitly and validate each target, never fuzzy-match.'],
 'canonical_changes': False,
}
(P/'independent-integration-routing-graph-review-20260927.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
print('Saved independent routing/configuration review and', len(decisions), 'keyed graph decisions')
