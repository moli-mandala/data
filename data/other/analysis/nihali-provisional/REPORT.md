# Provisional etymology of the Jambu Nihali lexicon

## Scope, units, and headline

This analysis covers **4,299 attested Nihali database records** and assigns each one a provisional rank-1 hypothesis. A conservative form-plus-meaning comparison across the five lexical sources collapses them to **3,277 normalized lexeme clusters**. These clusters, rather than raw dictionary rows or proxy IDs, are the least misleading unit for historical proportions. The overlay creates 2,743 explicit proxy entries where no existing Jambu etymon can safely carry the claim.

At lexeme-cluster level, 2,143 (65.4%) have an externally attributed contact source and 1,133 (34.6%) remain in the Nihali residue. The remaining 1 cluster is retained as Other. At record level the corresponding figures are 2,950 (68.6%) external and 1,348 (31.4%) residue, with 1 Other. Source-attributed, cross-dictionary-propagated, manually adjudicated, or previously curated evidence supports 2,979 records.

The resulting historical hypothesis is that Nihali is best treated as the sole documented survivor of an **independent Central Indian lineage that underwent layered relexification**. Indo-Aryan is the numerically widest attribution in the expanded database, while Korku/Munda is the most historically diagnostic contact channel; other Munda comparisons may include an older layer, and Dravidian is smaller but real. Lexical comparison alone does not locate the independent lineage's earlier homeland, date it, or demonstrate a relationship to another isolate.

## Lexeme-cluster results

| Stratum | Lexeme clusters | Share |
|---|---:|---:|
| Nihali residue | 1,133 | 34.6% |
| Indo-Aryan | 935 | 28.5% |
| Korku | 586 | 17.9% |
| Korku+Indo-Aryan | 315 | 9.6% |
| Dravidian | 106 | 3.2% |
| Munda | 70 | 2.1% |
| Korku+Munda | 56 | 1.7% |
| Indo-Aryan+Dravidian | 18 | 0.5% |
| Indo-Aryan+English | 12 | 0.4% |
| Munda+Indo-Aryan | 10 | 0.3% |
| Korku+Munda+Indo-Aryan | 10 | 0.3% |
| Korku+Dravidian | 7 | 0.2% |
| Munda+Dravidian | 5 | 0.2% |
| Korku+Indo-Aryan+Dravidian | 5 | 0.2% |
| English | 4 | 0.1% |
| Korku+English | 2 | 0.1% |
| Other | 1 | 0.0% |
| Munda+Indo-Aryan+Dravidian | 1 | 0.0% |
| Korku+Munda+Dravidian | 1 | 0.0% |

### Family involvement (non-exclusive)

Mixed labels count under each named family in this table; rows therefore do not sum to the lexeme total.

| Named contact family | Lexeme clusters | Share of all clusters |
|---|---:|---:|
| Indo-Aryan | 1,306 | 39.9% |
| Korku | 982 | 30.0% |
| Munda | 153 | 4.7% |
| Dravidian | 143 | 4.4% |
| English | 18 | 0.5% |

### Contact evidence tiers

The family counts above preserve all printed attributions. This second view separates resolved parents from recheckable but unresolved comparisons and from explicitly qualified or weak evidence. Families remain non-exclusive.

| Family | Total labelled | Resolved parent/route | Recheckable source | Qualified | Questioned/weak |
|---|---:|---:|---:|---:|---:|
| Indo-Aryan | 1,306 | 428 | 727 | 67 | 84 |
| Korku | 982 | 10 | 808 | 85 | 79 |
| Munda | 153 | 7 | 90 | 24 | 32 |
| Dravidian | 143 | 87 | 26 | 13 | 17 |
| English | 18 | 5 | 11 | 0 | 2 |

Across all families, 557 clusters have an external parent node. An internal variant edge is not counted as external evidence unless its curated chain actually terminates at such a node; otherwise the cluster is graded from its source proxy. The remaining tiers preserve source hypotheses without pretending they are equally resolved. Every cluster and basis is listed in `nihali-contact-evidence-tier-audit.csv`; `nihali-family-contact-evidence-audit.csv` grades each family separately so one resolved member of a mixed label cannot promote the others.

A sensitivity bracket makes the consequence of those tiers explicit. The floor retains only family-specific resolved parents and manually corroborated proxies. The supported envelope additionally retains qualified resolved links, explicit unqualified and propagated source evidence, and manually plausible or route-ambiguous proxies, but excludes label-only, questioned, and manually weak/unresolved cases. These are evidence thresholds, not statistical confidence intervals.

| Family | High-specificity floor | Supported envelope | All labelled |
|---|---:|---:|---:|
| Indo-Aryan | 452 | 1,222 | 1,306 |
| Korku | 30 | 903 | 982 |
| Munda | 8 | 121 | 153 |
| Dravidian | 96 | 126 | 143 |
| English | 5 | 16 | 18 |

The row-level definitions and shares are in `nihali-family-evidence-bracket-audit.csv`. The wide gaps, especially for Indo-Aryan and Korku, are a warning against presenting the headline family labels as solved etymologies.

The contrast is historically informative: non-Korku Munda has only 8 high-specificity claims, compared with 96 Dravidian and 452 Indo-Aryan. Korku's floor is only 30, but its supported envelope reaches 903: most Korku evidence identifies a proximate route in the lexical sources rather than a terminal Korku parent node. This is why the audit supports profound Korku contact without converting the much smaller non-Korku Munda set into proof of Munda descent.

A separate route-recovery test checks all 854 Korku-labelled source-proxy clusters against the independently ingested Korku lexicon, excluding the provisional proxy forms themselves. It recovers 86 close form-and-meaning matches and 28 possible matches; 405 have only a close string, 297 remain weak or unmatched, and 38 lack a recoverable printed comparison form. This is deliberately stricter than accepting a dictionary's donor label at face value.

Among the 86 strongest route matches, 8 Korku forms have an accepted upstream Munda parent and 78 have no upstream etymology in the current graph. The latter count is a Korku etymological-coverage gap, not evidence that those words originated in Korku. It sharply limits any attempt to convert an immediate Korku route into an ultimate Munda percentage. Every selected Korku form, score, source, and upstream path is in `nihali-korku-route-audit.csv`.

The Indo-Aryan layer is equally non-monolithic. Of 1,306 Indo-Aryan-involved clusters, 1,032 explicitly name modern Hindi, Marathi, Bengali, or Konkani comparanda, 33 name Sanskrit without a modern IA language, and 24 name both. Another 127 use only a generic IA label, 76 resolve without a specific language in the source note, and 14 remain label-only at that granularity. 330 clusters also mention a Korku route. These labels distinguish comparative evidence, not borrowing dates, but they make a single prehistoric Indo-Aryan layer untenable. Row-level named languages, parents, and cautions are in `nihali-indo-aryan-route-profile.csv`.

### Diagnostic basic vocabulary

After removing 14 manually verified concept-linking false positives, a fixed 93-domain Swadesh-style screen finds 292 lexeme clusters linked to 91 covered concepts (HEART and SWIM have no linked Nihali record). Exclusions are listed in `nihali-core-vocabulary-exclusions.csv`. This is a diagnostic slice, not a replacement-rate calculation: synonyms and multiple source forms can create more than one cluster per concept.

| Stratum | Conservative baseline | Variant-propagation sensitivity |
|---|---:|---:|
| Nihali residue | 157 (53.8%) | 91 (31.2%) |
| Indo-Aryan | 39 (13.4%) | 50 (17.1%) |
| Korku | 37 (12.7%) | 51 (17.5%) |
| Dravidian | 18 (6.2%) | 34 (11.6%) |
| Korku+Indo-Aryan | 16 (5.5%) | 19 (6.5%) |
| Munda | 11 (3.8%) | 25 (8.6%) |
| Korku+Munda | 8 (2.7%) | 15 (5.1%) |
| Munda+Indo-Aryan | 3 (1.0%) | 3 (1.0%) |
| Other | 1 (0.3%) | 1 (0.3%) |
| Korku+Munda+Indo-Aryan | 1 (0.3%) | 2 (0.7%) |
| Korku+Indo-Aryan+Dravidian | 1 (0.3%) | 1 (0.3%) |

The residue accounts for 157 of 292 core-vocabulary clusters (53.8%), compared with 34.6% of all lexeme clusters under the strict baseline. However, 66 manually reviewed forms are plausible variants or transparent derivational/inflectional relatives of separately clustered source-attributed Nihali forms. Propagating those comparisons only in a sensitivity pass reduces the core residue to 91 (31.2%). The apparent core enrichment is therefore not robust to conservative under-clustering and must not be used as positive evidence for an independent lineage. The repeated residue remains historically important, but its interpretation rests on item-level evidence and the absence of demonstrated regular correspondences rather than this proportion.

The same under-clustering risk was screened across the full lexicon. An exhaustive review register contains 205 residual-to-contact pairs: 172 are judged lexical variants, 32 remain qualified, and 1 is rejected. Propagating only the clearer variants would reduce the strict residue from 1,133 to 961 clusters; including qualified cases gives a broad sensitivity floor of 929. The 204 supported or qualified rows involve 75 Indo-Aryan, 101 Korku, 39 Munda, and 25 Dravidian reference strata (non-exclusive). This sensitivity does not alter installed graph edges: it measures how much source-level attribution is hidden by conservative lexical clustering. Every pair, score, decision, and rationale is in `nihali-global-variant-sensitivity-audit.csv`.

Counting each covered basic concept once gives a more stable picture: 17 concepts are residue-only, 22 retain both residue and contact forms, 21 are contact-only with one named family, and 31 are contact-only with multiple named families. Thus 39/91 concepts retain at least one effective residual root, while 52/91 are contact-only under the current hypotheses. Contact involvement is also distributed rather than unitary: Korku occurs in 46 concepts, Indo-Aryan in 42, Munda in 20, and Dravidian in 13. This balance fits layered replacement better than descent from any single donor layer, but residual presence still means unmatched rather than inherited. Crucially, 16/17 residue-only concepts have at least one root attested in multiple sources and 14/17 have a root in Konow or Bhattacharya; the residue-only concepts supported solely by one source are WALK. Among the 22 mixed residue-plus-contact concepts, 11 have a replicated residual root. This establishes a stable residual lexicon without converting it into a demonstrated lineage. The 91 concept-level rows are in `nihali-core-concept-origin-profile.csv`.

The stricter Korku route-recovery screen contains 54 core-vocabulary proxy clusters: 14 independently recover a close Korku form with compatible meaning, 6 are possible, 18 recover only a misleading or semantically unsupported string, 9 are weak/unmatched, and 7 lack a recoverable form. The source attributions remain recorded, but only the first two categories independently corroborate a Korku route in the current database.

A second manual sensitivity pass collapses the 91 still-residual clusters to citation-form and dialect-variant root groups within each concept. It yields 56 provisional root hypotheses across 39 concepts. This is not a Proto-Nihali reconstruction: it is a denominator correction that prevents forms such as five separately clustered transcriptions of 'head' from counting as five independent roots. Groupings and rationales are in `nihali-core-residue-root-audit.csv`; multi-cluster decisions are maintained in `nihali-core-residue-root-review.csv`.

The resulting root inventory contains 16 roots attested in four or five sources, 9 in three, 4 in two, and 27 in one. 33 are present in Konow (1906) or Bhattacharya (1957). This measures documentary replication, not linguistic age: old loans can be stable, and single-source items can be genuine. Actual forms, source lists, alternatives, and a caution on every row are in `nihali-core-residue-root-inventory.csv`.

Closed-class material is not uniformly replaced. The concept-linked inventory gives the following effective-stratum profile after applying the reviewed core-variant sensitivity where available (family columns are non-exclusive):

| Domain | Linked clusters | Residue | Indo-Aryan | Korku | Munda | Dravidian |
|---|---:|---:|---:|---:|---:|---:|
| pronoun | 22 | 17 | 0 | 3 | 2 | 1 |
| demonstrative | 15 | 5 | 1 | 9 | 0 | 0 |
| interrogative | 15 | 11 | 1 | 0 | 0 | 3 |
| polarity | 8 | 6 | 1 | 2 | 0 | 0 |
| low numeral | 37 | 1 | 16 | 1 | 3 | 18 |

Pronouns, deixis, interrogatives, and polarity retain many residual forms, while the low numerals are much more heavily assigned to Indo-Aryan or Dravidian. This asymmetry is compatible with relexification of a pre-contact grammatical lexicon, but it is not a genetic proof: these linked clusters contain under-merged variants and synonyms, and contact can affect closed classes too. The complete, row-level diagnostic is in `nihali-closed-class-diagnostic-audit.csv`.

The strict core slice contains 23 Munda-involved clusters covering only 18 concepts. The family-specific audit finds just 2 high-specificity corroborated Munda claims; 1 resolved link is qualified, 1 is manually weak, 15 are explicit unhedged source comparisons, and 3 are label-only. Variant propagation raises the row count to 45 but adds no new concepts (20 total). This is replication of the same proposed comparisons, not an expanding cognate set, and it supplies no regular Nihali–Munda sound correspondence system.

### Evidence basis by lexeme cluster

| Evidence basis | Lexeme clusters |
|---|---:|
| record-local source/editorial note | 1,847 |
| unresolved residue | 1,069 |
| existing-curated | 211 |
| manually accepted source-free lead | 114 |
| manually deferred computational lead | 18 |
| manually rejected computational lead | 18 |

The evidence-basis buckets are mutually exclusive. One of the 115 accepted reviewed leads belongs to a multi-record cluster that already contains a curated edge, so it is counted under `existing-curated` rather than again under manually accepted leads.

There are 138 clusters for which the lexical sources preserve different donor labels. In 93 cases one label nests inside another, typically reflecting immediate Korku transmission versus an Indo-Aryan or other ultimate source. The remaining 45 cases have disjoint labels. Manual adjudication treats 26 as plausible staged contact chains, favors Indo-Aryan in 8, Korku in 4, and Munda in 1; 6 remain unresolved. The union labels stay in the audit so this adjudication does not erase the printed alternatives. Full evidence and rationales are in `nihali-source-label-variation-audit.csv`.

Cross-dictionary propagation extends source/editorial evidence to 468 records in 359 conservatively matched clusters.

All 118 threshold-generated source-free leads received an explicit human decision: 95 had a clear margin over alternatives and 23 were family-level ties. A separate reviewed escape hatch covers 25 transparent cultural loans that string scoring misses. A fourth register manually checks 19 low-fit or semantically deceptive parent resolutions where the donor attribution itself remains source-supported. Across the four registers, 115 were accepted, 29 rejected, and 18 deferred. Decisions and rationales are preserved in the audit tables, including `nihali-transparent-loan-review.csv` for the cultural-loan exceptions and `nihali-source-parent-review.csv` for source-parent corrections. Rejected and deferred source-free leads remain Nihali-residue proxy hypotheses; rejected source-parent matches retain their printed donor stratum as unresolved proxies. Within the parent-head register, 8 links were redirected to the source-supported head and 11 were downgraded to unresolved proxies.

A separate morpheme-level review finds 23 otherwise residual expressions with recognizable contact material: 18 transparent components, 4 qualified components, and 1 quarantined source/gloss conflict. Layer mentions are non-exclusive (Indo-Aryan 23, Munda 1, Other 1). These rows remain residual in the strict graph because component analysis alone does not create a whole-expression rank-1 edge. Most retain unresolved Nihali material; a fully contact-composed case is removed only in the explicit sensitivity pass. See `nihali-residue-contact-component-review.csv`.

### Residue threshold sensitivity

A score-only stress test deliberately relaxes the production gates for clusters that still remain residual. It does not create assignments:

| Threshold | Minimum score | Minimum margin | Flagged | Indo-Aryan | Dravidian | Munda | Unreviewed |
|---|---:|---:|---:|---:|---:|---:|---:|
| very loose | 0.650 | 0.025 | 479 | 256 | 190 | 32 | 449 |
| loose | 0.700 | 0.025 | 334 | 178 | 133 | 22 | 304 |
| moderate | 0.750 | 0.025 | 171 | 86 | 68 | 16 | 141 |
| high score, low margin | 0.800 | 0.025 | 70 | 38 | 22 | 9 | 40 |
| production-like score/margin only | 0.800 | 0.045 | 65 | 36 | 22 | 6 | 39 |
| very high score | 0.850 | 0.045 | 16 | 9 | 5 | 1 | 0 |

At the production-like composite score and margin alone, 65 residual clusters would be flagged, but 39 never passed the required component-level form, meaning, and donor-specific gates. At the very-high score setting, 16 remain, of which 16 were already rejected or deferred by manual review. Looser settings mainly generate Indo-Aryan and Dravidian candidates, reflecting donor-database size as much as history; etymologised Korku coverage is especially sparse. This is why lowering one cutoff cannot convert the residue into evidence for a particular family. Full definitions are in `nihali-residue-threshold-sensitivity.csv`.

### Shape of resolved external links

The following table counts distinct lexeme-to-parent links that reach an actual external Jambu node, excluding unresolved donor proxies. Parent shape compares the Nihali form with the stored parent, which may be reconstructed or ultimate. Surface shape instead uses the best automatically selected observed descendant under that parent. Both are descriptive, not sound-law tests.

| Parent family | Links | Parent exact/near | Surface exact/near | Surface moderate | Surface distant/unavailable |
|---|---:|---:|---:|---:|---:|
| Indo-Aryan | 422 | 78 | 276 | 89 | 57 |
| Dravidian | 98 | 8 | 37 | 21 | 40 |
| Munda | 33 | 10 | 17 | 1 | 15 |
| Other | 13 | 7 | 9 | 1 | 3 |
| English | 5 | 0 | 0 | 3 | 2 |

Resolution is mostly to a comparative index or reconstructed node, not to an observed surface donor: 362 links terminate at the generic Indo-Aryan node, 58 at Proto-Indo-Iranian, all 98 Dravidian links at Proto-Dravidian, and the Munda links split between Proto-Kherwarian (16) and Proto-Munda (17), with 5 direct English and 13 Persian links. Thus 'resolved' means that a database etymon was identified; it does not by itself identify the immediate donor, borrowing date, or direction.
The observed-surface column is a reproducible diagnostic rather than manual donor identification. It can recover a relevant reflex hidden beneath an awkward canonical display form, but dense etymological families can also supply an accidentally attractive surface. The selected IDs, forms, languages, glosses, and both similarity scores remain inspectable in `nihali-resolved-contact-shape-audit.csv`.

The source labels and resolved parents answer different historical questions. The matrix below counts which named route families occur in the source stratum of each resolved ultimate-parent link; route columns are non-exclusive.

| Resolved parent family | Links | Korku-labelled route | Munda-labelled route | Indo-Aryan-labelled route | Dravidian-labelled route |
|---|---:|---:|---:|---:|---:|
| Indo-Aryan | 422 | 112 | 3 | 421 | 5 |
| Dravidian | 98 | 8 | 4 | 14 | 90 |
| Munda | 33 | 24 | 23 | 6 | 1 |
| Other | 13 | 0 | 0 | 13 | 0 |
| English | 5 | 1 | 0 | 4 | 0 |

Most importantly, 112 of the 422 links that resolve to an Indo-Aryan parent carry a Korku route label, and 24 of the 33 Munda-parent links do so. This is direct database evidence that Korku is often the proximate conduit rather than a sufficient statement of ultimate ancestry.

The Munda subset has 33 resolved links, of which 10 are exact or near. Near identity is compatible with borrowing, while the more distant pairs require recurring correspondences before they can support inheritance. Manual review reduces the 33 links to 28 distinct Munda parent roots and classifies 11 as near contact-compatible, 15 as possible correspondences, and 7 as weak. The only repeated non-identity proposal is initial c~s in 5 links representing four parent roots; one root ('dance') is duplicated across two Nihali citation clusters. Four semantic sets are too few, and their remaining segments too heterogeneous, to establish a Nihali–Munda sound law. The full form-shape inventory and all 33 rationales are in `nihali-resolved-contact-shape-audit.csv` and `nihali-munda-correspondence-review.csv`.

The parallel Dravidian diagnostic collapses 98 resolved links to 66 parent roots. On a deliberately mechanical joint form-and-meaning screen, 22 roots are near contact-compatible and 12 are possible comparisons, while 27 have weak form fit and 5 have weak semantic fit. 45 roots begin with identity mappings; the only repeated non-identity initials are b~v (3 roots), h~k (2 roots), g~k (2 roots). These sparse patterns mix unrelated meanings and do not establish a Nihali–Dravidian sound law. The diagnostic nevertheless supports a genuine contact layer: close matches such as the low numerals, 'dog', 'cat', and 'cotton' are exactly the shapes that borrowing can preserve. Root-level rows and the reproducible thresholds are in `nihali-dravidian-correspondence-audit.csv`.

## Record-level audit results

| Stratum | Records | Share |
|---|---:|---:|
| Nihali residue | 1,348 | 31.4% |
| Indo-Aryan | 1,174 | 27.3% |
| Korku | 769 | 17.9% |
| Korku+Indo-Aryan | 521 | 12.1% |
| Dravidian | 166 | 3.9% |
| Korku+Munda | 110 | 2.6% |
| Munda | 93 | 2.2% |
| Indo-Aryan+Dravidian | 31 | 0.7% |
| Korku+Munda+Indo-Aryan | 23 | 0.5% |
| Munda+Indo-Aryan | 15 | 0.3% |
| Indo-Aryan+English | 14 | 0.3% |
| Korku+Dravidian | 13 | 0.3% |
| Korku+Indo-Aryan+Dravidian | 8 | 0.2% |
| Munda+Dravidian | 5 | 0.1% |
| English | 4 | 0.1% |
| Korku+English | 2 | 0.0% |
| Other | 1 | 0.0% |
| Munda+Indo-Aryan+Dravidian | 1 | 0.0% |
| Korku+Munda+Dravidian | 1 | 0.0% |

### Method

| Method | Records |
|---|---:|
| source-proxy | 1,780 |
| residue-proxy | 1,273 |
| source-resolved | 387 |
| cluster-proxy | 343 |
| existing-curated | 217 |
| manual-resolved | 127 |
| cluster-resolved | 125 |
| manual-rejected | 25 |
| manual-deferred | 22 |

### Confidence

| Confidence | Records |
|---|---:|
| low | 2,172 |
| unresolved | 1,298 |
| high | 619 |
| medium | 210 |

Low or unresolved analyses account for 3,470 records (80.7%). Confidence grades the resolution of a particular parent, not the mere presence of a donor label in an editorial/source note; these proportions must not be presented as a fully solved etymological dictionary.

### Resolution quality within donor proxies

The 1,629 lexeme clusters containing at least one unresolved donor proxy are separately graded below. A catalog-indexed or explicit comparison can be rechecked against a named form; a donor-label-only proxy preserves a source judgment but does not reveal the comparison that motivated it.

| Proxy evidence | Clusters | Share of proxy clusters |
|---|---:|---:|
| explicit-comparanda | 1,444 | 88.6% |
| donor-label-only | 108 | 6.6% |
| catalog-indexed | 77 | 4.7% |

A surface-form check against the best recoverable printed comparandum provides a second, independent triage. It is deliberately descriptive: compounds, historical sound change, and reconstructed citation forms can make genuine contacts look distant, while near identity alone cannot establish direction or inheritance.

| Best surface shape | All proxies | Unqualified | Questioned |
|---|---:|---:|---:|
| exact | 389 | 351 | 38 |
| near | 385 | 344 | 41 |
| moderate | 386 | 324 | 62 |
| distant | 322 | 251 | 71 |
| unscored | 147 | 105 | 42 |

Only 1,299 proxy clusters (79.7%) combine recoverable comparanda with no explicit uncertainty marker. 254 contain an explicit question or hedge; 108 give no recoverable compared form at cluster level. Direction is explicitly asserted for 89, while 56 are comparison-only and the remainder are attributions without a stated direction. These distinctions are tabulated in `nihali-source-proxy-quality-audit.csv`. The 20 critical-priority rows are weakly supported core-vocabulary proxies. Manual review corroborates contact in 8, finds the route ambiguous in 5, retains 3 as plausible, and leaves 4 unresolved. The unresolved cases do not count as positive evidence for genetic classification; the row-level reasoning is preserved in `nihali-core-source-proxy-review.csv` and repeated in the quality audit.

The 44 high-priority weak proxies involving Munda or Dravidian also received full manual review: 10 corroborated contact items, 11 route-ambiguous items, 15 plausible items, 5 weak comparisons, and 3 unresolved labels. This prevents questioned family tags from being mistaken for equal-strength evidence; preferred donors and full rationales are in `nihali-diagnostic-source-proxy-review.csv`.

The remaining 200 explicitly hedged proxies were then reviewed one by one: 54 corroborated, 38 route-ambiguous, 57 plausible, 30 weak, and 21 unresolved. Thus no explicitly questioned proxy remains unreviewed. These decisions grade the source comparison without silently rewriting its graph label; the complete register is `nihali-questioned-source-proxy-review.csv`.

### Lexical sources and ascertainment

The sources differ sharply in whether they print or inherit etymological commentary. Rows therefore measure documentation practices as well as the language.

| Source | Records | Direct donor note | Current external label | Residue |
|---|---:|---:|---:|---:|
| nagaraja2014 | 1,761 | 1,386 (78.7%) | 1,411 (80.1%) | 350 (19.9%) |
| mundlay1996 | 1,707 | 781 (45.8%) | 1,090 (63.9%) | 617 (36.1%) |
| bhattacharya1957 | 407 | 170 (41.8%) | 258 (63.4%) | 148 (36.4%) |
| varghesekumar2015noira | 234 | 0 (0.0%) | 124 (53.0%) | 110 (47.0%) |
| konow1906 | 190 | 23 (12.1%) | 67 (35.3%) | 123 (64.7%) |

Deduplicating within each source gives the following source-normalized profile. Family columns are non-exclusive, and cross-dictionary propagation is retained because the question here is the current hypothesis coverage of each source, not authorship of the label.

| Source | Clusters | External | Korku | Indo-Aryan | Munda | Dravidian | Residue |
|---|---:|---:|---:|---:|---:|---:|---:|
| bhattacharya1957 | 396 | 63.1% | 33.6% | 37.6% | 9.6% | 6.1% | 36.9% |
| konow1906 | 183 | 33.9% | 16.4% | 18.0% | 1.6% | 6.0% | 66.1% |
| mundlay1996 | 1,687 | 63.7% | 25.8% | 44.5% | 5.1% | 2.3% | 36.3% |
| nagaraja2014 | 1,735 | 80.0% | 44.4% | 42.8% | 5.8% | 7.1% | 20.0% |
| varghesekumar2015noira | 220 | 51.8% | 21.4% | 34.5% | 5.0% | 7.7% | 48.2% |

The normalized contrast is too large to read chronologically. Konow has only 33.9% external coverage, while Nagaraja has 80.0%; Nagaraja also supplies the densest Korku commentary. Non-Korku Munda remains a small share in every source (1.6%–9.6%), rather than becoming a dominant layer when dictionary size is controlled. Location, elicitation scope, and editorial policy remain confounded; complete counts are in `nihali-source-normalized-profile.csv`.

Direct attribution agreement is still sparser. Among the 701 clusters attested in at least two sources, only 119 receive the same direct family label from multiple sources and 90 receive compatible nested labels. 48 have disjoint direct labels; 298 are labelled by only one source, and 146 have no direct label in their own replicated rows. Silence is not disagreement, and matching labels are not fully independent because later dictionaries can repeat earlier analyses. The result explains why source hypotheses are retained but not treated as independently replicated cognate judgments; see `nihali-cross-source-attribution-agreement.csv`.

Family-specific replication makes that distinction explicit:

| Family | All labelled | Lexeme in 2+ sources | Family directly labelled in 2+ sources |
|---|---:|---:|---:|
| Korku | 982 | 323 | 58 (18.0%) |
| Munda | 153 | 51 | 5 (9.8%) |
| Indo-Aryan | 1,306 | 348 | 162 (46.6%) |
| Dravidian | 143 | 43 | 5 (11.6%) |

Only 5 of the 51 replicated Munda-labelled lexemes receive a direct Munda label in two or more sources, compared with 58/323 for Korku and 162/348 for Indo-Aryan. Thus repeated attestation of a proposed Munda item usually replicates the Nihali word, not the Munda analysis. Full zero/one/two-plus counts and the dependency warning are in `nihali-family-attribution-replication.csv`.

Cross-source replication by layer is also non-exclusive for the named contact families. ‘Early source’ here means attested in Konow (1906) or Bhattacharya (1957), not that the etymon itself has been dated.

| Layer | Clusters | In 2+ sources | Early-source attested | Nagaraja 2014 | All five |
|---|---:|---:|---:|---:|---:|
| Nihali residue | 1,133 | 140 (12.4%) | 259 | 347 | 2 |
| Korku | 982 | 323 (32.9%) | 156 | 770 | 2 |
| Munda | 153 | 51 (33.3%) | 39 | 101 | 2 |
| Indo-Aryan | 1,306 | 348 (26.6%) | 174 | 742 | 1 |
| Dravidian | 143 | 43 (30.1%) | 29 | 124 | 2 |

The contact layers replicate across sources at roughly 27–33%, versus 12% for the strict residue. That does not make the loans older than the residue: named comparisons are easier to propagate across dictionaries, and the five sources sample different places, times, and editorial traditions. The table is therefore a documentation-stability check, not a loan chronology; its rows are in `nihali-contact-layer-replication-audit.csv`.

The database's linked concept categories show that the strata are not distributed uniformly across the lexicon. Categories are non-exclusive.

| Layer | Concept-linked | Noun | Verb | Adjective | Numeral | Other |
|---|---:|---:|---:|---:|---:|---:|
| Nihali residue | 674 | 267 | 249 | 47 | 12 | 116 |
| Korku | 680 | 391 | 198 | 65 | 4 | 50 |
| Munda | 128 | 60 | 59 | 6 | 2 | 10 |
| Indo-Aryan | 857 | 533 | 180 | 94 | 21 | 46 |
| Dravidian | 105 | 37 | 46 | 8 | 11 | 7 |

Among concept-linked clusters, Indo-Aryan is strongly noun-heavy (533 nouns versus 180 verbs), as is Korku (391 versus 198), whereas the residue is more predicate-rich (267 nouns versus 249 verbs). This is compatible with layered lexical replacement rather than a single uniform donor process. It is not decisive: concept links are incomplete, source editors differ in coverage, and Dravidian and non-Korku Munda labels are themselves verb-rich. Full counts and cautions are in `nihali-layer-category-profile.csv`.

Surface word shape likewise fails to isolate a unique residual phonotactic system:

| Layer | Mean folded length | Final vowel | Multiword/compound | Retroflex | Aspiration | Nasalization |
|---|---:|---:|---:|---:|---:|---:|
| Nihali residue | 5.86 | 74.9% | 26.1% | 22.9% | 16.1% | 2.3% |
| Korku | 6.01 | 72.3% | 24.6% | 24.7% | 24.8% | 3.5% |
| Munda | 5.54 | 68.0% | 23.5% | 20.9% | 21.6% | 3.9% |
| Indo-Aryan | 5.80 | 71.2% | 18.6% | 20.4% | 25.7% | 2.6% |
| Dravidian | 5.59 | 75.5% | 33.6% | 28.0% | 12.6% | 1.4% |

Every layer has median folded length 5; residue and contact strata broadly overlap on final vowels, retroflexion, and compounding. Residual aspiration is lower than in the Korku and Indo-Aryan layers, but this unmodelled difference can reflect loan adaptation, morphological suffixes, and transcription practice. The negative result prevents the residue's surface profile from being used as a surrogate family classifier. Definitions and counts are in `nihali-layer-form-shape-profile.csv`.

Only 701 of 3,277 clusters are independently attested in at least two lexical sources; 140 of those replicated clusters remain residue throughout. Cross-source stability confirms that these are real Nihali lexical items, but it does not by itself distinguish inheritance from an old unrecognized loan. The complete replicated-residue register contains 12 very-strong, 28 strong, and 100 moderate replication grades; 25 are core clusters that remain residual after variant sensitivity. See `nihali-replicated-residue-audit.csv`.

## Interpretation

1. **Indo-Aryan is widest; Korku is the diagnostic contact center.** Indo-Aryan appears in the largest number of expanded-database clusters, but Korku dominates Kuiper's older core sample and supplies the clearest community-specific transmission layer. A Korku-labelled match is best read as the route of transmission, not necessarily the ultimate origin: Korku itself contains Indo-Aryan loans, so counting every such item as ultimately Munda would inflate the Munda layer.
2. **The Indo-Aryan layer is chronologically mixed.** It contains inherited Indo-Aryan etyma reached through Marathi/Hindi reflexes, recent Hindi/Marathi cultural vocabulary, and English loans often mediated by Indo-Aryan. Surface donor and ultimate etymon must therefore remain separate questions.
3. **The Dravidian layer is real but comparatively small and heterogeneous.** Some proposed comparisons are explicitly uncertain, and apparent Dravidian material may have passed through neighbouring Indo-Aryan or Munda varieties.
4. **The residue is historically important.** Repeated basic vocabulary survives there, but this analysis cannot turn a set of unmatched forms into a demonstrated Proto-Nihali lexicon. That requires recurrent sound correspondences across internal dialect evidence or an external relative, neither of which is presently available at the necessary scale.
5. **An argot-only account is unnecessary.** Some disguising or replacement vocabulary may exist, but a wholesale argot hypothesis predicts neither the persistent basic residue nor the layered, source-specific contact profile as economically as relexification of an independent language does.

### Repeated residue examples

The following are examples attested by at least three distinct lexical sources and not assigned an external source here. Stability across dictionaries supports their reality as Nihali lexemes; it does **not** by itself prove inheritance.

| Form | Gloss | Records | Sources |
|---|---|---:|---|
| cigam | ear | 5 | bhattacharya1957, konow1906, mundlay1996, nagaraja2014, varghesekumar2015noira |
| jiki | eye | 5 | bhattacharya1957, konow1906, mundlay1996, nagaraja2014, varghesekumar2015noira |
| kalen | egg | 5 | bhattacharya1957, mundlay1996, nagaraja2014, varghesekumar2015noira |
| mānḍo | rain | 5 | bhattacharya1957, mundlay1996, nagaraja2014, varghesekumar2015noira |
| nāni | who | 5 | bhattacharya1957, konow1906, mundlay1996, nagaraja2014 |
| bay | today | 4 | bhattacharya1957, mundlay1996, nagaraja2014, varghesekumar2015noira |
| cōn | nose | 4 | bhattacharya1957, mundlay1996, nagaraja2014, varghesekumar2015noira |
| jopo | water | 4 | bhattacharya1957, konow1906, mundlay1996, nagaraja2014 |
| kajar | above | 4 | bhattacharya1957, mundlay1996, nagaraja2014, varghesekumar2015noira |
| kāggo | mouth | 4 | bhattacharya1957, konow1906, mundlay1996, nagaraja2014 |
| kōgo | snake | 4 | bhattacharya1957, mundlay1996, nagaraja2014, varghesekumar2015noira |
| menge | tooth | 4 | bhattacharya1957, konow1906, nagaraja2014, varghesekumar2015noira |
| miān | how much | 4 | bhattacharya1957, mundlay1996, nagaraja2014, varghesekumar2015noira |
| nēto | ash | 4 | mundlay1996, nagaraja2014, varghesekumar2015noira |
| pāgo | tail | 4 | bhattacharya1957, mundlay1996, nagaraja2014, varghesekumar2015noira |
| bakko | hand and arm together | 3 | mundlay1996, nagaraja2014, varghesekumar2015noira |
| bardo | sickle | 3 | bhattacharya1957, mundlay1996, nagaraja2014 |
| beʈe | not | 3 | bhattacharya1957, mundlay1996, nagaraja2014 |
| biya | village | 3 | bhattacharya1957, mundlay1996, nagaraja2014 |
| cekṭo | axe | 3 | bhattacharya1957, nagaraja2014, varghesekumar2015noira |

## Method and safeguards

Existing rank-1 Jambu edges were retained. Repeated attestations were clustered only when their normalized forms and meanings agree, or when a near-identical form has compatible gloss evidence across distinct sources. Record-local donor evidence may propagate within such a cluster. Candidate resolution still requires compatible donor family, semantic overlap, phonological similarity, and a margin over alternatives. Unresolved printed comparisons become donor proxies rather than invented links.
The clustering deliberately under-merges synonym-only glosses rather than consulting the generated concept table, so a later database build cannot change its own lexical units.

Source spellings and diacritics are retained in the installed forms; accent folding and Unicode normalization are used only for candidate retrieval and clustering. The one catalog-number emendation made during this audit is `tongre/ṭongre` 'knee(-cup)': its printed DED(S) 2419 citation corresponds to DEDR 2983, whose Naikri `ṭoŋgre` is the exact comparison, rather than DEDR 2419 'neck'. The correction is documented in the source row. No printed lexical form was silently respelled. The contradictory `katʰarnāk` source/gloss case remains quarantined rather than normalized into a convenient etymology.

Records with no printed or cross-dictionary donor evidence are searched for review leads. Every machine lead that cleared the conservative threshold was then manually accepted, rejected, or deferred in `nihali-computational-candidate-review.csv` and `nihali-low-margin-candidate-review.csv`. Only accepted leads become external links; rejected and deferred leads remain Nihali-residue proxies. This prevents accidental short-form and remote-language coincidences from driving the historical conclusion. The complete scores, margins, alternatives, source wording, manual rationales, cluster IDs, and record IDs are in `nihali-provisional-etymology-audit.csv`; `nihali-provisional-lexeme-audit.csv` provides the deduplicated review view.

## Comparison with Kuiper's baseline

Kuiper's 1962 study classified a 503-item vocabulary as 180 direct Korku loans (36%), about 20 possible remnants of an earlier Munda layer (roughly 4%), 47 Dravidian items (9%), and 123 items without any known Indian correspondence (about 24%). He explicitly warned that incomplete Korku documentation and subjective borderline decisions make these figures approximate. The present database is over eight times larger, combines five partly overlapping sources, and distinguishes immediate donor from ultimate ancestry; its percentages therefore test the shape of his model but are not a direct replication.

## Evidence synthesis

No single percentage decides the origin question. The matrix below records what each independent diagnostic can and cannot support.

| Domain | Finding | Evidential weight |
|---|---|---|
| stable lexical residue | 1133 lexeme clusters remain residual; the reviewed core reduces to 56 root hypotheses, including 29 attested in at least two sources. | moderate |
| concept-level basic vocabulary | Of 91 covered concepts, 17 are residue-only, 22 mix residue and contact forms, and 52 are contact-only. | moderate |
| Korku transmission route | Among 854 Korku-labelled proxies, 86 independently recover close Korku forms; only 8 currently trace to an upstream Munda parent. | moderate |
| non-Korku Munda specificity | Munda has a high-specificity floor of 8 among 153 labelled clusters; only 5 of 51 replicated lexemes receive direct Munda labels in two or more sources. | moderate negative |
| Nihali–Munda correspondences | 33 resolved links represent 28 parent roots: 11 near contact-compatible, 15 possible, and 7 weak; no recurring sound law is demonstrated. | moderate negative |
| Indo-Aryan layer | The evidence bracket runs from 452 high-specificity to 1222 supported clusters; the layer is noun-heavy and includes Korku-mediated routes. | strong for contact; weak for origin |
| Dravidian layer | 98 resolved links collapse to 66 roots; 22 are near contact-compatible and 32 are weak on form or meaning, without a sound-law system. | moderate for contact; negative for origin |
| closed-class asymmetry | Pronouns have 17 residual links out of 22, while low numerals have only 1 out of 37. | moderate |
| source dependence | Among 701 replicated clusters, only 119 have exact direct family-label agreement; 298 are labelled by one source. | strong methodological caution |
| surface word shape | All five layers have median folded form length 5 and overlapping surface diagnostics. | negative control |
| morphology and pronouns | Zide reports that Nihali pronouns do not resemble Munda pronouns and that proposed case and verbal analyses do not yield a Proto-Munda-like inherited system. | moderate |
| broad structural typology | Grambank similarity does not place Nihali uniquely with Munda rather than regional controls. | negative control |
| argot hypothesis | Stable basic terms recur across early and late sources without a demonstrated productive disguise rule. | moderate negative |
| residue threshold sensitivity | A production-like score/margin-only relaxation flags 65 residual clusters, but 39 never pass the required component gates; the very-high setting leaves 16. | strong methodological caution |
| whole-lexicon variant sensitivity | A mechanical exact-gloss and >=0.72 whole-form screen, supplemented by explicitly reviewed derivational and inflectional families, covered 205 residual/contact cluster pairs; 172 are plausible variants and 32 remain qualified leads. | strong methodological caution |
| residual expression decomposition | A targeted audit of 23 residual expressions finds 18 transparent and 4 qualified contact components, plus one quarantined source/gloss conflict. | strong methodological caution |
| synthesis | No known donor layer explains the whole core, and no alternative family has a regular inherited correspondence system. | moderate overall, diagnosis by exclusion |

The full matrix also states the hypothesis supported, the alternative challenged, the limitation, and the underlying audit or publication for every row: `nihali-origin-evidence-matrix.csv`.

## Competing origin hypotheses

| Hypothesis | Fit to this lexical audit | Provisional confidence | Main unresolved test |
|---|---|---|---|
| Independent lineage with layered relexification | Best overall fit: stable residue plus separable Indo-Aryan, Korku/Munda, and Dravidian contact layers | Moderate, as a diagnosis by exclusion | Demonstrate internal history or identify an external relative with regular correspondences |
| Direct Munda affiliation | Explains some lexicon and areal/morphological traits, but the resolved lexical set lacks a recurring inherited correspondence system | Low on present lexical evidence; not excluded | Reconstruct shared innovations not attributable to Korku contact |
| Indo-Aryan or Dravidian affiliation | Poor fit: both behave as stratified donor layers, not as the source of the whole basic lexicon | Very low | Find inherited morphology and regular core cognates outside the known loan strata |
| Argot-only origin | Can explain some substitution or deformation but not the whole cross-source lexical system | Low as a complete account; plausible as a secondary process | Identify productive disguise rules and their recoverable base forms |
| Relationship to another isolate or macrofamily | Published comparisons are sparse and non-systematic | Unsupported | Establish multiple exclusive, semantically controlled correspondence series |

- **Independent-lineage/relexification hypothesis.** Kuiper's unidentified component and Zide's later appraisal treat the non-loan residue as potentially representing a lineage without a demonstrated living relative. This best predicts a stable core residue together with several donor-specific layers, but it remains a diagnosis by exclusion rather than a comparative reconstruction.
- **Munda-branch hypothesis.** Mundlay argued in the same 1996 volume for placing Nihali directly under Proto-Munda but outside the Northern and Southern branches, using lexical, grammatical, and ethnographic evidence. The present lexical audit finds substantial Korku/Munda material, but much of it is explicitly contact-attributed and it does not produce the regular inherited correspondence set needed to choose this genetic account. Ilia Peiros's independent core-list appraisal in the same volume found only nine Munda comparisons with a scattered distribution and likewise judged the relationship unconvincing. Morphology remains the strongest counterargument to an isolate analysis, but Mundlay's own presentation says the grammatical evidence is largely negative, the resemblances are not close, and structural erosion obscures the system. The lexical database cannot adjudicate inherited versus contact-induced grammar, so a future morpheme-by-morpheme reconstruction is essential.
- **Argot or deliberately disguised register.** Socially restricted vocabulary and semantic replacement may explain some forms, but do not by themselves explain the cross-source stability of residue terms for body parts, natural phenomena, pronouns, and basic predicates. The audit therefore treats argot formation as a secondary process, not a complete origin account.

### Structural sensitivity check (not a family test)

As an external control on the grammatical argument, 164 coded Nihali features in [Grambank](https://github.com/grambank/grambank/commit/37f73da55cf8b426c82383f46a972bc59ce6cf76) were compared with selected regional languages. Pairwise raw agreement is 74.4% with Korku (129 shared coded features), 73.2–77.1% with five other Munda languages, 73.8% with Marathi, 78.5% with Hindi, 77.4–81.0% with four selected Dravidian languages, and 79.6% with Kusunda. Chance-corrected kappa and positive-feature Jaccard measures preserve the same basic warning: no uniquely Munda cluster emerges from this small control.
Each pair uses only features coded for both languages; kappa corrects for marginal-value agreement and Jaccard compares features whose value is 1. No imputation, feature weighting, or optimization was used.

These figures cannot identify ancestry. Grambank features are synchronic, structurally dependent, unevenly missing, and highly susceptible to areal convergence; the comparison also lacks a phylogenetic or spatial model. Its value is negative: a broad claim that Nihali 'looks Munda' is not by itself discriminating evidence. Only shared innovations and morpheme histories can decide the morphological question. Inputs, shared-feature counts, and all three descriptive measures are preserved in `nihali-grambank-structural-sensitivity.csv`.

The provisional adjudication is consequently **independent lineage with profound Munda contact**, not because Munda affiliation is impossible, but because the positive evidence required to demonstrate it is still missing.

## Scholarly context

The interpretation agrees in broad outline with Kuiper's stratified treatment and with Zide's later caution: massive Korku-mediated relexification can coexist with an independent residue, while isolated similarities to Tibeto-Burman or wider Austroasiatic do not by themselves establish genetic affiliation. Zide further stresses that directionality is not always obvious for the small set of South Munda parallels. Those cautions are built into the proxy and confidence scheme here.
The current [Glottolog 5.3 classification](https://glottolog.org/resource/languoid/id/niha1238) also leaves Nihali as a standalone top-level entry rather than placing it inside Munda. That is corroborating scholarly practice, not independent proof of the hypothesis.
Shailendra Mohan's 2016 documentation report likewise describes several historical and local contact layers and says that proposed links to Kusunda, Ainu, Nostratic-Dravidian, and Greater Austric had not yielded conclusive genetic evidence. This audit does not retest those remote proposals form by form; it treats them as unsupported until a regular correspondence system is demonstrated.
John Peterson's 2021 areal-typological account treats the Satpura Range as a residual or accretion zone where Nihali, Korku, and Gondi survived successive regional expansions. He explicitly allows that pre-Munda languages were already present in such hill zones. That geography makes survival of an independent Nihali lineage historically plausible, but it is contextual fit rather than comparative proof and cannot date the language.
A 2025 methodological survey of language isolates uses Nihali as an example of an isolate whose grammar has been extensively remodeled through sustained multilingualism. Its recommended historical toolkit—dialect comparison, internal reconstruction, philological source study, and explicit contact analysis—also explains the boundary of the present result: this lexical audit advances the last two, but cannot substitute for the first two.
A Mohan-led [Endangered Language Documentation Programme project](https://cultureincrisis.org/projects/documentation-and-description-of-nihali-a-critically-endangered-language-isolate-of-india) targets a descriptive grammar, trilingual dictionary, and 20 hours of archived audio/video. Those materials are the right basis for testing shared morphological innovations, but they are not silently treated as part of this five-source lexical database. The distinction matters: this report's negative finding is absence of a demonstrated relationship in the audited evidence, not proof that no relationship can ever be found.
A 2017 genome-wide study reports excess haplotype sharing and recent population ancestry between sampled Bhil and Nihali groups. That makes Zide and Shafer's lost-Bhil-language scenario geographically and demographically interesting, but genes do not classify languages. The result is compatible with regional population continuity or interaction; it neither proves that old Nihali was the Bhils' language nor supports a specific linguistic family. In particular, this report makes no 'Paleolithic' or ancestry-component claim from the lexical residue.

Primary/contextual references: F. B. J. Kuiper, [*Nahali: A Comparative Study* (1962)](https://dwc.knaw.nl/DL/publications/PU00009788.pdf), especially pp. 48-51; Kuiper, [*The Sources of the Nahali Vocabulary* (1966)](https://sealang.net/sala/archives/pdf8/kuiper1966sources.pdf); Norman H. Zide, ["On Nihali" (1996)](https://www.mother-tongue-journal.org/wp-content/uploads/2025/08/2-Mother-Tongue-II-1996_text.pdf), pp. 93–100; K. S. Nagaraja, [*The Nihali Language* (2014)](https://hdl.handle.net/20.500.14705/8151); and Asha Mundlay, "Nihali Lexicon" (1996), in the same *Mother Tongue* volume, pp. 17–40; and Ilia Peiros, "Nihali and Austroasiatic" (1996), in the same volume, pp. 75–76; Shailendra Mohan, ["Describing Endangered Languages: Experiences from Nihali Documentation Project" (2016)](https://files.core.ac.uk/download/pdf/141880631.pdf), pp. 182–187; John Peterson, ["The Spread of Munda in Prehistoric South Asia: The View from Areal Typology" (2021)](https://www.isfas.uni-kiel.de/de/linguistik-und-phonetik/team/uploads/Peterson_Spread_of_Munda.pdf), pp. 109–130; Iker Salaberri et al., ["State of the Art of Research on Language Isolates" (2025)](https://doi.org/10.1075/tsl.135.intro), pp. 2–19; and Gyaneshwer Chaubey et al., ["The Genome-Wide Analysis of the Bhils" (2017)](https://www.isw.unibe.ch/e41142/e41180/e523709/e523717/2017b_ger.pdf), especially pp. 279–285.

## Limits and next tests

This is a hypothesis inventory, not a completed comparative reconstruction. The strongest next lexical step is independent verification of the unqualified source proxies against primary donor lexica, prioritizing the 405 Korku form-only matches and 297 weak/unmatched routes. Korku itself needs etymological expansion for the 78 strongest route forms that currently stop without an upstream parent. The 56 core residue-root hypotheses then need dialectal comparison and correspondence discovery rather than another round of string matching. On the grammatical side, person marking, case allomorphy, and verb morphology require morpheme-by-morpheme reconstruction from the new documentation corpus. Neither the residue percentage nor the typological profile can currently distinguish an ancient isolate from an unrecognized deep relationship whose comparanda have been lost.
