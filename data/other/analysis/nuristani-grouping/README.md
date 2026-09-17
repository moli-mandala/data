# Nuristani/CDIAL editorial grouping — 2026-09-13

The editorial policy is to organize Nuristani evidence under its corresponding CDIAL entry when
one can be identified. Proto-Nuristani reconstructions and their former attestations are siblings;
the grouping does not decide between inheritance from Proto-Indo-Iranian and borrowing from
Indo-Aryan. Lexical variants retain their true target inside the group.

## Implementation

`nuristani_grouping.py` replaces the former `apply_nuristani_cognates`,
`reparent_cdial_nuristani_reflexes`, and `apply_nuristani_borrowings` passes. It runs after durable
identity assignment and the etymology overlay, so source order and historical assignment rows
cannot rebuild the superseded hierarchy. The existing cognate and borrowing catalogs preserve
the source interpretations and provide the correspondence, without choosing different topologies.

The edge-model compatibility representation is `Kind=reflex, Rank=1`, explicitly qualified by
`grouping:cdial; inheritance versus borrowing unresolved` in `Note`. The `etymology-group` form
tag makes the site label the link **Grouped with** and explain the unresolved transmission route.
These are editorial group memberships, not accepted inheritance assertions. This exception is
also documented in the data README and AGENTS.md.

The old 359 blank PII placeholders become CDIAL redirects. Their public IDs still resolve. Real
PII and PNur reconstructions retain all their lexical content and source attribution; 125 matched
standalone Strand PII heads also become grouped siblings. Source article descendants in other
languages move with a mapped head (three records in this build).

## Correspondence inventory

- `data/nuristani_cognates.csv`: 460 previously reviewed PNur correspondences.
- `data/nuristani_borrowings.csv`: 558 previously reviewed PNur correspondences, including NurED.
- `data/nuristani_cdial_groups.csv`: 125 additional Strand PII heads.
- `head-matching-audit.csv`: the full 205-head screening inventory, including 80 unresolved heads.

Additional head matches use a unique vowel-accent-normalized CDIAL spelling, an exact spelling
disambiguated by an attested reflex, or a unique existing CDIAL target of a same-language reflex
with matching spelling (acute/grave vowel accents ignored; consonantal ś/ź retained) and overlapping gloss vocabulary. They are
editorial correspondence candidates used for grouping, not a new inference of inheritance or
borrowing. No fuzzy phonetic search is performed by the build. Unresolved heads are retained in
their existing groups; no mapping is invented for them.

## Installed result

`graph-migration-audit.csv` records each changed edge or retired placeholder. `validation.json`
records the counts and lexical-preservation audit.

- 11,371 nodes marked as CDIAL group members, including 1,018 PNur and 125 PII reconstructions.
- 359 obsolete blank PII nodes redirected; 359 IA-to-placeholder edges removed.
- 855,253 node IDs before and after; no spelling, original transcription, gloss, phonemic form,
  citation, or etymological prose changed. Only Tags, Status, and Redirect changed on forms.
- 403,267 resulting edges, down from 403,501.
- 5,618 changed alignment targets regenerated using the existing alignment implementation.
- Accepted-graph acyclicity, all edge endpoints/statuses, and policy idempotence checked.

This migrates the graph, preserving separate source attestations. It does not merge the
accent/transcription doublets identified in the initial duplication audit.

## Verification and remaining full gates

The focused suite passes: 16 tests. The tests cover both former inheritance and borrowing branches, multiple/nested mappings,
blank-node redirects, genuine PII content preservation, variant targets, CDIAL subsection aliases,
other-source and other-language article descendants, unresolved heads, alternate-edge collapse,
canonical addendum targets, compiled grouping invariants, and changed alignment origins.

A 433-node compact database built from four representative CDIAL entries and an unmapped PNur
entry passes the real browser-query code using a native SQLite adapter. Both Katë diʦ/díʦ records
and three PNur reconstructions occur together under CDIAL 6658; PNur siblings have no reflex
children. Kamviri kāsa/kāsá both point to CDIAL 3135. All four example labels are “Grouped with.”
The full Svelte check passes with 0 errors and 7 existing warnings.

The corpus-wide source rebuild, full test suite, full browser database rebuild, and live-browser
QA are deferred under the laptop resource policy. The configured data CI only runs on push/PR;
no publication was requested solely to obtain a remote runner. The local migration used bounded
form loading and streamed the large tables; app-query verification used the small compact fixture.
The default `.dbwork/jambu.db` and deployed site still need a normal database build/release to
reflect this change. No production release, push, or commit was performed.
