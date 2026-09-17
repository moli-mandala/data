# Yoshioka cross-reference review — 14 September 2026

## Result

Reviewed all **57 previously unresolved index entries**, containing **62 forms**,
against the pinned PDF's index lines and referenced root/stem contexts.
**56 forms now have supported links; six remain ambiguous.** The requested
cross-reference review is complete; full repository integration remains pending.

| Measure | Result |
| --- | ---: |
| Previously unresolved entries fully resolved | 51 |
| Previously unresolved entries partially resolved | 2 |
| Previously unresolved entries still wholly ambiguous | 4 |
| Reviewed form-level links accepted | 56 |
| Reviewed forms left without an accepted edge | 6 |
| Total fully resolved index entries, including the earlier 163 | 214 |
| Installed source rows | 4,886 (unchanged) |

The partial entries are `yoshioka-entry-682` and `yoshioka-entry-726`.
Some index lines list forms with different meanings, requiring form-level decisions.
This adds 52 edges, redirects four old index-list edges to their lexical entries,
and removes one ambiguous index-list edge. Total variant/inflection edges rise
from 1,816 to 1,867.

## Editorial decisions

- Match the complete printed form in the explicitly referenced root group.
  The root can be a homograph or have a different meaning from its subentry.
- `atúγunum` links to **raw, unripe**, with X class and its plural suffix retained.
- Entry 749's three forms resolve separately to **chop, cut down, part**,
  **make bloom**, and **make chop**.
- Identical `@-̈doon-` forms are distinguished by their printed references:
  **gón → make open**, **gún → make catch, make pack**.
- `@-̇sqan-` explicitly references **γan**, identifying **kill, make die,
  perform**. The unrelated **beautify, adorn, decorate** entry is outside that group.
- `tur` identifies **horn** under **ltur**; the other root article means **imitate**.
- Index `duqhúlan` omits the terminal stem hyphen found on `duqhúlan-` under
  **qhulán**. This one reviewed equivalence is accepted while retaining both
  source spellings. No general fuzzy spelling rule is introduced.
- Original notes, citations, grammar, spellings and keys are preserved. Resolved
  forms gain the target citation and complete grammatical string as
  `Referenced entry grammar:`, plus applicable grammatical/class/dialect tags.
  `digía-` gains its explicit plural tag; a plural suffix alone does not make
  an article's headword plural.

## Remaining ambiguity

Each form below matches two printed senses under the named root, and the index
gives no sense identifier. Both candidates are recorded; the form retains a blank
lexical gloss, `uncertain`, and no accepted edge. Entry 726's second form loses
its old list edge, which would otherwise inherit the first form's “be born” sense.

| Index key / form | Index PDF page | Printed target | Candidate meanings (key; PDF page) |
| --- | ---: | --- | --- |
| `yoshioka-entry-682` / **d-@-́γan-** | 528 | γan | be ended, be used up, be exhausted (`yoshioka-entry-1140`; 545)<br>chip, be worn out (`yoshioka-entry-1145`; 545) |
| `yoshioka-entry-726:variant:1` / **d-@-̈man-** | 530 | man | become aware, realise (`yoshioka-entry-1907:variant:1`; 572)<br>become numb (`yoshioka-entry-1908:variant:1`; 572) |
| `yoshioka-entry-791` / **du-γán-** | 532 | γan | be ended, be used up, be exhausted (`yoshioka-entry-1139`; 545)<br>chip, be worn out (`yoshioka-entry-1144`; 545) |
| `yoshioka-entry-829` / **duqhár-** | 533 | qhar | crack (`yoshioka-entry-2332`; 587)<br>bloom, blossom (`yoshioka-entry-2335:variant:1`; 587) |
| `yoshioka-entry-851` / **duún-** | 534 | gún | freeze (`yoshioka-entry-1082:variant:1`; 542)<br>catch, seize, pack, begin (+ INF DAT/ADE, or FINALIS of V) (`yoshioka-entry-1083:variant:1`; 542) |
| `yoshioka-entry-2770` / **tá-** | 602 | ltá | run after (+ADE/@-cí), follow, reach (`yoshioka-entry-1804:variant:1`; 568)<br>put on (`yoshioka-entry-1806:variant:1`; 568) |

## Evidence and validation

- All 57 index entries and their root/stem contexts were visually reviewed in
  **28 image groups**. Image hashes, source text, physical pages, candidates and
  decisions are in [crossreference-decisions.json](../data/other/forms/raw_data/yoshioka_2026/crossreference-decisions.json).
  The [row audit](../data/other/forms/raw_data/yoshioka_2026/crossreference-audit.csv)
  records all 62 final outcomes and hashes. The [package README](../data/other/forms/raw_data/yoshioka_2026/README.md)
  explains how to reproduce the images.
- The importer rejects changed inventories, source/target rows and root evidence
  before applying decisions. Source-local endpoints and cycles are checked.
- **60 focused tests pass**, with the previously reported unrelated corpus-count
  test explicitly deselected. They cover earlier resolutions, unchanged other
  rows, mixed-meaning lists, homographs, the hyphen equivalence, grammar retention,
  and stale-evidence rejection.
- Installed source and audit exactly match regeneration. The bounded source parse
  retains **4,886 rows, zero errors**, all spellings and keys, and durable IDs/aliases.
  This work does not write the global identity registry.
- All **3,949 original grammar notes** survive; **56 referenced grammar notes**
  are added. Plural-tagged rows total **481**, and class-tagged rows **2,367**.
- The nine bare-heading exclusions, 25 rejoined fragments, ten source-questioned
  donor statements, sound profile, bibliography and language registry are unchanged
  by this follow-up. No new ancestry claims are made.

## Deferred checklist gates

Dictionary/glossary and reference-linking addenda apply. This follow-up uses the
existing native-font snapshot; OCR, acquisition, new-language registration and
new bibliography entries are inapplicable. The full data build, full suite,
consolidated source-audit refresh and browser database/UI QA remain pending under
the workspace's 8 GB resource policy and requested partial scope. Publication was
not requested.

Representative corrected source entries for a later refreshed-app check:
`yoshioka-entry-105` (atúγunum), `yoshioka-entry-737:variant:1` / `738`
(@-̈doon-), `yoshioka-entry-749` and its two variants, and
`yoshioka-entry-2891` (tur). These are verified source rows; the app database
has not been refreshed or visually checked for this follow-up.
