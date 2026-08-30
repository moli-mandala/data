# ESR 2011-023 Koch Bangladesh manual recovery

State: **manual review and shared source-specific integration complete; consolidated build deferred**.

This package recovers Appendix A.3 of *The Koch of Bangladesh: A Sociolinguistic
Survey* (SIL Electronic Survey Report 2011-023, March 2011). Every lexical
reading was transcribed and verified by hand from rendered primary-source
pages. OCR, PDF text, embedded legacy glyphs, the legacy audit, and installed
forms were used only to locate material or in reconciliation after the manual
ledger was frozen; none supplied or verified a reading.

## Frozen manual ledger

- Primary PDF: 91 pages; SHA-256
  `d1b2d597c16fd0338ad47d2bf031566192c5ff4e26a6651de14a228df681fc10`.
- Wordlist: physical pages 43–62 / printed pages 42–61, items 1–307.
- 1,113 printed response/disposition lines.
- 2,149 conceptual cells and 2,159 expanded rows.
- Conceptual dispositions: 1,780 attested, 225 ambiguity-only, 25 blank, 119
  not used. Expanded rows: 1,789 attested, 226 ambiguous, 25 blank, 119 not
  used.
- Item 241/site `r` is the one mixed coordinate: resolved `akui̯ʃa` is retained,
  while the separate `tɛp` response with an unresolved modifier is excluded.
- No unresolved modifier was inferred. The complete coordinate list is frozen
  in `post_freeze_manifest.json` and the visible bases/notes remain in
  `staging_audit.tsv`.

## Staging and exclusions

`staged_forms.csv` contains 1,017 resolved Koch target attestations at codes
`b`, `c`, `q`, and `r`, with 1,017 unique immutable source `Entry_Key` values.
The 1,142 audit-only rows comprise 772 resolved A’tong/Bangla controls, 226
ambiguous rows, 25 blanks, and 119 globally not-used rows. Source lexical-
similarity groups are carried as source annotations and do not create cognate
or borrowing edges.

The report identifies `b` Nokshi, `c` Kholchanda, and `r` Chandabhoi as
Tintekiya Koch; `q` Uttor Nokshi as Chapra Koch; `l` Bharatpur and `m`
Nalchapra as A’tong controls; and `0` as standard Bangla. Source spelling
variation is preserved (`Bhoratpur/Bharatpur` and
`Namchapra/Nolchapra/Nalchapra`). The report does not print exact site
coordinates, so `site_metadata.tsv` leaves latitude and longitude blank.

## Legacy reconciliation and identity

The generator exhaustively compares all 2,159 frozen manual rows with 2,208
expanded legacy audit rows. The latter include 21 demonstrably spurious global
not-used expansions at items 7, 10, and 12. All 2,159 manual rows are covered;
28 receive two legacy source-response matches because identical repeated
responses were collapsed into one frozen row with all group labels preserved.

Legacy target key identity is retained for the same source occurrence. Of 875
old target keys, 728 remain canonical, 289 keys are newly assigned to manually
recovered occurrences, and 147 old keys retire (20 duplicate aliases and 127
keys belonging to ambiguity or another non-staged disposition). At item
241/site `r`, old key `silkochbd2011:i241:r:1` correctly remains attached to
the excluded ambiguous response; recovered `akui̯ʃa` receives
`silkochbd2011:i241:r:2`.

## Sound and bibliographic policy

All 1,789 attested expanded rows use 44 Unicode codepoints. The pinned
SIL-Bangladesh base profile covers every staged form with no replacement
character and requires no speculative addition. Manual source IPA remains
byte-for-byte in raw Form/Original and Phonemic; `sound_profile.txt` is only a
display conversion profile. `reference_metadata.json` preserves the title,
author order (Seung Kim, Sayed Ahmad, Amy Kim, Mridul Sangma), report number,
date, archive locator, page extent, extraction scope, and exact PDF hash.

Run `build_post_freeze_package.py` to regenerate all post-freeze artifacts.
Focused validation is in `tests/test_sil_kochbd_2011_post_freeze.py`. Shared
source-specific integration installs the 1,017-row target stage byte-for-byte,
registers the seven report-defined sites with blank coordinates, replaces the
reference provenance, and explicitly routes the pinned base profile; the exact
state is frozen in `shared_integration_manifest.json`. Global source-audit
regeneration, the consolidated build, opaque-ID reconciliation, graph/full
tests, browser database work, and commit remain deferred; see `INTEGRATION.md`.
