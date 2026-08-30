# SIL Bangladesh legacy-font manual recovery queue

Checked 2026-08-29.  This queue records attested survey responses that the existing legacy-font
decoder excluded because at least one printed glyph has no verified mapping.  These are not
accepted losses.  They must be recovered by direct visual transcription from rendered publisher
pages under `SOURCE_INGESTION_CHECKLIST.md`'s survey-wordlist and OCR/manual-review addenda.

## Mandatory transcription protocol

- Read every response from the rendered PDF by hand.  Rendered page crops are the only source of
  a reading and the only evidence that may verify it.
- The PDF text layer, raw PUA/CID strings, current font maps, OCR, existing installed forms, and
  earlier transcriptions may locate a cell but must not supply or confirm its reading.
- Review every retained form, not only the records currently excluded by the decoder.  The old
  audit is consulted only after independent entry for reconciliation.
- Record every source gap, disqualified/not-used item, ambiguity, illegible form, and exact
  page/column/item/site coordinate.  Do not guess an IPA symbol.
- Keep source-local ledgers, response-line evidence, expanded cell audits, checksums, profiles,
  registries, and focused tests.  Shared integration waits for the consolidated rebuild.

## Queued reports

| Report | Exact publisher snapshot | Printed wordlist pages | Existing audit state | Manual state |
|---|---|---|---|---|
| ESR 2011-023 *The Koch of Bangladesh* | exact PDF recovered from the 2017-08-09 historical publisher-URL capture; SHA-256 `d1b2d597c16fd0338ad47d2bf031566192c5ff4e26a6651de14a228df681fc10`; 91 physical pages; all 20 wordlist pages rendered at 300 dpi and the final pages rechecked at 600 dpi | printed pp. 42-61 / physical pp. 43-62 | frozen manual audit: 2,149 conceptual cells / 2,159 expanded rows = 1,780 attested, 25 printed blanks, 225 unresolved legacy-modifier cells, and 119 globally unused cells | **manual review and shared source integration complete; consolidated build pending**: all items 1-307 reviewed with zero pending cells; 1,017 unique target forms installed, while 772 controls, 226 ambiguous expanded rows, 25 blanks, and 119 not-used rows remain audit-only; seven sites, exact reference, and explicit 44-codepoint `sil-bangladesh` profile route registered; OCR/PDF text, legacy data, and installed forms supplied no reading |
| ESR 2011-040 *The Kurux of Bangladesh* | exact PDF recovered from the 2017-08-09 historical publisher-URL capture; SHA-256 `f2f06c25ac55462d6a40843539d8417e24a647bd1eb0bbe3f24ea3e45f0b9e4b`; 90 physical pages; all 19 wordlist pages rendered at 300 dpi | printed pp. 38-56 / physical pp. 39-57 | exhaustive frozen audit: 1,869 rows from 1,842 conceptual cells = 1,661 attested, 136 blanks, 72 not-used, and 27 retained variants | **manual review and shared source integration complete; consolidated build pending**: 1,365 target attestations installed with immutable source keys; 296 Bangla control forms audit-only; exact profile, reference, and six site/control metadata rows registered; no unresolved, ambiguous, or illegible lexical coordinate |
| ESR 2012-007 *The Garos of Bangladesh* | exact PDF recovered from the 2017-08-10 historical publisher-URL capture; SHA-256 `4248b409d816c153f95c09e50bf51f9e5ff90d456e3c8d9d13dc2eca6f8c4359`; 212 physical pages; all 42 wordlist pages rendered at 300 dpi | printed pp. 45-86 / physical pp. 52-93 | 5,264 audit records: 4,444 installed, 712 attested responses excluded for unverified glyphs, 91 per-site printed gaps, 17 globally unused items | **manual review active through item 155**: 2,635 conceptual cells = 2,556 ordinary attestations, one source-conflict cell with an attestation, 61 blank-only cells, and 17 not-used cells for whole-item 152; 2,728 attested response occurrences; item 12/site `p` remains the sole unresolved source conflict; items 156-307 pending |

The existing locator/reconciliation files are
`20260826-sil-kochbd-audit.csv`, `20260826-sil-kurux-audit.csv`, and
`20260826-sil-garobd-audit.csv`.  Together they expose 1,514 attested records previously omitted
solely by the incomplete font decoder.  Counts are response/audit records rather than a claim that
every printed item/site has exactly one response; alternatives and globally unused prompts require
separate conceptual-cell accounting in each source-local manual package.

## Current acquisition state

Rechecked 2026-08-29. Direct SIL publisher navigation still stops at its Cloudflare interstitial,
but broader searches of the historical `www-01.sil.org/silesr/<year>/` paths recovered exact
publisher-era Wayback captures for Kurux and Garo. The replayed bytes match the already pinned
canonical hashes exactly. Kurux is a 90-page PDF; physical pages 39-57 have been rendered as 19
300-dpi PNGs in `tmp/pdfs/kurux_manual/`. Garo is a 212-page PDF; physical pages 52-93 have been
rendered as 42 300-dpi PNGs in `tmp/pdfs/garo_manual/`. Direct visual checks of the first and last
render in each set confirm that both spans contain items 1-307. Their acquisition blocker is closed;
Kurux and Koch have completed exhaustive manual review, and Garo is complete through item 155.

The same historical-series search recovered Koch from a 2017-08-09 capture whose digest matches
the previously known but unreplayable current-URL capture. The exact 91-page PDF matches the pinned
canonical SHA-256; physical pages 43-62 are now rendered as 20 300-dpi PNGs in
`tmp/pdfs/kochbd_manual/`. Physical p. 43 / items 1-13 had already been independently transcribed
from a surviving higher-resolution render into the 91-cell source-local ledger in
`sil_kochbd_2011_manual/`: 85 cells are attested,
three are printed blanks, and three retain their visible base form only in the audit because a small
final modifier is ambiguous. Items 14-307 can now resume at physical p. 44. Newly available page
images are not permission to fall back to OCR, raw legacy glyphs, or the existing installed forms.

The Kurux publisher record is independently pinned as SIL Bangladesh archive entry `41654`: it
names the exact 89-page numbered report and explicitly states that wordlist comparisons were used.
Its live file link still returned HTTP 403 on 2026-08-29, but the verified historical replay now
provides the required page-image evidence. The difference between the official 89-page extent and
`pdfinfo`'s 90 physical pages is the unnumbered cover/front matter.

The acquisition record confirms all three page-image sets are now ready and remain pinned even though Kurux has already completed
its source-local manual review.

## Dispatch plan

Koch and Kurux have completed manual review and shared integration. Garo resumes at item 156 in its separate
page-bounded manual-transcription lane. Do not rebuild shared CLDF until the active survey packages
finish their source-local and source-specific shared-integration gates.
