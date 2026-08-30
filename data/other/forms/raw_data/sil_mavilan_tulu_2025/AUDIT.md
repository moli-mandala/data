# Complete source audit

Reviewer declaration: `hand-keyed-from-rendered-source; OCR-not-copied`.

Every Appendix A.2 response cell was visually inspected from a 400-dpi page
render, with 900/1200-dpi crops for dense glyphs. Every accepted transcription
was keyed from the rendered source. OCR, PDF-extracted text, and legacy rows
were used only to locate material or compare after independent entry; none
supplied, normalized, inferred, or verified an accepted form.

## Exact accounting

| Category | Target | Control | Total |
|---|---:|---:|---:|
| Conceptual / manually reviewed cells | 624 | 624 | 1,248 |
| Attested | 615 | 615 | 1,230 |
| Source blank | 9 | 9 | 18 |
| Ambiguous | 0 | 0 | 0 |
| Illegible | 0 | 0 | 0 |
| Staged forms | 615 | 0 | 615 |
| Attested controls excluded from forms | 0 | 615 | 615 |

The exhaustive audit output has 1,248 rows. All 18 blanks have cell-level
coordinates and the literal printed absence marker in the manual ledger.
`unresolved_readings.tsv` is header-only because no ambiguity or illegibility
remains.

Direct source inspection corrects the initial census: the appendix contains
208 prompts, not 210. Physical p.37 / printed p.31 ends at item 207; physical
p.38 / printed p.32 contains only item 208, Dust. Items 209–210 do not exist.

## Post-entry reconciliation

All 615 manually attested target cells were compared with the legacy install
only after hand entry: 556 exact strings, 59 literal-source differences, and 0
missing coordinates. Each difference is enumerated in its block audit and was
rechecked against the rendered source. The existing `markodi` profile covers
all staged source forms; no profile addition is needed.

The seven block audits and immutable TSV ledgers account for every source cell:
items 1–18, 19–39, 40–79, 80–119, 120–159, 160–189, and 190–208.
