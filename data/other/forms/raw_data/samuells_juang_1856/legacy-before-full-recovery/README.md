# Samuells's Juang vocabulary (1856)

This source-stage package covers every printed cell of the short Juang glossary on pp. 302–303 of Samuells's article. The original 790-page public-domain scan and rendered review pages are local cache, not installed assets. `reviewed_inventory.tsv` preserves 31 prompts, including all 11 holds; `import_source.py` reproducibly generates the audit and the 21-row source CSV. The water alternatives have distinct variant-qualified entry keys.

The article was printed in 1856 despite a 1857 date in the BHL catalog. Samuells says he collected the list himself, but does not tie individual entries to a speaker or one of his 1854–56 visits. The package therefore has a source-qualified Juang lect with blank coordinates. Dalton's 1866 Juang table is analysis only: the two lists differ, and Dalton does not name an original collector for his Juang column.

`visual-review-20260925.jsonl` records a complete 31-cell review against the original pages. In particular, the rice form is **Runkoo** in print, despite OCR `Kunkoo`. `overlap-review.json` records the one exact normalized comparison with an existing Juang source. The original accents and doubling remain in `Original`; any generated house transcription is a cautious approximation, not a reconstruction of the 1856 speakers' pronunciation.

The data build, graph checks, and browser QA are deferred by the user's direction. No etymology or cross-language link is inferred.
