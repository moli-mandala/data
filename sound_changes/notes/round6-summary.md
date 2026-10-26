# Round 6 — full audit, summary (2026-09-29)

Twelve group agents audited every lect (brief: BRIEF-audit.md; per-group write-ups under
`## Round 6 audit` and later-pass sections in each group's notes file). The coordinator had first
audited Sindhi, Hindi and Gujarati by hand (northwest.md §12–13, hindi.md §12,
rajasthani-gujarati.md §12). Before: `tmp/sound_changes/snapshots/pre-round6/`; both sides scored
with the final engine (`sound_changes/score_all.py`).

## Result (151 lects)

| metric | median before → after | pair-weighted before → after |
|---|---|---|
| ext NED, train | 0.3562 → 0.3439 | 0.3071 → 0.2937 |
| ext NED, test | 0.3626 → 0.3488 | 0.3156 → 0.3017 |
| ext exact, test | 15.8% → 17.8% | 23.3% → 25.4% |
| verb-menu NED, test | 0.3521 → 0.3250 | 0.2795 → 0.2469 |
| plain NED, test | 0.4434 → 0.4434 | 0.3938 → 0.3931 |

ext NED improved on train in 139 lects and on test in 131; it got worse in none.

## Engine and tool changes made during the round

- `residuals.py` (error classes, ending pairs; applies %source tails) and `gates.py` (verdicts).
- `verb_form`: headwords that are not -ti presents drop their final vowel (jita + -anaka), -tē/-ti
  + aya recognised; accents stripped from vowels only (ś no longer becomes s).
- `word_flags`: verbs glossed as survey clauses ("he killed", "come!, he came") are verbs.
- `gold_tokens`/`accent_tokens`: IPA stress marks ˈ ˌ stripped.
- **Extension-aware accent** (`ext_aned`, gated as `ext_acc`): the accent of the variant the
  extension-aware score picks. The bare-headword accent score, like plain NED, only sees the bare
  headword and blocked correct ending changes; the case-by-case waivers granted before it existed
  (Nuristani apocope, NW citation-rule removals) were re-gated and pass under it.
- Verb menu (global): -āpanaka, -āpanakam, -āpitavyakam (placebo-tested earlier in the day).
- **Per-lect menus**: `%verb-menu +ITEM …` / `%ext-menu +ITEM …` in a lect file, each item
  admitted only where it beats a same-shape placebo whose consonant survives, on train and test
  (evidence in the line's comment). Global additions of -aka, -alla, -iyanakam only leaked chance fits
  (placebos beat them in 9, 9 and 4 lects). In use: root + -aka (14 lects), -alla (Bhoj), -iyanakam
  (N, Ku, D, bhad, kc), -ikke (Darai), -itaka (W. Hindi / Rajasthani -yo pasts), -itasya (E. Hindi -is
  past), -ayikka/-āpayikka (Kho, Sh), -anta (Ko), -akkanakam (M); noun -iyakam (G, OG), pl. -akā
  (Rajasthani survey lects).

## Decisions

- Citation shortcuts removed where the extension menu derives the forms (survey -o in 9 Rajasthani
  lects; -ā/-o in P, khet, thar, awan, poth, kcch); plain NED rises there by design.
- Nuristani: full final -a apocope in Kt and Kam (the tonic-á "accent trace" is re-read as the -aka
  extension's vowel); reverted in PNur and Wg (verb score). Kt's verb train is +0.0014 (test −0.0028),
  a kept exception.
- Kept: 96 new central changes after re-centralising (`centralize.py --exact`, 427 uses).
- Declined: engine change to read Kotgarhi v́ as a vowel (notation of one source).

## Open follow-ups

- Punjabi tone; Sauji; Kashmiri nd/mb; Kalasha -ak.
- A thematic citation item for Strand's Nuristani verbs (*rava) that survives apocope.
- Most remaining error is lexical or source variation (doublets, regional forms within CDIAL's H.,
  survey suffixes, citation-form choice).
