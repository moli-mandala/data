# Simplifying the rule file (2026-10-05/06)

Goal (user): the per-lect chains were heavily duplicated; state shared changes once, merge rules that
do the same thing, and drop lect-specific lines that do not earn their place.

## Steps

1. **One file.** 151 one-lect files (11,610 lines) → 12 family files (exact merge, 3,993 lines) →
   one `rules/chains.sc` with the families as `%set`s (`families.py merge`, `unite`). Every lect's
   compiled chain was unchanged.
2. **Folding duplicates** (`tmp/sound_changes/families/compact.py`): a line held twice (lects ordered
   it differently) is folded into its most-used copy when that leaves each moved lect's output
   exactly unchanged (every fork of every menu variant, with latent accents and per-source tails).
   Family files: 379 moves. After the merges below, on chains.sc: 986 moves, 2,270 → 2,089 code lines.
   (A first run lacked the tail/accent comparison and changed 6 lects; they were restored.)
3. **Merging similar rules** (`tmp/sound_changes/similar/similar.py`, `merge.py`): each distinct line
   is fingerprinted by the (form → output) changes it makes on ~40k intermediate forms gathered
   from every chain; pairs are scored by Jaccard via an inverted index (~2 min). Per round, the
   most similar disjoint pairs are taken; each lect with the minority text tries the majority line
   in place and keeps it when no score gets worse (ext NED, verb, accent, tone; train and test).
   Regular-vs-`optional:` and near-identical pairs were tried too (the user: those differences may
   be mistakes). Rounds 1–3 at Jaccard ≥ 0.85, 4–6 at ≥ 0.7: 370 of 806 lect switches accepted
   (135, 94, 60, 43, 29, 9).
4. **Dropping lect-specific lines** (`ablate.py`): each line only one lect takes was removed,
   greedily, when no score got worse: 122 of 1,150 lines went.
5. **Cleanup** (`unused.py`): 71 central changes and 1 block that nothing referenced any more.

## Result

- Scores (after round 6): test ext NED 0.2679 → 0.2666, exact 30.73% → 30.89%; train 0.2627 → 0.2618; no lect worse on any gated score (ext, verb, accent, tone; train and test).
- Notable merges: `hiatus-v-loss` → `v-loss-optional` (31 lects; the regular rule was the sporadic
  one), `r-a-optional` → `r-vocalic-multi` (31 lects), `nasals-dental-merger-dardic` folded into
  `nasals-dental-merger`, `use X` lines replaced by the block wrapping X (finals-syncope,
  nasals-nc-anusvara, nasals-retroflexion, finals-ap-shortening…), duplicate central definitions
  (`hiatus.v`/`.v.2`, `hiatus.iiaa.final`/`.2`, `ai-au.aiau`/`.3`…).
- Backups of every stage: `tmp/sound_changes/backups/`.

## Not done yet

- Folding again after rounds 4–6 (a full fold takes ~3.5 h on chains.sc).
- Widening selectors: lines most of a family has (`families.py shared`) tried in the lects that
  lack them, gated like a law (this changes behaviour).

## Widening: filling the tree's branches (2026-10-06/07)

Goal (user): "fill up as many circles as possible", i.e. more rules shared over large groups.
`tmp/sound_changes/widen/widen.py`: the lect tree is average linkage on shared rule lines (no family
constraint); a line hangs on the smallest branch holding all its lects. Each lect of that branch
lacking the line tries it at the line's place in the file; light group testing (a lect tries all its
candidates at once, failing groups are halved) keeps an addition when no score gets worse. Four
passes (fit ≥ 0.5, 0.3, 0.25, 0.2 capped at 8 per lect; earlier rejections skipped): 912 + 1,209 +
861 + 703 additions. Two duplicate passes moved lects from a minority copy of a line onto its main
copy under the same gate: 560 + 17.

| | before simplifying | before widening | now |
|---|---|---|---|
| rule lines | 2,159 | 1,948 | 1,759 |
| single-lect lines | 1,131 | 1,042 | 919 |
| mean lects per line | 3.58 | 3.90 | 6.41 |
| lines with ≥10 / ≥30 lects | 171 / 22 | 153 / 32 | 222 / 60 |
| test ext NED / exact | 0.2679 / 30.73% | 0.2666 / 30.89% | 0.2608 / 31.55% |
| train ext NED | 0.2627 | 0.2618 | 0.2579 |

No lect is worse than at the start on any gated score. Views: chaintree (tree), gitgraph (git-style
commit graph), chaingraph (matrix + network) in tmp/sound_changes/.
