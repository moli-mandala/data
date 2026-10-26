# Blocks round — consolidate sound changes into shared include blocks (2026-09-30, 01:30 → 09:30)

**Stop editing 07:30; final message (≤ 200 words) by 07:45.** Checkpoint note (≤ 60 words) by 04:30.

## Goal

Consolidate the chains: the same historical change should be ONE shared unit, used by every lect that
underwent it, across families where the evidence supports it. The unit is an **include block**:
`sound_changes/blocks/NAME.sc`, pulled into a lect's chain with `include NAME` (README § Blocks;
`blocks/mia-core.sc` is the model). A block is an ordered sequence of changes applied at one point in the
chain; blocks may include other blocks; a block may hold `use` lines, plain rules and its own classes.

Consolidation means, per lect: replace the lect's own copies/formulations of a process by `include X`
(at the point where its first copy was, or where the history puts it), with **no score loss**. Neutral
consolidation (identical predictions) is a success. Gains are welcome. Staging is encouraged when it is
the better history: e.g. a broad early allophonic change in one block (intervocalic l > ḷ) and a
narrower later one in another (ḷ > r in a subset), instead of one-off l > r rules.

## Where you work (do NOT edit live files)

- Base (read-only): `tmp/sound_changes/snapshots/pre-blocks/rules/` (= live rules at 01:30),
  `…/pre-blocks/changes.sc`, `…/pre-blocks/scores.json` (full scores of the base).
- Your workspace: `tmp/sound_changes/blocks-<theme>/` with `rules/` (copy the base there) and
  `changes.sc` (copy). Write candidate lect files ONLY there.
- New block files go directly in `data/sound_changes/blocks/`, named `<theme>-….sc` (no live lect includes
  them until integration, so they cannot affect anything). Never edit `mia-core.sc` or other themes' blocks.
- Do not edit `sca.py`, `evaluate.py`, `changes.sc`, live `rules/`, or other themes' workspaces.
  If you need a new shared formulation, write it as a plain rule inside your block.

## Checks (light — the user dislikes heavy testing; no per-lect multi-variant sweeps, no placebo sweeps)

- Exact local gate when your edit can only change words matching a pattern (e.g. only words with `l`):
  `tmp/sound_changes/unify/local.py` (`local_delta(stem, cand_path, regex)`; ± on lect-level means).
- Otherwise the paired sample gate: `python tmp/sound_changes/unify/sample_score.py CAND_DIR OUT.tsv 200 STEM …`
  (CAND_DIR = your workspace; compares with the live chain on the same 200 etyma per split; columns:
  Δ ext, verb, ext_acc, ext_tone, plain for train then test).
- Pure-consolidation shortcut: if the lect's predictions are identical (bare + menu variants on its words),
  it is neutral and accepted (see `tmp/sound_changes/unify/prefix.py`, class `Lect`, for how to compare).
- Accept a lect's edit if ext train ≤ 0 and ext test ≤ 0, verb ≤ +0.001, ext_acc/ext_tone ≤ 0 (train & test).
- Decide the variant per lect by linguistics/its existing rules, test once, fall back if it fails.
- One scoring process at a time per agent, `nice`; the machine has 8 GB and 7 agents share it.
  Do not run `score_all.py`, `derive.py`, the DB build, or `centralize.py` — the coordinator does.

## Deliverables (in your workspace)

- `report.md`: blocks (file, what, why, history), which lects include each (and which were tried and
  rejected, with the reason), per-lect Δ (ext/verb/acc train/test), and ideas not done.
- `touched.txt`: one lect stem per line whose candidate file you want integrated.
- Candidate files `rules/OIA-<stem>.sc` for exactly those lects, based on the base snapshot.

## Hard rules (unchanged)

No `@morph`, `if:verb`/`if:ext`, numeral or `if:starred` shortcuts. Tags: give each block rule the usual
`@tag` before the em dash, and a note with 2–3 example words. Keep MIA-stage changes before NIA ones.
