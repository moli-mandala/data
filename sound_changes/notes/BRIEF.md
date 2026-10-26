# Brief: hill-climb OIA → daughter sound-change lists

You are one of 12 parallel agents. Each owns one group of lects (see `sound_changes/languages.tsv`,
column `Group`). The goal is an *ordered, linguistically motivated* list of sound changes from
OIA (CDIAL headwords) to each lect in your group, hill-climbed on an edit-distance eval, plus
notes that support a family-wide synthesis of shared changes and relative chronology.

Working directory: `/Users/aryamanarora/Documents/Code/jambu-all/data` (run everything from here).

## Read first
- `sound_changes/README.md` — the tool and metric.
- `sound_changes/sca.py` docstring — rule syntax (classes, sets, `$n` refs/modifiers, `C $1`
  geminate matching, `!X`, `(X)` optional, `#`, `repeat:` prefix, `%exclude-source KEY`).
- `sound_changes/CATALOG.md` — the shared tag vocabulary. **Every rule line must carry tags.**
- Worked examples: `sound_changes/rules/OIA-Pk.sc`, `OIA-H.sc`, `OIA-Kho.sc` (written before the
  tag convention — the owning agents should add tags to them).

## Commands (all cheap; single-threaded)
```
python3 sound_changes/evaluate.py score   sound_changes/rules/OIA-X.sc --loo --errors 30
python3 sound_changes/evaluate.py suggest sound_changes/rules/OIA-X.sc          # best next rules + insertion point (train)
python3 sound_changes/evaluate.py apply   sound_changes/rules/OIA-X.sc 'hásta' '*pōstaka'
python3 sound_changes/evaluate.py frontier sound_changes/rules/OIA-X.sc --out sound_changes/results/OIA-X.hand.frontier.csv
```
Pairs are in `sound_changes/pairs/OIA-X.tsv` (columns: Etymon_ID, Etymon, Reflex_ID, Reflex, Gloss,
Split, Source). `X` is the file stem in the `File` column of `languages.tsv` — usually the
Language_ID, but IDs differing only by case get a suffix (`As` → `As-Ul`, `as` → `as-ll`, `Ko` → `Ko-Ul`).
A greedy automatic baseline for each lect is being generated in the background into
`sound_changes/rules/auto/` and `sound_changes/results/OIA-X.auto.frontier.csv`; you may read
those files for ideas, but do not run `induce` yourself.

## How to work
1. For each lect, first look at the data: a sample of pairs, the sources mixed in it, and the
   transcription. **Check the notation.** If a source's notation is systematically different
   (digraph aspirates `kh`, `ə` for schwa, doubled vowels for length, IPA vs Indological letters),
   either add clearly tagged `@notation` rules at the *end* of the chain that map into that
   notation, or drop the source with `%exclude-source KEY` if it is malformed/unusable. If a whole
   lect's data is unusable, skip it and say why in your notes.
   **New (01:50):** when one *source* within a lect has its own notation, wrap source-specific
   `@notation` rules in a block — they run after the main chain only when scoring that source's
   pairs, and do not count as chain rules:
   ```
   %source webster boehm
   {i u} > {ī ū} / _ #        ; @notation — Webster writes final long vowels
   %end
   ```
2. Build a group-level base chain once (the changes shared by your group), then copy and
   specialise it per lect. Put shared changes in the same relative order across your lects unless
   the data says otherwise — the synthesis compares orders across files.
3. Hill-climb on the **train** split: `score --loo` (per-rule value), `--errors` (worst misses and
   residual confusions), `suggest` (data-driven next rule + best insertion point). Accept a rule
   only if it is a plausible *sound change* (or a clearly tagged systematic `@morph`/`@notation`
   rule) — no word-specific rules, no rules that only fit a handful of items, prefer class-based
   general rules. Remove rules whose `--loo` value is ≤ 0 unless they are needed for ordering.
   Try reorderings where rules interact (feeding/bleeding) and keep the order the data prefers.
4. Do **not** tune on the test split. Look at test numbers only in the final `score` / `frontier`.
5. Rough time budget: spend effort in proportion to data size, but get *every* lect to a
   reasonable first version before polishing any one. Small lects (< 300 etyma) can mostly reuse
   a sibling's chain with light edits.

## Hard constraints
- Write only: `sound_changes/rules/OIA-<your lects>.sc`, `sound_changes/results/OIA-<your lects>.hand.frontier.csv`,
  `sound_changes/notes/<your-group>.md`, and scratch files under `tmp/sound_changes/<your-group>/`.
- Do **not** edit `sca.py`, `evaluate.py`, `extract.py`, `CATALOG.md`, `pairs/`, other groups'
  files, or anything under `cldf/` / `data/`. No git commits, no builds, no `make`, no installs.
  If you need an engine feature, work around it and describe the need in your notes.
- Keep to at most one running Python process at a time (the machine is shared by 12 agents).
- Check the clock with `date`. **Stop editing rules by 06:35** local time. Then, sequentially for
  each of your lects, run `frontier` (it writes the ablation curve), and write your notes. **Be
  completely done by 07:05.** If you are running late, skip frontiers for the smallest lects.

## Notes file (`sound_changes/notes/<your-group>.md`) — the synthesis depends on it
1. Table: lect, #etyma, #rules, train/test NED (null → final), test exact %, auto-baseline test
   NED at a similar rule count if available, excluded sources.
2. Your group's base chain in order, with tags, and which lects deviate where.
3. Relative-chronology evidence: rule pairs whose order matters (swapping them clearly hurts
   train NED — say by how much), and what that implies.
4. Shared vs. lect-specific changes; sub-grouping suggested by the changes.
5. Anything novel, surprising or qualitatively interesting: unexpected correspondences, apparent
   irregularities that look systematic, likely borrowing strata (tatsama/tadbhava), dialect
   mixture in the sources, data/notation problems, CDIAL mislinkings.
6. Any new `@x-…` tags you defined, with definitions. Engine features you wished for.

Your final message should be a ≤ 250-word summary: the table's key numbers, the 3–5 most
interesting findings, and anything you could not do.
