# Brief, round 2: vowels and the Vedic accent

Same group, same files, same constraints as `BRIEF.md`:
- write only your lects' rule files, frontier CSVs, your notes file and `tmp/sound_changes/<group>/`;
- do not edit engine files;
- run at most one Python process at a time;
- tune on **train** only.

This round has one focus: **vowel changes and anything conditioned by the Vedic accent**. The
first round under-explored these because the tools only proposed changes conditioned on adjacent
segments.

## What is new in the engine (documented at the top of `sca.py`)
- **`C*` / `!#*` (zero or more) in environments.** Syllable-level contexts are now expressible:
  - first syllable: `# C* _`;
  - final syllable: `_ C* #`;
  - harmony with the next vowel: `_ C* {a á}`;
  - pretonic: `_ !#* Acc`;
  - post-tonic: `Acc !#* _`.
- **Built-in vowel classes** `Acc` (acute/grave), `Long`, `Short`.
- **`if:FLAG` rule prefixes**, with word-level flags derived per etymon: `verb`, `numeral`,
  `starred`, `root`, `accented`. Examples: `if:verb,!numeral a t i > n ā / _ #`,
  `if:!accented …`. Use them to fix citation-morphology rules that were hitting numerals and -ti
  nouns. `evaluate.py apply … --flags verb` shows a derivation with given flags.
- **Optional tokens in targets**, one token each: `(a) (y) a t {i ē} > i k / _ #`.
- **`$n` in environments and exceptions**, e.g. `// _ $1` (not before an identical segment).
- **Set difference** in classes and sets: `class Unasp = Stop - kʰ gʰ …`, `{Stop - t tʰ}`.
- **`%gold-drop tone`** (or `caron`, `circumflex`, or a literal mark) strips tone marks from this
  lect's attested forms, for sources whose tone the chain cannot predict.
- **Tokenisation fixes:** accented diphthongs (`saínya`) are now one segment, and they merge to
  `ai` when `@acc-loss` strips the accent. Editorial braces in headwords are ignored. Scores
  shift by ≤ 0.001.

## New tool
```
python3 sound_changes/evaluate.py probe sound_changes/rules/OIA-X.sc [--vowel a] [--top 25] [--min 12]
```
It aligns every OIA vowel, read with its accent, to both the attested and the predicted form.
It tabulates the outcomes by these features, singly and in pairs:
- `pos`: first, medial, last or mono syllable;
- `acc`: tonic, pretonic, posttonic or unaccented;
- `next`: the next vowel;
- `syl`: open or closed;
- `nextσ`: the next syllable light or heavy.

It lists the cells where one outcome dominates the attested forms but the chain predicts
something else, with an estimated net gain and a suggested rule. The estimate ignores
interactions; always confirm with `score --loo`. Contexts are read on the headword, so these
rules belong **early** in the chain, and accent-conditioned ones must come before `@acc-loss`.

`suggest` now also proposes first-syllable, final-syllable and next-vowel candidates.

About 25–30% of headwords carry an accent. About 40% are Turner's `*` reconstructions, which are
almost always unaccented. So an accent rule can only help the accented etyma; `if:accented` /
`if:!accented` separates them.

## Restart note (11:00)
The first round-2 agents were stopped at about 06:39, after editing most rule files. A snapshot of
the files as they stood then is in `tmp/sound_changes/snapshots/round2-partial/`; round-1 scores
are in `sound_changes/results/summary-round1.csv`. **Start by auditing those partial edits** in
your lects: run `score --loo` and keep only round-2 rules with positive LOO that read as sound
changes (or clearly tagged `@morph` fixes); revert the rest. If a lect's test NED rose by more than
0.005 over round 1, find out why. Any gain from `%gold-drop` is a change of scoring, not a sound
change: report it separately (score with and without the directive).

## What to do
1. For each lect, largest first, run `probe` and look for systematic vowel conditioning:
   - accent-conditioned outcomes: pretonic reduction or loss, tonic lengthening, a > o/ɔ under
     stress, accent-conditioned apocope or syncope;
   - syllable position: first-syllable vs final-syllable treatment, penult lengthening;
   - harmony or umlaut: the next vowel colouring this one;
   - weight: open vs closed syllables, light vs heavy next syllable;
   - vocalic r̩: the a/i/u split. Test the accent, the neighbouring consonants and the next
     vowel.
   - diphthongs, and long-vowel shortening beyond the two-mora law.
2. **Test the Vedic accent directly** wherever a vowel change looks positional. Is the accent a
   better conditioner than syllable position or weight? Report the result either way.
3. **Adopt only rules that pass**: positive train `--loo` value, and they read as sound changes.
   Tag them with CATALOG tags (`@stress` as a co-tag on accent-conditioned rules, `@umlaut`,
   `@vowel-shift`, `@syncope`, `@a-rounding`, `@final-vowel-loss`, …). Check their order against
   `@acc-loss`, length changes and apocope.
4. Where cheap, also fix the citation-morphology rules that misfire on numerals and -ti nouns,
   using `if:verb` / `if:!numeral`.

## Deadlines and reporting
Stop editing rules by **12:30**. Then rerun `frontier` for every lect you changed. Add a section
**"§7 Round 2: vowels and accent"** to your notes file (or §8 if you already have a §7)
containing:
- per lect: train/test NED before → after this round;
- every vowel/accent rule adopted, with its LOO value and examples;
- every hypothesis tested and rejected, with its number (especially accent tests);
- what the accent does and does not condition in your group.

Be completely done by **12:50**. Your final message should be ≤ 200 words, with the numbers and
the most interesting vowel/accent findings.
