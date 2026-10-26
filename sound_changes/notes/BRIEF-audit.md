# Round 6 — full error audit, lect by lect

**Deadline: stop editing 05:55, final message (≤ 120 words) by 06:05.** Untouched lects are fine: list
them as not done. Work largest lects first (by pairs), then smaller ones.

## Goal

For every lect in your group: classify the chain's remaining errors, find the *systematic* ones, and
fix those with regular sound changes (or by fixing rule bugs/ordering). Leave lexical and source
variation alone; say so in your notes.

## Method (what worked for Sindhi, Hindi, Gujarati today — see northwest.md §12–13, hindi.md §12,
rajasthani-gujarati.md §12)

1. `python3 sound_changes/residuals.py sound_changes/rules/OIA-<stem>.sc > /tmp/<stem>.txt` — classes
   of misses (share of error), random examples per class, and the commonest predicted→attested ending
   pairs. Read 50+ examples; then look for patterns covering ≥ 5 pairs.
2. For each candidate: copy the rule file to a scratch dir **keeping the file name** (`OIA-<stem>.sc`),
   edit it, and run `python3 sound_changes/gates.py sound_changes/rules/OIA-<stem>.sc <scratch>/OIA-<stem>.sc`.
   Test candidates one at a time, then the accepted ones together.
3. Before accepting, list the words a candidate makes better and worse (apply both chains to the
   dataset and diff); a rule that fixes 20 words but breaks 70 is lexical variation, not a rule.
4. Install only what passes. Write `## Round 6 audit` in your group's notes file: per lect, the error
   profile (top classes), kept changes with before→after (train/test ext, verb, plain, exact),
   rejected candidates and why, and what remains.

## Gates (gates.py prints the verdict)

- Primary: extension-aware NED on train improves; ext test not worse.
- Verb-menu NED train/test not worse by more than 0.001; extension-aware accent NED (`ext_acc`, the
  accent of the variant the extension-aware score picks; added mid-round) not worse.
- (Deprecated 2026-09-30: the plain bare-headword NED gate and the bare-headword accent score are no
  longer reported or gated.)

## Things to look for (found today)

- **Rule-order bugs**: e.g. medial syncope running before final vowel loss, so a final closed syllable
  loses its vowel (G aŋgana > *ā̃gṇ for ā̃gaṇ). Test moving the rule.
- **Rules that over-match**: `$1` ignores aspiration (S: aspirated geminates became implosive); check
  what your geminate/`$n` rules really match.
- **Hiatus leftovers** after consonant loss: -aī > -ī, -āo > -o, -ãu > -õ, medial ai/au > e/o.
- **Dissimilation / conditioning** that splits an apparently random outcome (S: initial g j stay
  plain when the word already has an implosive).
- **Metathesis/cluster outcomes** that need an early rule (S: dīrgha > ḍrigʰo via a copied r).

## Hard rules

- No `@morph` rules, no `if:verb` / `if:ext` / numeral conditions, no `if:starred` citation shortcuts.
  A stratum condition (`if:!starred` for a change the late expressive vocabulary did not undergo) is
  allowed only with evidence both ways, noted.
- Do **not** edit `evaluate.py`, `sca.py`, `changes.sc`, `derive.py`, or other groups' files. To change a
  shared `use NAME` rule for your lect, replace that line in your lect file with a local rule. If you
  think a menu item (EXT_MENU / VERB_MENU) is missing, report it with a placebo test (a same-shape
  suffix whose consonant survives in your lects, e.g. -āranaka vs -āpanaka) — do not add it.
- Per-lect verb formations: a `%verb-menu +ITEM …` line (top of the lect file) adds items to that
  lect's verb menu only. Add one only with a placebo test in that lect (real beats a same-shape
  placebo whose consonant survives by > 0.002 on train and is better on test); note the numbers.
- Don't run derive.py, the DB build, synth pipelines or frontier. One lect's scoring at a time per
  agent; keep memory modest (8 GB machine shared by 12 agents).
- Already done today (build on it, don't redo): S (§12–13), H (§12), G (§12), plus the verb_form fix and
  the new verb-menu items (-āpanaka, -āpanakam, -āpitavyakam) and gloss-gated denominal verb forms.
