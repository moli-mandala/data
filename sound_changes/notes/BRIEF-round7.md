# Round 7 — second full audit (2026-09-29, 07:45 → 15:45)

**Checkpoint: by 11:30 send a ≤ 60-word progress note. Stop editing 15:00; final message (≤ 150 words)
by 15:10.** Everything in BRIEF-audit.md still applies (method, tools, hard rules), with the changes
below. Read your group's Round 6 sections first; build on them, don't redo them.

## Gates (tightened: the test split has now been consulted for six rounds)

- `gates.py` verdict must be ACCEPT (or ACCEPT-IF-JUSTIFIED with a written bare-vs-extended case).
  The accent gate is now **ext_acc** (accent of the variant the extension-aware score picks).
- **No effect-size threshold** (dropped at the user's request): any change that passes the gates may
  go in. A multi-rule change (e.g. vv > bb + a-loss after i) is installed and gated jointly.
- **Train decides, test vetoes:** choose candidates by train evidence only; test must not get worse.
  Do not tune a rule's environment to test words.
- Every accepted rule needs a mechanism in its comment (what happened, with 2–3 example words).

## Scoring change (08:20, user decision): vowel qualities are no longer merged

Headline scores (plain, extension-aware, verb-menu) no longer fold ə ʌ ɛ ɔ ɪ ʊ into a e o i u (only
spelling variants are unified: ε → ɛ, ǝ → ə, υ → ʊ), and they drop **every** accent/tone mark from
vowels (circumflex, caron, contour tones too; the accent score keeps them). ɨ is now a vowel.
Re-baseline everything. Lects whose sources write lax/open vowels got worse (ktg, kul, G, K, BH …)
until their chains produce those vowels. **This is now focus item 0 for affected lects:**
- where the quality is phonemic/regular, derive it by sound change (e.g. Gujarati a+u > ɔ, a+i > ɛ;
  Kashmiri/Kotgarhi ə, ɔ from their conditioning);
- where it is one source's transcription of the same phoneme (a source writing ʌ for a, ɪ for i),
  add an `@notation` rule in a `%source <key>` block for that source only — never a chain rule;
- `residuals.py` shows the vowel-quality class; check sources with `gold_src` before deciding.

## Tone score (09:00): tone TYPE is now scored, and the Vedic accent is addressable

- New metric `tone` / `ext_tone` (gates.py prints both; ext_tone is gated): accent position as in the
  accent score (0.5 once if missing/misplaced) plus 0.25 when the accent is on the right vowel with the
  wrong tone class. Classes: high (acute, vertical line), rising (caron, macron-acute), falling
  (circumflex, macron-grave), low (grave). Only sources that write a rising/falling/low mark somewhere
  in the lect are tone-scored (acute-only sources mark stress, not a tone type).
- A rule that only writes tone/accent leaves every segmental score unchanged; gates.py then accepts it
  when ext_tone (or ext_acc) improves on train and neither gets worse on test ("prosody-only").
- A rule can write a tone directly (`{a ā} > {ǎ ā̌} / …`); an explicit rising/falling/low mark wins
  over the generic realised accent. Plain/ext NED ignore all marks, so tone rules cost nothing there.
- New feature `Vedic` (like `Stress`): true only on the vowel carrying the **Vedic** accent on the
  latent tier (after `V > $1-acc`), so tone rules can be conditioned on it
  (e.g. `Vedic > $1 …`, `{ā} > {ā̂} / _ … // …`). Placement-rule accents are `Stress`.
- Hypothesis to test in the tonal lects (Mai, Gowro, bhatr, Bshk, Sh, Phal, Kho, Kt): does the Vedic
  accent predict tone TYPE (rising vs falling), possibly with vowel length, syllable weight or a lost
  aspirate/laryngeal as co-conditioners? Compare against positional/weight-based placement; accept only
  what passes the gates (ext_tone included).

## Engine fixes after the start (re-baseline affected lects)

- 08:15 split-off accent marks on ɨ ə ʊ are dropped in plain scoring (Nuristani report).
- 08:20 precomposed έ ὲ are unified to ɛ + mark (West Pahari report): ktg, bhal, khash, bhad, Ash, Pr,
  Kt, Wg, D, Sh, Kal, Kho, Phal, Pas, Dm.
- 08:45 `realize()` keeps a rule-written falling/low tone (â à) on a Stress-carrying vowel (it was
  overwritten by the acute). Put tone-writing rules at the chain's end so later rules still match.
- 09:10 `%gold-drop <mark>` now also applies to the accent and tone scores (it used to exempt accent
  marks, so `%gold-drop caron` did nothing). Use it for a mark that is one source's letter, not accent/tone
  (Strand's glide ǐ); a scoring declaration, noted not gated.
- 09:25 `canon` splits every a+i / a+u token whatever its marks (ãĩ ãũ āi āu …), as it already did
  for ai/au: gold ãũṭab no longer costs an edit against predicted ã ũ ṭ a b (Mth, Bi, Bhoj …).
  Re-baseline before the next candidate.
- 09:40 CDIAL's ɔ̌ (house spelling of Turner's long centralised ɔ̈̄; 96 forms: Ku, Kho, bhal, ktg, Kal,
  Sh, Pas, Wg, Pr, Kt …) is no longer read as an accent or rising tone, so one such form no longer makes
  the CDIAL source tone-scored. Re-baseline accent/tone if your lects have it.
- 10:00 `%gold-drop <mark> @SOURCE` drops the mark from that source's forms only (Strand's ǐ in a lect
  whose other sources write real tones, e.g. bhatr).
- 10:35 verb flag (`word_flags`): the gloss's first word is read without punctuation ("releases," was
  missed: Dhp, Pk, Pa, OAw …), and a bare English verb gloss ("bring", "climb up"; list in
  english_verbs.txt) on an -ati/-atē/-ayati headword counts as a verb (Kullui). More words get the verb
  menu; re-baseline all lects. Verb-only @morph rules that were blocked may now be removable.
- 10:50 … and a leading sense number ("1) eat; 2) bite") is skipped (Kullui).
- 11:40 a bare (consonant-final, often starred) root glossed in the 3rd person or as a clause is a verb
  (*cāh "desires"); rule @tags are only those before the comment's em dash (an "@morph" mentioned in the
  note is no longer a tag). Re-baseline.

## Focus for this round (in order of expected value)

1. **Open items** listed in your notes (e.g. Punjabi tone, Sauji, thematic Nuristani citations,
   Kashmiri nd/mb, Kalasha -ak, Konkani/Marathi leftovers, Sinhala -iya doublets).
2. **Accent** where your lects mark it (Dardic tone/pitch lects, Kashmiri, Kalasha, Palula, Shina,
   Punjabi, Sindhi, Kho …): the extension-aware accent score (`ext_acc`) is new — audit the
   accent-placement rules (`$1+stress`, `Stress` class) against it: placement, tone-from-aspirate
   (Punjabi), vowel-length ↔ stress interactions.
3. **Consistency inside the group:** the same historical change written differently in neighbours
   (different environments, orders or members). Where the variants are equivalent or one generalises
   cleanly, make them identical local rules (so centralize.py can merge them) — only if each lect's
   gates pass.
4. **Near-misses** (NED ≤ 0.25) and **rules that fire rarely or end mostly in misses** — as before.
5. **Per-lect menus:** `%verb-menu` / `%ext-menu` items still need a same-shape placebo whose
   consonant survives, beaten on train and test in that lect.

6. **Legacy `@morph` chain rules** (10:05): 305 rules tagged `@morph` still sit in 130 chains (outside
   `%source` blocks), e.g. Khol's infinitive `a {i ē} > e n / _ #`. Test removing each (leave-one-out,
   then jointly); remove it when the gates pass. The ext/verb menus and per-lect menus should now derive
   what these rules pasted on. A plain-NED loss from removing one is the bare-vs-extended artifact
   (ACCEPT-IF-JUSTIFIED). Keep one only when removal fails a gate, and say which.
   **Neutral removal (10:15):** if removing it leaves ext, verb, ext_acc and ext_tone unchanged or better
   on train and test (gates.py says REJECT because ext train did not improve), remove it anyway: it
   only served the bare headword. Note "neutral removal" and the plain delta.

## Not allowed (unchanged)

No `@morph`, `if:verb`/`if:ext`, numeral or `if:starred` citation shortcuts; no edits to engine files,
`changes.sc`, other groups' files; no derive/DB/synth runs.
