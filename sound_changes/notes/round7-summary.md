# Round 7 — second full audit, summary (2026-09-29, 07:45 → 15:45)

Twelve group agents audited every lect again (brief: BRIEF-round7.md; per-group write-ups under
`## Round 7` in each group's notes file). Before: `tmp/sound_changes/snapshots/pre-round7/`; after
(before centralising): `tmp/sound_changes/snapshots/post-round7-precentral/`. Both sides were scored
with the final engine (`sound_changes/score_all.py`).

## Result (151 lects)

Final engine (after every fix below), both sides scored identically. Lower is better except exact.

| metric | median before → after | pair-weighted before → after | lects better / worse |
|---|---|---|---|
| ext NED, train | 0.3559 → 0.3426 | 0.3012 → 0.2895 | 127 / 0 |
| ext NED, test | 0.3597 → 0.3492 | 0.3088 → 0.2969 | 109 / 0 |
| ext exact, test | 16.4% → 17.9% | 24.7% → 26.6% | 71 / 1 (Buksa) |
| verb-menu NED, test | 0.3600 → 0.3396 | 0.2688 → 0.2622 | 33 / 3 |
| ext_acc, test (59 lects) | 0.4953 → 0.4800 | 0.4744 → 0.4557 | 28 / 2 |
| ext_tone, test (19 lects) | 0.5218 → 0.5218 | 0.4857 → 0.4560 | 10 / 1 |
| plain NED, test | 0.4515 → 0.4610 | 0.3975 → 0.4089 | 52 / 71 |

- Plain NED rises by design: 207 legacy `@morph` citation rules were removed, which only served
  the bare headword (the extension-aware score derives those forms).
- The remaining "worse" entries:
  - verb: Wg, PNur, Kvari are the recorded waivers; Chitwan verb train +0.0012 is marginally over
    the 0.001 tolerance and is kept;
  - ext_acc/ext_tone: Mai, Gowro, bhatr, Bshk, Dm changed their `%gold-drop` declaration (tone marks
    are now scored), so before and after count different pairs. Re-gated with the new declaration on
    both sides, all five pass (Mai ext_tone 0.4810/0.5055 → 0.4500/0.4781).
- Largest ext test gains: ktg 0.440 → 0.322, Darai 0.309 → 0.222, Dungra Bhili 0.487 → 0.408,
  Kvari 0.467 → 0.390, Dhundari 0.432 → 0.359, Domari 0.458 → 0.395, Desia 0.387 → 0.325,
  Kashmiri 0.387 → 0.328.
- Centralised: 66 new shared central changes (215 uses); 616 central changes, 5,558 uses.

## Scoring and engine changes made during the round

Scoring policy (user decisions):
- **Vowel qualities are no longer merged** in any score (ə ʌ ɛ ɔ ɪ ʊ stay distinct from a e o i u;
  only spelling variants ε/ɛ, ǝ/ə, υ/ʊ are unified). The headline scores drop every accent/tone mark;
  the accent score keeps position only.
- **Tone score** (`tone`, `ext_tone`): accent position (0.5) plus the tone *type* (0.25 for the right
  vowel with the wrong class: high / rising / falling / low). Only sources that write a
  rising/falling/low mark are tone-scored. New rule feature `Vedic` (the carried Vedic accent).
- **Gates**: no effect-size threshold; train decides, test vetoes; multi-rule changes gated jointly;
  prosody-only changes are accepted on ext_tone/ext_acc when every segmental score is unchanged;
  removing a legacy `@morph` rule is accepted when every gated score is unchanged ("neutral removal").

Bugs reported by the agents and fixed (each with a test; 46 tests):
- split-off accent marks on ɨ ə ʊ kept in plain scoring; precomposed έ ὲ not unified to ɛ;
- `realize()` overwrote a rule-written falling/low tone with the acute;
- `%gold-drop` did nothing for the accent/tone scores (now applies; `@SOURCE` restricts it to one
  source: Strand's glide ǐ in Kt, Kam, Ash, bhatr);
- `canon` did not split nasal/long diphthong tokens (ãĩ ãũ āi āu), costing one edit per pair (Mth,
  Bi, Bhoj …);
- CDIAL's ɔ̌ (house spelling of Turner's long centralised ɔ̈̄) was read as a rising tone, making whole
  CDIAL sources tone-scored;
- verb flag: glosses with punctuation after the first word ("releases,"), bare English verb glosses
  ("bring": Kullui), sense numbers ("1) eat"), and bare roots glossed in the 3rd person (*cāh
  "desires") were not recognised as verbs;
- rule tags were read from anywhere in the comment (an "@morph" mentioned in a note became a tag).
- (integration) a bare pronoun gloss ("we": asmad) counted as a survey clause, so the pronoun was
  scored as a verb; a clause now needs a word after the pronoun ("he killed").
- Data: CDIAL row f_f2ekngihsqnya (Kt "śikyà") is the OIA headword parsed as a reflex (stopgap
  `%gold-drop grave` in Kt; spun off as a separate task).

## Findings

- **Vowel qualities by rule** (the scoring change's item 0): lects whose sources write open/lax vowels
  recovered, and most now beat their pre-change scores. They got there by deriving the quality (Gujarati
  a+u > ɔ, a+i > ɛ; Rajasthani final -o > -ɔ in 13 lects; Kashmiri ɛ/ɔ lowering and -aka/-ikā umlaut;
  Kotgarhi/Kochi pretonic ə; Bengali-type a > ɔ with raising before a high vowel in Kamta and
  Bishnupriya; Desia a, o > ɔ; Darai a/ā > ə/a) or by a per-source `@notation` where one source
  spells a phoneme its own way.
- **Tone type (southern Dardic)**: the onset predicts it, not the Vedic accent. Rising follows voiced,
  breathy or aspirated consonants and h, and a final nasal (Maiyã, Gowro, Bateri, Bashkarik, Pashai);
  Tirahi has low. Shina (Schmidt): rising/falling follows where the long vowel came from (contracted
  = rising).
- **Vedic accent → tone type**: weak and consistent: a vowel that kept the Vedic accent leans
  falling, a stress retracted from a later Vedic accent leans rising.
  - Palula: 10% rising on the Vedic syllable, 64% with the accent two syllables later.
  - Degener's Shina: 85% grave on the Vedic syllable, 44% with the accent on the next.
  - Zoller's Kohistani lects: 50–66% falling vs 24–57%.
  - Only Palula's rule passes (ext_tone −0.0002); the Zoller biases shrink on test. The Vedic accent
    does not predict NW (P, D) stress position.
- **Prosodic isoglosses**:
  - one NW weight-stress rule (long penult, else first syllable) in S, srk, kcch, P, L, with a
    Hindko-type variant in khet/poth;
  - West Pahari initial stress in 9 more lects, with degemination after a long vowel (5 lects) and
    pretonic reduction.
- **Segmental**:
  - Lahnda ū > ū̃ before h (< kʰ tʰ: múkʰa > mū̃) in srk, poth, awan;
  - Rajasthani -aya-/-ada- > e before a consonant (12 lects);
  - Eastern Hindi -ita+-sya > -īs (6 lects);
  - Nepali/Kumaoni/Garhwali ut-+sibilant assimilation, final -Cy > -C, and nn staying dental vs n > ṇ;
  - Tharu ḍ > r flap (Majhi, Darai);
  - Domari early loss of final -a after a consonant, before the k of -aka was lost (kárman > kam
    vs taptá-ka > tátʌ);
  - post-sibilant deaspiration found independently in Shina-type Dardic (Phal, Ush) and in Romani
    (eur, RomWel).
- **Per-lect menus** (each beats a same-shape placebo on train and test):
  - MIA fem. -ī/-ā, -ika, -am (Pk, Pa, Dhp, NiDoc, As, Ap, Si, Md);
  - -āka in N, Ku, Garh, Dotyali;
  - the -av causative in RomWel/RomBalk;
  - -yo past in Dotyali.

## Decisions

- **Legacy `@morph` citation rules**: 313 in 132 chains before the round → 106 in 79 after. Every agent tested removal. Removed
  where the gates pass or where every gated score is unchanged ("neutral removal"; plain NED rises by
  design). Kept only where removal fails a gate (reason in each group's notes).
- **Nuristani apocope**:
  - A narrower rule (final -a lost after a single consonant but kept after a cluster) passed the gates
    outright, but only protected Strand's verb citations (-ikka, -tavya); nouns do not show it.
  - Rejected as a disguised citation shortcut.
  - Full apocope installed in Wg and PNur, as in Kt and Kam, with the verb gate waived: Strand cites
    verbs with a thematic -a that the chain removes (sárati > *salika).
- **Verb-gate waivers**, all for chance or citation matches:
  - Wg/PNur apocope, as above;
  - Kvari ɔ: vásati > bośo's old score was an -anaka chance match.
- Kept (removal worsens ext slightly): southern Dardic -ati/causative @morph rules in Mai, bhatr,
  Chil, Dm (+0.0007 to +0.0029 ext train); Mewati -ā past; MIA 11 of 19.

## Open items

- A Strand thematic-citation verb item that survives apocope (Nuristani).
- A tone rule combining onset and kept vs retracted Vedic accent (Palula, Degener's Shina, Kohistani).
- Punjabi tone (gains only on the bare form); Haryanvi tone; tone in dog/kgr/mand.
- Notations hidden inside one source key (Ku ɔ/ʌ/ε/ə and RomBalk i̦ inside CDIAL; Kochila superscript
  doublets; Webster æ/ɔ; Shackelford ɔ; Hendriksen v for u).
- The Kt śikyà data record.
- `gates.py`'s better/worse reflex count ignores `%source` tails (information only).

## Post-round unification (2026-09-29, after integration)

Informal, data-led unification; each change gated per lect (local exact gate for changes that only
touch words matching a pattern; paired 250-etymon sample gate otherwise), then one full score of
all lects against the post-round-7 scores: **0 gate violations**; ext train 0.3426 → 0.3421 median
(75 lects better / 0 worse), test 0.3492 → 0.3479 (37 / 0). Scripts in `tmp/sound_changes/unify/`;
snapshots `pre-core/`, `pre-fillin/`.

- **mr** (āmra > amba): 3 formulations in 45 lects; the unconditioned `m {r l} > m b` also turned
  initial mr- into b- (H mrakṣaṇa > *bākʰan). Now `m r > m b / V _` plus initial `m r > m / # _`
  where it helps (H, K, L, OAw, kcch, kiuth, ktg). Placement matters: P amb beside nimm (mb > mm
  earlier) needs mr > mb *late*; S āmo, ṭrāmo and Kho/eur/kaithal have mr > m. Installed in 45 lects.
  ut-st > uṭṭʰ rejected: CDIAL already spells uttʰ-, and reflexes are dental in ~97%.
- **MIA-first**: with the Pk or Pa chain prepended as a MIA stage, few lects fit, but most of the loss is
  proxy idiosyncrasy (leave-one-out on a 12-lect panel): final-consonant loss, -ayati > -ēti, ḍ > ḷ,
  Vedic-accent loss, r̩ > a, two-mora shortening, ṣṇ > ṇh, ny > ññ, Pali rC/hy/rdʰ. With the remaining
  common core, 48 of 137 NIA lects lose nothing (44 improve), 85 are within +0.005, 98 within +0.01;
  the misfits are Dardic, Kashmiri, Romani/Zargari and part of W. Pahari (they keep ś/ṣ and Cr).
- **Common MIA core block** (27 central changes, `use … ; @mia-core`, at the head of the chain;
  final -m > -ṁ and vv > bb dropped from the core — they broke -akam and -tavya > G -vũ): installed in
  22 lects (B, Or, OB, N, Ku, Garh, Dotyali, OP, Brj, OAw, SSr, Kannauji, Bagheli, WK, Rj, Bhatri,
  PauriBareli, kgr, thar, AdivasiOriya, Bishnupriya, mewari_bannoda); the lects' own copies removed.
- **Subfamily post-MIA blocks**: contiguous blocks fail everywhere (median +0.044 even when only
  reordering): post-MIA changes are shared as an ordered *sequence* interleaved with lect-specific
  changes. Fill-in instead (add the clade changes a lect lacks between its copies of their
  neighbours): 25 lects gain (e.g. RomBalk NT > ND, MB flap + dental n, Buksa l > r + CL, Bagri v- > b-).
- **NT > ND**: regular in the NW (78–93% of NT words where the reflex is unambiguous); lower after a
  long vowel (57–70%: V̄NT > ṼT first), but chain order already captures that (a `// Long _` variant
  loses in 37/41 lects). Narrow copies merged into the main formulation in 11 lects (10 neutral).
  **ND > NN** is lexical outside Bengali-Assamese and Tharu (20–25% in L, khash): not unified.
- **Shared post-MIA prefixes** (`prefix.py`): per subfamily, the longest ordered sequence of shared
  post-MIA changes that each lect can hoist into one contiguous block with every prediction unchanged
  (greedy; a lect leaves the trunk when the next change interacts with its own rules). Prefixes are
  short (2–6 changes) because lects diverge in how compensatory lengthening, contraction and final-vowel
  loss interleave; the order-free late changes (v- > b-, ṇ > n, anusvāra loss/nasalisation) open many.
  Installed as `# ── <clade> post-MIA prefix … ──` blocks in 62 lects (BH, ktg reverted: interactions
  outside the 400-word sample). Trunks: Rajasthanic 4 changes × 8/14 lects; Eastern 4 × 5/10; Hindi
  3 × 5/9; Shinaic 6 × 3/6; W. Pahari 5 × 4/16; Kohistani, Migratory, C/E Pahari 3 × 3; Bihari only
  intervocalic m > ṽ (11/16); Kashmiric none.
- **aya > ē**: attested -aya- nouns show e in 30–67% in many lects (Dardic: Pas upalayana > palēn,
  Kal krayá > kre, Kho kṣayá > ʦ̣hek, Gowro/bhatr vayám > be; Nuristani tráyaḣ > tre, tánaya > Wg
  tanǽ; M, OM, G, OG, Md, W. Pahari), but chains predicted almost none: most had aya > e only in -ayati
  causatives (Pali-style) or a(C)a > e after consonant loss. A general early `a y a > ē` (after ai/au >
  ē/ō) now in 38 lects; in M, G, Kal, OAw, Tor, D, KharI only with `// _ t {i ē} #`, because the
  -aya- of -ayati verbs is dropped in NIA (ākalayati > M ākaḷṇẽ), the mirror image of Pali. Where it
  was rejected (H, P, S, Si, Md, K …) the effect is ±0.0005: those chains already reach e via y-loss and
  a+a contraction, or show -iyā (hr̥daya > hiyā). bhatr reverted (ext_tone).
- **`include`**: blocks live in `sound_changes/blocks/NAME.sc` and are pulled in with `include NAME`
  (README § Blocks). `blocks/mia-core.sc` replaces the 27 copied `use` lines in the 22 core lects.
- **Spreading one-offs** (`spread.py`): per clade, the 6 most active changes that only 1–2 lects have,
  tried in every sister after its copy of the home chain's preceding change (586 insertions, sample
  gate): 46 passed; the best per lect installed in 33 lects (Mth, S, kcch reverted on the full check).
  E.g. compensatory lengthening from mewari_bannoda → 4 Bhil lects; final aspirate devoicing from
  Noiri → 4; final-vowel loss from dog → cur, cam, kgr, pan; pretonic shortening from Bhoj → Bi, Dang;
  -ā/-ī > -ə from awan → L, poth. Many one-offs are a lect's own wording of a change its sisters
  have (final-vowel loss, compensatory lengthening): merging those wordings is the next step.
- **Visarga menus: tried and reverted.** Menus with nominative visarga forms (-akaḣ, -anakaḣ …) and
  early visarga deletion scored neutral; deriving the -o/-ā from the visarga needs it to survive MIA,
  which the chains' early final-consonant rules prevent. Reverted at the user's request (2026-09-30):
  bare-stem menus, no ḥ/ḣ merge; scores identical to before (0 differences over 151 lects). An
  unrestricted bare nominative gave roots a thematic -a and a spurious gain: a caution for menu items.
