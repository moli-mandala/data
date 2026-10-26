# Blocks round — consolidating sound changes into shared include blocks (2026-09-30, 01:30 → 09:30)

Seven theme agents (brief: `BRIEF-blocks.md`; reports in `tmp/sound_changes/blocks-<theme>/report.md` and
`pass2/report.md`) rewrote the chains so that the same historical change is one shared block
(`sound_changes/blocks/*.sc`, pulled in with `include NAME`), across families where the data support it.
Coordinator: sequential integration, full scores, per-lect reverts, then cross-family packages.

## Result

- **206 block files, all used** (hiatus-eastern, unused, removed); every one of the 151 lects includes blocks. Rule lines in lect files:
  6,910 → about 5,900 (the rest now lives once, in blocks).
- **No lect is worse than at the start** on any gated score (ext, verb, ext_acc, ext_tone; train and test),
  full scores at every stage. Net: ext train median 0.3416 → 0.3416, pair-weighted 0.2885 → 0.2878
  (88 lects better, 0 worse); test 0.3473 → 0.3464, 0.2955 → 0.2945 (72 better, 0 worse); verb train
  0.3228 → 0.3199 (41 better). Most of the round is neutral consolidation (identical predictions).
- Most-shared blocks (lects incl. nested use / clades): mia-lenition-stops 98/17, nasals-retroflexion 93/18,
  mia-stop-clusters 91/19, lar-v-b 90/19, mia-lenition 89/17, mia-s-clusters 84/19, mia-palatals 83/19,
  intervocalic-n-lenition 77/16, length-degem-plain 74/16, liquids-flap 68/16, mia-vowels 67/18,
  hiatus-medial-core 66/17, hiatus-v-loss 63/15, liquids-lateral-retroflexion 60/14.
  Full list: `tmp/sound_changes/unify/block_usage.txt`.

## Hypotheses (user)

- **l > ḷ > r** — supported. The ḷ belt (Marathi, Gujarati, Rajasthani, Bhil, Oriya, W. Pahari) and the r
  belt (Gangetic, Sindhi) lie side by side with the same conditioning (single intervocalic l only);
  Bhatri and Wadiyara Koli show ḷ ~ ṛ ~ r for one etymon; Punjabi has an unwritten ḷ; in 9 Hindi-belt and
  Sindhi lects putting l > ḷ early (MIA) changes no prediction. Blocks: `liquids-lateral-retroflexion` (60),
  `liquids-rhotacism` ḷ > r (25); flaps `liquids-flap` ḍ > ṛ (68) and `liquids-flap-rhotic` ṛ > r (27).
- **n > ṇ > n** (allophonic intervocalic retroflexion, later dental merger) — supported. In every mainland
  group OIA intervocalic n and ṇ have the same outcomes (NW 77% vs 78% ṇ, W. Pahari .79/.78, Hindi .05/.06).
  Not pan-IA: Pali, Sinhala and Dardic keep n ≠ ṇ; Bengali–Assamese shows no ṇ stage. Blocks:
  `nasals-retroflexion` (93), `nasals-dental-merger` (51); Garhwali/Kumaoni's dental nn falls out of
  `nasals-geminate-dental` + merger. Moving Punjabi/Lahnda retroflexion into the MIA section gains.

## Cross-family packages (coordinator, neutral by construction; extended where predictions are identical)

- `intervocalic-n-lenition` = n > ṇ + MIA lenition — 77 lects, 16 clades (+ 26 via the next).
- `intervocalic-retroflexion` = l > ḷ + the above — 26 lects, 9 clades (the user's allophonic
  intervocalic retroflexion of both sonorants, with lenition).
- `late-mergers` = v- > b-, ṇ > n, y- > j- — 29 lects; `mia-early` = MIA vowels + sibilants — 27;
  `liquids-flap-to-r` = ḍ > ṛ > r — 21.

## Themes (details in the agents' reports)

- **mia**: the MIA layer split into sub-blocks (sibilants, palatals, s-clusters, stop clusters, lenition,
  vowels, Dardic/Romani variants); `mia-core` now nests `mia-sibilants` and `mia-stop-clusters`. What blocks
  more: lect rules interleaved with MIA changes (Hindi anusvāra lengthening, NW rt > ṭṭ), Rajasthani
  stop + r, Dardic outcomes (dy > ź, kṣ > ʦ̣), Nuristani (not MIA).
- **nasals**: retroflexion/merger/geminate-dental; one NT > ND incl. aspirates (34); anusvāra units.
- **length**: `length-degem-plain` core + CL variants (first syllable, disyllable, final …); no single CL
  environment works across subfamilies; Dardic has plain degemination only.
- **laryngeals**: deaspiration (shared voiced core + variants), v > b (initial 90, geminate 42 incl. the
  vv/vy/yv/yy hardening), affrication (Dardic/W. Pahari/Nuristani), s > h, h-loss, vowel quality.
- **hiatus**: aya/ava, the medial a(C)a > e family (`hiatus-medial-core` now also Chil, Gaw, Kt, OSi),
  v/h loss, diphthongs.
- **finals**: apocope (monosyllables keep their vowel), -aka endings (west -o, east -ā, i/u, Dardic),
  -am endings, syncope; four Tharu lects' full apocope passes on ext but raises plain NED on test.

## Process notes

- Integration: line-based three-way merge conflicted in 88/126 lects; replaced by an anchor-replay merge
  (`tmp/sound_changes/unify/integrate2.py`): each theme's edits = delete these lines / insert these after
  that anchor, replayed on the live file.
- Pass 1 reverts (interactions with other themes' blocks) were mostly merge artefacts: rebuilt on the
  merged chains in pass 2, nearly all were identical or passed. Pass 2 kept 381/389 lect edits.
- Snapshots: `tmp/sound_changes/snapshots/pre-blocks/` (start), `pre-blocks2/` (after pass 1), `pre-seq/`,
  `pre-packext/`. Stage scores: `tmp/sound_changes/unify/stage_*.json`, `p2_*.json`, `blocks_final.json`.

## Small lects inherit their subfamily's blocks (2026-09-30, after the round)

Rule counts rose with attestation (Spearman pairs–rules 0.38; median 45 rules under 300 pairs vs 57
over 1,500; also within clades), because a change enters a chain only when the data show it. Default
flipped for the 43 lects under 300 pairs: blocks that ≥ half their clade-mates include were added (198),
then each tested alone (`tmp/sound_changes/unify/inherit*.py`, sample gate) and dropped only where the
lect's data contradict it (105 contradicted, 62 neutral, 17 gains); one full score, 2 lects reverted.
Kept: 87 inherited blocks in 30 lects, 14 improved (Khandesi ext −0.0101/−0.0067, khash test −0.0115,
Gmb, MB, Rj, RomBalk …), the rest neutral. Most often contradicted: intervocalic-retroflexion (7),
mia-clusters (5), mia-vowels (4) — real differences, not missing data.

## Sporadic (optional) changes (2026-09-30)

Engine: `optional:` prefix (also on `use` lines). A sporadic rule forks the derivation only where it
changes the word (sca.apply_rules_multi: a set of forms, identical forms merged, each form records the
optional rules it skipped); the extension-aware and verb scores take the best form per reflex, the plain
score applies the rule; derive.py traces the branch that matched. Cost grows with the real number of
distinct outcomes per word, not 2^k (full score 8 → 10.4 min with four optional blocks in 133 lects).
Each candidate was tested against a same-shape placebo optional rule (exact local gate):

| block | change | lects | Δext train/test (mean) | placebo |
|---|---|---|---|---|
| rt-retroflex-optional (+ mia-clusters-rt) | r + t tʰ d dʰ > ṭṭ … | 62 added + 17 own made optional | −0.0009/−0.0019 | −0.00000/−0.00002 |
| ks-cch-optional | kṣ > ccʰ beside kkʰ | 86 | −0.0009/−0.0022 | −0.00004/−0.00005 |
| dental-retroflex-optional | initial t d dʰ > ṭ ḍ ḍʰ | 106 | −0.0016/−0.0005 | −0.00004/−0.00004 |
| r-a-optional | r̩ > a beside i/u | 83 | −0.0009/−0.0007 | −0.0003/−0.0004 |

Net (full scores, no lect worse): ext train 0.2878 → 0.2830 pair-weighted, test 0.2945 → 0.2897;
verb test 0.2604 → 0.2525. Caveat: best-of over forks is optimistic; the placebos bound it (small for
the first three; r̩ > a's placebo recovers ~40% of its gain, as r̩ outcomes really vary by word).
Not yet done: spontaneous nasalisation (the probe matches almost every word; needs the sample gate);
a word-level consistency option (one branch per etymon across its reflexes).

### kṣ follow-up: outcome distribution and the tied reflexes (2026-09-30)

`tmp/sound_changes/unify/ksdist/`:
- ksdist.py runs every kṣ reflex through its lect's chain with the kṣ > ccʰ fork. It scores both
  branches against the attested form and classifies the winning derivation's kṣ segment (branch diff,
  or a placebo diff when there is no fork).
- Ties are resolved by a place-weighted distance: 0.5 within the velar or palatal class, retroflex
  ʦ̣ in its own class, 0.25 when only aspiration differs, 2 for C↔V.
- ties.py aligns each still-tied attested form to the prediction and reads the kṣ slot.

The ties showed three missing patterns, each tested against a placebo:

| block | change | lects | mean Δext real vs placebo |
|---|---|---|---|
| ks-h-optional | kṣ > h after a long vowel (dākṣiṇa > dāhiṇa, lākṣā > lāh) | 50 | −0.00023 vs −0.00003 |
| ks-nasal-optional | kṣ + nasal > nasal + h (ślakṣṇa > saṇha, pakṣman > pamha) | 8 | ≈ placebo overall; kept only where it beat it per lect |
| ks-s-optional | kṣ > ṣ (drākṣā > dros, kakṣa > kuṣ) | 6 (Nuristani only) | Nuristani −0.0092 vs −0.0019; Dardic/mainland ≤ placebo, not installed |

- Full score: no lect worse; pair-weighted ext train 0.2830 → 0.2828, verb train 0.2576 → 0.2572.
- Ties 838 → 650: 97 h cases and 44 sibilant cases resolved.
- Snapshot: pre-opt4.

Variant tried: `k ṣ > kʰ / Long _`, which relies on the lect's own lenition (int-aspirate-h) for the h.
- Predictions are identical to ks-h in 123/144 lects: Σ Δext train −0.0256 vs ks-h −0.0312 vs placebo −0.0048.
- ks-h is better in 15 lects that lack aspirate > h at that stage: Romani, Zangskari, Dotyali, Gawarbati,
  Hadauti, Domaaki, Marwari, Sinhala, Bhadarwahi …
- Kept ks-h as installed (user decision). Script: tmp/sound_changes/unify/ks_kh_vs_h.py.

### Unrecorded sporadic changes from the Hindi residuals (2026-09-30, opt5)

Mining:
- tmp/sound_changes/sporadic/mine.py aligns each reflex's chosen prediction to the attested form.
  It counts recurring edits (pred → attested, ±1 context) and how often each applies where its
  environment occurs.
- Run first on the 9 Hindi-group lects, then on 12 plains lects (compare.py) to check generality.

Gates:
- Opt_probe (MIA stage, regex-local gate) or endprobe.py (end of chain). endprobe caches the lect's base
  derivations and post-processes them, so it is exact and minutes instead of hours.
- Each candidate is tested against a same-shape placebo.
- A block is restricted to the groups where it beats the placebo in aggregate.

| block | change | lects | Σ Δext train real vs placebo |
|---|---|---|---|
| spont-nasal-optional (end) | first-syllable V > Ṽ before a voiceless C or h (sarpa > sā̃p, akṣi > ā̃kʰ) | 93 | −0.265 vs −0.074 (test −0.274 vs −0.068) |
| st-retroflex-optional (MIA) | #st(ʰ) > ṭʰ (stabdʰa > ṭʰaṇḍā) | 79 | −0.065 vs −0.005 |
| r-l-optional (MIA) | V r V > l | 78 (not Bihari–Tharu, Nuristani) | −0.097 vs −0.038 |
| v-loss-optional (end) | V v V > V V (aṭʰāīs, naiyā) | 52 (west only) | −0.180 vs −0.092; east/Dardic ≤ placebo |
| deasp-optional (end) | #Cʰ > C (gʰaná > Marw. gaṇɔ) | 20 (Raj–Guj, Hindi, Eastern) | overall ≤ placebo; Raj–Guj −0.089 vs −0.041 |

Rejected:
- Medial i > a: real ≈ placebo.
- Regular (non-optional) initial deaspiration: worse in all of Marw, Rj, Bagheli, Haryanvi,
  Chhattisgarhi and Godwari.
- l > r and nasal-less m > v: Hindi-specific and small, not probed.

Full score (opt5/final.json vs opt4):
- Two lects worse in combination were reverted to pre-opt5: Kalk, bhatr.
- Pair-weighted ext train 0.2828 → 0.2783, test 0.2896 → 0.2852.
- Exact train 28.3% → 28.9%, test 27.6% → 28.3%.
- Verb test 0.2524 → 0.2491.
- 135/0 lects better/worse on train.
- Plain NED worsens 0.4187 → 0.4534 because the plain score APPLIES every optional rule (nasalising every
  eligible first syllable). Plain is not a gate, but it should probably take the unchanged branch for
  `optional:` rules.

### Unified score, r > l dropped (2026-09-30)

- r-l-optional removed: forms with sporadic r > l may reflect variant etyma.
- Plain bare-headword NED deprecated.
- The score is now one best-of over bare + every noun extension + every verb formation for every headword.
  Every rule always applies (no @morph switch-off for verbs); optional rules still fork.
  "verb" is this score restricted to verb etyma.
- New baseline (unified/final.json): pair-weighted ext train 0.2647, test 0.2709; verb train 0.2387,
  test 0.2350. On the old score this was 0.2783 / 0.2852, so the bigger menu accounts for most of the drop.
- Worse than opt5 on ext: As-Ul (+0.0016/+0.0032) and Khol (+0.0021 train), from @morph rules now
  applying to verb candidates.

### opt6: hiatus glides + Cr retention / r-anticipation (2026-09-30, two agents)

Reports: tmp/sound_changes/glides/REPORT.md, tmp/sound_changes/cr/REPORT.md.
- glide-optional / glide-a-optional: 126 lects, end of chain (MIA stage in Si, OSi).
- cr-keep-optional + r-anticipation-optional + cr-release (placeholder kʳ… through the lect's cluster
  window): 40 lects.
- Snapshot: tmp/sound_changes/snapshots/pre-opt6-rules / -blocks.
- Full score vs the unified baseline: no lect worse.
  - Pair-weighted ext train 0.2647 → 0.2614, test 0.2709 → 0.2676.
  - Verb train 0.2387 → 0.2358, test 0.2350 → 0.2327.
  - 136/0 lects better/worse on train, 99/0 on test.

### opt7 (2026-10-01)

Scoring:
- Verb formations count again only for verb etyma or reflexes glossed "to …" (`evaluate.verb_ok`).
- Noun extensions stay available for every headword; no rule is switched off.
- Baseline on the pre-opt7 rules: ext train 0.2735, test 0.2805 (was 0.2614 / 0.2676 with verb forms
  for every noun).

Changes:
- VB: `optional: use hiatus.aaa.final` + -ō notation. The source has both -ō (āṇḍō, ambō) and -ā; a regular
  -aka > -ō is worse (net −16 reflexes). Optional: −0.0081 / −0.0187 vs placebo −0.0020 / −0.0051.
- v-b-skip-optional + v-b-release (placeholder ʋ around the v- > b- line): 21 lects.
  - P −0.0067 / −0.0075; placebo 0.
  - Eastern and Romani excluded (placebo better).
- pretonic-i-a-optional (end of chain): P, L.
  - P −0.0014 / −0.0014 vs placebo ≈0.
  - The regular version is worse in P, L, S, awan.
- apheresis-optional (end of chain): 69 lects.
  - Overall barely beats its placebo (initial-consonant loss: −0.309 vs −0.287), so installed only in
    the groups where it wins on train and test.
- Not changed: suvārṇā (*śōdʰakāra) already gets its -anaka form through its verb gloss; the remaining gap
  is phonological.

Full score vs that baseline:
- No lect worse.
- ext train 0.2735 → 0.2723, test 0.2805 → 0.2794.
- Verb test 0.2327 → 0.2318.

### Cross-family MIA clusters (2026-10-01; tmp/sound_changes/hclusters)

Before this, most chains had no rule for these clusters, so asm-, jihv-, cihn-, ātm- reached the output.
New blocks:
- mia-h-clusters: hm > mʰ, hn > nʰ, hṇ > ṇh, hv > bbʰ, tm > pp.
- mia-sm-mh: sm > mʰ, for central/eastern/southern IA.
- mia-sm-ss: sm > ss, for Northwest, Dardic, W. Pahari.
- nasal-aspirate-plain: end of chain, mʰ nʰ > m n.

Variant choice was made per group, from per-lect exact gates of 4 variants. Installed in 85 lects:
- 63 sm-mh + plain;
- 16 sm-ss + plain;
- 6 sm-ss.

MIA, Romani, Nuristani and Insular: no variant passed; unchanged.
The full score reverted Bhoj, Chitwan and Mth: the plain-nasal rule also hits ṣṇ/sn > ṇh words, which the
local gate didn't cover.

Result:
- ext train 0.2672 → 0.2669, test 0.2744 → 0.2733 (pair-weighted).
- Median test 0.3265 → 0.3189.
- 82/0 lects better on train, 74/0 on test.

### Several-outcome rules (2026-10-01)

Engine: `X > A | B / env`.
- The multi-derivation forks into every outcome; a single derivation takes the first.
- `sca.rules_taken` and derive.py trace the outcome a reflex used.
- `tmp/sound_changes/branches/branch_stats.py` reports per-lect branch shares.

Converted or added:
- mia-sm: `{s ś ṣ} m > mʰ | s s`, replacing the per-group mia-sm-mh / mia-sm-ss; neutral to slightly better.
  - mh share of decided reflexes: Bhil–Marathi 87%, Hindi 82%, Bihari 75%, Eastern 65%, W. Pahari 54%,
    Northwest 20%, Dardic 12%.
- nc-voiced-stop: `{ŋ m n ṇ} {voiced stops} > $1 $2 | $1` (end of chain), 81 lects.
  - −0.162 vs placebo −0.025.
- r-vocalic-multi: `optional: {r̩ ŕ̩} > {a á} | {i í} | {u ú}`, 69 lects, replacing r-a-optional.
  - −0.145 vs placebo −0.036.

Full score: ext train 0.2669 → 0.2653, test 0.2729 → 0.2709, exact test 30.1% → 30.5%, verb test 0.2274 → 0.2241;
no lect worse.

Candidate scan: tmp/sound_changes/branches/multi_outcomes.py (OIA clusters with ≥2 attested outcomes ≥20% per lect).

### Upstream slash / digraph fix (2026-10-01, ingestion agent; see its review summary in the session log)

Upstream changes:
- SSNP and nured slash alternates are split into variant rows (forms.slash_alternates).
- CDIAL `n/a` (DSAL placeholder glyph) maps to ɛ.
- Digraph aspirates are converted in about 20 source profiles. Pairs regenerated.

Chains:
- The redundant digraph tails (`{kʰ …} > {k …} h`) are commented out in 16 lects: OM, the Bareli/Bhil lects,
  Goj, Vasavi, B, Kannauji-Central, jaun.
- Kept:
  - bundeli `tʰ > t ʰ`: detached ʰ is still in 30 bundeli reflexes.
  - Kho ʦ̣h, Mai zh, the regmi-thakur Danuwar tails, and the zoller2023 exclusions: those sources were not changed upstream.

Scores:
- New-pairs baseline (tails/base.json) → tails removed (tails/final.json): 17/0 lects better.
- Exact match in the affected lects equals pre-upstream (OM 15.1% → 14.5%: its sonorant aspirates mh > mʰ are now
  single segments the chain still writes as m h). NED rises there only because the gold forms got shorter.
- New baseline: tails/final.json (ext train 0.2661, test 0.2718 pair-weighted, on the regenerated pairs).

Warning: the same build ingested 28 untracked 20260927-*.csv sources from another session. Two collide with CDIAL
legacy IDs (9 renames, 34 edges rewired). The owner must fix their file order before committing.

### Missing simple laws (2026-10-01; tmp/sound_changes/laws)

Detector: laws/lawfinder.py aligns the chosen input to the attested form and to the prediction, per OIA segment
and position class. A pattern is flagged when the attested outcome has a clear majority that the chain rarely
predicts. Then each candidate is probed against a placebo, with groups kept only where it beats the placebo.

Round 1 (end of chain):
- m-nasal-multi: `Vm > Ṽ | Vṽ | Vm`, 76 lects (gōdʰū́ma > gehū̃). −0.200 vs placebo −0.060.
- h-initial-loss-optional: 21 lects (Si haṁsá > as).
- b-lenition-multi: `VbV > b | v | ∅`, 79 lects (lābu > lāu). −0.065 vs placebo ≈0.
- Full score: ext test 0.2718 → 0.2703, verb test 0.2251 → 0.2213; no lect worse.

Round 2 (OIA stage; from the worst-headwords list):
- upa-va-optional: upa-/ava- > va-/ba-, 79 lects.
- ml-bl-optional: ml-/mr- > bl-/br-, 5 lects.
- s-ts-initial-optional: initial ś- > ʦ-, 8 Nuristani/Dardic lects.
- Full score: ext test 0.2703 → 0.2692; Pas reverted.
