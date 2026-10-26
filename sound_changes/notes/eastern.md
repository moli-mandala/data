# Eastern group: OIA → Odia, Bengali, Assamese, Kamta–Rajbanshi–Surjapuri, Desia, Hajong, Bishnupriya, OB, MB

Rule files: `sound_changes/rules/OIA-{Or,B,A,BH,MH,KS,AdivasiOriya,Hajong,MB,OB,Bishnupriya}.sc`.
Frontiers: `sound_changes/results/OIA-<lect>.hand.frontier.csv`. Scratch tools and logs:
`tmp/sound_changes/eastern/` (`try.py` variant tester, `chron.py` order-reversal test, `feat.py`
feature matrix → `feature-matrix.tsv`, `chron-log.txt`, `peek.py`/`conf.py` error browsers,
`table.py` for the table below).

## 1. Results

NED = mean normalised edit distance, null (etymon unchanged) → final chain. "Auto" = the greedy
baseline's test NED at its last (largest) rule count, next to the hand chain's test NED at the same
count on the hand frontier. Caveats: the auto baseline scored *all* sources (for B it includes the
kim survey lists, which my file scores through `%source` notation blocks; for Or/AdivasiOriya it
includes beine2017bhatri, which I excluded), so the numbers are on slightly different pair sets.
Rule counts include `@acc-loss`, `@notation` and `@morph` lines of the main chain (not `%source`
blocks). The hand chain beats the auto baseline on test at the auto's own rule count for every lect
except KS: at 14 rules auto has 0.446 against hand 0.462, on 86 test pairs. On train the hand chain is
better there too (0.434 vs 0.451). Backward elimination is greedy, and the auto curve was optimised
forward.

<!-- TABLE -->
| lect | #etyma | #rules | train NED | test NED | test exact % | auto test NED @n (hand @n) | excluded / per-source |
|---|---|---|---|---|---|---|---|
| Or | 3850 | 41 | 0.588 → 0.318 | 0.602 → 0.322 | 20.6 | 0.393 @24 (0.343) | beine2017bhatri |
| B | 3196 | 43 | 0.616 → 0.330 | 0.622 → 0.336 | 23.2 | 0.467 @25 (0.365) | zoller2023; %source: kim-kim-sangma2012garo, kim-kim-sangma-ahmad2011tripura |
| A | 2653 | 46 | 0.648 → 0.343 | 0.668 → 0.353 | 22.3 | 0.471 @24 (0.392) | — |
| BH | 524 | 43 | 0.712 → 0.461 | 0.745 → 0.464 | 9.3 | 0.593 @13 (0.563) | — |
| MH | 459 | 40 | 0.630 → 0.420 | 0.643 → 0.406 | 14.9 | 0.495 @19 (0.443) | — |
| KS | 391 | 44 | 0.618 → 0.374 | 0.620 → 0.369 | 22.1 | 0.446 @14 (0.462) | — |
| AdivasiOriya | 203 | 40 | 0.666 → 0.443 | 0.696 → 0.398 | 12.3 | 0.548 @25 (0.434) | beine2017bhatri |
| Hajong | 195 | 44 | 0.644 → 0.358 | 0.631 → 0.330 | 20.5 | 0.494 @21 (0.394) | — |
| MB | 155 | 35 | 0.583 → 0.331 | 0.567 → 0.408 | 13.9 | 0.450 @22 (0.426) | — |
| OB | 131 | 33 | 0.566 → 0.333 | 0.607 → 0.364 | 15.6 | 0.484 @19 (0.411) | — |
| Bishnupriya | 129 | 44 | 0.667 → 0.331 | 0.636 → 0.247 | 38.2 | 0.479 @14 (0.374) | — |
<!-- /TABLE -->

Excluded sources / per-source notation:
- **B: kim-kim-sangma2012garo, kim-kim-sangma-ahmad2011tripura** — these *are* Bengali (colloquial
  Standard Bengali elicited as the contact language in the Garo and Tripura sociolinguistic surveys:
  `śurjo`, `bɔchor`, `ʃona`), so they belong, but they are phonetic while Turner's B. is an
  orthographic transliteration. Handled with `%source` blocks: digraph aspirates (`dh`, `tʃh`), no
  vowel length, ś (garo) or IPA ʃ tʃ dʒ ɾ ɽ ʈ ɖ ɸ (tripura), no retroflex letters (garo). About 7
  forms have dropped letters (`br śti`, `d n` for bɾɪʃʈi, dɪn: a lost ɪ glyph) and stay as noise.
- **B: zoller2023** (3 pairs: `hóũk`, `girgiṭī`) — Bangani (West Pahari) forms mislinked to B.
  (apparently a language-code collision on "B"); excluded.
- **Or and AdivasiOriya: beine2017bhatri** (117 / 132 pairs) — the Bhatri survey list. In Or it
  writes ə/ɔ, syllable dots (`no.kʰo`) and r as ṛ; in both it is phonologically a *different lect*
  (deaspiration `dud`, `pul`; c j > s z `sani`, `zib`; not notation), so a `%source` block would
  have to contain sound changes. Excluded; Bhatri deserves its own file (it would be a nice test of
  the Odia–Halbi link; its c > s, j > z parallels Assamese).
- AdivasiOriya (behera2021desia) writes plain i u as ī ū, ɔ for OIA a and a for OIA ā. Single source
  after the Bhatri exclusion, so these are ordinary tail `@notation`/`@a-rounding` rules.
- Bishnupriya (kim-kim2008bishnupriya) writes /i/ as ī: tail `@notation i > ī`.
- The toulmin lects (BH, MH, KS) are phonetic and noisy (phrases, suffixed forms `-ʸæ`, free
  variation ʦ/c/s, ʣ/z/j, ɸ, β, ɣ lenition allophones). BH writes the open vowel /a/ as `ā` and OIA a
  as ɔ; MH and KS write both a and ā for OIA a. None of this could be fixed by a clean notation rule,
  so I kept them whole.

## 2. Base chain (group order) and deviations

All eleven files share one order (one exception: in Or, v-loss follows the a-contraction, §3); the
group template is OIA-B.sc. Codes: Or, Dv = AdivasiOriya
(Desia), OB, MB, B, A, BH (Kamtapuri), MH (Rajbanshi), KS (Surjapuri), Hj = Hajong, Bp = Bishnupriya.

MIA block:
1. `V > $1-acc` @acc-loss
2. `{ai au} > {ē ō}` @ai-au, plus `a {i u} > {ē ō}@2` for accented diphthongs (see §6)
3. `r̩ > i` @r-vocalic (r̩ > u after labials / r̩ > a tested: no gain anywhere)
4. `Sib > ś` @sibilant-ś — the Magadhan merger. Only A (x/h), Bp and Hj (h) and the survey/toulmin
   ś letters show *which* sibilant it merged into; in Turner's B./Or. it is invisible (all `s`).
5. `{t tʰ d dʰ} y > {c c j j} {c cʰ j jʰ}@1` @palatal-Cy
6. `{k t p} ś > {k c c} {kʰ cʰ cʰ}@1` @ks (kṣ > kkʰ; kṣ > ccʰ tested: worse in every lect, +0.003)
7. `ś Stop > $2-asp $2+asp` @s-stop-aspirate; `ś Son > ś ś` @assim-sonorant
8. `Nasal {voiceless stops, j jʰ ḍ ḍʰ, d dʰ} > ṁ $2` @nasal-cluster — **three regional types**:
   - B-type (B, BH, MH, KS, Bp): anusvāra before everything except g gʰ b bʰ, which assimilate:
     `{ŋ m} {g gʰ b bʰ} > $1 $1` @x-eastern-nasal-assim (aŋga > āŋ, kumbʰīra > kumīr).
   - A-type (A, Hj): anusvāra only before voiceless stops, j, ḍ; **ŋg, nd kept** (aŋgāra > āŋgār,
     krandati > kāndurā); only `m b > m m` (kambala > kāmali). A also lengthens a before a kept NC
     (`a > ā / # (C) _ Nasal Stop`).
   - Or-type (Or, Dv, OB, MB): no anusvāra rule at all; the nasal consonant survives (gāṇṭʰibā,
     kāndibā, OB sāndʰi, MB āntare). Dv, OB and MB lengthen a before it (`a > ā / _ Nasal Stop`).
9. `l Stop > $2-asp $2`, `{r ḣ} C > $2-asp $2` @assim-r; `Stop {r y v l Nasal} > $1-asp $1`,
   `l {y v} > l l`, `Nasal {y v r l} > $1 $1` @assim-sonorant; `Stop Stop > $2-asp $2` @assim-stop
   (BH lacks `l Stop`: toulmin ɔlpo)
10. `C {r y v l} > $1 / # _`, `C C > $2 / # _` @initial-degem
11. `ā > a / _ #` @x-eastern-final-a-short — all except BH, MH, Dv (LOO ≤ 0 there)
12. `a y a > ē // _ t {i ē} #` @aya-e (outside the -ayati verb ending): Or, A, BH, MH, KS
    (−0.0002 to −0.0012). Not B (+0.0003), and not OB or MB (+0.003: their spelling keeps -aya-).
    `{kʰ gʰ tʰ dʰ pʰ bʰ} > h / V _ V` @int-aspirate-h (not BH, Hj: LOO −0.001, −0.0035; their
    phonetic sources keep many medial aspirates); `{ṭ ṭʰ} > {ḍ ḍʰ} / V _ V` @int-retroflex;
    `p > v / V _ V` @int-p-v; `{k g c j t d y} > ∅ / V _ V` @int-stop-loss
13. `C > ∅ / V (C) (C) V _ #` @final-cons-loss (polysyllables only: CDIAL's root etyma `*cul`,
    `*pʰaṭ` are bare citation stems; the unconditioned rule had negative LOO in B, A and the Kamta
    lects because it ate their only consonant)

NIA block:
14. @morph citation forms (Or first adds `∅ > ā / C _ #`: Odia allows no final consonant, so bare
    root and consonant-stem etyma take the -ā extension, −0.0048): B-type `(V) a {i ē} > ā` (verbal noun -ā: cālayati > cālā; also BH, KS,
    Hj); A `> iba`; Or `> ibā`; MH and Bp none (toulmin cites bare stems, the Bp survey -anī nouns;
    the rules had LOO ≤ 0); OB/MB none (3sg -ai; MB adds `a i > e / _ #`, 3sg -e).
15. `v > ∅ / V _ V` @v-loss — all except A (no signal either way, ±0.0005; A. keeps many w: zovār) and Dv (LOO < 0).
    Before it, `a v a > ō` @aya-e (návanīta > nɔnī) in A, B, BH, MH, Dv, Hj, Bp, OB (−0.0004 to
    −0.009); not Or, KS, MB (+0.0005 to +0.003).
16. `{a ā} {a ā} > ā` @hiatus — all except OB, MB (see §3). Or adds `a > ∅ / V _`.
17. Or only: `n > ṇ / V _ V` @n-retroflex, `l > ḷ / V _ V` @l-retroflex — *before* degemination.
18. `V G $2 > $1+long $3` @degem-lengthen — B-type everywhere; A first syllable only; Or first syllable
    and not before a long vowel (kappāla > kapāḷa, karmāra > kamāra); Dv not before a long vowel.
19. `G $1 > $2` @degem-plain
20. `V ṁ > $1+long+nas / _ C` @anusvara-vowel (toulmin and kim lects write no nasality: `> $1+long`;
    Or `> $1+nas`, LOO −0.0008, kept because without it ṁ would surface; the loss comes from
    numerals like -viṁśati > -isa). Not in OB, which spells the anusvāra (aṁbāḍa, kāṁvarū).
21. `{ḍ ḍʰ} > {ṛ ṛʰ} / V _` @flap — **ṛ in B, Or, MB; r in A, BH, MH, KS, Hj, Bp**. No flap in Dv
    (Desia haḍ, gɔḍ) or OB (Caryāpada spelling koḍi, niaḍi: LOO −0.002).
22. `a > ∅ / _ #` @final-vowel-loss — **only -a**: MIA -i -u survive in B, A, BH, MH, Hj, Bp, Dv
    (B `a`-only vs `{a i u}`: −0.0028; A −0.0078; BH −0.0036; MH −0.0083). KS loses all three
    (weakly: +0.0008 for a-only). Or, OB, MB: no apocope.
23. `{ī ū ī̃ ū̃} > {i u ĩ ũ}` @x-eastern-high-length-loss — every lect.
24. `ṇ > n` @n-dental — B, A, Kamta lects, Hj, Bp, Dv; not Or, OB, MB.
25. `v > b / # _` @v-b, generalised to `v > b` everywhere in Or, B, Hj, Bp, OB, Dv (−0.0007 to
    −0.0066; sarva > sab). A resists it (+0.0024): it keeps w. `y > j / # _` @y-j
26. `{ē ō ē̃ ō̃} > {e o ẽ õ}` @notation
27. Sibilant spell-out: B, OB, MB `ś > s` @notation (Turner writes all B. sibilants s); Or, Dv, MH, KS
    `ś > s` @x-eastern-s-dental; A `ś > h / V _` @s-h then `ś > x` @x-eastern-s-x; Hj, Bp
    `ś > h / # _` @s-h (then Hj ś > s, Bp keeps ś); BH keeps ś (toulmin writes ś).

Lect-specific tails:
- A: `{c cʰ j jʰ} > {s s z z}` @x-eastern-deaffrication (LOO +0.029).
- B `ṛʰ > ṛ`, A `rʰ > r` @deaspiration of the flapped aspirate (−0.0014, −0.0012). Or keeps ṛʰ
  (adding it: +0.0012), which is another small Odia vs Bengali–Assamese isogloss.
- BH: `a > o` @a-rounding (global; toulmin ɔ), `{c cʰ j jʰ} > {s s z z}`, `l > n / # _`
  @x-eastern-l-n, `Stop > $1-asp // # _` @deaspiration.
- KS: `Stop > $1-asp / _ #` @deaspiration; `ā > a / _ C (C) V C` @x-eastern-pretonic-shortening
  (−0.0040; the same rule costs MH +0.0036, B +0.0095, BH +0.0153). This is a Bihari-type shortening:
  Surjapuri is the Bengali–Maithili transition lect.
- BH, MH, KS: late `ā > ∅ / i _` @hiatus (iā > i: *bʰiyantara > bʰitar, rājikā > rāi). Adding it to
  B, A, Or, OB, Hj or Bp gives ≤ 0, so it is a Kamta-group isogloss.
- Hj, Bp: `{ā ā̃} > {a ã}` @x-eastern-length-loss; `a > o / _ C a #` @a-rounding placed *before*
  apocope; `Stop > $1-asp / _ #` @deaspiration; `{ṭ ṭʰ (ḍ ḍʰ)} > {t tʰ (d dʰ)}`
  @x-eastern-deretroflex.
- Dv: `a > o` @a-rounding, `{ā ā̃} > {a ã}` @x-eastern-length-loss, `ṇ > n`, `h > ∅ / V _ V`
  @h-loss, `Stop > $1-asp // # _` @deaspiration, notation ī ū; no flap (Desia keeps ḍ: haḍ, gɔḍ);
  no v-loss but `v > b` everywhere (kavāṭa > kabat, −0.0066).

## 3. Relative chronology (order reversals, ΔNED on train; + = the reversed order is worse)

Measured with `chron.py`, which moves rule A just past rule B and re-scores.
- **i/u length merger after compensatory lengthening.** Moving `{ī ū} > {i u}` before
  `V G $2 > $1+long $3`: B +0.031, A +0.023, Or +0.017, OB +0.018, MB +0.017, BH +0.029, MH +0.029,
  KS +0.034, Hj +0.023, Bp +0.019 (Dv 0: its notation rule re-lengthens). The long ī ū created by
  degemination (tiktá > titta > *tīt > B. tit(ā)) are shortened too, so the merger is later than MIA
  degemination in every eastern lect. This is the most robust ordering result in the group.
- **Final -ā > -a before hiatus contraction (and so before -aka > -aa > -ā).** Moving `ā > a / _ #`
  after `{a ā} {a ā} > ā`: B +0.029, Or +0.025, Hj +0.020, KS +0.013, A +0.005, Bp +0.005. OIA/MIA
  final -ā was shortened *before* the -aa (< -aka) sequences contracted to a new -ā, which then
  survived (B -ā, Or -ā, A -ā nouns). This is an Apabhraṁśa-style final shortening shared by the
  group. OB and MB have it too, but they have no contraction at all.
- **Flapping after degemination (B, A).** Moving the flap before degemination: B +0.0066, A +0.0045,
  Or +0.0013, Bp +0.0015. Geminate ḍḍ also ends as ṛ/r. MH, KS and BH show −0.002 to −0.003 for the
  reverse order, but only because the reversed rule flaps the first half of ḍḍ (`gorḍ`), which is a
  spurious fit. With a guard (`// _ {ḍ ḍʰ}`) the early order is no better than the late one
  (MH +0.0003, KS +0.004 at the time of testing).
- **Nasal vowel before flapping (B, A).** Moving `V ṁ > Ṽ` after the flap: B +0.0042, A +0.0027 (BH
  +0.0005, others 0). The ḍ of ṇḍ flaps only once the nasal has become vowel nasality
  (daṇḍa > dā̃ḍ > ḍā̃ṛ).
- **Or retroflex ḷ before degemination.** Moving `l > ḷ / V _ V` after degemination: +0.0040
  (`n > ṇ` +0.0006). Only singleton intervocalic l becomes ḷ; MIA ll > l stays l
  (śāla > sāḷa vs *ḍalla > ḍāli). In Or, v-loss also *follows* the a-contraction (reverse +0.0011;
  in B the opposite, −0.0006, i.e. no signal).
- **A: debuccalisation of ś before deaffrication of c.** If c j first merge with ś, z (so that c feeds
  x/h): +0.017. Dropping c > s altogether: +0.029. Order: ś > x/h, then c > s. So Assamese s < c is
  younger than x/h < ś, and old s never came back into the x/h series.
- **Hj, Bp: a > ɔ before apocope.** Moving `a > o / _ C a #` after apocope: Bp +0.017, Hj +0.012. The
  rounding is conditioned by a following (later lost) -a: vānara > banor, vatsara > bosor, pʰala > pʰɔl.
- **Hj, Bp, KS: final deaspiration after apocope.** Moving it before apocope: Hj +0.013, Bp +0.012,
  KS +0.003. Newly final aspirates (mēgʰa > mek, múkʰa > muk) deaspirate.
- **BH: a > ɔ after compensatory lengthening** (+0.058 reversed): lengthened ā (bʰāt) escapes rounding.
- Trivial feeding orders, which confirm the machinery rather than tell us anything:
  `Sib > ś` before kṣ (+0.006); intervocalic ṭ > ḍ before flap (+0.014–0.018); stop loss before
  hiatus contraction (+0.036 B, +0.051 Or).
- **Local search.** Moving every rule to every other position (`moves.py`, B full, A and Or within
  ±10) found no move worth more than 0.0011, so the chain orders are locally optimal.
- **OB → MB → B.** Toggling features (`feat.py`, ΔNED×100) gives:

  | change | OB | MB | B |
  |---|---|---|---|
  | apocope of -a | −8.11 (absent) | −7.74 (absent) | +4.62 (present) |
  | hiatus contraction | −1.97 (absent) | −0.17 (absent; its LOO was −0.0017, removed) | +0.49 (present) |
  | ṇ > n | −0.66 (absent) | +0.13 (neutral) | +0.53 (present) |
  | NC > Ṽ | −2.33 (absent) | −0.11 (neutral) | +2.59 (present) |
  | ī ū > i u | +1.31 | +1.77 | +4.29 |
  | ḍ > ṛ flap | −0.21 (absent: koḍi) | +0.25 | +2.04 |
  | anusvāra ṁ > Ṽ | −0.20 (absent: aṁbāḍa) | +0.59 | +2.83 |

  So the Bengali-internal order is: the i/u length merger is already in OB (the Caryāpada spelling
  confuses i/ī). Flapping and anusvāra > nasal vowel come in on the OB → MB boundary. NC > Ṽ,
  ṇ > n, hiatus contraction and apocope of -a come after MB.
  Caveat: OB and MB are cited in orthographic form (inherent vowel written, ṇ written as in the
  manuscripts), so "after MB" means "after the spelling of the MB texts". Apocope in speech may be
  older.

## 4. Shared vs lect-specific changes; sub-grouping

Feature matrix (`tmp/sound_changes/eastern/feature-matrix.tsv`): ΔNED×100 from toggling each change
in each file. `*` marks a present rule that was removed (positive = removal hurts); unstarred values
are for adding the change where it is absent (positive = addition helps). So **positive = the data
support the change**. `·` = not applicable.

<!-- MATRIX -->
| feature | Or | Dv | OB | MB | B | A | BH | MH | KS | Hj | Bp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| apocope -a | -4.80 | +4.08* | -8.11 | -7.74 | +4.62* | +4.15* | +3.02* | +3.15* | +6.04* | +4.41* | +6.47* |
| -ā > -a | +0.40* | · | +1.40* | +1.16* | +0.58* | +0.32* | · | · | +0.52* | +0.62* | +0.87* |
| ḍ > r (not ṛ) | -1.19 | · | · | +0.01 | -2.12 | +1.53* | +0.48* | +0.30* | +1.04* | +0.67* | +1.24* |
| ṇ > n | -1.72 | +0.24* | -0.66 | +0.13 | +0.53* | +0.52* | +0.94* | +0.92* | +1.36* | +1.32* | +1.39* |
| c j > s z | -3.02 | -2.29 | -2.60 | -2.84 | -4.44 | +2.89* | +0.98* | -3.93 | -4.29 | -1.82 | -4.32 |
| initial ś > x/h | -1.33 | -1.69 | -2.10 | -1.35 | -1.37 | +1.41* | -0.91 | -1.18 | -1.50 | +0.34* | +0.41* |
| a > ɔ | -9.64 | +2.15* | -13.82 | -12.38 | -2.96 | -5.22 | +3.01* | -2.57 | -4.11 | +1.19* | +1.74* |
| ā = a (no length) | -5.68 | +6.55* | -7.16 | -5.95 | +0.32* | -6.04 | -9.81 | -6.80 | -5.15 | +12.12* | +12.05* |
| ī ū > i u | +2.50* | +0.00* | +1.31* | +1.77* | +4.29* | +3.07* | +4.65* | +4.31* | +5.39* | +3.86* | +3.33* |
| ŋg mb assim | -0.24 | -0.28 | -0.62 | +0.14 | +0.21* | -0.03 | +0.32* | +0.16* | +0.36* | +0.03 | +0.67* |
| NC > Ṽ (any) | -3.46 | -2.18 | -2.33 | -0.11 | +2.59* | +1.50* | +0.17* | +0.45* | +0.50* | +0.32* | +0.89* |
| l > n initial | -0.26 | -0.76 | -0.29 | -0.18 | -0.17 | -0.52 | +0.24* | -0.23 | -0.11 | -0.17 | -0.18 |
| final deasp | 0.00 | +0.37* | 0.00 | 0.00 | -0.09 | -0.47 | +0.78* | -0.37 | +0.29* | +1.35* | +1.21* |
| deretroflex | -2.09 | -1.08 | -1.68 | -0.74 | -2.07 | -2.03 | -0.76 | -1.56 | -1.68 | +0.71* | +1.17* |
| l > ḷ (Or) | +1.53* | -1.38 | -1.83 | -1.63 | -2.15 | -2.31 | -2.51 | -2.00 | -2.60 | -2.59 | -2.05 |
| hiatus a a > ā | +0.17* | +0.60* | -1.97 | -0.17 | +0.49* | +0.32* | +0.18* | +0.71* | +0.89* | +0.07* | +0.00* |
| late iā > i | -0.06 | 0.00 | -0.35 | +0.11 | 0.00 | 0.00 | +0.32* | +0.40* | +0.33* | 0.00 | 0.00 |
| initial h loss | -0.35 | -0.12 | -0.22 | -0.31 | -0.54 | -0.42 | -0.31 | -0.34 | -0.33 | -2.01 | -0.15 |
| initial n > l | -0.61 | -1.44 | -0.65 | -1.26 | -0.68 | -0.64 | -1.00 | -0.66 | -0.66 | -1.05 | -1.21 |
| medial ś > h | -1.55 | -1.28 | -0.75 | -0.94 | -1.66 | +1.21* | -0.96 | -1.38 | -1.13 | -1.14 | -0.70 |
| voiced asp > deasp | -1.70 | -0.00 | -2.47 | -1.12 | -2.20 | -1.89 | -1.31 | -1.94 | -2.41 | -1.20 | -0.71 |
| all asp > deasp | -4.15 | -0.13 | -5.11 | -3.09 | -4.65 | -4.12 | -1.98 | -3.99 | -4.08 | -2.17 | -2.14 |
| intervoc. h loss | -0.46 | 0.00 | -2.12 | -1.34 | -0.13 | -0.43 | -0.04 | -0.71 | -0.26 | -0.13 | +0.18 |
| aya > e | +0.06* | 0.00 | -0.35 | -0.25 | -0.03 | +0.02* | +0.10* | +0.12* | +0.08* | 0.00 | 0.00 |
| ava > o | -0.12 | +0.96* | +0.35* | -0.32 | +0.10* | +0.22* | +0.17* | +0.04* | -0.05 | +0.58* | +0.53* |
| flap deasp ṛʰ>ṛ | -0.12 | 0.00 | 0.00 | 0.00 | +0.14* | +0.12* | +0.02 | 0.00 | 0.00 | 0.00 | 0.00 |
| v > b everywhere | +0.10* | +0.66* | +0.11* | 0.00 | +0.14* | -0.24 | -0.03 | -0.00 | 0.00 | +0.07* | +0.18* |
| pretonic ā > a | -0.72 | 0.00 | -1.26 | -1.20 | -0.95 | -1.17 | -1.53 | -0.36 | +0.40* | 0.00 | 0.00 |
| v-loss | +1.10* | -0.20 | +0.83* | +1.27* | +0.53* | +0.06 | +0.23* | +0.44* | +0.18* | +0.42* | +0.39* |
<!-- /MATRIX -->

Reading it:
- **Shared by the whole group:** the ś merger (visible only indirectly), -ā > -a, i/u length merger,
  intervocalic v loss (except A), hiatus contraction (except OB, MB).
- **Odia vs Bengali–Assamese.** Or (and OB, MB) keep final -a, keep NC clusters, keep ṇ, and have
  ḷ (Or only). B, A, the Kamta lects, Hj and Bp all share apocope of -a (but not of -i -u), NC > Ṽ (in
  some form), and ṇ > n. **But** OB and MB, the ancestors of B, still have -a, ṇ and NC in their
  spelling, so if the orthographic caveat is set aside, apocope and ṇ > n in B are post-MB. A's
  identical changes would then be parallel developments or areal spread, not a Bengali–Assamese
  node. The NC typology also cuts across: A (ŋg, nd kept) patterns with Hj, B with the Kamta lects.
  Desia (Dv) groups with Odia (no NC > Ṽ, lengthening before NC like OB/MB) but has apocope and
  ṇ > n like the northern lects. That is probably a southern-Odia innovation, or it comes from
  Desia's tribal-lingua-franca status.
- **Smaller isoglosses (under 0.002 each):** deaspiration of the flapped aspirate (B ṛʰ > ṛ,
  A rʰ > r; Or keeps ṛʰ). Generalised v > b (Or, B, OB, Hj, Bp, Dv), which A alone resists (+0.24
  against). aya > e in Or, A and the Kamta lects but not in B, OB or MB. ava > o nearly everywhere
  except Or, KS and MB.
- **Kamarupa grouping (A, Kamta–Rajbanshi, Hajong, Bishnupriya).** The clearest partition in the
  matrix is **ḍ > r (not ṛ)**: positive in A, BH, MH, KS, Hj and Bp; negative in B, Or and OB. This is
  the classic Kamarupi isogloss. Within it:
  - A alone has the full package: ś > x/h (+1.41), c j > s z (+2.89), ŋg nd kept.
  - BH (Kamtapuri) shares c j > s z with A (+0.97) but not ś > x/h (toulmin writes ś, −0.91). It has
    B-type NC and ɔ-rounding. So it is intermediate, as expected geographically.
  - MH (Rajbanshi) and KS (Surjapuri) have ḍ > r but neither c > s (−3.9, −4.3) nor x/h, and B-type
    NC. KS behaves as a Bengali–Bihari transition lect: it also loses -i -u, deaspirates finally
    and has Bihari-type pretonic ā-shortening (−0.004, which no other eastern lect accepts).
  - BH, MH and KS share a late **iā > i** contraction (+0.30, +0.44, +0.33). No other lect has it,
    so it is the one innovation that defines a Kamta–Rajbanshi–Surjapuri subgroup in these data.
  - Hj and Bp share initial ś > h (Bp: saptá > hat, śilā > hīl), complete loss of vowel length,
    pre-apocope a > ɔ, final deaspiration and loss of retroflexion (Hj +0.71, Bp +1.17). That is a
    Tibeto-Burman-contact profile, but it is not the Assamese c > s (−1.8, −4.3). Hj has A-type NC,
    Bp B-type.

**Cross-application** (`cross.py`): each lect's final chain scored on every other lect's train
pairs (train NED, per-source blocks off). Read down a column: the best *other* chain is the nearest
relative, with notation confounds. The kim and Desia lects write no vowel length, so their chains fit
one another first; toulmin's BH writes ā for /a/.

<!-- CROSS -->
| chain ↓ / data → | Or | Dv | OB | MB | B | A | BH | MH | KS | Hj | Bp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Or | 0.318 | 0.646 | 0.414 | 0.393 | 0.461 | 0.511 | 0.634 | 0.519 | 0.508 | 0.579 | 0.617 |
| Dv | 0.636 | 0.443 | 0.626 | 0.623 | 0.592 | 0.649 | 0.677 | 0.615 | 0.573 | 0.474 | 0.452 |
| OB | 0.426 | 0.609 | 0.333 | 0.361 | 0.471 | 0.523 | 0.608 | 0.493 | 0.479 | 0.553 | 0.590 |
| MB | 0.418 | 0.633 | 0.350 | 0.331 | 0.454 | 0.527 | 0.610 | 0.494 | 0.478 | 0.568 | 0.604 |
| B | 0.469 | 0.633 | 0.480 | 0.408 | 0.346 | 0.473 | 0.550 | 0.437 | 0.397 | 0.533 | 0.555 |
| A | 0.519 | 0.674 | 0.532 | 0.490 | 0.476 | 0.343 | 0.539 | 0.500 | 0.457 | 0.567 | 0.586 |
| BH | 0.578 | 0.665 | 0.589 | 0.516 | 0.487 | 0.486 | 0.461 | 0.519 | 0.479 | 0.526 | 0.533 |
| MH | 0.483 | 0.631 | 0.452 | 0.414 | 0.406 | 0.461 | 0.532 | 0.420 | 0.385 | 0.522 | 0.542 |
| KS | 0.485 | 0.628 | 0.517 | 0.441 | 0.394 | 0.486 | 0.550 | 0.438 | 0.374 | 0.504 | 0.523 |
| Hj | 0.565 | 0.555 | 0.562 | 0.534 | 0.506 | 0.538 | 0.609 | 0.530 | 0.480 | 0.358 | 0.383 |
| Bp | 0.618 | 0.546 | 0.600 | 0.567 | 0.565 | 0.592 | 0.645 | 0.581 | 0.530 | 0.418 | 0.331 |
<!-- /CROSS -->

- Or data: MB and OB chains fit best (0.418, 0.426), then B (0.461). The shared feature is
  unapocopated -a.
- **A data: the Rajbanshi (MH) chain fits better than the Bengali one** (0.461 vs 0.473; BH and KS
  0.486). B data: KS fits best (0.394), then MH (0.406) and MB (0.408). The Kamta lects sit between
  B and A.
- BH data: MH (0.532), then A (0.539), with B and KS last (0.550). MH data: KS, then B. KS data: MH,
  then B.
- Hj and Bp are mutual nearest neighbours (0.383, 0.418). They also share h < s and the TB-contact
  features.

- So the data favour a **dialect continuum with one sharp Kamarupi isogloss (ḍ > r)**, plus a
  second, Assam-internal one (ś > x/h, c > s), over a clean Bengali–Assamese node. Odia is the
  most distinct member: it keeps -a, NC and ṇ, and it innovates ḷ ṇ from singleton l n.

## 5. Other observations

- **Magadhan a > ɔ is invisible in CDIAL.** Turner transliterates B., A. and Or. orthographically
  (অ = a), and the scorer folds ɔ into o. Adding `a > o` costs B −2.9, A −5.2 and Or −9.4. The
  rounding is visible only in the phonetic sources (BH toulmin, Hj, Bp, Dv, and the B survey lists).
  There it shows two patterns: global in BH and Dv, before a lost final -a in Hj and Bp. In both
  patterns, ā from compensatory lengthening is exempt.
- **No umlaut / i-epenthesis signal.** Bengali epenthesis (`∅ > i / V _ C i`), umlaut
  (`a > e / _ C i`) and vowel harmony (`o > u / _ C i`) all made B, A and Or worse (+0.001 to +0.016).
  Turner's orthographic B. hides them; the colloquial survey lists are too few to show them.
- **MIA -i -u survive apocope** in B, A, Kamta, Hj, Bp and Dv (āji, kāru, āgun), unlike Hindi.
  -iā (< -ikā) stays -iā; a rule `-iā > -i` had LOO ≤ 0 and was removed.
- **Odia -a / -ā** is 50:50 lexical (ā from the -aka extension, not predictable from the etymon: no
  conditioning by syllable count, preceding vowel or cluster). It is the largest single residual in
  Or (≈950 a→ā confusions). B and A behave the same way (-ā in about 30 % of OIA -a nouns). The
  choice is not conditioned by the OIA accent either: final-accented, other-accented, unaccented
  and starred etyma all give 27–45 % -ā (`acc.py`). Odia -ika > -iā is regular (@morph).
- **Little tatsama contamination in CDIAL:** Turner leaves out learned forms, so OIA C+r/y/v clusters
  survive in only 1–2 % of relevant pairs, and intervocalic k t p in about 3 % (B, A, Or; `tatsama.py`).
  The main systematic exception is the **-triṁśat numerals, which keep tr** everywhere (B ektris,
  A ektrix, Or ekatrisa, against Hindi ikattīs). This may be a learned/compound-boundary effect.
  The ∅→k residual (61 in B, 61 in A, 80 in Or) is mostly the expressive -k(k)- extension
  (B jamak, ṭʰasak, gʰuskāna). Desia keeps more clusters (rakta > rɔkɔt), which gives it negative
  LOO for some MIA assimilation rules. I kept those core MIA rules for comparability; their values
  are within noise at n = 406.
- **Numerals** (-viṁśati, -navati, -catvāriṁśat) are a separate, irregular class. They also trigger
  the verb @morph rules (Or `aṭʰāĩsibā`). Exempting them cost more than it gained, because the
  exception also blocks real -ṁśati verbs (daṁśati > dãsibā).
- **Assamese v/w:** intervocalic v-loss gains nothing in A (±0.0005), unlike every other lect
  except Desia. A. keeps many w (zovār, tāu < tāpa), which fits A.'s w < v.
- **Root etyma** (`*cul`, `*ḍag`, `*pʰaṭ`) are reflected as reduplicated/expressive forms (culculiyā,
  ḍagmagāiba). They are the worst misses in every lect and are not sound change.
- **CDIAL parse errors (Sanskrit taken as a reflex):** under úpaviśati the Or., B. and A. "reflexes"
  `vásati` and `upaviṣṭa` are Sanskrit forms from Turner's comments. So are A. `ávadahati` under
  *uddahati and Or. `rōháyati` under *rupati; their accents give them away. They should be dropped
  from the reflex set.
- **Language-code collision:** zoller2023's "B" rows are Bangani, not Bengali.
- **CDIAL OB forms** include some that are really MB/Early-NIA 3sg verbs (`ahai`, `dīsai`). OB has
  no citation-form @morph, and they score fine as -ai.

## 6. Tags defined here; engine wishes

New tags:
- `@x-eastern-final-a-short` — word-final -ā > -a before the MIA -aka > -aa > -ā contraction
  (Apabhraṁśa-type final shortening).
- `@x-eastern-high-length-loss` — ī ū > i u merger, after compensatory lengthening.
- `@x-eastern-length-loss` — full loss of vowel length including ā > a (Hj, Bp, Dv).
- `@x-eastern-nasal-assim` — ŋg mb (> ŋŋ mm) > ŋ m instead of anusvāra + stop.
- `@x-eastern-s-x` — ś > x (Assamese velar fricative; medially ś > h is tagged @s-h).
- `@x-eastern-s-dental` — the merged ś > s (Odia, Rajbanshi, Surjapuri, Desia).
- `@x-eastern-deaffrication` — c cʰ j jʰ > s s z z (A, BH).
- `@x-eastern-l-n` — initial l > n (BH).
- `@x-eastern-deretroflex` — retroflex stops > dentals (Hj, Bp; contact).
- `@x-eastern-pretonic-shortening` — ā > a before a closed following syllable (KS; cf. Bihari).

Engine/segmentation issue that affects every group: the CDIAL profile splits *accented* diphthongs
into two segments (`saínya` → s a í n y a, `saúvarṇa` → s a ú v …, `áirāvaṇa` → á i r …), so
`{ai au} > {ē ō}` misses them. About 120 etyma across all pair files are affected. I added
`a {i u} > {ē ō}@2` right after it (−0.0002 to −0.0003 in Or, B, A).

Engine features I wished for:
- Optional tokens in *targets* (`(V) a {i ē} > ā`), so one @morph rule can cover -ati/-ayati/-yati.
- A syllable-count condition. `C > ∅ / V (C) (C) V _ #` stands in for "polysyllables only".
- A "not the first half of a geminate" condition (`// _ $1`), for flapping and similar rules
  applied before degemination.
- Some way to mark a lexical stratum (tatsama/semi-tatsama) or to allow two outputs (Or -a/-ā).
  Otherwise the residual stays dominated by lexically distributed variants.

## 7. Round 2: vowels and accent

Scratch: `tmp/sound_changes/eastern/r2b/` (probe outputs, `loo0-*` = audit of the partial round-2
files, `loo3-*` = final LOO, `cmp.py` two-chain item diff). Variant tests used `try.py` / `diff.py`.

### 7.1 Scores (round 1 = `results/summary-round1.csv` → round 2 final)

| lect | n train | train NED | test NED | Δtest |
|---|---|---|---|---|
| Or | 4752 | 0.3180 → 0.3091 | 0.3224 → 0.3101 | −0.012 |
| B | 3785 | 0.3305 → 0.3241 | 0.3360 → 0.3230 | −0.013 |
| A | 2706 | 0.3428 → 0.3335 | 0.3532 → 0.3531 | 0.000 |
| BH | 1338 | 0.4606 → 0.4518 | 0.4644 → 0.4570 | −0.007 |
| MH | 548 | 0.4199 → 0.4142 | 0.4060 → 0.3921 | −0.014 |
| KS | 321 | 0.3737 → 0.3737 | 0.3685 → 0.3685 | 0 (reverted, see 7.4) |
| AdivasiOriya | 406 | 0.4430 → 0.4430 | 0.3977 → 0.3977 | 0 (reverted, see 7.4) |
| Hajong | 270 | 0.3578 → 0.3453 | 0.3301 → 0.3148 | −0.015 |
| MB | 136 | 0.3315 → 0.3315 | 0.4082 → 0.4082 | 0 |
| OB | 113 | 0.3331 → 0.3331 | 0.3635 → 0.3635 | 0 |
| Bishnupriya | 110 | 0.3313 → 0.3313 | 0.2466 → 0.2466 | 0 (partial edits pruned) |

No `%gold-drop` directive is used in any eastern file, so none of these numbers is a scoring change.
The B gain includes the partial round's `a > o` inside the two kim-survey `%source` blocks
(colloquial Bengali ɔ for Turner's orthographic a): it is a real change (@a-rounding) but only
applies to those 169 survey pairs; see 7.2 for its size. The dead rule `a {i u} > {ē ō}@2` (0 fires
since the tokeniser now keeps accented diphthongs whole) was deleted from all eleven files.

**Adoption threshold.** Besides a positive LOO, I required each new round-2 rule to gain at least
one item-equivalent on train (ΔNED ≥ 1/n_train: 0.0002 in Or, 0.0037 in Hajong, 0.009 in
Bishnupriya). This pruned several partial-round rules in the small lects (7.4). Two lects whose
test NED rose by more than 0.005 over round 1 (KS +0.006, Desia +0.008) had gained only 3–4 ED units
on train from the same edits, i.e. noise. They are back to their round-1 chains.

### 7.2 Rules adopted (LOO on train; kept only if they read as a change or a tagged @morph)

Vowel / accent / syllable rules:
- **B `V G $2 > $1+long $3 / # C* _`** (@degem-lengthen, first syllable only; was unrestricted):
  unrestricted +0.0008 worse. B now matches A and Or: compensatory lengthening is a first-syllable
  process (bʰatta > bʰāt, but medial geminates just shorten).
- **A `ō > u / _ C* {i u ī ū}`** (@umlaut, early, right after @ai-au): LOO +0.0007, 20 fires
  (kṓṣṭʰa > kuṭʰī, prōṣṭʰī > puṭʰi). This is Assamese regressive height harmony. The same rule is worse in B
  (+0.0002), Or (+0.0010), BH (+0.0013), MH (+0.0004), Desia (+0.0035), Hajong (+0.0015) and OB (0),
  and in KS/Bp/MB it gains less than one item. Adding `ē > i` is worse everywhere (A +0.0007, B
  +0.0015, Or +0.0013). Placing the rule late (after the i/u merger or apocope) is +0.0009 worse
  than early, so the triggers are the older high vowels.
- **A `{r̩ ŕ̩} > a / _ C C`** (@r-vocalic, closed syllable; partial round): LOO +0.0007, 22 fires
  (vr̩ddʰa > bāṛʰā). The accent version (`ŕ̩ > a / _ C C`) is +0.0005 worse. In B the positional rule
  gains only +0.0002, which is below threshold, so it is not adopted there.
- **Hajong `a > o / # C* _ | _ C a # // _ ŋ`** (@a-rounding): ɔ in the first syllable as well as
  before a lost final -a. It is worth 0.0076 over the round-1 `_ C a #` alone. The `// _ ŋ` clause
  is worth 0.0049 (gaŋ, bʰaŋa, aŋgul, aŋtʰi keep a before the A-type retained ŋ(g)). LOO +0.0244.
  In Bishnupriya the same extension gains 0.0055, below its 0.009 threshold, so Bp keeps `_ C a #`.
- **Or `V m V > $1 $3+nas`** (@m-v): intervocalic m is lost and its nasality passes to the next
  vowel (kumāra > kuā̃ra, āmalaka > ãũḷā, gōdʰūma > gahũ). It goes before degemination. LOO
  +0.0025, 86 fires. Plain `m > ∅` −0.0022, `m > ũ` −0.0003, and nasalising the preceding vowel
  +0.0011. This is an **Odia-only isogloss**: the same rule costs B +0.0021, A +0.0019, Desia
  +0.0028, MB +0.0042, BH +0.0024.
- kim `%source` blocks in B: `a > o` (@a-rounding; partial round). It is worth 0.019 NED on the 169
  survey pairs. First-syllable-only is +0.009 worse and not-before-a-high-vowel +0.005 worse. The
  scorer folds gold ɔ into o, so the Bengali ɔ ~ o height harmony cannot be seen. `e > ɛ/æ` (not
  before a high vowel) is +0.003–0.006 worse on those pairs.
- B syncope `a > ∅ / V C _ C V` (@syncope, end of chain; partial round): LOO +0.0006, 165 fires.
  Not adopted elsewhere: MH gains 0.0005 (below threshold), and Or +0.0106, A +0.0042, MB +0.0078
  are all worse. Restricting it to syncope before i is worse in B (+0.0006).
- `ṁ > ∅ / i _` (partial round; Or +0.0007, B +0.0009, A +0.0010): anusvāra lost after i without
  nasalisation. It is mostly the numerals -viṁśati/-triṁśat (B bis, tis) plus kíṁcid and piṁṣati,
  i.e. the MIA vīsa, tīsa type. Kept, but it is numeral-driven.

Citation morphology (@morph; not sound changes):
- `if:starred a > a a / _ #` (partial round): Turner's starred etyma take the -aka > -ā extension.
  LOO Or +0.0032, A +0.0036, B +0.0017, BH +0.0035, MH +0.0029. Adding it is worse in KS, Hj, Bp,
  OB and MB, and in Desia it gains less than one item.
- `if:!numeral` (Or, B) / `if:verb` (A) on the -ati citation rules (partial round): the flags fix
  the numerals (aṭʰāisibā). Switching Or/B to `if:verb` costs +0.0005/+0.0004, and switching A to
  `if:!numeral` costs +0.0010. The difference is -ti nouns that happen to take the verbal ending.
- New: **-āti verbs**, `if:verb ā {i ē} > i b a / _ #` (A; LOO +0.0023; dádāti > diba,
  prábʰāti > pāiba), `> i b ā` (Or +0.0020) and `> ā` (B +0.0005). A post-tonic ā-shortening
  rule had looked like a positive accent effect in A (−0.0005); `diff.py` showed that all its gain
  came from these verbs.
- New: **MH `if:verb (V) {a ā} {i ē} > ∅ / _ #`**: toulmin cites Rajbanshi verbs as bare stems
  (karṓti > kar, krīṇā́ti > kin). LOO +0.0028. Round 1 had rejected the unflagged form because it
  also hit -ti nouns. The same rule in BH/KS is ≥ 0, and the Bp -anī version (−0.0047) is below
  threshold.

### 7.3 The Vedic accent: what it does and does not condition (all ΔNED on train, + = worse)

**Nothing in the eastern group is conditioned by the Vedic accent.** Wherever a vowel change looks
positional, the position or weight version beats the accent version.
- **Compensatory lengthening: first syllable, not accent.** I added an early tonic-lengthening rule
  `{á í ú} > $1+long / _ C C` and restricted positional CL to unaccented words. Measured against
  "no CL in accented words", it recovers only part of what positional CL gives. Accent-version vs
  no-CL-in-accented, against the positional base: B +0.0071 vs +0.0109; Or +0.0024 vs +0.0027; A
  +0.0045 vs +0.0074; BH +0.0080 vs +0.0106; MH +0.0094 vs +0.0118; KS +0.0119 vs +0.0119; Hj +0.021
  vs +0.027; Bp +0.019 vs +0.043; Desia +0.0076 vs +0.0114; OB +0.0099 vs +0.0128; MB +0.0054 vs
  +0.0127. The partial recovery is what one expects: the Vedic tonic is often the first syllable.
  "First syllable *and* tonic" (B +0.0070) shows that pretonic first syllables lengthen as well.
  Adding tonic CL on top of positional CL: B +0.0002, Or +0.0003, A +0.0001.
- **Odia's weight condition (no CL before a long vowel) is Odia-specific.** It is worth −0.0027
  in Or (from round 1). The same exception costs B +0.0068, A +0.0017, BH +0.0096, MH +0.0059, Hj
  +0.0011. KS alone gains (−0.0012, first syllable + not before long), but that is noise
  (see 7.1). Blocking CL only before a following ā: A +0.0012, B +0.0064.
- **Final -a / -ā (the Odia 50:50 split, also 30 % in B, A).** Not the accent: oxytone `á > aa / _ #`
  gives Or +0.0013, B +0.0044, A +0.0033, MH +0.0057, Hj +0.0161. Post-tonic final a gives Or +0.0043,
  B +0.0041, A +0.0031. Restricting to cluster-final gives Or +0.0023 (plain) or +0.0006 (tonic).
  The only predictor is Turner's *star (the `if:starred` @morph above).
- **Apocope of -i:** positional `{a i} > ∅` B +0.0004, A +0.0255, MH +0.0044, BH +0.0014; the
  post-tonic -i version B +0.0040, A +0.0282, MH +0.0016, BH +0.0006. Non-verb -i loss is B +0.006.
  The probe's "final -i → ∅ 71 %" cell in B is the -ati verbs, which the @morph rules already handle.
  MIA -i -u survive in the eastern group (confirms §5). In Hajong `{a i}` gains only 0.0028
  (< 1 item).
- **Tonic lengthening in open syllables** `{á í ú} > $1+long / _ C V`: Or +0.0074, B +0.0060,
  A +0.0092 (first syllable only: +0.0045 / +0.0053 / +0.0063). **Pretonic shortening**
  `{ā ī ū} > $1-long / _ !#* Acc`: Or +0.0038, B +0.0054, A +0.0045. **Post-tonic syncope**
  `{a i u} > ∅ / Acc C _ C V` (early): Or +0.016, B +0.014, A +0.018.
- **Post-tonic ā shortening** `ā > a / Acc !#* _ !#* V`: A −0.0005 and MB −0.0016 look positive,
  but in A the whole gain is the -āti verb @morph artefact (7.2), and MB's is under one item. Or 0,
  B +0.0007, MH +0.0011. Rejected.
- **Vocalic r̩:** `ŕ̩ > a` (accent) Or +0.0004, B −0.0002, A +0.0003, Hj −0.0009, BH +0.0004, MH
  −0.0005. `r̩ > a / _ C C` (position) is about the same size, and `r̩ > u` after labials is 0 to
  +0.0009 everywhere. The one sizeable signal is Hajong `r̩ > u / # C* _` (−0.0051), but it fires on
  only 2 train etyma, so it was rejected. r̩ > i remains the eastern default.
- **Rounding a > ɔ:** in BH the global late rule is best: first syllable only +0.0104, not before
  i/u +0.0079, non-final +0.0154. Early variants that could see the accent all lose 0.013–0.022 to
  it; among those, tonic-only is +0.0019 worse than all-a. Probe: BH tonic a > o 64 %, pretonic
  55–75 %. For Hj/Bp, an early tonic-only rule (`á > o // _ C C | _ #`) is +0.006–0.007 worse than
  the same rule on every first-syllable a. MH and KS show no rounding at all: MH global +0.0245;
  KS global +0.045, first syllable +0.034, before -a +0.018. Desia rounds globally (first syllable
  only +0.0073, not before high vowels +0.0073).
- **i/u length merger: unconditioned.** It is ī ū > i u in every position. Keeping length finally
  costs Or +0.0030, B +0.0051, A +0.0032. In the final syllable: +0.0030 / +0.0190 / +0.0142. In the
  first syllable: +0.019 / +0.034 / +0.027. Before a final -a: +0.010 / +0.010 / +0.004. The probe's
  only exception is Assamese pretonic ī (46 % ī, n = 24), which is too small to act on.
- Other rejects: `a > u / _ C* u` harmony +0.004–0.006 and `a > i / _ C* i` +0.037–0.046 (Or, B,
  A). The two-mora law `{ā ī ū} > $1-long / _ C C` is ≤ +0.0006 in every lect except KS (−0.0006,
  below threshold); CL masks it. `{ē ō} > {i u} / _ C C`: +0.010–0.011. Medial syncope before i in
  the Kamta lects (the probe's "medial a before i → ∅" cell): 0 to +0.013 in any placement. Excluding
  root etyma from CL: B +0.0006, A +0.0012.
- Accent tests that could not be run cleanly: accent-conditioned *medial* syncope and a late
  accent-conditioned ɔ need the accent to survive past @acc-loss. A marker vowel (a > ə early)
  breaks every literal-`a` rule in between; the control alone cost +0.027 in B. An engine wish: a
  per-segment "was tonic" feature that survives @acc-loss (or `Acc` matching on the headword
  position after rewrites).

### 7.4 Partial-round edits reverted or pruned
- KS `ō > u` umlaut (+0.0009, 7 fires), Bp `ō > u` (+0.0005), MB `ō > u` (2 fires), Hajong
  `{ō ē} > {u i}` (+0.0022, 4 fires): each under one item-equivalent.
- Hajong `a > ∅ / V C _ C V` (1 fire), Hajong `{a i}` apocope (reverted to -a only), Bp
  first-syllable rounding (+0.0055 < 0.009).
- KS: `V G $2 > $1+long $3 / # C* _ // _ C* Long` plus two-mora (train −0.0019 = 4 ED units; test 0.3656 → 0.3743). Desia:
  starred @morph (−0.0009) and syncope (−0.0008; test +0.008). All reverted.
- Hajong and Bishnupriya `{a ā} {a ā} > ā` now have negative LOO (−0.0044, −0.0045). The "gains" from
  removing it are forms like sakāla > hoal ≈ sokal, where Hajong has a tatsama-like k. The rule is
  kept as a shared eastern change.

### 7.5 What this says about the group
- The eastern NIA vowel system is organised by **word position** (first syllable vs the rest) and
  by the **Odia weight condition**. It is not organised by the Vedic accent. First-syllable CL,
  first-syllable ɔ in Hajong, and the absence of any tonic-lengthening or pretonic-reduction signal
  all fit the initial stress of modern Bengali, Assamese and Odia, which is independent of Vedic.
- New isoglosses: Odia intervocalic m > nasalisation (Or only); Assamese regressive height harmony
  ō > u (A only in these data); Odia's CL blocked before a long vowel (Or only). The B/A/Or
  first-syllable restriction of CL now separates the three standard languages from the Kamta,
  Hajong and Bishnupriya files, where it makes no difference (0.0000) because the toulmin and kim
  data have almost no medial geminate reflexes.

Frontiers for all eleven lects were rerun after the final edits (11:34). The logs are in
`tmp/sound_changes/eastern/r2b/frontier-*.log`. Section 1 still holds the round-1 table; the round-2
scores are in 7.1.

## 10. Extension-aware score (round 4)

Scratch: `tmp/sound_changes/eastern/r4/`. It holds `OIA-*.before.sc` (the chains as they stood when the
round started), `ext0-*`/`ext1-*` (`score --ext` before/after, all with the 7-item menu that includes -akka),
`extloo-*` (`--ext-loo`), `tryext.py` (variant tester: plain, ext and accent train NED), and `extan.py`
(residual ending mismatches, plus how much a candidate menu item would add).

### 10.1 Scores (plain NED / extension-aware NED; before → after)

| lect | train plain | train ext | test plain | test ext |
|---|---|---|---|---|
| Or | 0.3091 → 0.3091 | 0.2453 → 0.2453 | 0.3101 → 0.3101 | 0.2498 → 0.2498 |
| B | 0.3241 → 0.3240 | 0.2625 → 0.2594 | 0.3230 → 0.3232 | 0.2618 → 0.2581 |
| A | 0.3335 → 0.3334 | 0.2734 → 0.2718 | 0.3531 → 0.3533 | 0.2928 → 0.2902 |
| BH | 0.4518 → 0.4518 | 0.3731 → 0.3710 | 0.4570 → 0.4565 | 0.3793 → 0.3736 |
| MH | 0.4142 → 0.4141 | 0.3320 → 0.3255 | 0.3921 → 0.3968 | 0.3089 → 0.3061 |
| KS | 0.3737 → 0.3733 | 0.2753 → 0.2662 | 0.3685 → 0.3666 | 0.2524 → 0.2425 |
| AdivasiOriya | 0.4430 | 0.3614 | 0.3977 | 0.3637 (unchanged) |
| Hajong | 0.3453 → 0.3440 | 0.2728 → 0.2677 | 0.3148 → 0.3148 | 0.2547 → 0.2553 |
| MB | 0.3315 → 0.3321 | 0.3130 → 0.2876 | 0.4082 → 0.3936 | 0.3851 → 0.3491 |
| OB | 0.3331 | 0.3094 | 0.3635 | 0.3319 (unchanged) |
| Bishnupriya | 0.3313 | 0.2589 | 0.2466 | 0.1784 (unchanged) |

The accent scores of Or, B and A are unchanged. Frontiers were rerun for B, A, BH, MH, KS, Hajong and MB.

**Ending choices** (train, after; test in brackets where it differs notably):
- Or: bare 68, aka 15, ikā 10, akka 3, uka 2, aḍaka 1, araka 1, allaka 0.
- B: bare 70, aka 13, ikā 8, akka 4, aḍaka 2, allaka 1, araka 1, uka 1.
- A: bare 70, aka 14, ikā 9, akka 3, aḍaka 2, uka 2, allaka 1, araka 0.
- BH: bare 60, aka 16, ikā 10, akka 7, aḍaka 3, allaka 2, uka 2.
- MH: bare 58, aka 14, ikā 13, akka 8, allaka 3, uka 3, aḍaka 1.
- KS: bare 58, aka 16, ikā 9, akka 7, aḍaka 7, uka 2, allaka 1.
- Desia: bare 57, aka 18, ikā 13, allaka 4, uka 3, aḍaka 2, akka 2.
- Hajong: bare 67, aka 17, ikā 6, uka 6 (test 10), akka 3, aḍaka 1.
- MB: bare 79, aka 15, ikā 4, the rest ≤ 1.
- OB: bare 81, ikā 7, aka 4, uka 3, akka 3, aḍaka 2.
- Bishnupriya: bare 70, aka 17 (test 26), ikā 8, aḍaka 2.

### 10.2 Rules changed (LOO: plain; ext-LOO = on the extension-aware train NED)
- **The `if:starred a > a a / _ #` @morph rule doubles endings, as the coordinator suspected.** On -uka
  variants it gives -uā (gʰōṭuka > gʰoṛuā instead of -u). More importantly, it makes -∅ impossible for
  a starred etymon, because every variant then ends in a vowel. That shows up as the largest residual
  ending mismatch in B (110 reflexes of "pred -ā → gold -∅": *aḍḍa > āṛ, *ēkka > ek), BH (24) and MH (21).
  Its ext-LOO is **negative** in every lect that has it: B −0.0090, A −0.0080, MH −0.0086, BH −0.0067.
  Its plain LOO stays positive: removing it costs plain +0.0017 to +0.0036. So it is kept (the plain score
  needs it) and restricted where that is free:
  - B, A: `// k _` (not on -ka forms, so not on the extended variants). Plain 0 / −0.0001; ext −0.0010 /
    −0.0008.
  - Or, BH, MH: the same restriction costs plain +0.0004 / +0.0024 / +0.0018 for ext ≤ −0.0015, so
    the rule is unchanged there.
  - The cleaner fix is an engine one: let `if:` see whether a form is an extended variant
    (`if:!ext`). Then the -aka variant would supply the -ā that the starred rule approximates.
- **CL first syllable only** in BH, MH, KS and Hajong (@degem-lengthen; B, A and Or already had it). The
  -allaka and -akka variants showed CL wrongly applied medially (gʰorālā, gʰorāk). Plain 0 to +0.0002;
  ext −0.0022 (BH, MH, Hj) and −0.0009 (KS). The ext-LOO of the whole rule is +0.015 to +0.038. Bishnupriya:
  ext +0.0005, so not adopted there.
- **Syncope** (@syncope, end of chain):
  - MH and KS `a > ∅ / V C _ C ā #`: plain −0.0004; ext −0.0043 / −0.0082; ext-LOO +0.0043 / +0.0082
    (markaṭa-ka > mākrā, *gōṭṭa > goṭlā).
  - Hajong `a > ∅ / V C _ C V`: plain −0.0012, ext −0.0029, ext-LOO +0.0029 (markaṭa > makra).
  - B's existing rule now has `// ṛ _ ṛ | r _ r | l _ l | n _ n | t _ t | k _ k | ṭ _ ṭ`, so syncope creates
    no new geminate (-aḍaka came out gʰoṛṛā). Plain 0, ext −0.0006; ext-LOO of the rule +0.0050.
  - BH (placed before a > ɔ: plain +0.0012) and A/Or (plain +0.002 to +0.011) are rejected.
- **Glides:** B `∅ > y / {i u e} _ ā` @ya-sruti (was i only): plain −0.0001, ext −0.0006, ext-LOO +0.0024
  (kū́pa > kuyā, kētaka > keyā). A `∅ > v / {u o} _ ā` (kʰasa > kʰahuvā, ṭēraka > ṭeruvā): plain −0.0004,
  ext −0.0007, ext-LOO +0.0004. Or `∅ > y / i _ ā` is worse (+0.0014 plain).
- **MB late hiatus contraction** `{a ā} {a ā} > ā` @hiatus @aka at the end of the chain. Plain +0.0007
  (inside the 0.001 allowance), **ext −0.0254**, ext-LOO +0.0254, test ext 0.385 → 0.349. The -aka
  variants of MB came out -aa (gʰoṛaa), while MB spells bālā, ruṭʰā. Under the plain score the -aa/-ā
  choice was invisible. This revises §3/§4: MB has the contraction, at least word-finally, so it
  enters on the OB → MB boundary. OB still rejects it (plain +0.0065 even word-final only), in line
  with the Caryāpada spelling.
- Tested and rejected: Or `a > ā / u _ #` for -uka > -uā (ext +0.0004).

### 10.3 What the choices say
- **-aka is the productive masculine/neuter extension everywhere.** It gives -ā in Or, B, A and the Kamta
  lects, -a in the length-less Hajong, Bishnupriya and Desia, and -aa > -ā in MB. It is chosen by
  13–18 % of reflexes; Bishnupriya test has 26 %. **-ikā** (> -i; Desia -iā) is chosen by 8–13 %.
  **-akka** (> -ak, Or -aka) is chosen by 3–8 %, most in the Kamta lects (BH 7, MH 8, KS 7). It is
  the expressive -k(k)- that §5 had listed as a ∅→k residual (B jamak).
- **-ḍa** (-aḍaka > -ṛā/-rā) is productive only in the Kamta–Surjapuri belt (KS 7 %, BH 3–4 %) and
  weakly in B (2 %). **-la** (-allaka) shows mostly in Desia (4 %) and MH (3 %). **-ra** (-araka) is
  ≤ 1 % everywhere.
- **-uka** is chosen much more in Hajong (6–10 %, > -u) than anywhere else (1–3 %).
- The bare stem is chosen least in the Kamta lects and Desia (56–61 %) and most in OB and MB (79–81 %).
  The old orthographic sources rarely show the extended forms: 4 % -aka in OB against 13–18 % in
  the modern lects.

### 10.4 Should the menu change? (evidence from `extan.py`, train, eastern lects)
The largest remaining ending mismatches, after the fixes above:
- B: -ā→-∅ on starred etyma (see 10.2), then verbal nouns in -āna/-na (≈ 56; verbs are never
  extended, so this is citation morphology), -ā→-yā/-iyā (26), -i/-u→-∅ (58).
- Or: -a→-ī (22; the Odia -ī < -ikā/-inī written long), -ā→-iā/-uā (25), -a/-ā lexical (§5).
- A: -ā→-ī (9), -ā→-iyā/-uvā (14), -iba verb forms.

Candidates, as the number of train reflexes whose best NED improves (non-starred etyma only, so that
the starred-rule artefact is excluded) and the ext-NED gain:

| candidate | B | Or | A | verdict |
|---|---|---|---|---|
| **-ikaka / -iyaka** (> -iyā, Or -iā) | 59 / 0.0028 | 86 / 0.0025 | 37 / 0.0029 | **add**: real NIA adjectival/agentive -iyā (-ika-ka, -iya-ka), frequent in B/Or/A (oṛiyā, kũṛiyā, ālsiyā) |
| **-ukaka** (> -uyā, Or -uā, A -uvā) | 62 / 0.0025 | 59 / 0.0015 | 28 / 0.0019 | **add** (lower priority): real agentive/diminutive -uā (kaṛuā, kāruā, aḷasuā) |
| thematic **-a** replacing -i/-u | 87 / 0.0043 | 49 / 0.0019 | 30 / 0.0015 | real MIA thematisation of i/u-stems (ikṣu > ākʰ, agni > āg). It needs a new variant type (keep the stem, drop the final vowel), not a suffix. Worth considering |
| -la, -ra without -ka | 98, 82 | 56, 74 | 29, 24 | **do not add**. Most of the gains duplicate the thematic-a effect (ákṣu > ākʰ via "ākʰla" → ākʰ) or are surface fits |
| -ḍa without -ka | 29 | 35 | 27 | small; the -aḍaka item already covers it after apocope. Do not add |
| -ilaka (> -ilā, -ilī) | 37 | 75 | 29 | plausible (Or -iḷā/-ilā), mid priority |
| -ikka | 13 | 28 | 4 | do not add (akka already covers most) |
| -iya / -īya (bare) | 0 | 0 | 0 | useless here (merges with the -ikā output) |
| plain fem. -ī | 0 | 10 | 0 | useless here, because of the ī > i merger (identical to -ikā) |
| -āla / -āluka | 26 / 21 | 18 | 14 | small; surface |
| -o | 34 | 5 | 14 | no: these are Turner's B/A -o spellings of tatsama or final-cluster -a, not a suffix |

Menu items the eastern lects barely use: **-araka** (0–1 % everywhere, never chosen in A, BH, MH, KS,
Hajong or Bishnupriya train), **-allaka** in Or (0 %) and Hajong (0 %), and **-uka** outside Hajong
(1–3 %). I would keep -araka for other groups but it contributes nothing here.

**Main recommendation:** before adding items, give `if:` rules a way to see extended variants
(`if:!ext`). In this group the biggest distortion of the extension-aware score is not a missing
suffix: it is the -aka @morph approximation (the starred rule) double-counting against the -aka
variant.

## 11. Verb citation forms (round 5, cut short at 18:27)

Scratch: `tmp/sound_changes/eastern/r5/`. It holds `OIA-*.before.sc`, `v0-*` (`score --verbs` before),
`vdump.py` (each verb-menu form through the chain with `@morph` off) and `tryv.py` (plain / ext /
verb-menu / accent train NED per variant).

**Rule added** (Or, A): `v y > v v` @assim-sonorant, after `use assim-sonorant.lyv`. This is the
regular MIA vy > vv (> bb) assimilation; the chain had been leaving -tavya as -tabya. Train deltas:

| lect | plain | ext | verb-menu | accent |
|---|---|---|---|---|
| Or | −0.0002 | −0.0069 | **−0.0353** | unchanged |
| A | −0.0001 | −0.0003 | −0.0013 | unchanged |
| B | −0.0002 | −0.0001 | +0.0005 | — (not adopted) |

**Verb-menu NED** (test verbs; plain NED is unchanged by this round):

| lect | test verbs | plain | verb menu before → after |
|---|---|---|---|
| Or | 232 | 0.262 | 0.421 → **0.381** |
| A | 137 | 0.290 | 0.432 → 0.431 |
| B | 162 | 0.299 | 0.323 (not done) |

In Or, -tavyakam now wins 127 of 232 test verbs (it was 50).

**Why the eastern citation forms still lose to plain.** They continue formations that are not on the menu:
- Or -ibā and A -iba come from -**i**tavya(-ka) (MIA -iavva). Root + -tavya cannot give the -i- by
  regular sound change (kʰan-tavya > kʰantabā, not kʰanibā).
- B's -ā verbal noun (ā̃kā, cālā) comes from root + -aka (or -ita-ka). The menu's -anaka gives -nā,
  and -ita gives -i.

Recommendation to the coordinator: add **-itavya**, **-itavyaka** and root **-aka** to `VERB_MENU`.

**Not done:** B (only vy > vv tested), and all the small lects (BH, MH, KS, Hajong, MB, OB,
Bishnupriya; Desia has no verbs). Their verb-menu scores are already below plain; `v0-*.txt` has the
numbers.

## Round 6 audit

Scratch: `tmp/sound_changes/eastern/r6/`. It holds:
- `OIA-*.before.sc`, the chains at the start of the round;
- `res-*.txt`, the `residuals.py` reports;
- `final-gates.txt`, the before → after gates for every lect;
- helper scripts: `mkvar.py` + `g.sh` (build a variant in `scr/` and run `gates.py` on it), `cmpx.py` (which
  reflexes a candidate makes better or worse, on the extension-aware best variant), `subs.py` (substitution
  counts with position) and `vplacebo.py` (verb-menu placebo test).

### Scores (before → after; gates.py, all ACCEPT unless noted)

| lect | ext train | ext test | verb train | verb test | plain train | plain test | exact train |
|---|---|---|---|---|---|---|---|
| Or | 0.2103 → 0.2091 | 0.2118 → 0.2109 | 0.1601 → 0.1589 | 0.1461 → 0.1455 | 0.3125 → 0.3121 | 0.3187 → 0.3186 | 20.7 → 20.9 |
| B | 0.2434 → 0.2342 | 0.2467 → 0.2388 | 0.3059 → 0.2762 | 0.2855 → 0.2602 | 0.3261 → 0.3246 | 0.3315 → 0.3330 | 25.0 → 25.2 |
| A | 0.2860 → **0.2306** | 0.2999 → **0.2435** | 0.4385 → **0.1839** | 0.4180 → **0.1583** | 0.3371 → 0.3360 | 0.3559 → 0.3506 | 22.9 → 23.4 (accent 0.637 → 0.616) |
| BH | 0.3595 → 0.3552 | 0.3626 → 0.3574 | 0.4120 → 0.3948 | 0.3718 = | 0.4553 → 0.4593 (justified, below) | 0.4591 → 0.4680 | 9.3 → 9.0 |
| MH | 0.3136 → 0.3086 | 0.2965 → 0.2896 | = | = | 0.4172 → 0.4174 | 0.3998 → 0.4005 | 13.7 → 14.6 |
| KS | 0.2606 → 0.2602 | 0.2380 → 0.2372 | = | = | 0.3733 → 0.3727 | 0.3666 → 0.3685 | 20.9 → 21.2 |
| Hajong | 0.2592 → 0.2469 | 0.2507 → 0.2375 | 0.3123 → 0.2265 | 0.2649 → 0.1961 | 0.3440 → 0.3385 | 0.3148 → 0.3160 | 20.7 → 22.6 |
| Bishnupriya | 0.2430 → 0.2415 | 0.1588 = | = | = | 0.3313 → 0.3297 | = | = |
| AdivasiOriya | 0.3585 → 0.3431 | 0.3637 → **0.3287** | — | — | 0.4430 → 0.4434 | 0.3977 → 0.3884 | 7.9 → 7.6 |
| MB | 0.2825 → 0.2791 | 0.3404 → 0.3351 | = | = | 0.3321 → 0.3323 | 0.3936 → 0.3971 | 21.3 = |
| OB | unchanged (no candidate passed) | | | | | | |

### Error profile
The ending (last two segments) accounts for 34 % of the error in Or, 47 % in B and A, 43–52 % in the small
lects. Next come consonant insertions/deletions (12–18 %) and vowel quality (8–14 %).

Most of the ending error is lexical, or depends on which extension the reflex took:
- Or -a ~ -ī ~ -ā;
- Or -i vs -ī (orthographic);
- the Kamta survey suffix -ʸæ and locatives -ot/-t;
- the numeral -ttar.

The medial vowel-length and a/u/o confusions in Or and A are scattered: no context covers ≥ 5 pairs with
a clean split (checked with `subs.py`, contexts included). They are left alone.

### Changes kept
1. **Short final -i lost; MIA -ī (< -in) kept** (B, BH, Hajong).
   - `{a i} > ∅ / _ #` replaces the -a-only apocope. `i n > ī / _ #` is added before final-consonant loss.
   - The `cmpx.py` split is clean. It helps agní > āg, trī́ṇi > tin, rāśí > rās, paŋktí > pā̃t, and the
     bare verbs (-āi > -ā: ā́nayati > ānā). Before `i n > ī` was added, it hurt every -in stem
     (hastín > hāti, pakṣín > pākʰi, dʰanín > dʰani, mālin > māli).
   - With both rules, B has 183 reflexes better and 2 worse.
   - This revises round 2 ("MIA -i survives"): only the long -ī < -in survives; short -i is apocopated
     like -a.
   - Rejected in A (plain +0.0046, accent worse), MH (plain +0.0048), Desia (plain +0.0065) and
     Bishnupriya (plain +0.0026). A really keeps -i.
2. **Assamese / eastern -iba from the -itavya(kam) menu forms.** Two rules:
   - `v v > b b`: geminate vv > bb; A keeps single v as w.
   - `a > ∅ / i _ C`: a absorbed after i before a consonant, so -iavva > -ivva > -ibba > -iba. The
     `_ C` condition keeps -iā < -iyaka (āṭiyā, puriyā); the unconditioned `a > ∅ / i _` broke 79 of these.
   - In A the two rules alone take verb train 0.4385 → 0.1901 and ext train 0.2860 → 0.2339.
   - Also installed in MH, KS, Bishnupriya and MB (small gains), and the hiatus rule alone in B (verb
     −0.0016). BH and Hajong rejected them (ext test +0.0006 / +0.0007).
3. **A: intervocalic v-loss** (`use v-loss.v.intervocalic`, before -aka contraction). ext −0.0025, verb
   −0.0059, accent 0.637 → 0.616; 120 reflexes better, 42 worse. Round 1 had found no signal; under the
   extension-aware score the w-retaining forms (sēvā > xev, kavārí > kavāri) are a lexical minority.
4. **B: final h lost** (`h > ∅ / V _ i #` before apocope, `h > ∅ / V _ #` after it): rēkʰā́ > re,
   múkʰa > mu, gā́tʰā > gā, varāhá > barā. 19 better, 5 worse; the worse ones are doublets (dāh ~ dā).
   The Odia equivalent was not kept: its gains are doublets too (dahi ~ dai; net −5 ED).
5. **Final ŋg > ŋ** (`g > ∅ / ŋ _ #`; Hajong, A): gáŋgā > gāŋ, śŕ̩ŋga > hiŋ, while ŋg stays medially.
   Hajong ext −0.0022 train, −0.0057 test.
6. **Final -ā > -a added to BH, MH and Desia** (`use final-a-short.a`; the rest of the group already had it).
   Desia ext test 0.364 → 0.329; MH ext −0.0050.
   - For BH the plain train loss is +0.0032. This is the bare-versus-extended artefact: aṣṭā́ > āṭ, kʰáṭvā > kʰāṭ,
     jáŋgʰā > zāŋ, ū́rṇā > un improve, while the attested -ā forms (vḗlā > belā, śipʰā > śipā, vínā > binā)
     continue -ā-ka and are now taken by the -aka variant. ext test −0.0040, exact test 8.7 → 9.6 %.
7. **Hiatus leftovers:** Or `{i ī} > ∅ / {i ī} _` (aśītí > asi, the -śīti numerals) and Or `o > u / _ ā`
   (kṣōda-ka > kʰuā, *ḍōva-ka > ḍuā). MB ya-śruti `∅ > y / {i u e} _ ā` (dēvá-ka > deyā, nā́vika > nāiyā).

### Rejected
- **OIA intervocalic ḍ > ḷ/l** (Or tāḷi, nāḷi; B tālā, nālā; A suli). Tested unconditioned and after a long
  vowel: Or ext +0.0038 and B ext +0.0009. It is lexical (ḍ > ṛ is the regular outcome).
- Or `a > ∅ / C _ i #`: accent +0.083. Or `ṭʰ > ṭ` (the -ṣṭ- words): +0.0026.
- BH final `ś > s` (+0.0056 plain); BH deaspiration sparing ṭʰ (+0.0016).
- B late medial ā-shortening (+0.0083).
- A `a > o / V C _ C i` (+0.0037); A final h-loss (ext test +0.0005).
- OB intervocalic `n > ṇ` (+0.0020). OB: no candidate passed.

### Menu gaps (reported, not added; placebo = same-shape suffix whose consonant survives)
- **Root + -aka action noun** (B ā̃cā, cālā, the Bengali -ā verbal noun). Adding it to the verb menu
  changes verb train NED by −0.0763 in B, −0.0061 in Or and −0.0028 in A. The placebo root + -aṭaka gives
  −0.0030 / 0.0000 / −0.0003, and -itaka −0.0033 / −0.0006 / 0.
  - This is B's largest remaining class: 241 "n → ∅" edits, where -anaka gives -nā against -ā.
- **-ukaka (-uā/-uyā agentive)** for nouns (Or aḷasuā, kaṛuā; A kʰahuvā) remains as recommended in §10.4.

### Not done
- OB: no candidate passed.
- The small Kamta lects BH, MH and KS got only the shared fixes above; their residual is mostly toulmin
  noise (-ʸæ, -ot).

### Round 6 second pass

Re-baselined on the changed engine: `verb_form` keeps ś, stress marks are stripped, and `residuals.py`
applies the `%source` tails. The chains at the start of the second pass are
`tmp/sound_changes/eastern/r6b/OIA-*.before.sc`; the before → after gates are in `r6b/final-gates.txt`.
`near.py` groups near-misses (best-variant NED ≤ 0.25) by differing segment pair with context, and
`res35.py` is `residuals.py` with 35 examples per class.

| lect | ext train | ext test | verb train | verb test | plain train/test |
|---|---|---|---|---|---|
| B | 0.2342 → 0.2208 | 0.2388 → 0.2261 | 0.2762 → 0.2000 | 0.2602 → 0.1839 | unchanged |
| Or | 0.2091 → 0.2080 | 0.2109 → 0.2098 | 0.1589 → 0.1528 | 0.1455 → 0.1395 | unchanged |
| A | 0.2306 → 0.2225 | 0.2435 → 0.2333 | 0.1839 → 0.1557 | 0.1583 → 0.1215 | unchanged |
| MB | 0.2791 → 0.2755 | 0.3351 → 0.3304 | 0.2937 → 0.2667 | 0.5226 → 0.4944 | unchanged |
| OB | 0.2930 → 0.2792 | 0.3319 → 0.3201 | = | = | +0.0065 / +0.0156 (justified below) |

**Kept:**
- **`%verb-menu +aka`** (root + -aka action noun) in B, Or, A and MB.
  - Assamese is `OIA-A.sc`. `OIA-As-Ul.sc` is Ashokan Prakrit (mia-insular) and was not touched.
  - Gains (train / test verb NED): B −0.0763 / −0.0763, Or −0.0061 / −0.0060, A −0.0020 / −0.0066,
    MB −0.0270 / −0.0282. MB's placebos (+ara, +aṭaka) give 0.0000 / 0.0000; the coordinator's
    placebo numbers cover B, Or and A.
  - Not added in BH: it gains on train (−0.0156) but not on test, where the placebo -ara wins
    (−0.0128). Not added in Hajong (no change).
- **A: neuter -aṁ > -a against masculine -aa > -ā.** `a m > ã / _ #` goes before final-consonant loss,
  and `a ã > a / _ #` before the -aka contraction. So -āpitavyakam gives -āiba (karāiba), not -āibā.
  - ext −0.0078, verb train −0.0262, verb test −0.0302; plain unchanged.
  - The same pair costs B verb +0.0022 (B wants -anakam > -ā) and Or ext +0.022, verb +0.134 (Odia -ibā
    is the long neuter reflex). So this is an Assamese-only development.
- **OB: word-final -aa > -ā** (`a a > ā / _ #`). ext −0.0137 train, −0.0118 test. The plain loss is the
  bare-versus-extended artefact:
  - better: baddʰá > bādʰā, mūṣa > musā, práviṣṭa > paiṭʰā, paṇḍitá > pāṇḍiā (14 reflexes), which continue -aka;
  - worse: 3 words (nilaya > nilaa, stábaka, āmrātaka); the 32-pair test set moves by one word.
  - OB now has the contraction word-finally, like MB. This revises §10.2, which said OB rejects it.

**Tested and rejected:**
- B: stratum-conditioned ŋg (`if:!starred` nasal assimilation; ext +0.0002), neuter -aṁ > -a (verb
  +0.0022), late medial ā-shortening.
- Or: first-syllable lengthening before NC (the verb misses bandʰibā for bāndʰibā; ext +0.0001 train,
  +0.0010 test).
- A: h lost before i (kahiba > kaiba, dahi > dai): accent 0.616 → 0.653.

**What remains, by class** (30+ examples read per class):
- B consonant insertion/deletion (15 %) and vowel quality (10 %) are lexical:
  - r̩ > i/u/a;
  - i ~ e, u ~ o;
  - the doublets kēndu ~ kẽdu.
- B nasalisation (3 %) is B's spontaneous nasalisation (kācā ~ kā̃cā, hāsā ~ hā̃sā, pʰukā ~ pʰũkā). It is
  sporadic, not conditioned by the neighbouring consonant.
- Or verbs: the remaining misses are NC-vowel length (lexical), ṇ ~ n, and aspiration after a nasal
  (dʰuṇḍʰ- ~ dʰuṇḍ-).
- Near-misses:
  - Or: -i ~ -ī and -a ~ -ī are written-length and feminine alternatives;
  - B: -āna vs -ānā needs a neuter -āpana formation, which the menu lacks;
  - A: intervocalic h before i fails the accent gate.

### Round 6 third pass

This pass covers BH, MH, KS, Hajong, Bishnupriya and Desia. Before-files are in
`tmp/sound_changes/eastern/r6c/`, and the gates are in `r6c/final-gates.txt`.

| lect | ext train | ext test | verb | plain |
|---|---|---|---|---|
| BH | 0.3552 → 0.3550 | 0.3574 → 0.3565 | = | = |
| MH | 0.3086 → 0.3075 | 0.2896 → 0.2865 | = | = |
| KS | 0.2602 → 0.2601 | 0.2372 → 0.2323 | = | = |
| Hajong | 0.2469 → 0.2462 | 0.2375 = | = | = |
| Bishnupriya | 0.2415 → 0.2391 | 0.1588 → 0.1564 | = | = |
| Desia | 0.3431 → 0.3352 | 0.3287 → 0.3264 | — | = |

**Kept (all six): the Assamese neuter pair.** `a m > ã / _ #` goes before final-consonant loss, and
`a ã > a / _ #` before the -aka contraction. The neuter -aka-m then gives -a, against masculine -ā.
- The gain comes through the menu's -akam variant: an i/u-stem or verb headword takes the thematic neuter
  and loses its final vowel. In Desia, 17 reflexes are better and none worse: trī́ṇi > tīn, *uppari > ūpor,
  kʰā́dati > kʰa, svápati > so, náva > nũã.
- Together with round 6's first pass, the eastern group now shares:
  - the -aṁ vs -aa split;
  - the -iba pair (vv > bb, `a > ∅ / i _ C`) in A, MH, KS, Bishnupriya and MB;
  - short final -i loss in B, BH and Hajong.

**Verb menu:** `%verb-menu +aka` and its -ara placebo both give 0.0000 on train in MH, KS, Hajong,
Bishnupriya and Desia (Desia has no verb pairs), and MH's placebo alone improves test (−0.0159). The
Kamta toulmin lists cite bare stems (root wins). Not added. BH was rejected in the second pass.

**Fixes carried over from B/A:**
- final -i loss: already in BH and Hajong. MH (plain +0.0048), Bishnupriya (+0.0026) and Desia
  (+0.0065) rejected it in the first pass; KS already drops -i -u.
- -iba: already in MH, KS and Bishnupriya. BH and Hajong rejected it (ext test +0.0006 / +0.0007).

**Near-misses** (NED ≤ 0.25, by segment pair):
- In BH and MH they are dominated by the -akka variant against the toulmin locative -ot / -at (āgot,
  talat) and the survey suffix -ʸæ. These are morphology or source effects, left alone.
- BH also has x < k in pakṣá > pāxā and r ~ ṛ (source notation).
- The rest are one or two items each and lexical. In Hajong and Bishnupriya most are ś ~ h and
  s ~ ś; in Desia, -ɨ and the indradʰanuṣ variants.
- The remaining syncope misses (BH mākrā, Desia katra) were retested in round 6: they lose on
  plain (+0.0012) or on the accent/ext gate.

### Round 6 fourth pass

This pass covers BH, Desia, MH and OB. Before-files and gates are in `tmp/sound_changes/eastern/r6d/`.

| lect | ext train | ext test | plain train | plain test |
|---|---|---|---|---|
| BH | 0.3550 → **0.3513** | 0.3565 → **0.3504** | 0.4593 → 0.4556 | 0.4680 → 0.4621 |
| Desia | 0.3352 → 0.3342 | 0.3264 = | 0.4434 → 0.4423 | = |
| MH | unchanged | | | |
| OB | unchanged (no further candidate passed; its round-6 final -aa > -ā stays) | | | |

**Kept:**
- **BH: word-initial palatals stay affricates** (`{c cʰ j jʰ} > {ʦ ʦʰ ʣ ʣʰ} / # _` before the
  deaffrication; elsewhere c j > s z). This was the biggest consonant-quality class (s → ʦ word-initially:
  38 edits): caṇa > ʦānā, cakrá > ʦākā, cʰākana > ʦʰān.
  - Kamtapuri shares the non-initial deaffrication with Assamese, but toulmin's BH keeps initial ʦ ʣ.
  - The same rule in MH and KS (initial only, or everywhere) costs +0.027 to +0.043. There toulmin's c
    is c, so the ʦ-initial pattern is a BH-specific isogloss.
- **Desia: the -iba pair** (`v v > b b`, `a > ∅ / i _ C`), as in A, MH, KS, Bishnupriya and MB. It
  gains little, since Desia has no verb pairs; it is kept for group consistency.

**Rejected:**
- Desia syncope (`a > ∅ / V C _ C V`, final syllable or before r l ḍ n; placed before rounding). ext train
  −0.0054, but test +0.0037.
- Desia initial deaspiration: +0.0014 / +0.0025. The source mixes aspirated and plain forms.
- MH: keeping geminates (no degem-plain): +0.0124. Toulmin's makkā, cikkā, mittʰā are sporadic.
- BH: syncope before rounding (three variants) improves ext train −0.0021 to −0.0028 but not ext test
  (+0.0021 to +0.0028). Intervocalic k > x: +0.0041.
- OB: -iba pair (+0.0018).

**What remains:**
- BH: vowel quality in the Kamta vowels (o ~ ā, i ~ e) and the toulmin -ʸæ / -ot suffixes, which are
  source morphology.
- MH: sporadic geminates and ɛ ~ e/i ~ a (toulmin's front open vowel, lexically distributed).
- Desia: -ɨ endings, aspiration variation between sources, and a syncope that train supports but test
  does not.
- OB: orthographic -aa/-ā, ī and ṇ.

## Round 7

Scratch: `tmp/sound_changes/eastern/r7/`. It holds:
- `OIA-*.before.sc`, identical to `tmp/sound_changes/snapshots/pre-round7/rules/` (checked with `cmp`);
- `step0/`, the re-gates of the interrupted agent's edits;
- `cand/<label>/OIA-*.sc` + `out/<label>.txt`, every candidate and its gates verdict (`q*.txt` are the queues, `q.sh`
  runs them one at a time);
- `final/`, snapshot → final gates per lect (same engine);
- helpers `vq.py` (predicted vowel → attested segment by syllable and next vowel, per source), `ctx.py`
  (one predicted→attested pair against its match by neighbouring segment) and `extloo-*.txt` (per-rule ext LOO).

### Scores: pre-round-7 snapshot → final (`final/*.txt`; gates.py, same engine: the 11:38 engine)

| lect | ext train | ext test | verb train | verb test | plain train | plain test | ext-exact train | verdict |
|---|---|---|---|---|---|---|---|---|
| Or | 0.2073 → 0.2062 | 0.2088 → 0.2068 | 0.1575 → 0.1563 | 0.1451 → 0.1451 | 0.3116 → 0.3481 | 0.3174 → 0.3602 | 35.9% → 36.2% | ACCEPT-IF-JUSTIFIED |
| B | 0.2221 → 0.2214 | 0.2286 → 0.2264 | 0.2028 → 0.2025 | 0.1958 → 0.1958 | 0.3258 → 0.3396 | 0.3358 → 0.3492 | 38.0% → 38.2% | ACCEPT-IF-JUSTIFIED |
| A | 0.2222 → 0.2222 | 0.2317 → 0.2317 | 0.1655 → 0.1655 | 0.1323 → 0.1323 | 0.3360 → 0.3896 | 0.3501 → 0.4020 | 36.8% → 36.8% | REJECT |
| BH | 0.3747 → 0.3656 | 0.3769 → 0.3633 | 0.4392 → 0.4296 | 0.3741 → 0.3741 | 0.4759 → 0.4740 | 0.4872 → 0.4797 | 13.5% → 14.5% | ACCEPT |
| MH | 0.3130 → 0.3049 | 0.2906 → 0.2811 | 0.2864 → 0.2864 | 0.2815 → 0.2815 | 0.4223 → 0.4256 | 0.4021 → 0.4151 | 21.2% → 22.1% | ACCEPT-IF-JUSTIFIED |
| KS | 0.2677 → 0.2560 | 0.2352 → 0.2294 | 0.2490 → 0.2490 | 0.2833 → 0.2833 | 0.3821 → 0.3736 | 0.3782 → 0.3714 | 29.3% → 32.4% | ACCEPT |
| Hajong | 0.2549 → 0.2542 | 0.2546 → 0.2539 | 0.2624 → 0.2624 | 0.1961 → 0.1961 | 0.3457 → 0.3477 | 0.3331 → 0.3331 | 32.2% → 32.2% | ACCEPT |
| Bishnupriya | 0.2606 → 0.2491 | 0.1858 → 0.1564 | 0.2500 → 0.2500 | 0.1667 → 0.1667 | 0.3497 → 0.3420 | 0.2662 → 0.2368 | 29.1% → 32.7% | ACCEPT |
| AdivasiOriya | 0.4002 → 0.3432 | 0.3840 → 0.3225 | 0.4947 → 0.4704 | 0.5556 → 0.5000 | 0.4934 → 0.4439 | 0.4434 → 0.3884 | 8.9% → 14.0% | ACCEPT |
| MB | 0.2773 → 0.2738 | 0.3304 → 0.3294 | 0.2790 → 0.2684 | 0.4944 → 0.4944 | 0.3323 → 0.3477 | 0.3971 → 0.4064 | 25.0% → 25.7% | ACCEPT-IF-JUSTIFIED |
| OB | 0.2807 → 0.2769 | 0.3201 → 0.3170 | 0.3069 → 0.2931 | 0.0000 → 0.0000 | 0.3426 → 0.3396 | 0.3792 → 0.3792 | 22.1% → 23.0% | ACCEPT |

Plain losses in Or, B, A, MB, MH are the removed `@morph` citation rules (bare headword only; see below). A's REJECT
is its only change, a neutral `@morph` removal (ext, verb, accent unchanged). ext_acc and ext_tone are unchanged in every lect.

### Step 0: the interrupted round-7 edits

The previous agent had installed five edits without writing them up. Each was re-gated (snapshot → current file,
after the 08:15 engine fix):

| lect | edit | ext train | ext test | verdict |
|---|---|---|---|---|
| B | `{i ī} > ∅ / {i ī} _` (ii contraction, as in Or) | −0.0001 | −0.0001 | ACCEPT (5 better / 4 worse), kept |
| MB | neuter pair `a m > ã / _ #` + `a ã > a / _ #` (as in A) | −0.0026 | −0.0010 | ACCEPT (verb −0.0066), kept |
| OB | the same neuter pair | −0.0023 | −0.0031 | ACCEPT, kept |
| MH | `use syncope.a.vcca` → B's syncope (not between identical consonants) | −0.0013 | −0.0022 | ACCEPT, kept |
| Hajong | `use syncope.a.vccv` → the same B rule | −0.0006 | −0.0007 | ACCEPT, kept |

Its uninstalled scratch candidates (o > u / _ ā in A, B, MB; ya-śruti in the small lects; the neuter pair in Or)
were re-tested below; all were rejected (Or's neuter pair was already rejected in round 6).

### Vowel quality (focus 0)

Where the sources write ɔ ɛ æ (`vq.py` per source; Or, A, MB, OB are CDIAL-only here and Turner writes a/e/o, and
Bhatri's ə/ɔ list stays excluded), the chains used to write o/e and were scored correct only because ɔ ɛ were
folded. Two kinds of fix, decided by what the source does:

**Regular changes (chain rules).**
- **Desia a > ɔ, o > ɔ** (AdivasiOriya). Odia/Desia a is /ɔ/, and Desia has a single mid back vowel: behera writes
  ɔ in 187 forms and o twice (koḍe, soe). `a > ɔ` replaces `use a-rounding` (a > o): dánta > dɔt, *nakkʰa > nɔk,
  lavaṇá > nɔn; ā stays a (kavāṭa > kabat). Then `{o õ} > {ɔ ɔ̃}` after the macron notation: *gōḍḍa > gɔḍ,
  *cʰōṭṭa > cʰɔt, gōrūpá > gɔrū. ext train 0.3983 → 0.3600 → 0.3426, test 0.3865 → 0.3451 → 0.3264; 79 + 27
  reflexes better, 1 + 1 worse.
- **Kamta/Bengali a-rounding split with vowel harmony** (BH; Bishnupriya; B's kim survey blocks). The unconditioned
  a > o becomes three rules: a > o before a high vowel in the next syllable (harmony: dʰanín > dʰoni, catuṣka >
  ʦoki, hariṇá > horin), first-syllable a > ɔ (gʰara > gʰɔr, pʰála > pʰɔl, kadala > kɔlā), other a > o (kambalá >
  kɔmbol, lā́ŋgala > nāŋol). Unconditioned a > ɔ was worse (BH ext +0.0059; Bishnupriya +0.0139): toulmin and
  kim-kim write o in non-initial syllables and before i/u. Hajong rejects it (its survey writes o in the first
  syllable: sokal, nodi, kola; doublets bɔśor ~ bosor), and keeps a > o.
  - In B the change is visible only in the phonetic survey lists (Turner transliterates the spelling, a for /ɔ/),
    so the three rules sit in the two kim `%source` blocks, replacing their a > o.
- **e > ɛ in the first syllable before a low vowel or word end** (BH, MH; `e > ɛ / # C* _ C* {a ā #}`): the
  Bengali-type e > æ (æk, dækʰa). kṣḗtra > kʰɛt, vḗlā > bɛlā, *pēṭṭa > pɛṭ, bʰēḍra > bʰɛrā; before i u e stays
  (ṭēraka > ṭeṛi-). In MH the source-wide notation e > ɛ gained more on train (−0.0051) but failed test (+0.0005:
  the ek/tel doublets), so the conditioned rule went in (−0.0040 / −0.0045). Hajong rejects it (+0.0031).

**One source's letter (`@notation` in a `%source` block).**
- toulmin's KS writes every mid front vowel ɛ (e 0 times, ɛ 40): `%source toulmin {e ẽ} > {ɛ ɛ̃}`. ext −0.0092 /
  −0.0058.
- kim-kim's Bishnupriya writes ɛ (e twice in 144 forms): `%source kim-kim2008bishnupriya`. It beat the conditioned
  e-lowering on train (−0.0045 against −0.0023).
- B's Garo survey writes ɛ (mɛgh, tɛl, pɛt; e once): `{e ẽ} > {ɛ ɛ̃}` added to its block (−0.0002 / −0.0006).

BH's loss from the scoring change (ext test .350 → .377 on the pre-round-7 chain) is partly recovered (→ .363); the
rest is toulmin's own o ~ ɔ and e ~ ɛ ~ æ variation (below).

Not changed: A (Turner's 60 Assamese ɔ are scattered: kāla > kɔlā but most a stay a; initial ḍ for d occurs
only before a later r/retroflex but is 31 : 33 even there); Hajong's o/ɔ doublets.

### Other changes kept

- **Or: a > u before hiatus ā** (`a > u / _ {ā ā̃}`, after `o > u / _ ā`). Once intervocalic v/p/m are lost the a
  before the hiatus ā is rounded by the lost labial: *kuṭava-ka > kuṛuā, yavākāra > juāra, maṇḍapa-ka > maṇḍuā,
  bʰrama > bʰuā̃, namáyati > nuā̃ibā. Found with `ctx.py` (a→u is 18 : 6 before ā, 7 : 1 before ā̃, ≤ 0.2 elsewhere).
  ext −0.0011 / −0.0018, verb −0.0013 train; 34 better, 6 worse (compounds whose aā contracted: akṣavāṭa >
  ākʰaṛā, vēsavāra > besara). The `C _` version was slightly worse on train (−0.0010).
- **Kamta iā > i only medially** (BH, KS, MH: `ā > ∅ / i _ C` replaces `use hiatus.a.2`, `ā > ∅ / i _`). The
  contraction is real inside the word (*bʰiyantara > bʰitɔr, *vivaŋka > bɛkā), but word-finally it deleted the -iā
  of the -iyaka variant that toulmin attests (haridrā > haldiā, pradʰí > pāhiā, *kartyā > kɛciā, cʰāyā́ > cʰiā).
  ext-LOO flagged the old rule (removal −0.0011 BH, −0.0021 KS, −0.0021 MH). Dropping it outright passed in MH and
  BH but failed KS test (+0.0008) and cost BH the bʰiyantara words (10 worse); the medial-only rule: BH −0.0019 /
  −0.0009 (26 better, 0 worse), KS −0.0021 / 0.0000; in MH (dropped first, −0.0021 / −0.0028) the medial rule is
  score-identical and was added for consistency.
- **MH: vv > bb dropped** (`use v-b.vv.2`, the A. -iba source): MH's toulmin cites bare verb stems, so the rule only
  acted on nouns (gōrvara > gavar, cárvati > cavvā). ext −0.0007, test =; 2 better, 0 worse.
- **BH e-lowering environment written as in MH** (`{a ā #}` for `{ā #}`): score-identical in BH (no short a
  survives to that point), so the two lects share one rule text.

### Rejected (train decides, test vetoes)

- Unconditioned a > ɔ: BH +0.0059 ext train, Bishnupriya +0.0139; B kim blocks +0.0002 (the harmony version passed).
- Hajong: a-rounding split (+0.0054), first-syllable e-lowering (+0.0031), no aa contraction (`aka.aaaa`, flagged by
  ext-LOO −0.0009; gates REJECT); Bishnupriya no aa contraction likewise.
- MH source-wide e > ɛ notation: train −0.0051, test +0.0005 (ek ~ ɛk, tel doublets).
- o > u / _ ā (Or's hiatus rule) in A (+0.0003), B (+0.0005), MB (0.0000) — the interrupted agent's candidate.
- ya-śruti (`use ya-sruti.iuea`, as in B/MB) in BH (+0.0012), Hajong, KS (+0.0006), MH (+0.0015), OB (test +0.0052),
  Bishnupriya, Desia (0.0000).
- Syncope (B's rule) re-tested under the new scoring: BH before rounding train −0.0046 but test +0.0005; Desia
  before a > ɔ train −0.0069 but test +0.0037; KS replacing `syncope.a.vcca` +0.0009. Same verdicts as round 6.
- KS: dropping the neuter `final-nasal.aa` (ext-LOO −0.0011) failed test (+0.0048); dropping `hiatus.a.2` failed
  test (+0.0008) — the medial-only rule passed instead.

### Legacy `@morph` rules (focus item 6)

Each removed alone and jointly (`q11.txt`); removed when the gates pass or the removal is neutral (ext, verb,
ext_acc, ext_tone unchanged or better on train and test; `neutral.py` checks this). The plain loss is the bare
headword: verbs now get their citation form from the verb menu, which ignores `@morph` anyway.

| lect | rules | result | ext train / test | plain train / test |
|---|---|---|---|---|
| A | 3 × `if:verb … > i b a` (-iba infinitives) | removed, neutral (each alone neutral too) | 0 / 0 | +0.0535 / +0.0502 |
| B | `V a {i ē} > ā`, `a {i ē} > ā`, `if:verb ā {i ē} > ā` (-ā verbal nouns) | removed (ACCEPT-IF-JUSTIFIED) | −0.0001 / 0 | +0.0141 / +0.0147 |
| KS | the two -ā verbal-noun rules | removed (ACCEPT; m2 alone ACCEPT, m1 alone neutral) | −0.0005 / 0 | +0.0009 / +0.0029 |
| BH | the same two | removed, neutral | 0 / 0 | +0.0030 / +0.0038 |
| Hajong | the same two | removed, neutral | 0 / 0 | +0.0020 / 0 |
| MH | `if:verb (V) {a ā} {i ē} > ∅` (bare-stem verbs) | removed, neutral | 0 / 0 | +0.0028 / +0.0089 |
| Or | `∅ > ā / C _ #` (no final consonant), `if:verb a {i ē} > i b ā`, `if:verb ā {i ē} > i b ā` | removed, neutral (each alone neutral too) | −0.0000 / −0.0002 | +0.0370 / +0.0441 |
| Or | `V a {i ē} > i b ā / C _ #` (-Vyati/-ayati as -ibā) | **kept**: removing it fails ext (+0.0001 train, +0.0003 test; also in the joint removal; unchanged when retried on the 11:38 verb-flag engine) | | |
| MB | `a i > e / _ #` (3sg -ai > -e, gárjati > gāje) | removed, neutral after the 10:08 verb-flag fix (before it: ext train +0.0018) | 0 / 0 | +0.0154 / +0.0093 |

### Consistency and rule audit (focus 3–4)

- Per-rule ext-LOO (`extloo-*.txt`, `extloo2-{A,B,Or}.txt`) for every lect. B: no rule has a negative value; round 6's
  `h > ∅ / V _ i #` is now worth 0.0000 on ext (kept: its gain is on the bare headword). Or: only `use anusvara-vowel.vm.preconsonantal.2` (V ṁ > nasal vowel
  before C) is negative (−0.0005). Dropping it passes the gates (ext −0.0005 / −0.0006, verb −0.0026) but was **not
  installed**: it leaves a bare ṁ in the output, which Or writes in 3 of 6,000 forms; the gain is an alignment
  artefact (ṁ costs one substitution against ũ/ŋ/ñ where a nasal vowel costs two: vaṁśī > baũsi, saṁgʰātá >
  saŋgʰā), and it loses the kãsā, ãsuā, dãsibā type (23 worse). The real split it hints at, aṁś > aũs (māṁsá >
  māũsa, pāṁśu > pāũsa), failed as a rule (`ṁ > ũ / {a ā} _ {s ś}`: ext +0.0001 / +0.0004, 8 better, 12 worse). Besides the iā contraction and MH's
  vv > bb above, it flagged only tiny values: A `a y a > ē // _ t {i ē} #` (−0.0001), Desia `Stop Stop` (−0.0007;
  core assimilation, not tested), Bishnupriya `{ai au} > {ē ō}` (−0.0006) and the aa contraction in Hajong and
  Bishnupriya (both rejected by gates).
- **Desia degeminate lengthening in the first syllable only**, as in Or (`/ # (C) _` added): non-initial geminates
  (the -akka extension) degeminate without lengthening: kárṇa > kanɔ, páñca > pancɔ, pʰulla > pūlɔ. ext −0.0011 /
  −0.0014, plain +0.0010; 7 better, 1 worse.
- Now shared as identical local rules: B's syncope (B, MH, Hajong); the harmony a-rounding (BH, Bishnupriya; B's
  survey blocks write the same three rules with `{i u}`, as no long vowels reach them); first-syllable e-lowering (BH, MH); medial iā > i (BH, KS, MH).
- Rejected: Bishnupriya `use deretroflexion` for its `{ṭ ṭʰ} > {t tʰ}` (+0.0030: its ḍ is r, not d).
- Accent: the eastern sources write no accent or tone (the few marked reflexes are OIA citations and ś), so
  focus 2 does not apply; ext_acc and ext_tone are unchanged everywhere.

### What remains / not done

- BH: toulmin's ɔ ~ o in non-initial syllables (o 97, ɔ 30 after the harmony split) and the -ʸæ / -æ finals (66 +
  30 against 442 -ā; the same words also appear with -ā: ʦikā ~ ʦikʸæ, kuā ~ kuæ), which is variation in the survey,
  not a conditioned change. Syncope still passes train only.
- Hajong: its survey's o/ɔ doublets (bɔśor ~ bosor, gʰɔr ~ gʰor) and first-syllable o; no vowel-quality rule passed.
- Or: medial ā ~ a (first-syllable ā before ā is 61 : 54) and intervocalic h ~ ∅ (≤ 0.48 in every context) are
  lexical; of the `ctx.py` checks run (a/u, ḷ/l, h/∅, ā/a) only a > u before ā split cleanly. l before a prefixed
  root's ō stays l (ālōḍayati > āloṛibā, 3 : 1), a morpheme-boundary effect, not tested.
- A: Turner's scattered ɔ and ḍ-for-d; B: i ~ e, u ~ o before ā (≤ 0.11), the -āna verbal noun (menu gap from round 6).
- Desia: syncope passes train but not test (again); -ɨ endings.
- OB: only the step-0 neuter pair; no vowel-quality issue (CDIAL only).
- Focus 5 (per-lect `%verb-menu` / `%ext-menu` items with placebo tests) was not attempted this round.
