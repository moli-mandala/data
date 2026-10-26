# west-pahari — OIA → West Pahari (16 lects)

Files: `rules/OIA-{ktg,kul,bhal,kiuth,WPah,jaun,bhad,dog,kc,khash,mand,kvar,cur,kgr,cam,pan}.sc`,
`results/OIA-*.hand.frontier.csv`. Scratch tools (menu-driven hill-climbing, LOO pruning, canonical
re-ordering, pairwise order tests, isogloss matrix) are in `tmp/sound_changes/west-pahari/`.

Method in one paragraph: a Kotgarhi chain was written by hand from `OIA-H.sc` and the data
(ktg has 1.5k etyma), copied to all 16 lects, then each lect was hill-climbed on train only with
(a) a fixed menu of linguistically motivated alternatives (drop/keep ɔ, ḷ, ʦ, length loss,
apocope variants, degemination variants, Cr retention, sibilant merger, deaspiration …) accepted
greedily when they gained ≥ 0.5–1×10⁻³ train NED, (b) `suggest` and hand inspection of residual
confusions, (c) "borrowing" every rule present in a sibling file and keeping it if it helped
≥ 1×10⁻³ at its canonical position, and (d) LOO pruning of rules with value ≤ 0. Rules that fit
≤ 3 train items were removed unless they recurred in every search round (noted where kept).

## 1. Summary table

NED = mean normalised edit distance (lower is better). "null" = etymon unchanged. The auto
baseline stops at 11–25 rules, so I give its test NED at its largest rule count k and, for a
like-for-like comparison, the test NED of my own ablation frontier (`OIA-X.hand.frontier.csv`)
at the same k. (For kul and jaun the auto baseline is scored on the same pairs but without the
`%source` notation tails.)

| lect | #etyma | #rules | train NED null → final | test NED null → final | test exact | auto test NED at its max #rules (k) | hand frontier test NED at k rules | sources excluded / %source tails |
|---|---|---|---|---|---|---|---|---|
| ktg | 1535 | 49 | 0.722 → 0.456 | 0.707 → 0.448 | 9.6% | 0.546 (23) | 0.487 | — |
| kul | 859 | 43 | 0.696 → 0.462 | 0.682 → 0.438 | 14.1% | 0.566 (17) | 0.498 | blair2021kullu (tail 4) |
| bhal | 831 | 43 | 0.669 → 0.435 | 0.665 → 0.457 | 8.3% | 0.514 (25) | 0.489 | — |
| kiuth | 726 | 46 | 0.673 → 0.418 | 0.668 → 0.431 | 9.8% | 0.548 (20) | 0.471 | — |
| WPah | 464 | 42 | 0.707 → 0.477 | 0.689 → 0.457 | 8.2% | 0.530 (24) | 0.485 | — |
| jaun | 426 | 38 | 0.637 → 0.439 | 0.603 → 0.404 | 15.0% | 0.547 (17) | 0.465 | john2008jaunsari (tail 4) |
| bhad | 394 | 39 | 0.671 → 0.464 | 0.748 → 0.549 | 5.5% | 0.605 (18) | 0.596 | — |
| dog | 322 | 36 | 0.651 → 0.469 | 0.649 → 0.479 | 8.2% | 0.595 (13) | 0.518 | — |
| kc | 311 | 44 | 0.744 → 0.486 | 0.770 → 0.507 | 2.3% | 0.598 (18) | 0.582 | — |
| khash | 209 | 32 | 0.658 → 0.491 | 0.691 → 0.527 | 10.8% | 0.600 (13) | 0.583 | — |
| mand | 212 | 35 | 0.667 → 0.470 | 0.664 → 0.532 | 7.0% | 0.598 (11) | 0.582 | — |
| kvar | 204 | 38 | 0.703 → 0.455 | 0.697 → 0.534 | 6.8% | 0.550 (19) | 0.585 | — |
| cur | 181 | 38 | 0.678 → 0.413 | 0.712 → 0.451 | 5.3% | 0.556 (22) | 0.512 | — |
| kgr | 179 | 34 | 0.682 → 0.492 | 0.703 → 0.544 | 0.0% | 0.598 (19) | 0.630 | — |
| cam | 169 | 38 | 0.671 → 0.361 | 0.649 → 0.438 | 9.1% | 0.527 (15) | 0.481 | — |
| pan | 104 | 33 | 0.637 → 0.380 | 0.721 → 0.370 | 7.1% | 0.559 (12) | 0.499 | — |

At equal rule count k the hand frontier beats the automatic baseline on test in 14 of 16 lects
(by 0.01–0.08 NED; exceptions kvar and kgr, whose test sets have only 44 and 25 pairs). The full hand
chains reduce test NED by 0.13–0.35 from the null baseline.

Sources: `kul` mixes **kullui-org** (Turner-style, ʦ/ʣ, ɔ) with **blair2021kullu** (a village
survey: ə, `ts`/`c`/`z` for the affricates, ˈ stress marks, final -ə, tone written as loss of
aspiration). Blair is kept, with a `%source blair2021kullu` tail (ʦ→c, length removed, ɔ→a, final
-ə); it still scores ≈0.19 train NED worse than kullui-org (0.573 vs 0.387) because it
really is a different set of Kullu dialects. `jaun` mixes CDIAL Jaunsari with
**john2008jaunsari** (digraph aspirates `kh dh`, `ç` for ś, no length, nasality mostly
unwritten); the `%source john2008jaunsari` tail (length removed, Cʰ→C+h, nasal vowels
denasalised, ś→ç) brings John's pairs to the CDIAL level (train NED 0.440 vs 0.435).
`dog`, `mand`, `kgr`, `cam` mix CDIAL with Patyal (1–5), Gill and (Dogri) Brightbill & Turner; their
notations are close enough that no tail helped. Nothing is excluded.

## 2. The base chain (built on Kotgarhi) and where lects deviate

All files share one canonical rule order (enforced with `tmp/…/canon.py`, which re-sorts every
file into the same relative order and keeps the result only if train NED does not get worse).
Four files keep a slightly local order because a canonical move cost ≥ 0.001 train NED: `kc`
(length rule before the hiatus rule), `cam` (l > ḷ after the infinitive rewrite, degemination
before apocope), `kvar` (c > ʦ before the NIA nasal rule, which is written on the Stop class)
and `jaun` (ava > o before intervocalic p > v, whereas Bhalesi prefers apa > ava > o). The MIA
block follows `OIA-H.sc`; the NIA block is West-Pahari-specific. "Absent in" lists the lects whose file lacks the change (the data did not
support it, LOO ≤ 0, or a different variant was chosen).

| # | rule (Kotgarhi file) | tags | absent in | different variant in |
|---|---|---|---|---|
| 1 | `V > $1-acc` | @acc-loss | — | — |
| 2 | `{ai au} > {ē ō}` | @ai-au | khash, kvar, cam, pan | cur |
| 3 | `ṁ {k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ} > {ŋ ŋ ŋ ŋ n n n n ṇ ṇ ṇ ṇ n n n n m m m m}@2 $2` | @nasal-cluster | WPah, bhad, dog, mand, kvar, cur, kgr, cam, pan | — |
| 4 | `r̩ > i` | @r-vocalic | kgr, cam | — |
| 5 | `ṣ > ś` | @sibilant-keep | dog, mand, kgr, cam | — |
| 6 | `{t tʰ d dʰ} r > {ṭ͡ḷ ṭ͡ḷ ḍ͡ḷ ḍ͡ḷʰ} / # _ \| {ā ī ū ē ō} _` | @x-wpah-tr | kul, jaun, dog, kc, mand, kvar, cur, kgr, cam, pan | bhal, bhad, khash |
| 7 | `{t tʰ d dʰ} y > {c c j j} {c cʰ j jʰ}@1` | @palatal-Cy | kc, kvar, cur | — |
| 8 | `{k t p} {ś s} > {k c c} {kʰ cʰ cʰ}@1` | @ks | — | dog, kc, mand, kgr, pan |
| 9 | `{s ś} Stop > $2-asp $2+asp` | @s-stop-aspirate | — | — |
| 10 | `{r y} y > j j` | @palatal-Cy | bhal, dog, kc, khash, mand, kvar, kgr, cam | — |
| 11 | `r ṇ > n n` | @assim-r | bhal, cur, kgr | pan |
| 12 | `{r ḣ} C > $2-asp $2` | @assim-r | khash | pan |
| 13 | `{s ś} {y v r} > $1 $1` | @assim-sonorant | bhad, cur, kgr | dog, khash, cam |
| 14 | `Stop {r y v l Nasal} > $1-asp $1` | @assim-sonorant | kgr | WPah, dog, khash, mand, cam, pan |
| 15 | `Stop Stop > $2-asp $2` | @assim-stop | khash | — |
| 16 | `v v > b b` | @v-b | WPah, kc, khash, kvar, cur, pan | — |
| 17 | `a > o / # (C) C _ C V \| # _ C V // _ C ā` | @a-rounding | bhal, kiuth, WPah, jaun, bhad, dog, khash, mand, cur, kgr, cam, pan | kul, kvar |
| 18 | `l > ḷ / V _ V` | @l-retroflex | bhal, jaun, bhad, dog, khash, kgr, pan | — |
| 19 | `{kʰ gʰ tʰ dʰ pʰ bʰ} > h / V _ V` | @int-aspirate-h | — | — |
| 20 | `{ṭ ṭʰ} > {ḍ ḍʰ} / V _ V` | @int-retroflex | kc, pan | — |
| 21 | `{ḍ ḍʰ} > {ṛ ṛʰ} / V _` | @flap | — | kul, kiuth, bhad, pan |
| 22 | `p > v / V _ V` | @int-p-v | dog | — |
| 23 | `{k g c j t d y} > ∅ / V _ V` | @int-stop-loss | — | — |
| 24 | `ā v a a {i ē} > ā u ṇ õ / _ #` | @morph | mand, kvar | kul, bhal, kiuth, WPah, jaun, bhad, dog, kc, khash, cur, kgr, cam, pan |
| 25 | `a a {i ē} > ṇ õ / _ #` | @morph | mand, kvar | kul, bhal, kiuth, WPah, jaun, bhad, dog, kc, khash, cur, kgr, cam, pan |
| 26 | `a {i ē} > ṇ õ / _ #` | @morph | mand, kvar | kul, bhal, kiuth, WPah, jaun, bhad, dog, kc, khash, cur, kgr, cam, pan |
| 27 | `{ā ō ī} {i ē} > $1 ṇ õ / _ #` | @morph | mand, kvar | kul, bhal, kiuth, WPah, jaun, bhad, dog, kc, khash, cur, kgr, cam, pan |
| 28 | `m > v / V _ V` | @m-v | bhal, kiuth, khash | dog, mand, cur, kgr, pan |
| 29 | `n > ṇ / V _ V` | @n-retroflex | dog, khash | — |
| 30 | `{s ḣ t d m n} > ∅ / _ #` | @final-cons-loss | — | — |
| 31 | `∅ > ṇ õ / C _ #` | @morph | mand, kvar | kul, bhal, kiuth, WPah, jaun, bhad, dog, kc, khash, cur, kgr, cam, pan |
| 32 | `Nasal {k c ṭ t p} > $1 {g j ḍ d b}` | @nasal-voicing | bhal, bhad, khash, pan | — |
| 33 | `{v h} > ∅ / V _ V` | @v-loss @h-loss | — | bhal, kiuth, kgr, cam |
| 34 | `a a > o / _ #` | @hiatus | jaun, bhad, dog, khash | kul, kiuth, mand, kvar, cur, kgr, cam, pan |
| 35 | `{a ā} > ∅ / _ #` | @final-vowel-loss | — | kul, jaun, dog, cur, kgr, cam, pan |
| 36 | `a > ∅ / V C _ C` | @syncope | WPah, jaun, bhad, dog, cur | — |
| 37 | `ṇ > n / {r ṛ ḷ ṇ} _` | @x-wpah-retro-dissim | kiuth, WPah, dog, mand, kvar, kgr, cam | bhal, bhad, khash, cur |
| 38 | `Gx $1 > $2` | @degem-plain | bhad, khash, pan | kul, bhal, kiuth, WPah, jaun, dog, kc, mand, kvar, cur, kgr, cam |
| 39 | `ṭ ṭ > ṭ` | @degem-plain | bhad, khash, pan | kul, bhal, kiuth, WPah, jaun, dog, kc, mand, kvar, cur, kgr, cam |
| 40 | `V ṁ > $1+nas` | @anusvara-vowel | mand, cam, pan | kul, bhal, kiuth, WPah, kc, kvar, cur, kgr |
| 41 | `v > b / # _` | @v-b | — | — |
| 42 | `y > j / # _` | @y-j | — | — |
| 43 | `{ā ī ū ē ō} > {a i u e o}` | @vowel-shift | bhal, jaun, khash, mand, kgr, pan | kul, kiuth, WPah, bhad, dog, kc, kvar, cur, cam |
| 44 | `a > ∅ / V _` | @hiatus | — | — |
| 45 | `{c cʰ j jʰ} > {ʦ ʦʰ ʣ ʣʰ}` | @affrication | kiuth, WPah, jaun, dog, mand, cur, kgr, cam, pan | kvar |
| 46 | `ñ > n / _ {ʦ ʦʰ ʣ ʣʰ}` | @affrication | kiuth, WPah, jaun, dog, khash, mand, cur, kgr, cam, pan | — |
| 47 | `{ṭ͡ḷ ḍ͡ḷ ḍ͡ḷʰ} > {c j jʰ}` | @x-wpah-tr-c | kul, bhal, jaun, bhad, dog, kc, khash, mand, kvar, cur, kgr, cam, pan | — |
| 48 | `h > ∅ / V _` | @h-loss | kul, bhal, kiuth, jaun, bhad, dog, kc, khash, cur, kgr, cam, pan | WPah, kvar |
| 49 | `y > ∅ / C _` | @cluster-simplify | khash, kvar, pan | — |

**Lect-specific additions / variants** (all tagged in the files):

- `kul`: a > ɔ in *every* initial syllable (`a > o / # (C) (C) _`), not only open ones; only
  short a is apocopated (-ā kept); length lost in ī ū ē ō but kept in ā; tr-rules not supported.
- `kiuth`: no c > ʦ; tr > c merges with old c; pʰ > f; a lengthened in a closed initial syllable
  (`a > ā / # (C)(C) _ C C`, placed right after the MIA assimilations: compensatory lengthening
  before geminates and NC); strictly intervocalic, early flap; late v-loss (after apocope);
  ava > o; -aka > -u; infinitive -ṇu; initial a lost before a cluster.
- `jaun`: all short vowels lengthened in closed initial syllables (compensatory, also before NC)
  and NC > nasal vowel + C (`V Nasal > $1+nas / _ Stop`); ava > o; all short final vowels lost,
  then final i u > ī ū; infinitive -ṇō̃.
- `kc` (Kochi): Kotgarhi-like (ɔ, ʦ, l > ḷ, -o), plus final i > e, i > y before a, and
  sibilant/h + nasal > N + Nʰ.
- `kvar` (Kvari, Jouanne): ɔ; law of two morae only outside nasal clusters; c cʰ j jʰ > ʦ ʦ ʣ ʣ
  (affrication *with* deaspiration); all voiced aspirates deaspirated; NC > V̄ + C (Jouanne
  writes the nasal as a tilde on the stop, kāg̃i < káŋkata, which the scorer drops); **initial h
  lost** (hastatala > ātʰēli, *hilla > iluṛɔ; the rule hurts every other lect).
- `bhal` (Bhalesi): no l > ḷ, no ɔ, no post-nasal voicing; tr > ṭ͡ḷ kept, initially *and
  medially* (kṣḗtra > ʦʰeṭ͡ḷ, páttra > paṭ͡ḷ, nētra > nēṭ͡ḷi); geminates kept after
  short a only (agni > agg, kúṣṭʰa > kuṭṭʰ, but degeminated elsewhere); ND > nn
  (`{ṇ n} {ḍ d} > n n`, daṇḍá > ḍann); non-initial voiced aspirates deaspirated; ava > o;
  final ī > i; postvocalic v lost; ṇ > n also after ṭ ḍ.
- `bhad` (Bhadrawahi): **every stop + r > ṭ͡ḷ/ḍ͡ḷ(ʰ)** (priyakāra > ṭ͡ḷār, bʰrū́ > ḍ͡ḷʰauā̃,
  bʰadrá > bʰaḍ͡ḷ); no degemination; no post-nasal voicing; r + dental > retroflex geminate;
  flap only intervocalically and early (before -ati truncation); length lost in ī ū ē ō (kept
  in ā).
- `khash` (Khashali): initial tr > ṭ͡ḷ; Cr otherwise kept; no degemination, no post-nasal voicing;
  non-initial deaspiration; ṇ > n after ṭ ḍ.
- `pan` (Pangwali): Cr kept (tr > tl is attested once, *trāyaḣ > tlāī — too few items for a rule);
  geminates kept; no post-nasal voicing; postvocalic s ś > h; -ā kept; V + ā > vā; **r lost
  before a consonant without gemination** (bárkara > bakrū, vardʰ- > badʰ-), where every other
  lect has rC > CC; strictly intervocalic flap.
- `cur` (Churahi): postvocalic s ś > h (before degemination); ī ū shortened before a consonant;
  -ā kept; secondary ai > e; ava > o; postvocalic v lost.
- `cam` (Chambeali): ś ṣ > s; r + dental > retroflex; -ā kept; Cr kept; v lost after vowels;
  sibilant/h + nasal > aspirated nasal (as `kgr`); penultimate compensatory lengthening.
- `dog` (Dogri): Cr kept with anaptyxis (`∅ > a / C _ r`); ś ṣ > s; **initial voiced aspirates >
  voiceless unaspirated** (tonogenesis, Patyal/Brightbill: bʰadrá > pàdrā, gʰuṇa > kū̀ṇ) and all
  other voiced aspirates deaspirated; secondary ai au > e o; all short final vowels lost;
  pʰ > f (only 3 train items, but re-selected by every search round).
- `mand` (Mandeali): Cr kept; ś ṣ > s; initial voiced aspirates *kept* (contrast with Dogri);
  hiatus V + ā > vā (also Pangwali); postvocalic h lost; initial a lost before a cluster; no
  verbs in the (Patyal) data, so no infinitive rules.
- `kgr` (Kangri): no stop + sonorant assimilation rule survives pruning (Cr kept); ś ṣ > s;
  non-initial deaspiration; v vocalised before C; sibilant/h + nasal > aspirated nasal.
- `WPah` (CDIAL "WPah.", unspecified): a mixture — Kotgarhi-type tr > c and l > ḷ with Kiunthali-type
  pʰ > f and -ṇu; Cr kept; length kept only in final closed syllables.

## 3. Relative chronology

Method: for every pair of interacting rules present in a lect, reverse their order (move the later
rule directly before the earlier one, or the earlier directly after the later one, whichever
costs less) and record the train-NED increase (`tmp/…/chron.py`). Values are ×10⁻³ NED;
n = number of lects where both rules exist.

| order (earlier < later) | mean | n | per lect (×10⁻³) |
|---|---|---|---|
| `@int-stop-loss<@syncope` | +37.6 | 11 | ktg:+49 kul:+29 bhal:+32 kiuth:+40 kc:+39 khash:+15 mand:+13 kvar:+16 kgr:+11 cam:+84 pan:+85 |
| `V G $2 > $1+long<@degem-plain` | +28.2 | 1 | kvar:+28 |
| `@int-stop-loss<@degem-plain` | +21.7 | 13 | ktg:+22 kul:+25 bhal:+9 kiuth:+25 WPah:+26 jaun:+26 dog:+15 kc:+18 mand:+14 kvar:+26 cur:+44 kgr:+14 cam:+19 |
| `@a-rounding<@vowel-shift — length` | +16.3 | 4 | ktg:+20 kul:+1 kc:+28 kvar:+16 |
| `@final-cons-loss<@final-vowel-loss` | +13.6 | 16 | ktg:+9 kul:+18 bhal:+10 kiuth:+11 WPah:+17 jaun:+9 bhad:+8 dog:+6 kc:+14 khash:+8 mand:+10 kvar:+40 cur:+32 kgr:+3 cam:+11 pan:+11 |
| `@a-rounding<@degem-plain` | +12.1 | 4 | ktg:+14 kul:+0 kc:+21 kvar:+13 |
| `@sibilant-s<@ks` | +8.8 | 4 | dog:+4 mand:+14 kgr:+11 cam:+6 |
| `@degem-plain<V G $2 > $1+long` | +8.8 | 1 | cam:+9 |
| `@int-retroflex<@degem-plain` | +8.5 | 12 | ktg:+6 kul:+12 bhal:+6 kiuth:+6 WPah:+11 jaun:+5 dog:+6 mand:+13 kvar:+10 cur:+7 kgr:+7 cam:+13 |
| `@l-retroflex<@degem-plain` | +6.9 | 9 | ktg:+13 kul:+2 kiuth:+8 WPah:+7 kc:+11 mand:+7 kvar:+4 cur:+7 cam:+3 |
| `@tone @deaspiration — initial<@deaspiration @tone` | +6.8 | 1 | dog:+7 |
| `@s-h<@final-vowel-loss` | +6.7 | 2 | cur:+7 pan:+7 |
| `@int-stop-loss<post-vocalic a absorbed` | +5.5 | 16 | ktg:+10 kul:+5 bhal:+3 kiuth:+1 WPah:+8 jaun:+7 bhad:+7 dog:+6 kc:+5 khash:+4 mand:+11 kvar:+5 cur:-1 kgr:+4 cam:+0 pan:+13 |
| `@n-retroflex<@final-vowel-loss` | +4.1 | 14 | ktg:+4 kul:+5 bhal:+7 kiuth:+2 WPah:+8 jaun:+5 bhad:+4 kc:+3 mand:+6 kvar:+4 cur:+0 kgr:+8 cam:+1 pan:+0 |
| `@int-stop-loss<@hiatus — -aka` | +3.5 | 11 | ktg:+3 kul:+1 bhal:+0 kiuth:+1 WPah:+5 kc:+4 kvar:+1 cur:+10 kgr:+2 cam:+3 pan:+9 |
| `@affrication<@x-wpah-tr-c` | +3.1 | 1 | ktg:+3 |
| `@final-vowel-loss<@x-wpah-final-lengthening` | +3.0 | 8 | jaun:+6 bhad:+6 dog:+8 khash:+2 cur:+0 kgr:-2 cam:+0 pan:+3 |
| `-aka > -aa<@final-vowel-loss` | +2.8 | 11 | ktg:+3 kul:+1 bhal:+0 kiuth:+1 WPah:+5 kc:+3 kvar:+1 cur:+8 kgr:+2 cam:+1 pan:+5 |
| `@int-aspirate-h<@v-loss` | +2.6 | 16 | ktg:+1 kul:-0 bhal:+3 kiuth:+1 WPah:+3 jaun:+2 bhad:+2 dog:+2 kc:+2 khash:+4 mand:+1 kvar:+3 cur:+2 kgr:+6 cam:+5 pan:+3 |
| `@ks<@affrication` | +2.2 | 7 | ktg:+1 kul:+3 bhal:+2 bhad:-0 kc:+2 khash:+1 kvar:+6 |
| `@palatal-Cy<@affrication` | +2.0 | 5 | ktg:+2 kul:+2 bhal:+2 bhad:+2 khash:+2 |
| `@m-v<@v-loss` | +1.7 | 13 | ktg:+2 kul:+1 WPah:+2 jaun:+1 bhad:+2 dog:+0 kc:+4 mand:+0 kvar:+2 cur:+0 kgr:+6 cam:+2 pan:+0 |
| `@flap<@degem-plain` | +1.7 | 13 | ktg:+2 kul:+3 bhal:+1 kiuth:+5 WPah:-0 jaun:+4 dog:+2 kc:+0 mand:+0 kvar:-0 cur:+6 kgr:+0 cam:+1 |
| `@l-retroflex<-ati verbs` | +1.4 | 6 | ktg:+4 kul:+1 kiuth:+1 WPah:+0 kc:+3 cur:+0 |
| `@int-p-v<@aya-e` | +1.4 | 3 | bhal:+1 kiuth:+1 cur:+2 |
| `-ati verbs<@l-retroflex` | +1.4 | 1 | cam:+1 |
| `@flap<@x-wpah-retro-dissim` | +1.3 | 9 | ktg:+1 kul:+4 bhal:+0 jaun:+0 bhad:+3 kc:+1 khash:+0 cur:+0 pan:+2 |
| `@deaspiration<@x-wpah-nd-nn` | +0.9 | 1 | bhal:+1 |
| `@aya-e<@int-p-v` | +0.9 | 1 | jaun:+1 |
| `@degem-plain<@final-vowel-loss` | +0.7 | 4 | kul:+0 WPah:+0 kvar:+1 cam:+1 |
| `@n-retroflex<@x-wpah-retro-dissim` | +0.7 | 8 | ktg:+0 kul:+2 bhal:+3 jaun:+0 bhad:+0 kc:+1 cur:+0 pan:+0 |
| `@x-wpah-tr — <@assim-sonorant` | +0.6 | 6 | ktg:-0 bhal:+0 kiuth:+0 WPah:+0 bhad:+4 khash:+0 |

What this implies:

1. **Intervocalic stop loss precedes degemination** everywhere (+9…+44): the NIA geminate
   simplification is younger than the MIA lenition, so old geminates (bʰatta, putta) survive as
   stops. Likewise **int. ṭ > ḍ (> ṛ) precedes degemination** (+5…+13): ṭṭ > ṭ is never flapped.
2. **Syncope follows intervocalic loss** (+11…+85) and **final-consonant loss precedes apocope**
   (+3…+40): textbook, but recovered in every lect that has both rules.
3. **ɔ-rounding (a > ɔ in open initial syllables) precedes both degemination and the loss of vowel
   length** (ktg +14/+20, kc +21/+28, kvar +13/+16): ɔ arises only from *short* a in *open*
   syllables at a stage where CC (> C) still closed the syllable and ā was still long. Kotgarhi
   ɔ is therefore older than the degemination and the collapse of length; Vedic accent has no
   effect on it (open-syllable a gives ɔ equally with and without an accent on it), but it is
   blocked pretonically before a long ā (kaṭāha > kəṛā́) — a later, weight-based stress pattern.
   No rule in the group is conditioned on the Vedic accent, so none carries `@stress`.
4. **l > ḷ precedes degemination** (+2…+13 in all nine ḷ-lects): ll stays l (kʰalla > kʰāl),
   single intervocalic l > ḷ. It also precedes the truncation of -ati for infinitives (ktg +4).
5. **n > ṇ intervocalic precedes apocope** (up to +8; > 0 in 12 of the 14 lects with both rules).
6. **Kotgarhi tr > c postdates c > ʦ** (+3): the lateral affricate from tr (kept as ṭ͡ḷ in
   Bhalesi/Bhadrawahi) became c only after the old palatals had become dental ʦ, so Kotgarhi keeps
   c (< tr) ≠ ʦ (< c): trōṭáyati > coṛnõ vs cāráyati > ʦarnõ. In Kiunthali, which has no
   affrication, the two merge in c. The single Pangwali tl (< tr) would have to *follow* the MIA
   assimilation of Stop + l (else tl > tt), i.e. Cr survived the MIA cluster assimilation in the
   Chamba area and was lateralised later (one item; no rule kept).
7. **s > h (Churahi, Pangwali)**: in Churahi s > h must precede degemination (+6: old ss stays
   s) and apocope (+7); the Pangwali s > h precedes apocope (+7). In Chambeali s > h was supported only
   before I added the ś ṣ > s merger (which the data prefer); after the merger it has LOO ≤ 0 and
   was removed, so the Chamba s > h isogloss rests on Churahi + Pangwali.
8. **Dogri devoicing of initial voiced aspirates precedes general deaspiration** (+7): the
   devoicing (tonogenesis) is only initial and older than the loss of aspiration elsewhere.
9. `sibilant merger < kṣ` (+4…+14 in dog/mand/kgr/cam) only reflects that the kṣ rule is written
   on s after the merger (a notational ordering, not a chronology claim).
10. **The flap ḍ > ṛ is early (MIA-period)**: in Kiunthali and Bhadrawahi, where it is strictly
    intervocalic, it must precede the truncation of -ati (else *jʰaṭati > jʰaḍ-ṇu would not
    flap: gold jʰaṛnu); moving it there from the NIA block gained 1.4 and 2.3×10⁻³. In the
    other lects the rule is written `/ V _` and its position is indifferent, so it now sits in the
    MIA block in every file (Kullui also prefers the strictly intervocalic version there).
11. **ava > o** (Bhalesi, Kiunthali, Churahi, Jaunsari) follows intervocalic p > v in Bhalesi
    (apa > ava > o feeds; −0.9×10⁻³ for moving it there) and sits there in Kiunthali and Churahi,
    but precedes it in Jaunsari (+0.9×10⁻³ the other way) — a weak, lect-specific signal.
12. Orders that do **not** matter (±0): nasal-voicing vs flap, nasal-voicing vs degemination,
    apocope vs syncope, apocope vs degemination (except the four lects whose data prefer
    degemination *before* apocope: kul, WPah, kvar, cam — a weak signal, ≤ 1×10⁻³).

## 4. Shared vs lect-specific changes; sub-grouping

**Shared by (nearly) all 16:** accent loss; ks > kkʰ/ccʰ; s + stop > aspirated geminate; stop +
stop assimilation; intervocalic aspirate > h, ṭ > ḍ, p > v, simple-stop loss; final consonant
loss; intervocalic ḍ > ṛ; v > b and y > j initially; hiatus absorption of post-vocalic a;
sibilant + y/v/r > ss (śvaśura > sas-; 10 lects), ry/yy > jj, vv > bb, rṇ > nn (not Bhalesi);
ṣ > ś with **ś ≠ s kept** (except the Panjabi-adjacent dog/mand/kgr and cam). The **law of two
morae leaves almost no trace**: `{ā ī ū ē ō} > short / _ CC` has LOO ≤ 0 and was dropped in 12
of the 16 lects (kept only in kc, khash, pan, and in kvar outside nasal clusters) — long vowels
before (former) clusters surface long (dánta > dānd, bʰitti > bʰīt), consistent with either
non-application or a complete compensatory re-lengthening at degemination.

Isogloss matrix (from the rule files; `tmp/…/isogloss.py`):

```
feature                                    ktg   kul   bhal  kiuth WPah  jaun  bhad  dog   kc    khash mand  kvar  cur   kgr   cam   pan  
ɔ-rounding (a>o initial syll)              X     X     .     .     .     .     .     .     X     .     .     X     .     .     .     .    
c>ʦ affrication                            X     X     X     .     .     .     X     .     X     X     .     X     .     .     .     .    
l>ḷ                                        X     X     .     X     X     .     .     .     X     .     X     X     X     .     X     .    
NO post-nasal voicing                      .     .     X     .     .     .     X     .     .     X     .     .     .     .     .     X    
geminates (partly) kept                    .     .     X     .     .     .     X     .     .     X     .     .     .     .     .     X    
Cr>lateral (ṭ͡ḷ/tl) kept                   .     .     X     .     .     .     X     .     .     X     .     .     .     .     .     .    
tr>c                                       X     .     .     X     X     .     .     .     .     .     .     .     .     .     .     .    
Cr retained (no Stop + r assimilation)     .     .     .     .     X     .     .     X     .     X     X     .     .     X     X     X    
ś>s merger                                 .     .     .     .     .     .     .     X     .     .     X     .     .     X     X     .    
s>h                                        .     .     .     .     .     .     .     .     .     .     .     .     X     .     .     X    
final -ā kept (only short a lost)          .     X     .     .     .     .     .     .     .     .     .     .     X     X     X     X    
-aka>-o                                    X     .     X     .     X     .     .     .     X     .     .     .     .     .     .     .    
-aka>-u                                    .     .     .     X     .     .     .     .     .     .     .     X     .     .     .     .    
-aka>-ā                                    .     X     .     .     .     .     .     .     .     .     .     .     X     X     X     X    
inf -ṇā                                    .     X     .     .     .     .     .     X     .     .     .     .     X     X     X     X    
inf -ṇū/-ṇu                                .     .     X     X     X     .     X     .     .     X     .     .     .     .     .     .    
inf -ṇo/-ṇõ/-ṇō̃                           X     .     .     .     .     X     .     .     X     .     .     .     .     .     .     .    
deaspiration (voiced asp.)                 .     .     X     .     .     .     .     X     .     X     .     X     .     X     .     .    
lengthening before (degeminated) clusters  .     .     .     X     .     X     .     .     .     .     .     X     .     .     X     .    
final i u > ī ū                            .     .     .     .     .     X     X     X     .     X     .     .     X     X     X     X    
pʰ>f                                       .     .     .     X     X     .     .     X     .     .     .     .     .     .     .     .    
length loss (all/most vowels)              X     X     .     .     X     .     X     .     X     .     .     X     .     .     X     .    
m>v/ṽ intervocalic                         X     X     .     .     X     X     X     X     X     .     X     X     X     X     X     X    
n>ṇ intervocalic                           X     X     X     X     X     X     X     .     X     .     X     X     X     X     X     X    
syncope                                    X     X     X     X     .     .     .     .     X     X     X     X     .     X     X     X    
rṇ > nn (dental)                           X     X     .     X     X     X     X     X     X     X     X     X     .     .     X     .    
sibilant/h + nasal > aspirated nasal       .     .     .     .     .     .     .     .     X     .     .     .     .     X     X     .    
r + dental > retroflex                     .     .     .     .     .     .     X     .     .     .     .     .     .     .     X     .    
ṇ>n after ṭ ḍ too                          .     .     X     .     .     .     X     .     .     X     .     .     X     .     .     .    
```

Hamming-distance UPGMA over these features:
`((bhad,(bhal,khash):5.0):7.5,(((cur,pan):6.0,(kgr,cam):6.0):8.5,(((ktg,kc):2.0,(kul,kvar):6.0):6.0,((kiuth,WPah):7.0,(dog,(jaun,mand):7.0):7.5):9.5):10.3):12.3):14.1`

A second, rule-free measure: **chain transfer**. Apply lect i's whole chain to lect j's train
pairs (with j's own `%source` tails) and take the symmetric excess over j's own chain,
½[(NED(i→j) − NED(j→j)) + (NED(j→i) − NED(i→i))] (`tmp/…/transfer.py`). Small = the two lects
need nearly the same rule list. UPGMA over it (×10⁻³):

`((kvar,(kul,(ktg,kc):54):78):104,((khash,(bhal,bhad):31):39,((jaun,(kiuth,WPah):46):76,((cur,pan):53,(dog,(cam,(mand,kgr):25):52):76):88):104):106):136`

- nearest to ktg: kc (54), WPah (69), kul (78)
- nearest to kul: ktg (78), kc (78), WPah (81)
- nearest to bhal: bhad (31), khash (45), WPah (83)
- nearest to kiuth: WPah (46), jaun (75), kgr (95)
- nearest to WPah: kiuth (46), kc (69), ktg (69)
- nearest to jaun: kiuth (75), WPah (78), mand (88)
- nearest to bhad: bhal (31), khash (33), WPah (83)
- nearest to dog: kgr (52), mand (65), khash (86)
- nearest to kc: ktg (54), WPah (69), kul (78)
- nearest to khash: bhad (33), bhal (45), pan (79)
- nearest to mand: kgr (25), dog (65), cam (77)
- nearest to kvar: kul (97), kc (98), jaun (108)
- nearest to cur: pan (53), cam (57), bhad (85)
- nearest to kgr: mand (25), cam (28), dog (52)
- nearest to cam: kgr (28), cur (57), pan (71)
- nearest to pan: cur (53), kgr (66), cam (71)

The transfer tree and the isogloss tree agree on the main blocks, and the transfer tree (which
also sees rule *order* and every minor rule) is the cleaner of the two: Kotgarhi–Kochi–Kullui
(Satlaj/Kullu), Bhalesi–Bhadrawahi–Khashali (Bhadrawah group), Kiunthali–WPah–Jaunsari (southern
Mahasu/Sirmaur), and a Chamba–Kangra block in which Churahi–Pangwali pair up and Mandeali–Kangri
(25, the closest pair in the group) join Chambeali and then Dogri. Kvari (Jouanne) attaches only
loosely to the Kotgarhi/Kullui block, partly because of its different transcription.

Sub-grouping suggested by the changes (both trees):

- **Bhadrawah group (bhad, bhal, khash)** — the tightest cluster in both trees: Cr > lateral
  affricate ṭ͡ḷ/ḍ͡ḷ (all stops in Bhadrawahi; dentals, also medially, in Bhalesi, with a few
  velar + r forms too — gráhaṇa > ḍ͡ḷahəṇi; initial tr in Khashali), no post-nasal voicing (nt
  stays nt; in Bhalesi voiced ND > nn), geminates kept (bhad, khash; in Bhalesi after short a),
  non-initial deaspiration (bhal, khash), infinitive -ṇū, ṇ > n also after ṭ ḍ. Pangwali shares
  geminate retention and the absence of post-nasal voicing (and has one tr > tl form).
- **Satlaj/Kullu group (ktg, kc, kul; kvar loosely)**: a > ɔ in (open) initial syllables (kul:
  all initial syllables), c > ʦ (ktg, kc, kul; kvar with deaspiration), l > ḷ, n > ṇ, masculine
  -o/-ɔ (ktg, kc) vs -ā (kul), infinitive -ṇõ/-ṇo vs -ṇā (kul); Kotgarhi tr > c. Kotgarhi and
  Kochi are almost identical (isogloss distance 1–2 of ~30 features; transfer excess 54×10⁻³).
- **Southern Mahasu / Sirmaur (kiuth, WPah, jaun)**: no ʦ, no ɔ; infinitive -ṇu/-ṇō̃; Hindi-like
  compensatory lengthening in closed initial syllables (kiuth, jaun); tr > c and pʰ > f
  (kiuth, WPah); ava > o (kiuth, jaun). The CDIAL "WPah." forms sit here, next to Kiunthali.
- **Chamba–Kangra block (cur, pan | cam, mand, kgr, dog)**: -ā kept and -aka > -ā, infinitive
  -ṇā, final i u > ī ū; Cr kept (all but cur); ś ṣ > s (cam, mand, kgr, dog); postvocalic s > h
  in Churahi/Pangwali (the Chamba-proper pair); aspirated nasals from s/h + N (cam, kgr); Dogri
  alone devoices initial voiced aspirates (Panjabi-type tonogenesis) while Mandeali keeps them.
  Mandeali–Kangri is the closest pair in the whole group (transfer excess 25×10⁻³).
- The rule-presence tree places Jaunsari next to Dogri/Mandeali only through shared *absences*
  (no syncope, no ɔ, no ḷ); the transfer tree puts it with Kiunthali/WPah, which I take to be
  right. Kullui likewise groups with the -ā lects by isoglosses but with Kotgarhi/Kochi by
  transfer: its phonology is Satlaj-type, its morphology (-ā, -ṇā) Kangra-type.

## 5. Novel / surprising / data issues

- **The ṭ͡ḷ series.** CDIAL writes a retroflex lateral affricate ṭ͡ḷ/ḍ͡ḷ(ʰ) for Bhalesi, Bhadrawahi,
  Khashali from tr, dr, and in Bhadrawahi from *every* stop + r (pr, bʰr, gr → ṭ͡ḷ, ḍ͡ḷ, dl).
  Pangwali has one tl form. A single rule `Stop r > ṭ͡ḷ/ḍ͡ḷ(ʰ) / # _ | V̄ _` (voicing and aspiration
  kept; Bhalesi also after short vowels) models all of them; the Kotgarhi/Kiunthali/WPah tr > c is
  best derived *through* the same lateral affricate (see §3.6), suggesting tr > ṭ͡ḷ was once
  West-Pahari-wide. Bhadrawahi and Bhalesi also show sporadic **r-attraction to the initial
  stop** (tāmrá > ṭ͡ḷām, darbʰá > ḍ͡ḷʰubʰ, dā́tra > ḍ͡ḷāt, babʰrú > ḍ͡ḷʰabbū) — a Dardic-like
  r-metathesis — but general rules for it (`Stop V (C) r > ṭ͡ḷ V (C)`) cost +8…+24×10⁻³, so it is
  lexical, not regular.
- **rṇ > nn (dental)** in 12 lects (kárṇa > kān, kannɪ; ū́rṇā > ūn; no relevant items in cur,
  kgr, pan), but **Bhalesi keeps the retroflex** (kʌṇ, kʌṇi) — the rule costs +1.7×10⁻³ there.
  Also a Bhadrawah-group feature: ṇ > n after ṭ ḍ as well as after r ṛ ḷ ṇ.
- **Kvari loses initial h** (hastatala > ātʰēli, *hilla > iluṛɔ); the same rule hurts all 15
  other lects. **Pangwali loses r before consonants without gemination** (bárkara > bakrū).
  **Bhalesi keeps geminates after short a** only (agni > agg).
- **Aspiration** splits the group: voiced aspirates are kept in Kotgarhi (Hendriksen adds tone
  marks, which the scorer drops), Kullui, Kiunthali, Mandeali; lost non-initially in Bhalesi,
  Khashali, Kangri; lost everywhere in Kvari and Dogri, where Patyal/Brightbill Dogri devoices
  initial ones (bʰ- > p- with tone). Sibilant/h + nasal gives an *aspirated nasal* (kānʰ) in
  Chambeali, Kangri and Kochi. The "aspiration shift" onto the initial consonant visible in some
  Kochi/Kotgarhi words (divasá > dʰyaṛʰi, snāti > nʰεiṇo) did not generalise (rule rejected).
- **Kotgarhi keeps ṭṭʰ** (< ṣṭ, sth) while degeminating everything else (ʣeṭṭʰ, koṭṭʰ, hoṭṭʰ;
  −2.3×10⁻³ train NED for the exception).
- **Kotgarhi length**: Hendriksen marks vowel length almost only in monosyllables (ā in mono 40 vs
  10 short; in polysyllables 16 vs 121); length is therefore removed everywhere in `ktg`
  (keeping it in monosyllables loses more on the many suffixed citation forms).
- **Infinitives** are an isogloss in themselves: -ṇõ (ktg), -ṇo (kc), -ṇu (kiuth, WPah), -ṇū
  (bhal, bhad, khash), -ṇō̃ (jaun), -ṇā (kul, dog, cur, kgr, cam, pan). ṇ > n after r/ṛ/ḷ/ṇ
  (and in the Bhadrawah group also after ṭ ḍ) is a real dissimilation (car-nõ, gɔṇ-nõ, daḷ-nõ).
  The infinitive rewrite works best *after* intervocalic stop loss (-ati > -ai > -ṇõ), so
  stem-final stops are lost exactly as the data show (kʰā́dati > kʰā́ṇõ), and after l > ḷ
  (jválati > ʣɔḷnõ); i.e. the citation infinitive is built on the MIA stem, not on the OIA one.
  CDIAL starred roots (`*kʰuŋk`) are cited as infinitives too (`∅ > ṇõ / C _ #`, @morph), as are
  athematic -āti/-ōti verbs.
- **Learned / tatsama stratum.** Share of train pairs whose reflex is closer (by > 0.15 NED) to
  the unchanged OIA etymon than to the chain output — candidates for tatsama/semi-tatsama
  borrowing from Sanskrit/Hindi (`tmp/…/tatsama.py`):

  - ktg: 106/1803 = 5.9%  e.g. agnidʰā́na→gɛnn (pred agiṇ); áŋga→aŋgəṛ (pred aŋg); ánna→ɔnn (pred an); apūpá→pvṛɪ (pred ou)
  - kul: 30/748 = 4.0%  e.g. agnidʰā́na→gʰyānā (pred ogiāṇ); anuhāra→nuhār (pred oṇuār); ahám→hāũ (pred o); ēkaṣaṣṭi→kāhṭ (pred eśṭʰi)
  - bhal: 38/793 = 4.8%  e.g. *aḍḍ→aḍḍo (pred aṛḍnū); ántara→antra (pred antar); álpa→eipaṛo (pred alp); aṣṭātriṁśat→aṭʰεttri (pred aṭṭʰāṭ͡ḷiś)
  - kiuth: 32/680 = 4.7%  e.g. akṣōṭa→kanʰōr (pred kʰoṛ); añjalí→annal (pred ñjaḷi); apūpá→pūṛē (pred auv); kanya→kannā (pred kān)
  - WPah: 25/451 = 5.5%  e.g. agryá→agia (pred ajj); ārá→arśa (pred ār); ā́satē→āh (pred asṇu); índra→indra (pred indr)
  - jaun: 12/282 = 4.3%  e.g. átyēti→aś (pred ācē); ēraṇḍa→ēṇḍā (pred ērãḍ); kastūrī→kəṭhur (pred kātʰūrī); cákōra→cākurā (pred caōr)
  - bhad: 23/410 = 5.6%  e.g. accʰa→haccʰo (pred aʦʦʰ); áśru→ε̃kʰu (pred aśrū); áśru→ε̄ ͂kʰu (pred aśrū); āryikā→ij (pred āʣʣī)
  - dog: 28/301 = 9.3%  e.g. anidra→nī̃dra (pred anidar); anidra→nī̃drā (pred anidar); áśru→atʰrū̃ (pred asar); ástʰi→asti (pred atʰ)
  - kc: 21/314 = 6.7%  e.g. agnidʰā́na→gɛnn (pred agaṇ); *andʰikāra→nʰyāro (pred andʰar); ásti→asti (pred atʰe); ahám→hū̃ (pred o)
  - khash: 20/232 = 8.6%  e.g. áśru→antʰru (pred aśrū); áśru→ā̃tʰru (pred aśrū); áśru→εŋkʰru (pred aśrū); ásti→asũ ase (pred attʰī)
  - mand: 13/255 = 5.1%  e.g. anidra→nĩdrā (pred aṇidr); anuhāra→nuā̀r (pred aṇvār); áśru→aśrū (pred su); ástʰi→astu (pred tʰi)
  - kvar: 12/173 = 6.9%  e.g. andʰakāra→inārɔ (pred adaār); ahám→haũ (pred o); *kaṭaccʰu→karci (pred koṛaʦu); kambalá→kāmbēl (pred kabaḷ)
  - cur: 7/157 = 4.5%  e.g. ágra→aggar (pred ag); *kʰūha→kʰūhā (pred kʰū); jitá→jitṇā (pred jī); *ṭēr→ṭīr (pred ṭērnā)
  - kgr: 9/181 = 5.0%  e.g. anidra→nĩdrā (pred aṇidr); ástʰi→astʰu (pred atʰī); gáti→gaĩ (pred gṇā); dāḍima→dariūṇ (pred dāṛĩ)
  - cam: 10/147 = 6.8%  e.g. aŋgúli→ū̃gal (pred aŋguḷī); accʰa→haccʰā (pred acʰ); ēkastʰá→kiṭṭʰā (pred etʰ); *ēkkalla→killā (pred ekl)
  - pan: 2/100 = 2.0%  e.g. lōhá→luhā (pred lō); śr̩gālá→sagāl (pred śiāl)

  The diagnostic is noisy (it also catches derivatives and suffixed forms), but the recurring
  real cases are forms that keep OIA clusters the chain assimilates — anidra → nī̃drā,
  ástʰi → asti/astu, índra → indra, āntrá → ā̃tra, ántara → antra, cakrá → cakkar — i.e. a
  learned/semi-learned layer with preserved Cr and st, largest in Dogri (9%) and Khashali (9%),
  3–7% elsewhere. In the Cr-retaining lects (dog, mand, kgr, khash, pan, cam) part of this
  layer is absorbed by the Cr-retention rule itself, so the inherited-vs-learned boundary is
  not recoverable from the metric alone.

- **Residual error is morphological**: masculine -ɔ/-ā/-u and feminine -ī extensions that the
  CDIAL headword lacks (∅ → o is the top suggestion in ktg/kc, ∅ → ā in kul/cam), the causative
  -auṇõ, derivational -ṭ-, -ṛ-, -k-, and compound numerals (Kullui, Bhalesi). I did not add
  blanket "add -o/-ā" rules.
- **Dialect mixture**: `WPah` (unspecified) behaves like a blend of Kotgarhi and Kiunthali;
  `kul` Blair vs kullui-org differ in ɔ vs ə and final -ə (dialects, not only notation); Dogri
  CDIAL forms keep bʰ-, Patyal/Brightbill forms show tonal devoicing.
- **Data issues** (for the ingestion owners): reflexes contain a mangled `n/a` sequence where
  a vowel letter was (Kotgarhi gabʰīrá → `gn/arɔ`, títʰi → `tʰn/ar`, tirátē → `tn/arnõ`,
  *pratʰila → `pn/alɔ`: 18 ktg pairs; also Bhalesi, Khashali, and outside the group Kal., Sh.,
  Gaw., Dm., Phal.); Kiunthali bā́hya → `vitʰout` is the English gloss "without"
  transliterated as a reflex; Kotgarhi *bāhuṣpʰara has "reflexes" `kʰ-b`, `bʰ-pʰ`, `gʰ-pʰ`, `pʰ-p`
  (alternation notes, not words) and *dōhanīgʰaṭaka has the reflex `ṇ`; several reflexes are
  phrases (stʰitá, ā́satē → `neĩ áttʰi` "is not"); Bhalesi uses two glyphs for the lateral
  affricate (`ṭ͡ḷ` 38×, `ṭ͜l` 2×); Hendriksen's lax ʊ appears as the letter `v` (gv́ṇṭʰɔ, ʦvṛɪ),
  which the scorer cannot fold;
  etyma such as `*ga{l}lati`, `*ṭaüna` carry editorial braces/diaeresis; Kiunthali `'` (a
  tone/length mark on CāC monosyllables: kʰā'l, jā'g) is not folded by the scorer and cannot be
  predicted reliably (adding it to all CāC hurts).

## 6. New tags and engine wishes

| tag | meaning |
|---|---|
| `@x-wpah-tr` | stop + r (initial, or after a long vowel) > retroflex lateral affricate ṭ͡ḷ/ḍ͡ḷ(ʰ) |
| `@x-wpah-tr-c` | that lateral affricate > c (Kotgarhi, Kiunthali, WPah) |
| `@x-wpah-retro-dissim` | ṇ > n after r, ṛ, ḷ, ṇ (Bhadrawah group: also ṭ ḍ) |
| `@x-wpah-nd-nn` | voiced dental/retroflex stop assimilated to a preceding nasal (Bhalesi ND > nn) |
| `@x-wpah-aphaeresis` | initial a lost before a cluster |
| `@x-wpah-ph-f` | pʰ > f |
| `@x-wpah-final-lengthening` | final i u > ī ū |
| `@x-wpah-final-shortening` | final ī > i (Bhalesi) |
| `@x-wpah-final-i-e` | final i > e (Kochi) |

Engine wishes: (1) syllable/stress-aware environments ("in a monosyllable", "stressed syllable")
— Kotgarhi length, Kiunthali `'` and the pretonic block on ɔ needed awkward env/exception hacks;
(2) exceptions that can refer to the matched target (to exempt ṭṭʰ from degemination I had to
define a class `Gx`); (3) optional tokens inside targets (`(a y) a t i` silently never matches);
(4) a way to mark a rule as "morphological class" (verbs vs nouns) so that infinitive rewrites do
not fire on nominal -ati/-atē.

## 7. Round 2: vowels and the Vedic accent

Scratch for this round: `tmp/sound_changes/west-pahari/r2/` (`audit.py`, `menu2.py` + `H1/H2/H4.txt`
hypothesis menus tried at 11 anchor positions, `rnd.py`/`rnd2.py` ɔ-rounding variants, `synacc.py`
syncope-by-accent marker test, `finacc.py` apocope-by-accent test, `lenacc.py`, `batch.py`,
`table.py`). Round-1 files are `tmp/sound_changes/west-pahari/bak13/`; "round 1" numbers below
are those files rescored with today's engine (bhad/dog shift by ≤ 0.002 from the round-1 table).
No `%gold-drop` directive is used in any west-pahari file, so every change below is a change of
rules, not of scoring.

### 7.1 Per lect: round 1 → round 2

| lect | #rules | train NED | test NED | test exact |
|---|---|---|---|---|
| ktg | 49 → 51 | 0.4553 → 0.4481 | 0.4476 → 0.4395 | 9.6% → 10.0% |
| kul | 43 → 44 | 0.4616 → 0.4603 | 0.4384 → 0.4483 | 14.1% → 13.4% |
| bhal | 43 → 44 | 0.4341 → 0.4302 | 0.4569 → 0.4590 | 8.3% → 8.3% |
| kiuth | 46 → 48 | 0.4180 → 0.4136 | 0.4314 → 0.4285 | 9.8% → 10.4% |
| WPah | 42 → 44 | 0.4774 → 0.4686 | 0.4574 → 0.4493 | 8.2% → 8.2% |
| jaun | 38 → 39 | 0.4378 → 0.4318 | 0.4045 → 0.4049 | 15.0% → 15.0% |
| bhad | 39 → 40 | 0.4619 → 0.4605 | 0.5422 → 0.5455 | 5.5% → 5.5% |
| dog | 36 → 36 | 0.4678 → 0.4665 | 0.4803 → 0.4877 | 8.2% → 8.2% |
| kc | 44 → 44 | 0.4860 → 0.4746 | 0.5066 → 0.4866 | 2.3% → 5.7% |
| khash | 32 → 33 | 0.4915 → 0.4883 | 0.5265 → 0.5201 | 10.8% → 10.8% |
| mand | 35 → 35 | 0.4704 → 0.4704 | 0.5316 → 0.5316 | 7.0% → 7.0% |
| kvar | 38 → 38 | 0.4549 → 0.4519 | 0.5342 → 0.5399 | 6.8% → 6.8% |
| cur | 38 → 41 | 0.4130 → 0.3964 | 0.4382 → 0.4249 | 5.3% → 5.3% |
| kgr | 34 → 34 | 0.4916 → 0.4883 | 0.5435 → 0.5452 | 0.0% → 0.0% |
| cam | 38 → 39 | 0.3611 → 0.3576 | 0.4384 → 0.4384 | 9.1% → 9.1% |
| pan | 33 → 34 | 0.3801 → 0.3743 | 0.3700 → 0.3700 | 7.1% → 7.1% |

Train NED improves in 15/16 lects (mand unchanged); the largest gains are kc (−0.011), cur
(−0.017), WPah (−0.009), ktg (−0.007). Three lects' test NED rose; I looked at why (without
changing anything in response):
- **kul +0.010**: entirely the new final -i apocope. On test it hits feminine i-stems that carry
  many Blair/kullui reflexes, all keeping -i (ákṣi → ɔcʰi ×7 reflexes, raśmí → rɔśi ×5, nahí → nai
  ×3, muṣṭí → muṭʰi). On train the rule wins on bʰittí → bʰit, śā́nti → śānd, ṣaṣṭí → śɔṭʰ,
  *dʰūḍi → dʰuṛ but loses on gāli → gāḷi, kali → kɔḷi, buddʰi → budʰi. Kullui -i apocope is real
  but lexically split (it looked like "after a cluster", but `i > ∅ / C C _ #` is no better on
  train, −0.02×10⁻³), so its train LOO (+1.3×10⁻³) is fragile.
- **dog +0.007** (73 test pairs): the round-1 rule `V m > $1+nas / _ V` fell to LOO −0.2×10⁻³ after
  syncope was added and was removed per the LOO ≤ 0 policy; on test it was worth grā́ma → grã,
  sōmavāra → svār (+0.35 of the +0.54 summed cost). The rest is syncope over-applying to ālasa,
  *dugdʰadʰāraṇa.
- **kvar +0.006** (44 test pairs): a single item, jáṭā → ʣɔ́ṛɛ, blocked by the new "not before
  final-syllable ā" exception on ɔ-rounding.

### 7.2 Audit of the partial round-2 edits (06:39 snapshot)

All partial edits had positive or zero LOO and were kept (value = train-NED increase ×10⁻³ when
the edit is reverted):
- citation-morphology flags (`if:!numeral` on the -ati/-ayati infinitive rewrites, `if:verb` on
  `∅ > ṇV / C _ #`): ktg +2.4, bhal +1.1/+1.9, kiuth +1.0, WPah +1.5, jaun +3.8, cur +2.5/+3.5,
  kgr +1.8; 0 in bhad, dog, kc, khash, cam, pan (kept: they stop numerals from being cited as
  infinitives, harmless).
- syncope widened from a to `{a i u} > ∅ / V C _ C`: kiuth +1.2, khash +1.7, kvar +1.5,
  kgr +1.5, pan +4.0.
- `{i u} {a ā} > {ī ū} / _ #` (@hiatus: -ika/-ikā/-uka > -ī/-ū): cur +2.1, cam +3.5, pan +1.8.

### 7.3 Vowel rules adopted this round (LOO ×10⁻³ on train, fires = etyma changed)

| rule | tags | lects (LOO, fires) | examples |
|---|---|---|---|
| `{a i u} > ∅ / V C _ C` (new; lects that had no syncope) | @syncope | WPah +2.0 (46), bhad +1.5 (13), dog +2.4 (20), cur +3.4 (13) | placed late: after int. stop loss, before/around apocope |
| `i > ∅ / _ #` (WPah: `{i u} > ∅ / _ #`) | @final-vowel-loss | kul +1.3 (34), kiuth +1.4 (19), cur +5.2 (18), WPah +1.0 (23) | bʰittí > bʰit, ṣaṣṭí > śaṭṭʰ, viṁśatí > bīh |
| `{ā ī ū} > {a i u} / _ C V C* V` — long vowel shortened when two or more syllables follow | @vowel-shift | bhal +0.9 (18), kiuth +0.9 (8), jaun +2.1 (16), khash +1.5 (10) | the NIA (weight-based) pretonic shortening; see 7.4 for the accent version |
| `{a á} > o / _ C* #` before apocope — final-syllable a > ɔ | @morph | ktg +2.7 (719), kc +10.3 (142) | aŋguṣṭʰá > ɔṇṭʰo, *śaitala > śeḷo, accʰa > aʦʰo; loses on gʰōrá > gʰōr, śr̩gālá > śāḷ: the masc. direct-case -ɔ generalised to a-stems, a lexical class, hence @morph |
| `u C > $2 v / # _ ā` (after degemination) | @x-wpah-u-metathesis | ktg +1.8 (11) | uttāná > tvaṇɔ, uddʰārá > dʰvār, udvāpayati > bvauṇõ, *utkāla > kvaḷɪ, *ut-śvāsa > śváɪ |
| ɔ-rounding exception narrowed: `a > o / # (C) C _ C V \| # _ C V // _ C ā C* #` | @a-rounding | kvar (−1.45 vs the round-1 form; rule LOO +24) | a before a final-syllable ā stays a |

Removed: kc `@two-mora` (LOO fell to −1.1 once the -ɔ rule was in), dog `V m > $1+nas / _ V`
(LOO −0.2). Tried and **not** adopted because they fit ≤ 4 items: pan shortening (1 item),
pan `{ē ō} > {i u} / _ C* #` (1), dog shortening (4).

### 7.4 Accent tests (Δ train NED ×10⁻³ against the adopted chain; + = worse)

Accent rules sit before `@acc-loss`; where the late rule needed an accent condition I marked the
relevant vowel before `@acc-loss` (a > ə / pretonic, etc.) and restored it right after the target
rule. Rules on accented vowels were tried only before `@acc-loss`; all others at 11 anchors.

**1. a > ɔ in the first syllable (ktg, kc, kvar; kul has it in every initial syllable).** The raw
first-syllable table for Kotgarhi: open 68% ɔ vs closed 20%. By accent, open tonic 69%, open
pretonic 77%, open unaccented 64%: no accent effect (pretonic is if anything higher). Light vs
heavy next syllable: 75% vs 50%.
- block pretonically (`// _ !#* Acc`): ktg +1.5, kc +2.5, kvar +8.0; kul (all-initial rule) +4.7.
- only the tonic a in accented words: ktg +5.4, kc +6.1, kvar +16.7, kul +6.6.
- add tonic *closed* syllables to the open ones: ktg +3.8, kc +3.6, kvar +8.6; kul, conversely,
  loses +5.2 when closed syllables are limited to the tonic one.
- block before any heavy next syllable (long vowel or CC): ktg +2.4, kc +3.7, kvar +3.9, kul +1.0.
- block before a following ā (the round-1 exception): ktg/kc best form; limited to an
  *accented* ā́ it loses +1.5 (ktg); limited to unaccented ā it is neutral (+0.1). kvar prefers
  "before ā in the final syllable" (−1.45).
- Verdict: **ɔ is conditioned by syllable structure (open syllable, not before ā), not by the Vedic
  accent.** The "weight" effect in the raw table is the ā-exception; a general heavy-next-syllable
  condition does worse.

**2. Vowel length.** Pretonic shortening `{ā ī ū ē ō} > short / _ !#* Acc`: worse in 15/16
lects (+0.6 to +15.7); dog −4.2, but the gain is an artefact (amāvāsyā → ãs, dā́yādya → dāj via
hiatus contraction, and a verb) and the raw Dogri table shows pretonic and tonic long vowels
kept long equally (65%/65%). Post-tonic shortening +1.0 to +11.3 (dog −0.5). Shortening every
non-tonic long vowel in accented words +4.2 to +26.9 (dog −4.7, same artefact). Tonic
lengthening in open syllables +4.2 to +15.8 in all 16; in open first syllables +0.8 to +10.4;
all tonic vowels +3.7 to +28.4. By contrast the **positional** rule (long vowel shortened when
two or more syllables follow) is adopted in four lects. Length is conditioned by position
(the later NIA weight/penultimate stress), not by the Vedic accent.

**3. Syncope.** Share of medial short vowels lost (attested, train): pretonic 27–50% vs
posttonic 42–77% vs tonic 58–86% (ktg 40/58/70, kul 27/44/67, kiuth 39/77/86, bhal 35/60/60,
WPah 50/71/59). Syncope is therefore **not pretonic**; if anything pretonic vowels survive best.
Marker tests on the late syncope rule: protecting pretonic medial a is neutral (−0.0 to +1.9;
ktg −0.0, kul −0.1, kiuth +1.1, bhal +0.2, WPah +1.0, khash +0.6, kvar +1.9, kc 0.0);
protecting post-tonic medial a costs +2.3 to +25.5; protecting the tonic medial vowel +0.2 to
+5.5. An early accent-conditioned syncope (before `@acc-loss`) is worse than the late positional
one whether pretonic (+1.1 to +9.9) or post-tonic (+3.4 to +55.4). The positional rule
`V C _ C` (open non-initial syllable) already captures what the accent would add.

**4. Apocope.** Protecting a tonic final ā́ from apocope: 0 to +2.5 (16/16 no gain); tonic
final á: +3.0 to +18.9; losing a post-tonic final ā in the -ā-keeping lects (kul, cur, kgr,
cam, pan): 0 to +1.0; protecting tonic final í ú: −2.0 (pan, 1–2 items) to +4.5, 0 elsewhere.
Apocope is not accent-conditioned.

**5. Final-vowel lengthening** (`{i u} > {ī ū} / _ #`) in lects that lack it: ≤ −0.8 (kiuth,
before i-apocope was added; 0 after), 0 elsewhere; a tonic-only version gains nothing
(0 in 12 lects, +2.9/+3.4 in mand/dog). The lengthening applies to *secondary* final vowels
(-ika > -ī), after the Vedic accent is gone.

**6. Vocalic r̩.** r̩ > i stays the best single rule. ŕ̩ > a: −1.3 (kc) to +1.6; ŕ̩ > u −1.4 (cam) to
+3.4; r̩ > a in the first syllable +0.5 to +5.6 (kgr −1.1); r̩ > u after labials ≤ −0.6;
r̩ > a before a / r̩ > u before u ≥ −0.6. Kotgarhi's raw table hints at tonic ŕ̩ > a in first
syllables (36% a, n = 11, vs 9% pretonic, n = 23), too few items for a rule.

**7. Umlaut / harmony, all rejected**: a > e before final i +44 (ktg; similar elsewhere);
i > e / u > o before a: +3.6 to +27.8 in all 16; lowering i u > e o in closed first syllables
+0 to +18; ā > ɔ in open first syllables 0 to +8.7; a > i/u before i/u ≥ −0.9.

**8. Other probe leads rejected as non-phonological**: kiuth `{a á} > ∅ / _ C* #` (−3.3) only
turns participle headwords into infinitives (bʰagna → bʰāgṇu, cʰinná → cʰinṇu: CDIAL links
past participles to the verb's citation form); kgr `if:!accented u > ∅` (−3.5) and
`ā > a / _ C* u` (−4.8), mand `ā > ∅ / _ C* ā` (−3.4) fit a handful of compounds.

### 7.5 What the accent does and does not condition in West Pahari

It conditions **nothing** that the chain can use. Every West Pahari vowel change I could test —
a > ɔ, length loss and shortening, syncope, apocope, final lengthening, r̩ — is conditioned by
syllable structure or position read on the (Middle) Indo-Aryan form: open vs closed syllable,
a following ā, number of following syllables, medial vs final. Where the raw tables show an accent
correlation it goes the "wrong" way for a Vedic-stress story (pretonic vowels are the most stable
medially; pretonic open a rounds as often as tonic a). This matches the round-1 chronology: the
vowel changes are NIA (they follow intervocalic stop loss, and ɔ-rounding precedes degemination
and length loss), long after the Vedic pitch accent was lost. No adopted rule carries `@stress`.

New tag: `@x-wpah-u-metathesis` — initial u- lost with labialisation of the following consonant
before ā (Kotgarhi; Hendriksen writes the labial as `v`/ʊ).

Frontiers (`results/OIA-*.hand.frontier.csv`) were rerun at 11:36–11:38 for the 15 lects whose files
changed in round 2 (logs in `tmp/…/r2/frontier-*.log`); `mand` is unchanged since round 1.

## 9. Accent score (round 3)

Scratch: `tmp/sound_changes/west-pahari/r3/`. It holds the round-3 starting copies of the five files
(`OIA-*.sc`), the harness `h.py`, the hypothesis menu `hyps.py`, the anchor sweep `exp1.py`, the
combinations and Vedic hybrids `exp2.py`, per-item diffs `diff.py`, accent LOO `aloo.py`, the
source-marking tables `stats.py`, and `bugfix_diag.py` (see 9.4). All numbers are accent NED
(lower is better). I tuned on train only. Placement rules left plain NED unchanged, on train and
test, in all five lects (checked after every edit; `aloo.py` prints plainΔ = 0).

Anchors swept for every hypothesis:
- right after `@acc-loss`;
- `@a-rounding`;
- after `@int-stop-loss`;
- before the NIA `@hiatus`;
- before `@final-vowel-loss`;
- `@syncope`;
- `@degem-plain`;
- `@vowel-shift` (length loss);
- end of chain.

The hypothesis menu:
- **first**: first syllable.
- **first long**: first long vowel, else the first syllable.
- **first heavy**: first long vowel or first vowel before CC, else the first syllable.
- **weight**: the rightmost heavy non-final syllable, else the first (Hindi-type).
- **heavy penult**: a heavy penult, else the first syllable.
- **long penult**: a long penult, else the first syllable.
- **ā-quantity**: an ā in the second syllable takes the accent from a short first vowel;
  otherwise the first syllable.
- **penult** and **last** syllable.
- **verbs**: the infinitive stem syllable before -ṇõ.
- **tone after aspirate**: the vowel right after a surviving voiced aspirate, else the first syllable.
- **tonogenesis**: the vowel after an initial OIA voiced aspirate or h, or before the first medial
  one, else the first syllable.
- **Vedic hybrids** (`%accent vedic`): a positional rule only for the non-Vedic-accented, verb,
  non-verb or starred etyma.
- **fair Vedic**: the Vedic accent wherever its vowel survives; the first syllable when the
  accented final a/ā is lost to apocope, and in etyma without an accent.

### 9.1 Results per lect

| lect | acc. pairs tr/te | train: none / Vedic / final | **test: none / Vedic / final** | adopted |
|---|---|---|---|---|
| ktg | 804/191 | 0.5011 / 0.4970 / **0.4506** | 0.4679 / 0.4653 / **0.4120** | initial stress |
| bhal | 67/27 | 0.5097 / 0.5053 / **0.4542** | 0.5583 / 0.5543 / **0.5166** | ā-quantity + initial |
| dog | 24/10 | 0.6155 / 0.6165 / **0.5830** | 0.6180 / 0.6055 / **0.6039** | initial stress |
| khash | 30/12 | 0.5595 / 0.5620 / **0.5399** | 0.5208 / 0.5125 / **0.5292** | initial stress |
| mand | 25/5 | 0.6236 / 0.6187 / **0.5998** | 0.6800 / 0.6800 / **0.7000** | last syllable (after apocope) |

The khash and mand test sets have 12 and 5 accent pairs, so their test rises are one or two items.
They are reported as they came out, and nothing was changed because of them.

**Kotgarhi.**
- Adopted: `V > $1+stress / # C* _` at the end of the chain, `@accent-placement`. LOO **+0.0505**.
- Why: 88% of Hendriksen's marked accents are word-initial (714/804). In 4-syllable words the
  accent is more often the 2nd or 3rd syllable (24/26), but those are mostly causatives in -áuṇõ
  that the chain does not derive.
- The first-syllable rule scores the same at every anchor (0.4506–0.4507).
- Rejected: every hypothesis in the table below was worse than 0.4506, or gained less than
  0.0005 and failed a check (see the notes on the right).

| hypothesis | train accent NED | note |
|---|---|---|
| first long | 0.4577 early / 0.4529 before `@vowel-shift` / 0.4506 end | |
| first heavy | 0.4532–0.4542 / 0.4505 at end | |
| weight | 0.4526 early / 0.4518 before `@vowel-shift` / 0.4502 at end | the end-of-chain gain is 7 won vs 4 lost etyma, mostly verbs whose attested form lost its initial vowel (ʦʰvádṇõ, gʰvàṛnõ, cúśṇõ): segmental mismatch, not placement |
| heavy penult | 0.4516–0.4525 / 0.4502 at end | |
| long penult | 0.4509–0.4517 | wins on kaṭāha, kōṣṭʰāgāra, kumbʰakāra; loses on tanūbʰūta, *hr̩tkūṭa, bʰāginēya, halēṣā |
| penult | 0.4868 early … 0.4583 late | |
| penult before apocope | 0.4603 | |
| last | 0.5357 | |
| ā-quantity | 0.4516 early / 0.4504 before `@vowel-shift` | its ā rule has **LOO −0.0004** |
| verbs | 0.4505 / 0.4504 | |
| tone after aspirate | 0.4542 | |
| tonogenesis | 0.4503 | 17 etyma, mixed; most gains are words whose placed accent is deleted later |

A contraction that keeps the accent, `repeat: V a > $1` in place of `a > ∅ / V _`, leaves plain
NED identical. It helps the early placements (weight after `@int-stop-loss`: 0.4526 → 0.4511), but
none of them then beats "first", so it was not adopted.

**Bhalesi.** Three rules, placed right after `@int-stop-loss`:
- `ā > $1+stress / # C* Short C* _`, LOO **+0.0044**;
- `Long > $1+stress / # C* _`, LOO **+0.0249**;
- `Short > $1+stress / # C* _ // _ C* ā`, LOO **+0.0262**.

What the rules do: they give every word exactly one accent. An ā in the second syllable takes it
from a short first vowel; otherwise it goes on the first syllable.

Against plain first-syllable stress (0.4614) this is −0.0072:
- won: vivāhá > biā̀, aŋgāraka > aŋā́ri, pañcāśát > panʣā̀, *saṁjvāla > sanʣā́o;
- lost: *nidāgʰya > níggo, *kaŋkāla > káŋkāl.

Rejected:

| hypothesis | train accent NED |
|---|---|
| first long | 0.4568 |
| first heavy | 0.4579 |
| weight | 0.4567 |
| heavy penult / long penult | 0.4567–0.4602 |
| a long vowel after a short first vowel (not only ā) | 0.4557 |
| penult | 0.4604–0.4806 |
| last | 0.4875–0.5220 |
| verbs | 0.4614 (no change) |
| tonogenesis | 0.4592 |
| tone after aspirate | 0.4679–0.4861 |

**Dogri, Khashali, Mandeali (small).**
- Dogri and Khashali: initial stress right after `@acc-loss`. LOO +0.0324 (dog) and +0.0196
  (khash).
- Mandeali: `V > $1+stress / _ C* #` at the end of the chain. LOO +0.0237. After apocope this is
  the old penult; penult before apocope scores 0.6013.

Rejected:

| lect | hypothesis | train accent NED | note |
|---|---|---|---|
| dog | ā-quantity | 0.5794 | 1 etymon, upādʰyāya |
| dog | first long | 0.5804 | |
| dog | heavy penult | 0.5801 | |
| dog | tonogenesis | 0.5830 | same as initial |
| khash | first heavy | 0.5366 | 1 won, bíḍāla; 1 lost, paśurūpá |
| khash | ā-quantity | 0.5380 | |
| khash | tonogenesis | 0.5399 | same as initial |
| khash | penult | 0.5433–0.5537 | |
| khash | last | 0.5687 | |
| mand | first | 0.6133 | |
| mand | tonogenesis | 0.6049 | |
| mand | first long | 0.6021 | |
| mand | weight | 0.6033 | |

None of these gains is more than one etymon, so none was adopted.

### 9.2 The Vedic accent, given its best chance (train accent NED)

| model | ktg | bhal | dog | khash | mand |
|---|---|---|---|---|---|
| none | 0.5011 | 0.5097 | 0.6155 | 0.5595 | 0.6236 |
| Vedic carried | 0.4970 | 0.5053 | 0.6165 | 0.5620 | 0.6187 |
| Vedic; first syllable for etyma without an accent | 0.4694 | 0.4751 | 0.6009 | 0.5433 | 0.6116 |
| Vedic for verbs; first syllable for the rest | 0.4614 | 0.4627 | 0.5877 | 0.5399 | 0.6216 |
| Vedic for non-verbs; first syllable for verbs | 0.4863 | 0.5041 | 0.6165 | 0.5620 | 0.6104 |
| fair Vedic | 0.4559 | 0.4656 | 0.5893 | 0.5466 | 0.6143 |
| **adopted chain** | **0.4506** | **0.4542** | **0.5830** | **0.5399** | **0.5998** |

In the "Vedic for verbs" row, khash equals the adopted chain because its chain predicts the same
accent. Item by item in Kotgarhi, fair Vedic beats initial stress on 2 etyma and loses on 31.

**Conclusion.** The Vedic accent never wins, whether as a whole or for any class (verbs,
non-verbs, starred or accented etyma). Carried over, it gains at most 0.004–0.005 over no accent,
and in Dogri and Khashali it is worse than no accent. This matches §7.5: the West Pahari accent is
a new, positional one, not a continuation of the Vedic accent. No rule carries `@stress`:
- **initial stress** in Kotgarhi, Dogri and Khashali;
- **ā-quantity** in Bhalesi: an ā in the second syllable pulls the accent off a short first vowel.
  This is the context in which Kotgarhi keeps a unrounded, written ə (kaṭāha > kəṛā́);
- **final stress (the old penult)** in Mandeali.

### 9.3 How the sources mark accent

- **Kotgarhi.** The only source is CDIAL (Hendriksen). 45% of pairs are marked, evenly across
  the alphabet, on short and long vowels alike (672 and 108 marks). The grave (low tone) and caron
  appear in 41% of the pairs whose etymon has a voiced aspirate or h, and in 2% of the others.
  Pairs with a voiced aspirate or h are marked 75% of the time; the others 40%. So the
  accent-marked subset over-represents tone words. Their tone sits on the first syllable, so this
  does not change placement.
- **Bhalesi and Khashali.** Only 9–10% of pairs are marked. Grave marks cluster in words with a
  voiced aspirate or h (18–20% vs 4–11%).
- **Dogri and Mandeali (Patyal, Gill).** These sources mark tone, not stress. Words with a
  voiced aspirate or h are marked 49% and 33% of the time; the others 5% and 7%. Here the accent
  score is mostly a score of where the tone falls.
- **Notation.** No West Pahari file has a `%source` block or writes a visible accent or tone
  mark, so the notation costs nothing.

### 9.4 Engine issues (not fixed; for the coordinator)

1. **Accented diphthongs lose their accent position.** `sca._acc_vowel` strips the mark and adds
   the acute at the end of the token, so `ái`/`áu` become `a`+`í`/`ú` after `canon`. Examples:
   báɪ, bʰàu, śáuṇ.
   - Affected: 34/804 Kotgarhi accent pairs (dog 1, mand 6).
   - Size: a scratch-only patch that keeps the mark where it was moves Kotgarhi train from 0.4506
     to 0.4486 (test 0.4120 → 0.4085). Mandeali does not change, because Patyal writes the mark on
     the second letter.
   - With the patch, Kotgarhi ranks the ā-quantity rule placed before `@vowel-shift` (0.4475)
     ahead of first syllable (0.4486). The Bhalesi rule would then also fit Kotgarhi.
2. **A missing accent costs less than a wrong one.** A misplaced accent costs 1.0 (two half-errors)
   and a missing one 0.5. So a rule that leaves a word unaccented can score better than one that
   guesses. The negative LOO of Kotgarhi's ā rule comes partly from this. I adopted no rule that
   leaves words unaccented.
   - Suggestion: count a missing predicted accent as a misplacement.
3. **Deletion drops the accent.** A latent accent is lost when its vowel is deleted by a
   contraction (`a > ∅ / V _`). Rewriting the rule as `V a > $1` keeps it. The engine could
   instead move the accent to the surviving neighbour.

Frontiers were rerun for ktg, bhal, dog, khash and mand at 12:47–12:49. The CSVs now carry
`train_aned` and `test_aned`; the logs are `tmp/…/r3/frontier-*.log`.

## 10. Extension-aware score (round 4)

Scratch: `tmp/sound_changes/west-pahari/r4/`. The folder holds:
- `orig/`: the 16 files at the start of round 4;
- `x.py`: harness;
- `base2.py`: before/after tables `before2.txt` and `after2.txt`;
- `cand.py`, `cand2.py`: candidate edits;
- `pdiff.py`: plain-NED item diffs;
- `resid.py`: residual endings and extra-suffix probes (`menu2.txt`);
- `samp.py`: samples of the reflexes behind each candidate;
- `loo4.py`: LOO values (`loo4.txt`);
- `frontier-*.log`.

All numbers use the 13:29 menu (aka, akka, ikā, uka, aḍaka, allaka, araka), and the "before"
column was rescored with it.

Every change had to meet three conditions:
- train extension-aware NED improves;
- plain train NED gets no more than 0.001 worse;
- train accent NED does not get worse.

I read "do not change the accent score" as "do not make it worse". The accent-placement rules
are untouched. Train accent NED moved in only three lects, and only down, because segmental fixes
improved the forms: ktg 0.4699 → 0.4659, mand 0.6165 → 0.6099, khash 0.5769 → 0.5752.

### 10.1 Before → after (plain NED / extension-aware NED)

| lect | train plain | train ext | test plain | test ext |
|---|---|---|---|---|
| ktg | 0.4481 → 0.4456 | 0.4053 → **0.3890** | 0.4395 → 0.4353 | 0.3875 → 0.3716 |
| kul | 0.4603 → 0.4596 | 0.3873 → **0.3711** | 0.4483 → 0.4475 | 0.3851 → 0.3663 |
| bhal | 0.4302 → 0.4298 | 0.3685 → 0.3681 | 0.4590 → 0.4590 | 0.3943 → 0.3943 |
| kiuth | 0.4136 → 0.4140 | 0.3783 → **0.3432** | 0.4285 → 0.4265 | 0.3912 → 0.3551 |
| WPah | 0.4686 (unchanged) | 0.4047 | 0.4493 | 0.3962 |
| jaun | 0.4318 → 0.4303 | 0.3775 → **0.3646** | 0.4049 → 0.4049 | 0.3445 → 0.3439 |
| bhad | 0.4605 → 0.4605 | 0.4240 → **0.4178** | 0.5455 → 0.5455 | 0.4974 → 0.5055 |
| dog | 0.4665 → 0.4603 | 0.4090 → **0.3840** | 0.4877 → 0.4693 | 0.4432 → 0.3990 |
| kc | 0.4746 → 0.4679 | 0.4320 → **0.4137** | 0.4866 → 0.4963 | 0.4417 → 0.4412 |
| mand | 0.4704 → 0.4675 | 0.4229 → **0.3771** | 0.5316 → 0.5258 | 0.4900 → 0.4183 |
| khash | 0.4883 → 0.4869 | 0.4458 → **0.4374** | 0.5201 → 0.5191 | 0.4906 → 0.4634 |
| kvar | 0.4519 → 0.4479 | 0.3918 → **0.3574** | 0.5399 → 0.5399 | 0.4471 → 0.3865 |
| cur | 0.3964 (unchanged) | 0.3279 | 0.4249 | 0.3723 |
| kgr | 0.4883 → 0.4858 | 0.4207 → **0.4136** | 0.5452 → 0.5452 | 0.4604 → 0.4657 |
| cam | 0.3576 → 0.3571 | 0.2973 → 0.2968 | 0.4384 → 0.4384 | 0.3647 → 0.3647 |
| pan | 0.3743 (unchanged) | 0.3154 | 0.3700 | 0.3185 |

- Train extension-aware NED improves in 13 lects, by up to −0.046 (mand) and −0.035 (kiuth,
  kvar). Plain train NED also falls in 11 of them. Test plain NED rose only in kc.
- kiuth plain train rises +0.0004, within the 0.001 tolerance.
- kc test plain +0.010: from the -ɔ exception on test items. Nothing was changed because of it.
- bhad test ext +0.008 and kgr test ext +0.005 are small test sets.

### 10.2 Ending-choice shares (train, after)

| lect | bare | aka | akka | ikā | uka | aḍaka | allaka | araka |
|---|---|---|---|---|---|---|---|---|
| ktg | 71% | 7% | 2% | 11% | 4% | 2% | 1% | 1% |
| kul | 57% | 15% | 2% | 12% | 6% | 5% | 1% | 2% |
| bhal | 68% | 12% | 1% | 10% | 5% | 2% | 1% | 2% |
| kiuth | 66% | 18% | 1% | 4% | 4% | 3% | 2% | 2% |
| WPah | 72% | 12% | 2% | 2% | 7% | 2% | 2% | 2% |
| jaun | 64% | 13% | 8% | 9% | 4% | 2% | 0% | 0% |
| bhad | 76% | 9% | 2% | 2% | 5% | 2% | 1% | 2% |
| dog | 64% | 18% | 3% | 8% | – | 4% | 1% | 2% |
| kc | 74% | 6% | 3% | 12% | 2% | 1% | 1% | 2% |
| mand | 54% | 20% | 2% | 7% | 5% | 4% | 4% | 4% |
| khash | 73% | 9% | 3% | 7% | 3% | 3% | 1% | 1% |
| kvar | 64% | 14% | 2% | 11% | 1% | 7% | 2% | – |
| cur | 67% | 15% | 1% | 9% | 3% | 2% | – | 4% |
| kgr | 58% | 23% | 2% | 7% | 3% | 2% | 2% | 4% |
| cam | 68% | 19% | 1% | 5% | 2% | 1% | 2% | 1% |
| pan | 70% | 10% | – | 11% | 3% | 3% | 1% | 2% |

Before this round -aka was chosen far less often because several chains turned it into the wrong
ending:
- kiuth: 7% before → 18% after;
- mand: 7% → 20%;
- ktg: 3% → 7%;
- dog: 11% → 18%.

### 10.3 Rules changed

LOO values are train extension-aware NED ×10⁻³ when the rule is removed (or its new exception
reverted); a positive value means the rule helps. Plain NED ×10⁻³ is given in brackets.

**-aka > -aa > masculine ending** (`@hiatus @aka`)
- **mand**, `a a > ā / _ #` after apocope, with apocope made `{a ā} > ∅ / _ # // a _`:
  contraction LOO +42.3 (plain +2.6), exception +39.1.
- **khash**, same form: +8.3 (+0.2); exception +5.7 (+1.3).
- **kiuth**, same form, replacing the old `a a > u`: +32.9 (+1.2); exception +31.6 (+0.7).
  Kiunthali's masculine ending is -ā (25% of endings); it had been -u.
- **bhad**, same form with -o: +6.2 (0.0); exception +5.1 (0.0).
- **jaun**, `a a > o / _ #` before apocope: +11.9 (+0.5).
- **kvar**, `a a > o / _ # // V _` placed before the old `a a > u`: +33.2 (0.0). Jouanne writes
  -o. After a further vowel (-V.a.a, *tapaka > tau) the old -u outcome remains; replacing `a a > u`
  outright costs +1.45 plain on that one item.
- **dog**, a chronological rewrite of apocope:
  1. `ā > a / _ #`: MIA final shortening, jihvā > jibbʰa. LOO +13.7 (plain −2.1). Removing it
     alone breaks the chain, so its value is the package's.
  2. `a a > ā / _ #`: +21.2 (+0.9).
  3. `{a i u} > ∅ / _ #`: the contracted -ā survives.

  Also new: `%source brightbill-turner2007dogri` with `ā > a` (@notation). That source writes ā
  without the macron (panī, sona, kela); on its own the block is worth plain −4.2.

**-ikā / -uka contraction** (`@hiatus`)
- **kul**, `{i u} {a ā} > {ī ū} / _ #`: +14.6 (−0.0). Kullui had -iā.
- **kgr**, `i ā > ī / _ #`: +7.1 (+2.5).
- **mand**, `i ā > ī / _ #`: +2.0 (+0.2).
- **ktg**, `u a > ū / _ #`: +5.0 (+0.8). It had -uo.
- **kc**, `u a > ū / _ #`: +2.1 (−0.5).

**Short a absorbed before ā** (`a > ∅ / _ ā`, `@hiatus`; kumbʰakāra > kumbʰaār > kumhār)

| lect | LOO | plain |
|---|---|---|
| kul | +1.5 | +0.7 |
| bhal | +0.4 | +0.4 |
| kiuth | +1.1 | −0.4 |
| jaun | +1.0 | +1.0 |
| dog | +1.2 | +1.2 |
| kc | +0.9 | −0.2 |
| khash | +1.7 | +0.4 |
| kvar | 0.0 | +4.1 |
| cam | +0.5 | +0.5 |

It was not adopted in WPah, bhad, cur and kgr, where it made one of the two scores worse.

**The ktg/kc `@morph` rule** `{a á} > o / _ C* #` generalises the masculine -ɔ to a-stems. It now
has an exception:
- ktg: `// Long C _ | V {r ḷ l} _`; reverting it costs +11.3 (plain +1.8).
- kc: `// Long C _`; reverting it costs +15.4 (plain +6.4).

With the exception, stems in a long vowel + C (-ār, -āḷ) and ktg's -ra/-la stems keep the bare form.
The -ɔ now comes from the -aka variant instead. The rule itself stays, because plain NED needs it
(removing it costs +2.7).

**Tried and rejected:**
- -aka > -ā in dog before apocope, or with a simple apocope exception: plain +4.5 to +6.2.
- -aka > -ā/-o in jaun with an apocope exception: plain +1.4.
- kiuth `i ā > ī`: plain +1.7.
- bhal `u a > ū`: ext +6.2 worse.
- `a > ∅ / _ ā` in cur (+1.6/+1.6), WPah (+0.4 plain), bhad and kgr.
- `l l l` reduction for -allaka (bhad/khash/pan): at most −0.3.
- Further -ɔ exceptions in ktg: after clusters (plain +5.8), after any long vowel (+1.6 plain);
  after s or a nasal (≤ −0.8 ext).

### 10.4 What the choices say

- **-aka is productive everywhere** and is the main extension: 6–23% of reflexes. The masculine
  direct ending differs by lect:
  - -ā: kul, kiuth, dog, mand, khash, kgr, cur, cam, pan;
  - -o/-ɔ: ktg, kc, bhal, WPah, jaun, bhad, kvar.

  Kangri (23%), Mandeali (20%), Chambeali (19%), Kiunthali (18%) and Dogri (18%) prefer -aka
  most strongly over the bare stem.
- **-ikā** (feminine -ī/-i) is second: 7–12%. It is lower in kiuth, WPah and bhad (2–4%), where
  the chain's -i does not match an attested -ī.
- **-ḍa, -la and -ra.**
  - -aḍaka is productive in kvar (7%), kul (5%) and mand/dog (4%).
  - -araka in kgr, cur and mand (4%).
  - -allaka is really productive only in mand (4%).
  - All three are marginal (≤ 2%) in ktg, bhal, WPah, jaun, bhad and kc.
- **-akka** is chosen 1–3% (jaun 8%). The chains make it -k/-kk or -ko. On inspection the jaun
  choices are consonant-final fits (-ak for attested -au, -ḍa, -na). They are not -kk-
  derivatives.

### 10.5 Should the menu change? (evidence from `resid.py`/`samp.py`)

Each candidate below was run through the current chains as an extra variant. The count is the
number of train reflexes whose best variant would improve, and Δ is the gain in extension-aware
NED ×10⁻³.

| candidate | reflexes improved (Δ) | verdict |
|---|---|---|
| -la without -ka | ktg 84 (6.9), kul 31, bhal 26, kiuth 21, mand 15 (9.1), kvar 10 (12.7) | **surface.** In ktg it only doubles the stem consonant (masta > mattʰo, *jʰōṭṭa > ʣʰoṭṭo); in kvar it fits unrelated segments |
| -ra without -ka | ktg 67 (6.0), jaun 24, kul 26 | **surface.** Same pattern |
| -ṭa/-ḍa without -ka | ktg 38, kul 23/21 | **surface** (nīcá > niṭo for niśṭo) |
| -āla / -ālu | jaun 36 (4.8), ktg 36, kul 29 | **surface.** It mostly supplies a final -ā after the -l is dropped (mūṣa > mūśāl for mūśā) |
| -iya | kc 16 (6.8), mand 9 (6.0), ktg 10 | mixed. A few are real -iya adjectives; most are surface (-io for attested -iṇo, -ubʰi) |
| -ikka | ≤ 12 per lect (≤ 1.0), except kc 8 (2.4) | **not needed** once -ikā contracts properly |
| plain feminine -ī (final vowel → -ī) | kiuth 38 (9.6), khash 3; 0 elsewhere | **real** morphology, but for Kiunthali the right fix is in the chain (-ikā > -ī). That fix costs +1.7 plain there, so I did not adopt it |
| -ūka | bhal 14 (3.4), ktg 17 | partly real (Bhalesi -ū diminutives such as kuṭṭū, guḍḍū); bhal `u a > ū` loses overall |
| final vowel → -ā (= the consonant-final bare form in ktg/kc) | ktg 227 (18.5), kc 43 (28.3) | **not a menu item.** It shows that the ktg/kc -ɔ generalisation still applies to many stems cited bare. Under an extension-aware score a bare variant *without* the -ɔ rule would be the principled fix; the plain NED keeps it. |

**Recommendation for West Pahari:**
- Add nothing. The large counts for -la/-ra/-ṭa/-āla are surface fixes that the menu's
  "no cost" design would reward without deriving anything.
- If one item is added, plain feminine -ī (Pk -ī, the NIA feminine) is the only well-attested
  candidate, and it matters only for Kiunthali.
- **Rarely chosen:**
  - -akka (≤ 3% except jaun; mostly surface; never chosen in pan);
  - -uka in dog and kvar (0–1%);
  - -araka in kvar and jaun (0%);
  - -allaka in jaun and cur (0–1%).

  -akka should stay only if other groups need it.

Frontiers were rerun at 13:40–13:47 for the 13 changed lects (all except WPah, cur and pan). The
logs are `tmp/…/r4/frontier-*.log`.

## 11. Verb citation forms (round 5, cut short at 18:27)

Scratch: `tmp/sound_changes/west-pahari/r5/`. It holds `orig/` (the files before this round),
`before.txt`, `after.txt`, `cand.py` and `vshow.py`.

**Formation.**
- The West Pahari infinitive, the -ṇ + vowel citation form, continues the **verbal noun in
  -anaka(m)**.
- The **neuter -anakam** gives three endings:
  - -ṇõ in Kotgarhi, and -ṇō̃ in Jaunsari;
  - -ṇu in Kiunthali and WPah (and -nu in Khashali);
  - -ṇū in Bhalesi, Bhadrawahi and Churahi.
- The **masculine -anaka** gives -ṇā (Chambeali, Pangwali, Kangri, Churahi, Dogri) and -ṇo
  (Kochi, WPah).
- The chains could not derive these without the `@morph` citation rules, for two reasons: final
  -m was dropped before the -aa contraction, and nothing removed the vowel before -n- after a
  closed syllable.

**Rules added.** Both go just before `@final-cons-loss` and are conditioned only by segments;
the `@morph` citation rules are left in place.
1. `a > ∅ / V C* C _ {n ṇ} a a (m) #` (`@syncope`): cakṣanakam > cakkʰṇaam. This is the analogue
   of the central -aḍaka/-araka syncopes. Added in all 13 lects below.
2. `a a m > X / _ #` (`@hiatus @final-nasal`): -aṇaam > -ṇX. Added in these lects:

   | lect | X |
   |---|---|
   | ktg | õ |
   | jaun | ō̃ (only after a consonant, `/ C _ #`; the unrestricted rule costs plain +1.06) |
   | kiuth, WPah, khash | u |
   | bhal, bhad, cur | ū |
   | dog | ā̃ |

   In cam, pan, kgr and kc the syncope alone does the work (the neuter would merge with -ṇā/-ṇo).

Every change met the round-5 conditions: plain train NED ±0.0000; `--ext` better in all 13; train
accent NED unchanged.

**Verb NED: plain / verb menu, before → after** (the plain score is unchanged by construction).

| lect | train verb pairs | plain (train/test) | train menu | test menu |
|---|---|---|---|---|
| ktg | 469 | 0.357 / 0.362 | 0.465 → **0.310** | 0.474 → **0.301** |
| kiuth | 163 | 0.281 / 0.244 | 0.391 → **0.247** | 0.388 → **0.221** |
| bhal | 130 | 0.358 / 0.347 | 0.473 → **0.324** | 0.444 → **0.290** |
| WPah | 100 | 0.425 / 0.467 | 0.412 → **0.332** | 0.387 → **0.290** |
| kc | 66 | 0.400 / 0.404 | 0.366 → **0.350** | 0.410 → **0.389** |
| jaun | 61 | 0.464 / 0.588 | 0.524 → **0.378** | 0.575 → **0.467** |
| bhad | 59 | 0.392 / 0.516 | 0.397 → **0.311** | 0.493 → **0.438** |
| cam | 50 | 0.269 / 0.288 | 0.243 → **0.232** | 0.279 → 0.292 |
| khash | 41 | 0.494 / 0.558 | 0.425 → **0.383** | 0.513 → **0.457** |
| cur | 35 | 0.272 / 0.393 | 0.273 → **0.191** | 0.374 → **0.333** |
| dog | 18 | 0.539 / 0.728 | 0.398 → **0.392** | 0.524 → **0.503** |
| pan | 17 | 0.297 / 0.427 | 0.360 → **0.327** | 0.371 → **0.323** |
| kgr | 9 | 0.450 / 0.875 | 0.445 → **0.414** | 0.857 → 0.857 |

The verb-menu NED now beats the plain citation rules in Kotgarhi, Kiunthali and Bhalesi, the
three lects the coordinator flagged.

**Not done:**
- **Lects:** kul, mand and kvar have 0–6 verb pairs.
- **Formations not modelled:** the Bhadrawahi -aṇ forms from -aṇa.
- **Checks not run:** leave-one-out values for the two new rules, and a test of retiring the
  `@morph` citation rules now that the regular rules cover them.

## Round 6 audit

Scratch: `tmp/sound_changes/west-pahari/r6/`:
- `orig/`: the 16 files before this round;
- `<stem>.txt`: `residuals.py` reports;
- `h.py`, `t.py`: per-candidate ext/plain/verb/acc deltas with the words each candidate makes
  better or worse;
- `bat.py`: the hiatus battery;
- `wb.py`, `wadd.py`: build a file and run `gates.py`;
- `log.txt`: the running log;
- `final_gates.txt`: `gates.py` on the original file vs the installed one, per lect.

All 16 lects were audited: largest first, then a second pass on the six largest.

### Error profile (shared by all 16 lects)

| class | share of error |
|---|---|
| **ending** (last two segments) | 38–53% |
| consonant inserted/deleted | 15–25% |
| consonant quality | 8–12% |
| vowel quality | 4–10% |
| syncope missing | 3–12% |
| uncontracted vowel sequences | 1–3% |

Most of the ending errors are **source or lexical variation**, which I left alone:
- Blair's Kullui final -ə and stress marks (ˈ), which the `%source` block already covers in part;
- Kiunthali `'` after a long vowel before a final C (kʰā'r, kā'n; 52 CDIAL forms, sporadic). As
  a rule (`∅ > ' / # C* Long _ C #`) it scores ext +1.4 and plain +16.0, so rejected;
- Kotgarhi's sporadic length marks (kʰā́r vs kʰar). Keeping ā in final closed syllables gives
  ext −5 but plain +4.3 and accent +7.9, so rejected;
- gender/extension choices the menu cannot see (-ā/-ī/-u added to a bare stem).

### Systematic errors fixed: hiatus leftovers after consonant loss

The one systematic error class common to all lects is vowel sequences left behind after
intervocalic consonant loss. Every rule was tested one at a time, then gated in combination
(`gates.py` ACCEPT for every lect).

The rules, each tagged `@hiatus` unless noted:

| rule | effect |
|---|---|
| `V $1 > $1` | identical vowels merge |
| `{a ā} ā > ā` | aā/āā > ā |
| `a i > e / _ C` | medial ai > e |
| `a u > o / _ C` | medial au > o |
| `o i > o / _ C` | medial oi > o |
| `o u > o / _ C` | medial ou > o |
| `a e > e / _ C` | medial ae > e |
| `a o > o / _ #` | final -ao > -o |
| `a ī > ī / _ #` | final -aī > -ī |
| `a ū > ū / _ #` | final -aū > -ū |
| `i u > u / _ #` | final -iu > -u |
| `i o > o / _ #` | final -io > -o (-iyaka) |
| `{o a} ā > ā / _ #` | final -oā/-aā > -ā |
| `{a i} e > e / _ #` | final -ae/-ie > -e (kc) |
| `a i > ī / _ #` | final -ai > -ī (kiuth) |
| `C > ∅ / _ $1 $1` | a consonant + geminate becomes the geminate: tatkṣaṇa > takkkʰ- > takkʰ-, -allaka lll > ll (`@cluster-simplify`) |

Which rules each lect received:

| lect | rules added (all end-of-chain, before any `%source` block) |
|---|---|
| ktg | V$1, ai>e/_C, oi>o/_C, ou>o/_C |
| kul | V$1/_C (medial only: the unrestricted rule costs verb +14 on 5 pairs), {o a}ā>ā/_#, iu>u/_#, {a ā}ā>ā, au>o/_C |
| bhal | V$1, {a ā}ā>ā, ai>e/_C, ao>o/_#, aū>ū/_#, ae>e/_C, iu>u/_# |
| kiuth | V$1, {a ā}ā>ā, iu>u/_#, ai>ī/_#, {o a}ā>ā/_#, ae>e/_C |
| jaun | V$1, ao>o/_#, aī>ī/_#, aū>ū/_#, io>o/_# |
| WPah | V$1, iu>u/_#, ae>e/_C, io>o/_# |
| bhad | V$1, ao>o/_#, C>∅/_$1$1 |
| kc | V$1, iu>u/_#, ao>o/_#, {a i}e>e/_# |
| dog | V$1, {a ā}ā>ā |
| mand | {a ā}ā>ā, oi>o/_C, V$1/_C |
| khash | V$1, ai>e/_C, ou>o/_C, aī>ī/_#, C>∅/_$1$1 |
| kvar | au>o/_C, iu>u/_# |
| kgr | V$1, {a ā}ā>ā |
| cur | aū>ū/_# |
| cam | V$1, {a ā}ā>ā, au>o/_C, aī>ī/_#, aū>ū/_# |
| pan | V$1, {a ā}ā>ā, aī>ī/_#, C>∅/_$1$1 |

### Before → after (installed file vs start of round; train / test)

| lect | ext | plain | verb | accent |
|---|---|---|---|---|
| ktg | .3662→.3603 / .3471→.3413 | .4457→.4427 / .4349→.4318 | .2820→.2796 / .2717→.2701 | .4659→.4588 / .4110→.4099 |
| kul | .3696→.3669 / .3651→.3629 | .4596→.4594 / .4475→.4463 | = | .6646→.6229 / .8875→.7625 |
| bhal | .3546→.3518 / .3797→.3739 | .4299→.4277 / .4590→.4548 | .3082→.3043 / .2885→.2771 | .4712→.4687 / .5507→.5427 |
| kiuth | .3269→.3225 / .3449→.3430 | .4140→.4122 / .4267→.4259 | .2164→.2156 / = | = |
| jaun | .3535→.3470 / .3344→.3314 | .4303→.4311 / = | = / .4611→.4532 | = |
| WPah | .3825→.3810 / = (.3380) | .4690→.4689 / = | .3265→.3270 (+0.0005) / = | = |
| bhad | .4032→.4006 / .4865→.4838 | .4605→.4595 / .5451→.5424 | = | .6658→.6640 / = |
| kc | .3901→.3774 / .4311→.4274 | .4679→.4642 / .4963→.4952 | .3240→.3229 / = | = |
| dog | .3713→.3698 / = (.3747) | .4603→.4589 / = | = | = |
| mand | .3703→.3672 / = (.4174) | .4675→.4648 / = | = | = |
| khash | .4156→.4101 / .4318→.4272 | .4869→.4845 / .5191→.5165 | .3801→.3740 / .4569→.4485 | .5752→.5719 / = |
| kvar | .3570→.3555 / = (.3846) | .4479→.4475 / = | – | = |
| kgr | .4079→.4046 / .4499→.4490 | .4858→.4839 / .5452→.5423 | = | .7059→.7013 / = |
| cur | .3090→.3086 / = (.3503) | = | = | – |
| cam | .2775→.2723 / .3613→.3479 | .3571→.3541 / .4384→.4250 | = | = |
| pan | .3173→.3054 / .2887→.2836 | .3743→.3658 / .3700→.3768 | = | – |

"=" means unchanged. Exact-match rates rose or held in every lect; ext-exact rose in 12 of 16.

Test plain NED rose in pan (+0.0068, 14 test pairs). Ext test still improves there, and the change
is one or two words, so I left it.

### Candidates tested and rejected (numbers are train deltas ×10⁻³)

**Syncope reorder** (the Gujarati fix) in bhad and dog, the two chains with syncope before
apocope: 0.00 change. Our syncope is `V C _ C` (no following vowel is required), so the order
does not matter.

**ktg syncope variants:**
- `V C _ C V` before apocope: ext +10.8, verb +38. The final-closed-syllable syncope is what
  derives -Cṇõ.
- Syncope after degemination: ext +10.8.
- `V C (C) _ C`: ext +5.3.

**ktg geminate retention.** Hendriksen writes geminates in about 620 forms.
- Keeping geminates after a short vowel: ext +10, 77 words better and 210 worse.
- After a only: ext +3.5, 47 better and 111 worse.

This is lexical/transcriptional, so rejected. Deleting `ṭ ṭ > ṭ` also loses: ext +1.6.

**kul tr > c.** Blair and kullui-org show kʰātra > kʰāc, trúṭyati > cuṭṇā, but also
triṁśat > tih, śvaitra > śetā. Three variants scored ext +0.4 to +0.9 (≈ 10 words better vs 10
worse). It is a dialect split, so rejected.

**Other bhal candidates:**
- `n n > n / _ #`: plain +3.0; Bhalesi keeps final nn.
- `l > ∅ / ā _ {# i}`: verb +1.0.
- `a i > ī / _ #`: test accent +0.0028.

**dog aphaeresis** `a > ∅ / # _ C (C) V C* V` (akṣōṭa > kʰoṛ, anuhāra > nùār; 9 better vs 4
worse): train ext −2.4, plain −2.3. Rejected on **test accent +0.008** (10 test accent pairs),
also with the stress placement moved after it. It is worth revisiting if the accent gate is
relaxed for single-item test sets.

**Other hiatus rules, per lect:** `a u > o / _ C` in ktg, jaun, WPah, bhad and khash (ext +0.3 to
+2.1); `a i > e / _ C` in bhad, jaun, kc and mand (ext +0.3 to +3.0).

### What remains
- **Endings (≈ 45%):** source notation and the bare/extended gender choice.
- **Consonant insertions:** compounds and prefixed forms whose attested reflex drops or adds a
  morpheme (pardādā, ʦʰvádṇõ).
- **Kotgarhi geminates and length marks:** sporadic in the source.
- **Initial a- loss (Dogri, Kotgarhi, Kochi aphaeresis):** a real pattern, but it cannot yet pass
  the accent gate.

### Round 6 second pass (non-ending classes)

**Method.** `r6/ops.py <stem>` collects every non-final edit operation in the best-variant
alignment of each missed pair: substitution, deletion or insertion, each with its left and right
neighbours. It prints them by frequency with examples, so conditioning can be read directly. I
read the top 25–35 operations, with examples, for dog, kgr, WPah, cam, kul, jaun and kc, then
ktg, bhal and kiuth. Each candidate was diffed word by word (`t.py`), then gated (`gates.py`).
`%source` edits were gated directly, because the harness does not rebuild source tails.

**Kept (all `gates.py` ACCEPT):**

1. **`v > u / V _ C`** (`@v-vocalisation`): postvocalic v before a consonant vocalises. This
   fixes the causative -āpayati > -āvṇ- > -auṇ- and vāmaná > bauṇ.

   | lect | train ext | train verb | test verb |
   |---|---|---|---|
   | ktg | .3603→.3538 (72 words better, 7 worse) | .2796→.2563 | .2717→.2513 |
   | kc | .3741→.3725 | .3229→.3155 | .3742→.3521 |
   | khash | .4101→.4098 | | |

   - **jaun: rejected.** Test verb rose +.016.
   - **kiuth: worse.** Kiunthali keeps -āvṇu, so the rule was not added.
2. **`a > ∅ / V C C _ C V`** (`@syncope`): medial a is also lost after a *closed* syllable when a
   vowel follows (antarāla > antrā́l, *kʰōttarūpa > kʰōtrū, *cʰōkkara > cʰokru). The existing
   `V C _ C` syncope needs a light syllable before it.
   - Without the following-vowel condition the rule over-syncopates final closed syllables
     (gōrvara > *gobr, kaŋkaṇa > *kaŋkṇ; 35 better vs 20 worse), so the condition is necessary.
   - **Kept** in bhal (test ext .3739→.3689), kiuth (.3223→.3188), khash (.4098→.4071), kgr
     (.4024→.3985), cur, pan and WPah.
   - **Rejected** in kul and mand (test ext +.0001 and +.0033), kc (verb +1.9×10⁻³), ktg (accent
     +3.3×10⁻³) and cam.
3. **`{a ā} > ∅ / # _ h V`** (`@aphaeresis`): initial a-/ā- is lost before an h that continues an
   OIA voiced aspirate (ākʰēṭa > āheṛā > heṛā, *adʰaōṣṭʰa > hoṭʰ, *abʰilōḍati > hiḷoḷṇā). Zero or
   one word worse in every lect.

   | lect | train ext | train plain |
   |---|---|---|
   | cam | .2723→.2679 | .3541→.3493 |
   | kgr | .4046→.4024 | .4839→.4814 |
   | kiuth | .3225→.3223 | |
4. **kc aphaeresis before a cluster.** Two rules:
   - `a > ∅ / # _ C C V C* V` (aŋguṣṭʰá > guṭʰo, akṣōṭa > kʰoṛ; 5 better vs 0 worse);
   - `{ŋ ñ ṇ n m} > ∅ / # _ C`, which drops the nasal the first rule leaves word-initially.

   Train ext .3774→.3741; test ext .4274→.4228.
5. **jaun `%source john2008jaunsari`: `ṛ > ḍ`** (`@notation`). John writes ḍ for the flap
   (lakaḍi, kukaḍi, kapaḍa, kuraḍi: 46 forms with ḍ vs 9 with ṛ). Train plain .4311→.4260, ext
   .3470→.3430; test ext .3314→.3297.

**Tested and rejected (lexical, dialectal or source variation):**

*Aphaeresis elsewhere*
- dog aphaeresis: fails on test accent even with the stress placed after it; the one test item is
  ardʰarātra.
- kgr general aphaeresis (`{a ā u} > ∅ / # _ C (C) V C* V`): accent +19×10⁻³; it also produced
  apakṣá > *bakʰā.
- The pre-cluster aphaeresis in the other 13 lects:
  - kul: test ext +.0036, also with the nasal rule;
  - ktg: 13 better vs 10 worse; with u- added, 17 vs 26;
  - bhal: 1 vs 15;
  - cam and cur: worse.

*Dialect mixtures*
- **WPah a-rounding** (`use a-rounding.cccvcv.except`). Train accent −0.112, but train ext only
  21 vs 20 words and **test ext +.019**. CDIAL's "WPah." mixes rounding and non-rounding
  dialects; the accent-marked forms come from the rounding ones.
- **WPah affrication** c/j > ʦ/ʣ: 28 vs 29 words, the same dialect mixture.

*Kotgarhi and Kiunthali*
- **ktg v / geminates / length** (see the first pass).
- **ktg `u > ∅ / # _ C C V`**: 4 better vs 16 worse (Kotgarhi keeps ubəḷnõ, ukəḷnõ).
- **kiuth closed-syllable lengthening** restricted to disyllables: 30 better vs 52 worse.

*Other lects*
- **kgr `s > h / V _ V`**: 2 better vs 3 worse.
- **dog gemination before r** (pā́tra > pāttar): 3 better vs 10 worse; accent +11×10⁻³.

**Source notation:** 99 Kullui pairs from Blair carry the IPA stress mark ˈ as a separate
segment (79 word-initial). `gold_tokens` does not strip it, so it scores as a missing segment
(59 of the top "∅→ˈ" operations). This is an engine issue: ˈ should be dropped with the other
stress marks, or read as accent. I did not add a notation rule to fake it.

**Cumulative round 6 (start of round → now; train / test ext):**

| lect | ext train | ext test | lect | ext train | ext test |
|---|---|---|---|---|---|
| ktg | .3662→.3538 | .3471→.3374 | bhad | .4032→.4006 | .4865→.4838 |
| kul | .3696→.3669 | .3651→.3629 | kc | .3901→.3725 | .4311→.4193 |
| bhal | .3546→.3512 | .3797→.3689 | dog | .3713→.3698 | = |
| kiuth | .3269→.3188 | .3449→.3409 | mand | .3703→.3672 | = |
| jaun | .3535→.3430 | .3344→.3297 | khash | .4156→.4071 | .4318→.4237 |
| WPah | .3825→.3806 | = | kvar | .3570→.3555 | = |
| kgr | .4079→.3985 | .4499→.4465 | cur | .3090→.3078 | .3503→.3490 |
| cam | .2775→.2679 | .3613→.3479 | pan | .3173→.3042 | .2887→.2836 |

`gates.py`, run on each original file against its installed file, gives ACCEPT for all 16
(`r6/final_gates2.txt`). Plain train NED is lower or unchanged everywhere except WPah
(+0.0002).

Test plain NED rose only in pan (+0.0068, 14 test pairs); train plain and test ext both improve
there.

### Round 6 third pass (verbs, near-misses, rule audit)

Snapshot: the files at the start of this pass are in `r6/p3start/`. New tools:
- `near.py`: near-misses (NED ≤ 0.25) that differ by a single operation, grouped with their
  context;
- `vmiss.py`: verb misses, showing the winning menu form against the attested citation;
- `rloo.py`: per-rule LOO on train ext/plain/verb/accent.

Kullui was re-baselined after the engine fix that strips ˈ ˌ: train ext 0.3669 → 0.3581 with no
rule change.

**Coordinator's `%verb-menu +iyanakam`** (passive -iya- + -anakam, placebo-tested by the
coordinator). Added near the top of the two files and confirmed with `gates.py`:

| lect | train verb | test verb | train ext | test ext |
|---|---|---|---|---|
| bhad | .2957→.2913 | .3978→.3884 | .3996→.3990 | .4801→.4785 |
| kc | .3150→.2918 | .3419→.3228 | .3724→.3676 | .4176→.4143 |

**Verbs.** In every lect with verbs the winning formation is the one derived in round 5 (-anakam /
-āpanakam; ktg 295 + 75 of 469 train verb pairs). The remaining verb misses are stem-internal:
- lexical vowel choices (ktg ʦakʰ- ~ ʦákʰ-, kiuth kʰāṭṇu ~ kʰaṭṇu);
- a ṇ–n dissimilation that is only sporadic (kiuth gaṇnu, kaṇṇu);
- sources' ʦ/ʣ spellings (WPah).

Every change that would help verbs has already been made: v > u (ktg, kc, khash) and dropping
the Bhadrawahi syncope (below).

**Near-misses:**
- **Kullui:** o→ā before r+C (kʰarpara > kʰāprā) vs o kept (kárman > kɔm, gárbʰa > gɔbʰi).
  - `a > ā / # (C)(C) _ r C`: 7 better vs 14 worse.
  - Rounding exception before clusters: 17 vs 58.
  - `k ś > cʰ / # _` (kṣurá > cʰuri): 2 vs 2.

  All three are lexical splits, so rejected.
- **Kotgarhi:** Hendriksen's **v for short u before a retroflex** (gvṛnõ, ʦvṛɪ, mv̄ḷ; 41 written v
  vs 7 written u before ṛ ṭ ṇ ḷ ḍ; u elsewhere).
  - `u > v / C _ {ṛ ṭ ṭʰ ṇ ḷ ḍ ḍʰ ṛʰ}`: ext −5.0, plain −4.9, verb −5.6; 41 better vs 6 worse.
  - **Rejected on accent** (+2.5×10⁻³). A stressed v́ is a consonant token, so its accent is
    invisible, and accented forms write the vowel as ó/ú instead.
  - Excluding clusters still costs accent +1.0.
  - **Engine suggestion:** read Hendriksen's `v` after a consonant and before a consonant as the
    vowel ʊ → u. The rule would then be a pure notation fix.

**Rule audit (`rloo.py`, all 16 lects).** Rules whose removal does not hurt train ext were gated
one at a time.

Kept removals:

| lect | rule removed | train | test |
|---|---|---|---|
| bhad | `use syncope.aiu.vcc` | ext .3990→.3980, verb .2913→.2741, plain +.0017 | ext .4785→.4720, verb .3884→.3537, plain −.0033 |
| pan | `use s-h.ss.postvocalic` | ext .3042→.3020 | = |
| WPah | the round-6 `i u > u / _ #` | ext .3777→.3767 | |
| cam | `use hiatus.a.postvocalic` | ext .2679→.2676 | |

- bhad: Bhadrawahi keeps the medial vowel (maraṇ, the -aṇ infinitives). Both lines are commented
  out, not deleted.
- WPah: the round-6 rule is redundant once the syncope is in.
- cam: covered by the round-6 V$1 and aā rules.

Rejected removals:
- **cur `nasal-voicing`:** test ext +.0088.
- **mand `assim-sonorant.stopyvlnas`:** test ext +.0116.
- **kiuth `aphaeresis.aa.cc`:** test verb +.0054.
- **kgr round-6 aā rule:** test ext +.0009.
- **pan `final-lengthening`:** gate ACCEPT, but test plain +.018, so rejected by judgement.
- **bhad `palatal-Cy.ryy`:** ext −.0004 is not enough to drop a regular change.
- **khash `syncope.a.vccnnaam`:** ext −.0001, same reason.

Redundant rules with zero effect were left in place: `use hiatus.a` in dog, kul, bhal, kiuth and
kc, and the overlap between V$1 and aā in dog and cam. The gate requires a strict improvement to
change anything.

**Cumulative, round 6 start → end.** The current engine was used for both files: IPA-stress fix
and the updated menus.

| lect | ext train | ext test | verb train | plain train |
|---|---|---|---|---|
| ktg | .3645→.3522 | .3456→.3359 | .2757→.2498 | .4457→.4422 |
| kul | .3608→.3581 | .3511→.3488 | = | .4513→.4511 |
| bhal | .3524→.3491 | .3797→.3689 | .2963→.2924 | .4297→.4276 |
| kiuth | .3254→.3173 | .3426→.3386 | .2102→.2041 | .4140→.4127 |
| jaun | .3527→.3422 | .3328→.3281 | = (test .4389→.4310) | .4303→.4260 |
| WPah | .3796→.3767 | = | .3135→.3132 | .4690→.4688 |
| bhad | .4023→.3980 | .4829→.4720 | .2957→.2741 | .4605→.4613 |
| kc | .3900→.3676 | .4295→.4143 | .3235→.2918 | .4679→.4601 |
| dog | .3713→.3698 | = | = | .4603→.4589 |
| mand | .3703→.3672 | = | = | .4675→.4648 |
| khash | .4141→.4056 | .4296→.4215 | .3740→.3658 | .4865→.4839 |
| kvar | .3570→.3555 | = | – | .4479→.4475 |
| kgr | .4079→.3985 | .4499→.4465 | = | .4858→.4798 |
| cur | .3080→.3067 | .3466→.3453 | = | = |
| cam | .2775→.2676 | .3613→.3479 | .2163→.2141 | .3571→.3493 |
| pan | .3173→.3020 | .3006→.2955 | = | .3743→.3663 |

`gates.py`, on each original file against its final file, gives ACCEPT for all 16
(`r6/final_gates3.txt`).

### Round 6 fourth pass (khash, bhad, kgr, WPah, dog, mand)

Snapshot: `r6/p4start/`. For each of the six lects I read the near-misses (`near.py`: one
differing segment, NED ≤ 0.25) and the full non-final operation profile (`ops.py`). Each
candidate was tested in all six lects, plus kul and cam where relevant, to find changes that pass
in several neighbours at once.

**Near-misses are few and scattered.** Across these six lects no segment pair occurs more than 11
times, and most occur 2–3 times. The residual is lexical, dialectal or source variation (see the
rejected list).

**Kept (`gates.py` ACCEPT):**
- **mand `ŋ > n / # _`** (`@cluster-simplify`): a word-initial ŋ left by aphaeresis becomes n
  (áŋgāra > ŋgār → ngār). Train ext .3672→.3654, plain .4648→.4631; test unchanged. It has no
  effect in kgr, dog, khash, bhad or WPah.

**Tested and rejected:**

*Degemination (bhad, khash).* Both chains keep geminates, and the attested forms vary
(mukkʰā̃ ~ mukʰ, kuṭṭʰal ~ kuṭʰal). Train ext ×10⁻³:
- general degemination: bhad +8.1 (57 better vs 46 worse); khash +5.7;
- final only: bhad +0.8 (plain +12); khash −2.3 but plain +14.8;
- after a long vowel: bhad −0.3 but accent +4.2.

*Other bhad candidates.*
- `ś > h / V _ (C) V`: ext −2.2 but plain +2.1 (over the 0.002 limit). The gains are the -dr̩śa
  pronominals (ētādŕ̩śa > ērhū, īdŕ̩śa > ìru, kīdr̩śa > kērhū); the losses are daśamá > daśãũ and
  úpaviśati > bēśaṇ. This is a narrow lexical set, not a regular change. In khash and WPah the rule
  scores ext +1.6.
- `C > ∅ / C _ C C` (ḍḍdʰ): one word, so not adopted.

*Across the group.*
- `∅ > b / m _ {l ḷ r}` (nirmala > nimbaḷ): ext worse in mand, dog, bhad and WPah. In kul train
  improves by 0.5, but test ext is +.0005, so rejected.
- `p > u / a _ {k ś ṣ}` (apakṣá > aukʰā): no effect. The p is already gone by that stage.

*WPah.* The residual is again the rounding / affrication dialect mixture (round 6 second pass). It
is not source-separable within CDIAL's "WPah.".

*dog.* The residual is aphaeresis (rejected on test accent in the second pass) and
gemination before r (pā́tra > pāttar; rejected).

**What remains in these six lects.** Mostly endings and lexical variation:
- source extensions (-ī, -ā, -u, -õ added to a bare stem: bhad amlu, śillõ);
- dialect-mixed rounding and affrication in WPah;
- sporadic geminates in bhad and khash.

The only change this pass is mand ext .3672→.3654; the other five lects' numbers stand as in the
third-pass table.

### Round 6: re-gate under the extension-aware accent (ext_acc)

Both candidates that had been rejected only on accent were re-gated with the new `gates.py`, which
scores accent on the variant the extension-aware score picks. **Both still fail; nothing
installed.**

**dog initial a- loss** (`a > ∅ / # _ C (C) V C* V`):
- Train: ext −.0024, plain −.0023, ext_acc +.0029.
- Test: ext_acc +.0062.
- With the stress placement moved after it, train ext_acc improves (.5161→.5138), but test
  ext_acc is still +.0062. The failing item is the single test word ardʰarātra > drā̀ttā.

**ktg Hendriksen v before a retroflex** (`u > v / C _ {ṛ ṭ ṭʰ ṇ ḷ ḍ ḍʰ ṛʰ}`):
- Train: ext −.0050, plain −.0049, verb −.0056, ext_acc +.0023.
- Test: ext_acc +.0016.
- The result is the same when the rule is placed before the stress rule. The written v is a
  consonant token, so an accent on it (v́) is invisible to the score. Only an engine reading of
  post-consonantal v as the vowel ʊ would let this notation rule pass.

## Round 7

Scratch: `tmp/sound_changes/west-pahari/r7/`:
- `snap/`: the 16 files at the start of round 7 (the same as `tmp/sound_changes/snapshots/pre-round7/rules/`);
- `start/`: the files as the interrupted agent left them;
- `vq.py`: predicted→attested vowel table for the best ext variant, by position (initial, medial or final syllable) and by the next vowel;
- `vex.py`: examples for one cell of that table;
- `gv.py`: attested vowel inventory by position and source;
- `g.py`: build a candidate from ops and run `gates.py`;
- `ed.py`, `ad.py`, `pd.py`, `vd.py`: per-pair ext, accent, plain and verb diffs;
- `log.txt`: the running log;
- `final_gates*.txt`: `gates.py` on each snapshot file against the installed file.

All numbers use the 2026-09-29 engine: lax vowels are distinct, marks are dropped from plain/ext, and
the sca.py fixes for split-off marks and accented Greek ε are in.

### Step 0: the interrupted round-7 edits

Only three files differed from the snapshot. Each had one line added after `@acc-loss`: initial
stress `V > $1+stress / # C* _` in WPah, bhad and kgr. Re-gated snapshot → current, all three are
**ACCEPT (prosody-only)**: the segmental scores are identical, and ext_acc and ext_tone improve on
both train and test.

| lect | ext_acc train | ext_acc test |
|---|---|---|
| WPah | .6079→.5646 | .5694→.5139 |
| bhad | .6195→.5647 | .5714→.4911 |
| kgr | .6090→.5659 | .4944→.4778 |

All three were kept. The stale "gates.py reports REJECT" comment was corrected, since `gates.py`
now has a prosody-only verdict.

### Focus 0: lax vowels (ə ɛ ɔ ɪ are scored as distinct vowels)

**Where the lax vowels are.** Share of attested forms with a lax vowel:

| lect | forms with a lax vowel | source |
|---|---|---|
| ktg | 1,244 of 2,232 | CDIAL = Hendriksen |
| kul | 455 of 1,521 | kullui-org ɔ; Blair ə |
| bhal | 139 of 999 | CDIAL |
| kc | 118 of 401 | CDIAL |
| WPah | 88 of 548 | CDIAL |
| kvar | 86 of 217 | Jouanne |
| jaun | 61 | John and CDIAL |
| bhad | 45 | CDIAL |
| khash | 41 | CDIAL |
| dog | 30 | mixed |
| kiuth, cur, kgr, cam, pan, mand | ≤ 14 each | |

`vq.py` tabulates predicted vowel → attested vowel by position and by the next vowel. `vex.py` then
lists the words in each cell. This separated regular outcomes from lexical and dialect scatter.

**Kotgarhi (installed; all rules have a mechanism comment in the file).**

1. **a-rounding writes ɔ.** The shared `a-rounding.cccvcv.except` was replaced by a local copy that
   writes ɔ, Hendriksen's vowel (añjalí > ɔndəḷ, *maggara > mɔgrɔ, ánna > ɔnn).
2. **Pretonic reduction** `{a i u} > ə / # C* _ C C* Long`, placed before the length merger. A
   short vowel in the first syllable is reduced before a long vowel, which took the stress:
   kaṭāha > kəṛā́, *ḍakkāra > ḍəkār, parā́ri > pərār, and every causative -āpayati (*karmāpayati >
   kəmauṇõ). An open-syllable-only variant scored worse (ext −1.9 vs −2.6 ×10⁻³). Restricting it
   to a costs ext +1.0 but gains accent 0.8; train ext decides.
3. **Final -o > -ɔ** (437 attested -ɔ vs 55 -o: *ālla > allɔ, karpaṭa > kapṛɔ).
   - The -o cases are the -ṇo infinitive and a few nouns.
   - An exception after n/ṇ, which would protect the infinitive, is a morphology proxy. The plain
     rule also scored better once the rounding rule was in (ext −61.6 vs −61.0 ×10⁻³, verb −4.2),
     so the plain rule was kept.
4. **Final -i > -ɪ** (298 attested -ɪ vs 62 -i: pippalī́ > pɪpḷɪ, kaŋkaṇa > kaŋgṇɪ).
5. **Closed-syllable reduction** `a > ə / V C C* _ C C | V C C* _ C #`. The unstressed a of a
   closed non-initial syllable becomes ə: akṣára > ákkʰər, kajjala > kaʣəḷ, úttarati > uttərnõ.
6. **Removed the round-2 `@morph` rule `{a á} > o / _ C* #`**, which generalised the masculine -ɔ
   to bare a-stems. It is commented out, not deleted.
   - Once -ɔ ≠ -o, the rule turns bare stems (ánna > an, kʰaśa > kʰɔ́ś) into -ɔ forms, and the
     -aka menu variant already supplies -ɔ where it is attested.
   - `gates.py`: ACCEPT-IF-JUSTIFIED. Train ext .3757→.3622, test .3561→.3451; 171 reflexes
     better, 16 worse.
   - Plain train +.0051 is the bare-vs-extended artifact: 173 of the plain losses are attested
     forms ending in -ɔ < -aka (*ālla > allɔ, kakkʰaṭa > kákʰṛɔ, accʰa > háʦʦʰɔ), where the bare
     headword used to borrow the ending from the @morph rule.
7. **Accent** (prosody-only, ACCEPT):
   - `Long > $1+stress / # C* ə C C* _` after the reduction rule, and `ə > $1-stress / # C* _` after
     the initial-stress rule. A reduced first syllable passes the stress to the following long
     vowel: kəṛā́, səbʰáuṇõ, bəʦʰáuṇõ.
   - ext_acc train .3802→.3792, test .3324→.3309 (13 reflexes better, 9 worse).
   - The words that get worse are the ə < i/u cases (kuśī́ > kɔ́śṭɔ, *hr̩tkūṭa > híkṛɔ).
8. **Removed `use aka.ua.final`** (-uka > -ua > -ū), found by rule LOO (`rloo_ktg.txt`).
   - Without the contraction, -ua loses its final a like any other final vowel and gives -u:
     dugdʰá > dudʰu, *gʰaṇṭa > gʰàṇḍu, vatsá > báʦʦʰu.
   - ACCEPT: train ext .3625→.3614, test .3459→.3440; 9 reflexes better, 0 worse.
9. **`%ext-menu +ṭṭaka +ṭṭikā`**: the MIA -ṭṭa- extension, masculine -ṭɔ and feminine -ṭɪ
   (kṓśa > kośṭɔ, kʰalla > kʰálṭɔ, ḍāla > ḍaḷṭɪ, bʰīru > bʰìrṭɪ).
   - Placebo +ppaka +ppikā: Δext train −.0048 vs −.0016, test −.0066 vs −.0017.
   - 50 reflexes better, 0 worse.
10. **Length kept in closed monosyllables** (`{ā ī ū ē ō} > {a i u e o} // # (C) (C) _ C (C) #`,
    Kochi's rule, replacing `length-loss.aiueo`): ḍāla > ḍāḷ, vā́la > bāḷ, śārka > śā́k.
    - ACCEPT-IF-JUSTIFIED: train ext .3566→.3482, test .3374→.3261; 64 reflexes better, 21 worse.
    - The plain loss (+.0091) was first read as Hendriksen's sporadic length, so the rule was
      rejected. The per-pair list shows it is bare-vs-extended: the losses are attested extended
      forms whose syllable is open (allɔ, kaḷɔ, tvaṇɔ, kʰédṇõ).
    - The 21 worse reflexes are Hendriksen's unmarked length (déś, sás).
11. **`use syncope.a.vcccv`**: medial a is also lost after a closed syllable when a vowel follows.
    This is the shared rule of bhal, kiuth and khash (*adʰyantara > jεndrɔ, *ṭiŋkara > ṭɪŋgṛɔ).
    - Round 6 rejected it in ktg on the bare-headword accent. Under ext_acc it passes: train ext
      .3482→.3441, test .3261→.3220; 63 reflexes better, 0 worse.
    - Re-tested in the other lects where round 6 rejected it:
      - **kul installed**: ext −.0014, test −.0003, verb −.0077; 17 better, 2 worse
        (váṇṭati > bɔṇḍṇā, saṁkaṭá > sɔŋgṛā).
      - **cam installed**: ext −.0004, test −.0022; 2 better, 2 worse.
      - kc still fails the verb gate (+.0017).
      - mand still fails on test ext (+.0033).

**Kochi (installed).**
- **a-rounding writes ɔ after a consonant** (gala > gɔḷo, *caḍʰati > ʦɔ́ṛʰno, pára > pɔri). The
  shared rule, left in place after it, still gives o in the two environments where Kochi has o:
  - word-initially (ávara > oru);
  - before v/y, where the o comes from contraction (ḍáyatē > ḍoa).

  The plain ɔ rule failed the verb gate (+.0123: ávara, *avaśalati, ḍáyatē, dávati). The version
  with the exceptions passes, with 28 reflexes better and 1 worse.
- **Pretonic and closed-syllable ə**: the Kotgarhi rules copied verbatim, so they are identical
  local rules that centralize.py can merge.
- **Removing the masculine -ɔ `@morph` rule** failed at first (ext_acc +.0069). After the 09:25–10:40
  engine fixes it passes: ext train −.0190, test −.0289; ext_acc unchanged. It was removed in the
  @morph audit below. The plain loss is 81 attested -o < -aka forms (bare-vs-extended).
- **`%ext-menu +ṭṭaka +ṭṭikā`**, as in Kotgarhi (nāḍī́ > naḷṭo, dvā́ra > daṭṭo, tumba > tumṭe).
  Placebo +ppaka +ppikā: train −.0044 vs −.0014, test −.0033 vs .0000.
  - The same pair fails the > .002 train margin in kul (.0012), bhal (.0013), jaun (.0004) and
    kiuth (.0011), so it was not added there.

**Kullui (installed).**
- **a-rounding writes ɔ.** kullui-org writes ɔ: adyá > ɔʣ, aṣṭā́ > ɔṭʰ.
- **A new hiatus rule `ɔ o > o / _ C`.** It stands in for the V$1 merger that o+o used to get
  (*adʰaōṣṭʰa > oṭʰ, kapʰōṇi > koṇ). Without it, ext_acc got worse on train and test.
- **Rounding blocked pretonically before ā** (`// _ C (C) ā`, as in ktg and kc): *ḍakkāra > ḍakār,
  *kalyā-hāra > kalār. Train ext −.0008, 10 reflexes better vs 2 worse; test −.0004.
- **Blair `%source` block**, as `@notation`:
  - `a > ə` (Blair writes Kullui short a as ə);
  - `ɔ > ə` (was `o > a`; ə beats a on train, .3707 vs .3737);
  - the final reduced vowel is now written ə (`∅ > ə / C _ #`, was a). This alone gave train ext
    −.021 and test −.022.

**Kvari (installed under the coordinator's small-sample verb waiver).** a-rounding and -aka now
write ɔ. Jouanne's ɔ continues a (pálāla > pɔraḷ, balivárda > bɔḷēd, jámbʰa > ʣɔbṛɔ), and her o
continues OIA ō (kōṣṭʰāgāra > koṭʰār, *gʰōraḍa > goṛēḍ).
- Train ext .4142→.3639, test .4668→.3903; 42 reflexes better, 3 worse.
- Verb menu train goes .3667→.5417 (.4944→.6111 after the verb-flag fix). That comes from a single verb
  pair out of 2 (3 after the fix), vásati > bośo. Its old
  score was an -anaka menu chance match: bosṇo matched the final -o, although the Kvari form has no
  -ṇ-. Its first-syllable o is one of three lexical o's for rounded a (with rasavatī > rosuigɔr,
  *tapaka > toi).
- The waiver is noted in the rule comment. The environment was not changed to get past the gate.

**Rejected or not attempted.**
- **WPah a > ɔ:** test ext +.011. This is the same rounding/non-rounding dialect mixture as in
  round 6.
- **WPah pretonic ə:** ext worse on train.
- **Bhalesi:**
  - a > ɛ is sporadic (30 attested ɛ vs 286 a);
  - ē > ɛ/ɛ̃ splits evenly with e (26 vs 25), and the ɛ̃ nasalisation is lexical;
  - no rule.
- **jaun, bhad, khash, dog:** the lax vowels are ≤ 15-token scatter per cell.
- **ktg `u > v / C _ retroflex`** (Hendriksen's v): still fails ext_acc (+.0021). A stressed v́ is
  a consonant, so the score cannot see its accent.
- **dog aphaeresis before the stress rule:** test ext_acc +.0062. The single test word is still
  ardʰarātra.

### Focus 2: accent (ext_acc)

The initial-stress placement used by ktg, dog, khash, WPah, bhad and kgr was tested in the six lects
that had no placement rule. All six are **ACCEPT (prosody-only)** and were installed right after
`@acc-loss`:

| lect | ext_acc train | ext_acc test | accent pairs (train/test) |
|---|---|---|---|
| kc | .6514→.3722 | .5345→.3429 | 6/7 |
| kul | .5264→.4194 | .6042→.5417 | 6/2 |
| cam | .6515→.6348 | .7000→.6000 | 6/1 |
| jaun | .6000→.5000 | – | 2/0 |
| kiuth | .7411→.7133 | = | 3/1 |
| kvar | .3333→.2500 | .6250→.5000 | 2/1 |

Most of these accent sets are tiny. After this round, 14 of the 16 chains place a stress; bhal
(quantity stress) and mand (final stress) keep their round-3 rules. cur and pan have no marked
forms.

**Tone.** dog, kgr and mand are tone-scored: Patyal and Gill write grave low tones.
- Most are Punjabi-type: low tone after a lost voiced aspirate or h (gʰuṇa > kū̀ṇ, mūrdʰán > mū̀ḍā,
  anuhāra > nùār).
- There are ≤ 8 such forms per lect, and the tone has to be written before deaspiration. Writing à
  that early would block later rules that target a.
- Not attempted. It needs a latent tone mark, like `$1+stress`.

### Focus 3: consistency
- ktg and kc now share the same length rule: length is kept only in closed monosyllables.
- ktg, kul and cam now share `syncope.a.vcccv` with bhal, kiuth and khash.
- bhad, jaun, kiuth, WPah and pan share the degemination after a long vowel.
- ktg and kc now share the same local ə rules.
- ktg, kc and kul all block a-rounding pretonically before ā.
- ktg with kc's `_ {v y}` exception fails on test (ext +.0004, verb +.0023), so the rounding
  environments stay lect-specific.

### Focus 6: legacy `@morph` rules (`morph.py`, `morph_*.txt`)

Every active chain `@morph` rule (outside `%source` blocks) was commented out: first all together
per lect, then one at a time, then the passing ones together. A removal was kept when
ext/verb/ext_acc/ext_tone were unchanged or better on train and test. Where ext is unchanged, this
is a "neutral removal": `gates.py` says REJECT only because ext train did not improve. The plain
loss is the bare-headword citation form these rules pasted on. The verb and ext menus derive the
same forms.

Rules removed (all commented out, with the reason in the file):

| lect | rules removed | kind | plain train Δ |
|---|---|---|---|
| ktg | -āpayati > -āuṇõ, if:verb -ayati > -ṇõ, -āti/-ōti > -ṇõ, if:verb C-root > -ṇõ | neutral | +.0124 |
| ktg | masc. -ɔ generalisation `{a á} > o` (focus 0) | ACCEPT-IF-JUSTIFIED | +.0051 |
| kc | -āpayati, -āti/-ōti, C-root, -ati > -ṇo | neutral | +.0210, then +.0291 |
| kc | masc. -ɔ generalisation | ext −.0190 / −.0289 | (+.0438 joint with -ati) |
| kiuth | all 4 | neutral | +.0677 |
| cur | all 3 | neutral | +.0531 |
| khash | both | neutral | +.0138 |
| jaun | the one | neutral | +.0114 |
| dog | the one | neutral | +.0007 |
| pan | the one | neutral | +.0471 |
| bhal | if:verb -ati > -ṇū, if:verb C-root > -ṇū | neutral | +.0249 |
| WPah | -ayati > -ṇu, if:verb C-root > -ṇu | neutral | +.0256 |
| bhad | if:verb -ayati > -ṇū, -āti/-ōti > -ṇū | neutral | +.0041 |
| kgr | if:verb -ati > -ṇā | neutral | +.0061 |
| cam | -ayati, -āti/-ōti, -ati > -ṇā | neutral | +.0062, then +.0923 |
| kul | -āti/-ōti > -ṇā | neutral | +.0005 |

Rules kept (removal fails a gate):

| lect | rule | failing gate |
|---|---|---|
| ktg | `a {i ē} > ṇ õ / _ #` (-ati) | ext train +.0007; ext_acc +.0014 (8 reflexes) |
| kul | `a {i ē} > ṇ ā / _ #` (-ati) | ext train +.0008 (8 better / 11 worse) |
| kul | `∅ > ṇ ā / C _ #` (C-root) | ext +.0030; ext_acc +.0083 |
| bhal | `a a {i ē} > ṇ ū / _ #` | ext +.0002 |
| bhal | `{ā ō ī} {i ē} > $1 ṇ ū / _ #` | ext test +.0003 |
| WPah | `a {i ē} > ṇ u / _ #` | ext +.0010 (ext_acc improves) |
| bhad | `a {i ē} > ṇ ū / _ #` | ext +.0004 |
| kgr | `∅ > ṇ ā / C _ #` | ext_acc +.0021 |
| cam | `∅ > ṇ ā / C _ #` | ext +.0017; ext_acc +.0083 |
| kc | `i > y / C _ {a ā}` (@morph @hiatus) | ext +.0031 |

What the kept rules still serve:
- the -ati rules: headwords that the verb flag misses (numerals in -ati, words without a verb gloss);
- the C-root rules: starred roots without a verb gloss.

**Engine reports while doing this** (all fixed by the coordinator):
- Kullui glosses without "to" ("bring", "climb up") and numbered glosses ("1) eat; 2) bite") left
  111 of 115 -ati headwords without the `verb` flag.
- After both fixes, kul's -ati rule still costs ext +.0008, so it is kept.
- The kul, ktg and cam numbers above were re-gated after the fixes.

### Other candidates this round
- **Degemination after a long vowel** `C > ∅ / Long _ $1`: a heavy syllable keeps at most two
  morae (dā́yādya > dāʣ, kārmika > kāmī, sārdʰa > sāḍʰō).
  - Round 6 rejected it in bhad on the bare-headword accent. With ext_acc it passes: bhad ext
    .4025→.4012, test .4844→.4825; 6 reflexes better, 2 worse.
  - The same local rule was then tested in the other 11 lects that keep geminates. It passes, and
    was installed, in:
    - jaun: ext −.0025, test −.0021; 7 better, 5 worse;
    - kiuth: −.0007, test −.0027; 4 better, 0 worse;
    - WPah: −.0002; 1 better;
    - pan: −.0013; 1 better.
  - Rejected in bhal (ext_acc +.0006), mand (verb +.0179) and dog (ext +.0006). No effect in khash,
    kgr, cam, cur and kvar.
- **khash final degemination** `C > ∅ / _ $1 #`: test ext +.0036. Vetoed.
- **Tone in dog** (low tone after an initial voiced aspirate):
  - The rule has to be written at the chain's end, so initial deaspiration was moved after it. Then
    ext_tone improves on train and test, but ext_acc gets worse (+.0026). An anaptyctic vowel
    (bʰrātr̩tvá > párā́it) gets a second accent. Rejected.
  - Written before deaspiration, the tone marks block later vowel rules and change ext (−.0011).
    That gain is an artifact.
- **bhal -ai > -ī** (round 6: rejected on test accent): now train ext −.0012 but test ext +.0001;
  10 better, 8 worse. Rejected.
- **bhal `l > ∅ / ā _ (V) #`** (ḍāla > ḍā, gāli > gāi, kāla+aka > kāo): 11 reflexes better vs 13
  worse, ext_acc +.0091. Lexical; rejected. `l > ∅ / ā _ #` alone: ext +.0002.
- **Rule LOO** (`rloo_{ktg,kul,kc,bhal,jaun,kiuth}.txt`):
  - the only effective removal was ktg `aka.ua.final` (above);
  - jaun `aka.aa.final.2` (-aka > -o): `gates.py` accepts removing it (ext −.0034, test −.0042),
    but only 50 reflexes get better and 119 worse. The gain trades the -o of the masculine for
    John's final -a (sā́ra > sara, pūra > pura). That is an ending choice, not a sound change, so it
    was not removed;
  - kiuth `aphaeresis.aa.cc`: test verb +.0054 again. Kept;
  - bhal `final-shortening`: ext −.0011, but the plain loss is genuine: bare -ī headwords attested
    with -i (amlī > amri, stʰālī́ > tʰāli). Kept;
  - the zero-effect `hiatus.a` lines (kul, kc) and kc `hiatus.oi` were left in place, since the gate
    needs a strict gain.

### Before → after (pre-round-7 snapshot vs installed, same engine, 10:57; `final_gates_c.txt`)

Both files were scored with the engine as of 10:57. "A-I-J" is ACCEPT-IF-JUSTIFIED: the only loss
is plain NED from the @morph removals (bare-vs-extended, focus 6). "Neutral @morph": ext, verb and
prosody are unchanged, and only plain rises. The ext_acc gains in WPah, jaun, bhad, kgr, cam, kc,
kul and kvar come from the initial-stress placement (step 0 and focus 2). In kiuth the stress rule
passed at install (.7411→.7133), but under the 10:57 engine it changes nothing (= .6652).

| lect | ext train | ext test | plain train | verb train | ext_acc train | ext_acc test | reflexes better/worse | verdict |
|---|---|---|---|---|---|---|---|---|
| ktg | .4444→.3441 | .4398→.3220 | .5019→.4880 | .2988→.2789 | .4624→.3680 | .4385→.3154 | 928/136 | ACCEPT |
| kul | .4171→.3685 | .4243→.3807 | .5096→.4606 | .2370→.1948 | .5264→.4194 | .6042→.5417 | 135/38 | ACCEPT |
| bhal | = .3561 | = .3815 | .4346→.4600 | = .2900 | = .4122 | = .4393 | 0/0 | neutral @morph |
| kiuth | .3186→.3179 | .3404→.3377 | .4131→.4812 | = .2160 | = .6652 | = .6429 | 4/0 | A-I-J (@morph plain) |
| WPah | .3904→.3903 | = .3388 | .4805→.5059 | = .3214 | .6423→.6137 | = .3750 | 1/0 | A-I-J (@morph plain) |
| jaun | .3540→.3515 | .3263→.3242 | .4359→.4434 | = .3301 | .6000→.5000 | = .0000 | 7/5 | A-I-J (@morph plain) |
| bhad | .4025→.4012 | .4844→.4825 | .4658→.4688 | = .3101 | .6297→.5656 | .5821→.5179 | 6/2 | A-I-J (@morph plain) |
| dog | = .3744 | = .3762 | .4657→.4664 | = .4193 | = .5397 | = .5184 | 0/0 | neutral @morph |
| kc | .4023→.3580 | .4521→.3931 | .4944→.5403 | .3280→.3135 | .5000→.2000 | .3389→.2167 | 69/9 | ACCEPT-IF-JUSTIFIED (@morph plain) |
| mand | = .3684 | = .4174 | = .4646 | = .3229 | = .5493 | = .6000 | – | unchanged |
| khash | = .4142 | = .4310 | .4904→.5036 | = .3792 | = .5071 | = .5045 | 0/0 | neutral @morph |
| kvar | .4139→.3636 | .4668→.3903 | .4746→.4551 | .4944→.6111 | .3333→.2500 | .6250→.2500 | 42/3 | REJECT on verb only (waiver) |
| cur | = .3065 | = .3453 | .3949→.4505 | = .2095 | = .0000 | = .0000 | 0/0 | neutral @morph |
| kgr | = .3995 | = .4439 | .4804→.4865 | = .4052 | .6698→.6219 | .4944→.4778 | 0/0 | neutral @morph + prosody |
| cam | .2693→.2690 | .3479→.3457 | .3510→.4495 | .2195→.2173 | .6515→.6348 | .7000→.6000 | 2/2 | A-I-J (@morph plain) |
| pan | .3070→.3056 | = .2955 | .3713→.4184 | = .3095 | = .0000 | = .0000 | 1/0 | A-I-J (@morph plain) |

Notes on the table:
- **kvar**: `gates.py` says REJECT only because of the verb menu, a single pair (vásati > bośo) out
  of 3. It is installed under the coordinator's small-sample waiver (see Kvari above).
- **ktg**, with all its changes, is ACCEPT outright: plain train improves as well
  (.5019→.4880).
- The Focus 2 table gives ext_acc at install time. Later engine fixes moved the baselines; this
  table has the current ones.

### Not done / open
- **Tone in dog, kgr and mand**: needs a latent tone mark (or `realize` support for a tone on a
  non-first vowel) to be written before deaspiration without blocking later rules.
- **Hendriksen's v for u before a retroflex (ktg)**: still an engine question. Reading
  post-consonantal v as ʊ would need the chain to produce ʊ.
- **Kochi pretonic stress** (the ktg-style shift to the long vowel): not tried.
- **Kochi closed-syllable syncope**: verb gate +.0017.
- **WPah dialect mixture** (rounding and affrication): not source-separable within CDIAL's
  "WPah.".
- **Kept @morph rules**: listed in focus 6. The -ati rules in ktg, kul, bhal, WPah and bhad still
  serve headwords without a verb flag.
