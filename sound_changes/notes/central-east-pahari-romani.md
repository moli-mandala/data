# central-east-pahari-romani — OIA → Pahari (N, Ku, Garh, Dotyali) and the migratory lects (Romani, Domari, Lomavren, Zargari)

Files: `rules/OIA-{N,Ku,Garh,Dotyali,eur,as-ll,Lom,RomWel,RomBalk,Zarg}.sc`,
`results/OIA-*.hand.frontier.csv`. Scratch scripts in `tmp/sound_changes/central-east-pahari-romani/`
(`q.py` variant tester, `chron.py` reorder tester, `central.py` / `control.py` feature tests,
`prune.py` LOO pruning).

## 1. Summary table

| lect | #etyma (train+test) | #rules | train NED null→final | test NED null→final | test exact % | auto test NED @≈same #rules | excluded sources |
|---|---|---|---|---|---|---|---|
| N | 3750 | 40 | 0.627 → 0.395 | 0.620 → 0.376 | 15.9 | 0.443 (21 rules) | zoller2023 |
| Ku | 2745 | 41 | 0.656 → 0.409 | 0.661 → 0.392 | 15.5 | 0.485 (20 rules) | zoller2023 |
| Garh | 696 | 43 | 0.693 → 0.372 | 0.695 → 0.383 | 23.2 | 0.515 (20 rules) | zoller2023 |
| Dotyali | 191 | 32 | 0.642 → 0.421 | 0.636 → 0.418 | 7.6 | 0.508 (24 rules) | — |
| eur | 512 | 42 | 0.706 → 0.399 | 0.749 → 0.426 | 15.5 | 0.574 (18 rules) | — |
| as-ll | 299 | 34 | 0.690 → 0.439 | 0.673 → 0.437 | 5.6 | 0.540 (17 rules) | herin2012domari |
| Lom | 160 | 31 | 0.732 → 0.384 | 0.719 → 0.400 | 12.1 | 0.586 (16 rules) | — |
| RomWel | 132 | 31 | 0.721 → 0.479 | 0.745 → 0.515 | 9.1 | 0.670 (14 rules) | — |
| RomBalk | 121 | 34 | 0.733 → 0.411 | 0.769 → 0.526 | 3.1 | 0.574 (16 rules) | — |
| Zarg | 111 | 32 | 0.760 → 0.519 | 0.751 → 0.494 | 11.4 | 0.673 (20 rules) | — |

Notes on the table:
- "auto" = the greedy baseline's test NED at its longest prefix (the auto curves stop at 14–24 rules);
  the hand frontier at the same rule count is given in §1b.
- Rules with train LOO ≤ 0 were pruned per lect (§6), so small lects have fewer rules than the
  group chain: a missing shared rule in RomWel/RomBalk/Zarg/Lom/Dotyali means *no evidence in
  ~120–250 train pairs*, not counter-evidence.
- `%source CDIAL … %end` block in `OIA-eur.sc`: CDIAL cites Romani verbs as bare stems (ker,
  kin, av), boretzky1994romani as 3sg -el; the chain produces -el and the CDIAL block strips it
  (−0.026 train NED). This is the only source block; it is tagged `@morph`.

### 1b. Hand frontier vs automatic baseline (test NED at equal rule counts)

| lect | test NED hand @5 / auto @5 | hand @10 / auto @10 | hand @k / auto @k (k = auto's longest) | hand full |
|---|---|---|---|---|
| N | 0.521 / 0.538 | 0.467 / 0.491 | 0.407 / 0.443 (k=21) | 0.376 (40) |
| Ku | 0.546 / 0.549 | 0.485 / 0.505 | 0.429 / 0.485 (k=20) | 0.392 (41) |
| Garh | 0.560 / 0.562 | 0.511 / 0.522 | 0.449 / 0.515 (k=20) | 0.383 (43) |
| Dotyali | 0.513 / 0.555 | 0.474 / 0.526 | 0.444 / 0.508 (k=24) | 0.418 (32) |
| eur | 0.586 / 0.640 | 0.537 / 0.591 | 0.483 / 0.574 (k=18) | 0.426 (42) |
| as-ll | 0.510 / 0.525 | 0.475 / 0.530 | 0.452 / 0.540 (k=17) | 0.437 (34) |
| Lom | 0.563 / 0.637 | 0.504 / 0.607 | 0.465 / 0.586 (k=16) | 0.400 (31) |
| RomWel | 0.654 / 0.697 | 0.628 / 0.660 | 0.604 / 0.670 (k=14) | 0.515 (31) |
| RomBalk | 0.710 / 0.657 | 0.654 / 0.629 | 0.587 / 0.574 (k=16) | 0.526 (34) |
| Zarg | 0.661 / 0.699 | 0.621 / 0.685 | 0.541 / 0.673 (k=20) | 0.494 (32) |

## 2. Base chains

### 2a. Pahari (N, Ku, Garh, Dotyali) — the OIA-H.sc MIA block plus Pahari NIA

MIA (same relative order as OIA-H.sc; additions marked +):
1. `V > $1-acc` @acc-loss
2. `{ai au} > {ē ō}` @ai-au
3. + `r̩ > r i / # _` @r-vocalic-ri (r̩ṇa > rin); + `r̩ > u / Lab _` @r-vocalic (vr̩kṣa > buk); `r̩ > i` @r-vocalic
4. `Sib > s` @sibilant-s
5. `{t tʰ d dʰ} y > cc/ccʰ/jj/jjʰ` @palatal-Cy
6. `{k t p} s > kkʰ ccʰ ccʰ` @ks @x-cep-ts-cch
7. `s Stop > $2-asp $2+asp` @s-stop-aspirate; `s Son > s s` @assim-sonorant
8. `Nasal Stop > ṁ $2` @nasal-cluster
9. + `ṁ {g gʰ b bʰ} > ŋŋ ŋŋ mm mm` @x-cep-nasal-voiced-assim (áŋgāra > aŋār, kumbʰakāra > kumāle; Garh: only mb)
10. `{r ḣ} C > $2-asp $2` @assim-r; `Stop {r y v l Nasal} > $1-asp $1` @assim-sonorant; `Stop Stop > $2-asp $2` @assim-stop
11. `C C > $2 / # _` @initial-degem
12. `{kʰ gʰ tʰ dʰ pʰ bʰ} > h / V _ V` @int-aspirate-h; `{ṭ ṭʰ} > {ḍ ḍʰ} / V _ V` @int-retroflex;
    + `a v a > o` @aya-e (before p > v; N −0.0011, Ku −0.0007, Garh −0.0033, Dotyali −0.0013); `p > v / V _ V` @int-p-v
13. `{k g c j t d y} > ∅ / V _ V` @int-stop-loss
14. + `{s t d m ḥ} > ∅ / _ #` @final-cons-loss (restricted to real OIA finals; starred roots like *cil keep theirs)
   (the H two-mora rule `{ā ī ū ē ō} > $1-long / _ C C` has LOO 0.0000 in N and −0.0014 in Ku —
   compensatory lengthening re-lengthens the same vowels — and was dropped from all four.)

NIA:
15. @morph `∅ > n u / C _ # // n _` — CDIAL's consonant-final starred *roots* (*ṭʰakk-, *cʰipp-,
    *ḍʰapp-) are cited in N./Ku./Garh./Dotyali as verbs (N −0.0115, Ku −0.0096, Garh −0.0039,
    Dotyali −0.0056; also eur −0.0016, Domari −0.0017 with -el / -ʌr). A citation convention, not
    a sound change, but it raises exact match too.
    @morph infinitive: `ā v a a {i ē} > ā u n u`, `a a {i ē} > n u`, `a {i ē} > n u` (N -nu, Ku -ṇo,
    Garh -ṇu, Dotyali cited in the past -yo)
16. `a a > o / _ #` @morph (-aka > -o; Garh -u)
17. N & Dotyali only: `{ḍ ḍʰ} > {r ṛʰ} / V _ V` @flap (single MIA -ḍ- > N. r: gʰaṭī > gʰari)
18. `V G $2 > $1+long $3 / _ V #` @degem-lengthen; `G $1 > $2` @degem-plain
19. `V ṁ > $1+long+nas / _ C` @anusvara-vowel
20. `{ḍ ḍʰ} > {ṛ ṛʰ} / V _` @flap (remaining ḍ, i.e. old ḍḍ/ṇḍ)
21. + `v > ∅ / V _ V` @v-loss (−0.003 train NED in every Pahari lect)
22. `{a i ā} > ∅ / _ #` @final-vowel-loss (final -u kept; Dotyali keeps -ā)
23. `repeat: {a ā} {a ā} > ā` @hiatus
24. lect-specific block (below), then `v > b / # _` @v-b, `y > j / # _` @y-j, notation.

Lect deviations:
- **N**: `ṇ > n` @n-dental; notation `{ē ō} > {e o}`, `{ī ū ī̃ ū̃} > {i u ĩ ũ}` (Turner's N. has no
  i/ī u/ū contrast: dropping it costs +0.032).
- **Ku**: `h > ∅ / V _ V` @h-loss; `i > y / C _ {a ā o u}` @hiatus; `n > ṇ / V _`
  @x-cep-n-retroflex (−0.0053); `{ī ū ī̃ ū̃} > {i u ĩ ũ} // _ #` @x-cep-high-vowel-shortening
  (non-final only; final -ī -ū kept).
- **Garh**: no r̩ > u, only `ṁ {b bʰ} > m m` (Garh keeps ŋg: kaŋgaṇ); `{a ā} u > o`, `{a ā} i > e`
  @vowel-shift (secondary au > ɔ, ai > ε; −0.0036/−0.0011); `h > ∅ / V _ V`; `i > y / C _ V`;
  `ī > i / _ #` @x-cep-final-i-shortening; `n > ṇ / V _`; `l > ḷ / V _` @l-retroflex (−0.0072);
  `ṇ > n / {r ṇ ḷ l} _` @x-cep-n-dissimilation (−0.0061: sārnu, kʰaṇnu); -aka > -u.
- **Dotyali** (eichentopf-tupper2019, one field source): verbs cited in the past (-yo, -eko) → @morph
  -yo; `∅ > y / V _ V` @ya-sruti (the source writes hiatus glides, −0.0087); notation
  `{ṭ ṭʰ ḍ ḍʰ ṛ ṛʰ} > dentals` (the source mostly writes retroflexes as dentals: pet, kā̃do);
  keeps final -ā.

### 2b. Migratory lects (eur, RomWel, RomBalk, as-ll = Domari, Lom, Zarg)

Order (all files; pruned rules simply missing):
1. `V > $1-acc`; `{s t d m n ḥ} > ∅ / _ #` @final-cons-loss (before the verb morph, so stems keep theirs)
2. @morph verbs: eur `-ati > -el` (+ CDIAL source block → bare stem), Domari `-ati > -ʌr`
   (Macalister cites 3sg), others bare stem; `a k a > o / _ #` @morph (-aka > -o, placed before
   cluster simplification so pakva > *paka does not become -o)
3. `{ai au} > {ē ō}`; `r̩ > u / Lab _` (eur, Balkan also after ś: śun, śung), `r̩ > i` @r-vocalic
4. `ṁ > ∅` @x-cep-anusvara-loss (no nasal vowels: viṁśati > biś)
5. `k ṣ > cʰ / # _` (Romani, Zarg), `k ṣ > kʰ` @ks
6. sibilants: Romani/Zarg `{ś ṣ} > ś` @sibilant-ś (s kept: sasto vs śero); Domari `ṣ > ś / _ Stop`
   @sibilant-keep + `{ś ṣ} > s // _ Stop` @sibilant-s; Lom `{ś ṣ} > s` @sibilant-s
7. `{t tʰ d dʰ} y > {c c j j}` @palatal-Cy; `{t p} s > cʰ` @x-cep-ts-cch
8. Romani/Zarg: `{b d g j} > {pʰ tʰ kʰ cʰ} / # _ V (C) VAsp` @x-cep-aspiration-shift
   (jambʰa > cʰam, *buddʰa > pʰuro); `VAsp > {kʰ cʰ ṭʰ tʰ pʰ} / # _` @x-cep-devoicing-asp
   (bʰrātr̩ > pʰral, dʰūma > tʰuv, gʰarma > kʰam); `VAsp > {g j ḍ d b}` @deaspiration.
   Domari: deaspiration only, no devoicing, no shift (bʰaginī > bēn, gōdʰūma > gēsū).
   Lom: `VAsp > h / V _ V` @int-aspirate-h then `VAsp > voiceless aspirates` everywhere
   (randʰ- > ərantʰ, labdʰa > altʰ) and `d > l / # _ V` @x-cep-lom-d-l (dēśa > les, dakṣ- > lakʰ).
9. `{ṇ n} {ṭ ḍ} > n r` @flap (maṇḍa > manro; Domari n d); `Nasal {k c t p} > $1 {g j d b}`
   @nasal-voicing (dant > dand); Welsh `{ŋ ṁ} {g gʰ} > ŋ`
10. `s Stop > $2 / # _` @initial-degem (stʰūla > tʰulo; Lom instead `s Stop > $2+asp`
    @s-stop-aspirate: hasta > atʰ, nāsti > natʰ)
11. `a v a > o` @aya-e (eur −0.0028, Welsh −0.0027, Zarg −0.0034, Lom −0.0003; not Balkan/Domari); `p > v / V _ V` @int-p-v; eur, Balkan `a > o / _ v` @a-rounding (sov, jov, sovlax; before m > v, so nāman > nav); Domari
    `{k g c j} > ∅ / V _ V` @int-stop-loss
12. `{t d} > l / V _ V` @x-cep-t-l (Domari > r)
13. `m > v / V _ V` @m-v; `r C > $2` @cluster-simplify (Welsh keeps rC); `{k g m} r > $1 / # _`
    @initial-degem (kr, mr simplify, **tr dr pr kept**: trin, drak, pʰral); `Stop Stop > $2`,
    `Stop {y v l Nasal} > $1`, `{ś s} {v y} > $1` @cluster-simplify (no Stop+r assimilation)
14. `{ṭ ṭʰ ḍ} > r / V _ | _ #` @flap; `C $1 > $1` @degem-plain
15. `v > b / # _`, `y > j / # _`, `y > ∅ / V _ V` @int-stop-loss (eur, Welsh, Zarg); `h > ∅` @h-loss
    (all four Romani lects incl. Zargari, −0.001…−0.002; not Lom, which keeps h: mahl, gihu)
16. `a > ∅ / V C _ C V` @syncope (eur, Domari, Lom, RomBalk, Zarg; bárkara > bakri)
17. `{a i u ā} > ∅ / _ #` (Domari keeps -a as -ʌ: only `{i u}`)
18. `{ā ī ū ē ō} > {a i u e o}` @x-cep-length-loss (Domari: only ā ī ū, @notation);
    `{ṇ ŋ ñ} > n` @n-dental; `{ṭ ṭʰ ḍ ḍʰ} > {t tʰ d dʰ}` @x-cep-retroflex-loss
19. notation/lect tails: eur, Welsh `cʰ > c` @deaspiration; Welsh `∅ > v / # _ a` @prothesis (vaŋār, vavēr; −0.004), `h > ∅ / V _ V` @h-loss,
    `kʰ > x` @x-cep-kh-x; RomBalk `kʰ > x / # _`, `{tʰ kʰ pʰ} > {t k p}` @notation (the Balk.
    source marks aspiration only on cʰ); Domari `{kʰ cʰ ṭʰ tʰ pʰ} > plain` @deaspiration
    (Macalister writes no aspiration) and `{ṭ ḍ} > {t d}`; Zarg `a > ā` @notation (Persian-style
    transcription writes /ɒ/ as ā: kān, māng, tʰān).

## 3. Relative chronology (Δ train NED when the order is reversed)

Pahari:
- Degemination-with-lengthening (penultimate) **before** final-vowel loss: N +0.0047, Ku +0.0054,
  Garh +0.0129, Dotyali +0.0165. The lengthening is conditioned on the old final vowel.
- The lengthening is **penultimate-only**: an unconditioned `V G $2 > $1+long $3` costs N +0.0052,
  Ku +0.0016, Garh +0.0056 (Dotyali alone prefers it, −0.0019) — i.e. stress-conditioned, as in Hindi.
- ŋg/mb assimilation (`ṁ {g b} > ŋŋ/mm`) **before** anusvāra → nasal vowel: N +0.0078, Ku +0.0090,
  Garh +0.0038, Dotyali +0.0084. I.e. it happened while the anusvāra was still consonantal (MIA),
  not as a late NIA simplification of Ṽg.
- N/Dotyali flap of *single* MIA -ḍ- to r **before** degemination: N +0.0014, Dotyali +0.0013
  (old ḍḍ stays ṛ/ḍ).
- Cy palatalisation bleeds general C+y assimilation: N +0.0021, Ku +0.0015, Garh +0.0057, Dotyali +0.0046.
- Ku intervocalic h-loss after aspirate > h (h-loss feeds on MIA h): +0.0017 (Garh ±0).
- Garh/Ku h-loss **before** v-loss, final-vowel loss and hiatus contraction (so the new a-a hiatus
  contracts): found by an exhaustive single-move search (`reorder.py`); Garh −0.0055 vs the late
  position, Ku −0.0002.
- (Trivial/notational, not evidence: C+sonorant assimilation feeds initial degemination; kṣ/ts rule
  written on `s` must follow the sibilant merger.)

Migratory:
- **t d > l (Romani, Lom, Zarg) / r (Domari) before cluster/geminate simplification** (mattá >
  mato, not *malo): eur +0.0058, Domari +0.0074, Lom +0.0044, Welsh +0.0028, Zarg +0.0081
  (Balkan −0.0015, 132 pairs). The lenition hit only singleton dentals while MIA-type geminates or
  clusters were still intact.
- **ṭ ḍ > r after cluster simplification** (so old ṭṭ/ḍḍ also flap: *pōṭṭa > por, haḍḍa > hero):
  Balkan +0.0095, Zarg +0.0028, Welsh +0.0018, Lom +0.0017, eur +0.0005; Domari −0.0010 (Domari
  keeps ṭṭ as a stop: pēṭ, kúrtʌ — so OIA-as-ll.sc puts the flap *before* simplification). So in Romani/Lom the dental lenition and the retroflex flap
  are chronologically distinct: dental lenition < simplification < retroflex flap.
- p > v before cluster simplification (pp does not > v): eur +0.0029, Domari +0.0022, Zarg +0.0030.
- m > v before r C > C (gʰarma > kʰam, not *kʰav): eur +0.0005, Zarg +0.0022 (weak).
- Lom: voiced aspirates > h intervocalically **before** the general devoicing of voiced aspirates:
  +0.0044 (gōdʰūma > gihu but randʰ- > ərantʰ).
- Cy palatalisation before C+y simplification: Lom +0.0045, Welsh +0.0076.
- Medial syncope applies while final vowels are still present: eur +0.0028, RomBalk +0.0047,
  Zarg +0.0053 (partly forced by the rule's environment).

## 4. Shared vs lect-specific changes; subgrouping

**Pahari vs Hindi.** The whole Hindi MIA block carries over. Pahari-specific, shared by all four:
ŋg/mb > ŋ/m (MIA-stage; Garh only mb), intervocalic v-loss, final -u retention, masc. -o
(< -aka; Garh -u). N (and Dotyali) vs Ku/Garh:
- single -ḍ- > r in N/Dotyali, ṛ in Ku/Garh (adding the N rule to Ku/Garh costs +0.006–0.008);
- intervocalic h-loss in Ku/Garh (−0.002/−0.004), not in N (+0.0006);
- postvocalic n > ṇ in Ku/Garh (−0.005), not N;
- glide formation iV > yV in Ku/Garh only; Garh alone: l > ḷ, ṇ > n after r/ṇ/ḷ, au > ɔ, ai > ε, mb
  but not ŋg assimilation.
Cross-application check (`readd.py`): every lect-specific rule above makes its sister lects
worse or is neutral when inserted at the analogous position (e.g. N's r-flap in Ku +0.0077 / Garh
+0.0074; Garh's l > ḷ in N +0.025 / Ku +0.027; Ku's n > ṇ in N ±0; Garh's ai > ε in Dotyali +0.009),
so none of them is a missed shared change. Likewise Lomavren's s-merger (+0.013–0.016), d > l
(+0.006–0.020) and st > tʰ (+0.004) and Domari's t d > r (+0.004–0.010) and deaspiration
(+0.019–0.026) are rejected by the Romani chains.
This supports the traditional Central Pahari (Ku, Garh) vs Eastern Pahari (N, Dotyali) split;
Dotyali patterns with N (r-flap, no n > ṇ); its h-loss evidence is too small to use (−0.0016).

**Migratory lects vs Central (Hindi) MIA.** Table: Δ train NED when a Hindi-type MIA variant is
put into each migratory chain (positive = the Central variant is worse):

| Hindi-type variant put into the chain | eur | RomWel | RomBalk | as-ll | Lom | Zarg |
|---|---|---|---|---|---|---|
| A s+stop>aspirated geminate (hattʰa) | +0.0029 | -0.0020 | +0.0044 | +0.0024 | n/a | +0.0034 |
| B Stop+r assimilated (tr>tt) | +0.0043 | n/a | +0.0015 | +0.0050 | +0.0060 | +0.0015 |
| C int. k g c j > ∅ | +0.0007 | +0.0043 | +0.0055 | +0.0000 | +0.0043 | -0.0007 |
| D int. aspirates > h | +0.0039 | +0.0008 | +0.0011 | +0.0048 | +0.0000 | -0.0007 |
| E sibilants > s | +0.0165 | +0.0091 | +0.0130 | +0.0025 | +0.0000 | +0.0133 |
| E' sibilants > ś (s too) | +0.0207 | +0.0132 | +0.0190 | +0.0237 | +0.0511 | +0.0251 |
| F no post-nasal voicing | +0.0038 | +0.0045 | n/a | +0.0033 | n/a | +0.0034 |
| G kṣ > cʰ | +0.0031 | +0.0028 | +0.0059 | +0.0031 | +0.0072 | +0.0039 |
| G' kṣ > kʰ | +0.0014 | +0.0000 | +0.0069 | +0.0000 | +0.0000 | +0.0022 |
| H r̩ > ri (no i/u) | +0.0000 | +0.0033 | n/a | -0.0004 | +0.0012 | n/a |
| I t d > ∅ (Central) not l/r | +0.0075 | +0.0035 | +0.0012 | +0.0077 | +0.0073 | -0.0007 |
| J t d kept | +0.0106 | +0.0019 | +0.0039 | +0.0125 | +0.0158 | +0.0049 |
| K int. voicing k t p > g d b | +0.0011 | +0.0076 | +0.0051 | +0.0000 | +0.0032 | +0.0000 |
| L ts ps > cʰ dropped | +0.0011 | n/a | +0.0021 | +0.0019 | +0.0011 | +0.0009 |

("n/a": the rule the variant replaces/attaches to is absent — e.g. Lom already has st > tʰ, Domari already has the velar loss, Welsh keeps rC.)

Wholesale test: replacing each migratory chain's own MIA block by the Hindi MIA block of
OIA-H.sc (assimilation to geminates, s+stop aspiration, intervocalic loss, sibilants > s, nasal
vowels…) and keeping the rest costs eur +0.087, Welsh +0.066, Balkan +0.071, Domari +0.054,
Lomavren +0.072, Zargari +0.060 train NED (`graft.py`). Domari is the least un-Central.

Control (the reverse): Romani-type variants in the Pahari chains all hurt: keeping s+stop +0.007–0.012,
keeping Stop+r +0.009–0.012, t d > l instead of loss +0.043–0.074, sibilants > ś +0.022–0.041,
initial voiced-aspirate devoicing +0.020, medial syncope +0.001–0.004.

What the migratory lects **share with Central/Hindi**: r̩ > i/u with labial conditioning (weak
evidence: variant H costs ≤ +0.003), ts/ps > cʰ (L: +0.001–0.002), kṣ > kʰ medially (Romani,
Domari, Lom; initial cʰ in Romani/Zarg), ava > o (Romani, Lom, Zarg), loss of intervocalic y,
initial v > b and y > j (Romani; Domari keeps v: vāl, vēsʌr — adding v > b costs Domari +0.008).
What they **do not** share:
- Stop+r assimilation (B +0.0015 … +0.0060 everywhere): Romani trin, drak, pʰral, medial -tr- kept;
  only kr/mr simplify. This is the clearest "pre-Central / NW-type" retention.
- s+stop → aspirated geminate (A mostly +0.002–0.005): Romani/Domari keep st, št (vast, kašt, ōśt).
  **Lomavren is the exception** (st > tʰ: hasta > atʰ, nāsti > natʰ; LOO +0.0166).
- loss of intervocalic dentals (I +0.004–0.008): instead lenition to l (Romani, Lom) / r (Domari),
  i.e. departure after voicing/spirantisation of -t- but before its loss.
- intervocalic aspirates > h (D +0.005–0.007 in Romani/Domari) — but Lomavren has it for voiced
  aspirates (gihu, mahl).
- sibilant merger as s: Romani and Zargari keep ś vs s (E +0.009–0.016), whereas **Domari and
  Lomavren merge to s** like Hindi (Domari keeps š only before stops).
- post-nasal voicing nt > nd (F +0.003–0.004) is a NW/Dardic-type feature absent from Hindi.
- Domari additionally shares Central intervocalic velar/palatal loss (LOO +0.0025).

Chronological reading: the ancestors left after r̩ > i/u, the ts/ps > cʰ and kṣ > kʰ changes, and
the voicing/lenition of intervocalic -t- (> ð?), but **before** (i) Central s+stop aspiration,
(ii) Stop+r assimilation, (iii) the loss of intervocalic stops, (iv) the sibilant merger in s.
Domari and Lomavren show more Central features (s merger; Domari velar loss; Lom st > tʰ,
aspirate > h) than Romani, which is the most archaic of the three on this evidence. Within the
group, the devoicing of voiced aspirates is shared by Romani and Lomavren (Lom everywhere, Romani
initially + aspiration shift) but **absent in Domari** (bēn, gēsū) — the "Romani vs Domari"
split by aspirate devoicing is directly visible in the data (Domari with devoicing: +0.02).
Medial a-syncope is shared by all migratory lects (−0.001 … −0.007) and absent in Pahari.

Masc. -o (< -aka) is shared by Pahari and Romani (non-Central; Hindi -ā).

## 5. Novel / surprising / data issues

- **ŋg > ŋ, mb > m in Pahari happened at the MIA (anusvāra) stage**, before nasal vowels formed
  (reordering costs +0.004–0.009). Welsh Romani has an independent ŋg > ŋ (vaŋlī).
- **Single vs geminate retroflex** in N: MIA -ḍ- (< -ṭ-, -ḍ-) > r (kuri, gʰari, tār), old -ḍḍ-/-ṇḍ- > ṛ.
- **Lomavren**: initial d > l (les, lakʰ, leval), voiced aspirates → voiceless aspirates everywhere
  (Armenian-type), st > tʰ and aspirate > h (Central-type). Tiny sample (149 train pairs).
- **Domari** has no voiced-aspirate devoicing and CDIAL (Macalister) writes no aspiration and no
  retroflexes; kṣ > kʰ > k is better than kṣ > c (círi < kṣurá is the exception).
- Two-mora shortening is invisible in Pahari (LOO 0 / −0.0014) because the penultimate
  compensatory lengthening restores the length it removes.
- Tatsama stratum is small: 2.0–2.3% of N/Ku/Garh pairs are closer to a learned rendering than to
  the chain output; 0.8% eur, 4.2% Domari.
- The dominant residual in every lect is unpredictable **gender/extension morphology**: final
  -o/-i/-u/-ā vs ∅ (N ∅→o 385, ∅→i 315, ∅→u 284 near misses; Ku ∅→o 358). A blanket
  `∅ > o / C _ #` would gain −0.02 in Ku by the NED length effect but lowers exact match; not
  added. OIA accent barely predicts -o (N oxytone 25% vs 20%).
- Sources: **herin2012domari excluded** (40 of 65 rows are inflected paradigm forms of two verbs
  with morpheme hyphens: ǧān-d-ōm, py-əme). **zoller2023 excluded** in N/Ku/Garh (digraph
  aspirates kh/bh and ə; even with a digraph `%source` block its rows score 0.62–0.67 NED — rare
  derived words). Dotyali (eichentopf-tupper2019) writes most retroflexes as dentals, ś sporadically,
  and hiatus glides; Zargari (rezaibaghbidi2003) gives inflected verb forms (te-pʰen-es, gölum,
  nā-jānā-vās) and Persian-style ā; Balkan Romani marks aspiration only on cʰ and uses `i̦`.
- Lomavren: 19 of 149 train reflexes end in a nominal -av (tatav, tʰulav, gilav, pelav) that no
  rule can predict; Zargari and Welsh show analogous masculine -o/-ō.
- Suppletive/lexical links in eur: `bʰávati → asom`, `tá → les`, `ātmán → pes`; RomBalk `muṣṭí → ṭ`.
- Tokenisation: CDIAL accents inside a diphthong split it (`saúvarṇa` → s a ú v…, `vaíra`), so the
  `{ai au} > {ē ō}` rule cannot see them (only a handful of etyma).
- Data errors / mislinkings seen: reflexes with HTML entity (`kãũlo &gt; kamlo`, N), German gloss
  as reflex (`mádʰu → branntvein`, eur), phrases and reduplicatives (`ḍagmagḍagāunu`,
  `cittā mā gamnu`, `romani-cel`, Domari light-verb compounds `díri-kerar`), a Sanskrit form as a
  Ku reflex (`*śvaśuragʰara → śvaśrūśvaśura`), `dʰvā́ŋkṣa → ḍkikicyāu` (N), `karambʰá → u` (eur).

## 6. Tags defined here (@x-cep-…) and engine wishes

- `@x-cep-nasal-voiced-assim` — ŋg > ŋ(ŋ), mb(ʰ) > m(m) (anusvāra + voiced velar/labial).
- `@x-cep-n-retroflex` — postvocalic n > ṇ (Ku, Garh).
- `@x-cep-n-dissimilation` — ṇ > n after r ṇ ḷ l (Garh).
- `@x-cep-high-vowel-shortening` — non-final ī ū > i u (Ku).
- `@x-cep-final-i-shortening` — final -ī > -i (Garh).
- `@x-cep-ts-cch` — ts, ps > (c)cʰ (co-tagged with @ks in the Pahari combined rule).
- `@x-cep-aspiration-shift` — Grassmann-like displacement of aspiration to the initial voiced stop
  (jambʰa > cʰam).
- `@x-cep-devoicing-asp` — voiced aspirates > voiceless aspirates (Romani initially, Lom everywhere).
- `@x-cep-t-l` — intervocalic t d > l (Romani, Lom, Zarg) / r (Domari).
- `@x-cep-anusvara-loss` — anusvāra lost without a nasal vowel (migratory lects).
- `@x-cep-length-loss` — loss of the vowel-length contrast.
- `@x-cep-retroflex-loss` — retroflex stops > dentals.
- `@x-cep-lom-d-l` — Lomavren initial d > l.
- `@x-cep-kh-x` — kʰ > x (Welsh, Balkan Romani).

Pruning: after hill-climbing, rules with train LOO ≤ 0 were removed iteratively (most negative
first) in every lect; the pruned shared rules in small lects are listed by the file diffs vs the
group chain (e.g. RomWel lacks r̩ > u, m > v, the aspiration shift, rC simplification).

Engine wishes: (1) a way to condition the whole-word citation morphology on part of speech
(verbs vs nouns ending in -ati; the -el/-ʌr/-nu morph also hits nouns like yuvatí); (2) %source
blocks that run *before* the chain end (the eur -el/stem difference is really a citation choice
that would be cleaner before t > l); (3) per-rule "evidence count" in the frontier output so
zero-fire rules in small lects can be distinguished from harmful ones.

## 7. Romani's position: Central origin vs NW/Dardic affinity

(Added on the coordinator's request, ~03:00. Other groups' rule files were read only; the
copies used for scoring are in `tmp/sound_changes/central-east-pahari-romani/_x_*.sc`, `_n_*.sc`;
scripts `cross.py`, the normalised run inline in the session log, tags from
`results/tags.csv` written by `synth.py tags` at ~02:57.)

### 7.1 Cross-scoring other groups' chains on the migratory lects

All splits pooled (train + test; nothing was tuned on these pairs). Each cell is mean NED of the
borrowed chain run unchanged on the migratory lect's pairs (own `%exclude-source` of the target
lect kept, the donor's `%exclude-source` and `%source` blocks dropped). Left of the slash: the
donor chain as is; right: the donor with every `@morph`/`@notation` line stripped.

| chain (group) | eur | RomWel | RomBalk | as-ll | Lom | Zarg |
|---|---|---|---|---|---|---|
| no rules | 0.714 | 0.726 | 0.740 | 0.687 | 0.729 | 0.758 |
| the lect's own chain | 0.404 | 0.486 | 0.433 | 0.438 | 0.387 | 0.513 |
| eur (own group (Romani)) | 0.435/0.446 | 0.559/0.561 | 0.523/0.519 | 0.581/0.570 | 0.515/0.502 | 0.566/0.569 |
| as-ll (own group (Domari)) | 0.566/0.625 | 0.620/0.654 | 0.601/0.654 | 0.438/0.511 | 0.565/0.609 | 0.702/0.680 |
| H (Central (hindi)) | 0.688/0.689 | 0.735/0.719 | 0.724/0.731 | 0.708/0.684 | 0.624/0.632 | 0.674/0.674 |
| N (Pahari (own group)) | 0.630/0.673 | 0.681/0.695 | 0.666/0.708 | 0.666/0.661 | 0.596/0.622 | 0.630/0.665 |
| L (NW) | 0.676/0.659 | 0.690/0.667 | 0.708/0.697 | 0.680/0.658 | 0.643/0.619 | 0.708/0.692 |
| S (NW) | 0.723/0.720 | 0.730/0.717 | 0.740/0.738 | 0.733/0.690 | 0.680/0.672 | 0.759/0.756 |
| P (NW) | 0.680/0.657 | 0.700/0.662 | 0.702/0.685 | 0.699/0.663 | 0.653/0.624 | 0.703/0.689 |
| D (NW (Domaaki)) | 0.632/0.635 | 0.667/0.657 | 0.622/0.638 | 0.614/0.599 | 0.615/0.628 | 0.676/0.684 |
| K (Dardic-N) | 0.672/0.657 | 0.701/0.685 | 0.701/0.692 | 0.637/0.637 | 0.645/0.622 | 0.698/0.681 |
| Sh (Dardic-N) | 0.637/0.625 | 0.674/0.655 | 0.648/0.643 | 0.636/0.613 | 0.657/0.636 | 0.670/0.661 |
| Kho (Dardic-N (Chitral)) | 0.587/0.589 | 0.666/0.655 | 0.610/0.609 | 0.616/0.586 | 0.627/0.620 | 0.646/0.659 |
| Kal (Dardic-N (Chitral)) | 0.562/0.552 | 0.638/0.625 | 0.590/0.580 | 0.595/0.581 | 0.570/0.549 | 0.649/0.635 |
| Phal (Dardic-N) | 0.639/0.642 | 0.657/0.658 | 0.665/0.669 | 0.671/0.673 | 0.636/0.635 | 0.649/0.651 |
| Bshk (Dardic-S) | 0.624/0.624 | 0.634/0.634 | 0.617/0.617 | 0.636/0.636 | 0.620/0.620 | 0.659/0.659 |
| Tor (Dardic-S) | 0.638/0.635 | 0.658/0.657 | 0.653/0.654 | 0.652/0.640 | 0.636/0.639 | 0.674/0.676 |
| G (Raj.-Guj.) | 0.690/0.699 | 0.735/0.722 | 0.716/0.738 | 0.714/0.684 | 0.630/0.636 | 0.695/0.698 |
| M (Marathi) | 0.727/0.724 | 0.741/0.727 | 0.754/0.764 | 0.739/0.713 | 0.686/0.689 | 0.703/0.700 |
| WPah (W. Pahari) | 0.600/0.587 | 0.644/0.640 | 0.633/0.648 | 0.649/0.650 | 0.601/0.620 | 0.658/0.650 |
| Si (Insular) | 0.657/0.626 | 0.704/0.684 | 0.661/0.637 | 0.593/0.571 | 0.616/0.575 | 0.748/0.734 |
| Pk (Prakrit) | 0.687/0.685 | 0.718/0.718 | 0.711/0.709 | 0.643/0.644 | 0.638/0.639 | 0.747/0.748 |

(The eur chain scores 0.435 rather than 0.404 on eur here because its CDIAL `%source` block is dropped
in all cross runs.)

This raw comparison favours the Chitral Dardic chains (Kalasha best for every Romani variety,
Lomavren and Zargari; Sinhala/Kalasha/Khowar for Domari) and puts Hindi near the bottom
(0.68–0.73). But most of that gap is **typological/notational**, not historical: the Dardic chains
drop vowel length, nasal vowels and geminates, which Romani/Domari/Lomavren also lack, while the
Hindi/Pahari/Gujarati chains produce long vowels, nasal vowels and geminates everywhere. To
control for this, the second run applies the same notation-neutralising tail to every donor
output *and* to the gold forms (`V > $1-nas`, `V > $1-long`, `C $1 > $1`, retroflexes and ṣ →
dental / ś), again with @morph/@notation stripped:

| chain (group) | eur | RomWel | RomBalk | as-ll | Lom | Zarg |
|---|---|---|---|---|---|---|
| null (no rules) | 0.648 | 0.661 | 0.657 | 0.610 | 0.676 | 0.700 |
| eur (Romani (own)) | 0.442 | 0.521 | 0.516 | 0.534 | 0.489 | 0.493 |
| as-ll (Domari (own)) | 0.551 | 0.593 | 0.572 | 0.431 | 0.527 | 0.607 |
| H (Central) | 0.534 | 0.557 | 0.558 | 0.529 | 0.475 | 0.567 |
| N (Pahari) | 0.534 | 0.551 | 0.545 | 0.527 | 0.485 | 0.568 |
| L (NW) | 0.532 | 0.548 | 0.545 | 0.509 | 0.472 | 0.574 |
| S (NW) | 0.608 | 0.612 | 0.615 | 0.555 | 0.543 | 0.650 |
| P (NW) | 0.530 | 0.544 | 0.531 | 0.517 | 0.477 | 0.570 |
| D (NW (Domaaki)) | 0.552 | 0.571 | 0.554 | 0.504 | 0.537 | 0.608 |
| K (Dardic-N) | 0.584 | 0.614 | 0.624 | 0.553 | 0.551 | 0.609 |
| Sh (Dardic-N) | 0.527 | 0.559 | 0.538 | 0.513 | 0.538 | 0.569 |
| Kho (Dardic-N) | 0.576 | 0.606 | 0.599 | 0.539 | 0.607 | 0.602 |
| Kal (Dardic-N) | 0.538 | 0.571 | 0.574 | 0.538 | 0.535 | 0.559 |
| Phal (Dardic-N) | 0.529 | 0.538 | 0.532 | 0.564 | 0.545 | 0.562 |
| Bshk (Dardic-S) | 0.540 | 0.562 | 0.537 | 0.557 | 0.551 | 0.593 |
| Tor (Dardic-S) | 0.551 | 0.584 | 0.569 | 0.559 | 0.560 | 0.601 |
| G (Raj.-Guj.) | 0.561 | 0.567 | 0.578 | 0.531 | 0.487 | 0.592 |
| M (Marathi) | 0.592 | 0.599 | 0.614 | 0.564 | 0.535 | 0.610 |
| WPah (W. Pahari) | 0.527 | 0.555 | 0.549 | 0.562 | 0.543 | 0.553 |
| Si (Insular) | 0.606 | 0.632 | 0.613 | 0.522 | 0.553 | 0.653 |
| Pk (Prakrit) | 0.583 | 0.615 | 0.607 | 0.523 | 0.532 | 0.643 |

After normalisation the donors are nearly indistinguishable (eur: W. Pahari/Shina 0.527,
Palula 0.529, Punjabi 0.530, Lahnda 0.532, Hindi 0.534, Nepali 0.534, Kalasha 0.538; Kashmiri,
Khowar, Sindhi, Marathi, Sinhala clearly worse at 0.58–0.61). Punjabi/Lahnda are marginally best
for Welsh and Balkan Romani and for Lomavren (L 0.472, H 0.475, P 0.477); for Domari the best
borrowed chain is **Domaaki** (D 0.504), then Lahnda (0.509), Shina (0.513). No outside chain
comes close to the migratory lects' own chains (eur 0.404 unnormalised); the eur chain run on
the other migratory lects is better than any non-migratory chain for Welsh, Balkan and Zargari
(0.49–0.52 normalised) but *not* for Domari (0.534 vs its own 0.431; Domaaki 0.504) — Domari is
not simply "Romani".
**Whole-chain scores therefore do not decide the question**; the diagnostic information is in
individual features.

### 7.2 Feature by feature (other groups' tags from results/tags.csv)

"Share" = the feature appears as a rule (tag or rule text) in that group's chains; counts are
lects/group. R = Romani (eur, Welsh, Balkan, Zargari), Dom = Domari, Lom = Lomavren.

| feature | R | Dom | Lom | Central (Hindi grp) | NW | Dardic-N | Dardic-S | W. Pahari | innovation or retention? |
|---|---|---|---|---|---|---|---|---|---|
| post-nasal voicing nt > nd (@nasal-voicing) | yes | yes | (no evid.) | 0/10 | 10/12 | 8/11 | 6/12 | 12/16 | **innovation**, NW/Dardic/WPah, not Central |
| anusvāra kept as consonant, no nasal vowel (`ṁ > ∅`-type) | yes | yes | yes | 0/10 (nasal V) | 9/12 | 0/11 (nasal V 10/11) | 10/12 | 3/16 | Central nasal vowels are the innovation; R shares the NW/Kohistani *retention* |
| no gemination; clusters simplify directly (@cluster-simplify, no @assim-stop) | yes | yes | yes | 0/10 | 2/12 | 11/11 | 12/12 | 1/16 | retention of the pre-geminate stage, Dardic-type (NW geminate like Central) |
| Stop+r kept (tr, dr, pr) | yes (trin, drak, pʰral) | yes | yes | 0/10 | S tr > ṭr; P/L assimilate | Kal/Kho keep; Sh tr > ʦ̣, K r-loss | Mai tr > ʦ̣, Pas Cr > l | tr > ṭḷ | **retention**; shared with Chitral Dardic (Kal, Kho); Sindhi/W.Pahari keep an r-reflex |
| s + stop kept (vast, kašt) | yes | yes (ōśt) | **no** (st > tʰ) | 0/10 keep | 1/12 keep (D) | 5/11 keep (Kho, Kal …) | 5/12 keep | 0/16 | **retention**, shared with Chitral/Kohistani Dardic and Domaaki; Lom has the Central/NW innovation |
| intervocalic t d > l / r (not lost) | l | r | l | 0 | 0 | Kho t > r | Mai, Chil t d > l | 0 | **innovation** shared only with a few Dardic lects (Khowar, Maiyã, Chilisso) |
| lenition ordered *before* cluster simplification | yes (eur, Lom, Zarg, Welsh, Dom) | yes | yes | 0/10 | 0/10 | 6/11 | 4/12 | 0/15 | Dardic-type ordering (in 0 Central/NW/WPah chains) — caveat: orders are only as reliable as each file |
| intervocalic k g kept | yes | lost (Central-type) | kept | lost | lost | K, Sh, Kal lose; Kho keeps | Mai loses | lost | R retention (Kho-like); Domari innovation shared with Central |
| intervocalic aspirate > h | no | no | **yes** (voiced asp.) | 10/10 | 9/12 | 0/11 | 0/12 | 16/16 | Central/NW innovation absent in R/Dom, present in Lom |
| medial voiced aspirates deaspirated | yes | yes (everywhere) | no (devoiced) | 0/10 | 1/12 (D) | 10/11 | 10/12 | 5/16 | **innovation**, Dardic-type (and Domaaki; Sinhala, Nuristani) |
| initial voiced aspirates devoiced (bʰ > pʰ) | yes | **no** | yes (everywhere) | 0 | Khol, Pothwari; Dogri (unaspirated) | 0 | 0 | Dogri | R/Lom innovation; nearest parallels Punjabi-area (Pothwari, Khol, Dogri) |
| aspiration throwback to initial (jambʰa > cʰam) | yes | no | no | 0 | 0 | Palula (Phal, voiced) | 0 | 0 | R innovation; Palula parallel |
| ṣ merges with ś, s kept (śero vs sasto) | yes | no (s; š before stops) | no (s) | Sib > s 10/10 | Sib > s 11/12 | 3-way kept | mostly kept | **ṣ > ś, 12/16** | R shares the W. Pahari **innovation**; Dom/Lom share the Central/NW merger |
| kṣ | initial cʰ, medial kʰ | kʰ | kʰ | kkʰ | kkʰ | ʦ̣ʰ / cʰ | ʦ̣ʰ | kkʰ | medial kʰ = Central/NW innovation; not Dardic ʦ̣ʰ |
| ts, ps > cʰ (macʰo) | yes | yes | yes | ccʰ | ccʰ | **ʦ** (K, Sh, Kho, Kal) | ʦ | ccʰ | **Central/NW innovation, not Dardic** |
| ava > o | yes (not Balkan) | no | yes | 9/10 | 9/12 | 0/11 | 0/12 | 2/16 | Central/NW innovation |
| medial a-syncope | yes | yes | yes | 3/10 | 6/12 | 0/11 | 0/12 | 11/16 | Central/NW/WPah, not Dardic (late, possibly parallel) |
| r̩ > u after labials, else i | yes | yes | yes | H: i only | P, S | Kho | Dm | i | NW/Chitral pattern rather than Hindi's |
| vowel-length loss | yes | partly | yes | 0 | 0 | 4/11 | 6/12 | 0 | Dardic-type (typological, likely parallel) |

### 7.3 Conclusion

- **Shared innovations with Central/NW MIA**, i.e. changes Romani must have taken part in before
  it left: ts/ps > cʰ (against Dardic ʦ), medial kṣ > kʰ (against Dardic ʦ̣ʰ), ava > o, r̩ > i/u,
  loss of intervocalic y, v- > b-; plus medial a-syncope, which Central/NW/W.Pahari share.
  None of these is specifically *Hindi* rather than generally non-Dardic Indo-Aryan.
- **Shared retentions with Dardic**, which do not prove a subgroup: s+stop kept, Stop+r kept, no
  gemination/compensatory lengthening, no nasal vowels, intervocalic aspirates not > h, k/g kept.
  These match the *Chitral* lects (Kalasha, Khowar) most closely, and that is why Kalasha tops the
  raw cross-score.
- **Innovations shared with the NW/Dardic/W.Pahari zone but not with Central**: post-nasal
  voicing (nd), deaspiration of medial voiced aspirates (Dardic), t > l/r (only Khowar, Maiyã,
  Chilisso), ṣ > ś with s kept (W. Pahari), initial devoicing of voiced aspirates (Punjabi-area
  parallels). These are areal, lenition-type or phonotactic changes, which are easy to share
  through contact or to acquire in parallel.
- **Verdict (Romani):** the evidence fits **early departure from a Central(-type) MIA
  continuum**, before the Central geminate/aspiration/intervocalic-loss package (§4 graft test:
  the Hindi MIA block costs +0.07–0.09), with the retained clusters an archaism rather than a NW
  diagnostic. It is followed by a formative period in the **north-west** that added post-nasal
  voicing, dental lenition and aspirate devoicing. A genuinely Dardic origin is contradicted by
  the non-Dardic ts > cʰ and kṣ > kʰ and by the absence of Dardic ʦ and 3-way sibilants. Among
  the NW-zone lects, Romani's closest innovation partners are W. Pahari (ṣ > ś) and the
  Punjabi/Dogri area (initial devoicing). In the normalised cross-score Punjabi/Lahnda and
  W. Pahari are also marginally the best whole chains. In short: **Central origin, early exit,
  then NW contact** — the classical Turner position, refined.
- **Domari** is the most Central of the three in its obstruent system: it has the s-merger,
  Central-type loss of intervocalic velars/palatals and kṣ > kʰ. It **lacks** the Romani/Lom
  devoicing of voiced aspirates and aspiration throwback, and deaspirates everywhere. It shares
  with Romani t > r/l lenition, post-nasal voicing, s+stop and Stop+r retention. Its best outside
  chain is **Domaaki** (the Dom language of Nager), then Lahnda — and Domaaki's own chain
  shares Domari's two most distinctive traits, deaspiration of voiced aspirates and retention of
  s+stop (it is the only NW-group lect without @s-stop-aspirate). So Domari points to a separate,
  more Central/NW-influenced exit, not a Romani offshoot.
- **Lomavren** combines the Romani-type core (t > l, Stop+r kept, ts > cʰ, devoicing of voiced
  aspirates — here everywhere) with Central/NW innovations that Romani lacks: st > tʰ, voiced
  aspirate > h, s-merger. It sits closest to the NW/Central chains (normalised L 0.472, H 0.475,
  P 0.477), so it probably left later than Romani, or from a more Central/NW-assimilated
  variety. The sample is tiny (160 etyma), and the Armenian-contact phonology (d- > l-, total
  devoicing) is layered on top.
- **Caveats:** feature presence in other groups is read from their current tags and rule texts,
  which differ in granularity; orders are only as good as each file. The small migratory
  samples (111–512 etyma) make single-feature deltas of ±0.002 noisy. A synth.py bug (below)
  affects tag counting for ś.

Tooling note: `synth.py tags` truncates tags that contain a combining diacritic — `@sibilant-ś`
is recorded as `@sibilant-` in results/tags.csv (it matches Eastern `Sib > ś` rules too). The
tag regex should accept combining marks (e.g. `@[\ẁ-ͯ-]+`).

## 8. Round 2: vowels and the Vedic accent

(Scripts: `tmp/sound_changes/central-east-pahari-romani/vq.py` (variant scorer), `vdiff.py`
(etyma whose prediction changes), `finacc.py` (final vowel × accent tables); pre-round files in
`r2bak/`, start-of-restart files in `*.r2start.sc`, LOO dumps in `r2loo/`. All numbers are train
NED unless marked; Δ < 0 is better.)

### 8.1 Audit of the interrupted round-2 edits
Kept, all positive LOO: the Pahari tonic-CC lengthening (N/Ku/Dotyali), the `if:!numeral` guards
and `if:verb,!numeral` citation rules (all lects), the weight-insensitive syncope
`V (C) C _ C V` (eur, Domari, Balkan). Changed: Garh `if:!numeral a {i ē} > ṇ u` (LOO −0.0005,
subsumed by the `if:verb` rule) and the same Ku rule (LOO 0.0000) dropped. Reverted: the Welsh
umlaut `if:!verb a > e / _ C* i` (7 items; the same rule costs eur +0.0034, Balkan +0.0049,
Lom +0.0065, Domari +0.0139, so lexical, not a change). No lect's test NED had risen by
more than 0.005 over round 1 at the restart. No `%gold-drop` is used in this group.

### 8.2 Scores (round 1 → round 2)

| lect | train NED | test NED | test exact % |
|---|---|---|---|
| N | 0.3949 → 0.3822 | 0.3765 → 0.3620 | 15.9 → 16.7 |
| Ku | 0.4084 → 0.3906 | 0.3922 → 0.3836 | 15.5 → 16.2 |
| Garh | 0.3718 → 0.3594 | 0.3826 → 0.3781 | 23.2 → 23.2 |
| Dotyali | 0.4212 → 0.4147 | 0.4161 → 0.4101 | 7.6 → 6.1 |
| eur | 0.3990 → 0.3932 | 0.4258 → 0.4249 | 15.5 → 15.5 |
| as-ll (Domari) | 0.4385 → 0.4335 | 0.4369 → 0.4361 | 5.6 → 4.2 |
| Lom | 0.3845 → 0.3767 | 0.3996 → 0.3996 | 12.1 → 12.1 |
| RomWel | 0.4773 → 0.4668 | 0.5155 → 0.5266 | 9.1 → 9.1 |
| RomBalk | 0.4106 → 0.3924 | 0.5261 → 0.5053 | 3.1 → 3.1 |
| Zarg | 0.5188 → 0.4993 | 0.4935 → 0.4746 | 11.4 → 13.6 |

(Round-1 values are the round-1 files rescored with the round-2 engine, `r2_before.txt`; they
differ from §1 by ≤ 0.0015.)

RomWel test rose by +0.011: 33 test pairs, and 3 items account for it (naḍá > *ner* vs *nai*,
lakuṭa > *lekur* vs *ruvlō*, smárati > *smer* vs *śerav*), all from the a > e rule (8.3), which
the train split (−0.0077) and both sister lects (Balkan, Zargari: test −0.021, −0.019) support.
Kept.

### 8.3 Vowel/accent rules adopted (LOO on train, examples)

Pahari
- `{á í ú} > $1+long / _ C C` @degem-lengthen @stress (from the interrupted round, re-tested):
  N +0.0006, Ku +0.0012, Dotyali +0.0017; Garh ±0 (not added). Hits the root syllable of
  -ati verbs and nouns outside the penult: rákṣati > rākʰnu, cárman > cām, bárkara > bākʰro,
  mátsya > mācʰo, stámbʰatē > tʰāmnu. Accent controls: the same lengthening of the *non-tonic*
  first-syllable vowel of accented words: N +0.0001 (≈ as good as tonic, −0.0006), Ku +0.0006,
  Dotyali +0.0036; of first-syllable vowels in unaccented words +0.0036/+0.0015/+0.0045; of
  every short vowel before CC +0.0059 (N), +0.0032 (Ku). The Vedic tonic is the best conditioner,
  with a small margin in N.
- `á > o / _ #` @final-vowel-loss @stress, **Ku only**: +0.0021 (215 etyma). Control: the same
  rule for barytone final -a costs Ku +0.0015. Oxytone a-stems keep a final vowel (-o) more often
  (Ku oxytone -o 39% vs barytone 29%; `finacc.py`). NED gain comes with a 0.6-point fall in exact
  match, so it is a weak rule. The oxytone rule beats the barytone control in every Pahari lect:
  N +0.0009 vs +0.0017, Garh (-u) +0.0033 vs +0.0074, Dotyali +0.0009 vs +0.0149. Only in Ku does
  it beat no rule. **The accent does condition final-vowel retention in Pahari, weakly.**
- `if:starred a > o / _ #` @morph (N +0.0065, Ku +0.0112, Dotyali +0.0048; Garh −u costs
  +0.0014): Turner's *a-stems are back-formations from NIA masc. -o/-ā forms, so -o is the
  expected citation form. It is the stem-class answer to the -o residue (N *-ra → -o 39% vs -ra 25%).
  It works better late, next to -aka > -o (N −0.0065 late vs −0.0059 early).
- `{ā ā̃} > {a ã} / # C* _ C* {ā ā̃}` @x-cep-pretonic-shortening (new tag: NIA shortening of a
  first-syllable ā before a following long ā, i.e. before the NIA stressed syllable): N +0.0012
  (73 etyma), Ku +0.0010, Garh +0.0014; Dotyali rejects (+0.0023; it keeps final -ā). gamār <
  grāmadāra, bʰãṛār < bʰāṇḍāgāra, pacās < pañcāśát, masā̃t < māsānta, aŋār, kasār, ahān. Only ā
  before ā: extending it to ī ū, or to ā before ī ū ē ō, costs +0.0009…+0.0015. **Chronology:**
  it applies after final-vowel loss and hiatus contraction (moving it before apocope costs N
  +0.0022, Ku +0.0009, Garh +0.0046), so it is a NIA, positional stress effect. The **Vedic**
  version (pretonic long-vowel shortening `{ā ī ū} > short / _ !#* Acc`) is rejected everywhere:
  N +0.0047, Ku +0.0022, Garh +0.0110, Dotyali +0.0064.
- `if:numeral ṁ > ∅ / i _ ś` @x-cep-anusvara-loss (MIA -iṁś- > -īs- in the decades: bis, tis,
  -tālis; N +0.0008, Garh +0.0006).
- Order: the -nu/-ṇo/-ṇu/-yo citation morphology now follows degemination/lengthening
  (N −0.0012, Ku −0.0004, Garh −0.0010, Dotyali ±0).

Migratory
- `{a i u} > ∅ / V (C) C _ C V` @syncope (eur +0.0102, Balkan +0.0055): high vowels syncopate
  as well as a (garbʰíṇī > kʰabni, sarpiṇī > sapni, táruṇa > terno, kʰāditá > xal-o). High-vowel
  syncope costs Lom +0.0019 and Zarg +0.0028 (they keep open-syllable a-syncope `V C _ C V`) and is
  ±0 in Domari. Weight: the weight-insensitive rule beats open-syllable-only in eur (−0.0014),
  Balkan and Domari (−0.0008); Lom and Zarg prefer open-only.
- `{a á} > e / # C C* _ {non-labial C} V` @vowel-shift (Romani a > e in an open first syllable):
  Welsh +0.0077, Balkan +0.0095, Zargari +0.0166 (kʰer < gʰara, pʰen- < bʰánati, śel < śatá,
  del < dádāti, lel < lábʰatē, mer- < máratē, tern-). Labials block it (sov, nav, jov; the
  unrestricted rule is worse by 0.0025 in Balkan). Absent from the non-Romani lects: Lom +0.018,
  Domari +0.010. eur ±0 (+0.0004 NED, +1.1 points exact; not adopted). It must apply to the OIA
  syllable structure (moved after the verb-stem citation rules it costs Zarg +0.011). Accent:
  tonic-only and non-tonic-only versions split the gain inconsistently (Balkan tonic +0.0051 /
  non-tonic −0.0095; Zarg 0 / −0.0084): **not accent-conditioned.**
- Domari `u > ∅ / _ #` replaces `{i u} > ∅` (−0.0022): final -i is kept (máni, sáli, asti, vasti,
  dī́li); -u is lost (vāyú > vai).

### 8.4 Hypotheses tested and rejected (Δ train NED)

Accent
- Romani syncope and the accent (eur, marker-vowel test that keeps the tonic vowel through the
  chain; sanity run identical to base): tonic vowel spared ±0.0000; syncope only pretonic in
  accented words +0.0015, only post-tonic +0.0013. (a-only version: tonic spared −0.0003, 4
  items.) **Romani syncope is accent-blind.**
- Romani/Domari final vowels vs accent: oxytone -á > -o eur +0.0018 (barytone +0.0001), Domari
  +0.0151, Welsh +0.0026, Zarg +0.0132; final -a/-ā/-i distributions do not differ between
  oxytones and barytones (eur -o 33% vs 32%; Domari -a 32% vs 30%). Domari: losing -a only in
  barytones +0.0012, only in oxytones +0.0041; losing barytone -i +0.0082, oxytone -í +0.0024.
  eur keeps final -i (−0.0028 for dropping i from apocope), but its oxytone-only version gives
  −0.0009 and the others reject the change (Balkan +0.0016, Zarg +0.0064, Lom +0.0036); not
  adopted. **No accent effect on final vowels in the migratory lects.**
- Pahari penultimate lengthening vs accent: restricting `V G $2 > long / _ V #` to unaccented
  words (the Vedic tonic rule alone serving accented words) costs N +0.0024, Ku +0.0025, Garh
  +0.0089, Dotyali +0.0106. **The penult (positional) lengthening is not replaceable by the Vedic
  accent;** the accent only adds the smaller tonic-CC rule.
- Vocalic r̩ (N, Ku): tonic ŕ̩ > a +0.0002/+0.0003, tonic ŕ̩ > i +0.0001, r̩ > a before i
  +0.0006/+0.0002, r̩ > u before u +0.0002/0. Too few accented r̩ (≈12 per lect).
- Pretonic Vedic syncope in N (`a > ∅ / V C _ C !#* Acc`) +0.0007; post-tonic +0.018.
- Pretonic ū > u (Garh −0.0014, 7 items; N/Ku ±0) and post-tonic long-vowel shortening in
  Dotyali (−0.0022, 4 items, all fem. -ā) are too small or artefacts.

Other vowel tests
- NIA schwa syncope in N/Ku (`a > ∅ / V C _ C V`, before or after apocope): −0.0002/+0.0011
  (N before apocope +0.0012). Pahari has no medial syncope.
- Accent-blind lengthening of every short vowel before CC: N +0.0059, Ku +0.0032;
  first-syllable (positional) lengthening N +0.0051.
- Ku hiatus shortening ā > a / _ V (suggest −0.0009) is mostly the -āuṇo causatives; not adopted.
- Zargari a > o before u (−0.0095, 4 items incl. kaḥ punar), medial i > e (−0.0049, 3 items),
  ō > u in open syllables (−0.0095, 4 items; costs eur +0.0073, Balkan +0.0189): rejected as too
  few items.
- Welsh umlaut (8.1); Domari/Lom a > e (8.3).

### 8.5 What the accent does and does not condition in this group
- **Does (weakly):** (i) closed-syllable lengthening of the Vedic tonic vowel before an OIA
  cluster outside the penult in Pahari (N, Ku, Dotyali; +0.0006…+0.0017); (ii) retention of
  final -a as -o in Pahari oxytone a-stems: better than barytones in all four lects, better than no
  rule only in Ku.
- **Does not:** Romani/Domari/Lom medial syncope (accent-blind; weight-insensitive in Romani);
  Romani/Domari final-vowel retention; Romani a > e; Pahari pre-stress shortening, which is
  NIA-positional (ā before a later long ā, after apocope), not Vedic-pretonic; the Pahari
  penultimate lengthening, which is positional and not replaceable by the tonic.
- The -o residue of Nepali/Kumaoni is a **stem-class/citation effect** (Turner's starred
  back-formations) far more than an accent effect: the starred rule gains −0.0048…−0.0112, the
  oxytone rule 0…−0.0021.

New tag: `@x-cep-pretonic-shortening` — NIA shortening of a first-syllable long vowel before a
later (stressed) long vowel (N, Ku, Garh). Engine note: the built-in `V`/`Acc` classes cannot
be used in set differences (`{V - á}`, `{Acc - á}` match only unaccented vowels), so
accent-sensitive late rules need marker vowels (ə, ä ï ü) inserted before `@acc-loss` and
removed after the rule. A way to keep the accent as a feature through the chain would help.

## 9. Accent score

(Script `tmp/sound_changes/central-east-pahari-romani/vqa.py` scores variants on train accent
NED; `accpos.py` tabulates where attested accents fall; `accerr.py` lists mismatches. Files
before this round: `*.r3start.sc`. Plain train/test NED is identical before and after every
edit (verified). No rule or `%source` block in this group writes accent marks. The other lects
have ≤ 11 accent-marked train pairs (Ku 11, Zarg 7, Garh/Lom 4, N/RomWel 2) and were not
tuned.)

### 9.1 Results

| lect | acc. pairs tr/te | train aNED none / Vedic / final | test aNED none / Vedic / final |
|---|---|---|---|
| as-ll (Domari) | 152/37 | 0.4339 / 0.4409 / 0.4022 | 0.4483 / 0.4660 / 0.4410 |
| RomBalk | 46/13 | 0.4362 / 0.4542 / 0.4305 | 0.5208 / 0.5327 / 0.5111 |
| eur | 28/3 | 0.4799 / 0.4889 / 0.4710 | 0.5833 / 0.5833 / 0.5595 |

Adopted (LOO = train accent-NED rise when the rule is removed):
- Domari `V > $1+stress / _ C* V C* #` at the end of the chain, @accent-placement, **+0.0317**.
  Macalister marks the penult of the attested word in 140 of 152 pairs (máni, tʌ́rnʌ, jaúʌr,
  dṓni); monosyllables are unmarked. Most remaining misses are segmental: the chain has one
  syllable too many (duhitŕ̩ > *duhíri* vs dī́ri, bíḍāla > *birála* vs blári; 20 pairs) or too few
  (-ʌr verbs jaúʌr vs *jábar*; 17). The rule gives the same score placed before apocope or at the
  end, because Domari keeps final -a/-i. Placed at the start of the chain (on the OIA penult) it
  gains only −0.0079, and before syncope −0.0303.
- RomBalk and eur `V > $1+stress / V C* _ C* #` at the end of the chain (last syllable of a
  polysyllabic output), @accent-placement, **RomBalk +0.0057, eur +0.0089**. Balkan Romani marks
  the last syllable in 45 of 46 train pairs (oxytone). But in 34 of them the accented syllable is
  the gender/extension suffix that the chain cannot predict (koró, pinró, cʰurí, makló), so the
  predicted monosyllable must stay unstressed. Stressing every last syllable, monosyllables
  included, costs RomBalk +0.0625 and eur +0.0122. Stressing only open finals: −0.0018 / −0.0009.
  The rule must follow apocope: before apocope or syncope it gains only −0.0005 in RomBalk.

### 9.2 Rejected (Δ train accent NED against the base without placement unless stated)
- Domari: last syllable +0.0540; first syllable −0.0314 (≈ penult, since most words are
  disyllables); first long vowel −0.0314; weight-sensitive "last if closed, else penult" +0.0252
  against the penult rule; "last if long" ±0 (length is already neutralised by then); verbs penult /
  others first syllable +0.0003 against penult; "first syllable if ≥ 3 syllables, else penult"
  −0.0010 against penult. Not adopted: it only compensates for missing syncope in the segmental
  chain.
- Romani: penult of polysyllables RomBalk +0.0052, eur +0.0155; first syllable +0.0777 / +0.0417.

### 9.3 The Vedic accent, given its best chance
- Vedic carried, alone: Domari +0.0069, RomBalk +0.0179, eur +0.0089. It is worse than no
  accent in all three.
- Vedic for accented etyma plus the positional rule for unaccented ones: Domari 0.4273, against
  0.4204 when accented etyma get nothing and 0.4022 when they get the penult. RomBalk 0.4506
  against 0.4327 (nothing) and 0.4305 (positional); eur 0.4880 against 0.4790 and 0.4710.
- Positional rule everywhere, with Vedic only as a fallback for monosyllables: RomBalk +0.0152
  and eur +0.0009 against the positional rule alone.
- **Conclusion:** in Domari and Romani the attested accent is a new positional stress: penult of
  the word in Domari, last syllable (usually the suffix) in Romani. It is counted on the NIA
  output after apocope. The Vedic accent predicts it worse than no accent at all, in accented
  etyma too. This agrees with §8: the Vedic accent conditions nothing segmental in the
  migratory lects either.

## 10. Extension-aware score

(All numbers use the 7-item menu aka, akka, ikā, uka, aḍaka, allaka, araka. Scripts in
`tmp/sound_changes/central-east-pahari-romani/`: `vqe.py` scores variants for ext-NED, plain NED
and accent NED together; `extshow.py` prints a headword's variants; `extres.py` lists residual
ending mismatches under the best-variant choice; `menu.py` scores menu additions and drops in
memory without touching the engine. Output is in `r4/`; round-4 start files are
`*.r4start.sc`. Train accent NED is unchanged in every lect. The accent numbers here come from
the updated engine and differ from §9: Domari train aNED is now 0.4160.)

### 10.1 Scores (before → after; ext = extension-aware NED)

| lect | train NED | train ext | test NED | test ext | train ending choice after (bare / aka / ikā / akka / uka / aḍaka / allaka / araka, %) |
|---|---|---|---|---|---|
| N | 0.3822 → 0.3823 | 0.3144 → 0.3093 | 0.3620 → 0.3630 | 0.2978 → 0.2930 | 63 / 11 / 12 / 5 / 4 / 3 / 2 / 0 |
| Ku | 0.3906 → 0.3908 | 0.3383 → 0.3333 | 0.3836 → 0.3860 | 0.3325 → 0.3326 | 70 / 13 / 6 / 3 / 1 / 3 / 1 / 1 |
| Garh | 0.3594 → 0.3603 | 0.3001 → 0.2927 | 0.3781 → 0.3784 | 0.3140 → 0.3120 | 71 / 19 / 3 / 1 / 0 / 3 / 1 / 1 |
| Dotyali | 0.4147 → 0.4153 | 0.3448 → 0.3355 | 0.4101 → 0.4101 | 0.3335 → 0.3207 | 57 / 18 / 6 / 12 / 2 / 4 / 2 / 0 |
| eur | 0.3932 → 0.3930 | 0.3368 → 0.3207 | 0.4249 → 0.4236 | 0.3731 → 0.3606 | 66 / 20 / 9 / 1 / 0 / 1 / 2 / 0 |
| as-ll (Domari) | 0.4335 → 0.4335 | 0.4030 → 0.3992 | 0.4361 → 0.4381 | 0.4081 → 0.4039 | 73 / 3 / 15 / 2 / 4 / 2 / 1 / 0 |
| Lom | 0.3767 → 0.3767 | 0.3562 → 0.3471 | 0.3996 → 0.3996 | 0.3880 → 0.3743 | 84 / 4 / 7 / 0 / 1 / 1 / 3 / 0 |
| RomWel | 0.4668 → 0.4656 | 0.4335 → 0.4292 | 0.5266 → 0.5266 | 0.4592 → 0.4562 | 78 / 11 / 5 / 3 / 1 / 1 / 2 / 0 |
| RomBalk | 0.3924 → 0.3905 | 0.3336 → 0.3123 | 0.5053 → 0.5053 | 0.4467 → 0.4123 | 61 / 21 / 13 / 2 / 0 / 2 / 2 / 0 |
| Zarg | 0.4993 → 0.4993 | 0.4824 → 0.4595 | 0.4746 → 0.4689 | 0.4356 → 0.3639 | 80 / 11 / 5 / 1 / 3 / 0 / 1 / 0 |

Test accent NED changed only in Domari (0.4323 → 0.4362, 37 pairs, from the `a a > a` rule) and
Zargari (one pair).

### 10.2 Rules changed (LOO = rise in train ext-NED when the rule is dropped; plain Δ in brackets)
- N, Ku: `if:starred a > o / C _ #`. The rule now fires only after a consonant, so -uka/-ikā
  variants no longer get a doubled ending (N *cōra-uka gave *coruo*). Its ext value is **negative**:
  dropping it would gain ext −0.0073 (N), −0.0057 (Ku) and −0.0102 (Dotyali, rule left
  unrestricted there because the C-only version costs plain +0.0060). But plain NED needs it
  (+0.0064, +0.0113, +0.0048 without it), so it stays. Under the extension-aware score it is
  redundant with the -aka variant. It is the plain-NED stand-in for "Turner's starred etyma
  continue -aka". The main residual in N and Ku is o→∅ (N 262, Ku 439): starred etyma whose bare
  -C reflex no variant can produce while the rule is on.
- N `a > ∅ / V C _ C {o ā i u} #` @syncope (+0.0032 [−0.0003]); Garh the same (+0.0074
  [−0.0009]); Ku with a heavy (long-vowel) first syllable only, `{ā ī ū ē ō} C _ C …` (+0.0019
  [−0.0001]; the general version costs Ku plain +0.0015). Penultimate a is lost before the
  gender/extension vowel, so -aḍaka/-allaka/-araka come out as -ṛo/-lo/-ro (ṭukro, bākʰro)
  instead of *-aro/-alo*. Dotyali rejects it (plain +0.0013).
- Ku `{i ī} {a ā} > ī / _ #` @hiatus (+0.0024 [0]) and Dotyali the same (+0.0093 [−0.0007]):
  -ikā > fem. -ī (kʰaṛī, cā̃dī). Dotyali previously produced *-iyā*.
- eur, RomBalk, RomWel, Zarg, Lom `{i ī} k ā > ī / _ #` @morph @aka (eur +0.0155 [−0.0002],
  RomBalk +0.0213 [−0.0019], RomWel +0.0043 [−0.0012], Zarg +0.0044, Lom +0.0091), plus eur/Zarg
  `u k a > ū / _ #` (+0.0006, +0.0020). Romani keeps intervocalic k, so -ikā used to come out as
  *-Ck* after syncope. The suffix k is lost as in `a k a > o`, and the long ī survives apocope
  (cʰuri, kangli, rakli). Plain NED also improves.
- Zarg `a k a > o / _ #` @morph @aka (+0.0165 [0]); Lom rejects it (−0.0013 only, not added).
- Domari `a a > a / _ #` @hiatus after the loss of intervocalic k (+0.0038 [0]): -aka merges with
  the -a stem. **Not adopted, blocked only by the accent constraint:** `{i ī} {a ā} > i / _ #`
  (-ikā > -i: ext −0.0247 more, plain −0.0013, *and* train accent NED −0.0039, i.e. better). Adding
  `u a > u` gives a spurious ext −0.066: -uka then deletes like -u, so "uka" stands in for the bare
  consonant-final form. Rejected.

### 10.3 What the choices say
- **-aka is the main extension everywhere:** Garh 19%, Dotyali 18% (+12% -akka, which yields
  -āk forms), RomBalk 21%, eur 20%, Ku 13%, N 11%, Zarg 11% (after the new -o rule). In Domari and
  Lom (3–4%) -aka merges with the bare stem.
- **-ikā is the second extension** (N 12%, RomBalk 13%, Domari 15%, eur 9%), i.e. feminine -ī
  (N -i, Ku -ī, Romani -i, Domari -i).
- -ḍa/-la/-ra extensions (aḍaka + allaka + araka) are productive only in Pahari (N 5%, Ku 5%,
  Garh 5%, Dotyali 6%) and marginal in the migratory lects (≤ 3%).
- **araka is never chosen** except in Ku/Garh (1%). In N, Dotyali, eur and Domari it gives the same
  output as aḍaka (ḍ > r), so ties go to aḍaka. That is a menu redundancy for lects with ḍ > r,
  not an absence of -ra.

### 10.4 Should the menu change? (`menu.py`; Δ train ext-NED, reflexes choosing the item)

| candidate | N | Ku | Garh | Dotyali | eur | Domari | Lom | RomBalk | Zarg |
|---|---|---|---|---|---|---|---|---|---|
| +ī (plain fem.) | −.0014 (27) | −.0009 (13) | −.0009 (3) | **−.0099 (14)** | **−.0153 (51)** | **−.0257 (60)** | −.0095 (9) | **−.0201 (17)** | −.0057 (5) |
| +iya / īya | 0 | 0 | 0 | −.0054 (9) | = +ī | 0 | 0 | −.0087 (13) | = +ī |
| +ikka | −.0004 (14) | −.0031 (96) | −.0019 (16) | −.0012 | 0 | −.0004 | 0 | 0 | 0 |
| +āla / ālu | −.0008 (32) | −.0005 | −.0002 | −.0007 | −.0055 (29) | −.0008 | **−.0154 (16)** | −.0036 (5) | −.0041 (5) |
| +ḍa (no -ka) | −.0014 (33) | −.0034 (64) | −.0030 (13) | −.0013 | −.0028 | −.0028 | −.0048 | −.0025 | −.0048 |
| +ra (no -ka) | −.0037 (116) | −.0049 (110) | −.0044 (17) | −.0054 | −.0012 | −.0006 | −.0018 | −.0013 | −.0024 |
| +la (no -ka) | −.0036 (109) | −.0050 (111) | −.0067 (26) | −.0054 | −.0026 | −.0007 | −.0017 | −.0025 | −.0025 |
| +ā (final ā) | −.0094 (258) | **−.0192 (370)** | −.0044 (19) | **−.0166 (29)** | −.0020 | 0 | −.0027 | 0 | −.0041 |

(These numbers were computed before the -ikā rules of 10.2. With -ikā > -ī in place, the Romani
+ī gains shrink by roughly the ikā rule's own value.)

Recommendations:
- **Add plain feminine -ī** (the MIA -ī feminine, not a surface fix). It is the largest real gain
  for Domari (60 reflexes), European and Balkan Romani and Dotyali. It also removes the need for
  chain rules like the -ikā contraction when a lect loses the -k- differently.
- **Add -ra and -la without -ka** (MIA -ra-/-la- extensions whose NIA reflex has no gender vowel,
  N/Ku -r/-l nouns): about 110 reflexes each in N and Ku, all Pahari lects gain 0.004–0.007.
  -ḍa without -ka is smaller (Ku 64).
- -ikka (Pk -ikka) helps only Ku (96 reflexes, −0.0031): optional.
- -āla/-ālu: only Lomavren (−0.0154, 16 reflexes; its -al/-av nominal) and eur (−0.0055). These
  are real (MIA -āla), but I would add it only if other groups need it.
- Do **not** add bare "-ā". It is the largest Pahari gain (Ku 370 reflexes), but it only relabels
  the stem vowel: the Ku/N -ā nouns are a mix of Hindi-type masc. -ā loans and fem. ā-stems, not
  an extension.
- **Drop or merge araka** for lects with ḍ > r (never chosen in N, Dotyali, eur, Domari, Lom,
  RomBalk, Zarg). **uka** is chosen by ≤ 4% (0–1 reflexes in Garh, eur, RomBalk), so it is low
  value outside N and Domari. Every other item is chosen somewhere.

## 11. Verb citation forms (round 5, cut short at 18:27)

**Nepali** is the only lect done. The citation form -nu continues the **neuter verbal noun
-anakam**, which gives MIA -aṇaaṁ, then -nu. Three sound changes were added and one existing
rule was already doing the syncope. Every change passed all four gates (train plain NED
0.3835 → 0.3824, train ext 0.3194 → 0.2894, accent unchanged):
- `a a m > u / _ #` @final-nasal @hiatus: neuter -aaṁ gives -u. Placed before final-consonant
  loss. Verb-menu NED −0.105.
- `v > u / V _ C` after v-loss: causative -āv-nu gives -āunu. −0.0057.
- `V G $2 > $1+long $3 / _ V C V #` @degem-lengthen: compensatory lengthening in the
  antepenult, giving rākʰnu. −0.0023.
- The round-4 syncope `a > ∅ / V C _ C {o ā i u} #` already turns -anu into -nu.

| lect | train verb NED, plain / menu | test verb NED, plain / menu, before → after |
|---|---|---|
| N | 0.3250 / 0.3696 → 0.2567 | 0.3019 / 0.3502 → 0.2995 / 0.2365 |

The verb menu now beats the plain chain. Winning forms (test): anakam 191, bare 18, anīya 15.

**Not done:**
- **Ku** needs syncope before -ṇo and n > ṇ ordered before the syncope.
- **Garh** needs the same neuter rule plus ṇ after consonants.
- **eur** and the other migratory lects: their bare stem wins, with no citation affix.
- **Dotyali** has no verbs flagged.

Before-files are in `tmp/sound_changes/central-east-pahari-romani/r5/`.

## Round 6 audit

Tools used: `residuals.py`, `gates.py`, and the scratch scripts `tmp/sound_changes/central-east-pahari-romani/subs.py`
(train-only counts of aligned pred→gold operations with their context), `extdiff.py` (words a candidate
makes better or worse under the extension-aware choice) and `cand.sh`/`mk.py` (candidate → gates). Files
before the round are in `…/r6/pre/`; the final gate runs are in `…/r6/final_gates.txt`. Every kept change passed
`gates.py` alone and again in the cumulative before→after run below.

### Results (before → after: train/test)

| lect | ext NED | verb-menu NED | plain NED | exact (plain) |
|---|---|---|---|---|
| N | .2693→.2629 / .2543→.2483 | .2052→.1925 / .1881→.1758 | .3889→.3890 / .3722→.3725 | 14.8→14.8 / 16.6→16.6 |
| Ku | .3269→.3005 / .3258→.2934 | .3132→.2350 / .3229→.2282 | .4020→.4041 / .3940→.3900 | 14.2→14.8 / 14.9→15.9 |
| Garh | .3018→.2786 / .3229→.2922 | .2720→.1735 / .2499→.1380 | .3608→.3591 / .3788→.3737 | 23.4→24.1 / 23.2→24.5 |
| Dotyali | .3252→.3243 / .3106→.3078 | – | .4202→.4203 / .4073→.4028 | 13.1→13.1 / 7.6→7.6 |
| eur | .3282→.3127 / .3605→.3415 | .3591→.3078 / .4183→.3641 | unchanged | unchanged |
| as-ll (Domari) | .3709→.3601 / .3949→.3847 | .4115→.3294 / .5170→.3952 | unchanged | unchanged |

Accent NED is unchanged, except Ku, where it improves (.7483→.7271).

Error profile (N, the largest; the others are similar): ending 37%, consonant insertion or
deletion 21%, consonant quality 9%, missing vowel 9%, vowel quality 7%, missing syncope 6%. The
dominant single residuals are menu or lexical, not sound change. N/Ku -āk vs -ā (the -akka
variant against Hindi-type -ā nouns, about 100 pairs each) and N -iyā vs -iyo are lexical; so
are -uvā (pūpa > puvā), the numerals, and compounds or reduplicatives.

### Kept changes
- **N**
  - Second-syllable syncope when two syllables follow:
    `a > ∅ / # C* V C _ C a C {u o ā i} #` (úttarati > utranu, *ākkaḍa > akranu, *ummal > umlanu;
    79 pairs better, 28 worse, the losers being roots with a long second vowel: gʰusārnu,
    masārnu). Ext −0.0018, verb −0.0066.
  - Ya-śruti glide before final -o: `∅ > y / {ā i e} _ o #` (āyo, hā̃siyo; 75 better, 7
    worse). Ext −0.0025.
  - Order fix: -aka > -o (`hiatus.aa.final`) now runs *before* the penultimate lengthening, so
    geminate stems lengthen (agni-aka > āgo, kaṣṭa-aka > kāṭʰo). Ext −0.0006.
  - The round-5 antepenult lengthening is restricted to the first syllable (`# C* _ V C V #`),
    because it lengthened second-syllable geminates (*ut-śakna > *ucʰāknu). Ext −0.0016,
    verb −0.0061.
- **Ku**
  - Order fix: `n-retroflex` (n > ṇ) now runs before the syncopes, and a new rule
    `a > ∅ / V C _ ṇ o #` follows it (utarṇo, ucʰalṇo, āṭṇo, ā̃kṇo). The first version had the
    syncope before n > ṇ, so it never fired; fixing that gave another ext −0.006. Together:
    ext −0.015, verb −0.070.
  - `v > ∅ / ā _ ṇ`: causative -āvaṇo > -āṇo (agʰāṇo). Verb −0.0094.
  - `hiatus.aa.final` moved before the penultimate lengthening, as in N. Ext −0.0029.
  - **Removed** the round-2 oxytone rule `á > o / _ #`. Under the extension-aware score it
    duplicates the -aka variant and hides the bare -C reflex (kālá > kāl, cittá > cit). Ext
    −0.0063 and test plain −0.0039, but train plain +0.0020. This is the bare-vs-extended
    artefact the brief allows: the -o forms are -aka derivatives, and exact match rises
    14.3 → 14.8%.
- **Garh**
  - `n-retroflex` moved before the syncope (-ṇu after consonants: kosṇu, rakʰṇu, bacṇu).
    Ext −0.018, verb −0.079.
  - n-dissimilation extended to ṛ: `ṇ > n / {r ṛ ṇ ḷ l} _` (ucaṛnu, upāṛnu). Ext −0.0018,
    plain −0.0018.
  - `ā v > o / _ ṇ`: causative -āvṇu > -ɔṇu, as in the secondary āu > ɔ (jaḷɔṇu, pilɔṇu).
    Verb −0.0121.
- **Dotyali**: added the missing `final-cons-loss` (every other Pahari chain has it; -akam
  gave *-ām*). Ext −0.0010, test plain −0.0045.
- **eur**: i-umlaut of the 3sg ending, `a > e / _ l i #` after t > l: -ati > -ali > -eli > -el
  (pʰerel, anel, avel). Ext −0.0155, verb −0.051. It is rejected by the small lects that cite
  bare stems (RomBalk, Zarg, Lom); Welsh ±0.
- **Domari**: `i > ∅ / a r _ #` after t > r: -ati > -ari > 3sg -ʌr (kílʌr, dáuʌr), while other
  final -i stays. Verb −0.082, ext −0.011.

### Rejected
- N intervocalic h-loss (+0.0008 ext, +0.0046 test ext): N keeps h, unlike Ku and Garh.
- N `n > ∅ / i _ #` (in-stems) −0.0002 and `a i > i / _ #` −0.0003: too few items.
- N `degem-lengthen.vg.vv`: verb test +0.0023.
- N dropping the antepenult lengthening: mixed results; restricted instead.
- Ku final `l > v`: 35 better, 54 worse. Ku sources have both -l and -v (ā̃cal/ā̃cav), which
  is source variation.
- Ku general syncope before n/ṇ/l/r + o/ī: ext +0.017.
- Garh dropping final -ī shortening: plain +0.0032. The sources disagree on ī/i.
- Garh hiatus reorder: ±0.
- Domari early final -a loss: accent +0.09, plain +0.015.
- Lom initial kʰ > x: +0.007. The source writes both.
- eur `y > v / a _ a` (-aya- > -ava-): −0.0006 only, and it creates bad v-clusters through
  syncope; not kept.

### Not done / left alone
Lom, RomWel, RomBalk and Zarg were inspected. Their residuals are Zargari inflected forms,
kaḥ punar, the Balkan source's `i̦` notation and Lomavren -av nouns: lexical or source
variation, so the chains are unchanged. N -āk/-ā and -iyo/-iyā need a menu item or are
lexical. N numerals are untouched (numeral conditions are not allowed). Domari initial k/x
and its light-verb compounds are source variation.

### Round 6 second pass

**The four small lects** (`residuals.py` profiles; share of the error mass):

| lect | missed pairs | ending | C ins/del | C quality | V quality | missing syncope | missing V |
|---|---|---|---|---|---|---|---|
| Lom | 145 | 57% | 15% | 9% | 5% | 6% | 6% |
| RomWel | 142 | 50% | 14% | 10% | 10% | 9% | 3% |
| RomBalk | 129 | 49% | 21% | 12% | 6% | 5% | 4% |
| Zarg | 165 | 42% | 24% | 14% | 7% | 4% | 5% |

The regular residue that was tested (all rejected by the gates):
- RomBalk nr > rn metathesis (pirnó, arno, xarno): train ext −0.0015, but test accent +0.028 and
  test plain +0.0036.
- RomWel final -o > -ō (the source writes long ō: kerō, cavō): train ext −0.0225, but test ext
  +0.0041 on 33 pairs.
- Zarg final nd > n (dān, kʰān, cʰin): ext −0.0002, plain +0.0023.
- Lom initial kʰ > x (plain +0.0069, see above).

The rest is lexical or source-specific:
- **Lom:** the -av nominal suffix (gilav, pelav), ə-reduction written by one source (kʰəl, nəkl).
- **RomWel:** long-vowel notation.
- **RomBalk:** the `i̦` notation for ṇḍ/nr and ṭṭ.
- **Zarg:** inflected verb forms (-ipey, te-…-es), kaḥ punar > kon/kos/qostu, Persian-style q.

**Deeper pass on N, Ku, Garh, eur** (non-ending classes; `subs.py` context counts and
`extdiff.py` better/worse lists). None of these passed:
- **N h-loss before i/u after a/ā** (dai, karāi, bāira): 30 better, 14 worse, ext −0.0004 but
  test ext +0.0001. Both variants are attested (dahi/dai, pahiyā), so this is variation. It
  stays out, like the general intervocalic h-loss.
- **N initial d > ḍ before ā** (ḍā̃s, ḍā̃ṛ): 16 better, 44 worse. Lexical.
- **Ku -ama- > -ãũ-** (bʰãũr, tãũri, asãũ): when placed late it also hits old -mm-/-mb-
  (kambala; 14 better, 20 worse). At the MIA stage it gives only −0.0001 ext. Not kept.
- **eur mb > m** (cam, cumi): 2 better, 14 worse; it feeds intervocalic m > v, and both camb and
  cam are attested.
- **eur -aya- > -av-** (calavel): blocked by the medial syncope (it gives *calvel*). It needs a
  syncope exception before v, and there is not enough data.

Remaining non-ending classes, as read:
- **N:** the passive/intransitive -inu (gʰisinu, cʰalinu, oirinu, about 50 pairs) needs an
  -iyanakam verb-menu item (Pk -ijja-). This is a menu question, not a sound change; untested.
- **N, Ku:** the ordinals -ama > -ãũ.
- **Ku, Garh:** secondary vowel shifts written by one source (ɔ, ε, ʌ; for example kā̃mav).
- **Garh:** k > g after a nasal vowel (pā̃gu, bāŋgu, 4 pairs).

Final numbers are unchanged from the first pass (table above): the second pass added no rules.

### Round 6 third pass

**(1) Verb-menu item -iyanakam (passive -ya- + -anakam) against a placebo.** Script:
`tmp/…/menutest.py`, which monkeypatches `evaluate.VERB_MENU` only. Verb-menu NED, train/test:

| lect | base | +iyanakam (winner count) | +iranakam placebo (winner count) |
|---|---|---|---|
| N | .1925/.1758 | .1849/.1686 (115) | .1887/.1718 (74) |
| Ku | .2350/.2282 | .2275/.2219 (75) | .2336/.2270 (20) |
| Garh | .1735/.1380 | .1709/.1371 (7) | .1726/.1380 (1) |
| eur | .3078/.3641 | .3068/.3641 (2) | .3068/.3641 (3) |
| Domari | .3294/.3952 | .3268/.3952 (1) | .3268/.3952 (1) |

Dotyali has no verbs.

- In N and Ku the real item beats the placebo clearly: about half the N gain and five-sixths
  of the Ku gain are specific to it.
- With the menu item in place, N also needs a hiatus contraction `i a > i / _ n u #`
  (-iyanu > -inu: gʰisinu, cʰalinu, oirinu). Under the patched menu the item plus that rule
  gives N verb .1925 → .1767 train and .1758 → .1586 test, ext −0.002, and passes the gates.
- Under the official menu the rule is neutral (verb test +0.0007) and was **not installed**.
- **Recommendation: add -iyanakam** (Pk -ijja-, the N/Ku passive-intransitive -inu/-iṇo).
  It is not wanted for the Romani lects or Domari.

The same test for a bare causative stem -āpaya, with placebo -āraya (Romani -av, the
causative cited as a bare stem):

| lect | base | +āpaya | +āraya |
|---|---|---|---|
| RomWel | .4057/.3431 | .3945/.3315 | .4046/.3319 |
| RomBalk | .3090/.4390 | .3004/.4057 | .3040/.4390 |
| Lom | .2967/.4706 | .2948/.4574 | .2950/.4706 |
| eur | .2950/.3728 | .2944 | .2940 |

(The eur and RomWel base values differ from earlier tables: the engine changed during the
round.) The evidence is modest, from 3–6 verbs per lect. Adding -āpaya is worth considering
only if other groups see the same.

**(2) Near-misses** (NED ≤ 0.25, a single differing segment; `near.py`):
- **N:** -āk/-ā (38) and -iyo/-iyā (12) are menu or lexical. -no/-nu (14) comes from the anaka
  vs anakam tie. -inu (13) is the missing menu item above. Glide y before ā was tested with
  gates: 34 better, 58 worse; rejected.
- **Ku:** -ik/-i (44, the ikka variant against short -i). Making final -ī short everywhere
  was tested: ext +0.0072; rejected, as Ku sources write both -ī and -i. The rest are
  singletons: kʰ/cʰ < kṣ, niṣk- > nik-, and nn > n rather than ṇ (ucʰino, 3).
- **Garh:** no group larger than 3 (nasalised -yũ, k > g after a nasal vowel).
- **eur:** groups of 2–4 only: -aya- > -av- (see the second pass), epenthesis in final -Cr
  (muter, astér, 3), final devoicing nj > nc (2), -asal without umlaut (2).

Nothing in the near-misses reaches the ≥ 5-pair threshold with a clean condition.

**(3) Verb misses: which menu form wins, and the attested citation.**
- **N** (1002 train verb pairs): winners anakam 608, āpanakam 199, anaka 49, anīya 38. The misses
  are almost all -nu (562). They are segmental errors in the stem (prefix contractions such as
  abʰi- > oi-, ut- > u-), or passive -inu, which needs the menu item.
- **Ku** (675): anaka 527, āpanaka 52. The misses end in -ṇo (401); the causes are the stem or
  -āuṇo vs -āṇo variation.
- **Garh** (140): anaka 119. 61 misses in -ṇu and 14 in -nu, from the n-dissimilation variation.
- **eur** (180): ati 65, root 46, ami 33. Misses end in -el (49), which is segmental, and -av
  (15), the causative the menu lacks.
- **Domari** (55): ati 26, root 18. -ʌr (26), where the misses are stem errors.
- **RomWel, RomBalk, Lom:** root wins. The misses are the -av causatives (8, 5, 5).
- **Zarg:** root wins, but 10 misses are the inflected -ipey forms, which the source cites and no
  menu item covers.

No new rules were installed in this pass. The final table is as in the first pass.

### Round 6: per-lect verb menus

The coordinator's placebo run (real item beats a same-shape placebo by > 0.002 on train and
test) gave N +iyanakam +aka, Ku +iyanakam +aka, Garh +aka. Here -aka is root + -aka as a verbal
noun, tested against the placebo -ara. My own placebo numbers for -iyanakam against
-iranakam are in the third-pass table.

The `%verb-menu` lines were added near the top of the three files and gated against the files
without them (plain and accent unchanged):

| lect | ext train | ext test | verb train | verb test |
|---|---|---|---|---|
| N (+iyanakam +aka) | .2629→.2599 | .2483→.2456 | .1925→.1802 | .1758→.1647 |
| Ku (+iyanakam +aka) | .3005→.2977 | .2934→.2913 | .2350→.2230 | .2282→.2181 |
| Garh (+aka) | .2786→.2778 | .2922→.2898 | .1735→.1702 | .1380→.1289 |

The N rule `i a > i / _ n u #` (-iyaṇaaṁ > -ianu > -inu) now passes the gates: ext
.2599→.2578 / .2456→.2428, verb .1802→.1720 / .1647→.1547, plain unchanged. It makes 142
pairs better and 52 worse; the losers are -anu verbs where the passive variant was winning by
accident (kahanu, cāhanu). **Installed.**

The Ku analogue `y a > i / C _ ṇ o #` makes 19 better and 69 worse, with verb +0.0036.
**Rejected**: Ku keeps the glide (koṭyūṇo, ugalyaṇo); its passive is -yaṇo/-yūṇo, not -iṇo.

Final extension-aware NED, train/test, against the start of round 6:

| lect | ext NED |
|---|---|
| N | .2693→.2578 / .2543→.2428 |
| Ku | .3269→.2977 / .3258→.2913 |
| Garh | .3018→.2778 / .3229→.2898 |

## Round 7

Scratch: `tmp/sound_changes/central-east-pahari-romani/r7b/` (`cand.py` builds a candidate from the
installed file, `g.sh` runs `gates.py`, `wdiff.py` lists the reflexes a candidate makes better or worse,
`show.py` lists reflexes containing given letters, `morph_loo.py` does the @morph leave-one-out). All
numbers below are under the scoring of the day (vowel qualities distinct, all accent/tone marks dropped
from plain/ext/verb) and the engine fixes up to 11:40 (the final table was re-run after the last one). Every kept change
passed `gates.py` alone against the file of the moment, and every lect passes the final snapshot → file run.

### Results (pre-round-7 snapshot → final, same engine; train / test)

`final_gates.txt`. Plain NED rises in most lects because of the @morph removals, which only served the
bare headword (the bare-vs-extended case the brief allows; see the table below). The accent columns are
ext_acc; ext_tone is only scored in eur (a CDIAL grave), where it moves with ext_acc.

| lect | ext NED | verb-menu NED | ext_acc | plain NED |
|---|---|---|---|---|
| N | .2575→.2477 / .2421→.2306 | .1764→.1751 / .1682→.1655 | ±0 | .3889→.4235 / .3716→.4051 |
| Ku | .2993→.2817 / .2912→.2674 | .2328→.2196 / .2362→.2213 | (not scored) | .4049→.4539 / .3895→.4374 |
| Garh | .2943→.2560 / .3067→.2734 | .1861→.1721 / .1923→.1812 | ±0 | .3673→.4164 / .3791→.4413 |
| Dotyali | .3348→.3019 / .3169→.2937 | .5042→.3756 / .6292→.5625 | – | .4203→.4215 / .4028→.4028 |
| eur | .3085→.2973 / .3406→.3223 | .3031→.2849 / .3854→.3575 | .3339→.3199 / .4792→.4375 | .3930→.4196 / .4236→.4380 |
| Domari | .4297→.3688 / .4579→.3949 | .4442→.3620 / .5023→.3909 | .4494→.3778 / .4731→.4036 | .4960→.4949 / .5110→.5067 |
| RomWel | .4223→.4135 / .4345→.4303 | .4180→.3901 / .3781→.3689 | ±0 | .4656→.4938 / .5342→.5523 |
| RomBalk | .3070→.3041 / .4007→.3955 | .3229→.3125 / .4075→.3798 | .2624→.2602 / ±0 | .3905→.4374 / .5053→.5190 |
| Zarg | .4304→.4300 / ±0 | ±0 | .6107→.5964 / .3750→.2500 | .4978→.5136 / .4689→.4651 |
| Lom | ±0 | ±0 | ±0 | .3856→.4027 / .4106→.4086 |

Lom changed only by neutral @morph removals.

### Step 0: the interrupted round-7 edits, re-gated

The previous agent had installed thirteen changes across eight rule files before it was stopped. Each was re-gated under the new
scoring, alone (leave-one-out against the current file) and together (pre-round-7 snapshot → file).

| lect | change | verdict | train ext Δ (LOO) |
|---|---|---|---|
| N | ut- assimilates before a sibilant (`t > ∅ / # u _ {ś ṣ s}`) | kept | −0.0002 |
| Ku | same ut- rule | kept | −0.0003 |
| Ku | second-syllable syncope as in N | kept | −0.0009 |
| Ku | ya-śruti glide before final -o, as in N | kept | −0.0020 |
| Garh | secondary au > ɔ, ai > ɛ, -āv(a)ṇu > -ɔṇu (were o, e) | kept | −0.0097 (ɔ), −0.0047 (ɛ) |
| eur | shared a > e in an open first syllable (`vowel-shift.aa…`) | kept | −0.0022 |
| eur | -aya- > -ava- plus a syncope that spares the vowel before v (joint) | kept | −0.0029 |
| eur | oxytone stress on monosyllables too | kept (prosody only) | ext_tone −0.0080, ext_acc −0.0080 |
| Domari | short a outside the first syllable > ʌ | kept | ext −0.032 (whole file) |
| RomBalk | `%verb-menu +āpaya` (bare causative -av) | kept | verb −0.011; placebo re-run below |
| Zarg | oxytone stress placement added | kept (prosody only) | ext_acc −0.0143 |
| RomBalk | oxytone stress on monosyllables | **reverted** | ext_acc ±0 (only bare `acc` improved) |
| RomWel | oxytone stress placement added | **reverted** | every score ±0 |

The two reverted stress rules change nothing that the gates read, so `gates.py` rejects them.

### Vowel qualities (the new scoring keeps ə ʌ ɛ ɔ ɪ ʊ distinct)

Letters in the golds, by source: Ku CDIAL ɔ 47, ε 21, ʌ 19, ə 12; Garh CDIAL ɔ 64, ε 32; Domari CDIAL ʌ 143
(herin2012 ə 28, but that source is excluded); Lom CDIAL ə 24; RomWel CDIAL ɔ 8; Dotyali æ 6; N, eur,
RomBalk and Zarg have none to speak of.

- **Garh** (regular): besides the round-7 au > ɔ, ai > ɛ, two more changes give ɔ:
  - MIA ava > ɔ, not o (`use aya-e.ava` replaced by a local `a v a > ɔ`): pavana > pɔṇ, návanīta > nɔṇi,
    dʰavalī > dʰɔḷi, navarātra > nɔrāttu. 6 better, 2 worse (bʰoṇu, loṇ).
  - final -am (after apocope) > -ɔ̃: aṣṭamá > āṭʰɔ̃, daśamá > dasɔ̃, pañcamá > pā̃cɔ̃, saptamá > sātɔ̃. Only the
    -ama ordinals end so in the data. The rule has no numeral condition; {a ā} m was worse (gʰarmá > gʰām).
  - Jointly: ext .2791→.2764 / .2946→.2884, plain −.0044 / −.0048, verb +.0004 / −.0049 (within tolerance).
- **Ku** (not addressable): the ɔ ʌ ε ə forms are one CDIAL sub-source inside the Ku. data (the same rows write
  ʦ ʣ, ś and ṛ for n: ʦɔkũṛ, ʣɔ̄̃yā, kʌũṛə, bɛ̄̃ṛ). Every row has Source = CDIAL, so a `%source` @notation block
  cannot single them out, and a chain rule would break the majority. Left alone.
- **Domari**: short a > ʌ in the first syllable too (xʌr, sʌp, gʌm beside kam, kal, asti, angul): 29 better,
  50 worse, ext +.014. Rejected; the first-syllable vowel varies.
- **Lom**: first-syllable i > ə (kʰəl, cʰən, tʰəl, nəkl): train ext −.0038, but test ext +.0040 (vetoed); with u
  as well, ext +.027. Source variation.
- **RomWel** ɔ̄ for ā (sɔ̄r, pɔ̄lē) and **Dotyali** æ (śæp, rujæko) are a handful of lexical cases.

### Romani -av causative (per-lect `%verb-menu +āpaya`, placebo -āraya)

Verb-menu NED train/test (real item must beat the placebo by > .002 on train and be better on test):

| lect | base | +āpaya | +āraya (placebo) | verdict |
|---|---|---|---|---|
| RomWel | .4057/.3431 | .3945/.3315 | .4046/.3319 | **added** (carav, xulav, burav) |
| RomBalk | .3244/.4390 | .3131/.4057 | .3195/.4390 | **kept** (acʰav, ladav, pusav) |
| Lom | .3104/.4929 | .3085/.4796 | .3087/.4929 | no: train margin .0002 |
| Zarg | .4949/.3725 | .4926/.3725 | .4949/.3725 | no: test tie |
| Domari | .3854/.4000 | .3854/.4000 | .3849/.4000 | no effect |
| eur (+āpayati, 3sg -avel) | .2821/.3439 | .2791/.3358 | .2784/.3309 (ārayati) | no: placebo wins |

(Base values are from before the 10:35 verb-flag fix; the verdicts are relative.)

The placebo was re-run after the 10:35 verb-flag fix: RomWel .4009 → .3901 (āpaya) vs .3998 (āraya) on
train, .3594 → .3488 vs .3491 on test; RomBalk .3226 → .3119 vs .3180, .4075 → .3798 vs .4075. Both still
pass. Lom (.3090 vs .3092) and Zarg (test tie) still fail.

### Legacy @morph rules (focus item 6)

`morph_loo.py`: drop each @morph rule, then all droppable ones together. A rule is dropped when ext,
verb, ext_acc and ext_tone are unchanged or better on train and test; the plain loss is the
bare-vs-extended artefact (these rules only rewrote the bare headword's ending). Run three times: before and after
the 10:35 verb-flag fix, which freed some -ayati rules, and after the 11:40 fix (consonant-final roots
glossed in the 3rd person are verbs), which freed the Domari C-final-root rule.

| lect | removed | kept (the gate that fails without it) | plain Δ train/test of the removal |
|---|---|---|---|
| N | C-final roots +nu; -āpayati > -āunu; -ati, other -Vti > -nu | -ayati > -nu (ext +.0001) | +.036/+.038 (ext −.0003/−.0015) |
| Ku | C-final roots +ṇo; -āpayati > -āuṇo; other verbs > -ṇo; then -ayati > -ṇo | – | +.036/+.038, then +.018/+.017 |
| Garh | -āpayati > -āuṇu; other verbs > -ṇu; then -ayati > -ṇu | C-final roots +ṇu (ext +.0004) | +.059/+.061, then +.018/+.017 |
| Dotyali | -ayati > -yo; C-final roots +yo (after `+itaka`, below) | -ati > -yo (ext +.0004) | +.0015/0, then +.0055/0 |
| eur | C-final roots +el; -ati, other -Vti > -el; final `el > ∅` (jointly, ext −.0020) | -ayati > -el (ext +.0002); -uka, -ikā, -aka (ext +.001 to +.04, ext_acc) | +.029/+.020 |
| Domari | -ati, other -Vti > -ʌr; then -ayati > -ʌr; then C-final roots +ʌr | – | +.011/+.011, then +.006/+.004, then +.001/−.002 |
| Lom | -āpayati > -av; -ayati, other -Vti > ∅ | -ati > ∅ (ext +.0008); -ikā > -ī (ext +.009) | +.018/−.002 |
| RomWel | -ayati, -ati, other -Vti > ∅ | -ikā > -ī (ext, ext_acc); -aka > -o (ext +.015) | +.033/+.018 |
| RomBalk | -ayati, other -Vti > ∅; then -ati > ∅ | -ikā > -ī, -aka > -o (ext +.02/+.05, ext_acc) | +.024/+.008, then +.023/+.005 |
| Zarg | -ayati, -ati, other -Vti > ∅ (ext −.0003) | -uka, -ikā, -aka (ext +.002 to +.015) | +.016/−.004 |

The kept noun rules (-aka > -o, -ikā > -ī, -uka > -ū) are what the extension menu needs in Romani, where
cluster simplification would otherwise create new -aka sequences. They are chronology, not citation
morphology.

### New changes kept (each gated alone against the file of the moment; train/test ext Δ)

- **N**
  - `%ext-menu +āka` (placebo -āra): the -ā nouns (karā, kaṛā, ukerā) beside -o < -aka. Ext .2571 →
    .2505 vs .2558 (placebo) / .2421 → .2364 vs .2416; 195 reflexes better, 0 worse. The 47 near-misses
    -āk/-ā (the -akka variant standing in for -ā) were this.
  - Final -Cy loses y after apocope (gamya > gam, maulya > mol, *calyā > cāl): −.0005/−.0005.
  - āma > āũ (āmá > āũ, vā́ma > bāũ, śyāmá > sāũ; geminate kárman > kām untouched) and the in-stem
    -in > -i (gōtrin > goti, dʰanín > dʰani): jointly −.0008/−.0015, 23 better, 2 worse.
  - Medial -ia- > i after consonant loss (*prasvidana > pasinā, *bʰiyajyatē > bʰijnu): −.0003/−.0001.
    (ia > e instead: 9 better, 16 worse.)
  - va-śruti before final -ā (yugá > juvā, gaḍḍuka > karuvā): −.0001/−.0003; extended to long ū ō
    (pūpa > puvā, kū́pa > kuvā, madʰū́ka > mauvā): −.0004/±0.
  - Final -n of an-stems lost (kárman > kām, nā́man > nāũ, rṓman > rãũ): −.0007/−.0014; in-stems keep
    it for -in > -i (V _ # instead: 15 better, 13 worse).
  - Final ṛʰ > ṛ (*ḍʰaḍḍʰa > ḍʰāṛ): −.0001/±0.
- **Ku**
  - `%ext-menu +āka` (placebo -āra): ext .2958 → .2913 vs .2951 / .2873 → .2802 vs .2873; 99 better, 0 worse.
  - Geminate nasals stay dental: `{ṇ n} {ṇ n} > n n`, and n > ṇ moved before degemination so it only
    hits single n (kárṇa > kān, ū́rṇā > ūn, cʰanna > cʰāno). Jointly with +āka: −.0068/−.0092.
  - Final -Cy loss as in N (kalya > kāl, kulyā́ > kūl): −.0004/−.0002.
  - First-syllable compensatory lengthening before a geminate when two syllables follow, the N rule
    verbatim (*aṭṭanaka > āṭṇo, aŋgana > āŋaṇ): −.0031/−.0046, 83 better, 55 worse. (Lengthening in every
    first syllable: 144/134, verb train +.0019; rejected.)
  - ā shortened before final hiatus -u/-i (gʰāta > gʰau, taḍāga > talau, *upalāya > ulau): −.0012/−.0017.
    Not in N (26 worse) or Garh.
  - va-śruti before -ā as in N (śúka > suvā, *nirmukʰa > nimuvā), with long ū ō (pūpa > puvā, lōhá >
    luvā): −.0003/−.0002, then −.0003/±0.
  - Final -n of an-/in-stems lost (kárman > kām, ātmán > āp, gōsvāmin > guśɛ̃): −.0015/−.0008, 29 better, 3 worse.
  - -ama/-āma > -ãũ as in N (ásama > asãũ): −.0001/−.0004 (one train word).
  - Final ṛʰ > ṛ (kōṭʰa > koṛ): −.0001/−.0016.
- **Garh** (besides the vowel-quality rules above)
  - l > ḷ moved before degemination and limited to single l, so old ll stays l (bʰalla > bʰali, *talla >
    tāl, vallī > bali vs. jválati > jaḷɔṇu), and final -Cy loss (tōlya > tol, híraṇya > hiraṇ): jointly
    −.0054/−.0011, 20 better, 3 worse.
  - Post-nasal voicing ŋk > ŋg (páŋka > pā̃gu, vaŋkú > bā̃gu, kaŋkaṇa > kaŋgaṇ): −.0020/±0, 6 better, 0
    worse. In N and Ku the same etyma are attested both ways (ṭā̃g ~ ṭā̃k): not installed there.
  - Geminate nasals stay dental (as Ku): −.0030/−.0032.
  - First-syllable lengthening before a geminate (as N, Ku): −.0031/−.0023 (kurkurá > kūkur, ṭʰakkura >
    ṭʰākur).
  - `%ext-menu +āka` (ext .2637 → .2600 vs .2635 / .2819 → .2791 vs .2819) and -ia- > i (bʰr̩ta > bʰiṛu):
    jointly −.0048/−.0042.
  - Final -n of an-/in-stems lost (kárman > kām, cárman > cām, gōsvāmin > gusɛ̃): −.0029/−.0021.
- **Dotyali**
  - ut- assimilation as in N/Ku (*ut-stʰita > utʰyo): −.0009/±0.
  - `%ext-menu +āka` (ext .3335 → .3191 vs .3335 / .3169 → .2990 vs .3154) and -ia- > i: jointly
    −.0163/−.0192, 33 better, 0 worse (dā̃dā, tārā, musā, kukurā).
  - `%verb-menu +itaka` (past participle + -aka: the -yo past the source cites; placebo -iraka): verb
    .4214 → .3620 vs .4214 / .5500 → .5000 vs .5500. Then i before final -yo is lost after a consonant
    (māryo, maryo, ānyo, sutyo): ext −.0087/−.0025, verb −.078/−.083. Dotyali verbs are only scored since
    the 10:35 verb-flag fix.
  - Final -n lost as in Ku and Garh (nā́man > nām ~ nāũ): −.0023/±0.
- **eur**
  - Voiceless stops deaspirated after a sibilant (aŋguṣṭʰá > angušt, ṓṣṭʰa > vušt, svastʰa > sasto):
    −.0024/−.0027, ext_acc −.006. The same rule in RomWel: −.0057/±0; in RomBalk, Lom, Zarg, Domari it
    changes nothing.
  - a > e in the first syllable no longer before s or h (hásati > asal, prahasati > prasal, bahís > avri):
    local copy of the shared rule, −.0016/±0. In RomWel, RomBalk and Zarg the local copy is identical in
    effect to the shared rule, so they keep `use`.
- **Domari**: bare final -a after a consonant is lost, after intervocalic k is lost and before -aa
  contracts, so -aka derivatives keep -a > -ʌ: kárman > kam, kaṇṭa > xand, kṣīrá > kir vs. taptá-ka >
  tátʌ, kāla-ka > xálʌ. Ext −.0288/−.0247, verb −.0046/±0, ext_acc −.0015/−.0133; 71 better, 2 worse.
  Plain +.0094: the bare-vs-extended case (the bare headword now loses -a, and the -a forms are -aka
  derivatives the extension score finds). The same loss placed late (it also hits -aka) improves ext but
  worsens ext_acc +.04 to +.06.
- **RomWel**: `%verb-menu +āpaya` (above) and the sibilant deaspiration (vaŋuśt, ast).

### Rejected this round (besides those above)

- Ku: final ī/ū kept long in monosyllables (kūt, kīl, gʰūs): ext −.0052/−.0035 but plain +.0036/+.0067,
  70 better, 31 worse; the same etyma are cited both ways (ūn ~ un, ūŋ ~ uŋ). Source variation.
- Ku -iī > -i; Ku ia > i (54 worse); Ku and N ḍ > l after a long vowel (culo ~ cuṛo, dālim ~ dāṛim: doublets).
- N e a > e; N final ā-shortening; Dotyali second-syllable syncope and general syncope (1–6 spurious
  improvements through -araka); Dotyali -Cy loss (train ±0).
- eur -Cr epenthesis (muter, aster: 6 better, 5 worse); eur final j > c (net −1); eur and Domari
  `%ext-menu +āka` (placebo as good).
- Domari: final -i loss (it also removes the -ari > -ʌr 3sg: 38 worse); length kept in the stressed
  penult (47 worse).
- RomBalk: ř written i̦ from ṭ ḍ ṭṭ ṇḍ (kai̦o, pei̦): 19 better, 19 worse, because the same etyma are
  also cited with r (por ~ poi̦, pinro ~ pii̦o). Two notations inside one CDIAL source key.
- Accent: nothing new passes. eur has 28 accent-marked train forms (15 final, 8 penult, several of them
  initial-stress Balkan forms such as tálo, bóko); Domari's penult rule already covers its 152.
- Rules that fire rarely: `fire.py` counts the train forms (bare and menu) each rule changes. In N every
  rule but the new āma > āũ (7) changes ≥ 15; in the other lects every segmental rule changes ≥ 5.
- Consistency tried and not installed: Ku -ia- > i (32 net worse), Garh second-syllable syncope (verb
  +.009), Garh/N -aṇu syncope (no effect), ā-shortening in hiatus in N/Garh, Garh ṛʰ > ṛ (train ±0),
  Garh -nya > -n (1 better, 1 worse), Ku/N -ika menu item (no effect), y-glide before -ā (test worse).

### Not done / left alone

- The Ku lax-vowel sub-source (ɔ ʌ ε ə, ʦ ʣ) and the RomBalk i̦ (ř) forms: both need a way to address one
  notation inside the CDIAL source key; `%source` cannot do it.
- N and Ku: ordinals and decades (-ārĩs/-ālis, -sata/-hattar), -iyo/-iyā, -l/-r and ḍ > r/l doublets,
  compounds: lexical. N -ai > -i after -ikā fails only on ext_acc (three accent-marked N forms).
- eur verbs: CDIAL cites bare stems, but final-cons-loss (needed for nāman > nav) also strips the verb-menu
  root (ān- > a); fixing it would need a vform condition, which is not allowed.
- Lom -av nouns (gilav, pelav), Zarg -ipey inflected verbs, RomWel long -ō: lexical or one-source.
- Accent/tone: no new placement rule passes; Domari and eur were audited (above).
