# Nuristani — OIA ↔ Nuristani correspondence chains

Group `nuristani`: Wg (Waigali / "Nuristani Kalasha"), Kam (Kamviri), Kt (Katë / Kamkata-vari),
Ash (Ashkun), PNur (Strand's Proto-Nuristani reconstructions), Pr (Prasun), Gmb (Tregami).
Rule files: `sound_changes/rules/OIA-{PNur,Kt,Kam,Wg,Ash,Pr,Gmb}.sc`. Scratch tools and logs:
`tmp/sound_changes/nuristani/` (`corr.py` segment-correspondence tables, `try.py`/`t.sh` rule
trials, `chron.py` order swaps, `oracle.py` neutralisation oracle, `loans.py` stratum
diagnostics, `bysrc.py` per-source scores; `chronology.txt`, `oracle.txt`, `loans.txt` outputs).

**Framing.** Nuristani is not descended from OIA. It is a third branch of Indo-Iranian. The
files are *correspondence* chains read from the OIA side. Shared "changes" such as intervocalic
loss are tagged with the ordinary CATALOG tags when the correspondence looks like the same
process. `@nur-archaism` marks places where OIA has merged a PIIr contrast that Nuristani keeps.
`@nur-innovation` marks Nuristani-internal developments. The PNur chain is the group base, and
every modern file is a copy of it with lect-specific deletions and a lect block at the end.

## 1. Summary table

NED on the train/test split (null = no rules → final). Nulls include the per-source notation
tails, which is why they differ slightly from the auto baseline's zero point. `nured` pairs are
excluded in every file (§5.4). "rules" counts chain rules; `%source` tail rules are listed
separately (a rule shared by two sources counts twice). The auto baseline is scored with nured
included and without tails, so its null is 0.002–0.013 higher. At the auto baseline's largest
rule count, the hand chain beats it by 0.02 (Pr) to 0.10 (Gmb) test NED.

| lect | etyma | rules | train NED null → final | test NED null → final | test exact | auto-baseline test NED (hand at same #rules) | excluded |
|---|---|---|---|---|---|---|---|
| PNur | 754 | 35 (+0 source-block) | 0.6249 → 0.4249 | 0.6111 → 0.4397 | 9.9% | 0.5034 @18 (auto max) ; hand @18: 0.4610 | nured |
| Kt | 1.00k | 40 (+3 source-block) | 0.7055 → 0.5230 | 0.7084 → 0.5416 | 5.4% | 0.6295 @19 (auto max) ; hand @19: 0.5836 | nured |
| Kam | 1.02k | 39 (+1 source-block) | 0.6806 → 0.5010 | 0.6805 → 0.5211 | 5.2% | 0.6108 @13 (auto max) ; hand @13: 0.5768 | nured |
| Wg | 1.06k | 34 (+6 source-block) | 0.6946 → 0.5454 | 0.6818 → 0.5659 | 2.9% | 0.6405 @13 (auto max) ; hand @13: 0.5837 | nured |
| Ash | 859 | 36 (+8 source-block) | 0.7050 → 0.5406 | 0.7025 → 0.5257 | 5.5% | 0.6439 @15 (auto max) ; hand @15: 0.5567 | nured |
| Pr | 647 | 44 (+1 source-block) | 0.7771 → 0.6367 | 0.7595 → 0.6400 | 3.9% | 0.7092 @16 (auto max) ; hand @16: 0.6878 | nured |
| Gmb | 123 | 31 (+0 source-block) | 0.7165 → 0.5266 | 0.7012 → 0.4724 | 13.3% | 0.5896 @15 (auto max) ; hand @15: 0.4935 | nured |

## 2. The base chain (PNur) in order

| # | rule | tags | note |
|---|---|---|---|
| 1 | `V > $1-acc` | @acc-loss | |
| 1b | `{n s t d} > ∅ / V _ #` | @final-cons-loss | OIA stem-final consonants absent (nā́man > Wg nām, sáras > *sar); −0.0010 to −0.0039 in every lect |
| 2 | `{ai au} > {e o}` | @ai-au | |
| 3 | `{ē ō} > {e o}` | @notation | OIA e/o are long by definition |
| 4 | `a t {i e} > a / _ #` | @morph | verbs cited as stems (Kt: `> e`, Strand's Kt citation form) |
| 5 | `k ṣ > ʦ̣` | @ks @nur-archaism | mākṣiká > *māʦ̣i |
| 6 | `ś v > ś p` | @nur-archaism | *ćw > śp as in Iranian: áśva > *uśpa, śváśura > *iśpaśur |
| 7 | `ś r > ṣ` | @nur-archaism @cluster-simplify | *śrāya > *ṣai, śŕ̩ŋga > *ṣiŋ |
| 8 | `ś > ʦ / V _ \| V ṁ _` | @nur-archaism @affrication | PIIr *ć: dáśa > *daʦ, āśá > *āʦa, viṁśatí > *viʦi |
| 9 | `c cʰ > ʦ`, `cʰ > ʦ` | @nur-archaism @affrication | OIA (c)cʰ < PIE *sḱ: ā́gaccʰati > *āʦa, cʰāyā́ > Kt ʦāvá |
| 10 | `h > ʣ` | @nur-archaism @affrication | PIIr *ȷ́h: himá > *ʣim, hŕ̩daya > *ʣara |
| 11 | `j > ʣ / _ {a ā}` | @nur-archaism @affrication | PIIr *ȷ́: jā́nu > *ʣā̃; j + front vowel (< PIIr *ǰ) stays j |
| 11b | `s > ś / _ C` | @x-nur-s-sh @nur-innovation | s > š before consonants, placed before cluster simplification (a late position costs 0.0006 to 0.004): stēná > *śtana. Not Ash or Pr |
| 12 | `r {t tʰ d dʰ} > ṭṭ/ḍḍ` | @assim-r | geminate, so it escapes flapping: kartana > *kaṭala |
| 13 | `r {ś ṣ} > ṣ`, `ṇ > ∅ / r _` | @assim-r | párśu > *pāṣu, kárṇa > *kār |
| 15 | `Asp > $1-asp` | @deaspiration | all ten aspirates |
| 15b | `n > ṇ / V _ V` (PNur only) | @x-nur-n-flap | Strand's *ṇ for intervocalic n (*daṇi, *gaṇa), the input of Kt/Kam ř̃. Wg, Ash and Pr reject it (+0.001 to +0.004) |
| 16 | `{ṭ ḍ} > ṛ / V _ V` | @int-retroflex @flap | maṭʰa > *maṛa |
| 17 | `p > v / V _ V` | @int-p-v | nāsāpuṭa > *nāsāvuḍa |
| 18 | `{k g t d y v b} > ∅ / V _ V` | @int-stop-loss @v-loss | ā́gata > *ā |
| 18b | `ṁ > ∅` | @anusvara-vowel | *after* lenition: like nt, ṁ protected the next stop (saṁbʰāvayati > Kam sāmā) |
| 19 | `ñ > ∅` | @nasal-cluster @cluster-simplify | páñca > *pãc |
| 20 | `n > ∅ / V _ {t ṭ}` | @nasal-cluster @nur-innovation | antár > *ātar, jīvantá > *jĩta |
| 21 | `ā > ∅ / C _ #` (PNur, Kam, Wg); `{a ā} > ∅ / C _ #` (Ash, Pr, Gmb); Kt: -a only after single C | @final-vowel-loss | final -ā is lost already in PNur (drā́kṣā > *drās, vḗlā > *vel); final -a only in the daughters other than Kam, see §4 |
| 22 | `C $1 > $1` | @degem-plain | |
| 23 | `C {y v} > $1` | @cluster-simplify | pakvá > *pak |
| 24 | `{k p} {t ṭ} > $2` | @cluster-simplify | śuktá > *suta |
| 25 | `{ī ū} > {i u}` | @nur-innovation @vowel-shift | Nuristani has no high-vowel length |
| 26 | `{a ā i} > ∅ / V _` (Ash, Gmb: `{a ā}`; Wg: `{a ā i u}`) | @hiatus | then, in Kt/Kam/Pr/Gmb, `a > ∅ / _ V` (a + i/u → i/u: kakubʰá > Kt kuv) |
| 27 | `r̩ > i` | @r-vocalic | kr̩ṣí > Kt kiṣ, ŕ̩kṣa > Kt iʦ |
| 28 | `C V r > $1 r $2 / # _ C` | @metathesis @nur-innovation | kūrma > Kt kr̆um, dīrgʰá > Kt dragar̆ |
| 29 | `a > ā / # _ C` | @nur-innovation @vowel-shift | PNur *ā- for OIA a-; Kam generalises it (see below) |
| 30 | `y > i / # _` | @notation | Strand writes *i- for y- (PNur and Kam only) |

Where each lect departs from the base (lect blocks are appended after rule 30):

- **Kt**: verbs `-ati > -e` (@morph). The j rule stays conditioned. Rule 21 drops -a only after
  a single consonant (`a > ∅ / V C _ #`). After a cluster it is kept as ə, which is better than
  dropping it everywhere by 0.0014 (gōṣṭʰá > guṣṭá).
  Drops `y > i` and `ś v > ś p` (loo 0). `{n ṇ} > r̆ / V _ V` (@x-nur-n-flap) goes in the
  lenition block, right after flapping: it must precede final-vowel loss (+0.0101 in Kt if
  reversed). Lect block: `ʣ > z`,
  `ṛ > r̆`, `r > r̆ / # _`, `o > u`, `{d ḍ b g} > ∅ / Nasal _`.
- **Kam**: keeps final -a (Strand's ə). Drops `ś v > ś p` (loo 0). Drops rule 29 in favour of
  the strand notation block `a > ā / # (C) (C) _` (@x-nur-a-quality, §5). Has the same
  `{n ṇ} > r̆` flap in the lenition block as Kt. Lect block: the Kt block without `ʣ > z`, since
  Kam keeps ʣ, plus `ā > a / V C _ | V C C _` (non-initial ā reduced like a, −0.0023).
- **Wg**: drops the cʰ rules, `a > ā / # _ C` and `y > i`. Final -a is kept in the chain and
  dropped only by the CDIAL/Buddruss tail. Strand writes a final ə that Morgenstierne/Turner and
  Buddruss do not (−0.0027 vs. chain-level loss). Lect block: `ʣ > z`,
  `r > ź / # _`, `o > u`, `ʦ > ś / V _`.
- **Ash**: `ś > ʦ` unconditioned. `j > ʣ` unconditioned. No cʰ rules, no early s > š. Rule 21
  also drops -a. Verbs are cited in Strand's -ie (@morph), and the CDIAL/Buddruss tail strips
  it (−0.0063).
  Lect block:
  `ʣ > z`, `r > ź / # _`, `{d ḍ b g} > ∅ / Nasal _`, then the chain shift `ʦ > s` ≺ `c > ʦ`.
  Also, early: `ṣ > ś / {i ī u ū} _` (kaniṣṭʰá > kaṇiśt, muṣṭí > muśt; −0.0017).
  `r {ś ṣ} > ṣ` must precede the unconditioned `ś > ʦ` (párśu > pāṣú; −0.0005).
- **Pr**: `j > ʣ` unconditioned. `{d dʰ} > l` placed *before* deaspiration and lenition. The
  r-assimilation rules are dropped (loo ≤ 0), because Pr simply loses r in clusters.
  No metathesis. Rule 21 also drops -a. Lect block: `ʣ > z`, `ʦ > z / V _`, `r > ź / # _`, `o > u`, `a > u`,
  `t r > c`, `r > ∅ / C _ | V _ C`, `ā > a / C _`, `ṛ > r`, `ʦ̣ > ṭ`, `e > i`. Also, early:
  `ṣ ṭ > ś t`, placed after `ś > ʦ` (muṣṭí > müśt; −0.0022 in total).
- **Gmb** (123 etyma): drops `ś v`, the cʰ rules, j, the r-assimilation, flapping and rule 29,
  because they do not fire or have loo ≤ 0 on 120 pairs. Lect block: `r > ʣ̣ / # _`,
  `o > u`, `{d ḍ b g} > ∅ / Nasal _`, `ʦ > ś / V _`.

## 3. Relative chronology (order swaps; train ΔNED when the pair is reversed)

Full log: `tmp/sound_changes/nuristani/chronology.txt` (final rule files). Every entry below
*hurts* when reversed, in every lect with enough data. The one exception is Wg p > v, where
intervocalic p is simply lost.

| earlier ≺ later | PNur | Kt | Kam | Wg | Ash | Pr | Gmb |
|---|---|---|---|---|---|---|---|
| `s > ś / _ C` ≺ kt/pt > t (its order vs lenition is neutral) | +.0007 | | +.0006 | | — | — | |
| deaspiration ≺ intervocalic loss | +.0050 | +.0032 | +.0048 | +.0018 | +.0034 | +.0041 | +.0026 |
| p > v ≺ intervocalic loss | +.0017 | +.0009 | +.0018 | −.0017 | +.0004 | +.0026 | 0 |
| intervocalic loss ≺ degemination | +.0056 | +.0055 | +.0053 | +.0092 | +.0060 | +.0035 | 0 |
| intervocalic loss ≺ C+y/v simplification | +.0051 | +.0008 | +.0017 | +.0039 | +.0025 | +.0002 | 0 |
| intervocalic loss ≺ kt/pt > t | +.0032 | +.0027 | +.0024 | +.0036 | +.0037 | +.0030 | +.0066 |
| intervocalic loss ≺ nt > t | +.0078 | +.0039 | +.0048 | +.0067 | +.0034 | +.0066 | +.0061 |
| intervocalic loss ≺ ṁ > ∅ | +.0008 | +.0003 | +.0005 | +.0003 | 0 | 0 | 0 |
| intervocalic loss ≺ final -ā loss | +.0017 | +.0013 | +.0008 | +.0017 | — | — | — |
| intervocalic loss ≺ final -a loss | — | +.0074 | — | — (tail) | +.0104 | +.0068 | +.0167 |
| Kt/Kam `{n ṇ} > ř̃` ≺ final-vowel loss | | +.0101 (-a), +.0019 (-ā) | +.0015 (-ā) | | | | |
| `ś r > ṣ` ≺ `ś > ʦ` | +.0010 | +.0004 | +.0006 | +.0005 | +.0026 | +.0008 | 0 |
| Ash `ś > ʦ` ≺ `ʦ > s`; `ʦ > s` ≺ `c > ʦ` | | | | | +.0120; +.0093 | | |
| Pr `{d dʰ} > l` ≺ intervocalic loss | | | | | | +.0076 | |
| Pr `ś > ʦ` ≺ `ʦ > z / V _` | | | | | | +.0029 | |
| Pr `t r > c` ≺ `r > ∅ / C _` | | | | | | +.0008 | |

Also: r-assimilation must produce a *geminate* retroflex (`rt > ṭṭ`), so that it escapes the
later flapping of single intervocalic ṭ. The single-segment version costs 0.0009 to 0.0028 in
all six lects that have the rule. The nasal-cluster rule `nt > t` placed *before* lenition
costs 0.0063 to 0.0070 (PNur), so the nasal protected the stop through the lenition period.
Anusvāra behaves the same way, though the effect is weaker.

What this implies. The Nuristani correspondence set behaves like a single ordered history:
(1) the palatal and cluster archaisms and the r + dental assimilation, then (2) deaspiration,
then (3) loss of single intervocalic stops and glides (ṭ ḍ > ṛ, p > v), then (4) cluster
simplification (degemination, C+y/v, kt/pt, nt, ñc, anusvāra), with no compensatory lengthening, and
finally (5) vowel changes and the loss of final -a in the modern lects. This is the
Dardic-like profile in which clusters simplify *after* lenition, so their outputs stay
intervocalic and do not lenite. It is the reverse of the MIA pattern, where assimilation
produces geminates that resist lenition. In Ashkun, *ʦ went to s before *č went to ʦ, a drag
chain. In Prasun, *d/*dʰ became l before intervocalic loss could remove them (mádʰu > mū́lu,
ádʰi > āli).

## 4. Shared vs lect-specific; subgrouping

- **Shared by all** (as correspondences): the palatal archaisms (ś/j/h > ʦ/ʣ; ṣ/ś in clusters),
  full deaspiration, loss of single intervocalic stops, cluster simplification without
  gemination, no high-vowel length, and the hiatus contraction.
- **Kt + Kam**: `ṛ > ř`, initial `r- > ř-`, intervocalic `n, ṇ > ř̃` (before final-vowel
  loss), `o > u` and
  `nd/ṇḍ/mb/ŋg > n/ṇ/m/ŋ` form a single block, identical in both files. Kam differs only in
  keeping *ʣ (Kt z), reducing non-initial ā, and its vowel notation (§5). This fits their status as close sisters.
- **Wg + Gmb** (+ partly Ash): initial `r-` fortition to a sibilant or affricate
  (Wg/Ash ź, Gmb ʣ̣). In Wg and Gmb only, post-vocalic *ʦ becomes ś (`ʦ > ś / V _`, @x-nur-ts-sh):
  Wg pacīś, Gmb daś, bāś, sātāś. Kt/Kam have duʦ and -iʦ instead. The *initial-r* change is
  shared by *all* modern lects under different outputs (Kt/Kam ř, Wg/Ash/Pr ź, Gmb ʣ̣), so it
  looks like one Proto-Nuristani fortition of r- with divergent phonetics. It is tagged
  `@x-nur-initial-r` everywhere.
- **Ash** stands apart with the chain shift *ʦ > s, *č > ʦ, and it has unconditioned *ǰ > ʣ > z.
- **Pr** is the most divergent, as expected. It has the only `d > l` (and dʰ), `tr > č`, voicing
  of post-vocalic *ʦ to z, `a > u/ɨ`, loss of r after consonants, and `ʦ̣ > ṭ`. It also
  *lacks* the first-syllable r-metathesis that the other five share. On the base chain it has
  the highest residual (test NED 0.64 vs 0.44–0.56), even though it has the largest lect block.
- **Final vowels**: final -ā is lost everywhere, including in Strand's PNur (−0.0012 to
  −0.0028 in every file). Final -a is lost in Ash, Pr and Gmb, and in Kt after a single consonant. It is kept in PNur and
  in Kam, and in Strand's Waigali (§5.4). The Wg loss is source-dependent: kept in the Strand
  subset, dropped in the Morgenstierne/Buddruss subsets. The same source split does *not* hold
  for Kt, Ash or Pr, where letting any single source keep -a costs 0.004–0.009. For Kam, adding -a loss costs +0.024, even when verbs are protected, because Strand's
  Kamviri keeps a final ə in most nouns. The loss of final -i/-u is not supported anywhere
  (|Δ| ≤ 0.002).
- Suggested tree from the shared lect blocks: {Kt, Kam} vs {Wg, Gmb} vs Ash vs Pr. Wg–Gmb is
  the most secure pair after Kt–Kam. Ash shares ź- with Wg and the nd > n rule with Kt, Kam
  and Gmb, but has its own sibilant shift.

## 5. Findings

### 5.1 How much is *not* predictable from OIA (the PIIr palatal problem)
OIA merged several PIIr contrasts that Nuristani keeps, so from the OIA side some
correspondences can only be guessed:
- OIA **ś** continues *ć (Nur ʦ) but also appears in IA loans and assimilated forms (Nur ś).
- OIA **j** is *ȷ́ (Nur ʣ) or *ǰ from PIE *g/gʷ + front vowel (Nur j).
- OIA **h** is *ȷ́h (Nur ʣ), *ǰh (Nur j; hánati > Kam jār̆̃á), or dissimilated (hásta > *dast).
- OIA **kṣ** is three-way: ʦ̣ (mākṣiká, kṣurá), ʦ (ŕ̩kṣa, tákṣati, dákṣiṇa, kukṣí) and
  ṣ (yakṣá, kṣípyati, sākṣin).

`oracle.py` measures what an oracle would gain if it knew the right branch for each etymon. It
picks, per etymon, the best of the alternative outputs for ś/j/h/kṣ (72 rule-set variants). A
control oracle toggles six linguistically meaningless binary alternatives
(k|c, t|ṭ, m|n, p|b, d|ḍ, g|j; 64 variants), to measure how much an oracle over that many
variants gains by chance. Results (train):

| lect | base NED | oracle gain ś/j/h/kṣ | % of residual | control gain | % | net | % by segment ś / j / h / kṣ | etyma helped / etyma with ś,j,h,kṣ |
|---|---|---|---|---|---|---|---|---|
| PNur | 0.425 | 0.0142 | 3.3% | 0.0044 | 1.0% | **2.3%** | 1.3 / 0.3 / 1.0 / 0.6 | 47 / 155 |
| Kt | 0.523 | 0.0144 | 2.8% | 0.0046 | 0.9% | **1.9%** | 1.1 / 0.2 / 0.7 / 0.7 | 58 / 184 |
| Kam | 0.501 | 0.0133 | 2.7% | 0.0061 | 1.2% | **1.5%** | 1.2 / 0.2 / 0.6 / 0.6 | 58 / 192 |
| Wg | 0.545 | 0.0130 | 2.4% | 0.0050 | 0.9% | **1.5%** | 0.8 / 0.6 / 0.4 / 0.6 | 53 / 196 |
| Ash | 0.541 | 0.0135 | 2.5% | 0.0075 | 1.4% | 1.1% | 0.9 / 0.1 / 0.7 / 0.8 | 52 / 168 |
| Pr | 0.637 | 0.0092 | 1.4% | 0.0070 | 1.1% | 0.3% | 0.5 / 0.2 / 0.5 / 0.3 | 28 / 132 |
| Gmb | 0.527 | 0.0139 | 2.6% | 0.0043 | 0.8% | (1.8%) | — (5 etyma) | 5 / 26 |

(The j figure fell from 0.8% to 0.2–0.3% once the rule was conditioned on the following vowel,
see below.) The key result has three parts:
1. OIA's neutralisations are real and measurable. About a third of the etyma that contain
   ś/j/h/kṣ would be better predicted with the other branch.
2. They account for only **≈1.5–2.3% of the residual error** (≈0.01 NED) beyond what an oracle
   gains by chance. They are not what makes Nuristani hard to predict from OIA. Most of the
   residual comes from derivation and compounding in the cited forms, from unpredictable vowel
   quality (Kt/Kam ā ~ o, u ~ ɨ, Pr a ~ u ~ ɨ ~ i), and from notation mixing.
3. Part of the j neutralisation *is* recoverable from OIA itself. `j > ʣ / _ {a ā}` beats
   `j > ʣ` everywhere except Ash and Pr (PNur −0.0026, Kt −0.0017, Wg −0.0015, Kam −0.0010).
   The reason is that OIA j + front vowel mostly continues PIIr *ǰ from the second
   palatalisation (jīvantá > Kt ját, jiyā́ > já, jyḗṣṭʰa > Kam jésṭ), while ja-/jā- is mostly
   *ȷ́ (jā́nu, jā́mātr̩, janítra, jánti, jámbʰa > ʣ/z). In Ash and Pr, *ǰ itself became ʣ/z, so
   the distinction is moot there. The h split (hánati, hántr̩ > j-; himá, hŕ̩daya, hári > ʣ-)
   has no such OIA-side cue.

The palatal gain is near chance in Ash and Pr, for two reasons. Ash re-neutralised the series
itself: *ʦ > s and *č > ʦ make OIA ś and c partly collide again. Pr is too noisy for the oracle
to separate from the control.

### 5.2 Other archaisms that OIA does *not* neutralise
These are fully predictable from OIA and help:
- **(c)cʰ > ʦ** (PIE *sḱ; Iranian s): ā́gaccʰati > Kt āʦe, Kam ā́ʦa; cʰāyā́ > Kt ʦāvá;
  upaccʰada > Kt vaʦó. Positive loo in PNur/Kt/Kam/Pr. Wg and Ash do not want it: Wg has
  IA-type c in gōccʰā > gūca and cʰagalá > culǽ, and Ash turns c into ʦ anyway.
- **śv > śp**, Iranian-like: áśva > *uśpa, śváśura > *iśpaśur, śvāvídʰ > *śpāi.
- **śr > ṣ, rś > ṣ**: *śrāya > *ṣai, párśu > *pāṣu.

### 5.3 IA loans vs inherited items (`loans.py`, `loans.txt`)
Background: in Jambu's data model, CDIAL's Nuristani items are editorial *groupings*. They
sit in the rank-1 `reflex` slot with `grouping:cdial` and assert neither inheritance nor
borrowing (data/AGENTS.md). So the pairs mix inherited words and loans by construction.
The diagnostic counts reflexes of OIA ś (outside clusters) with ʦ/ʣ/z (Nur-type) against those
with ś (IA-type):

| | PNur | Kt | Kam | Wg | Ash* | Pr | Gmb |
|---|---|---|---|---|---|---|---|
| ś → Nur-type | 48 | 132 | 91 | 62 | 121 | 64 | 2 |
| ś → ś | 49 | 71 | 70 | 92 | 25 | 40 | 14 |

(*Ash counts s as Nur-type because of its *ʦ > s shift, so its Nur-type share is inflated.)
- In **Kt/Kam** 35–43% of ś-items keep ś. They are dominated by culture words, the classic
  IA-loan stratum: dēśá 'country' > deś, pā́śa 'snare' > pōś, vā́śī 'adze' > vóś, *puśśī 'cat'
  > pśāś, kúśala, śāli 'rice', śāṭa 'cloth', maśáka. Initial ś- behaves the same way. In Kam,
  ś- stays ś in śará, śarád, śāli, śā́lā, śū́la, śū́ra and śāṭa, while ʦ- appears in a core set:
  śā́kʰā > ʦóa, śāka > ʦó, śíras > ʦer̆, śákr̩t > ʦiṭ, śūnyá > ʦur̆̃i, śaŋkú > ʦoŋ, śikyà > tiʦa.
  So `ś > ʦ` is kept post-vocalic only. Making it unconditioned, or initial except before a
  liquid, costs 0.001 to 0.004.
- In **Wg/Gmb** the ś-majority is *not* (only) borrowing. The numerals -daśa > -āś/-īś/-ēś
  (Wg pacīś, cadēś; Gmb daś, bāś, sātāś, trũś) are core vocabulary, and Kt/Kam have -iʦ/-uʦ
  in the same items. Wg/Gmb therefore shifted post-vocalic *ʦ > ś. This is modelled as a
  Duke-of-York pair (`ś > ʦ / V _` … `ʦ > ś / V _`). Each rule has loo 0 in isolation because
  its partner undoes it, but the pair records the history and the Wg–Gmb shared innovation.
  Strand's PNur also reconstructs *daś, *bāś, *satāś beside *daʦ, which suggests the
  reconstructions themselves absorb this split.
- **Aspirates** are absent from every inherited Nuristani form. The 18 Wg (CDIAL/Turner)
  reflexes with aspirates are IA loans or IA-contaminated: nacʰun 'dance' (< nr̥tyati via MIA
  nacc-), cʰúp, ʦ̣ʰīr 'milk' (kṣīrá), erecʰún, cʰuvulláh, kucʰok, vʰishtur. Kt and Pr have
  1–3 each. Strand's PNur has none. The 4 PNur-labelled forms with aspirates (*kaṁḍhaga,
  *nāṁdha, *acchi, *maṁdhānigā) all come from the excluded NuRED source.
- **h kept as h**: only Wg havís > hutá. No other lect keeps OIA h in a CDIAL-linked item.
- **Intervocalic stops**: 90–98% of OIA single intervocalic t/k/p are lost in every lect and
  source. The few retained cases are morphological, not loans: early syncope protects the stop,
  as in rucaka > Kt/Kam ẓúk, Ash ẓōk, Gmb ʣ̣ok, and so do agent -tr̥ and compounds. By the
  lenition criterion, the IA-loan stratum in CDIAL's Nuristani citations is small. By the
  ś criterion it is sizeable in Kt/Kam. The two criteria disagree because culture words
  borrowed early from IA with ś also underwent the Nuristani lenition.

### 5.4 Notation and data problems; per-source notation blocks
Every lect except Kam and PNur mixes Strand's transcriptions with Morgenstierne/Turner (CDIAL)
and Buddruss. From 01:50 the engine supports `%source KEY … %end` blocks. These rules run after
the chain only for that source's pairs and do not count as chain rules. Each file's blocks are
at its end. Blocks adopted (train ΔNED when added):

| lect | source | block rules | ΔNED |
|---|---|---|---|
| Kt | CDIAL | `ā > a` (Morgenstierne's Kati does not write Strand's â); `{i u} > {ī ū} / _ #` | −0.0096, −0.0009 |
| Kt | strand | `y > i / # _` (Strand writes y- as i/ǐ) | −0.0018 |
| Kam | strand | `a > ā / # (C) (C) _` (@x-nur-a-quality; moved here from the chain) | −0.031 as a rule |
| Wg | CDIAL | `{i u} > {ī ū} / _ #`, `{e o} > {ē ō}` | −0.0014, −0.0010 |
| Wg | strand | `u > ɨ` (Strand writes Waigali's central ɨ where others write u), `y > i / # _` | −0.0030, −0.0020 |
| Ash | CDIAL, buddruss | `ā > a` | −0.0046 / −0.0036 |
| Ash | CDIAL | `{e o} > {ē ō}`, `u > ū / _ #` | −0.0017, −0.0018 |
| Ash | strand | `a > ā / # (C) (C) _` (as Kam), `y > i / # _` | −0.0025, −0.0016 |
| Pr | CDIAL | `{i u} > {ī ū} / _ #` | −0.0020 |
| Wg | CDIAL, buddruss | `a > ∅ / C _ #` (no final ə; the chain keeps -a) | −0.0027 |
| Ash | CDIAL, buddruss | `i e > ∅ / _ #` (strip Strand's verb citation -ie, set by @morph in the chain) | −0.0063 with the @morph |

- **Strand's â/a**: the house profile maps Strand's â [a] to ā and his a [ə] to a. In Kam, OIA
  *a* of the **first syllable** therefore surfaces as "ā", and later a's as "a". This is the
  biggest single rule in the group (−0.031 in Kam). Strand's Ashkun behaves the same way
  (−0.0025 as a strand block), but his Kati does not: the rule hurts the Kt strand subset
  (+0.009). As a chain rule it hurts Kt, Wg and Ash, which mix in other notations.
- **Kt is two notations**: CDIAL (Morgenstierne) Kt hardly marks ā (25 tokens vs Strand's 351),
  writes ə/ɔ, and uses ẓ where Strand writes ř (rúh > ẓu vs r̆u). A `r̆ > ẓ / # _` CDIAL block
  does not pay (+0.0013).
- **`nured` excluded everywhere** (`%exclude-source nured`). It has its own notation: ċ = ʦ,
  ë = ə (not folded by `gold_tokens`), n̆, and capital archiphonemes Z/M/N/L/K. Notation
  blocks were tried once the engine supported them (`ʦ > ċ`, `a > ë`, `e > ë`, `u > ü`) and
  gain ≤ 0.0003. Even with them, nured's per-source NED stays at 0.62–0.72, against 0.51–0.65
  for the other sources, because its entries are compounds and derivatives (Kt *Mézir-mün*,
  *marë́n̆iŋ-aźi*). So it remains excluded as unusable, not merely differently transcribed.
- **Verb citation forms differ by source**: Strand Kt -e, Strand Kam -a (causatives often -oa),
  Strand Ash -ie, CDIAL mostly bare stems. Kt gets `-ati > -e` (−0.016). Ash gets `-ati > -ie`
  plus a CDIAL/Buddruss tail that strips it (−0.0063). An -oa rule for Kam causatives does
  not pay.
- **Compounds and phrases** dominate the worst misses in every lect (e.g. Kam kránda >
  kr̆āŋ kr̆āŋ lāsa). They set a high floor: even PNur, a clean reconstruction list, bottoms out
  at ≈0.43.
- **PNur reconstructions** keep final vowels and include IA-type ś in culture words (śāṭa >
  *śaṛa, śāli > *śāli, vā́śī > *vāś). They are a lexicon of Nuristani *words*, early loans
  included, not a filtered inherited set.
- CDIAL mislinkings or odd links noticed in passing: jyḗṣṭʰa > Pr vr̩ṣabʰṓ jyā́yān (English
  gloss material in the reflex field), dvā́daśa > Pr dvā́daśa (the OIA form copied as the reflex).

### 5.5 Smaller observations
- r-metathesis (@metathesis) out of closed first syllables (kūrma > kr̆um, *margā > mr̆oŋ,
  dīrgʰá > dragar̆) holds in all lects except Pr, but it is weak (−0.0003 to −0.0019).
- Kamviri u ~ ɨ (kʰura > kɨr, dʰūmá > dɨm, putrá > pɨtr) is frequent (240 ɨ tokens) but not
  predictable from OIA context: every conditioning tried (after velars, monosyllables, closed
  syllables) hurts.
- Kt/Kam OIA ā ~ o (mā́sa > mós, bʰāga > bok, aṭṭāla > āṭól vs kāṇá > kār̆̃á) likewise resists
  conditioning. Neither the OIA accent nor final position nor the following consonant works.
- dva > du and dvā́daśa > *bāś (dv > b, as in Iranian): too few items for a rule.
- `ʦ̣ > c` would help the CDIAL/Buddruss subsets of Wg and Ash (−0.0013 each). Strand writes
  ʦ̣ in the same kṣ-items (mākṣiká > māʦ̣i). It is unclear whether this is Turner's
  transliteration of Morgenstierne's retroflex affricate or a real dialect difference, so no
  rule was added.
- Final -a after m is lost in PNur and Kam (dʰūmá > *dum, grā́ma > *grām; −0.0024/−0.0019),
  though they keep -a elsewhere. This was left out because it has no plausible phonetic
  motivation.

## 6. Tags defined here

| tag | meaning |
|---|---|
| `@nur-archaism`, `@nur-innovation` | as in CATALOG (sister-branch correspondences) |
| `@x-nur-initial-r` | word-initial r- fortition: Kt/Kam ř, Wg/Ash/Pr ź, Gmb ʣ̣ |
| `@x-nur-dz-z` | *ʣ (< PIIr *ȷ́, *ȷ́h) deaffricated to z (Kt, Wg, Ash, Pr; not Kam, Gmb) |
| `@x-nur-n-flap` | intervocalic n, ṇ > ř̃ (Kt, Kam), before final-vowel loss |
| `@x-nur-s-sh` | s > š before consonants (PNur, Kt, Kam, Wg, Gmb; not Ash, Pr) |
| `@x-nur-ts-sh` | post-vocalic *ʦ > ś (Wg, Gmb) |
| `@x-nur-ts-s` | Ashkun *ʦ > s (first step of the chain shift *ʦ > s, *č > ʦ) |
| `@x-nur-pr-d-l` | Prasun *d, *dʰ > l in all positions, before intervocalic lenition |
| `@x-nur-pr-tr-c` | Prasun *tr > č |
| `@x-nur-pr-ts-t` | Prasun *ʦ̣ > ṭ |
| `@x-nur-a-quality` | Kam: OIA a of the first syllable = Strand's â [a] (house ā); later a = ə |

**Engine features wished for**
1. `%source` blocks (added at 01:50) solved most of the notation mixing and are used in Kt, Kam,
   Wg, Ash and Pr. The remaining wish is *part-of-speech (or etymon-class) conditioning* for
   citation morphology. Verbs are cited as -e (Strand Kt), -ie (Strand Ash), -a (Strand Kam),
   -ún (CDIAL Wg) or bare stems, and today that can only be approximated through the OIA
   ending -ati.
2. *Morpheme or compound boundaries* in the gold (or an option to score against the longest
   matching sub-word). Compounds and citation suffixes are the largest single source of error.
3. *Stratum flags* (inherited vs IA loan) on pairs, so that `@nur-archaism` rules can be scored
   on the inherited stratum only.

## 7. Strata: candidate IA loans vs inherited items among the CDIAL-grouped pairs

**Purpose.** Jambu stores CDIAL's Nuristani links as neutral `grouping:cdial` edges, which
assert neither inheritance nor borrowing. This section classifies every linked pair so that a
human curator can later decide between `reflex` and `borrowed` edges. It is *candidate-level
evidence only*; no data were edited.

**Files.** The per-item table is `tmp/sound_changes/nuristani/strata.csv`: all 10,679 pairs of
the seven lects, all splits and all sources, nured included. Columns: Lang, Etymon, Etymon_ID,
Reflex, Reflex_ID, Source, Split, Gloss, ned_nur, ned_ia, ned_ia_best, ia_best_chain, ned_ctrl,
signals, verdict, sanity. Scripts: `strata.py` computes, `strata_report.py 0.15 write` assigns
verdicts and writes the CSV, and `strata_summary.py` produces the tables below.

### 7.1 Method
- **ned_nur**: NED under the lect's own chain, including its `%source` notation tails.
- **ned_ia**: mean NED under five neighbouring Dardic chains (Kal, Pas, Kho, Dm, Gaw), used
  read-only. Each chain is copied with its own `@morph`/`@notation` lines and `%source` blocks
  stripped. The lect's own verb-citation `@morph`, chain-level `@notation` rules and `%source`
  tails are added, so the comparison is not about citation or transcription conventions.
  `ned_ia_best` and `ia_best_chain` give the best single chain.
  I use the mean rather than the minimum because the minimum over five chains acts as an
  oracle: it is as good as the Nuristani chain on average. The difference
  ned_nur − ned_ia is centred on the lect median, because every Dardic chain is on average
  0.04–0.10 worse than the lect's own chain.
- **ned_ctrl**: the lect's own chain with the palatal archaism rules removed (`ś > ʦ`,
  `h > ʣ`, `cʰ > ʦ`). This is the cleanest comparison, because it differs from ned_nur *only*
  in the diagnostic segments. It is only used for etyma that contain ś, h or cʰ, and not for
  śv (because of śv > śp).
- **Signals** (string diagnostics on OIA etymon → reflex):
  - IA-type:
    - `asp`: the reflex has an aspirate. Nuristani has none.
    - `ś-kept`: OIA ś (not in śr/śv/rś/kṣ) with ś and no dental affricate in the reflex.
    - `ś-kept-liq`: the same for initial ś before a liquid (see §7.4).
    - `h-kept`
    - `kṣ-IA`: kʰ/cʰ/kṣ in the reflex.
  - Nur-type: `ś>ʦ`, `h>ʣ`, `j>ʣ` (ja-/jā- only), `cʰ>ʦ`, `kṣ>ʦ`, `śv>śp`.
  - Lect-specific exemptions:
    - Wg/Gmb: post-vocalic ś is their *inherited* outcome (§5.3), so only initial ś counts.
    - Ash: s from ś is ambiguous, because Ash *ʦ > s coincides with a Dardic (Pashai-type)
      s < ś. Ash `ś>ʦ` is therefore not used as Nur evidence.
    - ś…ś/ṣ etyma (`ś-kept-assim`: śúṣka, *śuṣṭa, śuṣi, śmáśru, śváśura) are not used,
      because their OIA ś can be an assimilated *s (Av. huška).
- **Verdicts**:

  | verdict | rule |
  |---|---|
  | `IA-loan` | IA-type diagnostic (signal, or ctrl better by ≥ 0.10), no Nur-type diagnostic, not contradicted by the chain comparison |
  | `inherited` | the mirror image of `IA-loan` |
  | `conflict` | diagnostics disagree, or a diagnostic is contradicted by the chain (|Δ| ≥ 0.15) |
  | `lean-IA` / `lean-inherited` | only the chain comparison, centred Δ ≥ 0.15 or ≤ −0.15 |
  | `undecided` | no evidence either way |

  Etymon-level verdicts (the tables below) are strict if any pair is strict and no pair
  contradicts it.

### 7.2 Sanity check
The sanity set was defined a priori, by OIA etymon. **Core** vocabulary (1,375 pairs):
numerals, pronouns, body parts, kin terms, basic nature, wild and domestic animals of the
region, basic verbs. **Culture** words (452 pairs): crops, cloth, metals, trade, religion and
administration, tools, `*puśśī` 'cat'. At the pair level:

| verdict | core | culture | precision |
|---|---|---|---|
| IA-loan | 21 | 53 | 72% culture (**93%** without the 17 -daśa/viṁśati numeral pairs; base rate 25%) |
| inherited | 278 | 1 | **99.6%** core (base rate 75%) |
| lean-IA (chain only) | 147 | 64 | 30% culture: barely above the base rate |
| lean-inherited (chain only) | 228 | 77 | 75% core: no information |

So the diagnostic verdicts are usable, but the whole-chain comparison against Dardic chains is
nearly uninformative on its own. Its AUC for separating culture from core items is
0.53–0.57, for every IA chain and for their mean and minimum. Its value is as a veto:
contradicting a diagnostic turns the item into `conflict`. The ś/h/cʰ-only control
(AUC 0.56 over *all* items, most of which lack the segments) carries the useful signal.
Evidence types behind strict IA-loans, with sanity hits:

| evidence | pairs | core | culture |
|---|---|---|---|
| ś-kept | 181 | 20 | 31 |
| ś-kept-liq | 63 | 0 | 19 |
| asp | 26 | 1 | 3 |
| ctrl-only | 18 | 0 | 0 |
| kṣ/h | 2 | 0 | 0 |

The 20 core false positives are almost all -daśa numerals and viṁśatí with ś (PNur *daś,
*bāś, *satāś, Ash viśí). Those are exactly the items whose ś is inherited in Wg/Gmb (§5.3).
Strand's PNur includes these Wg-type forms, so they should not be read as loans. In Ash and Pr
they are ambiguous.

### 7.3 Counts and loan rates per lect
| lect | etyma | IA-loan | inherited | mixed | lean-IA | lean-inh | undecided | strict loan rate (95% CI) |
|---|---|---|---|---|---|---|---|---|
| PNur | 754 | 29 | 63 | 11 | 131 | 181 | 339 | 31% (23–42) |
| Kt | 1004 | 36 | 101 | 2 | 111 | 212 | 542 | 29% (22–38) |
| Kam | 1020 | 34 | 83 | 1 | 157 | 214 | 531 | 31% (23–40) |
| Wg | 1057 | 35 | 66 | 1 | 121 | 259 | 575 | 35% (26–45) |
| Ash | 859 | 19 | 23 | 0 | 80 | 213 | 524 | 44% (30–59) |
| Pr | 647 | 21 | 67 | 1 | 118 | 141 | 299 | 27% (18–38) |
| Gmb | 123 | 1 | 7 | 0 | 15 | 26 | 74 | 12% (2–47) |

The rate is IA-loan / (IA-loan + inherited) at the etymon level, numerals excluded. Only
10–15% of etyma get a strict verdict; most items carry no diagnostic segment.

**Does the loan rate differ between lects?** Not detectably. Kt 29%, Kam 31%, Wg 35%,
Pr 27%, PNur 31% and Ash 44% (numbers only comparable up to Ash's re-neutralised sibilants)
all have overlapping 95% intervals. Gmb (1 of 8) is too small. The one comparable diagnostic,
initial ś-, points the same way:

| lect | ś kept | ʦ/ʣ/z | rate ś-kept (95% CI) |
|---|---|---|---|
| PNur | 21 | 8 | 72% (54–85) |
| Kt | 26 | 15 | 63% (48–76) |
| Kam | 26 | 14 | 65% (50–78) |
| Wg | 22 | 16 | 58% (42–72) |
| Ash | 7 | 12 | 37% (19–59) |
| Pr | 11 | 4 | 73% (48–89) |
| Gmb | 1 | 0 | 100% (21–100) |

Kt/Kam, Wg and Pr all keep ś- in 58–73% of initial-ś items. Ash's lower share is its s-merger,
not fewer loans. So the hypothesis that Kt/Kam are less exposed than the southern lects (Wg,
Gmb) or Pr is **not supported** by these data.

The **source** effect is much larger than the lect effect:

| lect | source | pairs | IA-loan | inherited | rate |
|---|---|---|---|---|---|
| PNur | strand | 940 | 30 | 76 | 28% |
| Kt | CDIAL | 883 | 32 | 89 | 26% |
| Kt | nured | 95 | 1 | 11 | 8% |
| Kt | strand | 1273 | 33 | 110 | 23% |
| Kam | strand | 2022 | 62 | 143 | 30% |
| Wg | CDIAL | 1147 | 39 | 70 | 36% |
| Wg | buddruss-waigali1992 | 71 | 0 | 3 | 0% |
| Wg | strand | 641 | 9 | 42 | 18% |
| Ash | CDIAL | 852 | 16 | 22 | 42% |
| Ash | buddruss-wama2006 | 136 | 0 | 6 | 0% |
| Ash | strand | 789 | 9 | 24 | 27% |
| Pr | CDIAL | 680 | 18 | 57 | 24% |
| Pr | buddruss | 102 | 2 | 7 | 22% |
| Pr | morgenstierne | 510 | 15 | 32 | 32% |
| Pr | nured | 55 | 1 | 7 | 12% |
| Pr | strand | 55 | 0 | 2 | 0% |
| Gmb | morgenstierne | 103 | 1 | 8 | 11% |

Turner's CDIAL Nuristani citations (Morgenstierne's material) have 26–42% strict loans, while
Strand's items for the same lects have 18–27%. The contrast is sharpest for Wg (36% vs 18%) and
Ash (42% vs 27%). The likely explanation is selection: Turner linked a Nuristani word to an OIA
headword mainly when it resembled the IA form, which favours loans. Strand also supplies OIA
comparanda for inherited words.

### 7.4 Semantic fields
Fields come from keyword classes over the CDIAL/source glosses of unique etyma, so the counts
are approximate.

- **IA-loan**: society/religion/abstract 19 (19%) · other 17 (17%) · tools/artefacts/house 13 (13%) · crops/food/plants 12 (12%) · animals 12 (12%) · kin/person 12 (12%) · body 7 (7%) · nature/weather 7 (7%) · numeral 3 (3%)
- **inherited**: other 57 (35%) · numeral 22 (14%) · crops/food/plants 18 (11%) · animals 17 (11%) · kin/person 12 (7%) · body 11 (7%) · nature/weather 10 (6%) · tools/artefacts/house 8 (5%) · society/religion/abstract 6 (4%)

The loan stratum is richer in society, religion and abstract terms (19% vs 4%): oath/purity
śúddʰi, neighbour vēśíya, guest/foreigner vidēśyà, country dēśá, clever kúśala, tribute śāmya,
hero śū́ra, to wish/think *vaśati, dance nŕ̩tyati, to plunder viluñcati, demon piśācá. It is
also richer in tools, weapons and the house (13% vs 5%): adze vā́śī, trap pā́śa, spear śū́la,
śalyá, *śēlla, shirt *kʰēśśa, stable/house śā́lā, peg *kōkka, bag. Food items are rice śāli,
walnut akṣōṭa, sour milk kṣīrá and oats *śyāmākka. Persons and trades are shoemaker
carmakāra and shepherd paśupā́. The inherited stratum has the numerals, the body, kinship
(jā́mātr̩, jā́nu, hŕ̩daya, dánta …), ʣ-words for cold/snow/winter and verbs of basic motion.

Caveats for curators:
- **Initial ś- before a liquid** (`ś-kept-liq`, 32 etyma): śarád 'autumn' (in all six lects),
  śará 'reed', śáru 'arrow', śáras 'cream', śarabʰá 'markhor', śárya 'reed', śāli, śā́lā,
  śū́la, śū́ra, śalyá, śilī. The sanity-set members of this group are all culture words
  (rice, stable), but autumn, reed and markhor are unlikely borrowings. A Nuristani
  conditioned development *ć- > ś- before r/l cannot be excluded. It is contradicted by
  śíras > Kam ʦer̆ and *śarva > *ʦarva, but it deserves checking against Strand's PNur before
  these are relabelled.
- **Iranian loans mimic the archaisms.** Iranian also has *ć > s and *ȷ́(h) > z, so a
  Persian or Pashto loan shows Nur-type z/ʣ. sahásra '1000' > Kt āzor, Kam āʣor, PNur *azār
  is `inherited` here but is surely Persian hazār. Items whose Nur-type signal is z rather
  than ʣ, or which have a Persian-looking shape, need a second look.
- **Wanderwörter** such as *puśśī 'cat', *cʰagana 'goat/kid' (Kt cū̃ vs PNur *ʦ̣aṇa) and
  gōccʰā are classified by their shape. Their direction of borrowing is not settled by it.
- Items in the `inherited` class are only as good as the palatal diagnostic. About 99% of
  strict verdicts in the sanity set are right. That is still one or two errors per hundred,
  so a curator should read the evidence column before relabelling.

### 7.5 The 20 clearest candidate IA loans (etymon level, numerals excluded)
Ranked by the number of lects with a strict IA-loan verdict, minus twice the number with a
strict inherited verdict.

| etymon | gloss | lects | reflexes | evidence |
|---|---|---|---|---|
| śarád | autumn | Gmb, Kam, Kt, PNur, Pr, Wg | Gmb śarei, Gmb śarēi, Kam śārvor, Kam śārvór | ś-kept-liq×12 |
| *kāśira | bright | Ash, Kam, Kt, PNur, Pr | Ash kaśirǽ, Kam kāśara, Kam kāśára, Kt kaśerə | ś-kept×9, pal×9, chain×4 |
| śā́lā | shed, stable, house | Kam, Kt, PNur, Pr, Wg | Kam pśol, Kam pśól, Kt pśol, Kt pśól | ś-kept-liq×8, chain×1 |
| śāli | rice | Kam, Kt, Pr, Wg | Kam śāli, Kam śālí, Kt śælí, Pr śili | ś-kept-liq×9 |
| vēśíya | inhabitant | Ash, Kam, Kt, PNur | Ash vaśi, Ash vaśí, Kam viśa, Kam viśá | ś-kept×8, pal×8, chain×6 |
| kúśala | right, proper; competent, clev | Kam, Kt, PNur, Pr | Kam kśɨla, Kam kśɨ́la, Kt kuśula, Kt kuśúla | ś-kept×8, pal×7, chain×2 |
| *vaśati | to wish | Kam, Kt, PNur, Pr | Kam viśa, Kam viśá, Kt vaśe, Kt vaśé | ś-kept×8, pal×8, chain×3 |
| śītalá | cold | Kt, PNur, Pr, Wg | Kt śala, Kt śalá, Kt śilə, PNur *śalai | ś-kept×8, chain×1 |
| śáras | cream, skin on boiled milk; th | Kam, Kt, PNur, Wg | Kam kṭliśɨr, Kam śɨr, Kam śɨ́r, Kt śur | ś-kept-liq×7, chain×2 |
| śúddʰi | cleansing, purity; verificatio | Kam, Kt, PNur, Pr | Kam śɨt, Kam śɨ́t, Kt śut, Kt śút | ś-kept×7, chain×5 |
| vā́śī | sharp-pointed knife or adze | Kam, Kt, PNur, Pr | Kam voś, Kam vóś, Kt voś, Kt vóś | ś-kept×7, pal×7, chain×4 |
| *śaṇṭʰara | infertile, barren, castrated;  | Kam, Kt, PNur, Pr | Kam uśtra, Kt vuśtrë́, PNur *śā̃tara, Pr iśtür | ś-kept×4 |
| śarabʰá | a kind of deer | Kam, Kt, PNur | Kam śārtor̆, Kam śāru, Kam śārú, Kt śaruv | ś-kept-liq×7 |
| vidēśyà | foreign | Ash, Kt, PNur | Ash vidiśa, Ash vidiśā, Ash vidiśā́, Kt vadaśo | ś-kept×7, pal×7, chain×1 |
| gōccʰā | furrow of upper lip | Kam, Kt, PNur | Kam guca, Kam gucá, Kt guca, Kt gucá | pal×6, chain×2 |
| dēśá | weather (obsolescent) | Ash, Kam, Kt | Ash deśī́, Kam deś, Kam déś, Kt deś | ś-kept×5, pal×5, chain×5 |
| *puśśī | cat | Kt, PNur, Pr | Kt pśæs, PNur *piśā̃, PNur *piśā̃ś, Pr pśigi | ś-kept×5, chain×2, pal×1 |
| śará | reed from which arrows are mad | Kam, Kt, PNur | Kam śɨr̆, Kt śur̆, PNur *śaṛ | ś-kept-liq×3, chain×2 |
| ścōta | oozing | Kam, PNur, Wg | Kam śtɨ, PNur *śtu, Wg aśtɘ | ś-kept×3 |
| śabála | variegated, spotted | Kam, PNur | Kam śāa, Kam śā́a, Kam śɨr̆̃, Kam śɨ́r̆̃ | ś-kept×5 |

Also clear, but in a single lect: Wg (CDIAL) aspirated items such as nacʰún 'dances'
(nŕ̩tyati), ʦ̣ʰīr 'sour milk' (kṣīrá), aʦ̣ʰṓṛ 'walnut' (akṣōṭa), cʰuvulláh 'shoemaker'
(carmakāra), villú-cʰún 'plunder' (viluñcati), erecʰún 'protect' (ā́rakṣati) and hutá
(havís, with h kept). Also Kam pā́śa > poś 'trap', Kt *kʰēśśa > kēś 'shirt', Wg śāmya > śām
'tribute', Pr úṣṭra > üścúr 'camel' and Pr piśācá > peśāśi 'demon'.

### 7.6 The 20 clearest inherited items
| etymon | gloss | lects | reflexes | evidence |
|---|---|---|---|---|
| hŕ̩daya | heart | Ash, Gmb, Kam, Kt, PNur, Pr, Wg | Ash źidi, Ash źidí, Ash źidī́, Gmb ʣo | h>ʣ×24, pal×18, chain×11 |
| himá | cold, frost, snow (in <i>himáv | Ash, Gmb, Kam, Kt, PNur, Pr, Wg | Ash zīm, Ash źim, Ash źím, Ash źīm | h>ʣ×20, pal×15, chain×9 |
| jā́mātr̩ | daughter's husband | Ash, Kam, Kt, PNur, Pr, Wg | Ash zamā́, Ash zamə́, Ash zāmā, Ash zāmā́ | j>ʣ×15, chain×2 |
| hári | golden, yellow; green | Gmb, Kam, Kt, PNur, Pr, Wg | Gmb ʣaramr̆̃o, Gmb ʣaramr̆ṍ, Gmb ʣaro, Gmb ʣarṓ | h>ʣ×12, pal×11, chain×6 |
| jātá | born; son | Ash, Kam, Kt, PNur, Wg | Ash zaga, Ash zagā, Ash zagə́, Ash zāga | j>ʣ×15, chain×12 |
| bā́hya | external | Ash, Kam, Kt, PNur, Pr | Ash bazū, Ash bəzḗi, Ash bəˈzēi, Kam bāʣgal | h>ʣ×14, pal×14, chain×11 |
| rúh | growth, sprouting | Ash, Kam, Kt, PNur, Pr | Ash źū, Ash ẓū, Ash ẓū́, Kam ẓuã | h>ʣ×13, chain×3 |
| hayana | *winter | Ash, Gmb, Kam, PNur, Wg | Ash ziã, Ash zíã, Ash zē̃, Gmb a-ʣẽ | h>ʣ×13, pal×11, chain×9 |
| hántr̩ | panther | Gmb, Kam, Kt, Pr, Wg | Gmb jõṭ, Gmb jō̃ṭ, Kam jɨt, Kam jɨ́t | h>ʣ×12, chain×5 |
| saṁdaṁśá | tongs, pincers | Kam, Kt, PNur, Pr, Wg | Kam sānuʦ, Kam sānúʦ, Kt sunuʦ, Kt ʦanuʦ | ś>ʦ×11, pal×7, chain×2 |
| śiṁśumā́ra | porpoise, Delphinus gangeticus | Kam, Kt, PNur, Pr, Wg | Kam ʦāʦam, Kam ʦāʦám, Kt ċaċë́m, Kt ʦaʦə́m | ś>ʦ×11, pal×5 |
| kā́śatē | shines, is visible | Kam, Kt, PNur, Pr, Wg | Kam kāʦa, Kam kāʦá, Kt kʦe, Kt kʦé | ś>ʦ×10, pal×7, chain×7 |
| janítra | origin, birthplace; procreativ | Ash, Kam, Kt, PNur, Pr | Ash zātr, Ash zā́tr, Kam ʣotr, Kam ʣótr | j>ʣ×9 |
| ā́gaccʰati | comes to | Kam, Kt, PNur, Pr, Wg | Kam āʦa, Kam ā́ʦa, Kt āʦ, Kt āʦe | cʰ>ʦ×8, pal×6, chain×6 |
| jámbʰa | tooth, eye-tooth, tusk; jaws | Ash, Kam, Kt, PNur, Wg | Ash zámī, Kam ʣāmoa, Kam ʣāmóa, Kt zamuv | j>ʣ×8, chain×6 |
| jā́nu | knee | Ash, Kam, Kt, PNur, Wg | Ash zā̃, Kam ʣó̃, Kam ʣõ, Kt zú̃ | j>ʣ×8, chain×6 |
| kaiśya | hair | Gmb, Kam, Kt, PNur, Wg | Gmb kẽʦ, Kam keʦ, Kam keʦ kāno, Kt keʦ | ś>ʦ×7, pal×5 |
| sahásra | 1000 | Ash, Kam, Kt, PNur, Wg | Ash āʣār, Kam āʣor, Kt āzor, PNur *azār | h>ʣ×5, pal×3, chain×1 |
| prāśa | eating, food | Kam, Kt, PNur, Pr | Kam pr̆oʦ, Kam pr̆óʦ, Kam pr̆āʦuk, Kam pr̆āʦúk | ś>ʦ×18, pal×8, chain×5 |
| hrī́ | shame | Ash, Kam, Kt, Pr | Ash źirik, Ash źirík, Ash źirīk, Kam jara | h>ʣ×13 |

(sahásra is almost certainly a Persian loan; see the caveat above.)

### 7.7 What the curator should take from this
- About 100 etyma (~290 pairs) are strict candidate IA loans, and about 160 etyma (~810 pairs)
  are strictly inherited. The rest cannot be decided from sound correspondences alone, because
  they contain no diagnostic segment.
- The most actionable subsets:
  - the `asp` items in Wg CDIAL, which are unambiguous;
  - `ś-kept` culture words shared by Kt/Kam/Pr: kúśala, vā́śī, pā́śa, dēśá, vēśíya, vidēśyà,
    śúddʰi, *puśśī, maśáka, śāli, śā́lā;
  - the `inherited` items with h>ʣ or j>ʣ, where the Nuristani archaism is unambiguous.
- Loan rates do not distinguish the lects. The difference between sources (Turner/Morgenstierne
  vs Strand) is larger than any difference between lects.

## 8. Round 2: vowels and accent

Scratch: `tmp/sound_changes/nuristani/r2/` (`OIA-*.r1.sc` = round-1 files; `acc.py` tabulates an
OIA vowel's attested outcome by accent, position, syllable count, weight, next vowel and source,
with conditional-entropy ranking; `fin.py` counts final-vowel retention by accent; `cmp.py` diffs
two rule files item by item; `var.py` batch-tests rule variants via `../try.py`). No lect uses
`%gold-drop`, so every change below is a chain change, not a scoring change.

### 8.1 NED before → after (train / test; round 1 rescored with the current engine)

| lect | round 1 | partial round 2 (06:39 snapshot) | final round 2 | rules |
|---|---|---|---|---|
| PNur | 0.4249 / 0.4397 | 0.4174 / 0.4268 | **0.4163 / 0.4256** | 42 |
| Kt | 0.5226 / 0.5416 | 0.5204 / 0.5423 | **0.5086 / 0.5206** | 46 |
| Kam | 0.5007 / 0.5211 | 0.4986 / 0.5150 | **0.4905 / 0.4911** | 49 |
| Wg | 0.5452 / 0.5633 | 0.5427 / 0.5626 | **0.5409 / 0.5634** | 41 |
| Ash | 0.5406 / 0.5257 | 0.5404 / 0.5257 | **0.5357 / 0.5247** | 39 |
| Pr | 0.6364 / 0.6397 | 0.6359 / 0.6397 | **0.6329 / 0.6344** | 45 |
| Gmb | 0.5266 / 0.4724 | 0.5221 / 0.4990 | **0.5224 / 0.4724** | 31 |

**Audit of the partial round-2 edits.** The stopped agent had added (a) posttonic apocope of
final -a (early marker `a > ă / Acc !#* C _ #`, deleted at the final-vowel-loss slot, then
unmarked) in PNur, Kt, Kam, Wg, Gmb; (b) the same for nominal -i (`if:!verb`) in PNur, Kam, Wg;
(c) Kam `ū́ > ɨ / # C* _`; (d) `if:!numeral` on every `-ati` citation rule. (d) is kept
everywhere (0 to −0.0005; `if:verb` instead costs +0.0001 to +0.0093 because the verb flag misses
many verbs). (a) and (b) are kept in PNur, Kt, Kam, Wg (tested below). (c) was replaced by an
accent-blind rule. In Gmb, (a) was reverted.

**Why Gmb's test NED rose by 0.027.** The posttonic-only rule keeps the -á of oxytones, but
Tregami loses it: five of the 30 test pairs are oxytones (saptá > sut, putrá > putr,
himá > ʣim, vr̩kká > veṭek, trivatsá > triś), each now predicted with a final -a. The
train gain (−0.0045) came from something else. The monosyllable sá > sa was kept, and some
unaccented (starred) etyma happened to keep -a. On train, too, tonic-á items lose it
(dʰūmá > dum, vāsá > vās, bʰāsá > bās, parṇá > paṭ). Accent-blind loss restricted to
polysyllables (`{a ā} > ∅ / V C* C _ #`) recovers the train gain (0.5224) and the round-1 test
value (0.4724). Gmb is therefore a *negative* accent result.

### 8.2 Rules adopted (train LOO in the final files; examples)

Frontiers were rerun for all seven lects (`sound_changes/results/OIA-*.hand.frontier.csv`, logs in `r2/frontier-*.log`).

| lect | rule | LOO | examples / note |
|---|---|---|---|
| PNur | `a > ă / Acc !#* C _ #` … `ă > ∅ / C _ #` (posttonic apocope) | −0.0065 (trio) | vā́sa > *vās; kará > *kara |
| PNur, Wg, Kam | `if:!verb i > ĭ / Acc !#* C _ #` … (posttonic -i) | −0.0011, −0.0008, −0.0010 | |
| Kt | posttonic -a lost after clusters too | −0.0034 | Kt already drops -a after a single C |
| Kam, Wg | posttonic apocope | −0.0036, −0.0017 | |
| PNur, Kt, Kam, Ash | `ī > ∅ / V C* C _ #` (final long -ī lost like -ā) | −0.0010, −0.0020, −0.0011, −0.0009 | vā́śī > voś; short -i stays (Kt: dropping -i too costs +0.0100) |
| Kt, Kam | `ā > o / C _ C* #` (ā in the last syllable of the *modern* word), after *o > u | −0.0023, −0.0060 | mā́sa > mós, bʰāga > bó, śā́lā > pśól, nā́man > núm, aṭṭāla > āṭól; but kāṇá > kār̆á, dvā́ra > dārá |
| Kt, Kam | `a > u / # C C* _ C* #` (a of a post-apocope monosyllable) | −0.0079, −0.0013 | náva > nú, padá > pú, dánta > dút, páñca > púc, sáras > súr |
| Kam | `{ū ū́} > ɨ / # C* _ C* V C* #` (ū of the first syllable of a disyllable) | −0.0010 | dʰūmá > dɨm; replaces `ū́ > ɨ / # C* _` (0.0004 worse) |
| Wg, Ash | `ā > a / _ C* {i u}` (ā shortened before a short high vowel) | −0.0018, −0.0008 | bʰārika > bari, dānin > dáni, *tānuva > tanü |
| Pr | `{a ā} > i / # C* _ C* i` (first-syllable i-umlaut) | −0.0030 | antiká > itī, aṣṭʰi > ici, hári > źiaræ, gaviṣá > giṣ |
| Ash | `{a ā} > ∅ / V C* C _ #` (apocope only in polysyllables) | −0.0019 | sá keeps its vowel |
| Ash | `{ē ḗ} > i / _ C* V` (non-final ē raised) | −0.0010 | |
| Gmb | `{a ā} > ∅ / V C* C _ #` (replaces the partial posttonic version) | train = partial, test −0.027 | see §8.1 |

**Chronology.** Apocope ≺ Kt/Kam `ā > o`: reversing costs +0.0037 in Kt and +0.0083 in Kam.
*o > u ≺ `ā > o`: +0.0056 and +0.0121. This is a drag chain: old *o rose to u, then ā moved
into o. Apocope ≺ Kt/Kam `a > u`: +0.0077 and +0.0028. So the Kt/Kam vowel shifts operate on
the post-apocope syllable structure. Pr umlaut placed early beats a late position (+0.0005 to
+0.0017).

### 8.3 Hypotheses tested and rejected (Δ train NED vs the final file; + = worse)

Accent:
- **Final -a, tonic vs posttonic.** The only place where the Vedic accent does condition
  something, and only in part.
  - *Chain test:* also dropping tonic -á costs +0.0055 in PNur, +0.0030 in Kam, +0.0024 in Kt,
    0.0000 in Wg. In Ash, Pr and Gmb the opposite holds: *keeping* tonic -á costs +0.0033,
    +0.0045 and (test) +0.027.
  - *Accented vs unaccented words:* letting unaccented (mostly starred) etyma lose -a too costs
    +0.0060 in PNur, +0.0044 in Kam and +0.0029 in Wg. Part of the "accent" gain is therefore
    the starred/unstarred split, not the accent.
  - *Alternatives:* immediately posttonic only (`Acc C* _ #`) costs +0.0026 in PNur. Weight
    instead of accent (`a > ă / C C _ #`) costs +0.0105 in PNur, +0.0089 in Kam and +0.0029
    in Wg.
  - *Direct counts* (`fin.py`, share of reflexes ending in a vowel, OIA disyllables in -Ca):
    tonic vs posttonic 58 vs 38% in Wg (z = 3.0), 59 vs 47% in PNur, 50 vs 40% in Ash,
    59 vs 44% in Pr (z ≈ 1.6), but 50 vs 50% in Kt and 49 vs 52% in Kam. In trisyllables there
    is no effect in any lect (|z| < 1.1).
- **Final -ā́, -ī́, -í kept when tonic:** −0.0002 to +0.0021 for -ā́ (rejected everywhere).
  Keeping tonic -ī́ is worse in PNur, Kt, Kam and Ash. For posttonic vs all -i: PNur −0.0001,
  Wg −0.0002, Kam +0.0010.
- **Kt/Kam ā ~ o:**
  - Kt Strand forms, first syllable: o in 30% of tonic vs 28% of pretonic ā.
  - Kam, OIA disyllables: 63% vs 47%, but trisyllables with a tonic ā give only 15% o.
  - Chain: restricting `ā > o` to tonic ā or to non-tonic ā costs +0.0033 or +0.0026 in Kam. A
    separate tonic-penult `ā́ > o` adds only −0.0006 in Kam and +0.0003 in Kt.
  - **The conditioner is the modern syllable structure (last syllable after apocope), not the
    Vedic accent.**
- **Kt/Kam a ~ u:** in OIA disyllables, u in 44% of tonic vs 43% of pretonic first syllables
  (Kt), 42% vs 36% (Kam). The conditioner is syllable count, not accent: in Kam, u/ū > ɨ in
  disyllables is 47% (tonic) vs 50% (pretonic), against 13% in trisyllables and 0% in tetrasyllables.
- **Strand's â [a] vs a [ə] in the first syllable (Kam):** â in 96% of tonic, 96% of pretonic
  and 94% of unaccented first syllables. It is purely positional, as modelled by the round-1
  strand tail. In Kt Strand forms it is ≈50/50 whatever the accent (40% â in tonic and pretonic
  syllables alike), so no rule.
- **Prasun a ~ u ~ ɨ ~ i:** first-syllable a > u in 39% of tonic, 38% of pretonic and 29% of
  unaccented syllables. A tonic-only a > u costs +0.0211. Restricting a > u to the first syllable
  costs +0.0032, to monosyllables +0.0068, and blocking it before t/n costs +0.0007. The
  umlaut rule works whatever the accent: tonic-only +0.0018, non-tonic-only +0.0003.
- **Pretonic shortening or reduction:** `ā > a / _ !#* Acc` costs +0.0018 in Wg, +0.0057 in
  PNur, +0.0025 in Kt, +0.0097 in Gmb, and ≈0 in Kam, Ash and Pr. Pretonic `ē > i` performs no
  better than the accent-blind version (Ash −0.0009 vs −0.0010) and is rejected in Wg, Kam and
  Kt (+0.0008 to +0.0011).
- **Syncope:**
  - Medial open-syllable a, early (pre-lenition): +0.010 to +0.025, whether posttonic, pretonic
    or unconditioned. Late, accent-blind: +0.0009 to +0.0032 (Ash −0.0003, not adopted).
  - Aligned medial-a loss rates show no consistent accent pattern across lects. Posttonic is the
    *best* preserved in Kam (24% lost vs 38–57%) but not elsewhere.
- **Tonic lengthening, a > ā under the accent:** not seen. In Wg, first-syllable a gives ā in
  10% of tonic and 10% of pretonic syllables.
- **r̩ a/i/u split:** about 100 tokens per lect, with outcomes scattered over 6–8 vowels and no
  accent or neighbour pattern; not modelled.

Non-accent vowel rules rejected:
- i-umlaut outside Pr: +0.0018 to +0.0079.
- u-harmony `a > u / # C* _ C* u`: +0.0007 to +0.0059 (Pr −0.0003).
- Kt/Kam `ā > o` in the other lects: PNur +0.0117, Wg +0.0069, Ash +0.0149, Gmb +0.0253 (Pr 0).
- Monosyllable `a > u` elsewhere: PNur +0.0077, Wg +0.0024, Ash +0.0074, Gmb +0.0154.
- `ā > a / _ C* {i u}`:
  - in Pr, Kam and Kt: −0.0006, −0.0005 and −0.0003 (too small to adopt);
  - in Gmb: −0.0017, but it fires on only 3 etyma, so it was not adopted;
  - in PNur: +0.0036. Strand keeps *ā in the reconstructions.
- Final -ī loss in Pr and Gmb: +0.0015 and +0.0014. Final short -i loss in Kt: +0.0100.

### 8.4 What the accent does and does not condition in Nuristani
- The Vedic accent predicts **one** thing in these correspondence sets: whether OIA final -a
  survives, in PNur (Strand), Kt and Kam, and as a raw tendency in Wg. Oxytone -á is kept more
  often than posttonic -a, and the effect is confined to disyllables.
- Even there, about half of the chain gain is the accented/unaccented (i.e. unstarred/starred)
  split. It could equally reflect Nuristani stem classes that coincide with OIA oxytone
  derivatives (-tá, -ná, -ká), rather than a PIIr accent surviving in Nuristani.
- In Ash, Pr and Gmb, tonic -á is lost like the rest.
- **Vowel quality is never accent-conditioned.** Every quality alternation flagged in round 1
  follows the *modern* word shape instead:
  - Kt/Kam ā ~ o and a ~ u: last syllable or monosyllable after apocope;
  - Kam u ~ ɨ: disyllables;
  - Prasun a ~ i: i-umlaut.
  These are Nuristani-internal developments that postdate final-vowel loss. They are a
  Kt–Kam shared innovation (rejected in every other lect), which strengthens the {Kt, Kam}
  node of §4. Wg (with Ash) instead shortens ā before a high vowel.
- Negative result: none of the tested pretonic weakening, tonic lengthening or syncope patterns
  holds, so these data give no evidence that the PIIr mobile accent survived into Nuristani.

## 9. Accent score

Scratch: `tmp/sound_changes/nuristani/r3/`. It contains `atry.py` (batch rule variants, printing
train/test plain NED and accent NED under either accent model), `apos.py` (where the attested
accent falls, by syllable count and source), `ahat.py` (Strand's â vs the accent), `vcmp.py`
(Vedic vs positional, per headword accent class), the round-2 copies `OIA-*.r2.sc` and the
frontier logs. Every placement rule leaves plain NED unchanged (train/test identical to §8).
No rule or `%source` block in these files writes a visible accent mark, so notation does not
interfere with the score. PNur and Gmb have no accent-marked forms.

### 9.1 Accent NED (train / test)

| lect | acc. pairs tr/te | none | Vedic carried | final chain |
|---|---|---|---|---|
| Kam | 694/171 | 0.5270 / 0.5378 | 0.5216 / 0.5201 | **0.4920 / 0.4889** |
| Kt | 634/154 | 0.5367 / 0.5525 | 0.5316 / 0.5417 | **0.5062 / 0.5081** |
| Ash | 578/128 | 0.5706 / 0.5522 | 0.5706 / 0.5465 | **0.5622 / 0.5379** |
| Wg | 260/68 | 0.5597 / 0.5987 | 0.5651 / 0.6080 | **0.5510 / 0.5939** |
| Pr | 174/38 | 0.6217 / 0.7176 | 0.6163 / 0.7220 | **0.6139 / 0.7131** |

### 9.2 Adopted rules (end of chain, after apocope; accent-NED LOO on train)

| lect | rules | LOO |
|---|---|---|
| Kam | monosyllable `V > $1+stress / # C* _ C* #`; disyllable last `/ # C* V C* _ C* #`; 3+ σ penult `/ V !#* _ C* V C* #` | +0.0154, +0.0160, +0.0034 |
| Kt | last syllable `V > $1+stress / _ C* #` | +0.0305 |
| Ash | the three Kam rules | +0.0039, +0.0027, +0.0017 |
| Wg | disyllable last; 3+ σ penult (the monosyllable rule has LOO −0.0007 and was dropped) | +0.0054, +0.0032 |
| Pr | last syllable | +0.0078 |

Attested pattern (`apos.py`): monosyllables are always accented. Disyllables carry the accent on
the last syllable 72–92% of the time, and 14–36% on the penult. In trisyllables the accent is
on the penult in Kam (66%, antepenult 31%) and Strand's Kt (58–71%). In Ash (61–81%) and Wg
(47–65%) it is on the final. Wg nevertheless prefers the penult rule to the last-syllable rule
(0.5517 vs 0.5547), because its chain outputs are often a syllable longer than the attested
forms.

### 9.3 Rejected

Placement position, all Δ vs the final chain:
- Last syllable placed before final-vowel loss:
  - Kam 0.5152 (+0.023), Kt (old penult) 0.5286, Ash 0.5704, Wg 0.5573.
  - Pr 0.6119 (−0.0020): the only lect that prefers it. It predicts *no* accent for apocopated
    words, and a missing accent costs 0.5 against 1.0 for a wrong one. This reflects that cost
    asymmetry, not prosody, so it was not adopted.
- Old penult before apocope: Kam 0.5168, Ash 0.5653, Wg 0.5638, Pr 0.6253.
- 3+ σ penult in Kt: +0.0016 (also "3+ σ penult only if vowel-final" +0.0003, Strand-only
  penult in a `%source` block −0.0001). In Pr: 0.6174 vs 0.6139.
- Weight (heavy last, else penult):
  - Kam: no gain. A heavy-penult rule alongside the others leaves 0.4920 unchanged.
  - Ash +0.0060, Wg +0.0198, Pr +0.0082.
- Verbs: `if:verb` penult in Pr ±0. Verbs stressed on the stem in Kam +0.0006. Final-stressed
  verbs in Kam −0.0002, not adopted.
- Penult for vowel-final disyllables instead of the final: Kam +0.031, Kt +0.026. The surviving
  final vowel is stressed.

**Vedic accent, given its best chance** (`%accent vedic`; positional rules only where they
apply, restricted with `if:!accented`):

| test | Kam | Kt | Ash | Wg | Pr |
|---|---|---|---|---|---|
| Vedic for accented headwords, last σ for the rest | 0.5090 | 0.5179 | 0.5684 | 0.5615 | 0.6150 |
| positional mono/disyllables, Vedic (else penult) in 3+ σ | 0.4932 (+0.0012) | 0.5078 (+0.0016) | 0.5651 (+0.0029) | 0.5545 (+0.0035) | — |
| Vedic only for vowel-final accented disyllables (the oxytone-á class) | 0.4991 (+0.0071) | 0.5135 (+0.0073) | | | |

Every test loses to the purely positional chain on train. On *test*, the Vedic-in-trisyllables
variant is slightly better in Ash (0.5355 vs 0.5379) and worse in Wg (0.6002 vs 0.5939). With
68–128 test pairs, and tuning on train only, this was not adopted.

**Why the Vedic accent comes second** (`vcmp.py`, same accented etyma, share of accent-NED
contributions):
- *Paroxytone disyllables:* the two predictions coincide in 72–85% of etyma (Kam, Kt, Ash, Pr).
  Posttonic apocope turns the old tonic syllable into the last syllable: vā́sa > vās.
- *Oxytone disyllables in Kam:* they coincide in 70%, because oxytone -á is kept (§8) and
  the kept final vowel is stressed.
- *Where the two differ:* earlier-accented trisyllables (Kam −0.0081, Kt −0.0061, Ash −0.0053,
  Wg −0.0049) and oxytone trisyllables. There the positional rule wins, and it is never worse
  by more than 0.001 in any class.
- So the Vedic accent scores well only where round-2 apocope has already made it the last
  syllable. **It carries no information beyond the modern last-syllable/penult rule.**
- The oxytone -á finding is consistent with this. The -á of oxytones survives, and it is
  stressed as the last syllable, not because it was the Vedic tonic. Surviving posttonic -a is
  also stressed in Kam and Kt, and giving those words the Vedic accent costs +0.007.

**Strand's first-syllable â is not an accent mark** (`ahat.py`, polysyllables with accent marks):
- *Kam:* when the first vowel is â (house ā), the accent sits on it in only 7% of disyllables
  and 8% of trisyllables. It sits on other first vowels in 41% and 49%.
- *Kt (Strand):* 6–8% on â, 1–3% on a, 42–54% on other vowels.
- *Ash (Strand):* 0–14% on â.
- So â marks an *unstressed* full-vowel first syllable. This fits §8: under the modern
  last-syllable stress, stressed a/ā became u/o in Kt/Kam (`a > u`, `ā > o`). An unshifted
  first-syllable a/ā therefore signals an unstressed syllable, and Strand writes it â, distinct
  from reduced ə.

**Conclusion.** In Nuristani the accent is a modern positional stress:
- the last syllable of the post-apocope word in mono- and disyllables;
- the penult in longer words (Kam, Ash, Wg);
- the last syllable throughout in Kt and Pr.

The Vedic accent never beats it, in any lect or any word class. Its second place is an echo of
posttonic apocope.

## 10. Extension-aware score

Scratch: `tmp/sound_changes/nuristani/r4/`:
- `menu.py LECT [split] [items]` gives, per reflex, the best variant over bare + `EXT_MENU`, the
  cost of dropping each menu item, and the gain from adding candidate suffixes, with examples;
- `etry.py` tests rule variants on plain, accent and extension-aware NED.

**No rule was changed in this round.** No change passed all three conditions, so the rule files,
the accent scores and the §9 frontiers stand.

### 10.1 Scores (train / test) and ending choices

(Accent NED is as printed by the current engine. It has been rescored since §9, with the same
rule files.)

| lect | plain NED | ext-aware NED | chosen (train): bare / aka / ikā / uka / aḍaka / allaka / araka |
|---|---|---|---|
| Kam | 0.4905 / 0.4911 | 0.4223 / 0.4385 | 68 / 7 / 6 / 5 / 10 / 1 / 3 % |
| Kt | 0.5086 / 0.5206 | 0.4513 / 0.4650 | 72 / 9 / 5 / 5 / 6 / 1 / 1 % |
| Wg | 0.5409 / 0.5634 | 0.5028 / 0.5168 | 78 / 3 / 5 / 1 / 8 / 2 / 2 % |
| Ash | 0.5357 / 0.5247 | 0.4714 / 0.4643 | 67 / 14 / 6 / 4 / 5 / 2 / 3 % |
| PNur | 0.4163 / 0.4256 | 0.3729 / 0.3830 | 79 / 5 / 6 / 2 / 4 / 1 / 3 % |
| Pr | 0.6329 / 0.6344 | 0.5826 / 0.5898 | 73 / 10 / 6 / 1 / 9 / 1 / 0 % |
| Gmb | 0.5224 / 0.4724 | 0.4626 / 0.4340 | 69 / 12 / 9 / 2 / 3 / 2 / 2 % |

### 10.2 How the chain treats the variants

Every lect loses intervocalic k, and resolves the hiatus towards the second vowel. So
gʰōṭaka/-ikā/-uka come out as:
- Kt, Kam: gur̆a / gur̆i / gur̆u; Wg guṛa…; Ash goṛa…;
- Pr: guru / guri / guru, because Pr has a > u, so -aka and -uka merge. That is why dropping
  `aka` costs Pr +0.0000.

-aḍaka, -allaka and -araka give -Vr̆a / -Vla / -Vra. So in Nuristani the "aka" variant is not a
masculine ending. It is just "the stem keeps a final -a". Its unique wins are words where the
modern form keeps a final vowel that the chain apocopates:
- Kam úśpa < áśva, āsá < ásti, ār̆á < ā́rā;
- Ash ānā́.

The chain's final-vowel treatment is correct as an ending rule. The *extended* forms do not
need fixing: there is no contraction or ordering error, and the outputs are what the
Nuristani sound laws predict.

**The -k suffix pattern.** Many Nuristani reflexes end in a *surviving* -k: Kt kavék < kapṓta,
Wg aŋarī́k < aŋgāryā́, Pr uskuk < ā́skauti, ā́sturuk < āstara, Ash apəlḗk < *alpala-, Kam
kumlík < kr̩muká. That is the suffix *-kka (MIA -kka) or a post-lenition Nuristani *-k, not
-aka, whose k is lost.

I tested whether a chain rule could produce it. Early loss of final -a after k (`{a ā} > ∅ /
V k _ #` before lenition, so that headword -ka keeps its k) makes things worse:
- plain NED +0.0007 (Pr), +0.0027 (Kt), +0.0043 (Wg);
- ext-aware NED +0.0006 to +0.0077;
- accent NED also worse.

Headword -ka *does* lose its k (rucaka > ẓúk is a syncope case, §5.3). The -k forms are a
separate suffix, so this is a menu question, not a chain rule. Nothing was adopted.

### 10.3 Should the menu change? (candidates scored by adding them to the menu; train Δ ext NED, reflexes strictly improved)

| candidate | Kam | Kt | Wg | Ash | PNur | Pr | Gmb | verdict |
|---|---|---|---|---|---|---|---|---|
| **-ukka / -ikka / -akka** | −0.0042 (53) | −0.0048 (74) | −0.0077 (70) | −0.0062 (65) | −0.0027 (20) | **−0.0149 (124)** | −0.0044 (5) | **add one geminate-k item** (see below) |
| -āla / -āli (−ka) | −0.0002 / −0.0012 | −0.0053 / −0.0027 | −0.0025 / −0.0026 | −0.0021 / −0.0007 | −0.0021 / −0.0038 | −0.0086 / −0.0066 | −0.0042 / −0.0021 | no: the wins are final-vowel fixes (ánna > ānó, ā́sa > ā́sə), not -l- |
| -ra / -ḍa / -ṭa / -la (no -ka) | −0.0036 to −0.0050 | −0.0030 to −0.0034 | −0.0026 to −0.0029 | −0.0024 to −0.0032 | −0.0016 to −0.0030 | −0.0016 to −0.0072 | ≤ −0.0023 | no: surface fixes (ávi > vé, áva > vo) |
| plain -ī / -ā / stem truncation (∅) | −0.029 (≈250) | −0.014 to −0.018 | −0.008 to −0.010 | −0.003 to −0.005 | −0.024 | −0.001 to −0.005 | −0.005 | **no**: see below |
| -iya / -īya / -ika / -uki / -aki | 0 | 0 | 0 | 0 to −0.0013 | 0 | −0.0010 (-īya) | ≤ −0.0006 | no: after k-loss and hiatus they coincide with ikā/uka |

On the three recommendations:
- **Add a -kka item** (one of -akka/-ikka/-ukka suffices; the three overlap almost entirely).
  It is real morphology:
  - MIA -kka is well attested (Pkt. -kka, NIA -k);
  - the Nuristani -k/-ik/-uk suffix is productive in Strand's and Morgenstierne's material
    (kavék, aŋarī́k, uskuk, kumlík);
  - it survives because *-kk- escaped lenition.

  It is the largest *morphological* gain in every lect except Kam and PNur (≈0.003–0.015, 50–124
  reflexes in the large lects). It would matter for the Dardic lects too.
- **Do not add "-ā/-ī/∅"**, even though it is the largest raw gain in Kam, PNur and Kt. The
  replacement of the final vowel by -ā or -ī (both lost in Nuristani) amounts to optional
  apocope. It only compensates for sources that vary in writing a final ə, and would make the
  score blind to the apocope rules.
- **Keep all six existing items.** None is unused: dropping one costs +0.0013 to +0.0166, except
  Pr aka/aḍaka/araka (+0.0000; they merge with uka through a > u) and Gmb aḍaka (0).

### 10.4 What the choices say

The bare stem wins for 67–79% of train reflexes in every lect. No Nuristani lect prefers -aka,
for the reason given in §10.2. Ash has the highest aka share (14%), which mostly reflects the
final vowels its CDIAL/Buddruss forms keep.

The -ḍa/-ra shares are Kam 10%/3% and Wg, Pr 8–9% aḍaka. They are mostly *not* the MIA -ḍa/-ra
suffixes. Their unique wins are:
- r̆/ṛ reflexes that the chain does not derive from the headword's consonants (gala > gur̆,
  aŋgúli > āŋiur̆);
- compounds (kránda > kr̆āŋ kr̆āŋ).

The one genuinely productive extension besides -k is **-l-**, a diminutive, often combined with
-ik:
- kr̩muká > Kt kəŋelik, Wg kunalik / trũelī́, Kam kumlík;
- *trip > Kam trupála, Kt travéla;
- *jʰalla > Kt ziul.

The allaka item picks these up (1–2%), and they are real.

### 10b. Rescored with -akka in the menu (EXT_MENU = aka, akka, ikā, uka, aḍaka, allaka, araka)

No rules were edited. Plain and accent NED are unchanged.

| lect | ext-aware NED train (old → new) | test (old → new) | akka chosen train / test |
|---|---|---|---|
| Kam | 0.4223 → 0.4198 | 0.4385 → 0.4353 | 7% / 7% |
| Kt | 0.4513 → 0.4485 | 0.4650 → 0.4627 | 6% / 6% |
| Wg | 0.5028 → 0.4990 | 0.5168 → 0.5143 | 10% / 9% |
| Ash | 0.4714 → 0.4661 | 0.4643 → 0.4615 | 6% / 3% |
| PNur | 0.3729 → 0.3702 | 0.3830 → 0.3811 | 4% / 4% |
| Pr | 0.5826 → 0.5688 | 0.5898 → 0.5764 | 13% / 11% |
| Gmb | 0.4626 → 0.4582 | 0.4340 → 0.4340 | 4% / 0% |

Gains, train:
- Pr −0.0138, close to the −0.0149 predicted for -ukka in §10.3;
- Ash −0.0053, Wg −0.0038, Kt −0.0028, PNur −0.0027, Kam −0.0025, Gmb −0.0044.

The test gains are of the same size, so the -k suffix generalises. -akka takes shares mainly
from -aḍaka (Kam 10 → 6%, Wg 8 → 1%, Pr 9 → 5%, Kt 6 → 3%). Some of the earlier -aḍaka
choices were therefore stand-ins for -k forms. In Pr and Wg, -akka is now the most frequent
non-bare ending.

## 11. Verb citation forms

Scratch: `tmp/sound_changes/nuristani/r4/vtry.py`. For each variant it prints train plain NED,
accent NED, noun extension-aware NED, and verb plain vs verb-menu NED. No frontier was run.

**Formations.** In every Nuristani lect the verb menu already beats the plain score. The winning
form is the 3sg present `-ati`, shown as "bare" because the headword *is* -ati, or the root. The
citation forms continue the present (or a bare stem), never a verbal noun or infinitive:
- *Strand Kt -e* (dāmayati > dāmé, *cakṣati > ṣaʦ̣é, *cʰidati > ʦie) and *Strand Kam -a* both
  continue -ati: after the loss of intervocalic t, *-ai either contracts or loses its i.
- *CDIAL/Buddruss* cite bare stems (Wg ʦen, Ash bal, Pr ps), which the root form covers.
- *Strand Ash -ie* is written as a hyphenated suffix (kōn-ie, ʦ̣akar-ie). Its e can come from
  -ai as in Kt. The i- is not derivable (see below).
- The occasional anaka/tavyakam winners in Wg, e.g. strand lāsavũ, are too few for a rule.

**Rule added (Kt, Ash).** `{a ā} i > e / _ #` (@hiatus @ai-au) is placed before the hiatus
rules. It is a plain segmental contraction of final *-ai; no `if:` condition, no @morph.

| lect | test verbs | plain NED (test) | verb-menu test before → after | train checks (plain / accent / noun-ext / verb-menu) |
|---|---|---|---|---|
| Kt | 20 | 0.5973 | 0.5396 → **0.5125** | 0.5091→0.5071 / 0.5350→0.5327 / 0.4426→0.4382 / 0.4952→0.4472 |
| Ash | 17 | 0.4971 | 0.4119 → 0.4119 | 0.5359→0.5357 / 0.5886→0.5884 / 0.4609→0.4602 / 0.4928→0.4870 |
| Wg | 29 | 0.5525 | 0.4470 (unchanged) | rule rejected: verb-menu 0.4175→0.4336, plain +0.0018 |
| Pr | 37 | 0.5716 | 0.5184 (unchanged) | rejected: verb-menu +0.012, plain +0.0013 |
| PNur | 12 | 0.4014 | 0.3240 (unchanged) | rejected: verb-menu +0.047, accent +0.029 |
| Kam | 0 test (6 train) | — | — | rejected: plain +0.0086 |

The Kt gain is large because Strand's -e is simply the regular outcome of -ati. The same rule
also improves nouns in -ati/-adi, as well as the accent score.

**Rejected alternatives:**
- In Ash, `{a ā} i > i e / _ #`, which derives the whole -ie, cuts train verb-menu NED further
  (0.4600). It worsens the accent score (+0.0002), so it fails the criteria. Strand's hyphen also
  marks -ie as a suffix rather than a sound change.
- `{a ā} i > i / _ #` has no gain in any lect.
- In Wg, Pr, PNur and Kam, the Strand/Buddruss forms keep -a or a bare stem. There, *-ai
  losing its i (the existing hiatus rule) is correct, and contraction to -e hurts.

Citation @morph rules are left in place for the plain score.

## Round 6 audit

Method as in `notes/BRIEF-audit.md`: `residuals.py` for the error classes, then single-candidate
trials with `gates.py` against the installed file (scratch copy with the same file name), a
word-level diff of every candidate, and a combined test before installing. Scratch tools:
`tmp/sound_changes/nuristani/r6/`.
- `dump.py` writes the best extension-aware variant per pair, with the `%source` tails and the
  lect's `%verb-menu` items applied as in `evaluate.ext_metrics`.
- `diff.py` lists the words a candidate makes better or worse.
- `subs.py` counts aligned predicted→attested segment operations in near misses.
- `adiff.py` gives per-etymon accent-NED differences.
- `mk.py`/`run.sh` build and score a candidate.
- The round-5 rule files are in `r6/orig/`.

No `@morph`, `if:`, numeral or starred conditions were added. No shared `use` line was added,
removed or changed; every new rule is an ordinary local line in the lect file.

### R6.1 Scores, round 5 → round 6 (before the apocope decision; final numbers in R6.6)

Installed files, train / test. Both columns are scored with today's engine (evaluate.py/sca.py
as of 02:20; re-checked at 02:45), from `gates.py r6/orig/OIA-X.sc rules/OIA-X.sc`.

| lect | ext NED | verb-menu NED | plain NED | accent NED | exact (plain) |
|---|---|---|---|---|---|
| Kt | 0.4365→**0.4247** / 0.4537→**0.4364** | 0.4343→0.4245 / 0.5058→0.4725 | 0.5071→0.4976 / 0.5249→0.5077 | 0.5327→0.5226 / 0.5544→0.5372 | 8.4→9.4% / 9.9→10.3% |
| Kam | 0.4171→**0.4077** / 0.4313→**0.4200** | 0.1778→0.1111 / – | 0.4905→0.4806 / 0.4911→0.4792 | 0.5225→0.5128 / 0.5248→0.5149 | 10.5→11.4% / 10.1→11.1% |
| Wg | 0.4805→**0.4772** / 0.4979→**0.4945** | 0.4051→0.3968 / 0.4316→0.4281 | 0.5409→0.5376 / 0.5634→0.5597 | 0.6075→0.6044 / 0.6610→0.6571 | 6.3→6.6% / 2.9→3.2% |
| Ash | 0.4586→**0.4489** / 0.4537→**0.4464** | 0.4694→0.4420 / 0.4080→0.3333 | 0.5357→0.5275 / 0.5238→0.5201 | 0.5884→0.5801 / 0.5683→0.5618 | 6.7→7.4% / 5.5→6.9% |
| Pr | 0.5550→**0.5498** / 0.5668→**0.5662** | 0.5708→0.5574 / 0.4983 (=) | 0.6333→0.6289 / 0.6369→0.6357 | 0.6490→0.6462 / 0.7704→0.7651 | 1.6→1.7% / 3.5% (=) |
| PNur | 0.3602→**0.3554** / 0.3721→**0.3706** | 0.3277→0.3227 / 0.3032 (=) | 0.4163→0.4112 / 0.4256→0.4243 | 0.5964→0.5393 / 0.5879→0.5542 | 12.7→13.3% / 12.7% (=) |
| Gmb | 0.4558→**0.4519** / 0.4340→**0.4174** | – | 0.5224→0.5135 / 0.4724→0.4557 | 0.6571→0.6161 / 0.8750 (=) | 8.3% (=) / 13.3→16.7% |

- Every lect passes every gate. Every metric improves or stays level on train and test.
- PNur's accent score rests on very few accent-marked pairs, so it jumps by several points for
  one word.
- The Ash verb-menu figures include the coordinator's `%verb-menu +aka` (R6.3).

### R6.2 Error profile

Round-5 files, `residuals.py` with the `%source` tails (as fixed mid-round). Shares are of the
error mass; the last column is the pair count after the round-6 files.

| lect | missed pairs | ending | C inserted/deleted | C quality | syncope missing | V quality | epenthesis | round 6 |
|---|---|---|---|---|---|---|---|---|
| Kt | 1938 | 49% | 18% | 9% | 9% | 7% | 4% | 1907 |
| Kam | 1792 | 50% | 19% | 9% | 9% | 5% | 5% | 1768 |
| Wg | 1752 | 54% | 18% | 8% | 6% | 6% | 6% | 1747 |
| Ash | 1659 | 51% | 16% | 10% | 5% | 7% | 6% | 1635 |
| Pr | 1337 | 43% | 21% | 11% | 8% | 8% | 6% | 1333 |
| PNur | 806 | 54% | 15% | 9% | 7% | 6% | 4% | 801 |
| Gmb | 132 | 54% | 19% | 6% | 8% | 7% | – | 131 |

The "ending" class is mostly final-vowel presence or absence: Kam ānó vs chain ān; Kt
-e/-o/-ɔ; Ash -æ; duplicate forms of one etymon that disagree. R6.4 has a large result on this.

### R6.3 Changes kept

Each change was tested alone against the file it was added to. Figures are Δ ext NED train /
test, and "better/worse" counts reflexes.

**Shared cluster and consonant rules** (the same in every lect where they passed)

- **`t s > ʦ`**: vatsá > Kt vaʦúr̆ / Kam vāʦár̆, *matsiya > Kt maʦí, útsa > PNur *uʦ,
  trivatsá > Gmb triś.
  - Kt, Kam, Wg, PNur and Gmb place it before kt/pt > t. Ash and Pr place it *late*: after Ash
    *ʦ > s, and after Pr `ʦ > z / V _`. The reflex there is ʦ, not s or z (mátsya > Ash mōʦ,
    vatsa > Pr vuʦu); the early position is half as good.
  - Kt −0.0023/−0.0048 together with the next rule; Kam −0.0002/−0.0020; Wg −0.0003/−0.0003.
- **`t > ∅ / _ {k m ʦ ʦ̣ ś s p}`**: t lost before another obstruent or m, after kt/pt > t.
  - ātmán > Kt/Kam āmú, utpātaka > upá, utkrāma > Kt vukṛom, *mr̩tka > Wg mūk.
  - Kt −0.0019/−0.0020 (42 better, 0 worse); Kam −0.0021/−0.0018 (31/0); Wg −0.0015/−0.0028
    (32/0).
  - In Ash/Pr the set is `{k m ʦ ʦ̣ ś p} | _ s C` (*ut-stʰāti > Ash ōst). Both ts rules together
    give Ash −0.0027/−0.0002 and Pr −0.0022/−0.0005.
- **`h > ∅ / {a ā} _ V`**, before `h > ʣ` (@h-loss), in Kt, Wg and Gmb.
  - kalaha > Kt kalá, prahāra > prōr, *bahira > Kt barú / Wg ber, sahita > sā, saṁnāha > Wg sanā.
  - After e/i, h gives z/ʦ (*dēhati > dez, lihati > lazé, *rēhā > r̆eʦ).
  - Kt −0.0010/−0.0002, Wg −0.0009/−0.0002, Gmb −0.0019/0.
  - sahásra > āzor/azār is the one regular exception. It is what sinks the rule in Kam, Ash and
    Pr (R6.4).
- **`r > r̆ / {k g p b m v} _`**: r after a labial or velar is the flap ř.
  - Examples: kūrma > Kt kr̆um, grāhá > gr̆óv, bʰrā́tr̩ > Kt/Kam br̆ó, prádadāti > pr̆e,
    mriyátē > Kam mr̆é, kárman > Ash kr̆em.
  - After t and d it is always r: Kt 109 tr and 42 dr, 0 flapped. So the unconditioned
    `r > r̆ / C _` costs Kt +0.0053.
  - Kt −0.0037/−0.0085 (59/9; verb-menu test 0.5125 → 0.4792); Kam −0.0043/−0.0071 (48/2);
    Ash −0.0035/−0.0035 (42/10).
  - Wg rejects it: it keeps pr, br, gr (R6.4).
- **`ṣ ṭ > ś t / {i ī u ū} _`**, after the ś > ʦ affrication.
  - niṣṭi > Kt niśte / Kam niśtá, muṣṭí > Kt miśt / Kam mr̆ɨ́śt, *śuṣṭa > śtu, viṣṭara >
    Kam viśtákāno, piṣṭá > PNur *piśa.
  - After a and after r̩ it stays ṣṭ (r̩ṣṭí > íṣṭ, pr̩ṣṭí > pṭí). Letting r̩ in fails the test
    split.
  - Counts after i/u: Kt 24 št vs 5 ṣṭ; Kam 17 vs 8.
  - Kt −0.0007/−0.0007, Kam −0.0009/−0.0010, PNur −0.0005/−0.0011.
  - Wg is neutral on train and +0.0010 on test, so not added. Ash already has `ṣ > ś / high V`,
    and Pr has an unconditioned `ṣṭ > št`.
- **`s {tʰ pʰ kʰ k} > $2 / # _`**, in Wg, Ash and PNur: initial s is lost before an aspirate and
  before k.
  - stʰāpáyati > Wg tā, stʰā́na > Ash tana, stʰālī́ > tāl, stʰīyátē > ti, skandʰá > kana.
  - Plain st- keeps it (stambʰa > śtɔ̌ma, strī́ > istri).
  - Wg −0.0010/−0.0004, Ash −0.0007/−0.0007, PNur −0.0004/0.
  - Kt and Kam fail on test (+0.0007/+0.0012: stʰála > śtal, stʰū́ṇā > śtiu). stʰū́ṇā keeps s
    everywhere (Wg uśtū́n, Ash istũ, PNur *śtu).

**Kt**
- `a > ā / _ r {t tʰ d dʰ ṣ ś ṇ n}` (@assim-r), before the r-assimilation: −0.0013/−0.0013
  (21/16).
  - karta > kāṭá, kartya > kāṭí, gʰarṣa > gā́ṣ, kárṇa > kór (via ā > o), mártya > moʦ.
  - There is no lengthening before rv or ry (sárva > suv, śárya > śúr̆). Including them flips
    the result to +0.0004 on train.
- `a > u / V C* C _ {s ʦ ʦ̣ ś ṣ k t ṭ z ʣ} #`: a in the final closed syllable of a polysyllable
  becomes u before an obstruent. This extends the monosyllable a > u.
  - −0.0017/−0.0019 (22/4): *ārasa > ārús, vasantá > vasút, cáturdaśa > śtrúʦ,
    *kuḍakk- > kur̆uk, *mūṣakk- > muzúk.
  - The unconditioned version fails the accent gate (+0.0016). Its losses are before r/r̆/m
    (kakkʰaṭa > kakar̆, maraṇa > mará, andʰakāra > anər).
- `ā > ∅ / V C* _ V`, after the prevocalic a-loss: −0.0007/0 (nāsāpuṭa > nāsúr̆,
  *vātaputrī > vútr, śrā́yati > ṣē).
  - The unrestricted `{a ā} > ∅ / _ V` fails on test, because initial ā- stays (*āturati >
    ātravé, *āpula > ā́la).

**Kam**
- The Kt set, where each part passes on its own:
  - ā-hiatus −0.0001;
  - ar-lengthening −0.0008/0 (39/0: mártya > móc, ardʰá > āṭá, árdʰa > āṭó̃);
  - ts −0.0002/−0.0020;
  - t-loss −0.0021/−0.0018;
  - r̆ after labials/velars and ṣṭ > št (above).
- **Bug fix in the ū > ɨ rule.** ɨ is not a vowel in the segment profile
  (`sca.kind('ɨ') == '?'`).
  - Effect: once the early `{ū ū́} > ɨ` fired, later `V _ V` rules no longer saw a vowel. So
    gūtʰa kept its t (*gɨta), pū́ti gave *pɨt and yū́kā gave *iɨk.
  - Fix: the rule now writes ü, which the engine treats as a vowel and which does not occur in
    Kam's data. A new line `ü > ɨ` after accent placement turns it back into ɨ.
  - −0.0012/0 (gūtʰa > gɨ, pū́ti > pɨ, yū́kā > ɨ).

**Wg**: ts, t-loss, h-loss and initial s-loss, as above.

**Ash**
- ts and t-loss (the late ts, see above); r̆ after labials/velars; initial s-loss.
- **a before a nasal cluster becomes o in monosyllables** (−0.0016/0; 7 better, 0 worse).
  - Examples: dánta > dó̃t, páñca > pó̃ʦ, gandʰá > gōn, māṁsá > mós, *kʰandati > kōn. This is
    the Ash counterpart of Kt/Kam monosyllable a > u.
  - Implementation, with a marker vowel å:
    1. `{a ā} > å / C _ {n ñ ṁ} C` runs before the nasal-cluster rules;
    2. å contracts like a in hiatus;
    3. late, `å > o / # C* _ C* #`, and å > a elsewhere.
  - Word-initially and in polysyllables a stays (ánna > ānā, páñcadaśa > paʦūs,
    saṁdaṁśá > sandús). Without those restrictions the rule loses (test +0.0014).
  - A first version used ã as the marker. Its unmarking step silently removed genuine
    nasalisation that a later `$1+nas` rule produced.
- **`%verb-menu +aka`**, the coordinator's placebo-tested item, confirmed here with gates.py:
  - verb-menu 0.4569 → 0.4420 train, 0.3962 → 0.3333 test; ext 0.4502 → 0.4489 /
    0.4493 → 0.4464;
  - placebo `+ara`: 0.4569 → 0.4543 / 0.3962 → 0.3962.
  - The wins are CDIAL/Buddruss stem + -a/-ā citations (*utplāvayati > uplā,
    *ut-stʰāpayati > ō̃stā́, *avalagyati > uluŋa).
  - The same item with the same placebo does nothing in the other lects. Kt −0.0054 train, 0
    test (placebo 0); Kam, Wg, Pr, PNur 0 (placebo −0.0069 in Pr). Not added there.

**Pr**
- ts (late) and t-loss −0.0022/−0.0005.
- `ā > ∅ / V C* _ V` −0.0003/0 (jā́mātr̩ > zami).
- **`{d dʰ} > l // {n ṇ d} _ | _ {d dʰ}`**: Pr d/dʰ > l, but not after a nasal and not in the
  geminate dd(ʰ).
  - Those stay d: gandʰá > gúndu, índra > indr, buddʰi > bɨdɨ, yuddʰá > ɨd, uddʰāna > udosik.
  - −0.0030/−0.0001 (19/2).

**PNur**
- ts and t-loss −0.0023/−0.0004.
- `a > ā / _ r {ṇ n}` −0.0016/0 (kárṇa > *kār, parṇá > *pār). Strand's PNur has no
  lengthening before r + dental stop (*kaṭā, *gada, *kaṭala); the Kt set is +0.0009 on test.
- ṣṭ > št and initial s-loss (above).

**Gmb**: ts, t-loss and h-loss together −0.0039/−0.0167. The test gain is trivatsá > triś.

### R6.4 Rejected candidates

Δ ext NED train / test unless stated.

**Final -a: full apocope wins the extension-aware score but fails the gates.** This is the
largest single signal of the round.

Two tests:
- `{a ă} > ∅ / C _ #`, i.e. every final -a after a consonant lost, tonic -á included;
- `a > ă / C _ #` before accent loss, i.e. the -a of unaccented (mostly starred) words also
  apocopated, tonic -á kept.

| lect | test | ext NED Δ (train / test) | reflexes better / worse | plain Δ (train / test) | accent Δ (train / test) |
|---|---|---|---|---|---|
| Kt | all -a | **−0.0166 / −0.0199** | 231 / 19 | +0.0042 / +0.0005 | +0.0071 / +0.0062 |
| Kam | all -a | **−0.0337 / −0.0382** | 355 / 19 | +0.0206 / +0.0206 | +0.0276 / +0.0307 |
| PNur | all -a | **−0.0264 / −0.0277** | 173 / 14 | +0.0135 / +0.0099 | +0.029 / +0.056 |
| Wg | all -a | **−0.0080 / −0.0066** | 117 / 7 | +0.0030 / +0.0065 | +0.0005 / +0.0023 |
| Kt | unaccented -a | −0.0136 / −0.0099 | 192 / 19 | +0.0016 / +0.0057 | +0.0055 / +0.0162 |
| Kam | unaccented -a | −0.0197 / −0.0168 | 230 / 18 | +0.0097 / +0.0067 | +0.0112 / +0.0098 |

Why the metrics disagree:
- With apocope, the bare form matches the consonant-final reflexes: gandʰá > gun,
  *cāṭṭa > coṭ, gaṇá > Kam gur̆, pakvá > PNur *pak.
- The reflexes that do end in a vowel are then matched by the -aka variant (guṣṭá, kāṭá).
- The plain and accent scores only see the bare headword. There, the kept final vowel (and the
  final-syllable stress on it) wins.
- So this is exactly the bare-vs-extended artifact that the plain gate's exception describes.
  The accent gate has no such exception, so nothing was installed.

**Recommendation for the coordinator:** under the extension-aware score, the Kt/Kam/PNur/Wg
"kept final -a" (§8.2's tonic-á retention and Kam's ə) is better read as the -aka
extension's vowel than as a surviving OIA -a. If the accent gate were made
extension-aware, or waived for this bare-vs-extended case, full apocope should be adopted in
these four lects.

**Other rejections**
- **Kam `{u ú ū ū́} > ɨ` in the first syllable of disyllables** (extends the ū rule to short u).
  - ext −0.0061/−0.0061, 45 better and 2 worse: kúla > kɨr̆, túṣa > tɨs, tulā́ > tɨr̆.
  - It fails on accent +0.0072/+0.0077. `realize` only accents segments of kind V, and ɨ is
    not one, so every ɨ́ in the gold is an accent miss.
  - This is an engine/profile limitation, not a bad rule. Making ɨ a vowel in the profile
    would let it through. I did not touch sca.py or the profile.
- **Wg ND > N** (`{d ḍ b g} > ∅ / Nasal _`, as in Kt/Kam/Ash/Gmb; or {d ḍ} or {d ḍ g} only).
  - ext −0.0015 to −0.0022 train, −0.0017 to −0.0037 test; 38–66 better against 15–23 worse.
    Across sources 88 reflexes lose the stop and 21 keep it.
  - It fails on accent (+0.0045 to +0.0055 train). The accent-marked Wg forms are the ones that
    keep the stop (gúnd, andá, kuṇḍǽi, deṇḍī́, vindá-ún), and NED's max-length denominator
    favours the longer prediction (ganá vs gún scores worse than gandá vs gún).
  - Real, but blocked by the gate.
- **h-loss in Kam** (−0.0018 train, +0.0003 test: nahí, sahásra), **Ash** (test +0.0001) and
  **Pr** (verb-menu +0.0026). PNur passes ext but fails accent (+0.036, a tiny sample).
- **ar-lengthening** in Wg (+0.0002/+0.0018; Wg vaṣ, kaṣ), Ash (+0.0003), Gmb (+0.0022) and
  Pr (verb-menu +0.0026; Pr has no r-assimilation in its file, and vártra > vaṭ is CDIAL).
- **ā-hiatus (`ā > ∅ / V C* _ V`)** in Wg (+0.0022), Ash (+0.0005) and Gmb (+0.0017).
  - Wg, Ash and Gmb resolve hiatus towards the *first* vowel (adyāpi > ajā,
    jā́mātr̩ > zamā/zāmā).
  - Kt, Kam and Pr have the opposite, prevocalic a-loss. This is a clean split within the group.
- **Wg `r > r̆ / {k v} _` or `{k v m} _`**: +0.0003 / +0.0009. Wg keeps pr, br and gr (br 19 vs
  br̆ 2).
- **Initial s-loss in Kt/Kam** (test +0.0007/+0.0012) and **Gmb** (+0.0035).
- **ṣṭ > št in Wg** (test +0.0010), and the version that includes r̩ in Kt/Kam (test +0.0039 /
  +0.0025).
- **Kt/Kam i-glide before u** (ū > iu in disyllables; Strand/CDIAL gǐu, dǐum, pǐu): Kt
  −0.0004/+0.0009, and +0.0034 when u is included. It is lexical: sū́ra and śū́ra stay u,
  although Kam has ɨ in the same words.
- **Kam medial i > a** (Strand ə), three variants: +0.0001 to +0.0015 train. This is notation
  variation.
- **Ash intervocalic `{n ṇ} > r̆`**: test +0.0007 (jána > zen, śōdʰana > sōnə).
- **Ash final nasal after apocope → nasal vowel** (pēṣaṇa > pisã́, maraṇa > mārā̃,
  sumanas > sumõ): ext −0.0008/−0.0006, but verb-menu test +0.0028, because the -aṇa verbal
  nouns lose their n (vḗnati > vēṇ).
- **PNur nasal vowel before nt/ñc** (*pãc, *dãt): +0.0037. Strand also has *āt, *cit, *mat.
- **Wg a before NC > u in the first syllable of OIA disyllables**: it fires only on gandʰá; plain
  +0.0009.
- **Pr d-loss before a consonant** (udgama > agam): it breaks udgama/*udgʰāla through the later
  g-loss; accent +0.0046.
- **Kt `V $1 > $1`** (identical-vowel contraction): 1 word, accent +0.0002.
- **ND > N only word-finally** (`{d ḍ b g} > ∅ / Nasal _ #` in place of the shared rule), to
  keep andá, índra and kandá in Ash:
  - Ash −0.0004/−0.0006 ext, but accent +0.0010 and plain +0.0011;
  - Kt +0.0033, Kam +0.0048, Gmb +0.0044.
  - Dropping the rule in Ash costs +0.0015. The shared rule stays.
  - Note: the class name `Nasal2` exists only inside changes.sc; written in a lect file it is a
    literal token and never matches.
- **Kt final-syllable o > u before a nasal** (nā́man > núm, grā́ma > gr̆úm): 4 better, 3 worse
  (nɔm, gr̆om). Source variation, not installed.

### R6.5 What remains

- **Final vowels (≈45–55% of the error).** See the apocope result in R6.4. The remaining final
  -o/-e/-æ/-ɔ endings (Kam -oa, Kt -e/-ɔ, Ash -æ) are suffixal and vary within a source.
- **Vowel quality in the first syllable.**
  - Kt i ~ ə ~ a (45 i→a vs 56 i→i).
  - Wg monosyllabic a ~ o ~ u (CDIAL: 65 a, 29 o, 28 u).
  - Pr's unconditioned a > u (many u → a/o/e misses).
  - Round 2 tested every positional and accentual split, and none holds. These are dialect or
    source differences.
- **Initial prothesis and loss.**
  - Kt/Pr v- before rounded initial vowels (utkrāma > vukṛom, but u- in 18 of 32).
  - Pr initial p- > v- (31 v vs 55 p). This looks like inherited vs borrowed vocabulary
    (páñca, padá, pántʰā vs pāṇḍava, piśācá), not a phonological condition.
  - Pr initial d stays d in dr- and in some loans (dárvi > dur, druṇa > duru-).
- **PIIr palatal residue** (ʦ vs ś: dēśá > Kt déś, vā́śī > voś, kúśala > kśulə): §5.1, not
  predictable from OIA.
- **Medial syncope** (Kt/Kam āmuñcati > āmcá, avalambatē > ālmé): see §8.3; still not
  regular.
- **Engine.** ɨ is not a vowel (see Kam above). A profile fix would let Kam/Wg rules produce ɨ
  early and carry an accent on it.

### R6.6 Coordinator decision: full final -a apocope (Kt, Kam, PNur, Wg), and a near-miss pass

**Apocope adopted.** The coordinator waived the accent gate for this case only.
- Installed `{a ă} > ∅ / C _ #`: in Kt it replaces `{a ă} > ∅ / V C _ #`; in Kam, PNur and Wg
  it goes right after `use final-vowel-loss.a.c.3`.
- The earlier `ă` marking rules stay but are now redundant.
- Δ against the installed files, train / test:

| lect | ext | plain | accent | verb-menu |
|---|---|---|---|---|
| Kt | −0.0166 / −0.0199 | +0.0042 / +0.0005 | +0.0071 / +0.0062 | −0.0048 / −0.0033 |
| Kam | −0.0337 / −0.0382 | +0.0206 / +0.0206 | +0.0276 / +0.0307 | 0 / – |
| PNur | −0.0264 / −0.0277 | +0.0135 / +0.0099 | +0.029 / +0.056 | +0.0018 / **+0.0107** |
| Wg | −0.0080 / −0.0066 | +0.0030 / +0.0065 | +0.0005 / +0.0023 | **+0.0016** / 0 |

**Reinterpretation.** §8.2/§8.4 read the survival of tonic -á as a trace of the Vedic accent.
Under the extension-aware score that final vowel is better read as the vowel of the -aka
extension (gōṣṭʰa+aka > Kt guṣṭá, karta+aka > kāṭá). The OIA -a itself is lost everywhere.
- The accent-conditioned apocope of round 2 was therefore partly scoring the bare headword
  against reflexes built on the extended stem.
- The plain and accent scores only see the bare headword, which is why both got worse.

**Flag:** the verb-menu gate, which was not waived, is also exceeded in PNur (+0.0018 train,
+0.0107 test) and Wg (+0.0016 train).
- Cause: -itavya menu forms that had matched Strand's thematic -a verb citations lose their
  -a (ravatē > *rava, yábʰati > *iava, jī́vati > *jiva, cyávatē > Wg cɨva).
- It is the same artifact on the verb side: the -a belongs to a citation formation missing
  from the verb menu, a thematic stem in -a.
- Installed anyway, as decided. A `%verb-menu` thematic item with a placebo test would be the
  proper fix.

**Near-miss pass (NED ≤ 0.25, by differing segment pair), Kt and Kam, after apocope**
- Kept, **Kt `i > a / # C* _ C V #`**: first-syllable i of a disyllable that ends in a vowel is
  reduced to ə, written a.
  - *nijyati > najé, *nīḍāti > nar̆e, lihati > lazé, *niccʰāya > naʦé, r̩ṣabʰá > aṣə.
  - ext −0.0040/−0.0024, verb-menu −0.0103/−0.0175, accent −0.0021/0 (51 better, 11 worse).
  - Kam rejects the same rule: +0.0011 accent, 34 better vs 42 worse.
- Kept, **Kam `{u i} > ∅ / V C _ C V`**: syncope of a short high vowel after the first syllable.
  - āmuñcati > āmcá, ārōgya > ārgú, mā́nuṣa > mānśa, *bahira > bārá.
  - ext −0.0015/−0.0008, verb-menu −0.0444 train (42 better, 13 worse).
  - The u-only version is −0.0006/−0.0008. Kt rejects u-syncope (accent +0.0004).
- Rejected:
  - r̩ > a before a vowel: Kt ext test +0.0003; Kam test +0.0010.
  - The Kt verb -e vs bare-stem split (kā́satē > kas vs dāmé): source variation.
  - Kt first-syllable a-loss before a final -e (kaṣati > kṣe, pacyátē > pcé): 13 better,
    38 worse (gaṇáyati > gar̆é, cálati > calé), +0.0026/+0.0045.
  - Medial a-syncope before r/l (andʰakāra > Kam ānrá, karta > kāṭlá): Kam +0.0006/+0.0014,
    Kt −0.0005/+0.0002 with accent +0.0005.
  - Kam -oa verb endings: suffixal.

**Final scores, round 5 → round 6** (`gates.py r6/orig/… rules/…`, train / test). Ash, Pr and
Gmb are unchanged from R6.1.

| lect | ext NED | plain | accent | verb-menu | exact | gates vs round 5 |
|---|---|---|---|---|---|---|
| Kt | 0.4365→**0.4041** / 0.4537→**0.4142** | 0.5071→0.5001 / 0.5249→0.5076 | 0.5327→0.5276 / 0.5544→0.5434 | 0.4343→0.4094 / 0.5058→0.4517 | 8.4→10.4% / 9.9→10.6% | all pass |
| Kam | 0.4171→**0.3725** / 0.4313→**0.3810** | 0.4905→0.5011 / 0.4911→0.4995 | 0.5225→0.5403 / 0.5248→0.5453 | 0.1778→0.0667 / – | 10.5→11.5% / 10.1→12.3% | plain, accent worse (waived apocope) |
| Wg | *(apocope reverted, R6.7; final = R6.1: 0.4805→0.4772 / 0.4979→0.4945)* | | | | | all pass |
| PNur | *(apocope reverted, R6.7; final = R6.1: 0.3602→0.3554 / 0.3721→0.3706)* | | | | | all pass |

### R6.7 Follow-up: thematic verb item tested, apocope reverted in PNur and Wg

The waiver covered accent only, so the PNur and Wg verb losses had to be recovered or the
apocope reverted. I tested `%verb-menu +a` (root + thematic -a) against same-shape placebos,
each against the file with apocope:

| lect | item | verb-menu Δ (train / test) | ext Δ (train / test) |
|---|---|---|---|
| PNur | +a | −0.0026 / 0 | −0.0003 / 0 |
| PNur | +u (placebo) | −0.0033 / 0 | −0.0003 / 0 |
| PNur | +i (placebo) | 0 / 0 | 0 / 0 |
| Wg | +a | 0 / 0 | 0 / 0 |
| Wg | +u (placebo) | −0.0013 / 0 | −0.0002 / 0 |
| Wg | +i (placebo) | 0 / 0 | 0 / 0 |

- +a does not beat the placebos, and the PNur verb test loss (+0.0107) stays.
- The reason is structural: a thematic -a added to the root is itself removed by the new
  `{a ă} > ∅ / C _ #`, and -ā by the -ā apocope. No citation item ending in -a can survive a
  full-apocope chain.
- **The apocope is therefore reverted in PNur and Wg.** It stays in Kt and Kam, where the
  verb-menu score improves or is level.
- The installed PNur and Wg files are exactly the R6.1 state, and all gates pass against
  round 5:
  - PNur: ext 0.3602→0.3554 / 0.3721→0.3706; verb 0.3277→0.3227 / 0.3032 (=).
  - Wg: ext 0.4805→0.4772 / 0.4979→0.4945; verb 0.4051→0.3968 / 0.4316→0.4281.
- For PNur and Wg, the apocope evidence of R6.4 stands but is blocked by Strand's -a verb
  citations. One way out is to generate those citations from a stem that is not a final -a,
  but that is a menu design question for the coordinator.

### R6 fourth pass (Pr and Wg class by class, then Kt/Kam near misses; gate now on ext_acc)

`gates.py` now gates accent on `ext_acc`, the accent scored on the variant that the
extension-aware score picks. So some candidates rejected earlier on bare-headword accent were
re-run.

**Kept** (Δ ext train / test against the file at the time; all pass `gates.py`)
- **Pr `a > u // _ #`**: a word-final a stays a/ə.
  - Monosyllables: gatá > ga, nakʰá > na, yábʰati > iza. The -a of extended stems: ā́sa > āsa,
    áp > ā́və.
  - −0.0147/−0.0155, verb-menu −0.0190/−0.0110 (121 better, 27 worse). This is the largest
    single Pr gain of the round.
  - Rejected variants: keeping a in the whole final syllable (test ok, train +0.0029); keeping
    it in non-first syllables (+0.0065); keeping it in open syllables (+0.0046).
- **Pr `ā > a / C _ // _ #`**: word-final ā stays long (śā́kʰā > ʦāv, dʰrā́yati > lār,
  pāyáyati > pār). −0.0015/−0.0050.
- **Pr `{u i} > ∅ / V C _ C V C* #`**: penult syncope of a short high vowel, including u from
  *a.
  - āmuñcati > āmju, dʰamáni > lümī́, kr̩muká > kumlugi, prasupti > pust.
  - −0.0038/−0.0053 (114 better, 30 worse). It failed only the old bare accent gate, because of
    one unrelated test form (yunákti > yū́pu).
- **r lost before a labial** (after lenition, so v survives).
  - **Wg** `r > ∅ / V _ {p pʰ b bʰ m v}`: pū́rva > pə̄v, *śarva > ʦə̄v, sárva > sab,
    dārva > dāv, cárman > cām, nirmala > nimal, carpaṭa > capā́l. −0.0017/−0.0003 (21 better,
    3 worse: gárbʰa > grop, dárvi > dur).
  - **Kt, Ash, PNur, Gmb** without m, since rm metathesises there (kūrma > kr̆um):
    - Kt −0.0013/−0.0004 (20/2);
    - Ash −0.0010/−0.0003;
    - PNur −0.0015/0 (*sav, *capa);
    - Gmb −0.0008/0.
  - Kam rejects both versions (*tarpati > trāp, dárvi > dur).
- **Prothetic i- before initial st.**
  - **Wg** `∅ > i / # _ ś t {a ā r}`: stara > iśtor, staraṇa > iśtaræ̃, stāva > iśtāo. Not
    before u (stū́pa > śtu); the unrestricted version is +0.0002 on test. −0.0015/0.
  - **Ash** `∅ > i / # _ s t`: stŕ̩ > istā́, strī́ > istrī́. −0.0010/0.
  - Kt rejects it: 0 better, 28 worse.
- **Kt early -a loss after a single labial** (`{a ā ă} > ∅ / V {p pʰ b bʰ v} _ #` before
  final-consonant loss and lenition, then `{p b} > v / V _ #`).
  - apūpá > úv, *ēva > ev, kakubʰá > kúv, lāva > r̆óv. −0.0023/−0.0015 (13 better, 0 worse).
  - It must run before final-consonant loss, so that śrávas (> ṣu) is not caught.
  - Rejected in Kam, Wg, Ash and PNur, on plain, verb or ext_acc.
- **Kt `u > ∅ / V C _ C V`** (medial u syncope, as Kam): āmuñcati > āmcé, *utpusati > aps.
  −0.0003/−0.0008, ext_acc −0.0007/−0.0010. It failed only the old bare accent gate.
- **Kam final-syllable a > u before obstruents** (the Kt rule).
  - Rejected before the apocope; after it, −0.0024/−0.0004 (35 better, 10 worse).
  - Examples: vasantá > vāsút, *āraśa > ālúʦ, turá > tɨruk.
- **PNur `h > ∅ / {a ā} _ V`**: −0.0026/−0.0002. It had failed only the bare accent gate.

- **Final nasal after apocope → nasal vowel** (`V > $1+nas / V C* _ {n ṇ} #`, then the n goes),
  in **Ash** and **Wg**.
  - Ash: pēṣaṇa > pisã́, maraṇa > mārā̃, sumanas > sumõ. −0.0008/−0.0007, ext_acc
    −0.0011/−0.0010. It had failed the verb test under the old verb menu; it passes now.
  - Wg: káraṇa > karã, niṣádana > niṣã́, aśáni > aśē̃́. −0.0003/−0.0019, ext_acc test −0.0116.
  - No effect in Kt/Kam, where n has already become r̆. Pr is worse on verbs (+0.0047).
- **Wg ts > ʦ moved after post-vocalic ʦ > ś** (as in Ash/Pr); the t-loss set drops s and adds
  `_ s C`. útsa > ɨʦ, *pītsa > pɨʦ, vatsalá > vuʦalá. −0.0010/0 (5 better, 0 worse).

**Rejected** (fourth pass)
- **Wg ND > N**: still fails on ext_acc +0.0032 train and verb-menu test +0.0014.
- **Kam ɨ for short u**: ext_acc +0.0023/+0.0015, the engine's ɨ problem.
- **Unconditioned final-syllable a > u** (dropping the obstruent condition): Kt ext_acc
  +0.0004, verb test +0.0056; Kam +0.0024.
- **Pr vowels.**
  - Tabulating OIA first-syllable a by onset, coda and syllable count gives u ≈ 45% in every
    cell, so there is no conditioning to find beyond the final-a exception.
  - Also rejected: dropping or restricting `ā > a / C _` (+0.0022/+0.0044); `e > i // _ #`
    (no train change).
- **Initial s-loss in Kt/Kam** and **h-loss in Kam**, re-run after the apocope: still worse on
  test.

**Apocope re-check under the new gate.** Δ from the same file without the apocope rule to the
installed file:
- **Kt**: ext_acc −0.0245 / −0.0344 and ext −0.0170 / −0.0211. Plain is +0.0032 / −0.0010.
  Verb-menu is **+0.0014** train / −0.0028 test: the -tavya menu forms lose their -a, the same
  artifact as in PNur.
- **Kam**: ext_acc −0.0464 / −0.0491, ext −0.0362 / −0.0396, plain +0.0193 / +0.0208 (the
  justified bare-vs-extended loss). Verdict ACCEPT-IF-JUSTIFIED.
- **PNur/Wg, re-gated on the current files:**
  - PNur: ext −0.0267 / −0.0277, ext_acc 0 / 0, but verb-menu +0.0018 / +0.0107.
  - Wg: ext −0.0081 / −0.0066, but ext_acc +0.0012 / +0.0023 and verb-menu +0.0014 train.
  - Both still fail, so both stay reverted.

**Final scores, round 5 → round 6** (current engine, `gates.py r6/orig/… rules/…`; train /
test). This table supersedes R6.1 and R6.6.

| lect | ext NED | ext_acc | plain | verb-menu | verdict vs round 5 |
|---|---|---|---|---|---|
| Kt | 0.4361→**0.3998** / 0.4530→**0.4109** | 0.4475→0.4029 / 0.4627→0.4096 | 0.5071→0.4988 / 0.5249→0.5073 | 0.4416→0.4141 / 0.4453→0.4002 | ACCEPT |
| Kam | 0.4171→**0.3701** / 0.4313→**0.3806** | 0.4392→0.3840 / 0.4553→0.3984 | 0.4905→0.5007 / 0.4911→0.4988 | 0.1778→0.0667 / – | ACCEPT-IF-JUSTIFIED (plain: apocope) |
| Wg | 0.4802→**0.4724** / 0.4979→**0.4923** | 0.5112→0.5018 / 0.5497→0.5328 | 0.5409→0.5330 / 0.5634→0.5575 | 0.4003→0.3921 / 0.4352→0.4309 | ACCEPT |
| Ash | 0.4598→**0.4468** / 0.4537→**0.4445** | 0.4937→0.4792 / 0.4821→0.4720 | 0.5357→0.5248 / 0.5238→0.5194 | 0.4800→0.4497 / 0.4510→0.3708 | ACCEPT |
| Pr | 0.5549→**0.5297** / 0.5668→**0.5404** | 0.5421→0.5161 / 0.6252→0.5922 | 0.6333→0.6277 / 0.6369→0.6351 | 0.5842→0.5456 / 0.4983→0.4782 | ACCEPT |
| PNur | 0.3602→**0.3513** / 0.3721→**0.3704** | 0.5321→0.4750 / 0.4976→0.4639 | 0.4163→0.4077 / 0.4256→0.4243 | 0.3277→0.3227 / 0.3032 (=) | ACCEPT |
| Gmb | 0.4558→**0.4511** / 0.4340→**0.4174** | 0.5684→0.5563 / 0.7500 (=) | 0.5224→0.5121 / 0.4724→0.4557 | – | ACCEPT |

## Round 7

Method as in `notes/BRIEF-audit.md` and `BRIEF-round7.md`. Scratch tools are in the session scratchpad
`r7nur/`: `mk.py`/`run.sh`/`dump.py`/`diff.py`/`subs.py` from round 6; `vq.py`/`vm.py` for
pred→gold vowel matrices per source; `ctx.py` for a pred vowel's gold outcome by position and
neighbours; `vdiff.py` for per-verb chosen variants (bare + VERB_MENU, as `verb_metrics`) before
and after; `ablate.py` for a leave-one-rule-out pass. The round-7 starting files are in `r7nur/orig/`. They already contain the early-morning
round-7 lines (Wg monosyllable stress, PNur final stress and final-nasal rules, Pr prothetic
i-), which were taken as the start state.
No `@morph`, `if:`, numeral or starred condition was added, and no shared `use` line was changed.

### R7.1 Engine and scoring issues found (reported; fixed by the coordinator)

- **Accent marks on ɨ ə ʊ survived in gold.** The profile splits their combining acute off into its
  own token, and `gold_tokens` re-attached it after the accent-drop step, so gold ɨ́ (Kam Strand, about
  70 tokens) and ə́ (Kt CDIAL) could never match. Fixed 08:15; re-baselined.
- **Strand's ǐ is a glide letter, not a tone.** Every caron in Kt (57 Strand + 1 CDIAL), Kam (13) and
  Ash (20) is Strand's non-syllabic i (gǐu, ʦǐe, ǐuṣ, kōṣ-ǐe). The accent and tone scores read it
  as the accented vowel, stressing the glide. Now `%gold-drop caron` in Kt, Kam and Ash: a scoring
  declaration (coordinator), noted, not gated.
- **CDIAL ɔ̌** is the house rendering of Turner's ɔ̈̄ (long centralised ɔ: Kt yákr̩t > yɔ̈̄k → yɔ̌k,
  also Wg 3, Pr 1). One such form made the whole CDIAL source tone-scored in its lect. Fixed globally.
- **Kt CDIAL śikyà** (f_f2ekngihsqnya) is the OIA headword, parsed as the Kt reflex from the "GM 22"
  citation. Its Vedic grave made every CDIAL Kt pair tone-scored. Stopgap: `grave` added to Kt's
  `%gold-drop`, to be removed when the record is fixed.
- After these fixes no Nuristani lect has a tone-scored source: the "tone types" were all notation.
  So the Vedic-accent → tone-type hypothesis has nothing to test here.

### R7.2 Vowel qualities after the scoring change (task a)

`vm.py`/`vq.py` tabulate every aligned pred→gold vowel pair per source. The lax/central vowels come
almost entirely from two places: Strand's ɨ (Kam, Wg, some Pr) and Morgenstierne's ə/æ/ɔ as
cited by CDIAL (Kt, Ash, Pr, Wg). Only two of these are regular.

**Kept**
- **Kam `{u ú ū ū́} > ɨ / # C C* _ C* V C* #`** replaces the round-2 ū-only rule and its round-6
  ü placeholder (ɨ is now a vowel to the engine; the placeholder removal alone is score-neutral).
  u/ū of the first syllable of an OIA disyllable, after a consonant, is centralised: kúla > kɨr̆,
  túṣa > tɨs, buddʰa > bɨt, dʰūmá > dɨm, pū́ti > pɨ. ext −0.0043/−0.0045, ext_acc −0.0049/−0.0047,
  plain −0.0006/+0.0026; 25 reflexes better, 6 worse (kunta > kut, śúci > suc, puṇḍra > puṭ).
  Word-initial u- stays (úpa > u; the version with initial u- is identical on ext but +0.0013 plain);
  every first syllable (trisyllables too) is +0.0073/+0.0030.
- **Kt `%source CDIAL`: `a > ə / V C* _`** (@notation). Morgenstierne's Kati, as Turner cites it,
  writes every non-first-syllable a as ə, including the -aka extension's final vowel: andʰakāra >
  andərə, ā́sa > asə, kʰarvá > kavə, hári > zər̆ə. First-syllable a stays a (60 a vs 13 ə). ext
  −0.0064/−0.0017, verb −0.0039/−0.0153, ext_acc −0.0051/−0.0018. Restricting it to medial syllables
  gives only −0.0023/−0.0004.
- **Wg `%source CDIAL`: `{ũ ẽ õ ĩ} > {ū̃ ē̃ ō̃ ī̃}`** (@notation, installed with the Wg a+NC rule in
  R7.5). Morgenstierne/Turner write Waigali nasal vowels long (ū̃ 32 vs ũ 8, ē̃ 19 vs ẽ 5); Strand
  writes them short (ũ 28, ẽ 24). Alone it is neutral, because the chain made almost no nasal vowels.

**Rejected / not regular**
- Pr Morgenstierne ɨ for u: 17 ɨ vs 58 u in the first syllable, with no onset/coda/position split
  (final u is 55 u : 1 ɨ). Pr CDIAL ə: scattered over pred u, a, ī, with no position that wins.
- Wg Strand ɨ vs u (the existing `u > ɨ` notation): ɨ wins in every context (52 vs 18); the u
  cases cluster in starred expressive words (*kʰuṭṭa, *bukka, *muṭṭa), which cannot be conditioned.
  No ɨ-harmony either (ɨCa: 14 a vs 3 ɨ).
- Ash CDIAL final -æ/-ə: `a > æ / V C* _ #` gives −0.0035 train but test +0.0026; `a > ə` gives
  −0.0011 / +0.0008. Morgenstierne writes the final reduced vowel as æ, ə or a in the same position,
  so it is source-internal variation.
- Wg `a a > æ / _ #` (-aka > -æ, CDIAL uruŋæ, tanǽ): +0.0024/+0.0074, because Strand's -a needs
  the same variant. `a a > ā / _ #` (CDIAL verb citations in -ā): +0.0078/+0.0075.
- Kt CDIAL ɔ for o: 7 ɔ vs 7 o in monosyllables; not tried.
- Kam `r̩ > a` (the real source of most i→a gains below): train −0.0010 but test +0.0043; first
  syllable only −0.0005/+0.0048.

### R7.3 Thematic verb citations and apocope in PNur and Wg (task b)

The round-6 flag was that no citation item ending in -a survives full apocope. I re-examined it
with `vdiff.py`, which gives each verb's chosen variant before and after.
- **The item already exists: `-ati` survives apocope.** At apocope time the word ends in -a-i (the t
  is lost earlier), so the final -i protects the -a. Hiatus removes the i later.
- **What apocope breaks are Strand's thematic stems with a kept root consonant.** Before apocope,
  the best variant for *rava, *iava, *jiva, *ciava and *vaʦava was -itavya. The menu suffix's -va-
  stood in for a stem v that the chain loses between vowels. Strand's PNur verbs keep that v in 5
  cases (ravatē, yábʰati, jī́vati, cyávatē, *vikṣuvati) and lose it in 4 (bʰávati *bo, bʰāvayati
  *ba, píbati *pia, yuvátē *io), so v-retention is lexical.
- **After the 10:35 verb-flag fix, one real derivation also breaks:** sárati > Strand *salika /
  *sirika (Wg sirika, test). Its root + -ikka variant loses its final -a.
- **Placebos** (PNur, apocope file): `%verb-menu +aka` 0/0; the placebo `+ara` −0.0043 verb train
  (itself a chance match). No per-lect item was added.

**Tried and withdrawn:**
- Full apocope with a waiver, installed at 09:00. After the verb-flag fix, Wg verb test was +0.0041,
  failing the waiver's ≤ 0.001 condition, and PNur had a real derivation among its losers
  (sárati). Reverted at 11:20.
- Apocope except after a cluster, `{a ă} > ∅ / C _ # // C C _`. It passes every gate (Wg ext
  −0.0054/−0.0058, 145 better / 0 worse; PNur −0.0165/−0.0154, 72 / 0) because it keeps the -ikka
  and -(i)tavya citations. But nouns do not show the exception: vowel-final reflexes are 46% after
  an OIA cluster vs 59% after a single consonant in Wg, and 55% vs 60% in Strand's PNur (útsa > ɨʦ,
  ṓṣṭʰa > ɨ̄ṣṭ, pakvá > *pak, púṣya > *puś, bʰaktá > *bat). Labelled **verb citation -a, not
  phonology** and ruled out by the coordinator as a citation shortcut in disguise.

**Installed (coordinator decision, as Kt/Kam):** `use final-vowel-loss.aa.c` after
`final-vowel-loss.a.c.3`. The numbers below are from 11:30; R7.7 has the 11:40 rescoring. Label:
**"verb gate waived: Strand's verb citations keep a thematic -a
(source convention), which the chain's apocope removes"**. Deltas are against the round-7 start file.
- **PNur**: ext −0.0270/−0.0254; ext_acc 0/0; ext-exact 18.4→24.5% / 17.7→22.1%; 137 reflexes
  better, 14 worse. Plain +0.0094/+0.0027 (bare-vs-extended); verb −0.0005/+0.0231.
  - Against the restricted rule: ext −0.0105/−0.0100.
  - Verbs worse (Strand; old → new variant, NED):
    - itavya chance matches (the suffix's -va- standing in for a stem v): ravatē *rava 0 → 0.2;
      yábʰati *iava 0 → 0.2; cyávatē *ciava 0.2 → 0.4; *vikṣuvati *vaʦava 0.5 → 0.571;
      jī́vati *jiva (test) 0 → 0.2;
    - menu-suffix chance matches, no stem v: sájati *saʦ̣ia 0.333 → 0.4;
      *caŋkr̩tati *ʦ̣ākara (tavya) 0.667 → 0.7; r̩dʰáti *arā (tavya) 0.833 → 0.857;
    - real -ikka derivation: sárati *salika, *sirika (test) 0.167 → 0.333.
  - Verbs better: mostly Strand's *-āv causatives (prāpayati *prāv 0.2 → 0, *pusati *pusāv,
    miśrayati *miṣāv, citráyati *citrāv, kárṣati *kiṣāv, dʰāpayati *samdāv, *kʰuddati *kɨdāv), plus
    vḗnati *vẽta, niṣṭʰīvati, *pratilati.
- **Wg**: ext −0.0096/−0.0080; ext_acc −0.0008/−0.0059; 280 better, 19 worse. Plain +0.0050/+0.0035;
  verb +0.0010/+0.0041.
  - Against the restricted rule: ext −0.0029/−0.0019, but ext_acc +0.0003/−0.0036 (the coordinator
    ruled that the start file is the baseline).
  - Verbs worse: cyávatē cɨva (itavya 0.25 → itavyakam 0.4), pāráyati pari-sta (0.286 → 0.429),
    yábʰati iava (0 → 0.2), sárati sirika (test, ikka 0.167 → 0.333). Better: grantʰáyati grū̃ṭ.

**Open item (again):** a Strand thematic-citation menu item that survives apocope. It would need
the root's final consonant kept (rav-a, iav-a) and a stem vowel that is not word-final at apocope
time. No OIA formation in the menu does that. Adding one needs a placebo test against a same-shape
item whose consonant survives.

### R7.4 Pr a > u (task c)

Removing `a > u // _ #` again: 86 reflexes better, 278 worse. The words that keep a are ordinary
vocabulary: dánta ʔatam, padá val, vána ban, saptá sat, páttra vaṭ (but śaptá śüt, bʰaktá but,
páñca vuc). Per source (a-better / u-better): CDIAL 30/161, Morgenstierne 39/91, Buddruss 11/13,
Strand 6/11. No accent, onset, coda or source split is available; this is lexical or dialectal
(Prasun villages). No change.

### R7.5 Near misses and consistency (task d)

**Kept**
- **Kam: non-initial ā reduction moved after the last-syllable ā > o, with a final-ā guard.**
  - Found by the ablation: removing `ā > a / V C _ | V C C _` improved ext.
  - Cause: it ran before `nur-innovation.a.cc` and so bled it. With the order fixed, the ā of the
    last syllable becomes o: áŋgāra > āŋó, aṭṭāla > āṭól, āstāra > śtór, utkrāma > ugr̆óm,
    prahāra > pr̆ór, yamarāja > imró.
  - Non-final ā still reduces to a: niḣsārayati > nusára, *karttāra > kātrá.
  - New line `ā > a / V C* _ #` before a.cc: a word-final non-initial ā stays a (utpātaka > upá,
    *ut-stʰāti > uśtá, *ut-śrāyati > uaṣa).
  - ext −0.0017/−0.0010, ext_acc −0.0023/−0.0010, verb 0/0 (after the verb-flag fix); 17 better,
    3 worse (manā́k, dantamāṁsa). The reorder alone is −0.0006/−0.0010.
- **Kt: the same final-ā line** (identical local rule, for consistency). −0.0003/0, ext_acc
  −0.0007/0. It rests on one etymon (utpātaka > upá better, *āṇḍajāta > āzo worse).
- **Pr `use h-loss.h.aav`** (as Kt/Wg/Gmb/PNur). It failed the verb gate in round 6 and passes after
  the verb-flag fix. prahāra > pār, kalaha > kala, mahallaka > melik, *nijahāti > nza. ext −0.0010/0,
  verb −0.0018/0; 10 better, 2 worse (bāhú > vaz, ahám > unʣū).
- **Wg: a before a nasal cluster > ũ in a post-apocope monosyllable**, using Ash's marker method:
  - `{a ā} > å / C _ {n ñ ṁ} C` runs before the nasal-cluster rules;
  - å contracts like a;
  - late `å > ũ / # C* _ C* #`, and å > a elsewhere.
  - This is Waigali's counterpart of Ash å > o and of Kt/Kam monosyllable a > u.
  - Monosyllables: dánta > dũt/dū̃t, páñca > pũc, pántʰā > pũt, kantʰā > kū̃t, *vañja > mũʦ.
    Polysyllables keep a (antár > ater, páñcadaśa > pacīś, kāntā > kā̃ta).
  - With the CDIAL long-nasal notation (R7.2): ext −0.0012/−0.0009 with full apocope, −0.0014/0
    without. Writing ū̃ in the chain is weaker (−0.0005/0), and õ is worse (+0.0004).

- **PNur and Kam `a ā > ā`** (before `hiatus.aai.postvocalic`). When a consonant is lost between
  a and ā, the pair contracts to long ā instead of losing the ā.
  - PNur: prahāra > *prār, *pragāmaka > *prāma, *hayanavāra > *ʣẽvār, *savāra > *suvār.
    ext −0.0006/−0.0020; 5 reflexes better, 1 worse (andʰakāra).
  - Kam: the long ā then takes the last-syllable ā > o (*hayanavāra > ʣā̃vór, kṣapā́ > ṣóã).
    ext −0.0007/0, ext_acc −0.0009/0; 5 better, 0 worse.
  - Kt fails on test (+0.0003); Pr (ext_acc +0.0011) and Wg (0 / +0.0009 ext_acc) are not better.

**Kept, then reverted**
- ~~Kam `i > a / # C* _ C V #`~~ (Kt's round-6 rule, for group consistency) — installed, then **reverted** after the 10:35 verb-flag fix (verb +0.0022/+0.0028 from newly flagged verbs: lihati > liʣá, *vināti > viná, vílapati > vilā, stʰīyátē > tia). The numbers below are from before the fix.
  - First-syllable i of a post-apocope disyllable ending in a vowel > ə (Strand â/a): tŕ̩ṇa > tār̆̃í,
    bʰr̩ta > bār̆á, śītalá > śālá, hŕ̩daya > ʣārá.
  - ext −0.0009/−0.0008, ext_acc −0.0019/−0.0019, plain +0.0012/+0.0016.
  - The reflex balance is poor (34 better, 42 worse; mostly i < r̩ gains vs plain i losses: nī́la >
    nilá, lihati > liʣá, *likka > liká). It passes on NED mass, not on counts.
  - `# C C* _ C V` (any first syllable) is +0.0038.

**Rejected**
- Syncope made uniform across the group:
  - Kt `{u i} > ∅ / V C _ C V` (Kam's version): ext −0.0004/−0.0004, but verb +0.0016.
  - Wg u- or {u i}-syncope: ext_acc +0.0024 / +0.0022.
  - Ash u- or {u i}-syncope: ext +0.0002 / +0.0001.
- Wg ND > N, re-run after apocope:
  - {d ḍ b g}: ext −0.0021/−0.0037, but ext_acc train +0.0030 and verb test +0.0014.
  - {d ḍ} only: ext_acc +0.0025.
  - The accent-marked Wg forms are the ones that keep the stop (gúnd, andá).
- Wg: dropping the CDIAL/Buddruss final-a tail now that the chain apocopates: verb +0.0153/+0.0110.
- PNur: dropping `nur-innovation.v.vcnn` (final-nasal vowel nasalisation, added this morning for
  consistency with Ash/Wg) is −0.0002/0. That is 2 reflexes of one lexeme (dʰamáni/dʰamana *dama)
  against *karã, *śiŋdrõ, so it is kept for consistency.
- Medial `a a > ā / _ C` (the same contraction for a + a): Kt −0.0021/−0.0033 and Kam
  −0.0032/−0.0042 pass the gates, PNur fails (verb test +0.0025). **Not installed**: 47 of Kt's 71
  and 52 of Kam's 68 improved reflexes come from the +akam variant. There -akam > -ām > -om matches
  unrelated o-endings (Kt kará > karov, *kʰuṭṭa > kuṭov; Kam *ṭappa > ṭāpóã, pūra > púr mos). With
  +akam excluded, Kt is about even (3.6 vs 2.9 NED mass). The real cases (cáya > có, *lavaṇḍa >
  lāváṇ) are too few to separate from the extension artifact.
- Round-6 rejections re-run after the apocope and verb-flag changes, all still rejected:
  - h-loss in Kam (−0.0012 train, but test +0.0001) and Ash (−0.0014 / +0.0001). The single
    test-side loser is sahásra > azār.
  - Initial s-loss before an aspirate in Kt (test +0.0007) and Kam (test +0.0012, verb +0.0083).
  - r-loss before a labial in Kam (+0.0003 train).
  - ar-lengthening in Wg (+0.0001/+0.0019, ext_acc +0.0070) and Ash (0).
  - ṣṭ > št in Wg (test +0.0009); ā-hiatus in Wg (train 0).
  - Final-nasal vowel nasalisation in Pr (+0.0007, verb +0.0044) and Kam (0).
  - Kt/Kam apocope with the cluster exception (+0.017 / +0.015): Kt and Kam lose -a after clusters
    too (gōṣṭʰa > guṣṭ).
- Other near-miss clusters checked and found lexical:
  - Kam/Kt l > r̆ after u (kúla kɨr̆, kʰōla kur̆ vs mála mul, pʰála pul);
  - Kam final -o (5 ā-stem words vs 41 consonant-final);
  - Kt final-syllable o > u (strand o 32 : u 9);
  - Ash first-syllable ā > o;
  - PNur a > ā before r;
  - Wg monosyllable a > o/u (a 71 : o 26 : u 18).

**Leave-one-rule-out ablation** (`ablate.py`, Δ ext train/test, ext_acc, verb for each chain rule
removed):
- Wg, Kt, Ash, Pr, Gmb: every chain rule pays for itself on ext train, except the redundant
  ă/ĭ markers (zero effect since apocope). No removal improves ext train.
  - Small test-only effects: Kt affrication.j.aa −0.0005, Pr hiatus.a.prevocalic −0.0020.
  - These were not acted on, because train decides.
- PNur: only `nur-innovation.v.vcnn` (−0.0002, above).
- Kam: `ā > a / V C _ | V C C _` (−0.0003/−0.0016 when removed; verb +0.0028). This led to the
  reorder kept above.
- Observation (not actionable): CDIAL writes a final high vowel long when it is stressed and short
  after an earlier stress. Counts (long : short) under final stress / under earlier stress:
  Pr 52:17 / 20:36, Wg 41:6 / 8:36, Ash 53:6 / 11:22. The `{i u} > {ī ū} / _ #` CDIAL notation
  could follow the stress if the chains predicted non-final stress. They place it finally, so no
  split is possible yet.

### R7.6 Legacy @morph rules (focus item 6)

Each was removed alone, then jointly where a lect has two, and re-run after the 10:35 verb-flag
fix. Δ is candidate − installed (train / test).
- **Removed (neutral removal)**:
  - PNur `a t {i e} > a / _ #`: ext, verb, ext_acc 0/0; plain −0.0042/−0.0082 with apocope
    (+0.0015/+0.0017 without).
  - Pr `if:verb a t {i e} > a / _ #`: ext, verb, ext_acc 0/0 (verbs never see @morph in ext/verb
    scoring); plain +0.0024/+0.0010.
  - Kt `a t {i e} > e / _ #`: after the verb-flag fix every metric is 0/0; Strand's -e comes from
    -ati through `ai-au.aai.final`.
  - Wg `a t {i e} > a / _ #`: after the fix, ext 0/−0.0004, verb and ext_acc 0; plain
    +0.0017/−0.0027 with apocope (+0.0040/+0.0008 without).
- **Kept** (the gate that fails when it is removed):
  - Kam `a t {i e} > a / _ #`: ext +0.0002/+0.0025, ext_acc test +0.0033. Plain would gain
    −0.0155/−0.0246, but ext test is the veto.
  - Ash `a y a t {i e} > i e / _ #`: ext train +0.0004.
  - Ash `a t {i e} > i e / _ #`: ext +0.0007/+0.0009; both together +0.0013/+0.0009.
  - Pr `a y a t {i e} > a / _ #`: ext train +0.0005.
  - These serve the noun-flagged headwords in -ati/-ayati that the sources cite with the verb ending.


### R7.7 Scores, round-7 start → end

Both columns are scored with the engine as of 11:40 (all fixes in R7.1 plus the verb-flag, canon
and @tag fixes), from `gates.py r7nur/orig/OIA-X.sc rules/OIA-X.sc`, train / test. The start files
are the round-6 end states.

| lect | ext NED | ext-exact | ext_acc | verb-menu | plain | verdict |
|---|---|---|---|---|---|---|
| Kt | 0.4097→**0.4030** / 0.4259→**0.4243** | 14.9→15.4% / 13.3→13.3% | 0.4104→0.3963 / 0.4209→0.4136 | 0.4123→0.4079 / 0.4378→0.4293 | 0.5021→0.5011 / 0.5143→0.5144 | ACCEPT |
| Kam | 0.3701→**0.3633** / 0.3843→**0.3788** | 18.7→19.3% / 17.5→18.0% | 0.4030→0.3920 / 0.4269→0.4207 | 0.3701→0.3671 / = | 0.5086→0.5062 / 0.5136→0.5153 | ACCEPT |
| Wg | 0.4771→**0.4674** / 0.4953→**0.4873** | 8.3→9.7% / 6.1→7.2% | 0.5136→0.5129 / 0.5339→0.5285 | 0.3919→0.3929 / 0.4783→0.4819 | 0.5360→0.5410 / 0.5587→0.5622 | verb waived (apocope); plain bare-vs-ext + @morph |
| Ash | 0.4600 / 0.4617 (=) | 8.7% / 10.4% (=) | 0.4929→0.4902 / 0.5007→0.4974 | = | = | ACCEPT (the ext_acc gain is the %gold-drop declaration) |
| Pr | 0.5390→**0.5380** / 0.5476 (=) | 3.8% / 3.9% (=) | 0.5369→0.5366 / = | 0.5921→0.5915 / = | 0.6291→0.6313 / 0.6366→0.6385 | ACCEPT-IF-JUSTIFIED (plain: @morph neutral removal) |
| PNur | 0.3488→**0.3212** / 0.3667→**0.3392** | 18.4→24.7% / 17.7→22.1% | = | 0.3116→0.3111 / 0.2736→0.2967 | 0.4074→0.4170 / 0.4243→0.4222 | verb waived (apocope); plain bare-vs-ext |
| Gmb | 0.4523 / 0.4257 (=) | = | = | = | = | no change |

- Kam i > a, re-tested after the 11:40 fix, still fails: verb +0.0021/+0.0025.

### R7.8 What remains

- **Strand thematic verb citations** (open item, R7.3): a menu item that keeps the root's final
  consonant and a non-final stem vowel at apocope time.
- **Pr a > u** (R7.4) and Pr's many initial-consonant changes (p- > v-, initial loss): lexical or
  dialectal.
- **Wg ND > N** stays blocked by ext_acc: the accent-marked forms keep the stop.
- **Kam/Ash h-loss**: blocked by one test word (sahásra > azār).
- **Final-vowel endings** (-oa Kam verbs, -æ/-ə Ash and Wg CDIAL, Kt -e/-ɔ) are suffixal or
  source-internal, as in round 6.
- **Tone**: after the notation fixes no Nuristani source writes a tone type, so the Vedic-accent →
  tone-type test does not apply to this group.
- **CDIAL final-vowel length follows stress** (R7.5 observation). It needs non-final accent
  placement first.
