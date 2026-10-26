# mia-insular — OIA → MIA (Pk, Pa, As, Ap; Gāndhārī, Niya, KharI) and Insular (Si, OSi, Md)

Files: `sound_changes/rules/OIA-{Pk,Pa,As-Ul,Ap,Dhp,NiDoc,KharI,Si,OSi,Md}.sc`, frontiers in
`sound_changes/results/OIA-*.hand.frontier.csv`, scratch tools in `tmp/sound_changes/mia-insular/`.

## 1. Summary table

Train/test NED: null (no rules) → full chain. "Auto" = the greedy baseline's test NED at its
largest rule count, with my frontier's test NED at the same rule count in brackets. Rule counts
exclude `%source` notation-block rules.

| lect | #etyma | #rules | train NED | test NED | test exact | auto test NED (hand at same n) | sources excluded / per-source blocks |
|---|---|---|---|---|---|---|---|
| Pk (Prakrit) | 7932 | 46 | 0.447 → 0.204 | 0.460 → 0.206 | 38.9% | 0.279 @23 (0.226) | block: sheth1923 (homorganic nasals, ē ō) |
| Pa (Pali) | 5473 | 41 | 0.337 → 0.142 | 0.347 → 0.144 | 54.5% | 0.213 @16 (0.161) | — |
| Si (Sinhala) | 3816 | 45 | 0.634 → 0.373 | 0.647 → 0.394 | 12.6% | 0.487 @13 (0.454) | — |
| Dhp (Gāndhārī) | 1333 | 26 | 0.467 → 0.187 | 0.460 → 0.188 | 49.5% | 0.250 @15 (0.218) | blocks: gandhari (ṇ > n), CDIAL/Brough (no ṁ, intervocalic ṇ) |
| Md (Dhivehi) | 783 | 49 | 0.729 → 0.450 | 0.727 → 0.471 | 7.3% | 0.570 @19 (0.537) | block: fritz (ṣ for r̆) |
| NiDoc (Niya) | 588 | 26 | 0.497 → 0.274 | 0.480 → 0.282 | 24.0% | 0.335 @10 (0.300) | — |
| As (Ashokan) | 568 | 29 | 0.486 → 0.303 | 0.504 → 0.323 | 16.6% | 0.377 @9 (0.372) | block: andersen1990 (m for ṁ; kept, not excluded) |
| OSi (Old Sinhala) | 543 | 29 | 0.617 → 0.418 | 0.608 → 0.409 | 9.5% | 0.469 @9 (0.437) | block: paranavitana (ț ņ ļ) |
| Ap (Apabhraṁśa) | 157 | 29 | 0.605 → 0.448 | 0.560 → 0.381 | 0.0% | 0.463 @9 (0.404) | block: sheth1923 |
| KharI (Kharoṣṭhī inscr.) | 137 | 12 | 0.503 → 0.289 | 0.503 → 0.288 | 17.2% | 0.353 @12 (0.288) | — |
| MIA (unspecified) | 119 | — | skipped | | | 0.671 @6 | data unusable (see §5) |

## 2. Base chain (mia-insular) and deviations

The base chain is `OIA-Pk.sc`; every other file is a copy of it (MIA lects) or of `OIA-Si.sc`
(Insular), with shared rules kept in the same relative order. Order of the MIA block:

| # | rule (Pk form) | tags | deviations |
|---|---|---|---|
| 1 | `V > $1-acc` | @acc-loss | all |
| 1b | `∅ > i / # C _ l` | @anaptyxis | Pk Pa (klēśa > kilēsa) |
| 2 | `t > ṭ / # p r a _ i` | @x-mia-insular-prati | Pk Pa As Ap Si OSi; Dhp writes it `p r a t i > p a ṭ i` (r lost, paḍi-); absent NiDoc KharI Md (0 fires / ≤ 0) |
| 3 | `{ai au} > {ē ō}` | @ai-au | all |
| 4 | r̩: Pk `a / _ dental`, else `i`; Pa, As, KharI, OSi `a`; Si, Md `u` after labials else `a`; Dhp `u` after labials else `i`; NiDoc `u` after labials else `ri`; Si additionally `t > ṭ / r̩ _` | @r-vocalic (@r-vocalic-ri NiDoc) | lect-specific, see §4 |
| 5 | `∅ > a / Them _ #`, `∅ > a / C C _ #` | @morph | Pk Pa As Ap (bare roots / consonant stems thematised) |
| 6 | `m > ṁ / _ #`, `repeat: C(x) > ∅ / _ #` | @final-cons-loss | all; Pk Pa As Ap keep final ṁ (class Cx excludes m) |
| 7 | `i ṁ > ī / _ Sib` | @anusvara-vowel | Pk (Md has it late) |
| 8 | `Sib > s` | @sibilant-s | all except Dhp NiDoc KharI (merger costs +0.014…+0.022 there) |
| 9 | `{t tʰ d dʰ} y > c c / j j…` | @palatal-Cy | all |
| 10 | `r y > j j` (Pa, Dhp: `∅ > i / r _ y` first) | @palatal-Cy @assim-r / @anaptyxis | Pa Dhp have anaptyxis riy; rule ≤ 0 (dropped) in Si OSi Md As Ap |
| 11 | `{k t p} s > kkʰ ccʰ ccʰ` | @ks | not Dhp NiDoc KharI (kṣ kept); NiDoc late kṣ > c̄h |
| 12 | `s Stop > $2-asp $2+asp` | @s-stop-aspirate | Dhp NiDoc KharI: only `ṣ Stop` (+ Dhp `s tʰ > ṭṭʰ`); s + stop kept |
| 13 | `{s h} Nasal > $2 h` | @s-nasal | Pk Pa Ap (≤ 0 in Si OSi Md As) |
| 14 | `s Son > s s`, `j ñ > ṇṇ/ññ`, `h y > jjʰ/yh`, `v y > v v`, `Nasal Nasal`, `n y > ññ`, `Nasal {y v}` | @assim-sonorant … | Pa: ññ, yh metathesis; Pk: ṇṇ, jjʰ |
| 15 | `Nasal Stop > ṁ $2` | @nasal-cluster | Pk Ap Dhp NiDoc KharI As (As late); Pali keeps the homorganic nasal; Insular has no ṁ stage |
| 16 | `r {t tʰ d dʰ} > ṭṭ ḍḍ…` | @x-mia-insular-r-retroflex | As Si Md (Pa only rdʰ > ḍḍʰ; Pk: hurts) |
| 17 | `{r ḣ} C > $2-asp $2`, `l Stop > …` | @assim-r | not NiDoc KharI (r + C kept in Kharoṣṭhī documents); KharI only `r Sib > Sib` |
| 18 | `Stop {r y v l Nasal} > $1 $1`, `Stop Stop > $2 $2`, `l {y v} > l l` | @assim-sonorant @assim-stop | Dhp: no stop + sonorant assimilation at all (≤ 0; only `y > ∅ / C _` late); NiDoc: only Stop + Nasal; KharI: Stop + y v l Nasal, r kept |
| 19 | `C {r y v l} > $1 / # _`, `C $1 > $2 / C _`, `C C > $2 / # _` | @initial-degem @cluster-simplify | Si OSi drop the last (general degemination covers it) |
| 19b | `s s > s / {ī ū ē ō ā} _` | @degem-lengthen | Pk only (īśvara > īsara; Pa 0) |
| 20 | `{ā ī ū ē ō} > $1-long / _ C C` | @two-mora | Pk Pa As Ap; masked (LOO 0) in Si OSi Md by the later general shortening, so dropped there; not visible in Kharoṣṭhī spelling |
| 21 | `a y a > ē / _ t {i ē}` | @aya-e | Pk Pa Ap Dhp NiDoc (verbs only); **As lacks it** |
| 22 | `a v a > ō / # _` | @aya-e | Pk Pa Ap Dhp Si Md OSi (in the Insular files at the head of the Insular block; its position is free: ΔNED ≤ 0.0001) |
| 23 | `{kʰ gʰ tʰ dʰ pʰ bʰ} > h / V _ V` | @int-aspirate-h | Pk; NiDoc only kʰ gʰ (added after the matrix below was measured); Ap, Pa, As, Dhp, KharI: hurts |
| 24 | `{ṭ ṭʰ} > {ḍ ḍʰ} / V _ V` | @int-retroflex | Pk Ap; Pa `{ḍ ḍʰ} > {ḷ ḷʰ}`; Insular `{ṭ ḍ} > ḷ` |
| 25 | `p > v / V _ V` | @int-p-v | Pk Ap Dhp NiDoc Si OSi Md |
| 26 | ya-śruti + `{k g c j t d} > ∅ / V _ V` | @ya-sruti @int-stop-loss | Pk (Ap without ya-śruti); Si Md lose only `{k g c t y}` (d, j>d, b kept); Dhp NiDoc voice instead (@int-voicing) |
| 27 | `n > ṇ` | @n-retroflex | Pk Ap only |
| 28 | `y > j / # _` | @y-j | Pk Ap only |

Insular-specific early rules (inside the MIA block): Si `ā > a / _ #` (OIA final -ā shortened
early, −0.006), `ā > æ / # (C) _ C {i ī e ē}` umlaut, then `{i ī} > a / C _ #`; Md `a > u / _ C u #`.
Insular block (Si, then OSi and Md deviations): deaspiration of all series → **`d > ∅ / V _ V`
(OIA d, dʰ) before `c j > s d`** (so the new d from j survives: −0.0018 Si, −0.0010 OSi, −0.0015 Md)
→ (Md: `h > ∅ / # _` then `s > h / # _`, `y > d / # _`) → `p > v` → loss of k g c t y (not OSi) →
hiatus `a > ∅ / _ V` (Si) → `ṭ ḍ > ḷ` (Md: `ṭṭ > r̆` first) → **degemination without lengthening** →
prenasalisation (`N + voiced stop > ⁿD`, `N + voiceless > T`) → long-vowel shortening (OSi: also
final) → hiatus glides → umlaut → contractions → anusvāra loss → (Md: p > f, final -u after r,
apocope, late `s > h / V _ V`, `i ṁ > ī / _ s`, final `-ā > -i`, ṇ > n).

## 3. Relative-chronology evidence (swap or move one rule; ΔNED on train)

| lect | order kept | alternative | Δ train NED | reading |
|---|---|---|---|---|
| Si | lenition (p > v, stop loss) ≺ degemination | degemination first | +0.041 | geminates escaped lenition: classic counter-feeding; degemination is post-lenition |
| Si | `ṭ ḍ > ḷ` ≺ degemination | swapped | +0.0059 (OSi +0.0025) | ṭṭ stays ṭ, only single ṭ flaps |
| Si | deaspiration ≺ `c > s` | swapped | +0.0033 (Md +0.0036, OSi +0.0021) | cʰ must lose aspiration before the affricate becomes s |
| Si | deaspiration ≺ stop loss | loss first | +0.0032 | intervocalic aspirates were lost like plain stops (no Pk-style `> h`) |
| Si | `ā > æ` umlaut ≺ `-i > -a` | swapped | +0.0006; moving umlaut after it +0.0008 | umlaut conditioned by the original final -i |
| Md | `c > s` ≺ `s > h` | s > h first | +0.0027 (several orders tried) | CDIAL Md. c- > s- mostly stays s |
| Md | `h- > ∅` ≺ `s- > h-` | h-loss at end | +0.0042 vs −0.0014 | OIA h- lost (hásta > at) before the new h- from s- |
| Si Md OSi | `d > ∅ / V _ V` ≺ `j > d` | d in the later loss set | −0.0018 (Si) | j > d is younger than the loss of OIA intervocalic d |
| Md | degemination, apocope ≺ `s > h / V _ V` | s > h early | +0.0038 | degeminated ss also > h (áśva > aha), and word-final s after apocope stays (divasa > duvas) |
| Md | `ṭṭ > r̆` ≺ degemination | swapped | +0.0055 | |
| Md | `-a > -u / V r _ #` ≺ apocope | swapped | +0.0075 | |
| Dhp | voicing ≺ (orthographic) degemination | degemination first | +0.021 | Gāndhārī geminates did not voice → the no-geminate spelling is not a sound change before lenition |
| Pk | `a y a > ē` ≺ loss of y | y-loss first | +0.0040 | contraction precedes y-loss |
| Pk | ya-śruti ≺ stop loss | swapped | +0.0043 | (definitional: ya-śruti is the stop-loss outcome between a-vowels) |
| Pk | stop loss ≺ `-atē > -ai` | swapped | +0.0036 | |

Presence/absence matrix: ΔNED (train) when each Pk-type change is appended at the *end* of each
final chain. A clear + means the lect does **not** have the change (the attested forms still
show the unchanged segment); 0 means the change is already in the chain (or its input is gone),
e.g. Pk for its own lenitions, Insular for aspirate > h (deaspiration came first). Dhp n > ṇ is
≈0 only because the gandhari notation block writes n for ṇ.

| appended change | Pk | Pa | As-Ul | Ap | Dhp | NiDoc | KharI | Si | OSi | Md |
|---|---|---|---|---|---|---|---|---|---|---|
| `n > ṇ` | 0 | +0.0279 | +0.0188 | 0 | +0.0005 | +0.0254 | +0.0057 | +0.0331 | +0.0278 | +0.0557 |
| `{kʰ gʰ tʰ dʰ pʰ bʰ} > h / V _ V` | 0 | +0.0079 | +0.0149 | +0.0033 | +0.0057 | +0.0029 | +0.0054 | 0 | 0 | 0 |
| `{ṭ ṭʰ} > {ḍ ḍʰ} / V _ V` | 0 | +0.0049 | +0.0043 | 0 | +0.0032 | +0.0053 | +0.0075 | +0.0068 | +0.0016 | 0 |
| `p > v / V _ V` | 0 | +0.0095 | +0.0101 | 0 | +0.0008 | +0.0008 | +0.0037 | +0.0042 | +0.0034 | 0 |
| `{k g c j t d} > ∅ / V _ V` | 0 | +0.0578 | +0.0586 | 0 | +0.0610 | +0.0380 | +0.0366 | +0.0318 | +0.0297 | +0.0164 |
| `{k t} > {g d} / V _ V` | 0 | +0.0456 | +0.0425 | 0 | +0.0078 | +0.0187 | +0.0185 | +0.0222 | +0.0286 | +0.0136 |
| `y > j / # _` | 0 | +0.0027 | +0.0083 | 0 | +0.0051 | +0.0040 | +0.0036 | +0.0020 | +0.0020 | 0 |
| `{ś ṣ} > s` | 0 | 0 | 0 | 0 | +0.0193 | +0.0212 | +0.0139 | 0 | 0 | 0 |
| `{ā ī ū ē ō} > $1-long / _ C C` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `a y a > ē / _ {t d} {i ē}` | 0 | 0 | +0.0022 | 0 | +0.0005 | 0 | +0.0015 | 0 | +0.0023 | 0 |
| `r > l` | +0.0260 | +0.0276 | +0.0011 | +0.0144 | +0.0414 | +0.0449 | +0.0412 | +0.0274 | +0.0206 | +0.0336 |
| `{kʰ gʰ cʰ jʰ ṭʰ ḍʰ tʰ dʰ pʰ bʰ} > {k g c j ṭ ḍ t d p b}` | +0.0369 | +0.0407 | +0.0326 | +0.0184 | +0.0217 | +0.0175 | +0.0306 | 0 | 0 | 0 |
| `C $1 > $2` | +0.0424 | +0.0446 | 0 | +0.0047 | 0 | 0 | 0 | 0 | 0 | 0 |
| `s > h / V _ V` | +0.0135 | +0.0137 | +0.0142 | +0.0033 | +0.0086 | +0.0014 | +0.0029 | +0.0116 | +0.0161 | +0.0019 |
| `v > b / # _` | +0.0123 | +0.0125 | +0.0124 | +0.0174 | +0.0125 | +0.0122 | +0.0056 | +0.0122 | +0.0122 | +0.0118 |
| `{ṇ} > n` | +0.0313 | +0.0100 | 0 | +0.0149 | 0 | 0 | +0.0047 | +0.0025 | +0.0035 | 0 |

## 4. Shared vs lect-specific changes; sub-grouping

**Shared by every lect with usable data** (the MIA core): accent loss, ai au > ē ō, loss of
final consonants, palatalisation of dental + y, assimilation of stop + stop and stop + nasal,
anusvāra/homorganic nasal before stops, Kharoṣṭhī/Brāhmī-independent loss of the second member
of initial clusters. Everything else splits the group into four clear sets:

1. **Central/Western literary MIA — Pk, Ap** (Pk ⊃ Ap): the full lenition package (aspirate > h,
   ṭ > ḍ, p > v, loss of k g c j t d with ya-śruti), n > ṇ everywhere, y- > j-, aya > ē in
   verbs. Ap is too small (157 etyma) to show anything Pk lacks; ya-śruti and r̩ > a before
   dentals hurt it (Apabhraṁśa sources write the hiatus unglided: piāsa).
2. **Pali, Ashokan** — the conservative "MIA core" only: every Pk lenition *hurts* when appended
   (table in §3). Pali-specific: r̩ > a (not i), jñ/ny/ṇy > ññ, hy > yh, rdʰ > ḍḍʰ (but rt > tt),
   intervocalic ḍ > ḷ, vv > bb, homorganic nasals. Ashokan (Brāhmī spelling: no geminates, ē ō
   written e o, ī ū mostly short) has ṇ > n (eastern), jñ > ññ and rt > ṭṭ; r > l is only partial (r 143 :
   l 87 train pairs) and was dropped once rt > ṭṭ was in; and — unlike Pali, Pk and even Gāndhārī — **no aya > ē** (+0.0025 when added: pūjayati,
   dasayitu).
3. **North-west (Kharoṣṭhī) — Dhp, NiDoc, KharI**: three sibilants kept (merger +0.014…0.022),
   kṣ kept (NiDoc: special c̄h intervocalically), s + stop kept but ṣ + stop > ṭʰ, sv > sp (Dhp,
   NiDoc; the same process as the dardic-north group's @x-dardic-north-sv-sp), no n > ṇ.
   Lenition = **voicing/spirantisation without loss**: Dhp t k ṭ > d g ḍ, tʰ dʰ > s (θ ð), c j > y,
   p > v; NiDoc only k ṭ > g ḍ (t stays: -ti vs Dhp -di), written ǵ ḍ', plus kʰ gʰ > h (likʰati >
   lihati; the other aspirates stay); KharI (inscriptions) none at
   all. r + C: assimilated in Dhp, kept in NiDoc and KharI (KharI only r + sibilant > sibilant).
   KharI alone simplifies *initial* s + stop to an aspirate (stūpa > tʰuba; −0.0046), which hurts
   Dhp and NiDoc (+0.003).
   Stop + sonorant: Dhp spells the clusters unassimilated or simplified (no assimilation rule
   survives; C + y > C late), NiDoc assimilates only stop + nasal, KharI stop + y v l nasal.
   So Dhp ⊃ NiDoc ⊃ KharI in lenition, matching literary > documentary > monumental registers.
4. **Insular — Si, OSi, Md**: deaspiration of all aspirates (instead of Pk's aspirate > h),
   c > s, j > d, degemination without compensatory lengthening, prenasalised voiced stops, loss
   of the nasal before voiceless stops, general shortening of long vowels, ṭ ḍ > ḷ, umlaut.
   OSi lacks the loss of intervocalic stops (orthographic conservatism of the Sigiri graffiti /
   inscriptions — or the loss is post-8th c.); Si loses only the voiceless stops, g and y (d,
   j > d, b stay); Md goes furthest (s > h, y- > d-, p > f, apocope, ṭṭ > r̆). Md and Si share
   ava- > o-, c > s, j > d, loss of OIA intervocalic d before j > d, prenasalisation; Md alone has
   s > h, h- loss and y- > d- in the chain.

Sub-grouping suggested: {Pk, Ap} vs {Pa, As} vs {Dhp ⊃ NiDoc ⊃ KharI} vs {Si ≈ OSi, Md}. The
Insular lects start from a *Pali/Ashokan-like* stage — no n > ṇ (+0.031…0.045 when added), no
aspirate > h, no ya-śruti — and then innovate on their own (lenition to v/y/∅ is independent of
Pk's: Si keeps d and b, which Pk loses).

### Relative chronology between MIA stages (what Pali lacks that Prakrit has)
Treating the presence matrix as stages: Ashokan and Pali stop at the cluster assimilations and
the law of two morae; Pali adds aya > ē (in verbs) and ḍ > ḷ; Gāndhārī adds intervocalic voicing
(not loss) and keeps the sibilants; Prakrit adds aspirate > h, ṭ > ḍ, p > v, total loss of
simple intervocalic stops with ya-śruti, n > ṇ and y- > j-; Apabhraṁśa (small data) continues
Pk. Within Pk, the chain requires aya > ē *before* the loss of intervocalic y (+0.0040 when
swapped), i.e. contraction predates the lenition layer. In Insular, lenition precedes
degemination (+0.041 when swapped) and deaspiration precedes c > s (+0.0033/+0.0036/+0.0021 in
Si/Md/OSi): the Insular deaspiration is early, before the affricate became s, while Pk's
aspirate > h is a different, later-looking change that Insular never had.

## 5. Other findings, data problems
- **Sinhala s**: CDIAL Si. keeps s in 276 of ~490 train pairs with initial s/ś/ṣ (h 125, zero
  ~90). s > h (initial or medial) *hurts* Si and OSi and is kept only in Md. Sinhala's s > h is
  well known; the CDIAL Si. data is dominated by learned/restored forms (tatsama stratum) and
  lists s- and h- variants side by side (sobinavā / hobinavā). Same for c > s (71) vs c > h (25).
- **aya > ē is morphological**: restricting it to -ayati/-ayatē verbs beats the general rule in
  Pk (−0.0025), Pa (−0.0045) and Dhp (−0.0095): nouns keep -aya- (ālaya, kṣaya, naya).
- **prati- > paṭi-** (Pk paḍi-, Si piḷi-, Dhp paḍi- with r lost) is a morpheme-specific
  retroflexion, systematic across Pk, Pa, As, Si and Gāndhārī (tagged @x-mia-insular-prati).
- Intervocalic inherited y: Pk (CDIAL) keeps it between a-vowels (valaya, ālaya): −0.0011.
- Pk only: a long vowel before ss keeps its length and ss > s (īśvara > īsara, śīrṣa > sīsa;
  −0.0005; 0 in Pali, which has issara) — the Pk/Pali split of the two-mora outcomes.
- Initial C + l anaptyxis (klēśa > kilēsa) in Pali and Prakrit; ry > riy in Pali and Gāndhārī.
- Si: OIA final -ā shortened early (latā > la-, sēnā > sena; −0.006), before -aka > -aa > -ā
  creates new long finals; verbs of all classes (-ati/-āti/-ōti) cited as -navā / Md -anī.
- Gāndhārī retains st/sp (vistiṁna, spa) but assimilates ṣṭ > ṭʰ and stʰ > ṭʰ.
- Old Sinhala: r̩ > a, no stop loss, all vowels short (script), ṁ/prenasalisation unwritten.
- Si final -i > -a (ṣaṣṭi > saṭa, giri > gira, bʰīti > biya) must follow the ā > æ umlaut.
- Md: intervocalic s > h is *late* (after degemination and apocope): áśva > aha but divasa >
  duvas; geminate ṭṭ > r̆ (ṣ in Fritz) while single ṭ > ḷ.
- **Sources**: Pk sheth1923 writes homorganic nasals and ē ō before geminates (a %source block,
  −0.005 train NED); Dhp gandhari writes n for ṇ, Brough (CDIAL Dhp.) omits anusvāra and writes
  intervocalic ṇ (%source blocks); As andersen1990 writes m for ṁ and cites verbal roots (car, pap,
  cy) — kept with a %source block (its NED 0.31 ≈ CDIAL's); OSi paranavitana writes ț ņ ļ (OCR
  variants); Md fritz writes ṣ for r̆.
- **MIA (unspecified, 119 etyma) skipped**: 39/224 reflexes are ≤3-segment fragments (ṭ, ḍḍ, ā,
  all), 40 are OIA forms (dr̩śyátē, durhr̥d), many pairs are unrelated look-ups (jā́yatē → lippai,
  laggai; karṓti → dinna). No chain file written.
- **Gāndhārī compounds**: the gandhari dictionary links compounds and names to the simplex etymon
  (vyādʰi → vadʰimaranaśoga…, pakṣa → saṁkileśapakṣiya, nánda → Budʰanaṁda); these dominate the
  worst misses and inflate Dhp NED. Similar suffix/compound noise (-aya, -iya, -ka) is most of the
  residual in Pk and Pa.
- Numerals (-catvāriṁśat > -yāla, -āḷīs) are a large irregular block in Pk and Md.

## 6. New tags
| tag | meaning |
|---|---|
| `@x-mia-insular-prati` | retroflexion of t in the prefix prati- (paṭi-/paḍi-/piḷi-) |
| `@x-mia-insular-r-retroflex` | dental > retroflex after r or r̩ (ártʰa > aṭṭʰa, kr̩ta > kaḷa); cf. @x-hindi-rt-retroflex, @x-nw-rt-cerebral |
| `@x-mia-insular-r-l` | r > l (eastern/Māgadhī Ashokan) — defined but finally not in any chain (As: net ≈ 0, dropped) |
| `@x-mia-insular-sv-sp` | sv, śv > sp, śp (Gāndhārī, Niya); same as @x-dardic-north-sv-sp |
| `@x-mia-insular-nd-nn` | nd, ndʰ > ṁn (Niya); cf. @x-wpah-nd-nn, @x-bihari-tharu-nd-nn |
| `@x-mia-insular-p-f` | p > f (Dhivehi) |
| `@x-mia-insular-final-u` | final vowel after r > u (Dhivehi) |
| `@x-mia-insular-final-i-a` | final -i > -a (Sinhala) |
| `@x-mia-insular-tt-r` | geminate ṭṭ > r̆ (Dhivehi) |
| `@x-mia-insular-anusvara-loss` | anusvāra lost before a consonant without lengthening (Si, OSi); cf. @x-cep-anusvara-loss |
| `@x-mia-insular-j-d` | j > d (Insular), co-tagged with @affrication for c > s |

Engine wishes: (1) `%source` blocks that can also *precede* parts of the chain (e.g. a
different anusvāra spelling that interacts with later rules); (2) a way to exempt starred
verbal roots / compounds (`*jʰal`, compound reflexes) without per-word rules; (3) an `apply`
option that prints the per-source tail. My scratch tools (tmp/sound_changes/mia-insular/an.py:
`chron`, `scan`, `move`, `sub`) did adjacent-swap and candidate-at-every-position scans.

## 7. Round 2: vowels and the Vedic accent

Scratch for this round: `tmp/sound_changes/mia-insular/r2/` (`t.py` = insert/replace/delete trials
with better/worse pair counts; `rtab.py` = r̩ outcome table by accent class and labial context;
`ftab.py` = final -a outcome by accent class; `var/` = Sinhala variants with the accent kept
through the umlaut and -i > -a rules; `probe_*.txt`, `score_*.txt`).

**Audit of the partial round-2 edits (06:39 snapshot).** Only Ap, Md and Si had been edited; all
their round-2 rules had positive train LOO and were kept: Ap final `{ā ī ū} > {a i u} / _ #`
(+0.0028), Md first-syllable `{a ā} > e / # C* _ C* i` (+0.0042), Si first-syllable
`ā > æ / # C* _ C* {i e}` (widened from `# (C) _`, +0.0019) and `a > æ / # C* _ C C i` (+0.0016),
and `if:!numeral` on the Si/Md verb-citation rules (numerals such as viṁśati no longer become
-navā/-anī). No `%gold-drop` is used in any mia-insular file. Two round-1 Md rules had fallen to
≤ 0 and were removed: `r̩ > u` after labials (−0.0003) and `a > i / # C _ {r ḷ l} i` (0 fires; the
new umlaut takes its input).

### 7.1 Scores (train / test NED)

"r1" = round-1 summary; "r1 file now" = the unchanged round-1 file under the current engine
(tokenisation fixes shift scores by up to 0.002); "r2" = final file.

| lect | #etyma | r1 train / test | r1 file now | r2 train / test | r2 test exact | #rules r1 → r2 |
|---|---|---|---|---|---|---|
| Pk | 7932 | 0.2036 / 0.2061 | 0.2030 / 0.2054 | **0.2021 / 0.2055** | 38.9% | 46 → 49 |
| Pa | 5473 | 0.1422 / 0.1443 | 0.1417 / 0.1436 | **0.1415 / 0.1435** | 54.7% | 41 → 42 |
| Si | 3816 | 0.3728 / 0.3945 | 0.3727 / 0.3940 | **0.3699 / 0.3901** | 12.3% | 45 → 47 |
| Dhp | 1333 | 0.1871 / 0.1883 | 0.1855 / 0.1883 | 0.1855 / 0.1883 (unchanged) | 49.5% | 26 |
| Md | 783 | 0.4496 / 0.4710 | 0.4495 / 0.4710 | **0.4378 / 0.4680** | 8.1% | 49 → 51 |
| NiDoc | 588 | 0.2743 / 0.2817 | 0.2737 / 0.2817 | **0.2717 / 0.2776** | 24.7% | 26 → 27 |
| As | 568 | 0.3032 / 0.3233 | 0.3023 / 0.3233 | 0.3023 / 0.3233 (unchanged) | 16.6% | 29 |
| OSi | 543 | 0.4175 / 0.4086 | 0.4174 / 0.4086 | **0.4154 / 0.4078** | 9.5% | 29 → 30 |
| Ap | 157 | 0.4477 / 0.3807 | 0.4477 / 0.3807 | **0.4369 / 0.3850** | 0.0% | 29 → 32 |
| KharI | 137 | 0.2890 / 0.2882 | 0.2869 / 0.2882 | 0.2869 / 0.2882 (unchanged) | 17.2% | 12 |

No lect's test NED rose by more than 0.005. Ap's test rose by 0.0043 (34 test pairs; the train
gain comes from final shortening and ī > i, ē > e). Frontiers were rerun for Pk, Pa, Si, Md,
NiDoc, OSi and Ap. Dhp, As and KharI files are unchanged from round 1. Pa's round-1
`∅ > i / r _ y` (viriya) now has LOO −0.0000 (5 better / 6 worse); it is kept as a known Pali
anaptyxis.

### 7.2 Adopted vowel/accent rules (train LOO; examples)

| lect | rule | tags | LOO | examples |
|---|---|---|---|---|
| Si | `r̩ > i / _ !#* Acc` (before @acc-loss) | @r-vocalic @stress | +0.0005 (13 better / 10 worse pairs) | pretonic r̩: mr̩ttikā́ > mæṭi, gr̩hī́ta > giya; tonic and starred r̩ stay > a |
| Si | `ā > æ / # C* _ C* {i ī e ē}`, `a > æ / # C* _ C C {i ī}` (from the partial run) | @umlaut | +0.0019, +0.0016 | ālapati > ælavum, ástʰi > æṭa |
| Md | `{u ú} > i / _ C* {i ī}` | @umlaut | +0.0012 | u fronted before i |
| Md | `{ī ī́} > e / _ C* #` | @vowel-shift | +0.0010 | āsandī́ > eⁿde |
| Md | `a > u / C _ r V` (after lenition, before ṭṭ > r̆) | @vowel-shift | +0.0025 (31 fires; 17 better / 6 worse) | karōti > kuranī, tarati > turenī |
| Md | `{a ā} > e / # C* _ C* {i ī}` (partial run) | @umlaut | +0.0042 | āmiṣa > em |
| Pa | `r̩ > u / _ C* {u ū}` | @r-vocalic @umlaut | +0.0002 (9 fires) | pr̩tʰu > putʰu, mr̩du > mudu |
| Ap | `{ā ī ū} > {a i u} / _ #` (partial run) | @vowel-shift | +0.0028 | Ap final shortening |
| Ap | `ī > i`, `ē > e` (end of chain) | @vowel-shift | +0.0056, +0.0016 | dīpa > dippa, adʰīna > adʰiṇṇa, aśītí > asii; CDIAL Ap. e 19 : ē 11 |
| Pk | `r̩ > ā / _ #` | @morph @r-vocalic | +0.0003 | r-stems in the nom. sg.: pitr̩ > piā |
| NiDoc | `r̩ > a / _ #` | @morph @r-vocalic | +0.0020 | pitr̩ > pita, mātr̩ > mata |
| Pk | `if:verb ō > a / _ t {i ē} #`, `ā > a / {n ṇ} _ t {i ē} #` | @morph | +0.0004, +0.0002 (14 / 2) | karōti > karai; jānāti > jāṇai, gr̩hṇāti > giṇhai |
| OSi | `{ē ḗ} > ∅ / t _ #` | @morph | +0.0021 | middle -tē verbs cited without the ending (avalambatē > olambu) |

Tried and dropped as fitting only a handful of items: Dhp `r̩ > u / _ C* u` (2 fires), NiDoc
`if:verb ō > a` (3 fires), Ap `ē > i / _ #` (1 fire), OSi `a > ∅ / _ V` (6 better / 1 worse).

### 7.3 Accent tests (Δ train NED when added; + = worse). All rejected unless marked.

- **Pretonic shortening** `{ā ī ū} > {a i u} / _ !#* Acc`: Pk +0.0088, Pa +0.0114, As +0.0049,
  Ap +0.0038, Si +0.0009, Dhp +0.0005, OSi 0. Post-tonic shortening: Pk +0.0022, Pa +0.0035.
  MIA has no accent-conditioned shortening; long vowels shorten only by the two-mora law.
- **Does the tonic resist the two-mora law?** (unaccented long vowels shortened before CC before
  @acc-loss, tonic ones left long): Pk +0.0024 (14 better / 116 worse), Pa +0.0024 (19 / 68). No.
- **Accent and consonant gemination** (Pischel's nīḍá > neḍḍa, tailá > tella type:
  long vowel + C → short + CC): pretonic Pk +0.0035 (2 better / 59 worse), Pa +0.0035 (2 / 47);
  tonic Pk +0.0036 (8 / 60); post-tonic +0.0010 (1 / 29); unaccented words +0.0143. The few
  cases are lexical; no accent rule.
- **Pretonic a > i / a > u (Pischel's majjhima type)**: Pk +0.0043…+0.0111, Pa +0.0046…+0.0147
  (≤ 7 better per ~150–400 worse). Rejected.
- **Pretonic aphaeresis** `{a i u} > ∅ / # _ C C* Acc` (araṇya > raṇṇa): Pk +0.0035.
- **Accent and -aya- > ē** (causative -áyati vs denominative -ayáti): restricting aya > ē to
  either accent pattern hurts Pk (+0.0048 / +0.0071); `if:accented` / `if:!accented` both hurt
  (+0.0040 / +0.0031). The residual -ēi/-ai alternation is lexical (thematisation of causatives).
- **Accent-conditioned syncope** (Si, Md): post-tonic `a > ∅ / Acc C _ C V` Si +0.0125, Md
  +0.0279; pretonic Md +0.0036. Rejected.
- **Final vowels (apocope / -u)**: share of OIA final -a kept as a vowel is flat by accent class
  in Si (oxytone 83%, other accented 83%, unaccented words 84%; a:∅:u the same). Md: oxytone
  final -a lost 28%, post-tonic 39% — a weak tendency (≈ 15 items), not testable as a rule
  because apocope is late in the chain.
- **Si final -i > -a with accent carried through** (variant files in `r2/var/`): exempting
  tonic final -í costs +0.0014; exempting -i right after the tonic 0.0000. Not accentual.
- **Si umlaut with accent carried through**: umlaut only of tonic ā/a +0.0022, only of
  unaccented +0.0010, only before an accented i +0.0030, only before unaccented i +0.0002. The
  umlaut ignores the accent of both target and trigger.
- **Tonic ā > æ in Si** (first syllable) +0.0035; ā > æ before an accent +0.0044.
- **Vocalic r̩ — accent vs labial context vs next vowel.** Outcome shares (rtab.py, train):

  | lect | tonic a/i/u | pretonic a/i/u | post-tonic a/i/u | unaccented words a/i/u | after labial u | elsewhere u |
  |---|---|---|---|---|---|---|
  | Pk | 26/51/12 | 34/40/19 | 27/33/23 | 42/31/16 | 31% | 9% |
  | Pa | 39/32/18 | 51/27/22 | 75/9/12 | 64/13/15 | 36% | 8% |
  | Si | 21(+æ 14)/29/7 | 29/39/12 | 56/3/12 | 36(+æ 10)/13/21 | 20% | 13% |
  | Dhp | 40/60/0 | 14/43/43 | 14/57/29 | 19/38/38 | 73% | 9% |
  | NiDoc | 45/9/45 | 15/46(ri)/8 | — | — | — | — |

  Rules: Pk `ŕ̩ > i` −0.0000, `ŕ̩ > a` +0.0006, pretonic `r̩ > i` +0.0002; Pa `ŕ̩ > i` 0.0000,
  pretonic i +0.0007, `if:accented r̩ > i` +0.0012; Dhp pretonic `r̩ > a` +0.0043, `ŕ̩ > a`
  +0.0005; As all +0.0005…+0.0010; NiDoc `ŕ̩ > a` −0.0007 alone but +0.0005 once final
  `r̩ > a` is in (the tonic r̩ of NiDoc are nearly all oxytone -tŕ̩ kinship stems: pitŕ̩, mātŕ̩,
  duhitŕ̩ — morphology, not accent). **Si is the only lect where the accent is adopted**
  (pretonic `r̩ > i`, −0.0005; `if:accented r̩ > i / # C* _` −0.0003, the same rule in unaccented
  words +0.0004). Labial context raises u everywhere (Dhp 73% vs 9%) but only in Dhp does u
  become the majority (rule kept, LOO +0.0030); in Pk/Pa/As/Si `r̩ > u / Lab _` hurts
  (+0.0004/+0.0005/+0.0010/+0.0003) because a (Pa, As) or i (Pk) stays the plurality after
  labials. u before a following u helps Pa (−0.0002) and is neutral in Pk, Si, NiDoc.
  The consistent accent-looking pattern — i commoner in words with a Vedic accent, a commoner in
  Turner's starred (unaccented) headwords — is at least partly a stratum effect: starred etyma
  are later and more often continue the a-dialect (Pa, As, Si forms).

### 7.4 Kharoṣṭhī vowel length (Dhp, NiDoc, KharI)

Length is notation, not phonology: the gandhari dictionary and CDIAL's Dhp. forms write no long
vowels at all (0 of 1599 reflexes), NiDoc has 24 long-vowel letters in 821 reflexes and KharI 10
in 169, all in sanskritised spellings (atīta, āneṁti). Each file ends with `V > $1-long`
(@notation); removing it costs NiDoc +0.076, KharI +0.097. No evidence either way on real
shortening — the script cannot show it.

### 7.5 What the accent does and does not condition in mia-insular

- It conditions **nothing** in Pali, Prakrit, Ashokan, Gāndhārī, Apabhraṁśa or Old Sinhala
  vowels that the data can see: no pretonic shortening, no tonic protection from the two-mora
  law, no accentual gemination, no a > i/u, no accent-conditioned syncope or apocope.
- The one small positive result is **Sinhala pretonic r̩ > i** (tonic and starred r̩ > a);
  NiDoc's tonic r̩ > a is morphological (oxytone r-stems).
- Umlaut (Si ā/a > æ, Md a > e, u > i) and Si final -i > -a are position-/trigger-conditioned
  and accent-blind.
- The accent's apparent effects in `probe` cells are mostly the accented/starred split (Vedic vs
  Turner-reconstructed headwords), i.e. lexical strata, and suffix morphology.

## 10. Extension-aware score (round 4)

Scratch: `tmp/sound_changes/mia-insular/r4/` (`te.py` = trial edits scored on plain and
extension-aware train NED; `tails.py` = residual ending mismatches under the best variant;
`endchoice.py` = which variant explains each attested ending; `menu.py` = add/drop analysis
for menu candidates, computed in-process without touching `evaluate.py`; `extB_*` / `extA_*` =
before/after `score --ext` with the 13:29 menu including -akka; `OIA-*.r3.sc` = files before this
round).

### 10.1 Scores (menu: aka, akka, ikā, uka, aḍaka, allaka, araka)

| lect | plain train | plain test | ext train | ext test | chosen on train (bare / aka / akka / ikā / uka / aḍaka / allaka / araka, %) |
|---|---|---|---|---|---|
| Pk | 0.2021 | 0.2055 | 0.1777 | 0.1794 | 82 / 7 / 0 / 6 / 3 / 1 / 0 / 0 |
| Pa | 0.1415 | 0.1435 | 0.1132 | 0.1137 | 88 / 8 / 0 / 3 / 1 / 0 / 0 / 0 |
| **Si** before | 0.3699 | 0.3901 | 0.3276 | 0.3415 | 78 / 2 / 4 / 6 / 6 / 0 / 2 / 1 |
| **Si** after | 0.3708 | 0.3898 | **0.3127** | **0.3314** | 70 / 7 / 5 / 6 / 9 / 1 / 1 / 0 |
| Dhp | 0.1855 | 0.1883 | 0.1639 | 0.1667 | 87 / 5 / 1 / 2 / 0 / 3 / 0 / 2 |
| **Md** before | 0.4378 | 0.4680 | 0.3798 | 0.4048 | 66 / 11 / 7 / 1 / 12 / 1 / 0 / 2 |
| **Md** after | 0.4379 | 0.4675 | **0.3682** | **0.3931** | 65 / 12 / 7 / 2 / 11 / 1 / 0 / 2 |
| NiDoc | 0.2717 | 0.2776 | 0.2417 | 0.2535 | 84 / 10 / 1 / 2 / 1 / 0 / 1 / 1 |
| As | 0.3023 | 0.3233 | 0.2837 | 0.3003 | 87 / 6 / 0 / 3 / 1 / 1 / 1 / 1 |
| OSi | 0.4154 | 0.4078 | 0.3973 | 0.3930 | 85 / 5 / 0 / 3 / 4 / 1 / 0 / 1 |
| Ap | 0.4369 | 0.3850 | 0.3917 | 0.3622 | 65 / 8 / 3 / 11 / 6 / 2 / 2 / 3 |
| KharI | 0.2869 | 0.2882 | 0.2574 | 0.2475 | 79 / 14 / 0 / 5 / 1 / 0 / 1 / 1 |

Only Si and Md changed; all other rows are unchanged. The accent scores (Si 0.8244, Md 0.8089,
train n = 4) are unchanged. Frontiers were rerun for Si and Md.

### 10.2 Rules changed (train LOO: plain / extension-aware)

| lect | rule | tags | plain LOO | ext LOO |
|---|---|---|---|---|
| Si | `a {k g} a > ā / _ #` just before the loss of intervocalic k g c t y. Previously -aka → -aa and the hiatus rule `a > ∅ / _ V` removed one a, giving -a, so the masculine -ā was never derived. | @aka | −0.0003 | +0.0072 |
| Si | `u a > u / _ #` before glide insertion (-uka → -u, not -uva; CDIAL Si. -u 530 : -uva 104) | @hiatus @aka | −0.0006 | +0.0078 |
| Md | `u a > u / _ #` (-uka → -u: ukunu, kudu) | @hiatus @aka | −0.0001 | +0.0115 |

Tried and rejected:
- Pk `{k g} > y / i _ ā #` (-ikā → -iyā): +0.0003 plain, +0.0016 ext. CDIAL Pk. -iā 212 : -iyā 156, so -iā is correct as it stands.
- Si `{i u} k a > {ī ū}`: +0.0124 ext.
- Ap `a a > a / _ #`: plain +0.0032.
- Pa `r̩ > ā / _ #`: ≈ 0.

The @morph citation rules do not interact with the variants. The verb rules (-navā, -anī, -ai) never see extended forms, and Pk's thematisation (`∅ > a / Them _ #`) needs a final consonant.

### 10.3 What the choices say

- **-aka in Prakrit.** The chain gives -aka > -aya (ya-śruti), the modal CDIAL Pk. spelling (-aya 497, -aga 176, -aa 93, -ō 39, -aō 13 reflexes).
  - `endchoice.py`: -aya, -aga and -aa reflexes are explained by the aka variant in 46–54% of cases, and by headwords already ending in -aka in 15–31%.
  - -iya reflexes choose aka in 57% (-īya/-iya + ka). The -ō/-aō nominatives (47) are explained by nothing on the menu.
  - So Pk -aa/-aya/-aga does come from -aka, but the nominative -ō is not modelled.
- **-aka in Apabhraṃśa.** Ap has almost no -aa/-ao (1 + 0 reflexes). Its 14 -u endings choose uka 64%, but they are pronouns and the nom./acc. -u (tuhu, ehu, ajju). They are not -uka, so uka is only a surface match for them.
- **Pali, Ashokan, Kharoṣṭhī lects, OSi.** They keep -aka/-aga/-aǵa as written; aka is chosen 5–14%. KharI is the highest at 14%.
- **Sinhala and Dhivehi.**
  - **Si:** -aka has two outcomes, -ā via k-loss and contraction (now derived) and -aya (90 reflexes pick akka → -aka as the nearest). -ikā > -iya; -uka > -u.
  - **Md:** -aka and -uka both reduce (-aka > -i via -ā > -i; -uka > -u). uka and aka are chosen in about 11–12% of reflexes each, the highest -ka shares in the group besides Ap's ikā.
- **-ḍa, -la and -ra.** The -aḍaka/-allaka/-araka extensions are marginal everywhere (≤ 3%; Ap 2–3% on 185 pairs). The -ḍa/-la/-ra that do occur are mostly suffixless (see the menu proposals below).

### 10.4 Should the menu change? (evidence: `menu.py`, train, Δ ext NED when added or dropped, with the number of reflexes that switch with a strict gain)

**Add, in this order:**
1. **Plain feminine -ī** (headword final vowel → ī). This is real, productive MIA/NIA morphology: Pk karaṁḍī, kayalī, kiṁkarī; Pa kumārī, kappāsī.
   - Pk −0.0061 (425 reflexes), Pa −0.0035 (160), Si −0.0040 (105), Md −0.0051 (21), NiDoc −0.0080 (31), OSi −0.0086 (30), Ap −0.0110 (14).
   - The menu has only -ikā, which yields -iā/-iya, not -ī, in these lects.
2. **-iya/-īya** (-ika/-īya adjectives). Real morphology.
   - Pk −0.0041 (379), As −0.0064 (43), Pa −0.0016 (60), OSi −0.0031 (13).
   - Si gains nothing: -ikā already gives -iya there.
3. **-la and -ra without -ka.**
   - -la: Pk −0.0034 (283), As −0.0111 (65), Md −0.0057 (37), Si −0.0033 (116).
   - -ra: Pk −0.0031 (265), As −0.0103 (68), Md −0.0064 (44).
   - These are real -la-/-ra- suffixes (Pk aggala, aṁgāla, -illa; Si -ul/-al). In Ashokan some of the gain comes from root/stem citations (a surface fix).
   - Pk's allaka rarely fits, because Pk has **-illa**, not -alla. Replacing -allaka by -illaka would be closer, but it gains only 59 reflexes (−0.0003).
4. **Si/OSi final -u** (Si −0.0155, 388 reflexes; OSi −0.0078, 29). This is the Sinhala nominative/citation -u (aturu, aradu, ariṭu).
   - It is well attested, but it is an inflectional ending, not a stem extension.
   - Recommend it only if the menu is meant to cover citation endings (as for the Dardic -ī/-u). Pk -ō (−0.0012, 127) and Pa -ā (−0.0037, 175) are the same kind of case.
5. **Not worth adding:** -ṭa/-ḍa without -ka (≤ −0.0019 anywhere, 60–130 reflexes in Pk/Si, much of it lenition noise), -āla/-ālu (≤ −0.0038, Md only), -ikka (≤ −0.0015; 0 in Dhp, NiDoc, OSi, Ap, KharI), -ullaka (≤ 21 reflexes).

**Drop.** No item is harmful, since each is chosen only when it strictly fits.
- -aḍaka, -allaka and -araka are almost unused. The cost of dropping each, with the reflexes that would lose their best match:
  - -aḍaka: Pk +0.0001 (18), Pa 0 (1), Si +0.0003 (9), NiDoc 0, As 0.
  - -allaka: Pk +0.0003 (46), Pa +0.0001 (6).
  - -araka: ≤ +0.0012 except Md +0.0034 (24).
- -akka duplicates -aka wherever geminates are not written or are degeminated: dropping either costs 0.0000 in As, OSi and KharI. It is chosen 7–10% in Md and 4–5% in Si (as the nearest to -aka/-aya), and ≤ 1% in Pk, Pa, Dhp and NiDoc.
- In the menu's favour: dropping -aka costs Pk +0.0066 (697 reflexes), Pa +0.0075 (319), Si +0.0089, Md +0.0156. Dropping -ikā costs Pk +0.0064 (617) and Si +0.0141. Dropping -uka costs Md +0.0204 and Si +0.0086.

Caveat: the add/drop figures for Pk, Pa, Si and Md were computed before -akka joined the menu,
and those for Si before this round's two Si rules (`menu_big.txt`). The smaller lects
(`menu_small.txt`) were computed with -akka in the menu.

## 11. Verb citation forms (round 5)

Scratch: `tmp/sound_changes/mia-insular/r5/` (`tv.py` scores a rule edit for its change in plain,
extension-aware, verb-menu and accent NED; `vf.py` prints the verb-menu forms of a root;
`OIA-*.r4.sc` are the files before this round). Deadline cut short: only Sinhala and Dhivehi were
done.

| lect | test verbs | plain | verb menu before → after (test) | verb menu before → after (train) | winning form after |
|---|---|---|---|---|---|
| Si | 151 | 0.2631 | 0.3744 → **0.2185** | 0.4011 → 0.2556 | anakam 513/621 |
| Md | 51 | 0.3894 | 0.4691 → **0.3534** | 0.4186 → 0.3067 | anakam 127/168 |

**Formation.** Sinhala -navā and Dhivehi -anī both continue the neuter verbal noun **-anakam**.

**Sinhala.** Two rules were added:
- `a m > ā / _ #` (@final-nasal), before final-consonant loss;
- `∅ > v / a _ ā #` (@ya-sruti @hiatus), just before the hiatus a-loss.

The derivation is karaṇakam > karanakā > karanaā > karanavā.

**Dhivehi.** Two rules were added:
- the same `a m > ā / _ #`;
- `a ā > ī / _ #` (@hiatus), just before `a a > ā`.

This gives kuranī. Nouns in -aka keep -ā > -i (tāraka > tari).

**Checks on train.**
- Plain NED: Si +0.0000, Md +0.0007.
- Extension-aware NED: Si −0.0215, Md −0.0222.
- The accent scores are unchanged (Si 0.8244, Md 0.8089).

The `@morph` citation rules stay in place for the plain score.

**Not done:** Pk, Pa, Dhp, NiDoc, As, OSi, Ap, KharI. Pk and Pa already score better on the verb
menu than plain (they choose -ati).

## Round 6 audit

Scratch: `tmp/sound_changes/mia-insular/r6/`.
- `res_<stem>.txt`: `residuals.py` reports.
- `cand.py`: builds a candidate from edits, runs `gates.py`, and lists the train words that get better or worse (random samples).
- `cand/<name>/`: the candidate files.
- `extloo_*`: ext-LOO runs.
- `OIA-*.r5.sc`: the files before this round.

**Tool caveat.** `residuals.py` does not apply a lect's `%source` notation blocks. Its ending
pairs for Dhp (-ṇa → -na, 38 pairs) and part of Pk come from the gandhari and sheth1923
notation. The chain and `gates.py` apply those blocks, so these are not chain errors.

**Overall finding.** In the MIA and Insular lects, nearly every repeated residual turned out to be
doublets or source/dialect mixture inside one lect: CDIAL often lists both variants (Pk vattaṇa/vaṭṭaṇa, Si seya/sē, Md kam/kan,
NiDoc duta/duda, As rāja/lāja). Every candidate rule of that kind fixed and broke about the same
number of words (26/25, 54/55, 13/12, 5/6, 26/23, 72/77), so none was installed. The one
systematic gap was OSi verbs, which lacked the Si -navā derivation.

### Kept changes

**OSi.**
- `a m > ā / _ #` (@final-nasal), before final-consonant loss.
- `k > v / a _ ā #` (@int-stop-loss @ya-sruti), before `sibilant-s`.

These are the Si -navā derivation (-anakam > -anakā > -anavā) adapted to OSi, which keeps intervocalic k and so needs k > v directly. The verb-menu NED improves; plain is unchanged.

**Ap.**
- `u a > u / _ #` (@hiatus), at the end of the chain: final -ua (< -uka) > -u (ajju, paṇhu). It fixes 11 words and breaks 2.

**Si.**
- `%verb-menu +aka` (root + -aka). Against the placebo `+ara` it gains train verb −0.0022 vs −0.0000 and test −0.0015 vs −0.0008.
- The gain is marginal but beats the placebo. I did not add it to Md: there `+aka` gives −0.0016 and the placebo `+ara` −0.0007, which is too close.

| lect | train plain | train ext | train verb | test ext | test verb | ext exact (train) |
|---|---|---|---|---|---|---|
| Si | 0.3711 → 0.3711 | 0.3001 → **0.2998** | 0.2385 → 0.2363 | 0.3209 → 0.3207 | 0.2020 → 0.2005 | 19.5% |
| OSi | 0.4154 → 0.4154 | 0.3906 → **0.3889** | 0.3930 → 0.3457 | 0.3849 → 0.3842 | 0.4627 → 0.4556 | 10.4% |
| Ap | 0.4369 → 0.4358 | 0.3852 → **0.3798** | 0.1627 → 0.1627 | 0.3530 → 0.3501 | 0.3075 → 0.3075 | 6.5% → 8.1% |

The accent scores are unchanged.

Unchanged (train / test ext):
- Pk: 0.1703 / 0.1712
- Pa: 0.1100 / 0.1110
- Md: 0.3564 / 0.3787
- Dhp: 0.1624 / 0.1633
- NiDoc: 0.2291 / 0.2280
- As: 0.2683 / 0.2807
- KharI: 0.2559 / 0.2426

### Error profiles and rejected candidates (train, share of error in misses)

**Pk.**
- Profile: ending 33%, consonant ins/del 20%, consonant quality 19%, vowel quality 9%.
- The ending residue is a three-way spelling of -aka (-aya 497, -aga 176, -aa 93), -iā ~ -iyā, -ia ~ -iya, and feminine -ī / -ā not on the menu. Medially, the residue is y-śruti ~ g ~ ∅ (agaru/ayaru).
- Rejected:
  - `r dʰ > ḍ ḍʰ`: 26 better / 25 worse (uḍḍʰa ~ uddʰa, vaḍḍʰamāṇa ~ vaddʰamāṇa).
  - `r t > ṭ ṭ`: 54 / 55 (vaṭṭai ~ vattai); ext train +0.0001, verb +0.0013.
  - `u a > u / _ #`: 131 / 320.
  - Round 4 had already rejected `-ikā > -iyā`.
- All of these are dialect doublets within CDIAL Pk.

**Pa.**
- Profile: ending 34%, consonant ins/del 18%, consonant quality 17%.
- The residue is citation: -ka forms (antaka, aṭṭālaka) beside the bare stem, -in stems cited with n, feminine -ā / -ī, participles in -ita, and -ṇa ~ -na.
- None of it is a sound change, so no candidate was built.

**Si.**
- Profile: ending 36%, vowel quality 19%, consonant ins/del 15%, syncope 11%.
- Rejected:
  - Final -a apocope `a > ∅ / V C V {n s t} _ #`: 142 / 174, plain +0.0038 (as ~ asa, adan ~ adana).
  - Final hiatus `{e o} {a u} > {ē ō} / _ #`: gates ACCEPT (ext −0.0021) but 24 / 24. In the first syllable only: 13 / 12. The pairs are doublets (bʰēdá > beya and bē both listed; sē ~ seya); left alone.
- The verb classes -anavā 455 / -inavā 112 / -enavā 103 are not predictable from the OIA stem (-ya-, -aya- or thematic): each OIA stem type goes to all three. They are lexical conjugation classes.
- No Si rule has a negative ext-LOO.

**Md.**
- Profile: ending 40%, consonant ins/del 19%, vowel quality 15%.
- Rejected:
  - Late `t > ∅ / V _ V` (bʰakta > bai, muktā > mui): 19 / 70.
  - Final `m > n` (kan ~ kam): 5 / 6.
  - Deleting `{a i ī} > u / V r _ #`: its ext-LOO is −0.0005, but deleting it costs plain +0.0083 and accent +0.0278.
- The numerals (-āḷīs, -āvīs) are irregular.

**Dhp.**
- The tool caveat above covers the -ṇa pairs.
- The real residue is the gandhari dictionary's compounds and names (pakṣa > saṁkileśapakṣiya, várman > Budʰavaṁma), which make up most of "vowel/consonant missing".

**NiDoc.**
- Rejected:
  - `t > d / V _ a #`: 26 / 23 (duda ~ duta, nida ~ nita).
  - General `t > d / V _ V`: plain +0.0182, accent +0.2857.
- The documents write both.

**As.**
- The ending residue is the eastern nominative -e vs western -o (morphology).
- Rejected: `r > l` at 72 / 77 (lāja ~ rāja, aṁtala ~ aṁtara). This is east/west edict mixture in one lect, which one chain cannot fit.

**OSi.** Apart from the verb fix, the residue is the Si-type apocope ~ -a variation (ag ~ aga).

**KharI.** Not audited (137 etyma).

### Menu note
Si `+aka` is installed via `%verb-menu`. No new noun menu items are proposed beyond §10.4 (plain feminine -ī and -ā are the largest remaining ending classes in Pk and Pa).

### Round 6 second pass

This pass used the fixed `residuals.py` (it now applies the `%source` tails) and
`tmp/sound_changes/mia-insular/r6/near.py`. `near.py` takes each reflex's best variant (bare, noun menu or verb
menu) and groups the misses by differing segment pair with context. The lists are
`near_Si.txt` (NED ≤ 0.25), `all_Si.txt` (≤ 0.5), `near_Md.txt` and `all_Md.txt` (≤ 0.6).

**Kept, all passing `gates.py`:**

| lect | rule (position) | tags | better / worse words (train) |
|---|---|---|---|
| Si | `c > j / V _ V` (before `c j > s d`): kāca > kada, kávaca > kavada, durvacas > duvada | @int-voicing | 32 / 5 |
| Si | `a > u / C _ r u` (end): antara > aturu, kēsara > kesuru, úccarati > usuruvanavā | @umlaut | 33 / 13 |
| Si | `i a > i / _ #` (before the y glide): añjita > æⁿdi, abʰīta > abi, piṣṭa > piṭi | @hiatus | 280 / 175 |
| Md | `c > j / V _ V` (as Si) | @int-voicing | 2 / 1 |
| Md | `i a > i / _ #` (as Si) | @hiatus | 8 / 1 |
| OSi | `a > u / C _ r u` (as Si) | @umlaut | 2 / 1 |

The Si -iya ~ -i endings are partly doublets. The -ia contraction was kept because it is net
positive by a wide margin (ext test −0.0102) and it is the expected hiatus outcome. The
-iya forms that remain are the other stratum.

**Totals, before this round (r5 files) → now:**

| lect | plain train | plain test | ext train | ext test | verb train | verb test | exact (plain, test) | accent |
|---|---|---|---|---|---|---|---|---|
| Si | 0.3711 → 0.3701 | 0.3902 → 0.3873 | 0.3001 → **0.2928** | 0.3209 → **0.3091** | 0.2385 → 0.2331 | 0.2020 → 0.1976 | 12.3% → 12.5% | unchanged |
| Md | 0.4416 → 0.4402 | 0.4731 → 0.4720 | 0.3564 → **0.3545** | 0.3787 → **0.3761** | 0.2933 → 0.2893 | 0.3421 → 0.3328 | 7.7% | unchanged |
| OSi | 0.4154 → 0.4154 | 0.4078 → 0.4080 | 0.3906 → **0.3886** | 0.3849 → **0.3834** | 0.3930 → 0.3457 | 0.4627 → 0.4556 | 9.5% → 10.2% | — |

**Rejected in this pass:**
- **Si medial a > u before a sonorant.**
  - `/ V C _ {r l ḷ} V`: 160 / 206.
  - with n ṇ added: 231 / 598 (accent worse).
  - penult only: 124 / 188.
  - Forms like atura, maⁿgula and asuna sit beside atara, maⁿgala and asana. This is lexical.
- **Si y-umlaut** `a > æ / # C* _ C y` (madhyama > mæd-): 13 / 37.
- **OSi `c > j`:** 2 / 4. OSi spelling keeps the s.
- **Md r + u harmony:** 6 / 5.
- **Examined but not turned into candidates:**
  - **Si:** s- ~ h- ~ ∅ (≈ 230 near-miss ops). This is the learned/inherited stratum split noted in round 1; s > h still hurts.
  - **Si:** prenasalised vs plain voiced stops (ⁿḍ/ⁿg/ⁿd/ᵐb ~ ḍ/g/d/b, ≈ 60). No conditioning found (kaⁿḍa ~ kaḍaya, laᵐba ~ labba).
  - **Si:** the verb conjugation classes -anavā / -inavā / -enavā (a > i or e before n in ≈ 100 near-misses). They do not follow from the OIA stem.
  - **Si:** -aka > -aya ~ -ā ~ -uva.
  - **Si:** ḷ ~ l and ṇ ~ n.
  - **Md:** h- ~ ∅ (abu, akiri, is), which is post-s>h loss in part of the lexicon.
  - **Md:** the -enī verb class.
  - **Md:** -ak / -ik (akka-variant artifacts).
  - **Md:** the numerals.
- **KharI (audited):**
  - 128 missed pairs. 45% of the error is the ending, which is inscriptional inflection: loc. -e (ekadaśe, kupe), gen. -sa (bʰikṣusa, bʰadrasa), instr. -eṇa (utareṇa).
  - The rest is ṇ ~ n doublets (jaṇa, kahapana) and single items (jñ > ñ, rgʰ > ggʰ).
  - Nothing is systematic enough for a rule.

### Round 6 third pass (Si, Md)

**Kept:** Md `t > ṭ / r̩ _` (@rt-retroflex; the Si rule, placed before `use r-vocalic.2`). It
gives nŕ̩tyati > nar̆an(ī) and fixes 2 pairs with 0 made worse. It is a single etymon, but it is the shared Insular rule already in
Si, so it was installed for consistency.

Md now:
- train: plain 0.4400, ext **0.3539**, verb 0.2860;
- test: plain 0.4720, ext **0.3754**, verb 0.3328;
- accent unchanged.

Si is unchanged from the second pass (ext 0.2928 / 0.3091).

**Rejected or left alone, with evidence:**
- **Si æ outside the umlaut.**
  - The misses are first-syllable a → æ with no front vowel following (cʰattra > sæta, jāla > dæla, nābʰi > næba, madʰyama > mædum), 3–7 per context.
  - The general first-syllable `ā > æ` cost +0.0152 in round 4 and y-umlaut is 13 / 37. Lexical.
- **Si vowel length.** 46 near-misses are -ā vs -a after the y/v glide (anuvā ~ anuva, aniyā ~ aṇiya).
  - They come from the -aka > -ā derivation meeting -a citation forms.
  - Sinhala -ā vs -a follows animacy (masculine animate -ā), which is not phonological. Kept as is.
- **Si ya-śruti in a_a hiatus** (-aka > -aya, 89 near-misses):
  - word-final `∅ > y / a _ a #`: 15 / 19, ext train +0.0008;
  - general: 77 / 261.
  - The -aya ~ -ā split is a stratum/doublet.
- **Si prenasalised vs plain voiced stops.** No conditioning found (see the second pass).
- **Si verb classes, placebo-tested `%verb-menu` items** (train verb Δ):
  - `+yanakam` −0.0009 vs placebo `+ranakam` −0.0008;
  - `+ayanakam` 0.0000 vs placebo `+aranakam` −0.0011.
  - No real formation beats its placebo. Only `+aka` (installed earlier) does. The -inavā/-enavā classes stay lexical.
- **Md final -i loss.**
  - `i > ∅ / V {t n l} _ #` (kṣānti > ket): 6 / 6.
  - general `i > ∅ / V C _ #`: 12 / 11, plain test +0.0037.
  - Doublets.
- **Md, other classes:**
  - numerals with s- (sauvīs) against the chain's h-: probably borrowed or learned;
  - h- ~ ∅;
  - the -enī verb class;
  - -ak/-ik (akka-variant artifacts).
  - None was systematic enough for a rule.
  - Dhivehi r̆ (< ṭṭ) now also covers r̩ + t; single ṭ > ḷ is already in the chain. The Fritz source's ṣ for r̆ is a notation block.

## Round 7

Scratch: `tmp/sound_changes/mia-insular/r7/` (`snap/` = the pre-round-7 files; `cand.py` builds a
candidate from edits and runs `gates.py`; `loo.py` = leave-one-out of named lines with 6-decimal
deltas; `vsub.py` = vowel substitutions by context; `ae.py` = Si first-syllable a/æ by OIA context;
`vq.py` = vowel-quality characters per source; `g_*.txt` = gate outputs).

### Step 0: the interrupted round-7 edits, re-gated (snapshot → edit, new scoring)

Each change was gated alone against the snapshot (Si and Md in sequence). All pass; all kept.

| lect | change | ext train | ext test | net reflexes | verdict |
|---|---|---|---|---|---|
| NiDoc | `t > d / V _ a #` (jātá > jada, gʰr̩tá > gʰrida) | 0.2282 → 0.2269 | 0.2247 → 0.2192 | +8 (34/26) | ACCEPT (plain +0.0009) |
| OSi | `use rt-retroflex.t` (kr̩ta > kaḷa, as Si/Md) | 0.3883 → 0.3880 | = | 0 (1/1) | ACCEPT |
| Si | `i a > i / V C* _ #` replaces `use hiatus.ia.final.3`: no contraction in disyllables (gītá > giya, rátʰa > riya, nadī́ > niya) | 0.2928 → 0.2913 | 0.3090 → 0.3076 | +29 (30/1) | ACCEPT |
| Si | `{e o} {a u} > {ē ō}@1 / _ #` (bʰēdá > bē, yṓga > yō) | 0.2913 → 0.2894 | 0.3076 → 0.3044 | +4 (24/20) | ACCEPT |
| Md | the same `i a > i / V C* _ #` (consistency) | 0.3539 → 0.3539 | = | +1 | ACCEPT |
| Md | `i > ∅ / V {t n l} _ #` (kṣānti > ket, vēlli > vel) | 0.3539 → 0.3532 | 0.3754 → 0.3750 | 0 (6/6) | ACCEPT |

The Si e/o contraction and the Md -i loss are near-even on reflexes (doublets such as beya ~ bē, keti ~
ket stay); under the no-threshold rule they stay because they pass the gates.

### Vowel quality under the new scoring (focus 0)

The unmerged vowel scoring hardly touches this group. `vq.py` finds no ə ʌ ɛ ɔ ɪ ʊ ɨ in Si, Md, OSi,
Dhp, NiDoc, As, KharI golds. The only hits are stray non-Indic forms in CDIAL Pk/Pa/Ap (Pk śastá >
"śæśtə xudā́", bʰarukaccʰa > "βαρυɣάζα", Pa mā́rgati > "ʌ", Pa *lēŋguṭṭa > "nəguṭṭʰa", Ap ēṣá > hεs,
ēhɔ): data problems (Persian/Greek/NIA forms parsed as the MIA reflex), not chain targets. Sinhala æ/ǣ
was never folded, so the Si numbers did not move for this reason.

**Sinhala æ.** 19.8% of the Si miss error is vowel quality. `ae.py` tabulates first-syllable a/ā by the
OIA vowel and the next OIA vowel. Before a following i the attested outcome is split (a…i: æ 50, a 42, i
26; ā…i: æ 47, a 13), so the existing umlaut rules (ā before any C* + front vowel, a before CC + i) are
the regular part. Rejected widenings:
- `a > æ / # C* _ C* {i ī}` (single consonant too): ext +0.0052, 13 better / 163 worse; it catches
  prati-/pari- (piḷi-, piri-), where a > i before a liquid + i is the outcome.
- the same with e ē added: +0.0055, 14 / 172.
- restricted to an onset consonant and a non-liquid, non-retroflex single C: +0.0027, 6 / 72 (latā >
  liya, nakʰá > niya, katʰā́ > kiya: a > i before a lost stop + i wins).
- first-syllable æ with no front vowel following (áŋga > æⁿga 89 vs 748 a): lexical, as in round 6.

**Dhivehi** has no lax/centralised vowels in either source (one CDIAL æ). Its vowel-quality misses
(`vsub.py`) are the -anī/-enī verb class (a→e before -nī 45, first-syllable a→e 19), which is lexical
as the Si -anavā/-inavā/-enavā classes are, and scattered harmony doublets.

**Old Sinhala æ (installed).** CDIAL OSi. writes æ (57 forms: ā́viśati > ævisenavā, dāsá > dæsi) but
Paranavitana's Sigiri glossary never does (vāpī́ > vavi). The Si umlaut rules were added to OSi,
identical lines, and the Paranavitana `%source` block gets `{æ ǣ} > {a ā}` @notation (one source's
transcription of the same vowel). Jointly: ext train 0.3880 → 0.3876, test 0.3834 → 0.3796, plain train
+0.0008, ACCEPT. Without the notation line the umlaut alone fails (ext train +0.0007).
Rejected: Si final `{i ī} > a / C _ #` in OSi (+0.0093 ext, 7 / 32): OSi still keeps -i (stʰíti >
siți, stutí > tuti), so the Si -i > -a is later than the OSi documents.

### Consistency (focus 3) and near-misses (focus 4)

Installed (all ACCEPT; mechanism and examples in the rule comments):

| lect | rule | ext train | ext test | reflexes |
|---|---|---|---|---|
| Pa | `r̩ > i / _ ś` (kr̩śá > kisa, sadŕ̩śa > sadisa, vŕ̩ścika > viccʰika) | 0.1100 → 0.1098 | 0.1110 → 0.1108 | 12 / 4 |
| Pa | `m r > m b / V _` (āmrá > amba, tāmrá > tamba; initial mr- > m-: mrakṣa > makkʰa) | 0.1100 → 0.1098 | 0.1110 → 0.1109 | 8 / 0 |
| Pa | `ṁ l > l l` (saṁlapati > sallapati, saṁlēkʰa > sallēkʰa) | 0.1100 → 0.1099 | = | 6 / 0 |
| Pa | the three jointly | 0.1100 → 0.1095 | 0.1110 → 0.1106 | 26 / 4 |
| Pk | `m r > m b / V _` (āmrá > aṁba, tāmrá > taṁba) | 0.1702 → 0.1700 | 0.1712 → 0.1710 | 9 / 2 |
| As | `m r > m b / V _` + `r̩ > i / _ ś` (āmrá > aṁba, ētādŕ̩śa > etārisa) | 0.2679 → 0.2669 | 0.2798 → 0.2778 | 3 / 0 |
| Si | `m r > m b / V _` (āmrá > aᵐbē, tāmrá > taᵐba; doublet tam) | 0.2894 → 0.2892 | 0.3044 → 0.3043 | 6 / 4 |
| Md | `m r > m b / V _` (āmrá > aᵐbu) | 0.3524 → 0.3515 | = | 2 / 0 |
| Md | `use anusvara-loss` (as Si) replaces `ṁ {k c ṭ t p} > $2`: māṁsá > mas, pāṁśu > fas | 0.3531 → 0.3524 | = | 7 / 0 |

Pa `m r > m b` everywhere failed (verb test +0.0024; mrakṣ- > makkʰ- not bakkʰ-): the medial
environment is the regular one. Pk and Dhp already give i for r̩ś (default r̩ > i).

Rejected:
- Pa `use rt-retroflex.t` (āhr̩ta > āhaṭa, bʰr̩ta > bʰaṭa): 14 / 40, ext +0.0009; Pa writes both.
- As `{n ṇ} y > ñ ñ` (as Pa): 1 / 2, ext +0.0001.
- Si `r̩ > i / _ ś`: 1 / 3, ext +0.0001 (sādr̩śa > sari, dr̩śi > das).
- Pk near-misses (`near.py`, NED ≤ 0.25): the large groups are y-śruti spelling (-aya ~ -aga ~ -aa,
  -iā ~ -iyā) and ext-menu artifacts (-iaya); the phonological ones are doublets: niḣs- > ṇiss- 11 ~
  ṇīs- 8, kṣ- > kʰ- 93 ~ cʰ- 45, śr/śv > ss 42 ~ ṁs 7 (aṁsu). No rule.
- Pk `r dʰ > ḍ ḍʰ` (as Pa; ūrdʰvá > uḍḍʰa, vardʰaki > vaḍḍʰai): 26 / 25 again, ext test +0.0003 — the
  vaddʰ- ~ vaḍḍʰ- doublets of round 6. Other Pk ending residue: y-śruti spellings (-ya → -ga 120),
  verb participles in -ia on words the verb flag misses (-ēiā → -ia 30, -aiā → -ia 24: an engine/flag
  matter), kṣ > kkʰ ~ ccʰ (15), learned bʰ ~ h (18).
- Pa near-misses: -ṇa ~ -na, -in stems cited with n, feminine -ī/-ā, ṣaṭ > cʰa- (one numeral family),
  vy- ~ viy- (learned), dadʰāti > dahati (one root): citation or lexical.
- Morph-tagged rules were not rewritten as consistency candidates (Pa/As `∅ > a / n t _ #` vs Pk
  `∅ > a / C C _ #`); the @morph pass below removed Pa's.

**Kharoṣṭhī lects (Dhp, NiDoc), installed:**

| lect | rule | ext train | ext test | reflexes |
|---|---|---|---|---|
| Dhp | `s tʰ > ṭ ṭʰ` moved after `C $1 > $2`: the st cluster has its own Kharoṣṭhī sign, transliterated ṭṭh (stʰā́na > ṭṭʰana, ástʰi > aṭṭʰi, 14 : 2); ṣṭ stays single (duṣṭa > duṭʰa) | 0.1622 → 0.1602 | 0.1612 → 0.1601 | 16 / 1 |
| Dhp | `u r > r u / # d _` (durgá > druga, durbala > drubala; 6 : 0) | → 0.1615 | = | 6 / 0 |
| Dhp | `use assim-r.lstop` as Pk/Pa (álpa > apa, kálpa > kapa; śílpa > śilpa kept) | → 0.1617 | → 0.1605 | 5 / 1 |
| Dhp | `{ś s} v > s p` replaces `use sv-sp` (īśvará > ispara, víśva > vispa; sp 14 : śp 2) | → 0.1619 | = | 5 / 3 |
| Dhp | the four jointly | 0.1622 → 0.1589 | 0.1612 → 0.1593 | 32 / 5 |
| Dhp | `ś c > c` @notation, the c̄ sign (paścā́t > pac̄a, āścarya > ac̄aria) | 0.1589 → 0.1581 | = | 8 / 0 |
| Dhp | `ś r > ṣ` (śramaṇá > ṣamaṇa, śraddʰā́ > ṣadʰa, áśru > aṣu; ṣ 14 : ś 6) | 0.1551 → 0.1526 | 0.1547 → 0.1534 | 14 / 6 |
| Dhp | `Nasal $1 > ṁ $2 // # _`, as NiDoc (jñāna > ñana, not ṁñana) | 0.1526 → 0.1521 | = | 3 / 0 |
| KharI | `ś r > ṣ` (śramaṇá > ṣamaṇa, śrutá > ṣuta) | 0.2559 → 0.2530 | = | 2 / 1 |
| NiDoc | `r m > m`, `ś c > c` @notation: the m̄ and c̄ signs (kárman > kam̄a, dʰárma > dʰam̄a; paścā́t > pac̄a); the macron is not scored | 0.2265 → 0.2249 | = | 7 / 1 |

Left alone (doublets in the documents): NiDoc t ~ d, ṁn ~ n (gamaṁna ~ gamana, 6 : 25), s̱ ~ s (4 : 7),
ṣk > kʰ ~ k (Dhp 5 : 4, NiDoc 6 : 5); -anti/-e/-o endings are inflection.

**Other rejected candidates:**
- OSi final -a apocope `a > ∅ / V C _ #`: at the end, ext −0.0030 / −0.0012 but verb train +0.0145;
  before vowel shortening (protecting -ā), ext −0.0083 / −0.0070 but verb train +0.0054 and plain
  +0.024 (mahamata, vihara, abaya keep -a); with a consonant list, ext test +0.0080. OSi -a ~ ∅ is
  variable in the documents, as round 6 found.
- Si initial u dissimilation `u > i / # _ C C {u ū}` (utpʰulla > ipila, utkuṇa > ikiṇī): 4 / 8, ext
  +0.0002 (undura > uⁿdurā, uṣṇá > uṇu keep u).
- Ap near-misses (94 pairs ≤ 0.4): inflected forms (-ai, -ī, -ḍa), single items (tāvat > tāma). Nothing.
- Rules with no or negative ext-LOO value (`evaluate.py score --ext-loo`, Dhp NiDoc As OSi Ap KharI Md):
  only three. OSi `ā > æ / # C* _ C* {i ī e ē}` (LOO −0.0006): removal is ext train −0.0006 but ext
  test +0.0002 (veto), kept. Ap `ē > e` (−0.0007): removal costs verb train +0.0075 (the ē of -ēi
  causatives), kept. Md `a > u / _ C u #` (0.0000): removal does not improve ext train, kept.
- Si and Pa ext-LOO: Si has no rule at or below zero. Pa `use anaptyxis.ry` (vīrya > viriya) is at
  −0.00003: removing it lets ry assimilate to yy (kārya > kayya, niryāti > niyyāti, *karyatē > kayyati);
  gates ACCEPT (ext train −0.00003, test −0.0007, verb train −0.0007, verb test +0.0002 within the
  tolerance; 7 / 8 reflexes). Removed: Pali writes both (kāriya ~ kayya, vīriya), so the train evidence
  is a tie and the regular assimilation is now the chain's default; the -riya forms are the svarabhakti
  doublets.
- Pk ext-LOO: nothing below zero; the lowest is the kept @morph `ā > a / {n ṇ} _ t {i ē} #` (+0.00003).

### Per-lect noun menus (focus 5): feminine -ī / -ā

Round 6 left plain feminine -ī and -ā as the largest ending classes in Pk and Pa (Pk near-misses
a→ī after ṇ ḍ r l ≈ 90). Each lect was tested with `%ext-menu +ī`, `+ā` and the same-shape placebo
`+ū` (a one-vowel ending that is not a formation; no consonant-bearing placebo exists for a bare
vowel). Δ ext train / test:

| lect | +ī | +ā | placebo +ū | installed |
|---|---|---|---|---|
| Pk | −0.0058 / −0.0049 | −0.0028 / −0.0034 | −0.0008 / −0.0007 | +ī +ā (jointly −0.0085 / −0.0082, net +607) |
| Pa | −0.0034 / −0.0044 | −0.0036 / −0.0041 | −0.0008 / −0.0007 | +ī +ā (jointly −0.0063 / −0.0078, net +252; ext_acc train −0.029) |
| As | −0.0043 / −0.0031 | −0.0106 / −0.0140 | −0.0022 / −0.0045 | +ā |
| Dhp | −0.0014 / −0.0010 | −0.0031 / −0.0047 | −0.0025 / −0.0011 | +ā |
| NiDoc | −0.0073 / −0.0051 | −0.0080 / −0.0035 | −0.0029 / 0 | +ī +ā (jointly −0.0143 / −0.0086, net +52) |
| Ap | −0.0105 / −0.0005 | −0.0005 / 0 | 0 / 0 | +ī |
| KharI | −0.0029 / 0 | −0.0026 / −0.0038 | −0.0030 / 0 | none |
| Si | −0.0035 / −0.0045 | −0.0029 / −0.0051 | −0.0011 / −0.0004 | +ī +ā (net +100) |
| Md | −0.0047 / −0.0050 | −0.0014 / −0.0008 | −0.0020 / −0.0018 | +ī |
| OSi | −0.0092 / −0.0049 | −0.0053 / −0.0007 | −0.0073 / −0.0105 | none |

In the Insular lects +ū is a weak placebo (with vowel shortening it gives -u, a real ending from
-uka), so it is conservative there; OSi fails it. The placebo's own gain (−0.0007 to −0.003) is the
menu-inflation baseline; the installed items beat it on train and test.

**+ika (masc./neut. -ika > -ia).** The menu has -ikā, -ikka and -iyaka but not plain -ika; Pk residuals
show -a → -ia 59 (avalagna > ōlaggia, uddʰava > uddʰavia). Placebo +ira (r survives in every lect).
Pk +ika −0.0033 / −0.0030 vs +ira −0.0012 / −0.0012; Pa −0.0008 / −0.0009 vs −0.0006 / −0.0005: both
installed. Elsewhere: Md +ika −0.0036 / −0.0057 vs +ira −0.0001 / −0.0015, installed; in As, Dhp, NiDoc, Ap,
KharI, OSi, Si +ika changes nothing (the existing -ikā/-ikka/-iyaka variants already cover it) and
loses to the placebo.

**+am (neuter nom./acc. and adverbial -aṁ).** As near-misses: ∅ → ṁ at the end 26 (kā́ma > kāmaṁ,
kr̩tyá > kacaṁ); Pk ihá > ihaṁ, Pa ávara > ōraṁ. Placebo +um (m survives as ṁ). As −0.0090 / −0.0026 vs
−0.0004 / −0.0004; Pa −0.0014 / −0.0009 vs −0.0003 / −0.0001; Pk −0.0005 / −0.0004 vs −0.0001 / −0.0001:
installed in all three. Dhp, NiDoc, KharI, Ap: the placebo wins on train (Gāndhārī and Apabhraṃśa
nominative -u makes +um a real formation there), not added.

### Legacy @morph rules (focus 6)

Leave-one-out (`loo.py`, engine after the 10:35 verb-flag fix; Δ = removal minus current, 6 decimals),
then jointly per lect. 19 rules tested; 8 removed (3 in the first pass, 5 after the new noun menus
and the 10:35/11:40 engine fixes, second pass below).

| lect | rule | Δ ext train | Δ ext test | Δ plain train | decision (failing gate) |
|---|---|---|---|---|---|
| Si | `V t {i ē} > a n a v ā / _ #` | +0.000238 | −0.000707 | +0.0445 | keep (ext train) |
| Md | `V t {i ē} > a n ī / _ #` | +0.001024 | −0.000387 | +0.0535 | keep (ext train) |
| OSi | `{ē ḗ} > ∅ / t _ #` | +0.001707 | +0.001251 | +0.0021 | keep (ext train, test) |
| Ap | `∅ > a / Them _ #` | +0.001815 | +0.002941 | +0.0139 | keep (ext train, test) |
| Ap | `ē > i / {a ē} _ #` | 0 | 0 | +0.0014 | **removed** (neutral removal) |
| NiDoc | `r̩ > a / _ #` | +0.001000 | +0.003340 | +0.0014 | keep (ext train, test) |
| NiDoc | `ē > i / t _ #` | +0.000187 | 0 | +0.0022 | keep (ext train) |
| As | `∅ > a / Them _ #` | +0.000685 | 0 | +0.0017 | keep (ext train) |
| As | `∅ > a / n t _ #` | +0.000209 | 0 | +0.0007 | keep (ext train) |
| As | `ē > i / t _ #` | 0 | 0 | +0.0006 | **removed** (neutral removal) |
| Pa | `∅ > a / Them _ #` | +0.000013 | −0.000279 | +0.0004 | keep (ext train) |
| Pa | `∅ > a / n t _ #` | −0.000005 | +0.000203 | +0.0001 | keep (ext test veto) |
| Pa | `ē > i / t _ #` | +0.000057 | 0 | +0.0044 | keep (ext train) |
| Pk | `r̩ > ā / _ #` | +0.000101 | −0.000093 | +0.0003 | keep (ext train) |
| Pk | `∅ > a / Them _ #` | +0.000041 | −0.000278 | +0.0013 | keep (ext train) |
| Pk | `∅ > a / C C _ #` | −0.000026 | +0.000066 | +0.0004 | keep (ext test veto) |
| Pk | `if:verb ō > a / _ t {i ē} #` | 0 | 0 | +0.0004 | **removed** (neutral removal) |
| Pk | `ā > a / {n ṇ} _ t {i ē} #` | +0.000029 | 0 | +0.0002 | keep (ext train) |
| Pk | `ē > i / {a ē} _ #` | +0.000074 | 0 | +0.0036 | keep (ext train) |

Verb, ext_acc and ext_tone did not move for any removal. Joint removal is worse than each single one in
every lect (Pk ALL: ext +0.0008 / +0.0003; Pa ALL +0.00006 / −0.00008; As ALL +0.0009; Ap ALL +0.0018;
NiDoc ALL +0.0012). The kept rules still carry words the verb flag or the noun menu does not reach:
unflagged -ati/-atē headwords cited as -navā/-anī/-ai forms (Si, Md, Pk), thematised consonant stems
(diś > disā) and r-stem nominatives (pitr̩ > piā, pita). The three removed ones only served the bare
headword: -tē/-atē middles and -ōti verbs are now reached through the verb menu.

**Second pass** (after the +ī/+ā menus and the verb-flag engine fixes; same method):

| lect | rule(s) removed | Δ ext train | Δ ext test | Δ plain train | verdict |
|---|---|---|---|---|---|
| NiDoc | `ē > i / t _ #` | 0 | 0 | +0.0022 | neutral removal |
| Pa | `∅ > a / Them _ #` + `∅ > a / n t _ #` (jointly; singly −0.000058/−0.000440 and −0.000116/0) | −0.000175 | −0.000440 | +0.0004 | ACCEPT |
| Pk | `r̩ > ā / _ #` + `∅ > a / Them _ #` (jointly; singly −0.000046/−0.000093 and −0.000014/−0.000304) | −0.000060 | −0.000397 | +0.0015 | ACCEPT |

The new +ā item now derives the r-stem nominative (pitr̩ > piā) and the thematised -ā/-a citations
that the removed rules pasted on; the unused `class Them` lines were dropped from Pa and Pk. Still
kept after the second pass (failing gate): Si, Md verb-citation rules (ext train +0.0006, +0.0010);
OSi `{ē ḗ} > ∅ / t _ #` (ext train and test); Ap and As `∅ > a / Them _ #` (ext train); As
`∅ > a / n t _ #` (ext train); NiDoc `r̩ > a / _ #` (ext train and test); Pa `ē > i / t _ #` (ext train
+0.000057); Pk `∅ > a / C C _ #` (ext test +0.000066), `ā > a / {n ṇ} _ t {i ē} #` (ext train),
`ē > i / {a ē} _ #` (ext train). Removing all of a lect's remaining ones together is worse again
(Pa ALL ext train −0.000118 but plain +0.0048 carried by `ē > i`; Pk ALL +0.0003).

### Scores: pre-round-7 snapshot → now (same engine, run 12:23–13:40)

| lect | plain train | ext train | verb train | plain test | ext test | verb test | ext_acc train |
|---|---|---|---|---|---|---|---|
| Pk | 0.2021 → 0.2038 | 0.1684 → 0.1558 | 0.1517 → 0.1516 | 0.2055 → 0.2057 | 0.1692 → 0.1571 | 0.1635 → 0.1635 | 0.5563 → 0.5563 |
| Pa | 0.1415 → 0.1413 | 0.1091 → 0.0998 | 0.1053 → 0.1046 | 0.1435 → 0.1434 | 0.1101 → 0.0990 | 0.1140 → 0.1146 | 0.7185 → 0.6899 |
| Dhp | 0.1855 → 0.1783 | 0.1622 → 0.1521 | 0.0825 → 0.0758 | 0.1883 → 0.1854 | 0.1612 → 0.1534 | 0.0846 → 0.0785 | 0.0000 → 0.0000 |
| NiDoc | 0.2717 → 0.2733 | 0.2273 → 0.2104 | 0.2692 → 0.2666 | 0.2776 → 0.2806 | 0.2189 → 0.2038 | 0.2176 → 0.2063 | 0.2143 → 0.2143 |
| As-Ul | 0.3023 → 0.3020 | 0.2659 → 0.2455 | 0.3120 → 0.3120 | 0.3233 → 0.3224 | 0.2768 → 0.2582 | 0.3396 → 0.3396 | 0.0000 → 0.0000 |
| Ap | 0.4376 → 0.4389 | 0.3821 → 0.3716 | 0.4344 → 0.4344 | 0.3850 → 0.3887 | 0.3501 → 0.3496 | 0.3315 → 0.3315 | 0.0000 → 0.0000 |
| KharI | 0.2869 → 0.2841 | 0.2559 → 0.2530 | 0.7321 → 0.7321 | 0.2882 → 0.2882 | 0.2426 → 0.2426 | 0.2536 → 0.2536 | 0.0000 → 0.0000 |
| Si | 0.3701 → 0.3682 | 0.2926 → 0.2847 | 0.2354 → 0.2350 | 0.3873 → 0.3857 | 0.3093 → 0.2990 | 0.2129 → 0.2129 | 0.7331 → 0.7331 |
| OSi | 0.4154 → 0.4159 | 0.3878 → 0.3871 | 0.3251 → 0.3242 | 0.4080 → 0.4043 | 0.3831 → 0.3792 | 0.4696 → 0.4696 | 0.0000 → 0.0000 |
| Md | 0.4400 → 0.4382 | 0.3527 → 0.3421 | 0.3041 → 0.3041 | 0.4720 → 0.4723 | 0.3725 → 0.3614 | 0.3324 → 0.3324 | 0.7256 → 0.7256 |

ext_tone is unchanged in every lect (only Si, Md, Pa, Pk, NiDoc have a few stray marked CDIAL forms). ext_acc improves only in Pa (the +ā/+ī
variants carry the stray accented Pa forms); nowhere worse. Plain rises slightly where @morph rules
were removed (Pk train +0.0017, NiDoc test +0.0030, Ap +0.0013/+0.0037): the bare-vs-extended
artifact, as the brief expects. Pa verb test +0.0006 is inside the 0.001 tolerance.

### Not done
- Accent (focus 2): no lect in the group writes accents (a handful of stray Vedic-accented CDIAL forms in
  Si, Md, NiDoc); nothing to audit.
- KharI: only `ś r > ṣ`; the rest of its residue is inscriptional inflection (loc. -e, gen. -sa, instr. -eṇa).
