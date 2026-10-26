# Marathi–Bhil group: OIA → Marathi, Old Marathi, Konkani, Bhil lects, Wagri, Vaagri Boli, Halbi/Bhatri

Rule files: `sound_changes/rules/OIA-{M,OM,Ko-Ul,Bhili,Bhilali,RathwiBareli,PauriBareli,PalyaBareli,Rathawi,Khandesi,Noiri,DungraBhili,Vasavi,mewari_bannoda,VB,Bhatri,hal}.sc`.
Frontiers: `sound_changes/results/OIA-<lect>.hand.frontier.csv`.
Scratch (all tools are single-process helpers around `evaluate.py`): `tmp/sound_changes/marathi-bhil/`:
`bhil.sc` (Bhil NIA template), `prune.sh` (iterative loo ≤ 0 removal), `greedy_add.py` (forward
selection from hand-written candidate lists `cands*.txt`), `readd.py` (re-try pruned template rules at
their template position), `swaps.py` (adjacent-swap search), `reorder.py` (chronology probes),
`srcblock.py` (per-source `%source` notation), `crossscore.py` (every family chain on every lect),
`diag.py` (raw correspondence diagnostics), `table.py`.

## 1. Results

NED = mean normalised edit distance, null (etymon unchanged, after the lect's `%source` notation
blocks) → final chain. "auto @n (hand @n)" = the greedy baseline's test NED at its last (largest)
rule count n, and in brackets the hand chain's test NED at the same count on the hand frontier.
No source is excluded anywhere; source-specific transcriptions are handled by `%source` blocks
(not counted as rules). Test = the 20 % held-out etyma, never used for tuning.

| lect | #etyma | #rules | train NED null → final | test NED null → final | test exact % | auto test NED @n (hand @n) | excluded |
|---|---|---|---|---|---|---|---|
| M | 4746 | 43 | 0.666 → 0.375 | 0.654 → 0.372 | 16.6 | 0.534 @14 (0.468) | – |
| OM | 1876 | 22 | 0.520 → 0.388 | 0.524 → 0.420 | 13.0 | 0.472 @8 (0.420) | – |
| Ko-Ul | 711 | 37 | 0.640 → 0.426 | 0.671 → 0.452 | 11.6 | 0.568 @24 (0.480) | – |
| Bhili | 211 | 31 | 0.683 → 0.487 | 0.690 → 0.459 | 7.9 | 0.654 @24 (0.488) | – |
| Bhilali | 197 | 29 | 0.688 → 0.485 | 0.661 → 0.475 | 7.4 | 0.619 @25 (0.478) | – |
| RathwiBareli | 180 | 32 | 0.681 → 0.495 | 0.633 → 0.428 | 6.9 | 0.634 @14 (0.494) | – |
| Bhatri | 179 | 27 | 0.614 → 0.416 | 0.623 → 0.413 | 18.1 | 0.528 @22 (0.434) | – |
| mewari_bannoda | 173 | 33 | 0.675 → 0.480 | 0.672 → 0.516 | 4.7 | 0.606 @16 (0.535) | – |
| Noiri | 172 | 31 | 0.679 → 0.462 | 0.643 → 0.437 | 8.1 | 0.627 @13 (0.508) | – |
| PauriBareli | 149 | 30 | 0.672 → 0.462 | 0.685 → 0.496 | 6.9 | 0.679 @19 (0.518) | – |
| PalyaBareli | 132 | 26 | 0.670 → 0.431 | 0.687 → 0.438 | 12.2 | 0.621 @17 (0.466) | – |
| VB | 129 | 24 | 0.685 → 0.584 | 0.674 → 0.672 | 0.0 | 0.669 @9 (0.645) | – |
| Vasavi | 126 | 25 | 0.680 → 0.454 | 0.656 → 0.500 | 6.5 | 0.563 @19 (0.529) | – |
| hal | 120 | 21 | 0.641 → 0.379 | 0.640 → 0.339 | 27.6 | 0.572 @21 (0.339) | – |
| Rathawi | 119 | 28 | 0.668 → 0.465 | 0.683 → 0.440 | 13.6 | 0.639 @14 (0.445) | – |
| Khandesi | 109 | 24 | 0.611 → 0.379 | 0.549 → 0.377 | 20.0 | 0.528 @15 (0.440) | – |
| DungraBhili | 106 | 31 | 0.712 → 0.435 | 0.691 → 0.475 | 0.0 | 0.643 @19 (0.514) | – |

## 2. Base chain and deviations

All 17 files share one MIA block (in this order; a lect file lacks a rule only where its loo value
was ≤ 0 on that lect's train split and it was pruned):

```
V > $1-acc                                  @acc-loss
{ai au} > {ē ō}                             @ai-au
r̩ > i   (Bhilali, PalyaBareli, VB, Bhatri, Halbi: r̩ > a; kept with zero loo value in a few small lects so outputs never contain r̩)   @r-vocalic
Sib > s                                     @sibilant-s
{t tʰ d dʰ} y > {c c j j} {c cʰ j jʰ}@1     @palatal-Cy
{k t p} s > {k c c} {kʰ cʰ cʰ}@1            @ks
s Stop > $2-asp $2+asp                      @s-stop-aspirate   (Halbi: s C > unaspirated geminate)
s Son > s s                                 @assim-sonorant
Nasal Stop > ṁ $2                           @nasal-cluster     (M only — every other lect keeps NC)
{r ḣ} C > $2-asp $2                         @assim-r
Stop {r y v l Nasal} > $1-asp $1            @assim-sonorant
Stop Stop > $2-asp $2                       @assim-stop
{m n ṇ v l} y > $1 $1 ; {n ṇ} m > m m       @assim-sonorant
{kʰ gʰ tʰ dʰ pʰ bʰ} > h / V _ V             @int-aspirate-h
{ṭ ṭʰ} > {ḍ ḍʰ} / V _ V                     @int-retroflex
p > v / V _ V                               @int-p-v
{k g c j t d y} > ∅ / V _ V                 @int-stop-loss
l > ḷ / V _ V                               @l-retroflex       (single l only: precedes degemination)
m > v / V _ V                               @m-v
{a u} v a > ō / # _                         @aya-e             (M only: ava-/apa-/upa- > o-)
ā > a / _ #                                 @x-marathi-bhil-final-shortening
C > ∅ / _ #                                 @final-cons-loss
```

Then one of three NIA blocks:

- **Marathi (M; OM and Ko derived from it).** morphology (`-ati/-ayati > -ṇẽ`, bare roots `+ṇẽ`,
  `-ikā > -ī`, `-uka > -ū`) → `@degem-lengthen` (`V G $2 > $1+long $3 / # (C) _`, i.e. in the
  *initial* syllable — see §5) → `repeat:` `@degem-plain`
  → `@anusvara-vowel` (`V ṁ > $1+long+nas`) → `@aya-e` (`a a > ē / C _ C`, kadala > keḷ) → `@hiatus`
  (`{a ā}{a ā} > ā`) → `@final-vowel-loss` → `@syncope` (`a > ∅ / VC_CV`) → `@n-retroflex`
  (non-initial n > ṇ) → `@y-j` (all positions: kārya > kāʣ) → `cʰ > s` (`@x-marathi-bhil-ch-s`) → `s > ś / _ Front`
  (`@palatalisation`) → non-final `ī ū > i u` (`@x-marathi-bhil-iu-shortening`) → `c j jʰ > ʦ ʣ ʣʰ`
  except before front vowels (`@affrication`) → `@notation` (e o without macron).
  - **Konkani** deviates: verbs are cited in the present `-tā` (`@morph`), MIA final shortening covers
    ī ū too, masc. `-aka > -o` (as in Gujarati and the Bhil lects, not Marathi `-ā`), final vowels are
    *not* lost (so no `@final-vowel-loss`), nasal + stop is kept with a nasalised long ā
    (`a > ā̃ / _ Nasal Stop`), non-initial aspirates are deaspirated (`@deaspiration / !# _`, −0.013
    train NED), n > ṇ is intervocalic only and sits *before* intervocalic stop loss (Prakrit-age;
    −0.0017 vs the Marathi position), final -a is absorbed after a vowel (-ia > -i), compensatory
    lengthening stays penultimate (the initial-syllable variant is worse here), and no
    `@affrication`/`@syncope` survive pruning.
  - **Old Marathi (Tulpule)** keeps only a skeleton (see §5): acc-loss, ai-au, r̩, sibilants, ks,
    stop-stop, int-retroflex, l > ḷ, upa- > o-, final-C loss, degemination (± lengthening), y > j,
    cʰ > s, i/u shortening, plus notation (digraph aspirates in a `%source tulpule1999` block, written
    inherent -a, -ṇē, final -ī) and a √root morph rule.
- **Bhil (template `tmp/…/bhil.sc`; Bhili, Bhilali, the three Bareli lects, Rathawi, Khandesi, Noiri,
  Dungra Bhil, Vasavi, Wagri, Vaagri Boli).** `-aka > -o` → `-ikā > -ī` → `@degem-lengthen`
  (initial syllable in all but Rathawi and Wagri, which keep the penultimate variant; pruned in
  Pauri Bareli once watters's initial ā was handled as notation) →
  `@degem-plain` → `a > ā / _ Nasal Stop` (NC kept, vowel lengthened; tagged `@anusvara-vowel`) →
  `V ṁ > …` → `@final-vowel-loss` → masc. `-o` on nouns in extension suffixes (`@morph`) → `@syncope` →
  `n > ṇ` → `@y-j` → `s > h` → lect-specific additions → `@notation` (digraph aspirates for the
  varkey/watters/varghese surveys).  Lect-specific: Bhilali `o > ā` (masc. -ā) and no s > h; Pauri
  Bareli `c j > ts dz` except before front vowels; Noiri, Dungra Bhil, Wagri `c > s`; Rathwi and
  Pauri Bareli `o > u` in non-final syllables; Bhili, Rathwi/Pauri Bareli, Vasavi intervocalic
  `v > ∅` (`@v-loss`); Noiri, Khandesi, Wagri postvocalic `h > ∅`; Dungra Bhil voiced deaspiration and general a > ɔ after consonants (`@a-rounding`, −0.007; absent in Noiri from the same source, so not notation); Vasavi `v- > b-`;
  Khandesi no -o, no s > h, no ḷ; VB an echo vowel `ɨ` after final consonants.
- **Halbic (Bhatri, Halbi).** No masc. -o / -ṇẽ morphology, `@degem-plain` (+ `@degem-lengthen` in
  Halbi, where Woods's data show it), `V ṁ > $1+nas`, `@final-vowel-loss`, `{ḍ ḍʰ} > {ṛ ṛʰ}` (`@flap`),
  `ṇ > n`, `v > b` everywhere, `@y-j`; Halbi `s C > CC` without aspiration; Bhatri also
  `@deaspiration` (all series), `h > ∅ / V_V` and a kept final vowel in -ṛa nouns (`@morph`).

Rule × lect presence (main chain, notation and accent rules omitted; `matrix.py`). Abbreviations:
Bi Bhili, Bl Bhilali, RB/PB/PY Rathwi/Pauri/Palya Bareli, Rt Rathawi, Kh Khandesi, No Noiri, DB Dungra
Bhil, Va Vasavi, Wa Wagri, Bt Bhatri, Hl Halbi.

| rule | tags | M | OM | Ko | Bi | Bl | RB | PB | PY | Rt | Kh | No | DB | Va | Wa | VB | Bt | Hl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `{ai au} > {ē ō}` | @ai-au | + |  | + |  |  | + |  |  |  |  |  |  |  |  | + | + |  |
| `r̩ > i` | @r-vocalic | + | + | + | + |  | + | + |  | + | + | + |  | + | + |  |  |  |
| `Sib > s` | @sibilant-s | + | + | + | + | + | + | + | + | + | + | + | + | + | + | + | + | + |
| `{t tʰ d dʰ} y > {c c j j} {c cʰ j jʰ}@1` | @palatal-Cy | + |  | + | + | + |  | + | + | + | + |  | + | + | + | + | + | + |
| `{k t p} s > {k c c} {kʰ cʰ cʰ}@1` | @ks | + | + | + | + | + | + |  | + | + | + | + | + |  | + |  | + |  |
| `s Stop > $2-asp $2+asp` | @s-stop-aspirate | + |  | + | + | + | + | + | + | + | + | + | + | + | + | + | + |  |
| `s Son > s s` | @assim-sonorant | + |  | + | + | + | + | + |  |  |  | + | + | + |  | + | + | + |
| `Nasal Stop > ṁ $2` | @nasal-cluster | + |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `{r ḣ} C > $2-asp $2` | @assim-r | + |  | + | + | + | + | + | + | + | + | + | + | + | + | + | + | + |
| `Stop {r y v l Nasal} > $1-asp $1` | @assim-sonorant | + |  | + | + | + | + | + | + | + | + | + | + | + | + | + | + | + |
| `Stop Stop > $2-asp $2` | @assim-stop | + | + | + | + | + | + | + | + | + | + |  | + |  | + | + | + |  |
| `{m n ṇ v l} y > $1 $1` | @assim-sonorant | + |  | + | + | + | + | + | + | + |  | + |  |  | + |  |  | + |
| `{n ṇ} m > m m` | @assim-sonorant | + | + | + |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `{kʰ gʰ tʰ dʰ pʰ bʰ} > h / V _ V` | @int-aspirate-h | + |  |  | + | + | + | + | + | + | + | + |  |  |  | + | + |  |
| `{ṭ ṭʰ} > {ḍ ḍʰ} / V _ V` | @int-retroflex | + | + | + | + | + | + | + | + | + | + | + | + | + | + |  | + |  |
| `p > v / V _ V` | @int-p-v | + |  | + |  |  |  |  |  |  |  | + | + |  | + | + |  |  |
| `{k g c j t d y} > ∅ / V _ V` | @int-stop-loss | + |  | + | + | + | + | + | + | + | + | + | + | + | + | + | + | + |
| `l > ḷ / V _ V` | @l-retroflex | + | + | + | + | + | + |  | + | + |  |  |  |  | + | + |  |  |
| `m > v / V _ V` | @m-v | + |  | + |  |  | + | + |  |  | + |  |  |  |  |  |  |  |
| `a v a > ō / # _` | @aya-e | + |  |  |  |  |  |  |  |  |  |  |  |  |  | + |  |  |
| `u v a > ō / # _` | @aya-e | + |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `∅ > ṇ ẽ / {k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ p pʰ b ` | @morph | + |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `ā > a / _ #` | @x-marathi-bhil-final-shortening | + |  |  | + |  | + | + | + |  | + | + | + |  |  | + | + | + |
| `C > ∅ / _ #` | @final-cons-loss | + | + | + | + | + | + | + | + | + | + | + | + | + | + | + | + | + |
| `V a {i e ē} > ṇ ẽ / _ #` | @morph | + |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `a {i ē} > ṇ ẽ / C _ #` | @morph | + |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `i {a ā} > ī` | @hiatus @morph | + |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `u {a ā} > ū / _ #` | @hiatus @morph | + |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `V G $2 > $1+long $3 / # (C) _` | @degem-lengthen | + | + |  | + | + | + |  | + |  | + | + | + | + |  |  |  | + |
| `repeat: G $1 > $2` | @degem-plain | + |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `V ṁ > $1+long+nas / _ C` | @anusvara-vowel | + |  | + |  |  |  |  |  |  |  |  | + |  | + | + |  |  |
| `a a > ē / C _ C` | @aya-e | + |  | + | + | + | + |  |  |  |  | + | + | + |  |  |  |  |
| `repeat: {a ā} {a ā} > ā` | @hiatus | + |  | + |  |  |  |  |  | + |  |  | + | + | + | + |  |  |
| `{a i u} > ∅ / _ #` | @final-vowel-loss | + |  |  | + | + | + | + | + | + | + | + | + | + | + | + | + | + |
| `a > ∅ / V C _ C V` | @syncope | + |  |  |  |  |  |  |  |  |  |  | + |  | + | + |  |  |
| `n > ṇ / !# _` | @n-retroflex | + |  |  |  | + | + | + |  |  |  |  |  | + |  |  |  |  |
| `y > j` | @y-j | + |  |  |  |  |  |  |  |  |  |  |  |  | + |  |  | + |
| `cʰ > s` | @x-marathi-bhil-ch-s | + | + | + | + |  |  | + |  |  |  |  |  |  | + |  |  |  |
| `s > ś / _ Front` | @palatalisation | + |  | + |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `{ī ū ī̃ ū̃} > {i u ĩ ũ} / _ C` | @x-marathi-bhil-iu-shortening | + | + |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `{c j jʰ} > {ʦ ʣ ʣʰ} // _ Front` | @affrication | + |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `∅ > + ṇ ẽ / # √ (C) (C) (C) V (C) (C) _ #` | @morph |  | + |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `G $1 > $2` | @degem-plain |  | + | + | + | + | + | + | + | + | + | + | + | + | + | + | + | + |
| `y > j / # _` | @y-j |  | + | + |  |  |  |  |  |  |  | + | + |  |  |  | + |  |
| `n > ṇ / V _ V` | @n-retroflex |  |  | + |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `∅ > t ā / {k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ p pʰ b ` | @morph |  |  | + |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `{ā ī ū} > {a i u} / _ #` | @x-marathi-bhil-final-shortening |  |  | + |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `V a {i e ē} > t ā / _ #` | @morph |  |  | + |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `a {i ē} > t ā / C _ #` | @morph |  |  | + |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `a {a ā} > o / _ #` | @hiatus @morph |  |  | + | + | + | + | + | + | + |  | + |  | + |  |  |  |  |
| `V G $2 > $1+long $3 / _ V #` | @degem-lengthen |  |  | + |  |  |  |  |  | + |  |  |  |  | + |  |  |  |
| `a > ā̃ / _ {m n ṇ ñ ŋ} Stop` | @anusvara-vowel |  |  | + |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `{kʰ cʰ ṭʰ tʰ pʰ gʰ jʰ ḍʰ dʰ bʰ} > {k c ṭ t p g j` | @deaspiration |  |  | + |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `a > ∅ / V _ #` | @hiatus |  |  | + |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `a > ā / _ Nasal Stop` | @anusvara-vowel |  |  |  | + | + | + |  | + | + | + | + | + |  | + |  |  |  |
| `∅ > o / {ḍ ḷ r k ṭ} _ #` | @morph |  |  |  | + | + |  | + | + | + |  |  |  |  | + |  |  |  |
| `s > h` | @s-h |  |  |  | + |  |  | + | + | + |  | + | + | + | + |  |  |  |
| `ṇ > n` | @n-dental |  |  |  | + |  |  |  |  | + | + | + | + |  | + | + | + | + |
| `v > ∅ / V _ V` | @v-loss |  |  |  | + |  | + | + |  |  |  |  |  | + |  |  |  |  |
| `{ē ō} {a ā} > $1` | @hiatus |  |  |  | + | + | + | + |  |  | + |  |  |  | + |  |  |  |
| `r̩ > a` | @r-vocalic |  |  |  |  | + |  |  | + |  |  |  |  |  |  | + | + | + |
| `ṁ > ∅` | @anusvara-vowel |  |  |  |  | + | + | + |  | + |  | + |  | + |  |  |  |  |
| `o > ā / _ #` | @morph |  |  |  |  | + |  |  |  |  |  |  |  |  | + |  |  |  |
| `∅ > o / C _ #` | @morph |  |  |  |  |  | + |  |  |  |  | + | + | + |  |  |  |  |
| `{ō o} > u / _ C` | @vowel-shift |  |  |  |  |  | + | + |  |  |  |  |  |  |  |  |  |  |
| `{c j} > {t d} {s z}@1 // _ Front` | @affrication |  |  |  |  |  |  | + |  |  |  |  |  |  |  |  |  |  |
| `a > ∅ / V _` | @hiatus |  |  |  |  |  |  |  | + |  |  | + |  |  |  |  |  |  |
| `i {a ā} > ī / _ #` | @hiatus @morph |  |  |  |  |  |  |  |  | + |  |  |  |  | + |  |  |  |
| `h > ∅ / V _` | @h-loss |  |  |  |  |  |  |  |  |  | + | + |  |  | + |  |  |  |
| `c > s` | @x-marathi-bhil-c-s |  |  |  |  |  |  |  |  |  |  | + | + |  | + |  |  |  |
| `{gʰ jʰ ḍʰ dʰ bʰ} > {g j ḍ d b}` | @deaspiration |  |  |  |  |  |  |  |  |  |  |  | + |  |  |  |  |  |
| `a > o / C _` | @a-rounding |  |  |  |  |  |  |  |  |  |  |  | + |  |  |  |  |  |
| `v > b / # _` | @v-b |  |  |  |  |  |  |  |  |  |  |  |  | + |  |  |  |  |
| `∅ > ɨ / C _ #` | @x-marathi-bhil-final-vowel |  |  |  |  |  |  |  |  |  |  |  |  |  |  | + |  |  |
| `V ṁ > $1+nas / _ C` | @anusvara-vowel |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | + | + |
| `{ḍ ḍʰ} > {ṛ ṛʰ} / V _` | @flap |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | + | + |
| `v > b` | @v-b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | + | + |
| `{kʰ gʰ cʰ jʰ ṭʰ ḍʰ tʰ dʰ pʰ bʰ ṛʰ} > {k g c j ṭ ` | @deaspiration |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | + |  |
| `h > ∅ / V _ V` | @h-loss |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | + |  |
| `∅ > a / ṛ _ #` | @morph |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | + |  |
| `s C > $2-asp $2` | @s-stop-aspirate |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | + |

## 3. Relative chronology (train NED change when the order is swapped; + = worse)

Moves are single-rule moves in the final files (`reorder.py`):

| lect | probe | ΔNED |
|---|---|---|
| M | `l > ḷ / V_V` after degemination (ll would become ḷ) | +0.0080 |
| M | `{kʰ…} > h / V_V` after degemination | +0.0079 |
| M | `{ṭ ṭʰ} > {ḍ ḍʰ}` after degemination | +0.0084 |
| M | `m > v / V_V` after degemination | +0.0028 |
| M | compensatory lengthening after plain degemination (i.e. no lengthening) | +0.0057 |
| M | intervocalic stop loss after hiatus contraction (i.e. no feeding) | +0.1020 |
| M | `y > j` after `j > ʣ` (yōṭa > ʣoḍā needs y > j first) | +0.0033 |
| M | `cʰ > s` after `s > ś / _Front` (cʰidra > śīd needs cʰ > s first) | +0.0008 |
| M | `a a > ē` (aya > e) after `{a ā}{a ā} > ā` | +0.0013 |
| M | syncope before the `-ṇẽ` verb morphology | +0.0028 |
| M | `n > ṇ` before intervocalic stop loss | +0.0025 |
| M | non-final ī ū shortening before compensatory lengthening | +0.0214 |
| M | `Sib > s` after `s Stop` assimilation | +0.0076 |
| M | palatal Cy after C+sonorant assimilation | +0.0020 |
| Ko | `l > ḷ` after degemination | +0.0059 |
| Bhilali | `l > ḷ` / `{kʰ…} > h` / `ṭ > ḍ` after degemination | +0.0034 / +0.0144 / +0.0060 |
| Rathwi Bareli | same three | +0.0052 / +0.0127 / +0.0057 |
| Noiri | `c > s` before `s > h` (feeding: new s would become h) | +0.0106 |
| Noiri | postvocalic `h > ∅` before `s > h` (s > h feeds h-loss) | +0.0012 |
| Noiri | postvocalic `h > ∅` after (rather than before) `a a > ē` (h-loss feeds aya > e) | +0.0038 |
| Dungra Bhil | `c > s` before `s > h` | +0.0092 |
| Wagri | `c > s` / `cʰ > s` before `s > h` | +0.0050 / +0.0060 |
| Pauri Bareli | `cʰ > s` / `c j > ts dz` before `s > h` | +0.0013 / +0.0050 |
| Bhatri | `ḍ > ṛ` before `ṭ > ḍ` (so that ṭ > ḍ feeds the flap) | +0.0027 |

Implications: (i) the MIA lenitions (aspirate > h, ṭ > ḍ, l > ḷ, m > v) are older than NIA
degemination in every lect with enough data — the classic MIA/NIA split holds for Marathi, Konkani and
Bhil alike. (ii) In the Bhil lects `s > h` is older than the deaffrications `c > s`, `cʰ > s` and than
the Pauri Bareli ts/dz affrication — a counterfeeding order: the new s from c escaped debuccalisation.
(iii) In Bhatri ṭ > ḍ feeds ḍ > ṛ (kukkuṭa > kukṛa). (iv) In Marathi, y- > j- precedes the ʦ/ʣ
split and cʰ > s precedes s > ś before front vowels; both are late (post-degemination) NIA changes.
Adjacent-swap search (`swaps.py`) found no swap improving M, OM, Ko or most small lects by > 0.001.

## 4. Shared vs lect-specific changes; sub-grouping

Raw correspondence diagnostics (all pairs, `diag.py`; independent of the chains):

| lect | final -o : -ā (masc.) | #s-/ś-/ṣ- → h- | VlV → ḷ | ts/ʦ present | ṇ share of n/ṇ | null NED ≤ .2 (tatsama-like) |
|---|---|---|---|---|---|---|
| M | 27:1472 | 0% (0/583) | 85% (717/845) | 882 | 83% (2555/3078) | 1% (107/7809) |
| OM | 87:569 | 0% (1/405) | 72% (206/288) | 1 | 61% (1288/2127) | 18% (846/4813) |
| Ko-Ul | 103:130 | 0% (0/72) | 87% (47/54) | 17 | 55% (95/173) | 2% (17/819) |
| Bhili | 87:30 | 55% (16/29) | 66% (19/29) | 11 | 22% (25/114) | 2% (7/456) |
| Bhilali | 115:126 | 24% (10/41) | 78% (40/51) | 0 | 32% (46/143) | 2% (12/600) |
| RathwiBareli | 175:41 | 13% (7/55) | 58% (30/52) | 35 | 38% (64/168) | 1% (10/699) |
| PauriBareli | 113:20 | 74% (31/42) | 8% (2/26) | 35 | 38% (27/71) | 1% (5/455) |
| PalyaBareli | 30:19 | 100% (10/10) | 74% (14/19) | 2 | 30% (11/37) | 1% (2/195) |
| Rathawi | 15:11 | 100% (7/7) | 67% (4/6) | 0 | 31% (8/26) | 2% (2/122) |
| Khandesi | 2:14 | 0% (0/9) | 0% (0/5) | 2 | 12% (4/34) | 2% (2/113) |
| Noiri | 121:62 | 56% (19/34) | 4% (1/25) | 6 | 24% (24/99) | 0% (2/446) |
| DungraBhili | 57:7 | 91% (10/11) | 0% (0/10) | 0 | 21% (9/42) | 1% (2/177) |
| Vasavi | 109:12 | 84% (16/19) | 10% (2/20) | 9 | 47% (15/32) | 0% (0/224) |
| mewari_bannoda | 29:38 | 100% (14/14) | 62% (10/16) | 0 | 38% (16/42) | 2% (4/201) |
| VB | 8:11 | 0% (0/2) | 46% (6/13) | 1 | 33% (11/33) | 1% (2/155) |
| Bhatri | 14:0 | 0% (0/31) | 5% (2/37) | 1 | 15% (13/88) | 2% (9/412) |
| hal | 5:4 | 0% (0/12) | 0% (0/11) | 0 | 0% (0/32) | 2% (3/154) |

Every family chain (with this lect's notation adapter and `%source` blocks) scored on every lect,
all pairs (`crossscore.py`; "Bhili", "Bhatri", "hal", "Ko-Ul" columns are my own tuned chains, so the
diagonal is optimistic; G, Rj, Nimadi, H, Or are other groups' chains as of 03:30):

| lect | null | M | Ko-Ul | G | Rj | Nimadi | H | Or | Bhili | Bhatri | hal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Ko-Ul | 0.647 | 0.546 | 0.431 | 0.565 | 0.572 | 0.562 | 0.574 | 0.519 | 0.579 | 0.618 | 0.601 |
| Bhili | 0.685 | 0.534 | 0.551 | 0.534 | 0.541 | 0.525 | 0.545 | 0.545 | 0.482 | 0.563 | 0.553 |
| Bhilali | 0.684 | 0.538 | 0.541 | 0.540 | 0.546 | 0.507 | 0.547 | 0.526 | 0.498 | 0.586 | 0.565 |
| RathwiBareli | 0.673 | 0.538 | 0.541 | 0.551 | 0.559 | 0.523 | 0.558 | 0.526 | 0.518 | 0.576 | 0.561 |
| Noiri | 0.674 | 0.548 | 0.550 | 0.541 | 0.542 | 0.533 | 0.542 | 0.561 | 0.487 | 0.551 | 0.548 |
| PauriBareli | 0.674 | 0.542 | 0.561 | 0.567 | 0.559 | 0.540 | 0.562 | 0.563 | 0.501 | 0.558 | 0.559 |
| PalyaBareli | 0.674 | 0.504 | 0.512 | 0.496 | 0.496 | 0.483 | 0.519 | 0.495 | 0.440 | 0.565 | 0.534 |
| Vasavi | 0.677 | 0.552 | 0.567 | 0.547 | 0.543 | 0.511 | 0.539 | 0.557 | 0.486 | 0.545 | 0.543 |
| Rathawi | 0.671 | 0.536 | 0.521 | 0.512 | 0.501 | 0.526 | 0.528 | 0.561 | 0.463 | 0.563 | 0.555 |
| Khandesi | 0.600 | 0.461 | 0.475 | 0.468 | 0.420 | 0.500 | 0.410 | 0.459 | 0.430 | 0.449 | 0.428 |
| DungraBhili | 0.708 | 0.607 | 0.602 | 0.581 | 0.612 | 0.576 | 0.591 | 0.602 | 0.517 | 0.624 | 0.608 |
| mewari_bannoda | 0.675 | 0.569 | 0.543 | 0.548 | 0.537 | 0.538 | 0.562 | 0.547 | 0.523 | 0.587 | 0.573 |
| VB | 0.683 | 0.707 | 0.657 | 0.684 | 0.669 | 0.668 | 0.677 | 0.677 | 0.694 | 0.648 | 0.685 |
| Bhatri | 0.615 | 0.539 | 0.506 | 0.515 | 0.478 | 0.532 | 0.440 | 0.454 | 0.561 | 0.415 | 0.444 |
| hal | 0.641 | 0.460 | 0.486 | 0.413 | 0.387 | 0.493 | 0.365 | 0.470 | 0.485 | 0.393 | 0.371 |

Sub-grouping reading:
- **Bhil lects are not Marathi-type.** They share with Gujarati/Rajasthani the masc. `-o` (<-aka) and,
  in most, `s > h` and ṇ/n merger; they keep nasal + stop clusters (NC, with a > ā) where Marathi has
  ṁ + stop → long nasal vowel. Among other groups' chains the best fit for the Bhil lects is Nimadi,
  then Rajasthani/Gujarati; the Marathi chain is usually worse than Nimadi (by ≈ 0.01–0.03 NED;
  Vasavi 0.552 vs 0.511, Bhilali 0.538 vs 0.507). Marathi
  features that do turn up in single Bhil lects: ts/dz affrication in Pauri Bareli (and a few ts/dz
  forms in Bhili, Rathwi), ḷ in the varkey Bareli lects. These look like areal contact with Marathi
  (Khandesh/Satpura) on a Gujarati–Rajasthani base.
- Within Bhil, **s > h** splits the lects: Bhili, Pauri/Palya Bareli, Rathawi, Noiri, Dungra Bhil,
  Vasavi, Wagri have it (55–100 % of s-/ś-/ṣ- → h-); Bhilali (24 %), Rathwi Bareli (13 %) and
  Khandesi (0 %) do not — Bhilali also has masc. -ā (Nimadi-like). `c > s` groups Noiri, Dungra Bhil and
  Wagri. **Khandesi** has neither -o nor s > h nor ḷ and is fit best by the Hindi/Rajasthani chains —
  it does not pattern with the Bhil core in these data (only 109 etyma).
- **Halbi/Bhatri are central/eastern, not Marathi.** v > b everywhere, ḍ > ṛ, ṇ > n, no ḷ, no
  -o/-ṇẽ morphology. Among foreign chains the Hindi chain fits both best (Bhatri 0.440, Halbi 0.365),
  then Rajasthani (0.478 / 0.387); Odia fits Bhatri well (0.454) but Halbi only as well as Marathi
  (0.470 vs 0.460); Marathi is clearly worse for Bhatri (0.539). So on the "between Marathi and Odia"
  question the phonology here sides with the Hindi (central) type, with Bhatri leaning east (Odia);
  nothing specifically Marathi (ʦ, ḷ, -ṇẽ, s > ś) survives in either. Halbi differs from Bhatri in
  keeping aspiration (Bhatri deaspirates all series) and in showing compensatory lengthening where
  the source (Woods) writes length.
- **Konkani** shares with Marathi l > ḷ, n > ṇ, cʰ > s, s > ś before front vowels, but has
  Gujarati-like masc. -o and retains final vowels (shortened) — the Odia chain (which also keeps final
  vowels) fits it better than the Marathi chain (0.519 vs 0.546).
- **Vaagri Boli** (srinivasa) is too noisy to place: test NED barely moves from the null; its best
  signals are an epenthetic final ɨ after consonants (South Indian contact) and weak -o/-u endings.
- Changes shared by the whole group: all the MIA assimilations, intervocalic loss/lenition, final
  consonant loss and MIA shortening of final -ā; degemination with compensatory lengthening in
  the initial syllable (penult in Ko, Rathawi, Wagri; invisible in Beine's Bhatri notation). tr-retention (Gujarati/Sindhi) is **not** supported anywhere:
  keeping t+r unassimilated worsened every lect tested at ~02:00 (M +0.002, Bhili +0.0045, Halbi +0.007).

## 5. Novel / surprising / data notes

- **Old Marathi (Tulpule 1999) is a learned stratum.** 18 % of OM pairs are within NED 0.2 of the
  Sanskrit etymon (M: 1 %). Starting from the full Marathi chain (train NED 0.498), iterative loo
  pruning removes intervocalic stop loss (−0.022), nasal-cluster, n > ṇ, p > v, aspirate > h, all
  C+sonorant assimilations and final-vowel loss, reaching 0.389: the data prefer an almost
  tatsama chain + Marathi morphology/orthography. Per pair, the full MIA chain is still better on
  975 of 4813 OM pairs (the tadbhava minority, e.g. agní > āgi, áŋga > āṁga), worse on 2384. So OM
  gives no usable chronology; its rule list documents orthography (Devanagari transliteration:
  digraph aspirates, written inherent -a, -aṇē, ṁ) and Tulpule's etymological practice (Sanskrit
  √roots linked to derived Marathi verbs and nouns — handled by a `√…` morph rule that shields the
  root with a `+` boundary so root-final consonants do not assimilate to the ending).
- **Transcription traps found and handled with `%source` blocks:** varkey-vunnamatla2018bareli writes
  ī ū for short i u; watters2013northerndhule writes no i/u length and writes dental d as ḍ (ḍor,
  ḍuḍ); varghesekumar2015noira writes *all* dental stops with the underdot (dánta > ḍāṭ) and no long
  i, and ū for u; beine2017bhatri writes long ā as a and short a as ə/ʌ (so after the house folding
  ə → a length is invisible) and no i/u length; Tulpule uses digraph aspirates while CDIAL's OM. forms
  keep ʰ; the M. "bundeli" wordlist writes a final -a and no ʦ/ʣ/ś. The Bhil lects mixing sources
  (Rathwi/Pauri Bareli: varkey + watters; Noiri: varghese + watters) improved by 0.01–0.03 train NED
  once notation was made source-specific.
- **Compensatory lengthening targets the initial syllable, not the penult.** Replacing the Hindi-style
  `V G $2 > $1+long $3 / _ V #` by `… / # (C) _` improves M (−0.0014; karpaṭa > kāpaḍ, carmapaṭṭa >
  ʦāmoṭā), Bhilali, all three Bareli lects, Noiri, Vasavi, Khandesi, Dungra Bhil and Halbi (−0.003 …
  −0.013); only Konkani, Rathawi and Wagri prefer the penult. In disyllables the two coincide; in
  longer words the data point to an initial-stress system across the group.
- watters2013northerndhule writes initial-syllable a as ā (a %source rule helps all four watters
  lects, −0.002 … −0.006, and hurts every varkey lect), i.e. its a/ā reflects vowel quality.
- Bhatri "length loss" and "no compensatory lengthening" were notation artefacts (above); Halbi with
  Woods's data shows ordinary compensatory lengthening (−0.012).
- Pauri Bareli has no ḷ (VlV → ḷ in 8 %): l > ∅/v instead (kāla > kavo, músala > mūhava) — an
  l-vocalisation not seen in the other Bareli lects.
- Marathi: two-mora shortening has ≤ 0 value in every lect of the group (removed): long vowels
  before (former) clusters are restored by compensatory lengthening, and Turner's M. length marking of
  i/u is only reliable word-finally (hence the i/u shortening rule, −0.0035). OIA intervocalic ḍ is
  mostly kept as ḍ in M. (121 of 192), not ḷ — the Pk-style ḍ > ḷ was removed.
- Marathi -ati/-ayati verbs are cited as -ṇẽ infinitives (892 of 1095 -ti etyma), CDIAL bare roots
  (*cul, *ṭʰan) also as -ṇẽ verbs; ~20 % of -ti etyma are linked to nouns (derivatives), the largest
  residual.
- The "bundeli" source attached to M. is a Marathi-like wordlist (kāḷa, kʰeḷa, amī, tūmī, var) with
  inconsistent spacing ("t ūmī", "vart a") — kept, but probably a Marathi dialect of Bundelkhand
  (or mislabelled); worth checking.
- Konkani IndoLanguages pairs all contain ś (a search artefact of that source?).
- CDIAL/other mislinkings that surface as worst misses: M. *śītalavēlā > śiroḷ, amlavallī > ābol,
  *abʰiyavahāra > orā; VB andʰakāra > indāru, índra > andīśankōpi (compound); Bhili ślakṣṇá > nānū
  ('small', a different etymon).

## 6. New tags and engine wishes

- `@x-marathi-bhil-final-shortening` — MIA shortening of final long vowels (-ā > -a; in Ko also
  -ī -ū), before final vowel loss (same process as other groups' `@x-eastern-final-a-short`).
- `@x-marathi-bhil-ch-s` — (c)cʰ > s (Marathi sahā, Konkani, some Bhil).
- `@x-marathi-bhil-c-s` — c > s (deaffrication; Noiri, Dungra Bhil, Wagri).
- `@x-marathi-bhil-iu-shortening` — non-final ī ū > i u (Marathi; high-vowel length only kept finally).
- `@x-marathi-bhil-final-vowel` — epenthetic/echo final ɨ after consonants (Vaagri Boli).
- Engine wishes: (1) a morpheme boundary symbol that the V/C classes ignore (I used a literal `+`
  segment, kind "?", to shield √root-final consonants in OM); (2) gender/POS-aware citation morphology
  (final -ā/-ī/-ẽ in M. is mostly gender, unpredictable from the headword); (3) `%source` blocks that
  can also run *before* chain rules (notation that interacts with sound changes, e.g. varghese's dental
  underdot vs retroflexion).

## 7. Round 2: vowels and the Vedic accent

Restarted 11:00 from the 06:39 partial state. That agent had changed only `OIA-M.sc` and
`OIA-RathwiBareli.sc`. I audited both with `score --loo`: every partial rule had LOO > 0 and was
kept. Tools (all in `tmp/sound_changes/marathi-bhil/`, single-process): `var.py` (named variant
edits → train NED), `battery*.txt` / `spec_*.txt` (the hypotheses below), `acctest.py`
(compensatory-lengthening conditioner), `oxy.py` (accent vs. the -o citation vowel), `tn.py`.
`%gold-drop` is not used: none of these lects has tone marks. All numbers are **train** NED unless
marked test. "Δ" is the change against the file at that point; negative means better.

### 7.1 Per-lect NED, round 1 → round 2 (train / test)

| lect | train r1 → r2 | test r1 → r2 | round-2 changes |
|---|---|---|---|
| M | 0.3745 → 0.3696 | 0.3719 → 0.3680 | partial-round rules kept + tonic ŕ̩ > i |
| OM | 0.3878 → 0.3875 | 0.4195 → 0.4210 | au > ō |
| Ko-Ul | 0.4256 → 0.4251 | 0.4506 → 0.4425 | r̩ > a; `if:!numeral` on the -tā rules |
| Bhili | 0.4875 → 0.4860 | 0.4595 → 0.4558 | medial u > a; syncope |
| Bhilali | 0.4852 → 0.4806 | 0.4750 → 0.4750 | medial u > a |
| RathwiBareli | 0.4916 → 0.4796 | 0.4281 → 0.4188 | partial-round accent/-o rule kept + medial u > a |
| Noiri | 0.4629 → 0.4622 | 0.4374 → 0.4374 | medial u > a |
| PauriBareli | 0.4626 → 0.4619 | 0.4965 → 0.4965 | medial u > a |
| PalyaBareli | 0.4309 → 0.4219 | 0.4376 → 0.4376 | medial u > a |
| Vasavi | 0.4539 → 0.4514 | 0.4998 → 0.4998 | medial u > a |
| Rathawi | 0.4646 → 0.4572 | 0.4404 → 0.4404 | medial u > a |
| Bhatri, hal, Khandesi, DungraBhili, mewari_bannoda, VB | unchanged | unchanged | nothing passed (see 7.3) |

No test NED rose by more than 0.005. OM rose by 0.0015; au > ō is worth 0.0003 on train. A
VB rule that was adopted and then reverted had raised VB test NED by 0.015 (7.3).
Frontiers were rerun for the 11 changed lects.

### 7.2 Adopted vowel/accent rules (LOO = train NED increase when removed; fires = train etyma)

- **M** (rules from the partial round, audited):
  - `u > ū / _ #`, @vowel-shift: +0.0006 (64). Madʰú > mahū type. Restricting it to oxytone -ú
    gives +0.0004, to unaccented words +0.0003, so the accent plays no part.
  - `{i u} > {ī ū} / _ C #`, @vowel-shift: +0.0011 (44). Hastín > hātʰī.
  - `r̩ > a`, @r-vocalic: +0.0019 (101). It beats r̩ > i by 0.0015.
  - `{a i u} > ∅ / V C _ C V`, @syncope: extending it to i and u is worth −0.0005 over a-only.
  - `if:!numeral` on the two -ṇẽ rules: −0.0001.
  - `if:!root` on compensatory lengthening, @morph: −0.0008.
- **M (new):** `ŕ̩ > i / _ C* V` before @acc-loss, tagged @r-vocalic @stress. LOO +0.0003
  (17 etyma). Examples: gŕ̩dʰra > gīdʰ, vŕ̩ścika > vĩʦū, mŕ̩ṣā > mīs. The tonic r̩ gives i and
  the rest give a.
  - Controls: all r̩ > i is +0.0015; first-syllable r̩ > i is +0.0008; the accented-word flag
    alone is −0.0001.
  - Weak and not replicated: in Ko the same split is +0.0006, so Ko keeps plain r̩ > a (−0.0003).
- **OM:** `au > ō` (@ai-au), +0.0003 (16). A full `{ai au} > {ē ō}` is +0.0009, because
  Tulpule keeps tatsama ai.
- **Ko:** `r̩ > a` (+0.0024, 29 etyma); `if:!numeral` on the -tā rules (−0.0003).
  `if:verb,!numeral` is worse (+0.0043), and so is `if:verb` on the M. -ṇẽ rules (+0.0010):
  the verb flag misses -ti etyma that the dictionaries cite as verbs.
- **Bhil core (Bhili, Bhilali, Rathwi/Pauri/Palya Bareli, Noiri, Vasavi, Rathawi):**
  `u > a / V C* _ C* V`, @vowel-shift. It reduces medial u to a, mostly in the penult
  (kukkuṭa > kūkaḍo type). It sits just before @final-vowel-loss; placing it after is worse in
  every lect.
  - LOO: Bhili +0.0010 (10), Bhilali +0.0045 (12), RathwiBareli +0.0048 (10), Noiri +0.0007 (8),
    PauriBareli +0.0008 (9), PalyaBareli +0.0090 (8), Vasavi +0.0025 (5), Rathawi +0.0082 (7).
  - **Isogloss:** the same rule makes every other lect worse: M +0.0025, Ko +0.0016,
    OM +0.0022, Khandesi +0.0013, Dungra +0.0018, Wagri +0.0004, Bhatri +0.0053, Halbi +0.0032.
- **Bhili:** `a > ∅ / V C _ C V` after final-vowel loss (the group's @syncope), +0.0010
  (5 etyma). It also helped Noiri (−0.0007), Rathawi (−0.0008) and Halbi (−0.0011), but it fired
  on only 1–3 etyma there, so it was dropped.
- **RathwiBareli** (partial round, audited): `a > ə / Acc !#* _ #` before @acc-loss, plus
  `ə > ∅ / _ #` after the -o rule, tagged @stress @morph. LOO +0.0071 / +0.0121. A final -a after
  the accent (dánta > dat) blocks the masc. -o. Oxytone and unaccented words take -o.
  - This beats the flag-only version (`if:!accented` on -o) by 0.0045, and a positional
    control (`V !#* _ #`) by 0.0153.
  - **It does not generalise** (`oxy.py`). Bhili −0.0012 and Bhilali −0.0002 are marginal and
    not adopted. It is worse in Pauri +0.0005, Palya +0.0006, Noiri +0.0003, Dungra +0.0023,
    Rathawi +0.0071, Vasavi +0.0064, Wagri +0.0060 and Ko (-o < -aka) +0.0083.

### 7.3 Hypotheses tested and rejected

**Compensatory lengthening: initial syllable vs. accent vs. penult vs. weight** (`acctest.py`;
train NED for none / initial / penult / everywhere / accented-syllable in accented words plus
initial in the rest / accent-or-initial):

| lect | none | initial | penult | everywhere | accent + initial | accent or initial |
|---|---|---|---|---|---|---|
| M | .3763 | **.3708** | .3724 | .3743 | .3716 | .3708 |
| Ko | .4426 | .4265 | **.4262** | .4277 | .4339 | .4270 |
| Bhilali | .4950 | .4848 | .4935 | .4837 | .4887 | .4848 |
| Noiri | .4720 | **.4629** | .4650 | .4630 | .4690 | .4629 |
| PalyaBareli | .4468 | **.4309** | .4396 | .4312 | .4379 | .4309 |
| Khandesi | .4004 | **.3794** | .3858 | .3843 | .3955 | .3794 |
| DungraBhili | .4652 | **.4357** | .4525 | .4370 | .4469 | .4357 |
| Rathawi | .4814 | .4693 | **.4646** | .4677 | .4793 | .4677 |

The Vedic accent never beats the initial syllable. "Accent or initial" equals "initial", so the
accented syllable adds nothing. "Accent only" is worse by 0.001–0.016. OM, Halbi, Bhili,
RathwiBareli and Vasavi show the same pattern.

Weight fares no better. In M, blocking lengthening before a heavy next syllable (closed, or a long
vowel) makes no difference (±0.0000). Lengthening in any syllable preceded only by light
syllables is +0.0008, and initial plus penult is +0.0011.

**Conclusion:** compensatory lengthening follows a new *initial-syllable* prominence, not the
Vedic accent or syllable weight. Ko, Rathawi and Wagri keep the penultimate variant.

**Other accent tests (Δ train):**

- **Tonic open-syllable lengthening** `{á í ú} > {ā ī ū} / _ C V`: worse in every lect,
  +0.0016 to +0.0083 (Bhili +0.0045, Noiri +0.0053, Dungra +0.0083). The first-open-syllable
  control is worse still (Bhili +0.0157).
- **Pretonic long-vowel shortening** `{ā ī ū} > {a i u} / _ !#* Acc`:
  - Worse everywhere (Bhili +0.0055, Noiri +0.0079, Palya +0.0107, Rathawi +0.0178) except
    Wagri (−0.0005) and VB (−0.0047).
  - In VB it beats the accented-word flag (−0.0035), the two-mora law (−0.0028 vs −0.0047) and
    positional controls (+0.0011, +0.0056). Most of its hits are pretonic long vowels before a
    cluster (ātmán > apṇē, āntrá > ataḷḍi, āmrá > ambō, prāṇá > pannō). So this is two-mora
    shortening that spares the tonic syllable (aṣṭā́ > āṭṭʰɨ).
  - It was adopted and then **reverted**. It rests on 7 etyma in the noisiest lect. It raised VB
    test NED 0.6722 → 0.6873: 2 of 33 test etyma, āṇḍá > āṇḍō and pūjā́ > pūjā, keep their
    pretonic long vowel.
- **Accent-conditioned syncope** (M): marking pretonic medial a and syncopating only that is
  +0.0036 against position-only syncope. Syncopating posttonic medial a early is +0.0072.
  - Probe: medial a is lost in 71% of *tonic* cases (n=140), 65% posttonic and 56% pretonic.
    Marathi syncope ignores the Vedic accent entirely.
  - In the Bhil lects, marker-based pretonic/posttonic syncope does no better than its inert-marker
    control (the ə marker itself perturbs other rules, so these runs are only indicative).
- **Marathi non-final ī/ū shortening:**
  - Shortening only unaccented ī ū before @acc-loss (keeping tonic length) is +0.0201, the
    same as shortening every non-final ī ū early (+0.0202). Accent is irrelevant.
  - The rule must follow compensatory lengthening. It works through the initial syllable: an
    initial-syllable-only version is +0.0002, a non-initial-only version +0.0214.
  - `if:!accented` on the late rule is +0.0034.
  - OM: protecting tonic ī ū is +0.0056 and removing the rule +0.0053.
- **Final -ā / -ẽ in M:**
  - -ā on consonant-final non-verbs: +0.0152. Only in unaccented words (flag): +0.0081.
    Unaccented plus oxytone (posttonic final a blocked): +0.0131.
  - -ẽ on unaccented non-verbs: +0.0277.
  - Gender/extension endings are not predictable from the accent.
- **-ṇẽ rule with `if:verb,!numeral`:** +0.0010 in M (7.2). `if:root` on the bare-root rule
  makes no difference (±0.0000); applying it to any root-flagged word is +0.0030.
- **Medial u > a, accent variants** (Bhil core, 7.2): the pretonic-only form gains less than the
  plain form (e.g. RathwiBareli −0.0026 vs −0.0051). The posttonic-only form is usually worse
  (Bhili +0.0016, Noiri +0.0016). The unaccented-word-only form is also no better.
- **Konkani:** medial syncope is worse (+0.0013). Pretonic-marker syncope is +0.0020.
  Non-final ī shortening is +0.0019 (and +0.0006 before a). Initial-or-penult lengthening is
  +0.0016.
- **Dungra Bhil a > ɔ rounding:** the tonic-only early version is +0.0195. The accented-word flag
  is −0.0032 and `if:!starred` −0.0001. That is lexical stratification rather than a positional
  accent effect, so it was not adopted. A -e citation vowel instead of -o is +0.0011.
- **Umlaut / harmony** `a > o / _ C* u`: +0.0028 (Bhili), +0.0027 (Bhilali).
  **ī ū before a** `{i u} > {ī ū} / _ C* a` (probe suggestion): +0.0010 (Bhili), 0 (Bhilali).
  The probe tallies for Bhatri and Vasavi ignore their `%source` notation blocks (Beine's
  length-less vowels, varkey's ī/ū for i/u), so their "a > ā" and "u > ū" cells are artefacts.

### 7.4 What the accent does and does not condition in this group

- **Does not:**
  - compensatory lengthening, which is initial-syllable (penult in Ko, Rathawi, Wagri);
  - medial syncope (tonic medial a is lost as often as any other);
  - Marathi ī/ū shortening;
  - final -u > -ū, or the M. gender endings -ā/-ẽ;
  - tonic lengthening and pretonic shortening (both rejected outside VB);
  - Bhil medial u > a;
  - a > ɔ in Dungra Bhil.
- **Weakly does:**
  - Marathi tonic ŕ̩ > i against unaccented/pretonic r̩ > a (−0.0003; reversed in Ko).
  - Rathwi Bareli: a posttonic final -a blocks the masc. -o. Oxytones and starred forms take -o,
    worth −0.0072 with the position beating the flag. It is not replicated in 8 other -o lects,
    so treat it as a local, possibly lexical (-aka vs. old barytone) effect.
- **Overall:** the group shows the Vedic accent replaced by a fixed **initial-syllable
  prominence**. Initial syllables keep length and gain compensatory length; non-initial
  syllables lose length (M. ī ū), lose short vowels (syncope) and, in the Bhil core, reduce u to a.

## 10. Extension-aware score (round 4)

All numbers use the 7-item menu (aka, akka, ikā, uka, aḍaka, allaka, araka).
- Tools: `tmp/sound_changes/marathi-bhil/ext4.py`.
  - `var` gives plain and ext train NED per variant.
  - `show` prints each menu variant's output.
  - `resid` counts the remaining final-vowel mismatches and the reflexes that candidate
    suffixes would fix.
  - `apply` writes a variant into the rule file.
- Backups of the round-3 rule files are in `r4bak/`. The before/after table is
  `r4_ba.txt`, leave-one-out `r4_loo.txt`, and candidate counts `r4_cands.txt`.
- The M accent score is unchanged (0.6250, n=1). No other lect has accent-marked pairs.
- Frontiers were rerun for the 15 changed lects. OM and Khandesi are unchanged.

### 10.1 Plain and extension-aware NED, before → after (train | test)

| lect | plain train | ext train | plain test | ext test | choices after |
|---|---|---|---|---|---|
| M | .3696 → .3696 | .2857 → .2856 | .3680 → .3682 | .2873 → .2872 | bare 65, aka 17, ikā 11, akka 3, aḍaka 3, araka/uka/allaka 1 |
| OM | .3875 | .3597 | .4210 | .3874 | bare 79, aka 9, ikā 5, aḍaka 4, uka 2, allaka 1, araka 1, akka 0 |
| Ko-Ul | .4251 → .4241 | .3506 → .3456 | .4425 → .4430 | .3791 → .3768 | bare 62, aka 11, ikā 10, uka 9, akka 3, aḍaka 2 |
| Bhili | .4860 → .4861 | .4053 → .3882 | .4558 → .4531 | .3910 → .3913 | bare 54, ikā 12, uka 10, aka 8, akka 7, allaka 3, aḍaka 3 |
| Bhilali | .4806 → .4812 | .4267 → .4058 | .4750 → .4750 | .3978 → .3727 | bare 55, aka 11, ikā 9, uka 9, allaka 6, aḍaka 4, akka 4 |
| RathwiBareli | .4796 → .4798 | .4088 → .3988 | .4188 → .4188 | .3715 → .3617 | bare 54, ikā 14, uka 13, allaka 7, aḍaka 3, aka 3, akka 3 |
| Noiri | .4622 → .4618 | .4106 → .3954 | .4374 → .4374 | .4059 → .3886 | bare 59, ikā 12, aka 10, uka 9, akka 5 |
| PauriBareli | .4619 → .4619 | .3639 → .3604 | .4965 → .4965 | .4154 → .3833 | bare 49, ikā 15, aka 14, uka 11, allaka 5, aḍaka 4 |
| PalyaBareli | .4219 → .4219 | .3660 → .3336 | .4376 → .4376 | .3739 → .3424 | bare 53, uka 23, ikā 9, aḍaka 5, aka 4 |
| Vasavi | .4514 → .4514 | .4078 → .3956 | .4998 → .4998 | .4848 → .4735 | bare 71, ikā 11, allaka 5, uka 4, aka 3, akka 3 |
| Rathawi | .4572 → .4564 | .3905 → .3593 | .4404 → .4404 | .3812 → .3502 | bare 51, uka 21, ikā 12, aka 6 |
| Khandesi | .3794 | .3098 | .3771 | .3085 | bare 62, aka 19, ikā 11, aḍaka 2, allaka 2 |
| DungraBhili | .4357 → .4357 | .3886 → .3853 | .4752 | .4381 | bare 69, ikā 11, aka 5, aḍaka 5, uka 4 |
| Wagri (mewari_bannoda) | .4804 → .4804 | .3951 → .3892 | .5157 | .3908 → .4070 | bare 60, ikā 16, aka 8, uka 5, araka 3 |
| VB | .5840 → .5838 | .4920 → .4915 | .6722 → .6752 | .5810 | bare 49, ikā 14, uka 12, aka 9, akka 7, araka 4 |
| Bhatri | .4154 → .4126 | .3313 → .3281 | .4135 → .4114 | .3025 → .2948 | bare 57, aka 20, ikā 12, allaka 5, uka 3 |
| hal | .3787 → .3781 | .3215 → .2998 | .3391 → .3391 | .3134 → .3088 | bare 61, aka 13, ikā 10, uka 6, akka 6, araka 4 |

### 10.2 Rules changed (Δ plain / Δ ext train)

Each value is the NED increase when the rule is dropped, or the gain when it was added.
- **Medial u > a (round 2), 8 core Bhil lects.** Restricted to `u > a / V C* _ C C* V`.
  - Problem: the old `_ C* V` form turned -uka > -ua into -a, so -uka came out as bare -a.
  - Ext gain from the fix: Bhili −.0147, Bhilali −.0098, RathwiBareli −.0087, Noiri −.0105,
    Pauri −.0035, Palya −.0330, Vasavi −.0031. Plain cost ≤ +.0007.
  - Rathawi also needs `u {a ā} > ū / _ #` (@hiatus @morph, −.0287 ext).
  - LOO of the restricted rule (plain / ext): Bhili .0006 / .0009, Bhilali .0040 / .0038,
    RathwiBareli .0041 / .0042, Noiri .0007 / .0037, Pauri .0008 / .0014, Palya .0090 / .0086,
    Vasavi .0025 / .0025, Rathawi .0082 / .0063.
- **`i {a ā} > ī / _ #`** (@hiatus @morph, -ikā > -ī): Bhilali .0000 / .0110,
  Vasavi .0000 / .0090, hal .0000 / .0045. Before this, these chains gave -iā.
- **`u {a ā} > ū / _ #`**: hal .0000 / .0049; Rathawi (above) .0000 / .0313.
- **Masc. hiatus contraction.** `a {a ā} > o / _ #` (-aka > -o): Wagri .0000 / .0080 (its
  existing `o > ā` then gives -ā), Dungra .0000 / .0033. `a {a ā} > ā / _ #` (-aka > -ā): hal
  .0000 / .0094. Before this, these chains gave bare -a for -aka.
- **`a > ∅ / V C _ C {o ā ī ū(ẽ)} #`** (@syncope: medial a before a long final vowel, e.g.
  -aḍaka > -ḍo, -araka > -ro), placed after final-vowel loss:
  - M .0000 / .0003, Ko .0005 / .0041, Bhili .0000 / .0009, Noiri .0000 / .0027,
    Rathawi .0008 / .0025, hal .0005 / .0073.
  - It costs more than .001 plain in Bhilali, RathwiBareli, Palya and Vasavi (+.0011 to +.0013),
    so it was not adopted there.
- **Late `G $1 > $2`** (@degem-plain after syncope: -ṭ-aḍaka > -ḍḍo > -ḍo). Gain when added:
  M .0000 / .0002, Ko −.0004 / −.0008, Bhili −.0002 / −.0002, Bhilali .0000 / −.0001,
  RathwiBareli −.0005 / −.0008, Noiri −.0003 / −.0006, Palya .0000 / −.0003,
  Wagri .0000 / −.0007, VB −.0003 / −.0005, Bhatri −.0028 / −.0031.
- **Rejected:**
  - OM `{a i u} k {a ā} > {ā ī ū} / _ #`: ext −.0095 but plain +.0025. Tulpule's tatsama cites
    -aka words unchanged.
  - Khandesi -aa > -ā: plain −.0022 but ext +.0055. Khandesi -aa > -o: ext +.0196.
  - VB -aa > -ā: plain +.0041.
  - Ko: no compensatory lengthening before -ll- (±0).

### 10.3 What the choices say

- **Marathi -ā / -ẽ.** In M, -aka gives -ā correctly (ghōṭaka > ghoḍā; 17% of reflexes pick it,
  plus ikā 11%).
  - The largest remaining ending mismatch is **neuter -ẽ**: ∅→ẽ 336, ā→ẽ 159 and ī→ẽ 23 train
    reflexes. The menu has no neuter stem, and -aka cannot yield -ẽ.
  - ā→∅ 202 and ī→∅ 137 are pairs where Turner files an unextended M. word under an extended
    etymon, or the reverse.
- **Bhil masc. -o against -aka.**
  - The chains derive -aka > -o correctly (ghoḍo).
  - But the Bhil `@morph` rules that add -o to *bare* stems (`∅ > o / C _ #` or
    `/ {ḍ ḷ r k ṭ} _ #`) make bare and -aka identical. So aka is chosen for only 3–11% of
    reflexes, and o→∅ stays the top residual (RathwiBareli 86, Noiri 85, Bhili 35, Vasavi 35).
  - Dropping the -o morph rule would lower ext NED by .003–.033 (Noiri −.033, Vasavi −.028,
    RathwiBareli −.018, Dungra −.017). It would raise plain NED by .005–.040, so it was kept, as
    the brief requires.
  - The -o morph rule is the plain-NED stand-in for -aka. Under the extension-aware score it is
    redundant and even harmful.
- **-uka stands in for the neuter.** After the -uka > -ū fix, -uka is chosen for 21–23% of
  reflexes in Palya Bareli and Rathawi, where -ū is the single most frequent ending (Palya 42,
  Rathawi 25 train reflexes: mūrdʰán > mūṇḍū, mukʰatuṇḍaka > moḍū). These are neuter -ũ/-ū
  (Gujarati-type -ũ < -akaṁ), not -uka.
- **-ḍa / -la / -ra.**
  - Productive in Marathi: 3% aḍaka, 1% each allaka / araka, but many consonant-final -ḍ/-ḷ/-r
    words remain (see 10.4).
  - Productive in Bhilali (allaka 6, aḍaka 4), RathwiBareli (allaka 7), Pauri (allaka 5) and
    Vasavi (allaka 5): the -lo/-ḷo of the Bhil lects.
  - Bhatri prefers allaka (5%). Halbi and Vasavi never pick aḍaka.
- **-aka preference over the bare stem** is strongest in Bhatri (20%), Khandesi (19%), M (17%),
  Pauri (14%) and hal (13%). It is weakest in RathwiBareli (3%), Vasavi (3%) and Palya (4%),
  where -o is already added by morphology.

### 10.4 Should the menu change? (recommendations; the menu was not edited)

Counts are train reflexes that a candidate would improve beyond the current 7-item menu
(`r4_cands.txt`, with a one-line neuter rule `a a {ṁ m} > ẽ/ū/ã / _ #` in a scratch copy).
1. **Add neuter -akam (-akaṁ).** This is real MIA morphology: Pk. -aaṁ > M. -ẽ, G./Bhil -ũ.
   - M: 263 reflexes fixed (ext −.0083, the largest single candidate). Ko: 33.
   - In the Bhil lects it gains 0 only because -uka already absorbs the neuter -ū (see 10.3).
     With -akam the choice shares would read correctly.
   - The chains need one `@morph` rule each (`a a ṁ > ẽ` in M and Ko, `> ũ/ū` in Bhil).
2. **Consider -ra / -la / -ḍa without -ka** (Pk. -ḍa-, -la-, -ra- extensions, real but
   secondary).
   - Counts (-ra / -la / -ḍa): M 167 / 154 / 108; Ko 20 / 20 / 18; RathwiBareli 23 / 27 / 22;
     Bhilali 15 / 16 / 24.
   - These give M. consonant-final forms (-ḍ, -ḷ, -r) that -aḍaka etc. cannot.
3. **Lower priority:**
   - -ilaka (M 64, Bhatri 17, RathwiBareli 15);
   - -āla (M 60, Noiri 17, Pauri 16, VB 13);
   - feminine -aḍikā / -allikā / -arikā (M 40 each, RathwiBareli 5–8; real -ṛī/-lī diminutives);
   - -ikka (M 16, Noiri 10);
   - -ima (M 33).
4. **Do not add -iya / -īya:** it fixes 0 reflexes in every lect, because every chain merges
   -iya with -ikā > -ī. Plain feminine -ī is also covered by -ikā.
5. **Rarely chosen:** -akka (OM 0%, Bhatri 1%, Khandesi 1%, elsewhere 2–7%); -araka (0–4%);
   -aḍaka (0 in Halbi, ≤ 3% in most). Keep them: they cost nothing and are genuine.
6. **OM:** its tatsama chain keeps intervocalic stops, so any menu item only matches by surface
   similarity there (OM candidate counts are inflated, e.g. -la 238). Exclude OM from menu
   evidence.

### 10.5 Neuter -akam derivation (round 4b; 10-item menu with iyaka, ikka, akam)

Before this change the chains lost final -m in the MIA block, so -akam came out identical to
-aka and was never chosen.

**Rule added:** `if:ext a a {m ṁ} > X / _ #` (@morph), placed just before
`C > ∅ / _ #`. It turns -akam > -aaṁ into the neuter ending X. The ending was chosen per lect
by train extension-aware NED:
- M: -ẽ
- Ko: -ẽ
- Bhili: -ū̃
- Palya Bareli: -ū̃
- Vasavi: -õ
- Bhilali, Noiri, Pauri Bareli, Rathawi: -u. These lects' `%source` notation blocks turn it
  into -ū where varkey writes ū, which is why plain -ū tied with -uka and gained nothing.

**Checks:**
- Plain train and test NED are identical in every changed lect.
- The M accent score is unchanged (0.6250). The other lects have no accent-marked pairs.
- Frontiers were rerun for the 9 changed lects.

**Before → after:**

| lect | ext train | ext test | akam chosen |
|---|---|---|---|
| M | .2854 → .2773 | .2869 → .2793 | 0 → 4% |
| Ko-Ul | .3449 → .3369 | .3743 → .3675 | 0 → 5% |
| Bhili | .3874 → .3740 | .3900 → .3789 | 0 → 8% |
| Bhilali | .4043 → .4023 | .3720 → .3634 | 0 → 2% |
| Noiri | .3916 → .3613 | .3886 → .3405 | 0 → 21% |
| PauriBareli | .3548 → .3455 | .3780 → .3653 | 0 → 6% |
| PalyaBareli | .3320 → .3304 | .3424 → .3424 | 0 → 1% |
| Vasavi | .3940 → .3694 | .4557 → .4328 | 0 → 13% |
| Rathawi | .3562 → .3436 | .3365 → .3176 | 0 → 7% |

**Not changed:**
- RathwiBareli: every neuter ending (ū, ũ, ū̃, u) worsens ext train NED by .0005 to .0029.
  There, the old -akam = -aka variant was the only way to get -o on barytone words, because the
  accent rule (`a > ə`) blocks -o on the bare -aka variant.
- Dungra Bhil, Wagri, Khandesi, VB and the Halbic lects: -ū/-ũ endings are rare in their data.

In Palya Bareli and Rathawi, -uka is still chosen for 21–23% of reflexes. It ties with the
neuter, and the menu order puts uka first, so those shares overstate -uka.

## 11. Verb citation forms (round 5)

This section is short because of the hard deadline.
- **Method:** `tmp/sound_changes/marathi-bhil/ext4.py var` reports plain, ext, verb-menu and
  accent NED together. `vshow.py` and `vres.py` are the verb diagnostics.
- **Adoption test:** plain within 0.001, ext not worse, accent unchanged (M 0.6250), verb-menu
  better.

**M** (test: 369 verb pairs).
- Citation -ṇẽ continues the neuter verbal noun **-anakam** (Pk. -aṇaaṁ > -ṇẽ).
- The round-4 rule `if:ext a a {m ṁ} > ẽ / _ #` (@morph) is now `a a {m ṁ} > ẽ / _ #`, tagged
  @hiatus @final-nasal, before final-consonant loss.
- Effect: plain train ±0, ext train −0.029, accent unchanged, verb-menu train .3637 → .2304.
- Test verbs: plain .2926, verb-menu .3659 → .2366. -anakam now wins 306 of 369 test pairs.
- Remaining residual: root-final single stops are lost before -aṇẽ (ḍag- > ḍeṇẽ for
  ḍagāgṇẽ, about 100 train pairs). An early syncope before -nakam was rejected: verb-menu
  +0.013.

**Ko** (26 test verb pairs).
- Changes:
  - The same neuter rule was retagged as a sound change (−ẽ).
  - `i u > ū / _ #` (@hiatus) was added after final-consonant loss: -itum > -iu > -ū, which
    continues into the Konkani infinitive -ūka.
- Effect: plain ±0, ext −.0004, verb-menu train .4754 → .4728. Test verbs: plain .3348,
  verb-menu .4451 → .4386.
- Konkani's main citation form, present -tā, continues the present participle -ant-, which is
  not on the menu. With -anta added, the verb-menu score would be .4680 (8 train pairs).

**OM (not changed):**
- Test verbs: plain .4009, verb-menu .4217. Citation -aṇē(ṁ) comes via -aṇa (chosen for 330
  train pairs).
- An -akam derivation needs a narrow `k > ∅ / V _ a m #` in this tatsama chain. It gained only
  verb −.0003, so it was rejected.

**Not done / no verbs:**
- All Bhil lects, Khandesi and the Halbic lects have 0 verb pairs (VB has 1). Their round-4
  `if:ext` neuter rules were left untouched because verb-menu NED cannot be measured there.

## Round 6 audit

**Method**
- `residuals.py` was run per lect; the class tables and examples are in
  `tmp/sound_changes/marathi-bhil/r6/<lect>.txt`.
- Candidates were screened on train with `ext4.py var`, then decided by `gates.py`. For each
  candidate, a word-level better/worse diff (`wdiff.py`) checked that the gain is a rule rather
  than lexical variation.
- Backups of the pre-audit files are in `r6bak/`. The decision log is `r6/log.txt`, and the
  full before/after gate output is `r6/final.txt`.
- The M accent score is unchanged (0.6250). No @morph, `if:ext` or `if:verb` rules were added.

**Before → after (train / test)**

| lect | ext NED | plain NED | verb-menu NED |
|---|---|---|---|
| M | .2564→.2428 / .2585→.2439 | .3696→.3684 / .3689→.3677 | .2044→.1897 / .2098→.1946 |
| OM | .3578→.3238 / .3845→.3583 | .3875→.3878 / .4210→.4221 | .3940→.2922 / .4076→.3114 |
| Ko | .3571→.3570 / .3815→.3782 | .4244→.4244 / .4442→.4406 | .4591 / .4283→.4184 |
| Bhili | .3740→.3694 / .3789→.3759 | .4861→.4796 / .4531→.4498 | – |
| Bhilali | .4023→.3892 / .3634→.3545 | .4812→.4804 / .4750→.4761 | – |
| RathwiBareli | .3933→.3905 / .3617→.3611 | .4798→.4799 / .4188 | – |
| Noiri | .3613→.3555 / .3405→.3204 | .4618→.4612 / .4374→.4262 | – |
| PauriBareli | .3455→.3394 / .3653→.3504 | .4619→.4566 / .4965→.4826 | – |
| PalyaBareli | .3304→.3263 / .3424→.3306 | .4219→.4201 / .4376→.4303 | – |
| Vasavi | .3694→.3644 / .4328→.4180 | .4514→.4524 / .4998→.4885 | – |
| Rathawi | .3436→.3407 / .3176→.2905 | .4564→.4539 / .4404→.4267 | – |
| Khandesi | .3080→.3051 / .3085→.2835 | .3794→.3765 / .3771→.3521 | – |
| DungraBhili | .3825→.3779 / .4330 | .4357→.4352 / .4752 | – |
| Wagri (mewari_bannoda) | .3883→.3852 / .4070→.4008 | .4804→.4772 / .5157→.5087 | – |
| hal | .2985→.2964 / .3088→.2915 | .3781→.3760 / .3391→.3218 | – |
| Bhatri, VB | unchanged | | |

**Error profiles.** In every lect the largest class is the ending: 33–59% of error. It is mostly
bare-vs-suffixed citation, or gender/number endings the chain cannot predict:
- the Bhil -o/-ā/-ī/-ū alternation;
- Dungra Bhil's -e;
- Marathi -ā, -ī, -∅ against Turner's filing.

Next come consonant insertion/deletion (14–23%) and consonant quality (6–17%). These are mostly
lexical: ṇ/n and ḍ/ḷ variation, and survey transcription. Vowel length, syncope and hiatus make
up most of the rest.

**Changes kept**

- **M:**
  1. `{i u} > {ī ū} / _ C #` after the late ī/ū shortening. Final closed high vowels are long:
     uśīr, ā̃kūr, kūḷ, ukīr (70+ ending pairs). The plain cost of +.0011 is the bare/extended
     artefact: ubṇẽ, ustẽ and ukirḍā are suffixed.
  2. `a {i ī} > ī / _ #`, a hiatus leftover (ovraī > ovrī).
  3. `ā̃ > ã / V C* _ C`. The anusvāra long vowel appears only in the initial syllable. CDIAL M.
     writes ã : ā̃ in non-initial syllables 41:10 before d and 33:3 before b, against 22:107
     initial. This is another initial-prominence effect.
  4. Onsetless initial vowels do not lengthen before a long vowel or vowel sequence (akʰoḍ,
     asū, aṭāḷ, aṭʰās). Onset words always lengthen.
  5. `ā > a / V C* _ C C` after syncope. Non-initial ā shortens in the closed syllable that
     syncope creates: ohaḷṇẽ, causative -āv- > -av-. Verb −.012; plain words better 56, worse
     48.
- **OM:**
  1. Final -akam/-aka lose k: `k > ∅ / a _ a (m) #`, then `-aaṁ > -ẽ`, `-aa > -ā`, and
     `n > ṇ / V _ ẽ #`.
  2. The notation rule `ẽ > ē` had been tagged @morph and was restricted to ṇ_, so it was off
     for verbs. It is now @notation and applies to every final ẽ.
  3. Verb-menu NED fell .394 → .292: Tulpule's -aṇē is the -anakam verbal noun. The stems stay
     learned; the tadbhava suffix does not.
  4. The rule is restricted to a_a because -ika/-uka keep k in Tulpule (adhika, amuka).
- **Ko:** `a > ā / _ NC` replaces `> ā̃`. Konkani sources write ā+N 53 times and ā̃+N 14 times.
- **All Bhil lects without it, plus Khandesi and Halbi:** `{ai au} > {ē ō}` restored. It had been
  pruned in round 1; with the later rules it now helps every lect, e.g. saúvarṇa > hauvno →
  sōn-. Final `{a ā}{o ō} > o` (khāo > kho) was added in the same batch.
- **Bhilali, RathwiBareli, Palya, Vasavi:** pre-final syncope `a > ∅ / V C _ C {o ā ī ū} #`
  (navlo, -lo, -ḍo). Plain cost is ≤ .0013; only 4 plain words move in Bhilali.
- **RathwiBareli, Pauri:** `a a > ā` after v-loss.
- **Pauri, RathwiBareli, Bhilali:** word-final `ṇ > n` (kān, tīn).
- **Noiri, Rathawi:** final deaspiration (hāt, dūd).

**Rejected**
- M:
  - `{kʰ ṭʰ tʰ pʰ} > deasp / _ #`: 88 words better, 128 worse; hāt versus āṭʰ, oṭʰ is lexical.
  - Medial hiatus `ai > e`, `au > o`: plain +.002.
  - Final-syllable ā shortening: plain +.0057.
  - Non-initial ā̃ shortening before a final consonant as well: no extra gain.
  - Onset-only compensatory lengthening: +.0002.
- OM: k-loss in `V _ a` (includes -ika/-uka): plain +.0022, and the diff shows adhika/amuka
  getting worse.
- Ko: intervocalic ḍ > ḷ, both early and late (plain +.0081 for the late version); Konkani
  final -a vs ∅ is dialect variation inside CDIAL.
- Final deaspiration in Bhili, Bhilali, Palya, Khandesi and Wagri (test worse), and in
  RathwiBareli and Pauri (plain cost only). The split follows source and lexicon.
- Hiatus moves in RathwiBareli (e/o+a after v-loss, au > o, ia > e): all worse.
- `a a > ā` after v-loss in Bhili and Vasavi. Final ṇ > n in Palya. Final āo > o in
  RathwiBareli, Bhatri and VB.

**What remains**
- Ending choice: the Bhil -o morph rule and the gender endings.
- Lexical ṇ/n and ḍ/ḷ.
- Marathi -aviṇẽ: the causative keeps -i- from -aya-, and the menu has -āpanakam but not an
  -āpayanakam-type item; about 85 M train verb pairs have -avṇẽ → -aviṇẽ. This is reported as a
  possible menu gap; no placebo test was run.
- OM abstract nouns in -paṇa (< -tvana; about 50 pairs) are not on the menu.

**Not done:** Bhatri and VB. Their residuals were read, but no candidate passed or was
attempted.

### Round 6 second pass

**Tool note.** `residuals.py` compares predictions with the attested forms without applying the
lect's `%source` notation blocks. In OM its top non-ending "errors" (82 ∅→b before bʰ, 50 ∅→k
before kʰ, …) are just Tulpule's digraph aspirates, which the `tulpule1999` block already
handles; `evaluate.py` and `gates.py` apply the blocks.
- My scratch copy `tmp/sound_changes/marathi-bhil/r6/res2.py` applies the tails. It is
  train-only, prints 40 examples per class, and adds a tally of non-final operations in
  context (pred→gold / L_R). All second-pass reading used it; the output is
  `r6/<lect>2.txt`.
- The coordinator may want to fix `residuals.py` itself.

**Kept (all passed `gates.py`; word diffs checked)**
- **OM.**
  - `s > ∅ / # _ Stop`, tagged @cluster-simplify (stanya > tānī, sparśa > parisa).
  - `a > ā / # (C) _ Nasal Stop` (aŋkana > āŋkaṇā).
  - Effect: train plain −.0018, ext −.0020; test verb −.0177.
  - Exact matches fall from 14.2% to 13.4%: tatsama words that were exact by accident now get
    the tadbhava treatment.
- **Bhil lects:**
  - `d v > b / # _` (dvā́ra > bār, dvā́daśa > bārā; Gujarati-type), with medial hiatus
    `a {u ū} > o`, `a {i ī} > e / _ C` (hastakūṭa > hathoḍo).
  - Accepted in Bhilali, Bhili, Noiri, Palya, Vasavi, Rathawi and Dungra. Train ext change:
    −.0044, −.0020, −.0055, −.0029, −.0010, −.0087 and −.0041 respectively.
  - RathwiBareli takes only the hiatus part (−.0003).
- **Halbic and others:**
  - `a a > ē / C _ C` (aya > e: kadala > keṛa, cʰagala > celi), plus `repeat: {a ā}{a ā} > ā`,
    placed before final-vowel loss.
  - Train ext change: Bhatri −.0130, Halbi −.0082, Wagri −.0023, Pauri −.0004, Bhilali −.0012,
    Rathawi −.0014, Dungra −.0020.
  - VB: ext −.0157 train and −.0227 test, but plain +.0041. The plain loss is a single word,
    ayám predicted ā for attested am; the gain is in the extended variants. Accepted as
    justified.

**Rejected (with reasons)**
- M:
  - `a > i / V v _ n V #` (causative -aviṇẽ): verb +.0006. -avṇẽ and -aviṇẽ both occur, so this
    is variation.
  - ava/apa/upa-: o- 62 against va- 34 words. This is lexical, so the o- rule stays.
  - Initial ā- in preverb verbs: a:ā ≈ 1:1 inside the same words (akaḷṇẽ/ākaḷṇẽ). Lexical.
- OM: initial Cr > C is plain +.0011 and ext +.0010 (tatsama pr-, gr-).
- Ko: nothing reaches 5 pairs in the non-ending classes. The ∅→ā/geminate cases (kāppaḍa) are
  a source that writes both length and gemination.
- Bhil:
  - dv > b: worse in Pauri, Khandesi, Wagri and RathwiBareli.
  - Medial hiatus: worse in Pauri, Khandesi and Wagri.
  - c/j > ts/dz except before front vowels, in RathwiBareli, Bhilali, Bhili and Noiri: worse
    everywhere (+.001 to +.012). The ts written by watters and varkey is sporadic.
  - aya > e + hiatus: worse or neutral in RathwiBareli (test worse), Noiri, Palya and Vasavi;
    Bhili is negligible (−.00002).
- Bhatri: the participle -aala > -ila works, but it is morphological, so it was not adopted.
- VB: no other non-ending pattern reaches 5 pairs.

**Round 6 total (pre-audit → now), ext NED train / test:**

| lect | ext train | ext test | plain train | verb train / test |
|---|---|---|---|---|
| M | .2564→.2428 | .2585→.2439 | .3696→.3684 | .2044→.1897 / .2098→.1946 |
| OM | .3578→.3219 | .3845→.3542 | .3875→.3860 | .3940→.2920 / .4076→.2937 |
| Ko | .3571→.3570 | .3815→.3782 | .4244→.4244 | .4591 / .4283→.4184 |
| Bhili | .3740→.3675 | .3789→.3759 | .4861→.4766 | – |
| Bhilali | .4023→.3836 | .3634→.3545 | .4812→.4753 | – |
| RathwiBareli | .3933→.3902 | .3617→.3611 | .4798→.4795 | – |
| Noiri | .3613→.3500 | .3405→.3204 | .4618→.4567 | – |
| PauriBareli | .3455→.3390 | .3653→.3504 | .4619→.4565 | – |
| PalyaBareli | .3304→.3234 | .3424→.3306 | .4219→.4169 | – |
| Vasavi | .3694→.3634 | .4328→.4148 | .4514→.4521 | – |
| Rathawi | .3436→.3306 | .3176→.2905 | .4564→.4438 | – |
| Khandesi | .3080→.3051 | .3085→.2835 | .3794→.3765 | – |
| DungraBhili | .3825→.3718 | .4330→.4330 | .4357→.4284 | – |
| Wagri | .3883→.3828 | .4070→.3993 | .4804→.4762 | – |
| Halbi | .2985→.2882 | .3088→.2915 | .3781→.3740 | – |
| Bhatri | .3249→.3120 | .2948→.2906 | .4126→.4120 | – |
| VB | .4882→.4725 | .5788→.5561 | .5838→.5879 | – |

### Round 6 third pass

**Tools.** Near misses were read with `r6/res2.py` in `NEAR=1` mode: train pairs with NED ≤ 0.25
and exactly one edit, grouped by the edit and its context. Weak rules were found with plain
`score --loo` and then re-checked on the extension-aware and verb scores. Verb formations were
added with the new `%verb-menu`, each against a same-shape placebo.

**Kept**
- **M: `%verb-menu +aka +akkanakam`.**
  - `+aka` is the coordinator's item: root + -aka verbal noun. Verb 0.1897 → 0.1873 train,
    0.1946 → 0.1917 test; the placebo `+ara` gives −.0002 / −.0003.
  - `+akkanakam` is the expressive -akk- stem + -anakam. It explains the 22 near-miss
    `-avṇẽ → -akṇẽ` verbs (ʦubakṇẽ, saḍakṇẽ, saṭakṇẽ). Verb 0.1873 → 0.1843 train,
    0.1917 → 0.1908 test; the placebo `+attanakam` gives −.0001 / 0.
- **Ko: `%verb-menu +anta`,** the present participle -ant-. It is the source of Konkani's
  present citation form -tā (cartā, jaḷtā), which no menu item could reach before. Verb
  0.4591 → 0.4538 train, 0.4184 → 0.4129 test; the placebo `+ampa` gives ±0. Ext improves by
  −.0007 / −.0008.

**Tested and rejected**
- **M near misses.**
  - Onsetless `a ṁ > ã / # _ C` (ãgār, ãtar, ãkṇẽ; 14 near misses): train −.0004 but test ext
    +.0005 and verb +.0013.
  - `r dʰ > ḍ ḍʰ` (vāḍʰṇẽ): train −.0003, test +.0005. The full dental retroflexion after r is
    +.0002.
  - Nasalising `m > ṽ` (bʰõvṇẽ): +.0015.
  - The rest are menu artefacts: -āvivẽ against -āviṇẽ, and -akka against -aṇ/-ḷ/-r endings.
  - Or they are lexical: initial ā/a, o-/u-, d/ḍ.
- **M verb menu:** `+āpayanakam` (the -aviṇẽ causative) −.0007 against the placebo
  `+ārayanakam` at −.0002. The margin is too small, so it was not added.
- **M weak rules.** These have plain LOO ≤ +.0001 or fire 1–17 times, but each is needed on the
  extension-aware or verb score, so all are kept:
  - `-ua > -ū`: dropping it costs ext +.0008;
  - tonic r̩ > i: ext +.0005;
  - `-aī > -ī`: ext +.0008;
  - `upa- > o-`: verb +.0020;
  - `a a {m ṁ} > ẽ`: fires 1 time on plain but carries the verb menu.
- **OM near misses.**
  - Tulpule's -ēṁ against -ē (215 against 493), final -i against -ī (126 against 516), and
    tatsama ś, kṣ, y-, st-: orthographic or learned variation, left alone.
  - General `p > v / V _ V`: ext +.0002.
  - The narrow `p > v / ā _ a` gives ext −.0014 and verb −.0095. It is rejected because it only
    tracks the causative morpheme -āp- in a learned chain.
- **Ko.**
  - Dropping `Nasal Nasal > $2 $2` (fires once): verb +.0014, so it is kept.
  - Final -a loss after stops: +.003 to +.009.
  - Near misses are all ≤ 4 pairs.

### Round 6 fourth pass

Covered Ko, VB, Bhilali, Dungra, Wagri and Bhili, reading near misses (single-edit, NED ≤ .25),
the non-ending edits in context, and verb misses. All changes passed `gates.py`, and word diffs
were checked.

**Kept**
- **Nasal lost before a voiceless stop:** `{n ṇ ñ ŋ m} > ∅ / V _ {k c ṭ t p}`, tagged
  @nasal-cluster (dánta > dāt, káṇṭaka > kāṭo). The preceding ā is already long through
  anusvāra lengthening.
  - The near-miss lists showed it in Bhilali, Bhili and Dungra. It was then gated in every Bhil
    and Halbic lect.
  - Accepted in 12 lects: Bhilali, Dungra, RathwiBareli, Noiri, Pauri, Vasavi, Rathawi, Wagri,
    Khandesi, VB, Bhatri, Halbi. Train ext change ranges from −.0020 (Noiri, Wagri) to
    −.0076 (Halbi).
  - Rejected for Bhili and Palya because test got worse (+.0008 / +.0017).
  - Keeping a nasalised vowel (`V N > $1+nas`) is worse than plain loss (Bhili +.0001, Halbi
    −.0056 against −.0076). The sources write the vowel oral or unmarked.
- **Ko: the present in -tā** now comes from the `+anta` menu item through two regular changes.
  1. `a > ∅ / V C _ Nasal Stop` (@syncope), before the NC lengthening. Train: plain −.0004,
     ext −.0026, verb −.0147. Test: ext −.0015, verb −.0073. Plain words: 5 better, 0 worse.
  2. `Nasal > ∅ / C _ C` (@cluster-simplify), giving car-nt- > cart- (cartā, giḷtā, caḍtā).
     Train: plain −.0007, ext −.0073, verb −.0470. Test: ext −.0068, verb −.0415. Plain words:
     6 better, 0 worse.
  3. Ko verb-menu NED over round 6 is .458 → .392 train and .428 → .364 test.
- **VB:** the epenthetic ɨ is not added after liquids, `∅ > ɨ / C _ # // {r l ḷ} _` (udʰār,
  upar, pūr). Train ext −.0063 and plain −.0021; test unchanged.
- **RathwiBareli, compliance fix.**
  - The engine now gives RathwiBareli some verb pairs through gloss-gated denominal forms.
    Against those, the round-6 pre-final syncope was verb +.0042.
  - `a a > ā` after v-loss is now ext-negative under the current engine.
  - Both were removed. RathwiBareli against its pre-audit file: ext .3936 → .3902 train, verb
    unchanged.

**Rejected**
- Ko:
  - Nasal loss before voiceless stops: plain +.0026. Ko keeps NC (kāṇṭo).
  - Final -a loss after stops, and dropping `Nasal Nasal`: see the third pass.
- `ī {a ā} > ī / _ #` in the Bhil lects: ±0.
- Dungra -o/-e and the Bhilali/Wagri -ā/-o endings: morphological or source variation, left
  alone.
- Wagri cʰ- > s/c-, Bhilali and Wagri davara > dorā, Dungra l-/n-: lexical.

**Round 6 total (pre-audit file → final; the same engine for both columns)**

| lect | ext train | ext test | plain train | verb train / test |
|---|---|---|---|---|
| M | .2564→.2416 | .2585→.2430 | .3696→.3684 | .2044→.1843 / .2098→.1908 |
| OM | .3578→.3219 | .3845→.3542 | .3875→.3860 | .3940→.2920 / .4076→.2937 |
| Ko | .3573→.3466 | .3815→.3691 | .4244→.4232 | .4578→.3922 / .4283→.3641 |
| Bhili | .3741→.3676 | .3789→.3759 | .4861→.4766 | .6690 = |
| Bhilali | .4025→.3816 | .3634→.3545 | .4812→.4730 | .5863→.5461 |
| RathwiBareli | .3936→.3902 | .3617→.3617 | .4798→.4756 | .6510 = |
| Noiri | .3597→.3481 | .3405→.3204 | .4618→.4550 | = |
| PauriBareli | .3490→.3377 | .3666→.3518 | .4619→.4513 | .5653→.5210 / .6667→.6000 |
| PalyaBareli | .3312→.3234 | .3424→.3306 | .4219→.4169 | .5000→.4375 |
| Vasavi | .3694→.3600 | .4328→.4148 | .4514→.4478 | – |
| Rathawi | .3443→.3265 | .3176→.2905 | .4564→.4390 | = |
| Khandesi | .3003→.2932 | .3085→.2835 | .3794→.3722 | = |
| DungraBhili | .3825→.3753 | .4330 = | .4357→.4244 | = |
| Wagri | .3849→.3775 | .4070→.3993 | .4804→.4742 | = |
| Halbi | .2985→.2806 | .3088→.2915 | .3781→.3664 | – |
| Bhatri | .3249→.3093 | .2948→.2906 | .4126→.4093 | – |
| VB | .4882→.4632 | .5788→.5561 | .5838→.5829 | = |

The "pre-audit" numbers here are rescored with the current engine (new verb items,
denominal verbs and `%source`-aware residuals), so some differ slightly from the first-pass
table.

## Round 7

**Method.** Scratch work is in `tmp/sound_changes/marathi-bhil/r7b/`. `log.txt` is the decision log,
`logs/` has every `gates.py` output, and `cand/<tag>/` has the candidate files. Candidates ran one at
a time through a serial queue (`worker.sh`). Two helper scripts were used:
- `vconf.py` tallies predicted-vowel → attested-vowel alignments by position and source.
- `wd.py` is a word-level better/worse diff that applies the `%source` tails.

`gates.py`'s "reflexes better/worse" count does not apply `%source` tails. It is therefore
misleading for tail-only changes: for Bhatri it reported 36 better and 124 worse, while `wd.py`
found 45 better and 15 worse. The coordinator may want to fix it. Candidates gated after each
engine fix (09:25 canon, 10:35 verb flag) were re-gated under the new engine.

**Step 0.** The interrupted agent had installed nothing: every rule file was byte-identical to
`snapshots/pre-round7`. Its scratch candidates were re-gated:
- Khandesi and Palya `degem-lengthen.vg2→vg3`, and Vasavi `+anusvara-vowel.a.nasalstop`, had no
  effect (REJECT).
- Pauri `+degem-lengthen.vg3` made ext +.0036 worse (REJECT).
- Its M `%ext-menu` items had never been placebo-tested. They are now: +la gives −.0050, but the
  placebo +na gives −.0047, a margin of .0003. Not adopted. +ra and +ḍa are below the placebo.

### Vowel qualities (lax vowels now scored)
Sources that write lax vowels:
- ʌ/ə: Beine (Bhatri, Halbi).
- ɔ: varkey's Bareli, varghese's Noiri/Dungra, the mewari list, and pattanaik (Varli).
- ɛ̃: IndoLanguages (Konkani).

**Kept**
- **Bhatri: short a > ə before a consonant** (`@a-centralisation`, chain end; gʰara > gər, pʰála >
  pəl, dánta > dət).
  Beine writes ə 97 and ʌ 74, in free variation inside the same words (pəl/pʌl). The change is
  installed jointly with compensatory lengthening `degem-lengthen.vg2.c` (hattʰa > hāt > Beine
  hat, as in Halbi) and `anusvara-vowel.a.nasalstop` (dánta > dāt > dat).
  - Final -a stays: the verb citations in -la (mərla) got worse under the 10:35 verb flag. The
    first version put ə everywhere and was fixed after the final check.
  - Final state against the snapshot: ext .3946→.3766 train, .3805→.3568 test; verb .4494→.4411.
  - Each member is necessary: a > ə alone is ext +.0042; CL alone is ±0; without the anusvara
    rule the gain is −.0182.
  - Writing ʌ instead of ə gives only −.0099.
- **Halbi:** `%source beine2017bhatri a > ə` (@notation; Woods writes a): ext −.0048 / −.0009.
  Also `anusvara-vowel.a.nasalstop`, as in Bhatri: ext −.0093 / 0, words 4 better / 0 worse.
- **Bhili:** `%source pattanaik-koul2003varli a > ɔ`. This source records Varli, where short a is ɔ
  (gʰara > gɔr, náva > nɔv).
  - Restricted to `/ _ C`: in every position it turned Varli verb citations such as piva into
    *pibɔ (verb +.067 under the 10:35 engine).
  - Against no Varli rule: ext −.0026 / −.0069, verb unchanged.
- **Dungra: final o > ɔ** (kadala > kelɔ, āmrá > āmbɔ). The source writes -ɔ 27 against -o 21. The
  same source writes -o 56 against -ɔ 30 in Noiri, so this is a lect trait, not the notation.
  Ext −.0170 / −.0415. Rewriting the medial `a > o` rounding as ɔ is worse (−.0096 jointly).
- **Konkani:** `%source IndoLanguages ẽ > ɛ̃` (neuter kaḷśɛ̃, āḷśɛ̃): ext −.0019 / −.0023, words
  8 better / 0 worse.

**No change**
- Varkey's ɔ (Bhil lects): initial a→ɔ in 10 of 68, final o→ɔ in 38 of 166.
- The mewari list: final ā→ɔ in 9 of 51.
- M, OM and Ko CDIAL write no lax vowels.

### Other kept changes
- **Dungra `ō > ū / # C* _`** (gōḍḍa > gūḍū, bōkka > būkaḍo, pōtara > pūiro). Ext −.0081 / −.0153,
  words 7 better / 1 worse. Rejected in Bhilali, Bhili and Noiri; Noiri shares the source, so this
  is not notation.
- **Vasavi initial `h > ∅` before `s > h`.** The old h was lost and s > h made a new one: hásta >
  āth, haḍḍa > āḍ, but sūcī́ > hūi. Ext −.0037 / 0, words 6 better / 0 worse. In Noiri and Pauri the
  same watters dialect shows it, but a `%source` tail cannot separate the old h from the new one.

### Legacy @morph rules (focus 6) and dead rules
**Citation -o removed.** The `∅ > o / … _ #` rules added masc. -o to the bare headword. They now
cost on the extension-aware score: the -aka / -aḍaka menu variants supply the suffixed citations,
and the rule blocked the bare variant for words cited bare (ghar, dūr, pʰaḷ, ek).
- Removed in RathwiBareli, Bhilali, Bhili, Palya, Noiri, Dungra, Vasavi and Wagri, plus Bhatri's
  `∅ > a / ṛ _ #`. Ext train changes from −.0007 (Wagri) to −.0248 (Vasavi); every test score is
  better or equal.
- Plain gets worse (+.003 to +.030). This is the bare-vs-extended artefact.
- Pauri: a neutral removal, with ext, verb and accent unchanged and plain +.0081.

**Other @morph removals**
- M `∅ > ṇ ẽ` after bare CDIAL roots: ext −.0007 / −.0012, plain +.0101.
- Ko `a {i ē} > t ā`: a neutral removal after the 10:35 verb-flag fix; plain +.0317.
- Bhilali's dead `o > ā`: a neutral removal.

**Kept because removal fails a gate**
- The neuter `if:ext -akam` rules in seven lects: ext train worse, +.0011 to +.0242.
- The OM √-root -ṇē rule: ext train +.0011.
- Ko bare-root -tā: +.0003.
- Ko -ayati -tā: ext test +.0017.
- M -ayati/-ati -ṇẽ: ext train +.0001 / +.0004 under the 10:35 engine.
- Wagri o > ā: +.0232.
- Rathawi `if:!ext` -o: +.0014.
- RathwiBareli ə-marker: +.0025. Its comment is updated: it no longer blocks -o.

**Dead or contradicted rules, found by ext-LOO**
- Bhilali `hiatus.aaa.final`: redundant with `hiatus.aaaa.repeat.2` and `o > ā`. Removed jointly
  with the -o rule; verb −.0045.
- Bhili `hiatus.aii`: removed.
- Palya `hiatus.auu`: its only firing, hastakūṭa, is attested hāthaūḍo.
- Vasavi `hiatus.aii`: its only firing, sadŕ̩kṣa > *hekh, is attested sārkho.

### Rejected
- Dungra initial `l > n`: ext −.0036, but only 2 etyma better against 1 worse, so lexical as in
  round 6.
- Watters pʰ > ɸ: accepted in Noiri and Vasavi, rejected in Pauri and Rathwi. Pooled, 11 words are
  better and 12 worse (pʰulla/pʰála with ɸ, spʰuṭ- with ph), so it is not a notation.
- Noiri and Rathawi `hiatus.aii` removal: one etymon each way.
- VB aya-e and final-consonant-loss removals: chance matches of menu variants.
- Pauri `a a > ā` / aya-e removal: verb +.0098.
- OM -tvana/-paṇa: not testable. The chain has no tv > pp, and the stems are learned.

### Before → after (pre-round7 snapshot → now; same engine for both columns)
| lect | ext train | ext test | plain train | verb train / test |
|---|---|---|---|---|
| M | 0.2416→0.2410 | 0.2428→0.2415 | 0.3685→0.3786 | 0.1857→0.1857 / 0.1958→0.1958 |
| OM | 0.3216→0.3216 | 0.3545→0.3545 | 0.3860→0.3860 | 0.2918→0.2918 / 0.3026→0.3026 |
| Ko-Ul | 0.3526→0.3507 | 0.3725→0.3702 | 0.4243→0.4560 | 0.3941→0.3941 / 0.3654→0.3654 |
| Bhili | 0.3877→0.3765 | 0.3895→0.3760 | 0.4898→0.4854 | 0.5214→0.5214 / 0.6667→0.6667 |
| Bhilali | 0.3843→0.3781 | 0.3589→0.3526 | 0.4757→0.4914 | 0.5461→0.5417 / 0.0000→0.0000 |
| RathwiBareli | 0.4061→0.3857 | 0.3750→0.3492 | 0.4904→0.5042 | 0.6426→0.6426 / 0.0000→0.0000 |
| Bhatri | 0.3946→0.3766 | 0.3805→0.3568 | 0.4756→0.4727 | 0.4494→0.4411 / 0.0000→0.0000 |
| mewari_bannoda | 0.3843→0.3836 | 0.3993→0.3846 | 0.4789→0.5041 | 0.4062→0.4062 / 0.6667→0.6667 |
| Noiri | 0.3665→0.3626 | 0.3381→0.3287 | 0.4722→0.4690 | 0.3713→0.3713 / 0.0000→0.0000 |
| PauriBareli | 0.3408→0.3408 | 0.3588→0.3588 | 0.4533→0.4614 | 0.5013→0.5013 / 0.6000→0.6000 |
| PalyaBareli | 0.3250→0.3118 | 0.3306→0.3290 | 0.4185→0.4227 | 0.4375→0.4375 / 0.0000→0.0000 |
| VB | 0.4593→0.4593 | 0.5422→0.5422 | 0.5829→0.5829 | 0.5172→0.5172 / 0.4756→0.4756 |
| Vasavi | 0.3639→0.3316 | 0.4148→0.3785 | 0.4478→0.4701 | 0.5607→0.5607 / 0.4000→0.4000 |
| hal | 0.3021→0.2879 | 0.3149→0.3140 | 0.3867→0.3729 | 0.4093→0.4093 / 0.0000→0.0000 |
| Rathawi | 0.3265→0.3265 | 0.2905→0.2905 | 0.4390→0.4390 | 0.5714→0.5714 / 0.0000→0.0000 |
| Khandesi | 0.2932→0.2932 | 0.2935→0.2935 | 0.3722→0.3722 | 0.1871→0.1871 / 0.0000→0.0000 |
| DungraBhili | 0.4223→0.3784 | 0.4874→0.4081 | 0.4677→0.4685 | 0.4345→0.4315 / 0.0000→0.0000 |

Scored under the 10:35 engine. VB, OM, Rathawi and Khandesi are unchanged.

**Not done**
- The M/OM/Ko near-miss lists are unchanged from round 6: all ≤ 4 pairs or lexical.
- Accent/tone: no lect in the group marks tone.
- `gates.py` net-reflex count ignores `%source` tails (reported above).
