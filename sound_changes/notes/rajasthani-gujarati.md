# Rajasthani–Gujarati group: OIA → Gujarati, Old Western Rajasthani, Koli, Gujari, Rajasthani dialects

Rule files: `sound_changes/rules/OIA-{G,OG,OMarw,WK,Goj,Nimadi,Marw,marwari_bagra,mewari_dholpura,Rj,mewari_basad,had,hadothi_kelwada,dhundari_badagaon,dhundari_bamore,bagri_fatehabad,mewati_akera,mewati_jhambaus}.sc`.
Frontiers: `sound_changes/results/OIA-<lect>.hand.frontier.csv`.
Scratch tools in `tmp/sound_changes/rajasthani-gujarati/`: `try.py` (score one edit), `swap.py`
(order swaps), `reorder.py` (best position of every rule), `cross.py` / `ins.py` (test every group
rule in every lect), `prune.py` (iterative LOO ≤ 0 removal), `diag.py` (per-source dialect
diagnostics → `diag.txt`), `look.py` / `single.py` (error listings), `template.sc` (survey
template), `chron_final.txt` (all swap numbers quoted in §3).

## 1. Results

<!--TABLE-START-->
| lect | name | #etyma | #rules | train NED null → final | test NED null → final | test exact | auto test NED (its #rules) | hand test NED at that #rules | %source notation blocks for |
|---|---|---|---|---|---|---|---|---|---|
| G | Gujarati | 4549 | 49 | 0.6511 → 0.3772 | 0.6476 → 0.3734 | 17.0% | 0.5003 (@13) | 0.4596 |  |
| OG | Old Gujarati | 743 | 35 | 0.5727 → 0.3172 | 0.5756 → 0.3125 | 19.6% | 0.4159 (@17) | 0.3718 |  |
| OMarw | Old Marwari | 588 | 42 | 0.5897 → 0.3280 | 0.5659 → 0.3186 | 18.8% | 0.3938 (@22) | 0.3452 |  |
| Marw | Marwari | 445 | 42 | 0.6787 → 0.4560 | 0.6748 → 0.4726 | 6.9% | 0.5748 (@18) | 0.5002 | blocks: marwari mewari |
| mewari_basad | Malvi | 334 | 36 | 0.6441 → 0.4897 | 0.6588 → 0.5369 | 1.9% | 0.6060 (@17) | 0.5446 | blocks: mewari varghese-john-samuel2009malvi |
| mewari_dholpura | Mewari | 282 | 45 | 0.6810 → 0.4873 | 0.6926 → 0.5339 | 2.1% | 0.6414 (@28) | 0.5514 | blocks: mewari |
| Goj | Gujari | 262 | 40 | 0.6933 → 0.4826 | 0.6615 → 0.4679 | 6.9% | 0.5732 (@35) | 0.4680 | blocks: hugoniot-polster-ahmad-rajan2023easterngujari varghesekumar2015noira |
| had | Hadauti | 258 | 41 | 0.6726 → 0.4832 | 0.6529 → 0.5204 | 4.8% | 0.6113 (@21) | 0.5286 | blocks: hadothi mewari |
| dhundari_badagaon | Shekhawati | 246 | 37 | 0.6499 → 0.4883 | 0.6512 → 0.4738 | 8.3% | 0.5746 (@23) | 0.4762 | blocks: dhundari marwari |
| Nimadi | Nimadi | 231 | 33 | 0.6346 → 0.4523 | 0.6200 → 0.4601 | 6.5% | 0.5842 (@24) | 0.4746 | blocks: varkey-vunnamatla2018bareli vunnamatla-john-samuvel2012nimadi |
| bagri_fatehabad | Bagri | 230 | 40 | 0.6755 → 0.4749 | 0.6390 → 0.4505 | 10.9% | 0.5521 (@22) | 0.4790 | blocks: bagri |
| dhundari_bamore | Dhundari | 218 | 40 | 0.6601 → 0.4625 | 0.6537 → 0.4596 | 8.0% | 0.5778 (@22) | 0.4759 | blocks: dhundari |
| mewati_akera | Mewati | 213 | 33 | 0.6475 → 0.4422 | 0.6337 → 0.4689 | 10.5% | 0.5844 (@25) | 0.4996 | blocks: mewati |
| marwari_bagra | Godwari | 194 | 38 | 0.6713 → 0.4627 | 0.6749 → 0.5186 | 5.1% | 0.6434 (@21) | 0.5478 | blocks: marwari |
| hadothi_kelwada | Shahawadi | 169 | 36 | 0.6495 → 0.4168 | 0.6170 → 0.4474 | 14.6% | 0.5979 (@14) | 0.4822 |  |
| mewati_jhambaus | Ahirwadi | 167 | 32 | 0.6513 → 0.3784 | 0.6257 → 0.4265 | 9.8% | 0.6095 (@12) | 0.4702 | blocks: mewati |
| Rj | Rajasthani | 165 | 34 | 0.6571 → 0.4979 | 0.6212 → 0.4756 | 14.0% | 0.5981 (@16) | 0.5283 | blocks: mewari |
| WK | Wadiyara Koli | 112 | 37 | 0.7578 → 0.4911 | 0.7435 → 0.5883 | 2.4% | 0.6544 (@21) | 0.5855 |  |
<!--TABLE-END-->

(Round-1 numbers. Round-2 scores, after the vowel/accent round, are in §7.1.)

"auto" = the greedy baseline's test NED at its last rule (the induction stopped at 12–35 rules);
"hand at that #rules" reads our backward-elimination frontier at the same count. Null NED here
(for lects with `%source` blocks) already includes the block notation, so it differs slightly from
the auto files' zero-rule point; the auto baseline also scored the digraph sources raw. No lect was
skipped. No source is excluded: the three digraph-notation sources (Bareli in Nimadi; Hugoniot
et al. and Noira in Gujari) are scored through `%source` notation blocks (§2). G, OG and OMarw are
pure CDIAL; the Rajasthani lects are survey word lists (plus CDIAL in Marw). The small lects'
test sets are 40–120 pairs, so their test NED moves by ±0.01 with single words; train NED is the
more reliable number there.

## 2. Base chain (order as in the files)

### 2a. MIA block (all 18 files, same relative order; each lect keeps the subset LOO supports)
```
V > $1-acc                                    @acc-loss
{ai au} > {ē ō}                               @ai-au
r̩ > i          (G, WK: r̩ > r i / # _ ; r̩ > a)  @r-vocalic / @r-vocalic-ri
Sib > s                                       @sibilant-s
{t tʰ d dʰ} y > {c c j j} {c cʰ j jʰ}@1        @palatal-Cy
h y > j jʰ     (6 files)                      @palatal-Cy
{k t p} s > {k c c} {kʰ cʰ cʰ}@1               @ks
s k > k k      (OMarw, WK, Goj, Mewari, Malvi) @s-stop-aspirate (unaspirated exception)
s Stop > $2-asp $2+asp                        @s-stop-aspirate
s Son > s s                                   @assim-sonorant
Nasal Stop > ṁ $2   (not Gujari)              @nasal-cluster
r {t tʰ (d dʰ)} > ṭṭ(ʰ) (ḍḍ(ʰ))   (§4)        @x-rg-rt-retroflex @assim-r
{r ḣ} C > $2-asp $2                           @assim-r
d v > b b      (§4)                           @x-rg-dv-b
j ñ > ṇ ṇ (G, OMarw, Marw); t m > p p (15 files)   @assim-sonorant ; @assim-stop
y y > j j (G, OG, OMarw)                      @palatal-Cy
v v > b b (§4)                                @v-b
m {r l} > m b (§4)                            @x-rg-mr-mb
Stop {y v l Nasal} > $1-asp $1                @assim-sonorant
{ŋ ñ ṇ n m l v} y > $1 $1                     @assim-sonorant
Stop r > $1-asp $1   (G, OG: NDStop r — t/tʰ + r kept)       @assim-sonorant
Stop Stop > $2-asp $2                         @assim-stop
C C > $2 / # _       (G: C NR — tr- kept)     @initial-degem
n > ṇ / V _ V                                 @n-retroflex
{kʰ gʰ tʰ dʰ pʰ bʰ} > h / V _ V               @int-aspirate-h
{ṭ ṭʰ} > {ḍ ḍʰ} / V _ V                       @int-retroflex
m > v / V _ V    (OMarw, Marw, Hadauti, Shahawadi)           @m-v
p > v / V _ V                                 @int-p-v
{k g c j t d y} > ∅ / V _ V                   @int-stop-loss
{a u} v {a ā} > o / # _  (surveys: any position)             @aya-e
[verb-citation @morph, below]
{s t (d) n m (r) ḣ} > ∅ / _ #                  @final-cons-loss
```

### 2b. NIA block (shared order)
```
s > h / # _ V | V _ V            @s-h             WK, Mewari (dholpura), Godwari
ḍ > ṛ / V _ V   ({ḍ ḍʰ} in WK, Goj)   @flap        G, WK, Goj only (surveys and OWR write ḍ)
l > ḷ / V _ V                    @l-retroflex     all but OG, OMarw, Gujari, Malvi, Shahawadi, Ahirwadi
{c cʰ j jʰ} > {s s z z}          @x-rg-deaffrication  WK (after s > h and degemination); Godwari: initial c(ʰ) > s
[verb citation, G/WK]            @morph
a h {i ī} > e / _ C              @hiatus @h-loss
i {a ā} > ī / _ #                @hiatus          -ika/-ikā > -ī
a a > o / _ #  (OG: > a u)       @hiatus @morph   -aka > -o/-ɔ (OG -aü)
{i ī} {a ā} > e / _ C  (G)       @hiatus
{a ā} {a ā} > ā / _ !# ; a {o u} > o ; a > ∅ / V _     @hiatus
h > ∅ / V _ V                    @h-loss          Rajasthani surveys, Gujari (not Mewati, not G)
v > ∅ / V _ V                    @v-loss          Shahawadi, both Dhundari points, Gujari
ṇ ṇ > n n                        @n-dental
V G $2 > $1+long $3 / _ V #      @degem-lengthen  (not Gujari, Malvi, Shekhawati; OG: also # (C) _ ; OMarw: {i u} also # (C) _)
repeat: G $1 > $2                @degem-plain
V ṁ > $1+long+nas / _ C {a i u} | _ C #   @anusvara-vowel  (OG/OMarw keep the ṁ letter)
V ṁ > $1+nas                     @anusvara-vowel
{a i u} > ∅ / V C _ C V          @syncope         G, WK only — before apocope
{a ā i u} > ∅ / _ #              @final-vowel-loss  (OG/OMarw: only ā > a / _ #; pruned in Marw, Mewari,
                                                    Malvi, Nimadi, Dhundari: there final -i/-u surface as
                                                    the -ī/-ū the surveys cite, a-stems go to -o)
a > o / C _ #                    @morph           survey a-stems cited in masc. -o
{gʰ …} > {g …} / !# _  (both series in Godwari, Rj)   @deaspiration  Mewari, Godwari, Rj, Malvi, Bagri, Shahawadi; WK word-initial
v > b / # _   (Ahirwadi, Bagri: everywhere)   @v-b
y > j / # _                      @y-j
{ē ō} > {e o}                    @notation
{ū̃ ū} > {ũ u}  (G)              @vowel-shift
ṇ > n                            @n-dental        Malvi, Ahirwadi, Shahawadi, Rj (Hindi-type reversal)
a > ∅ / # _ C V                  @x-rg-aphaeresis  Marw, Mewari, Rj, Hadauti, Dhundari, Bagri, Gujari, Ahirwadi
j jʰ > y ; cʰ > c  (Gujari)      @x-rg-j-y ; @deaspiration
```

Verb-citation `@morph` (end of MIA block or start of NIA, before hiatus contraction): G and WK
`{a ā ē ō} {i ē} > v ũ / _ #` (-ati/-atē/-āti/-ēti/-ōti → infinitive -vũ; ~670 G etyma — the most
valuable non-trivial G rule); survey lects `{a ā ē ō} {i ē} > o / _ #` (questionnaire glosses
"X!, he Xed" are cited in the masc. past -o); Ahirwadi `> ā`; OG/OMarw keep 3sg -aï (`ē > i / a t _ #`
merges middle -atē into it; `a a > a / _ {i ē}` contracts -ayati).

`%source` notation blocks (not chain rules): all Rajasthani survey sources get
`{ã ā̃ ĩ ī̃ ũ ū̃ ẽ õ} Stop > $1-nas {homorganic nasal}@2 $2` (they write pāñc, kāṇṭo) and
`{i u} > {ī ū}` (they write ī ū for short i u; Nimadi `u > ū`, `ī > i`); the digraph sources get
`{kʰ gʰ cʰ jʰ ṭʰ ḍʰ tʰ dʰ pʰ bʰ} > {k … b} h`. Moving the survey notation out of the chain into
blocks left train NED unchanged (±0.001) and lets CDIAL Marw. items be scored in house notation.

Deviations from the shared order: WK has the anusvāra rules before l > ḷ (−0.0068 there; 132
pairs) and c > s after degemination, after s > h (§3); Goj has h > ∅ before a {o u} > o.

## 3. Relative chronology (train ΔNED when the two rules are swapped; + = the swap hurts)

| earlier → later | ΔNED of the swap | implication |
|---|---|---|
| l > ḷ → degemination | G +0.017, WK +0.017, Marw +0.020, Mewari +0.028, Godwari +0.019, Hadauti +0.022, Dhundari +0.024, Bagri +0.023, Rj +0.017, Nimadi +0.010 | only single -l- becomes ḷ, -ll- stays l: the ḷ/l split is older than degemination wherever ḷ exists |
| flap → degemination | G +0.018, WK +0.015, Goj +0.005; the reorder search moved the flap from the Hindi-file slot after degemination to the head of the NIA block (−0.0084 on G) | MIA single -ḍ- (< -ṭ-) > ṛ while -ḍḍ- is still geminate: G ghaṛɔ vs aḍvũ |
| ṇṇ > nn → degemination | G +0.009, OMarw +0.019, WK +0.018, Marw +0.017, Mewari +0.024, Hadauti +0.019, Dhundari +0.021, Bagri +0.019, Nimadi +0.007 | dental n of kān (< karṇa) vs ṇ of guṇ is fixed while the geminate exists |
| s > h → degemination | WK +0.032, Godwari +0.033, Mewari +0.031 | -ss- survives as s |
| s > h → c > s | WK +0.023, Godwari +0.004 | counter-feeding: s < c (saṛāvū̃ < caḍʰati) never becomes h, so deaffrication is younger than s > h in both Koli and Godwari; in Koli it also follows degemination (cc > c > s; moving it there: −0.0064) |
| compensatory lengthening → apocope | G +0.011, WK +0.034, Godwari +0.056, Hadauti +0.044, Shahawadi +0.025, Bagri +0.023, Rj +0.013 | hasta > hattʰa > hātʰ, conditioned by the later-lost final vowel; OG/OMarw show the lengthening with final vowels still written → lengthening pre-OWR, apocope post-OWR |
| syncope → apocope | G +0.0019, WK +0.019 | Gujarati/Koli syncope is computed on the pre-apocope word |
| aspirate > h → h-loss | Godwari +0.020, Gujari +0.011, Hadauti +0.007, Dhundari +0.003–0.007, Bagri +0.006, Rj +0.005 | feeding: Rajasthani loses the h continuing MIA medial aspirates |
| aspirate > h → non-initial deaspiration | Godwari +0.004, Malvi +0.008, Bagri +0.010 | the Rajasthani medial deaspiration is a late NIA change |
| rt > ṭṭ → general r + C assimilation | Godwari +0.003, Mewari +0.0025, Malvi +0.0035, Shahawadi +0.005, Hadauti +0.002, Bagri +0.002, Rj +0.002 | bleeding |
| dv > bb → C + v assimilation | Godwari +0.0055, Dhundari +0.004, Mewari +0.0024, OMarw +0.0019, Goj +0.0015, G +0.0012 | bleeding |
| int. stop loss → -aka contraction | G +0.050, WK +0.051, … | feeding (trivial) |

Not order-sensitive (swap ≈ 0): n > ṇ vs the MIA assimilations; aspirate > h vs stop loss;
r + C vs C + y/v; m > v vs nasal clusters. The per-rule best-position search (`reorder.py`)
found no further move worth > 0.002 in any file.

## 4. Shared vs lect-specific changes; sub-grouping

Shared by practically all files: the MIA block; n > ṇ (lost again in Gujari; ṇ > n in Malvi,
Ahirwadi, Shahawadi, Rj); ṇṇ > nn; l > ḷ; degemination with penultimate lengthening; -aka > -o/-ɔ;
ava-/apa- > o-; lengthening before anusvāra unless the next syllable is heavy; apocope (not in
OG/OMarw); **rt > ṭṭ** (data: G 52/85, Marw 7/10, ≈100 % at every survey point; the rule is kept
where LOO > 0: 11 files); **dv > bb** (bār 'door', bār- '12'; 9 files); **mr/ml > mb** (āmra > ām(b)o; G, WK, Gujari,
Nimadi, Marw, Godwari, Mewari, Malvi, Hadauti). One MIA isogloss splits the group: **-vv- > -bb-**
(sarva > savva > sab) helps Malvi, Nimadi, Shahawadi, Dhundari (Bamore), Mewati, Gujari and OMarw
(−0.0004 to −0.0016) but hurts G, OG, Marw, Mewari, Godwari, Rj (+0.0008 to +0.0033), i.e. the
Gujarati–Marwari–Mewari west keeps v (G sau/sav-) while the eastern and northern fringe has the
Hindi-type bb.

Tag presence per file (after LOO pruning; absence can mean "not supported by this lect's data" as
well as "did not happen"; files ordered roughly west → east/north):

| tag | G | OG | WK | OMw | Mw | God | Mew | Rj | Mlv | Nim | Had | Shw | Shk | Dhu | Bag | Mwt | Ahi | Goj |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `@ai-au` | x | x | x | x | x |  |  |  | x |  |  |  |  |  |  |  |  |  |
| `@anusvara-vowel` | x | x | x | x | x | x | x | x |  | x | x | x | x | x | x | x | x |  |
| `@assim-r` | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| `@assim-sonorant` | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| `@assim-stop` | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| `@aya-e` | x | x |  |  | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| `@deaspiration` |  |  | x |  |  | x | x | x | x |  |  | x |  |  | x |  |  | x |
| `@degem-lengthen` | x | x | x | x | x | x | x | x |  | x | x | x |  | x | x | x | x |  |
| `@degem-plain` | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| `@final-cons-loss` | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| `@final-vowel-loss` | x | x | x | x |  | x |  | x |  |  | x | x | x |  | x | x | x | x |
| `@flap` | x |  | x |  |  |  |  |  |  |  |  |  |  |  |  |  |  | x |
| `@h-loss` | x |  |  |  | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| `@hiatus` | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| `@initial-degem` | x |  |  |  | x |  | x | x |  |  | x |  | x |  | x | x |  |  |
| `@int-aspirate-h` | x | x |  | x |  | x |  | x | x | x | x |  | x | x | x | x | x | x |
| `@int-p-v` | x | x |  | x | x | x | x |  | x |  |  |  | x | x | x |  | x | x |
| `@int-retroflex` | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| `@int-stop-loss` | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| `@ks` | x | x | x | x | x | x | x |  | x | x | x | x | x | x | x | x | x | x |
| `@l-retroflex` | x |  | x |  | x | x | x | x |  | x | x |  | x | x | x | x |  |  |
| `@m-v` |  |  |  | x | x |  |  |  |  |  | x | x |  |  |  |  |  |  |
| `@n-dental` | x |  | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| `@n-retroflex` | x | x | x | x | x | x | x |  |  | x | x |  | x | x | x |  |  |  |
| `@nasal-cluster` | x | x | x | x | x |  | x | x |  | x | x | x | x | x | x | x | x |  |
| `@palatal-` | x | x |  | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| `@r-vocalic` | x | x | x | x | x |  | x | x |  | x |  |  | x |  |  | x |  | x |
| `@r-vocalic-ri` | x |  | x |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `@s-h` |  |  | x |  |  | x | x |  |  |  |  |  |  |  |  |  |  |  |
| `@s-stop-aspirate` | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| `@sibilant-s` | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
| `@syncope` | x |  | x |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `@v-b` |  |  |  | x | x |  | x | x | x | x | x | x | x | x | x | x | x | x |
| `@v-loss` |  |  |  |  |  |  |  |  |  |  |  | x | x | x |  |  |  | x |
| `@vowel-shift` | x |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `@x-rg-aphaeresis` |  |  |  |  | x |  | x | x |  |  | x |  |  | x | x |  | x | x |
| `@x-rg-deaffrication` |  |  | x |  |  | x |  |  |  |  |  |  |  |  |  |  |  |  |
| `@x-rg-dv-b` | x | x |  | x |  | x | x |  | x |  |  | x |  | x |  |  |  | x |
| `@x-rg-j-y` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | x |
| `@x-rg-mr-mb` | x |  | x |  | x | x | x |  | x | x | x |  |  |  |  |  |  | x |
| `@x-rg-rt-retroflex` | x |  |  |  | x | x | x | x | x |  | x | x |  | x | x |  |  | x |
| `@y-j` | x | x |  | x | x | x | x | x | x | x | x | x | x | x | x | x | x |  |

Per-source diagnostics (`diag.txt`; counts of gold reflexes in the given OIA context):

| lect / source | s- → h | VlV → ḷ | VnV → ṇ | c- → s | j- → y | final -o / -ā |
|---|---|---|---|---|---|---|
| G / CDIAL | 2/452 | 552/669 | 411/430 | 2/374 | 0/103 | 1005/68 |
| OG / CDIAL | 0/96 | 4/81 | 76/78 | 0/29 | 0/12 | 5/12 |
| OMarw / CDIAL | 0/58 | 17/77 | 35/38 | 0/26 | 0/12 | 70/19 |
| WK / zubair | 9/19 | 7/12 | 8/8 | 8/8 | 0/6 | 28/11 |
| Goj / rensch | 1/59 | 9/40 | 9/22 | 1/31 | **8/8** | 122/48 |
| Goj / hugoniot | 0/10 | 3/10 | 1/4 | 1/8 | 0/6 | 28/27 |
| Nimadi / vunnamatla | 0/48 | 52/72 | 20/33 | 0/33 | 0/13 | 155/43 |
| Nimadi / bareli | 0/19 | 22/32 | 6/12 | 0/17 | 0/3 | 75/34 |
| Marw / marwari | 21/45 | 45/56 | 27/37 | 1/36 | 0/17 | 201/44 |
| Marw / mewari | 16/31 | 24/30 | 7/12 | 5/15 | 0/3 | 105/40 |
| Godwari / marwari | **28/32** | 25/36 | 9/13 | 10/25 | 0/9 | 132/26 |
| Mewari (dholpura) | **64/110** | 77/97 | 25/41 | 17/53 | 0/23 | 239/172 |
| Rj / mewari | 7/16 | 11/17 | 3/9 | 0/13 | 0/5 | 14/28 |
| Malvi / mewari (Basad) | **27/30** | 1/9 | 12/21 | 5/25 | 0/11 | 121/47 |
| Malvi / varghese | 26/84 | 69/119 | 23/44 | 14/62 | 0/22 | 228/149 |
| Hadauti / hadothi | 2/51 | 56/77 | 18/29 | 0/45 | 0/14 | 172/149 |
| Shahawadi / hadothi | 0/13 | 2/11 | 3/8 | 0/10 | 0/5 | 36/30 |
| Shekhawati / dhundari | 0/29 | 21/28 | 11/20 | 0/24 | 0/8 | 79/40 |
| Dhundari / dhundari | 0/39 | 42/56 | 15/24 | 0/32 | 0/10 | 129/68 |
| Bagri / bagri | 0/34 | 33/46 | 17/24 | 0/26 | 0/11 | 107/83 |
| Mewati (akera) | 0/39 | 35/50 | 12/18 | 0/37 | 0/7 | 66/85 |
| Ahirwadi (jhambaus) | 0/14 | 9/17 | 4/8 | 0/9 | 0/3 | **6/40** |

Sub-grouping suggested by the chains:

* **Gujaratic = G + OG + WK.** Only these take the Gujarati chain: r̩ > a with initial r̩ > ri, the
  -vũ infinitive (G, WK), the flap written ṛ, and — decisively — **medial syncope before apocope**
  (adding it to WK: −0.0155; to any Rajasthani survey lect: +0.001 to +0.014). WK built from the G
  chain scored 0.5595 vs 0.5956 from the Rajasthani template at the time of the comparison.
  **tr retention** (t/tʰ + r unassimilated, also initial tr-) holds only in G and OG (full
  assimilation instead costs G +0.0013, OG +0.0020; dropping initial tr- costs G +0.0006:
  kʰetar, āmãtraṇ, OG traṇṇi); every other file, incl. OMarw (−0.0017) and Koli (−0.0019), prefers
  assimilation — a Gujarati trait already in Old Gujarati.
* **OG vs OMarw (Old Western Rajasthani).** Both keep final vowels (apocope costs +0.07) and the
  anusvāra spelling; both lengthen before degeminated clusters also word-initially (OG any vowel,
  −0.0115; OMarw only i/u, −0.0017 vs any vowel; G penultimate only, +0.0098 if extended). The
  modern split is already visible: OG tr-retention and -aka > -aü; OMarw -aka > -o, v- > b-,
  m > v, ṇṇ > nn, sk > kk — the Marwari profile.
* **South-western Rajasthani (Marwari, Mewari, Godwari, Basad-Mewari, Rj)**: s > h (Godwari 88 %,
  Mewari 58 %, Basad 90 %, Marwari sources ~50 %), non-initial deaspiration (Mewari, Godwari, Rj),
  rt/rd > ṭṭ/ḍḍ strongest (Godwari, Mewari, Malvi, Shahawadi), c > s in Godwari (and 17/53 in
  Mewari, 14/62 in Malvi). s > h is shared with Koli: an areal south-western trait across the
  Gujarati/Rajasthani boundary rather than a genetic node.
* **Central/eastern Rajasthani (Dhundari, Shekhawati, Hadauti, Bagri)**: no s > h (0/29–2/51),
  intervocalic h-loss (all four) and v-loss (both Dhundari points), -o, ḷ, ṇ retained;
  Shekhawati is the only central point where compensatory lengthening is not supported
  (pruned, LOO −0.0001).
* **Mewati–Ahirwadi**: no s > h, no intervocalic h-loss (adding it: +0.005/+0.006); Ahirwadi masc.
  and past participle -ā (6 -o vs 40 -ā; verb citation `> ā` −0.0068), v > b everywhere and ṇ > n
  — transitional to Western Hindi (Haryanvi/Braj). Akera is intermediate (-o 66 / -ā 85, keeps ṇ).
* **Malvi / Nimadi**: Malvi (Varghese) has no ṇ (ṇ > n −0.0035) and writes geminates without
  compensatory lengthening (degem-lengthen pruned: sat, lat, kūttaro); Nimadi keeps ṇ and ḷ and
  has lengthening (−0.0020 when re-added).
* **Gujari (Goj) is the outlier**: no compensatory lengthening (adding it +0.022; first-syllable
  short V + CC is written long in 29 % of Goj reflexes vs 32 % for plain short V, against 61 % vs
  25 % in Hadauti and 47 % vs 9.5 % in G), no n > ṇ (Rensch
  9/22), nasal + stop kept as clusters (`Nasal Stop > ṁ` +0.013), no -o generalisation of
  a-stems (+0.008), no s > h; j > y (Rensch et al.'s Pakistani data only: 8/8; Hugoniot's Eastern
  Gujari 0/6) and cʰ > c. These match its north-western (Panjabi/Pahari/Hindko) surroundings; the
  Rajasthani core (lengthening, -o, ṇ) is weak. Gujari fits none of the subgroups above: at most a
  Rajasthani base reshaped by north-western contact.

## 5. Novel / surprising / data issues

* **Stress-conditioned length before anusvāra (G and everywhere):** lengthening only when the next
  syllable is light (ā̃gaṇ, ā̃gḷī, kā̃ṭɔ vs ãgār, ãdʰārũ, ãgʰoḷvũ, tʰãbʰvũ). Unconditioned lengthening
  and final-syllable-only lengthening are both worse.
* **G vowel details**: r̩ > a beats r̩ > i (+0.0009 for i); u/ū merge (−0.0025) but i/ī must stay
  distinct (+0.0083); i + a in hiatus > ε medially (kaṇer < karṇikāra, divel < dīpataila; −0.0007);
  a(h)i > ε (εdʰāṇ < abʰijñāna, εraṇⁱ < adʰikaraṇī).
* **ḍʰ is not flapped in G** (paḍʰvũ, caḍʰvũ; flapping only ḍ: −0.0008).
* **sk/ṣk > kk (unaspirated)** in OMarw, WK, Gujari, Mewari, Malvi (sūkai, dukāl) beside the regular
  s + stop > aspirated geminate — the Prakrit sukka ~ sukkha doublet surviving dialectally.
* **ś in Gujarati**: 63 of the 87 G reflexes with ś continue OIA ś (keś, koś, niś, deśī) —
  a semi-learned stratum that escaped the sibilant merger; not modellable phonologically.
* **Tatsama doublets**: kʰet ~ kʰetar (kṣētra), citārɔ ~ citrār — part of G's tr retention is
  learned, which is why its gain in G is small although OG shows it robustly.
* **Survey transcription**: all survey sources write ḍ for the flap (any flap rule +0.004 to +0.02),
  nasal consonant + stop for nasal vowels, ī/ū for short i/u. Citation forms: masc. -o for a-stems
  (`a > o / C _ #`) and "X!, he Xed" (past -o) for verbs; plural -ā forms (andārā, kānaḍā, ḍorā)
  and -ḍo/-ṛī diminutives are the main irreducible residue (o → ā is the top confusion in Hadauti).
* **Dialect mixture inside lects**: `mewari_basad` ("Malvi") = Varghese's Malvi (s > h 26/84) +
  a Mewari survey point at Basad (s > h 27/30); s > h nets ≈ 0 and is left out. `Rj` is the Mewari
  survey again, not CDIAL "Rj.". `Nimadi` includes Bareli (a Bhil language) — with a digraph block
  it scores like the Nimadi source (0.458 vs 0.459). `Goj` mixes Pakistani Gujari (Rensch et al.),
  Eastern Gujari (Hugoniot et al.) and Noira (Varghese & Kumar; probably not Gujari). `Marw` mixes
  CDIAL Marw. (infinitive -ṇo; NED 0.33) with two surveys (0.49).
* **Data format problems**: Rensch et al. reflexes hold alternants joined by " / " (yano / yaṇā,
  āy / ay; 87 Goj pairs) and are scored as one string; some CDIAL reflex fields hold OIA words
  (G vilōḍayati → "saṁlōḍyatē", *nītiśālā → "nárāśáṁsa", *utskadati → "útkasati",
  āstʰāpana → "āstʰā́na", OG káścid → "kā́nicid") — mislinked citations.
* **Koli**: implosives ɓ ɗ ɠ in 16 pairs (ɠāro, ɗāṛo, ɓīz) are not regular enough for a rule
  (+0.003); initial voiced aspirates deaspirate (gāh < gʰāsa; −0.012); zubair's ṛ may cover the
  retroflex lateral (ukāṛvū̃ < utkalati).
* **Gujari deaffrication**: besides cʰ > c, Rensch has ź/ś for j/c in a few items (suraź, pānź,
  maśar, śat) — Kashmiri/Pahari-like, not modelled.
* **Western deaspiration area**: non-initial voiced aspirates deaspirate in Mewari, Godwari, Rj,
  Shahawadi, Malvi, Bagri (both series in Godwari and Rj, −0.0044/−0.0048); CDIAL Gujarati keeps all
  aspirates (+0.007 to +0.025 if removed).

## 6. New tags and engine wishes

* `@x-rg-deaffrication` — palatal affricates > sibilants: Koli c cʰ j jʰ > s s z z; Godwari initial
  c(ʰ) > s.
* `@x-rg-j-y` — Gujari (Rensch et al.) j jʰ > y (jana > yano, adya > ay).
* `@x-rg-aphaeresis` — loss of unstressed initial a- before CV (araghaṭṭa > rahaṭ, anyākāra > nyāro).
* `@x-rg-rt-retroflex` — r + dental stop > retroflex geminate (varta > bāṭɔ, vartati > vaṭvũ,
  vardʰita > baḍīyā); co-tagged `@assim-r`.
* `@x-rg-dv-b` — dv > bb (dvāra > bār, dvādaśa > bār-).
* `@x-rg-mr-mb` — m + liquid > mb (āmra > amba > ām(b)o).
* `@x-rg-dla` (round 4) — loss of the suffix vowel of -aḍa-/-ala- before the -o/-ī ending (-aḍaka > -ḍo).
* `@x-rg-weight` (round 2) — vowel length conditioned by syllable position and the weight of the next
  syllable (NIA stress: first syllable unless the second is heavy), as opposed to `@stress` (Vedic accent).

Cross-group equivalences for the synthesis: `@x-rg-rt-retroflex` = `@x-hindi-rt-retroflex` =
`@x-nw-rt-cerebral`; `@x-rg-dv-b` = `@x-nw-dv-b`; `@x-rg-mr-mb` = `@x-hindi-mr-mb`;
`@x-rg-deaffrication` ≈ `@x-marathi-bhil-c-s` / `@x-marathi-bhil-ch-s` (the Koli/Godwari c > s is the
western end of the Bhili deaffrication area).

Engine features that would have helped: set difference / class intersection (`Stop − {t tʰ}`;
worked around with the `NDStop` / `NR` classes, since `!{t tʰ}` also matches vowels); optional tokens
inside targets (`a (h) {i ī} > e`); gloss/POS conditioning for citation morphology (the -vũ rule also
hits numerals in -viṁśati); `%source` blocks for sound changes, not only notation, for
dialect-mixed lects (Malvi + Basad Mewari, Pakistani + Eastern Gujari); splitting reflex fields on
" / " into alternants.

## 7. Round 2: vowels and accent

Scratch tools (in `tmp/sound_changes/rajasthani-gujarati/r2/`): `audit.py` (LOO of the partial
round-2 edits against the round-1 rules), `exp.py` (score a block of edits; train NED split into
Vedic-accented and unaccented etyma), `cross*.py` (the same edit in every lect), `tab.py`
(cross-tab of attested vowel outcomes by probe features), `cmp.py`/`cmpf.py` (per-etymon changes),
`probe2.py` (the `probe` tool with `%source` notation blocks applied to the prediction: plain
`probe` ignores them, so on the survey lects its "i > ī", "u > ū" cells are notation artefacts).

### 7.1 Scores (round 1 → round 2)

| lect | name | #rules r1 → r2 | train NED r1 → r2 | test NED r1 → r2 | test exact r1 → r2 | round-1 rules re-scored on the round-2 engine (train / test) |
|---|---|---|---|---|---|---|
| G | Gujarati | 49 → 51 | 0.3772 → 0.3699 | 0.3734 → 0.3683 | 17.0% → 18.4% | 0.3771 / 0.3734 |
| OG | Old Gujarati | 35 → 36 | 0.3172 → 0.3106 | 0.3125 → 0.3108 | 19.6% → 20.8% | 0.3169 / 0.3125 |
| OMarw | Old Marwari | 42 → 44 | 0.3280 → 0.3222 | 0.3186 → 0.3151 | 18.8% → 19.6% | 0.3279 / 0.3186 |
| Marw | Marwari | 42 → 44 | 0.4560 → 0.4540 | 0.4726 → 0.4756 | 6.9% → 6.9% | 0.4560 / 0.4726 |
| mewari_basad | Malvi | 36 → 37 | 0.4897 → 0.4844 | 0.5369 → 0.5365 | 1.9% → 1.9% | 0.4890 / 0.5369 |
| mewari_dholpura | Mewari | 45 → 46 | 0.4873 → 0.4856 | 0.5339 → 0.5306 | 2.1% → 2.1% | 0.4889 / 0.5339 |
| Goj | Gujari | 40 → 41 | 0.4826 → 0.4837 | 0.4679 → 0.4679 | 6.9% → 6.9% | 0.4867 / 0.4679 |
| had | Hadauti | 41 → 42 | 0.4832 → 0.4815 | 0.5204 → 0.5182 | 4.8% → 4.8% | 0.4834 / 0.5204 |
| dhundari_badagaon | Shekhawati | 37 → 38 | 0.4883 → 0.4891 | 0.4738 → 0.4768 | 8.3% → 7.5% | 0.4903 / 0.4738 |
| Nimadi | Nimadi | 33 → 34 | 0.4523 → 0.4497 | 0.4601 → 0.4608 | 6.5% → 6.5% | 0.4546 / 0.4601 |
| bagri_fatehabad | Bagri | 40 → 41 | 0.4749 → 0.4766 | 0.4505 → 0.4513 | 10.9% → 10.9% | 0.4802 / 0.4505 |
| dhundari_bamore | Dhundari | 40 → 41 | 0.4625 → 0.4621 | 0.4596 → 0.4596 | 8.0% → 8.0% | 0.4656 / 0.4596 |
| mewati_akera | Mewati | 33 → 34 | 0.4422 → 0.4412 | 0.4689 → 0.4639 | 10.5% → 9.7% | 0.4431 / 0.4689 |
| marwari_bagra | Godwari | 38 → 39 | 0.4627 → 0.4626 | 0.5186 → 0.5146 | 5.1% → 5.1% | 0.4649 / 0.5186 |
| hadothi_kelwada | Shahawadi | 36 → 36 | 0.4168 → 0.4168 | 0.4474 → 0.4474 | 14.6% → 14.6% | 0.4191 / 0.4474 |
| mewati_jhambaus | Ahirwadi | 32 → 33 | 0.3784 → 0.3781 | 0.4265 → 0.4265 | 9.8% → 9.8% | 0.3796 / 0.4265 |
| Rj | Rajasthani | 34 → 35 | 0.4979 → 0.4935 | 0.4756 → 0.4717 | 14.0% → 14.0% | 0.4991 / 0.4756 |
| WK | Wadiyara Koli | 37 → 38 | 0.4911 → 0.4877 | 0.5883 → 0.6082 | 2.4% → 2.4% | 0.4911 / 0.5883 |

"r1" = `summary-round1.csv`. The last column re-scores the round-1 rule files on the round-2 engine (06:33, before any round-2 edit); against it every lect's train NED improved or stayed equal (Gujari 0.4867 → 0.4837, Shekhawati 0.4903 → 0.4891, Bagri 0.4802 → 0.4766), so the apparent train rises against the summary file are engine re-scoring. Test NED rose by more than 0.005 only in WK (§7.2); Marwari's +0.003 comes from medial `u > a` (reverting it: test −0.0030, train +0.0014).


No lect uses `%gold-drop` (no tone marks in these sources), so all changes are chain changes.

### 7.2 Audit of the partial round-2 edits (the 06:39 snapshot)

All partial edits had positive train LOO against the round-1 rule they replaced and were kept:
- `u > a / V C* _ C* V` (@vowel-shift; medial u > a, māṇuṣa > māṇas, OMarw kūkusa > kūkasa), in
  the 15 non-Gujaratic files: LOO +0.0005 (OMarw) … +0.0046 (Malvi).
- `i n > ī / _ #` (@morph, -in stems cited in the nom. -ī: hastín > hātʰī), G, OG, OMarw, Marw,
  WK: +0.0006 … +0.0019.
- compensatory lengthening restricted with `C*` (first syllable of a disyllable in G, OMarw,
  Mewari, Nimadi; first syllable of any word in OG and WK): +0.0008 … +0.0031 over the round-1
  `_ V #` rule.

**Wadiyara Koli test NED 0.5883 → 0.6082 (+0.020).** The whole rise comes from one edit, WK's
first-syllable lengthening `V G $2 > $1+long $3 / # C* _` (reverting it gives the round-1 test
number back but costs train +0.0015). It changes 5 of WK's 41 test pairs, and a single etymon
accounts for 0.016 of the 0.020: *matsiya → predicted mās, attested maccʰī / massī / macʰī
(the diminutive -ī makes the word trisyllabic, where the G-type "disyllables only" condition would
have kept a short). On train (132 pairs) the any-length, disyllable-only and round-1 variants
differ by one pair's worth (0.0015), so this is test-set noise on a 41-pair split, not a
systematic error. I kept the train choice; the G-type condition would be the consistent
alternative.

### 7.3 Vowel rules adopted this round (train LOO / gain over the rule they replace)

New tag `@x-rg-weight`: a vowel change conditioned by syllable position and the weight of the
next syllable (i.e. by the NIA stress, not the Vedic accent; `@stress` is reserved for
OIA-accent conditioning and no rule in this group earned it).

1. **Compensatory lengthening = first syllable, next syllable final or light** (G, OMarw, Mewari;
   surveys: `_ V # | # C* _ C* Short C V`):
   G `{a i e o ā ī ē ō} G $2 > $1+long $3 / # C* _ C* V (C) # | # C* _ C* Short C V`,
   LOO +0.0113; the light-syllable extension alone −0.0018 (markaṭa > mākaṛ, aṣṭamá > āṭʰam,
   pallava > pālav; heavy next syllable blocks it: rakṣapāla > rakʰvāḷ, *hastavr̩tti > hatʰɔṭī).
   Same extension: OMarw −0.0006 (−0.0010 together with `{i u}` initial lengthening
   written `# C* _`), Mewari (Dholpura) −0.0016, Bagri −0.0014, Godwari −0.0016, Shahawadi
   −0.0022, Rj −0.0027. It hurts Marw, Hadauti, Dhundari, Mewati, Ahirwadi, Nimadi, WK
   (+0.0004 … +0.006), which keep their round-1 condition.
2. **Anusvāra lengthening = first syllable, not before a heavy (long-vowel closed) syllable**
   (G, OG, OMarw): G `V ṁ > $1+long+nas / # C* _ C // _ C Long C`, −0.0019 vs the round-1
   "next vowel short" rule (jáŋgʰā > jā̃gʰ, dantaka > dā̃tɔ, saṇḍʰikā > sā̃ḍʰī, ṭáŋkati > ṭā̃kvũ
   vs áŋgāra > ãgār, añjīra > ãjīr, *kʰaŋkʰālayati > kʰãkʰāḷvũ, ēraṇḍa > erãḍ); LOO +0.0075.
   OG −0.0015, OMarw −0.0020. Both lengthenings together in G: −0.0037. In the survey lects the
   first-syllable version is slightly worse (+0.0004 … +0.0013) and was not adopted.
   So the round-1 "stress-conditioned" description holds, but the stress is the NIA
   initial/heavy-syllable stress: the lengthened syllable is the first one unless the second is
   heavy — the same condition for anusvāra and for degemination lengthening.
3. **G ū: `ū > u // # C* _ C V`** (early; LOO +0.0009), with u excluded from compensatory
   lengthening (dudʰ, muṭʰ) and the late merger reduced to `ū̃ > ũ` (LOO +0.0022): together
   −0.0011 vs the round-1 blanket `{ū̃ ū} > {ũ u}`. OIA ū survives only in an open first syllable
   (mūla > mūḷ, śūra > sūr, karpūra > kapūr vs sūtra > sutar, śūrpa > supṛũ). CDIAL's Gujarati
   u/ū spelling is loose, so this is partly orthographic. G-only: OG, WK 0, OMarw +0.0014.

### 7.4 Accent tests (train ΔNED; in brackets the change on the Vedic-accented etyma only)

Method: accent-conditioned variants are applied only to `if:accented` etyma (25–30 % of
etyma; Turner's starred forms carry no accent), with the round-1 rule kept for the others, so the
bracketed number compares "Vedic accent" with "weight/position" on the same words. Where the
accent has to survive past `@acc-loss`, I either wrote the rule before `@acc-loss` in OIA terms
(e.g. `if:accented Acc > $1+long / _ Nasal Stop`), or replaced `@acc-loss` by an explicit list
that spares one vowel, or protected a vowel with a marker (ə) restored right after the test rule.

| hypothesis | result |
|---|---|
| G anusvāra lengthening only under the Vedic accent | +0.0015 (+0.0051) |
| G … accent AND light next syllable | +0.0019 (+0.0065) |
| G … accent OR light next syllable | −0.0004 (−0.0015) = unconditioned lengthening in accented words (−0.0015); the first-syllable rule gives (−0.0034) |
| G … lengthen unless pretonic | +0.0016 (+0.0054) |
| OG / OMarw / Marw / Hadauti: tonic-only anusvāra lengthening vs none | +0.0026 vs +0.0035 / +0.0012 vs +0.0015 / +0.0061 vs +0.0068 / +0.0007 vs +0.0013 |
| OG compensatory lengthening only if the first syllable is tonic | +0.0053 (+0.0119); no lengthening at all +0.0081 (+0.0180); "unless pretonic" +0.0052 |
| OMarw / Marw / Hadauti tonic-only CL vs no CL | +0.0078 vs +0.0080 / +0.0102 vs +0.0094 / +0.0075 vs +0.0068 |
| G syncope: pretonic medial vowels protected | +0.0018 (+0.0063) |
| G syncope: tonic medial vowels protected | +0.0038 (+0.0129) |
| G `ŕ̩ > i` (tonic r̩ > i, else a) | +0.0004 (+0.0014) |
| OG `ŕ̩ > a` / `if:!accented r̩ > a` | +0.0004 / −0.0013 (see below) |
| G tonic ū́ keeps its length | −0.0005 (−0.0017); tonic ū́ in an open syllable −0.0006 (−0.0020); but any first-open-syllable ū −0.0010 (−0.0023): position ≥ accent |
| Survey citation: barytone final -a > ∅, rest -o (Marw) | +0.0031 (+0.0065); oxytone -á > ∅ +0.0096 (+0.0199) |
| Aphaeresis of initial a- blocked when a- is tonic (á-) | Marw +0.0021, Mewari +0.0023, Hadauti +0.0010, Dhundari +0.0005, Rj/Bagri/Gujari/Ahirwadi 0.0000 |

Descriptive counts (G, `tab.py`): compensatory lengthening in closed first syllables — a: tonic
49 % long, pretonic 39 %, unaccented 36 %; i: tonic 14 %, pretonic 43 %, unaccented 21 %.
Medial open short vowels lost (all causes): tonic 70 %, posttonic 71 %, pretonic 42 %,
unaccented 61 %. Final -a in the survey lects (Marw): posttonic (barytone) ∅ 41 % / -o 25 %;
tonic (oxytone -á) -o 42 % / ∅ 32 %; unaccented -o 44 % / ∅ 24 %.

**What the accent does and does not condition in this group.** Nothing that we can turn into a
rule. Every length change (anusvāra, compensatory lengthening, ū retention) is predicted better by
the position of the syllable and the weight of the next one than by the Vedic accent; in the
accented etyma the accent recovers a third of what the positional rule gets for anusvāra and
compensatory lengthening, and tonic-only lengthening is worse than no lengthening in Marwari.
Syncope ignores it (tonic medial vowels are lost as often as posttonic ones; protecting either
pretonic or tonic vowels hurts), and so does the aphaeresis of initial a-, which round 1 called
"unstressed": tonic á- is lost as readily. Two weak correlations, neither usable as a rule:
oxytone -á stems are cited with the -o citation ending more often than barytone ones in the
survey lects, and pretonic medial vowels survive more often in G (probably because pretonic
medial vowels sit in long words where syncope applies only once). The OG r̩ split is a stratum
effect, not accent: Turner's starred (unaccented) etyma prefer r̩ > a (−0.0024 on them), the
Vedic-attested ones prefer r̩ > i (+0.0016 for a), while tonic `ŕ̩ > a` does nothing (+0.0009 on
accented words). The a/i/u split of r̩ has no conditioning here: labial neighbours (r̩ > u)
0.0000, after dentals r̩ > i +0.0007, harmony before i −0.0000.

### 7.5 Other hypotheses tested and rejected (train ΔNED)

- **Gujarati syncope** is best unconditioned (`{a i u} > ∅ / V C _ C V`, before apocope):
  second syllable only +0.0018, all but the second +0.0092, penult only +0.0036, after long vowels
  only +0.0033, after short only +0.0075, before a long vowel only +0.0087; also after a closed
  syllable +0.0001. WK prefers penult-only syncope (−0.0060) but only through two words in -amaṇa
  (udgamana > ūgamṇũ, astamáyana > ātʰamṇũ); not adopted. Syncope in any non-Gujaratic lect:
  +0.004 … +0.029 (OG +0.029, Old Marwari +0.017) — Gujaratic-only and post-OG.
- **Two-mora-law shortening** (long V > short / _ CC): +0.003 … +0.017 everywhere; ā only:
  −0.0012 Shekhawati, −0.0018 Rj, +0.0004 … +0.0054 elsewhere. Not adopted — NIA
  re-lengthening makes it invisible.
- **G final -ī apocope** (`{a ā i u ī} > ∅ / _ #`): +0.0037 (fem. -ī is kept).
- **G first-syllable i > a before a rounded vowel** (vi-/ni- > va-/na-: vakʰuṭvũ, naṭʰor):
  −0.0004, entirely the vi-/ni- prefixes; not adopted (a prefix, not a sound change).
- **G medial u > a** (the OMarw rule) +0.0010; medial i > a +0.0036; u > a before u +0.0002;
  posttonic closed medial i > ī +0.0003; compensatory lengthening only for a (not i) +0.0009,
  i only in disyllables −0.0006 (marginal, not adopted).
- **Survey fem. citation -ā > -ī** (@morph): −0.0006 … −0.0036 in 9 survey lects but carried by
  ≤ 8 etyma (haridrā > haḷdī, *diddā > dīdī, vīṭā > bīṭī); not adopted.
- **Citation-morphology flags**: `if:!numeral` on the verb-citation rules changes nothing
  (0.0000 … +0.0019); `if:verb` hurts every survey lect (+0.003 … +0.014) — the -ati/-āti rule
  also correctly converts unflagged -ti etyma. No change.
- **Gujari medial a**: probe shows 58 % loss, but no syncope formulation helps (best −0.0005).

### 7.6 Relative chronology from this round

Both lengthenings read the weight of the second syllable on the pre-syncope, pre-apocope word
(dā̃tɔ, mākaṛ), so the stress placement that drives them is older than Gujarati syncope and
apocope, as the round-1 swap tests (§3) already required for compensatory lengthening. In G the
u/ū facts need two parts: the early `ū > u` outside open first syllables, and no lengthening of u;
the early rule alone, with u still lengthened and no late merger, is worse (+0.0056), because
the ū created by compensatory lengthening is then never shortened.

## 10. Extension-aware score (round 4)

Scratch tools in `tmp/sound_changes/rajasthani-gujarati/r4/`: `xexp.py` (an edit's effect on plain
and extension-aware train NED), `show.py` (derivations of the menu variants of test stems),
`resid.py` (ending mismatches left under the best variant), `which.py` (reflexes that pick a
given variant), `cand.py` (candidate menu items: how many train reflexes each would win from the
current menu), `ba.py` (before/after table), `before/` (the rule files as they stood at the start
of this round). All numbers use the 13:29 menu (aka, akka, ikā, uka, aḍaka, allaka, araka).
The accent score is unchanged in every lect (only G, n = 5, and OG, n = 1, have any accented pairs).

### 10.1 Scores (before → after this round)

| lect | train plain before → after | train ext before → after | test plain before → after | test ext before → after |
|---|---|---|---|---|
| G (Gujarati) | 0.3699 → 0.3638 | 0.3014 → 0.2914 | 0.3683 → 0.3650 | 0.3006 → 0.2914 |
| OG (Old Gujarati) | 0.3106 → 0.3106 | 0.2544 → 0.2544 | 0.3108 → 0.3108 | 0.2598 → 0.2598 |
| OMarw (Old Marwari) | 0.3222 → 0.3213 | 0.2850 → 0.2810 | 0.3151 → 0.3151 | 0.2786 → 0.2731 |
| Marw (Marwari) | 0.4540 → 0.4524 | 0.3985 → 0.3942 | 0.4756 → 0.4713 | 0.4278 → 0.4219 |
| mewari_basad (Malvi) | 0.4844 → 0.4843 | 0.4303 → 0.4257 | 0.5365 → 0.5352 | 0.4736 → 0.4749 |
| mewari_dholpura (Mewari) | 0.4856 → 0.4860 | 0.4186 → 0.4136 | 0.5306 → 0.5295 | 0.4612 → 0.4566 |
| Goj (Gujari) | 0.4837 → 0.4837 | 0.4071 → 0.4008 | 0.4679 → 0.4679 | 0.3587 → 0.3571 |
| had (Hadauti) | 0.4815 → 0.4820 | 0.4187 → 0.4123 | 0.5182 → 0.5189 | 0.4536 → 0.4445 |
| dhundari_badagaon (Shekhawati) | 0.4891 → 0.4892 | 0.4303 → 0.4260 | 0.4768 → 0.4726 | 0.4097 → 0.4034 |
| Nimadi (Nimadi) | 0.4497 → 0.4499 | 0.3996 → 0.3825 | 0.4608 → 0.4582 | 0.4134 → 0.4055 |
| bagri_fatehabad (Bagri) | 0.4766 → 0.4764 | 0.4175 → 0.4109 | 0.4513 → 0.4489 | 0.4059 → 0.3994 |
| dhundari_bamore (Dhundari) | 0.4621 → 0.4623 | 0.3928 → 0.3896 | 0.4596 → 0.4616 | 0.4043 → 0.4012 |
| mewati_akera (Mewati) | 0.4412 → 0.4412 | 0.3451 → 0.3390 | 0.4639 → 0.4619 | 0.3908 → 0.3796 |
| marwari_bagra (Godwari) | 0.4626 → 0.4626 | 0.3996 → 0.3957 | 0.5146 → 0.5146 | 0.4572 → 0.4522 |
| hadothi_kelwada (Shahawadi) | 0.4168 → 0.4168 | 0.3418 → 0.3353 | 0.4474 → 0.4474 | 0.3744 → 0.3744 |
| mewati_jhambaus (Ahirwadi) | 0.3781 → 0.3781 | 0.3214 → 0.3087 | 0.4265 → 0.4265 | 0.3533 → 0.3410 |
| Rj (Rajasthani) | 0.4935 → 0.4935 | 0.4253 → 0.4208 | 0.4717 → 0.4717 | 0.3846 → 0.3711 |
| WK (Wadiyara Koli) | 0.4877 → 0.4877 | 0.4045 → 0.4045 | 0.6082 → 0.6082 | 0.5531 → 0.5531 |

Plain NED changed by at most +0.0007 (train) in any lect; G gains 0.0060 plain from the root-citation rule. OG and WK were left unchanged (no edit improved their ext score). Test NED shown for information only; nothing was tuned on it.

### 10.2 How the chains treated the menu, and what was changed

Checked on gʰōṭa, hasta, karṇa, kuñcī, bʰallū, pāda with every extension (`show.py`):
- **-aka → masc. -o** works everywhere: G gʰoṛo (ɔ), OG gʰoḍau (-aü), OMarw, WK, Gujari and all
  survey lects gʰoḍo. The survey lects also give bare a-stems -o through the citation rule
  `a > o / C _ #` (@morph). That rule is still needed for plain NED, which is why these lects
  rarely "choose" -aka (3–7 %): their bare form already carries the ending. Ahirwadi's masc. is
  -ā (§4); `a a > ā / _ #` there gains ext −0.0006 but costs plain +0.0037, so it was not adopted,
  and the menu's -akka (→ -āk) stands in for -ā instead (see 10.4).
- **-ikā → -ī** works except in Nimadi and Malvi (the -ia hiatus was never contracted: kūñciā):
  added `i {a ā} > ī / _ #` (@hiatus @aka). Mewari (Dholpura) keeps -īā; contraction there
  costs ext +0.0005 (its reflexes are plural -ā / fem. -ī with a long root vowel).
- **-uka → -ū** was destroyed in the 15 files that have the round-2 medial `u > a` (it turned
  -uka into -aka). The rule now spares a u in hiatus before a lost stop:
  `u > a / V C* _ C* V // _ {k g c j t d y} V` (-uka > -ua > -ū; plain ≤ +0.0007, ext −0.0015 …
  −0.0141; Ahirwadi −0.0141). In Rj the phonological version costs plain +0.0015, so Rj excludes
  only the -uka suffix (`// _ k {a ā} #`, plain 0, ext −0.0049). Shahawadi also needed the
  contraction `u {a ā} > ū / _ #` (ext −0.0066).
- **-aḍaka → -ṛo/-ḍo, -allaka → -lo.** In the surveys the chains gave -aḍo/-alo (kanaḍo,
  ṭūkaḍo) where Hadauti and Marwari have ṭūkḍɔ, pʰūlḍɔ. New rule (@syncope, new tag
  `@x-rg-dla`): `a > ∅ / V C (C) _ {ḍ ḷ l ṛ} {o ī} #`, i.e. the suffix vowel of -aḍa-/-ala- is
  lost before -o/-ī. Adopted where it helps the ext score: G −0.0016, Hadauti, Marwari, Bagri,
  Godwari (−0.0007 … −0.0012 with the geminate-aware form), Gujari −0.0041 (camṛī, lakṛī), Mewati
  (Akera) −0.0021. It *hurts* Mewari +0.0011, Nimadi +0.0015, Ahirwadi +0.0021, WK +0.0028, Malvi
  +0.0007, Shekhawati +0.0006: these keep -aḍo/-aḍī (kūkaḍo). This is a clean dialect split: -ḍo
  in Hadauti–Marwari–Bagri–Godwari–Gujari–Akera–Gujarati, -aḍo in Mewari–Malvi–Nimadi–
  Dhundari–Ahirwadi–Koli. G also had -allaka → -ḷlo (ākʰāḷlo); `ḷ l > l` after syncope fixes it
  (ext −0.0007).
- **Compensatory lengthening before -aḍa-**: Hadauti and Dhundari (Bamore) now lengthen also in
  a first syllable before a light syllable (`_ V # | # C* _ C* Short C V`; kárṇa + -aḍa- > kānaḍā):
  ext −0.0022 / −0.0017, plain 0.0000 / −0.0005 (in round 2 this cost Hadauti +0.0006; the -uka
  fix removed that).
- **Citation morphology.** Turner's bare verbal roots (*aḍḍ, *aṇṭ; flag `root`) are cited as
  verbs: G `if:root ∅ > v ũ / G _ #` (plain −0.0060, ext −0.0078; 95 G reflexes ended in -vũ
  against a bare prediction), Marwari `if:root ∅ > o / G _ #` (plain −0.0021, ext −0.0004). The
  survey analogue helps plain NED elsewhere but worsens the ext score (a root's noun reflexes then
  lose the bare variant), so it is not adopted there; WK gains nothing.
- Rejected: G syncope blocked before a final short syllable (-aṇ, -ar, -aḷ residue: ā̃gaṇ, ātʰar):
  ext −0.0085 but plain +0.0025 (the bare form is then worse against neuter -ũ reflexes such as
  ā̃gṇũ: this is the neuter problem of 10.4); OG nom. -u for a-stems (plain +0.0419);
  `u {a ā} > ū / _ #` elsewhere (0 or worse).

Extension-aware train LOO (`score --ext-loo`) of the rules added or changed this round:

| lect | rule | ext LOO |
|---|---|---|
| G | `if:root ∅ > v ũ / G _ #` | +0.0078 |
| G | `a > ∅ / V C (C) _ {ḍ ḷ l ṛ} {o ī} #` | +0.0015 |
| G | `ḷ l > l` | +0.0007 |
| Marw | `if:root ∅ > o / G _ #` | +0.0005 |
| Marw / Hadauti / Bagri / Godwari / Gujari / Mewati | `a > ∅ / V C (C) _ {ḍ ḷ l ṛ} {o ī} #` | +0.0018 / +0.0003 / +0.0013 / +0.0019 / +0.0041 / +0.0021 |
| Nimadi / Malvi | `i {a ā} > ī / _ #` | +0.0103 / +0.0010 |
| Shahawadi | `u {a ā} > ū / _ #` | +0.0066 |
| 14 files | `u > a / V C* _ C* V // _ {k g c j t d y} V` | +0.0006 (Ahirwadi) … +0.0047 (Malvi); OMarw +0.0015, Marw +0.0011, Gujari +0.0034, Nimadi +0.0042 |
| Rj | `u > a / V C* _ C* V // _ k {a ā} #` | +0.0036 |
| Hadauti / Dhundari | CL `_ V # \| # C* _ C* Short C V` | +0.0179 / +0.0284 (whole rule) |

One older rule has a negative ext-LOO: the round-1 `i {a ā} > ī / _ #` in Mewati (Akera) −0.0017
and Ahirwadi −0.0025 (their -ikā/-ika reflexes keep -iyā/-ī-ā forms); it was found after the
editing freeze and left in place.

### 10.3 What the ending choices say

| lect | train ending shares (after) |
|---|---|
| G | bare 65%, aka 14%, ikā 11%, aḍaka 4%, akka 2%, allaka 2%, araka 1%, uka 1% |
| OG | bare 65%, aka 22%, ikā 4%, aḍaka 2%, uka 2%, allaka 2%, akka 1%, araka 1% |
| OMarw | bare 78%, aka 8%, ikā 5%, uka 4%, akka 4%, allaka 1%, aḍaka 1%, araka 1% |
| Marw | bare 68%, ikā 10%, aḍaka 7%, akka 4%, aka 4%, araka 4%, allaka 2%, uka 1% |
| mewari_basad | bare 68%, ikā 12%, akka 9%, aḍaka 3%, araka 3%, aka 2%, allaka 1%, uka 1% |
| mewari_dholpura | bare 56%, ikā 22%, akka 7%, aka 4%, aḍaka 4%, uka 3%, araka 2%, allaka 2% |
| Goj | bare 63%, aka 16%, ikā 12%, aḍaka 3%, akka 3%, uka 2%, araka 1%, allaka 1% |
| had | bare 61%, ikā 13%, akka 8%, aḍaka 6%, aka 4%, araka 4%, allaka 2%, uka 2% |
| dhundari_badagaon | bare 65%, ikā 10%, akka 8%, aḍaka 7%, aka 5%, allaka 3%, uka 2%, araka 0% |
| Nimadi | bare 66%, ikā 14%, akka 6%, aka 4%, uka 3%, aḍaka 3%, allaka 3%, araka 1% |
| bagri_fatehabad | bare 63%, ikā 15%, akka 8%, aka 4%, aḍaka 3%, uka 3%, allaka 2%, araka 1% |
| dhundari_bamore | bare 64%, ikā 12%, aka 7%, aḍaka 7%, akka 5%, allaka 3%, uka 1%, araka 1% |
| mewati_akera | bare 52%, ikā 14%, aka 13%, akka 13%, aḍaka 4%, uka 2%, allaka 1%, araka 1% |
| marwari_bagra | bare 65%, ikā 14%, aka 6%, akka 5%, aḍaka 5%, araka 4%, allaka 1%, uka 1% |
| hadothi_kelwada | bare 57%, aka 16%, akka 12%, ikā 5%, uka 4%, aḍaka 3%, araka 2%, allaka 1% |
| mewati_jhambaus | bare 61%, akka 15%, ikā 13%, uka 5%, aka 4%, aḍaka 2% |
| Rj | bare 60%, ikā 12%, aḍaka 8%, akka 8%, aka 6%, uka 4%, allaka 2%, araka 1% |
| WK | bare 59%, aka 14%, ikā 11%, aḍaka 11%, araka 3%, allaka 2% |

- **-aka over the bare stem**: the Gujaratic lects and Gujari choose -aka most (OG 22 %, G 14 %,
  WK 14 %, Gujari 16 %, Shahawadi 16 %). In the Rajasthani surveys -aka is hidden by the
  citation -o rule.
- **-ikā** (fem. -ī) is the most-chosen extension in every Rajasthani survey lect (10–22 %; Mewari
  22 %), against 4–5 % in OG/OMarw: the survey questionnaires elicit many fem./diminutive -ī
  nouns (kīḍī, cʰāḷī, pʰaḷī).
- **-aḍa-** is productive in Marwari, Hadauti, Shekhawati, Dhundari (6–13 %) and Koli (11 %),
  and least used in Gujari, Ahirwadi and G (2–3 %). **-alla-** is strongest in Bagri and
  Hadauti (up to 7 %); **-ara-** is marginal (≤ 4 %).
- **-uka** is rare (1–5 %): after the fix it is chosen mainly for bʰālū-type words.

### 10.4 Should the menu change? (for these lects)

Evidence: residual ending mismatches under the best variant (`resid.py`) and, per candidate, the
number of train reflexes on which it beats every current variant (`cand.py`; the chain derives
the candidate like any other extension; gain = drop in ext NED).

**Add, well supported:**
1. **Neuter -akam (→ G -ũ, OG -aüṁ, WK -ū̃).** The largest residual in G: 246 reflexes want
   -ũ on a bare stem and 46 more want -ũ for -o. The chain needs one rule for it,
   `a a ṁ > ũ / _ #` (G; OG `> a u ṁ`, WK `> ū̃`), which never fires on real headwords, so
   plain NED is untouched. With it plus `-akaṁ` in the menu, G ext train NED falls 0.2936 → 0.2658,
   with -akaṁ chosen by 809 of 5527 reflexes (15 %). OG 0.2544 → 0.2441 (62 of 695), WK
   0.4045 → 0.3828 (16 of 132); Gujari 7 of 753 and OMarw 1 of 595. The neuter -aṁ/-ũ is real,
   well-attested Apabhraṃśa/Old Gujarati morphology (-aüṁ), and its survival is a Gujarati
   isogloss. This is the most valuable addition for this group; the menu should also let a
   chain derive it, i.e. a nasalised variant of -aka.
2. **Plural/masc. -ā (-akāḥ).** The survey lects cite many nouns in the plural (andārā, kānaḍā,
   kāndā). A plain -ā ending wins 73 (Bagri), 59 (Godwari), 76 (Shekhawati), 78 (Hadauti),
   89 (Mewari), 124 (Malvi), 50 (Nimadi), 40 (Marwari), 125 (G) train reflexes, gains 0.02–0.03
   in the surveys. It is real morphology (masc. pl. -ā < -akāḥ; Hindi-type masc. -ā in Ahirwadi)
   but a number ending, not a stem extension.

**Add, weaker:**
3. **-iya/-īya** matters only where a chain fails to contract -ia: in Mewari -iya, -īya and a
   plain -ī each win the same 72 reflexes (fem. -ī with a long root vowel, where the -ikā variant
   gives -īā); elsewhere ≤ 8. G's -iyũ/-iyɔ residue (≈ 70 reflexes) needs -iya + neuter/masc.
   endings, i.e. -iyaka/-iyakam. Real (-iya- adjectives and diminutives), but small here.
4. **-ikka**: OG 21 (0.0035), G 23, Marwari 10, Gujari 10; real Pk -ikka but small.

**Surface fixes, do not add:** -ra, -la, -ṛa, -ḍa, -ṭa without -ka (15–60 reflexes per lect, but
they compete for the same words and mostly buy partial matches on the plural -ā and -ḍo forms);
-ālu/-āla (Gujari 46, Shekhawati 40, Bagri 42, G 90 — mostly stand-ins for other endings);
-aṇa (Rj 17 of 136: agent/verbal nouns, not an extension); plain fem. -ī (0–4 reflexes beyond
-ikā everywhere except OMarw, 54, where the chain shortens -ikā to -i).

**Drop or treat with care:**
- **-akka** is chosen 5–15 % in the survey lects (Ahirwadi 15 %, Mewati 13 %, Shahawadi 12 %) but
  for the wrong reason: these chains turn it into -āk/-ako, and it wins as a cheap stand-in for
  an unmodelled -ā/-a (Ahirwadi kālā, cʰɔrā, tārā ← predicted kālāk, cʰorāk, tārāk; Malvi keḷa,
  kaḷa ← kalako). No Rajasthani reflex continues -akk-. Its share should not be read as
  evidence; for these lects I would drop it (or charge for it).
- **-uka** and **-araka** are chosen by ≤ 5 % and ≤ 4 %; WK never chooses -uka or -akka,
  Ahirwadi never -araka. Keep them (they are real and cheap), but they do little here.

### 10.5 Addendum: neuter -akam derived (14:20, menu = aka, akka, ikā, ikka, iyaka, uka, aḍaka, allaka, araka, akam)

The chains lost -m through final-consonant loss, so -akam came out like -aka (G gʰoṛo) and was
chosen 0 %. Added before `@final-cons-loss` (unconditioned: it never hurts plain NED, so no
`if:ext` was needed):
- G `a m > ũ / _ #` + `a ũ > ũ / _ #` (before `a a > o`): gʰōṭakam > gʰoṛũ;
- WK and Gujari the same with ū̃ (goṛū̃, gʰoṛū̃);
- OG `a m > u ṁ / _ #` (Ap. -aüṁ: gʰoḍauṁ).
OMarw was tested and left out (ext −0.0001; no neuter); the Rajasthani survey lects have no neuter.

| lect | train plain before → after | train ext before → after | test plain / ext after | -akam share (train) |
|---|---|---|---|---|
| G | 0.3638 → 0.3638 | 0.2883 → 0.2625 | 0.3652 / 0.2662 | 13 % |
| OG | 0.3106 → 0.3094 | 0.2496 → 0.2414 | 0.3108 / 0.2462 | 7 % |
| WK | 0.4877 → 0.4877 | 0.4045 → 0.3828 | 0.6082 / 0.5279 | 12 % |
| Gujari | 0.4837 → 0.4831 | 0.3994 → 0.3974 | 0.4679 / 0.3558 | 1 % |

"Before" here is the chain just before this addendum, scored on the 14:20 menu. The accent score
is unchanged (G 0.8565 train, n = 5; OG 0.6429, n = 1). Frontiers were rerun for G, OG, WK
and Goj. So neuter -ũ < -akam is a Gujaratic isogloss (G, OG, Koli). Gujari shows only a trace
(1 %), and Old Marwari and Rajasthani show none.

## 11. Verb citation forms (round 5, cut short at 18:27)

Gujarati -vũ and Koli -vū̃ go back to the neuter seṭ gerundive -itavyakam (Ap. -evvaü). The verb menu offers root + -tavyakam, and the chain assimilated C + t (likʰtavya > litvũ), so -anakam (-ṇũ) was picked instead. The fixes use sound changes only:
- the neuter rules `a m > ũ / _ #` and `a ũ > ũ / _ #` (G; WK/Gujari ū̃; OG `a m > u ṁ`) were retagged from `@morph` to `@final-nasal` / `@hiatus`, so they now run for verbs too;
- G and WK: `∅ > i / C _ t a v y a` (@x-rg-set-i: connecting -i- of seṭ -itavya after a consonant root);
- G and WK: `e > ∅ / V C* _ v ũ #` (WK: `ū̃`; @syncope: likʰevũ > likʰvũ).

| lect | train verb pairs | plain | verb menu before → after | train plain / noun-ext |
|---|---|---|---|---|
| G | 1100 | 0.2618 | 0.3296 → 0.2218 (tavyakam wins 898) | unchanged |
| WK | 21 | 0.4244 | 0.4716 → 0.4542 | unchanged |

Rejected: `t > ∅ / C _ a v y a` (verb −0.1046, but it is ad hoc). OG and OMarw need nothing: their verb menu (ati/bare) already beats plain. Not done: Marwari (-ṇo < -anaka; the menu already beats plain, 0.334 vs 0.353), Gujari and the survey lects. The survey lects have no verb-flagged etyma, so the verb menu does not apply to them. Accent score unchanged (G, n = 5).

## §12 Gujarati residual pass (all 4,814 misses classified)

Error share: endings 41%, consonant loss/insertion 18%, epenthesis/syncope 14%, consonant quality
8%, vowel quality 7%, length 7%. Changes (train/test): ext 0.2500/0.2514 → 0.2318/0.2350, verb menu
0.2033/0.2003 → 0.1913/0.1915; plain +0.002/+0.001 (bare headwords scored against extended forms:
syncope that now waits for the suffix vowel no longer helps the bare form by accident).

- **Final vowel loss before medial syncope** (and the `if:root` citation rule before both, so
  consonant-final nouns do not become -vũ verbs): a final closed syllable keeps its vowel
  (aŋgana > ā̃gaṇ, āstara > ātʰar, aŋgula > ā̃gaḷ, ālasya > āḷas; ~155 pairs). The same reorder was
  accepted for Khetrani and Saraiki; rejected for Lomavren, Zargari, Wadiyara Koli.
- **y glide** in final -io/-iũ (ucʰiyɔ, kʰā̃ḍiyũ): ext −0.005.
- **-aī > -ī** (amī, gʰī), **-āo > -ɔ** (ojʰɔ, sāmɔ).
- **Verb menu (all lects): causative gerundive -āpitavyakam** (G -āvvũ: apāvvũ, gumāvvũ; Or -āibā):
  beats the -āritavyakam placebo in G and Or, neutral elsewhere.

Rejected: final -Cr > -C (kʰetar also attested), ε > a before -vũ after a cluster (āmãtravũ is the
exception; kʰaḷakvũ the rule). Remaining: citation variants (-ɔ/-ũ/-ī/bare, 39 + 28 + 22), l/ḷ, the
first-syllable length of degeminated roots (apvũ for āpvũ).

## Round 6 audit

Method as in BRIEF-audit.md: `residuals.py` for each lect, a tally of the aligned non-final edit
operations (`tmp/sound_changes/rajasthani-gujarati/r6/ops.py`), candidates gated one at a time
with `gates.py` (batch runner `r6/gate.py`), word-level diffs (`r6/diffw.py`), then the accepted
ones installed together. G was done by the coordinator (§12); its four fixes were tried in every
other lect. "Before" is the rule set at 00:42, rebuilt in `r6/before/`. Rules written as
`(as G §12)` in the files are the ported ones. Accent scores are unchanged everywhere.

### Scores (before → after)

| lect | kept | ext train | ext test | plain train | plain test | verb train | verb test | ext-exact test |
|---|---|---|---|---|---|---|---|---|
| OG (Old Gujarati) | neuter `a m > u ṁ` retagged from @morph to a sound change (verbs now reach -ivaüṁ); `a > ∅ / V _ u #`; `a > ∅ / i _ v v`; `∅ > y / {ā e} _ a #` | 0.2356→0.2277 | 0.2450→0.2372 | 0.3094→0.3088 | 0.3108→0.3093 | 0.1636→0.1542 | 0.1719→0.1642 | 27.4%→28.0% |
| OMarw (Old Marwari) | `v > ∅ / a _ i #` (-āmi > -avi > -ai: kāḍʰai, karāi); `∅ > y / {ā e} _ a #` | 0.2671→0.2638 | 0.2608→0.2534 | 0.3213→0.3210 | 0.3151→0.3151 | 0.2306→0.2207 | 0.1715→0.1449 | 25.0%→28.6% |
| Marw (Marwari) | -aī > -ī; y-glide -io > -iyo; `%source CDIAL` block ḍ > ṛ / V _ V (CDIAL Marw. writes the flap: kaṛo, gʰaṛī, ū̃ṛo) | 0.3903→0.3839 | 0.4186→0.4082 | 0.4524→0.4475 | 0.4713→0.4651 | 0.3260→0.3121 | 0.2708→0.2431 | 8.3%→9.6% |
| mewari_basad (Malvi) | -aī > -ī; `l > ḷ / V _ V` (Varghese writes ḷ 69/119) | 0.4091→0.4069 | 0.4653→0.4640 | 0.4843→0.4841 | 0.5352→0.5344 | 0.0000→0.0000 | 0.0000→0.0000 | 3.6%→3.6% |
| mewari_dholpura (Mewari) | y-glide | 0.4102→0.4070 | 0.4505→0.4496 | 0.4860→0.4857 | 0.5295→0.5295 | 0.0000→0.0000 | 0.0000→0.0000 | 2.6%→2.6% |
| Goj (Gujari) | -aī > -ī; y-glide | 0.3972→0.3931 | 0.3615→0.3615 | 0.4827→0.4834 | 0.4692→0.4692 | 0.0000→0.0000 | 0.0000→0.0000 | 13.2%→13.2% |
| had (Hadauti) | y-glide | 0.4042→0.3990 | 0.4361→0.4254 | 0.4820→0.4820 | 0.5189→0.5189 | 0.0000→0.0000 | 0.0000→0.0000 | 7.7%→7.7% |
| dhundari_badagaon (Shekhawati) | y-glide | 0.4259→0.4209 | 0.4004→0.3953 | 0.4892→0.4892 | 0.4705→0.4705 | 0.0000→0.0000 | 0.0000→0.0000 | 10.8%→10.8% |
| Nimadi (Nimadi) | -aī > -ī (plain +0.0017 from bare headwords; 7 better, 0 worse under ext) | 0.3740→0.3731 | 0.3940→0.3940 | 0.4499→0.4516 | 0.4582→0.4582 | 0.0000→0.0000 | 0.0000→0.0000 | 9.0%→9.0% |
| bagri_fatehabad (Bagri) | y-glide | 0.4018→0.3996 | 0.3914→0.3825 | 0.4764→0.4764 | 0.4489→0.4489 | 0.0000→0.0000 | 0.0000→0.0000 | 12.9%→12.9% |
| dhundari_bamore (Dhundari) | -aī > -ī | 0.3859→0.3851 | 0.4030→0.4030 | 0.4623→0.4634 | 0.4616→0.4616 | 0.0000→0.0000 | 0.0000→0.0000 | 10.3%→10.3% |
| mewati_akera (Mewati) | y-glide | 0.3390→0.3361 | 0.3752→0.3724 | 0.4412→0.4412 | 0.4587→0.4587 | 0.0000→0.0000 | 0.0000→0.0000 | 16.1%→16.1% |
| marwari_bagra (Godwari) | y-glide; -āo > -o | 0.3933→0.3916 | 0.4420→0.4337 | 0.4626→0.4625 | 0.5146→0.5146 | 0.0000→0.0000 | 0.0000→0.0000 | 5.1%→5.1% |
| hadothi_kelwada (Shahawadi) | y-glide | 0.3315→0.3262 | 0.3723→0.3644 | 0.4168→0.4168 | 0.4474→0.4474 | 0.0000→0.0000 | 0.0000→0.0000 | 22.0%→22.0% |
| mewati_jhambaus (Ahirwadi) | — (nothing passed) | 0.3077→0.3077 | 0.3394→0.3394 | 0.3781→0.3781 | 0.4265→0.4265 | 0.0000→0.0000 | 0.0000→0.0000 | 14.6%→14.6% |
| Rj (Rajasthani) | y-glide | 0.4202→0.4133 | 0.3700→0.3666 | 0.4935→0.4935 | 0.4717→0.4717 | 0.4286→0.4286 | 0.0000→0.0000 | 18.6%→18.6% |
| WK (Wadiyara Koli) | -aī > -ī; y-glide | 0.3861→0.3826 | 0.5318→0.5318 | 0.4877→0.4877 | 0.6082→0.6082 | 0.4449→0.4449 | 0.6410→0.6410 | 7.3%→7.3% |

### Which G §12 fixes carry over

- **y glide in final -io** (`∅ > y / i _ o #`; māriyo, dekʰiyo, evāḷiyo) is accepted in 12 lects:
  Marwari, Mewari, Hadauti, Shekhawati, Bagri, Mewati, Godwari, Shahawadi, Rj, Gujari, Koli
  (ext −0.001 … −0.007 train, test up to −0.011). It is neutral in Nimadi, Dhundari and Malvi.
  In Ahirwadi test got worse (+0.0016).
- **-aī > -ī** (`a ī > ī / C _ #`) is accepted in Malvi, Marwari, Gujari, Nimadi, Dhundari and
  Koli (bījalī, tūṭī). It is neutral or worse in the other survey lects and in OMarw, and only
  marginal in OG (not kept).
- **-āo > -o** helps only Godwari. It is rejected in 11 lects (+0.0002 … +0.0017).
- **Final vowel loss before syncope**: nothing to port. Only WK has syncope, and the coordinator
  already rejected the reorder there.
- **-aka > -o as a sound change** (`a a > o / _ #`, for the lects that reach -o only through the
  `if:!ext` citation rule) is rejected in Malvi (+0.0132) and Nimadi (+0.0073). Their
  questionnaires often cite the oblique/plural -a/-ā (keḷa, kaḷa, kānaḍā), and bare -aka > -a
  matches those. The masculine -o of these lects stays a citation choice. It is not the regular
  outcome of -aka.

### Lect-specific findings

- **OG** (error share: endings 32 %, consonant loss 20 %, vowel quality 12 %): Ap. -aü after a
  vowel loses its a (aṇurāu, āgʰu, karahu: 17 pairs). The gerundive -iavva- contracts to -ivva-
  (valivaüṁ). A glide appears in -āya/-eya (gʰāya, teya). The OG neuter rule had been retagged
  `@morph` by someone else, so verbs could not reach -ivaüṁ; it is a sound change again.
  Rejected: intervocalic v-loss (+0.0087); no anusvāra lengthening (+0.0055); G-type
  compensatory lengthening (+0.0053); -avi > -ai (+0.0002).
- **OMarw**: the 1sg/verb-menu forms in -āmi end in -ai, not -avi, so v (< m) is lost before
  final -i (verb test 0.1715 → 0.1449). Rejected: flap ḍ > ṛ (+0.0003); dropping the word-initial
  {i u} lengthening. That one gives ext −0.0025, but plain +0.0021, and the gains are almost all
  ut-/ud- prefix words (ugʰāṛo, ujāṛi, ulaṭo) against bare nouns such as dūkāla, bīnatī, so it is
  not the bare-vs-extended artefact.
- **Marwari**: 203 CDIAL pairs write the flap ṛ where the two surveys write ḍ, which is a
  notation block. Verb test 0.2708 → 0.2431. Widening the -aḍa-/-ala- syncope to final -i/-ā is
  rejected (+0.0007).
- **Malvi**: ḷ restored. Rejected: undoing ṇ > n (+0.0052); s > h (+0.0010 train; the Basad
  point has it, Varghese mostly not); disyllabic compensatory lengthening (ext −0.0017, but plain
  +0.0024 — the two sources disagree, Varghese writes no lengthening); syncope before -Cī (3
  words only).
- **Gujari**: compensatory lengthening is rejected again (+0.0123), as are first-syllable
  u > ū and initial y > j (Eastern Gujari keeps j; train −0.0009 but test +0.0026). Mixed sources.

### What remains (all lects)

The ending classes (42–57 % of the error) are mostly citation variation: bare vs -o vs plural -ā,
fem. -ī, and the survey questionnaires' mixed forms. The -akka/-ikka variants leave a stray -k
in 27 Hadauti misses; they are chosen as stand-ins for -ā. The medial 'a → ∅' class (85–107 per
lect) is lexical syncope, with no positional rule surviving the gates. l/ḷ and ḍ/ṛ vary within
sources, and s > h is a Basad/Mewari feature inside mixed lects.

### Round 6 second pass

"Before" is the rule set after the first pass (`r6/before2/`). Both sides were rescored with the
engine as fixed mid-round (source tails in `residuals.py`, IPA stress marks, the verb_form ś fix).

| lect | ext train | ext test | plain train | plain test | verb train | verb test | ext-exact test |
|---|---|---|---|---|---|---|---|
| G | 0.2318→0.2296 | 0.2350→0.2316 | 0.3660→0.3650 | 0.3664→0.3662 | 0.1913→0.1881 | 0.1915→0.1834 | 33.1%→33.6% |
| OG | 0.2277→0.2272 | 0.2372→0.2360 | 0.3088→0.3085 | 0.3093→0.3078 | 0.1542→0.1542 | 0.1642→0.1642 | 28.0%→28.0% |
| OMarw | 0.2638→0.2633 | 0.2534→0.2534 | 0.3210→0.3210 | 0.3151→0.3151 | 0.2207→0.2185 | 0.1449→0.1449 | 28.6%→28.6% |
| Marw | 0.3839→0.3816 | 0.4082→0.4057 | 0.4475→0.4491 | 0.4651→0.4637 | 0.3121→0.2767 | 0.2431→0.1986 | 9.6%→10.1% |
| had | 0.3990→0.3977 | 0.4254→0.4190 | 0.4820→0.4823 | 0.5189→0.5128 | 0.0000→0.0000 | 0.0000→0.0000 | 7.7%→7.7% |
| Nimadi | 0.3731→0.3728 | 0.3940→0.3940 | 0.4516→0.4512 | 0.4582→0.4582 | 0.0000→0.0000 | 0.0000→0.0000 | 9.0%→9.0% |
| mewati_akera | 0.3361→0.3303 | 0.3724→0.3606 | 0.4412→0.4408 | 0.4587→0.4469 | 0.0000→0.0000 | 0.0000→0.0000 | 16.1%→17.7% |
| Rj | 0.4133→0.4100 | 0.3666→0.3582 | 0.4935→0.4924 | 0.4717→0.4633 | 0.4286→0.4286 | 0.0000→0.0000 | 18.6%→18.6% |
| WK | 0.3826→0.3804 | 0.5318→0.5318 | 0.4877→0.4877 | 0.6082→0.6082 | 0.4449→0.4313 | 0.6410→0.6410 | 7.3%→7.3% |

Unchanged in the second pass: mewari_basad, mewari_dholpura, Goj, dhundari_badagaon, bagri_fatehabad, dhundari_bamore, marwari_bagra, hadothi_kelwada, mewati_jhambaus.

**Kept (all passed `gates.py`; word diffs checked):**
- **G `h > ∅ / C _ | _ C`** after syncope (durlabʰa > dulɔ, *dēhalla > delī, *lābʰana > lāṇī,
  paṭaha > paṛɔ): ext −0.0017/−0.0014. 109 reflexes better, 35 worse. The worse ones are a
  retained h (dohṇũ, mεhṇũ, cāhvũ), so this is a strong tendency, not an exceptionless change.
- **G compensatory lengthening also before the -evv- of -itavya** (the ε is later lost, so the
  syllable counts as light): *cipp > cīpvũ, arpáyati > āpvũ, *caṭṭ > cāṭvũ. This is the apvũ/āpvũ
  residue the coordinator flagged. Verb −0.0028/−0.0081, ext −0.0006/−0.0021; 51 better, 35
  worse. The worse ones are -ya- presents (pacyátē > pacvũ, lipyatē > lipvũ, sidʰvũ), where the
  verb menu builds the root with the y, so a false geminate is lengthened.
- **Stop + y/v/l/nasal assimilation restored** in Mewati (Akera), Hadauti, Rj and Marwari (round 1
  had pruned it on plain NED): trúṭyati > ṭūṭ-, aŋguṣṭʰya > gū̃ṭʰī. Ext −0.0052/−0.0055 (Mewati),
  −0.0013/−0.0063 (Hadauti), −0.0003/−0.0046 (Rj), −0.0007/−0.0013 (Marwari). Shekhawati
  rejects it (+0.0009).
- **Other pruned shared rules restored.** I tried every group-core rule a lect lacked
  (`r6/crossmiss.py`). Kept: OG ṇṇ > nn (−0.0005/−0.0012), Rj kṣ > kkʰ (−0.0030/−0.0039),
  WK intervocalic p > v (ext −0.0022, verb −0.0136), Mewati n > ṇ (−0.0006/−0.0063), Nimadi
  initial-cluster simplification (−0.0003). Rejected: l > ḷ in OG and OMarw (+0.013), the
  nasal-cluster rule in Malvi, Gujari and Godwari (+0.005 … +0.013: they write NC clusters),
  n > ṇ in Gujari.
- **`%verb-menu +aka`** (root + -aka verbal noun) in Marwari: verb −0.0307/−0.0444, ext
  −0.0016/−0.0012. Placebos +ara and +ala both give 0.0000. OMarw: verb −0.0022/0.0000, placebos
  0, kept. It is neutral in G and WK and negligible in OG (−0.0005). Rj has one train verb, so
  it was not installed there. The other survey lects have no verb-flagged etyma.

**Rejected, with reasons:**
- G: medial `a v a > o` (ext train −0.0007 but test +0.0005); v lost after a consonant (plain
  +0.021); intervocalic h-loss (+0.0008); rd > ḍḍ (+0.0007); i/u > a in a final closed syllable
  (ext −0.0006, but 35 better against 28 worse: tatsama dāruṇ, karuṇ, taruṇ keep u — lexical).
  The -Cr and l/ḷ residue is left as lexical (tatsama kʰetar, ḷ vs l within the same entries).
- Surveys: -ada-/-aya- > e (kadala > keḷā, nayana > neṇ) passes the gates in Hadauti,
  Shekhawati, Bagri and Marwari, but rests on about 3 words and hits -ayati verbs (māreo). Also
  cʰ > c (+0.001 … +0.005 everywhere), s > h in Marwari, Rj and Hadauti (+0.006 … +0.016), and
  the widened -aḍa-/-ala- syncope.
- Rule deletions (`r6/dropcheck.py`, each rule of each non-G lect) show two things:
  - Removing the survey citation rules (`a > o / C _ #`, the verb `> o` rule) would cut ext by
    0.018–0.033 in Hadauti, Bagri, Shekhawati, Godwari, Marwari and Mewari, but costs plain
    +0.006 … +0.044. This is the clearest bare-vs-extended artefact in the group: the survey
    masculine -o *is* the -aka extension. I left the rules in place, because they are the
    coordinator's `if:!ext` citation rules; it is flagged here for decision.
  - Removing -ikā > -ī (Malvi, Nimadi, Dhundari, Mewati, Ahirwadi) or -aka > -o (Shekhawati)
    also passes the gates (ext −0.001 … −0.010). Those gains come from plural citations (-iyā,
    -ā) matched by the uncontracted -iā/-a. The contractions are real, so they stay.
  - Removing Rj's final-consonant loss passes only because the -akam variant then leaves a
    surface -am. Removing Ahirwadi's aphaeresis turns on 2 words.
  - OMarw's word-initial {i u} lengthening: ext −0.0025/−0.0057 without it, but plain +0.0021,
    carried by ut-/ud- prefix words against bare nouns (dūkāla, bīnatī). Kept.

**Remaining:** survey citation variation (bare / -o / -ā / -ī / -iyā). Lexical syncope. l/ḷ, ḍ/ṛ
and s/h vary within the mixed sources (Basad vs Varghese in Malvi; the Marwari vs Mewari
surveys). Gujari's missing lengthening, a north-western trait (§4).

### Round 6 third pass: citation -o dropped, and near-misses

"Before" is `r6/before3/`, the rules after the second pass.

| lect | ext train | ext test | plain train | plain test | verb train | verb test | ext-exact test |
|---|---|---|---|---|---|---|---|
| Marw | 0.3816→0.3562 | 0.4057→0.3804 | 0.4491→0.4881 | 0.4637→0.4923 | 0.2767→0.2374 | 0.1986→0.1278 | 10.1%→14.2% |
| mewari_basad | 0.4069→0.3994 | 0.4640→0.4507 | 0.4841→0.5040 | 0.5344→0.5471 | 0.0000→0.0000 | 0.0000→0.0000 | 3.6%→6.5% |
| mewari_dholpura | 0.4070→0.3894 | 0.4496→0.4245 | 0.4857→0.5102 | 0.5295→0.5528 | 0.0000→0.0000 | 0.0000→0.0000 | 2.6%→6.2% |
| Goj | 0.3931→0.3920 | 0.3615→0.3599 | 0.4834→0.4834 | 0.4692→0.4692 | 0.0000→0.0000 | 0.0000→0.0000 | 13.2%→13.2% |
| had | 0.3977→0.3794 | 0.4190→0.4074 | 0.4823→0.5119 | 0.5128→0.5382 | 0.0000→0.0000 | 0.0000→0.0000 | 7.7%→9.5% |
| dhundari_badagaon | 0.4209→0.3890 | 0.3953→0.3630 | 0.4892→0.5030 | 0.4705→0.4810 | 0.0000→0.0000 | 0.0000→0.0000 | 10.8%→15.8% |
| Nimadi | 0.3728→0.3511 | 0.3940→0.3827 | 0.4512→0.4717 | 0.4582→0.4739 | 0.0000→0.0000 | 0.0000→0.0000 | 9.0%→12.3% |
| bagri_fatehabad | 0.3996→0.3658 | 0.3825→0.3559 | 0.4764→0.4821 | 0.4489→0.4676 | 0.0000→0.0000 | 0.0000→0.0000 | 12.9%→18.8% |
| dhundari_bamore | 0.3851→0.3647 | 0.4030→0.3761 | 0.4634→0.4811 | 0.4616→0.4968 | 0.0000→0.0000 | 0.0000→0.0000 | 10.3%→14.9% |
| marwari_bagra | 0.3916→0.3662 | 0.4337→0.4085 | 0.4625→0.4948 | 0.5146→0.5519 | 0.0000→0.0000 | 0.0000→0.0000 | 5.1%→7.7% |

Unchanged: G, OG, OMarw, mewati_akera, hadothi_kelwada, mewati_jhambaus, Rj, WK.

**Citation -o dropped** (coordinator's decision): `a > o / C _ #` (`if:!ext` in some files).
- Removed in Marwari, Mewari, Hadauti, Shekhawati, Bagri and Godwari.
- Removed in Malvi, Dhundari and Nimadi together with the sound change `a a > o / _ #`
  (-aka > -ao > -o), because those chains otherwise had no -o source at all.
- Where the chain did not already lose final -a, `a > ∅ / C _ #` was added: Marwari, Mewari,
  Malvi, Dhundari, Nimadi. In Marwari it alone gives ext −0.018/−0.021 and plain −0.006.
- Nimadi's removal failed the gates without apocope (ext test +0.004). With apocope it passes
  (ext −0.0217/−0.0112).

Plain NED rises by 0.006–0.039. This is the bare-vs-extended artefact: the attested -o forms
continue -aka, and the menu derives them. In Hadauti, 75 reflexes got better and 5 worse.
- Better (bare stems): bāpo > bāp, peṭo > peṭ, dūdʰo > dūdʰ, dūro > dūr, ājo > āj.
- Worse: masculine -o words whose -aka form the chain misses (kāṇṭo for kā̃ṭɔ, now kaṇṭo;
  kānd for kāndɔ).
- Ext test exact matches rise sharply: Shekhawati 10.8 % → 15.8 %, Bagri 12.9 % → 18.8 %,
  Marwari 10.1 % → 14.2 %.

**Near-misses (NED ≤ 0.25, by differing segment pair; `r6/near.py`):**
- **Marwari**: verbal nouns in -aṇo keep a medial a the data lacks (hasaṇo for hasṇo, jāgaṇo,
  kʰodaṇo, dʰaraṇo). The -aḍa-/-ala- suffix syncope now covers -aṇa- too:
  `a > ∅ / V C (C) _ {ḍ ḷ l ṛ ṇ n} {o ī} #`. Verb −0.039/−0.071, ext −0.0019/−0.0019. Gujari
  and Bagri take -ara- instead (camṛī, kukro): ext −0.0011/−0.0016 and −0.0006.
- **G**: the top near-miss pairs are:
  - ∅ → a before ex-geminates: *aḍakk- > aṛkvũ for aṛakvũ. Syncope sees the degeminated form;
    running degemination after syncope is rejected (ext +0.0044).
  - -iyo → -iyũ: 56 pairs. The neuter -iyakam is not on the menu; I recommend adding it.
  - ā → a in -āvvũ causatives: tested, no gain.
  - ī → i in -īCvũ roots: excluding i from lengthening gives +0.0011.
  - Rhythmic ā-shortening after a long vowel: +0.0014.
  - rd > ḍḍ: +0.0007.
  No change kept.
- **OG**: medial a+ā contraction (+0.0013) and medial y glides (train +0.0005, test −0.0041) are
  rejected. The rest is Apabhraṃśa spelling variation (-u/-aü, -i).
- **Rj**: only 1–3 pairs per type. Nothing systematic.

### Round 6: `%ext-menu +iyakam` (neuter -iyakam > G -iyũ)

Each item is tested against the placebo `+irakam` (gates, ext train / test):

| lect | +iyakam | placebo +irakam | kept |
|---|---|---|---|
| G | −0.0022 / −0.0019 | −0.0005 / −0.0006 | yes |
| OG | −0.0013 / −0.0024 | −0.0003 / −0.0010 | yes (-iyauṁ) |
| WK | −0.0005 / 0.0000 | −0.0005 / 0.0000 | no (ties the placebo) |
| Gujari, OMarw | 0.0000 | 0.0000 … −0.0019 | no |
| Marwari, Hadauti, Mewari | 0.0000 | −0.0003 … −0.0014 | no (no neuter) |

Plain, verb and accent scores are flat in every case. The neuter -iyũ is a Gujarati/Old Gujarati
feature only, consistent with the -akam result in §10.5.

### Round 6 fourth pass

"Before" is `r6/before4/`.

| lect | ext train | ext test | plain train | plain test | ext-exact test |
|---|---|---|---|---|---|
| Marw | 0.3562→0.3511 | 0.3804→0.3725 | 0.4881→0.4881 | 0.4923→0.4923 | 14.2%→14.2% |
| mewari_basad | 0.3994→0.3884 | 0.4507→0.4363 | 0.5040→0.5040 | 0.5471→0.5471 | 6.5%→6.5% |
| Goj | 0.3920→0.3891 | 0.3599→0.3534 | 0.4834→0.4834 | 0.4692→0.4692 | 13.2%→14.3% |
| had | 0.3794→0.3596 | 0.4074→0.3871 | 0.5119→0.5119 | 0.5382→0.5382 | 9.5%→11.3% |
| dhundari_badagaon | 0.3890→0.3718 | 0.3630→0.3514 | 0.5030→0.5030 | 0.4810→0.4810 | 15.8%→19.2% |
| Nimadi | 0.3511→0.3455 | 0.3827→0.3761 | 0.4717→0.4717 | 0.4739→0.4739 | 12.3%→12.3% |
| bagri_fatehabad | 0.3658→0.3583 | 0.3559→0.3495 | 0.4821→0.4821 | 0.4676→0.4676 | 18.8%→19.8% |
| dhundari_bamore | 0.3647→0.3567 | 0.3761→0.3603 | 0.4811→0.4811 | 0.4968→0.4968 | 14.9%→14.9% |
| mewati_akera | 0.3303→0.3228 | 0.3606→0.3492 | 0.4408→0.4408 | 0.4469→0.4469 | 17.7%→19.4% |
| marwari_bagra | 0.3662→0.3558 | 0.4085→0.3993 | 0.4948→0.4948 | 0.5519→0.5519 | 7.7%→9.0% |
| hadothi_kelwada | 0.3262→0.3158 | 0.3644→0.3465 | 0.4168→0.4168 | 0.4474→0.4474 | 22.0%→24.4% |
| Rj | 0.4100→0.3821 | 0.3582→0.3340 | 0.4924→0.4924 | 0.4633→0.4633 | 18.6%→23.3% |

Unchanged: G, OG, OMarw, mewari_dholpura, mewati_jhambaus, WK.

**Shared finding: the survey plural citation -ā.** The largest near-miss pairs are the same in
all six lects the coordinator named:
- `k → ∅ #` (7–21 pairs): the -akka variant yields -āk/-ak and wins as a stand-in for -ā/-a.
- `∅ → ā #` / `∅ → a #` (5–24 pairs): kāndā, āmbā, kāḷā, gāra, ḍāvā.

The questionnaires cite many masculine nouns in the plural (-ā < -akāḥ), which the menu lacks.

`%ext-menu +akā` (masc. pl. -akāḥ > -ā) against the placebo `+arā`, ext train/test:

| lect | +akā | placebo +arā |
|---|---|---|
| Rj | −0.0279 / −0.0242 | −0.0089 / −0.0092 |
| Hadauti | −0.0198 / −0.0203 | −0.0052 / −0.0053 |
| Shekhawati | −0.0173 / −0.0116 | 0.0000 / 0.0000 |
| Malvi | −0.0110 / −0.0145 | −0.0057 / −0.0077 |
| Shahawadi | −0.0105 / −0.0179 | −0.0038 / −0.0033 |
| Godwari | −0.0103 / −0.0092 | −0.0041 / −0.0027 |
| Dhundari | −0.0080 / −0.0159 | −0.0018 / −0.0056 |
| Bagri | −0.0075 / −0.0064 | −0.0016 / −0.0029 |
| Mewati | −0.0074 / −0.0114 | −0.0021 / −0.0049 |
| Nimadi | −0.0055 / −0.0067 | −0.0024 / −0.0034 |
| Marwari | −0.0052 / −0.0079 | −0.0025 / −0.0019 |
| Gujari | −0.0029 / −0.0065 | −0.0011 / −0.0043 |
| Mewari (Dholpura) | −0.0014 / −0.0009 | −0.0025 / −0.0027 — not kept |
| Ahirwadi | −0.0012 / −0.0033 | 0.0000 / −0.0033 — not kept (ties on test) |

It is installed in the 12 lects where it beats the placebo on train and test. Plain, verb and
accent scores are flat.

In Ahirwadi the masculine singular itself is -ā (§4), so -akā adds little there. Mewari's
plurals are mostly -iyā/-īā forms (andārā, āmbīā), covered by the -ikā variant.

**Tested and rejected:** `%ext-menu +itaka` for the past-participle -yo citations (dekʰyɔ,
māryɔ) gives 0.0000 in all six. These etyma are verbs by gloss but carry no `verb` flag in the
survey lects, so neither menu reaches them — worth checking in `word_flags`. The -ada-/-aya- > e
pattern (keḷɔ, gelɔ, nerɔ) recurs in all six, but in only about 3 words each (see the second pass).

**Residue in the non-ending classes:**
- medial a-loss before -Cī is lexical (kūkḍī but kūkaḍo);
- geminates are inserted in -o forms (kaddɔ, dāddɔ, rāttɔ);
- ḷ/r (kāro) and s/h vary by source;
- deaspiration of dʰ in Hadauti occurs in 3 words.
None of these passes as a regular rule.

### Round 6 fifth pass: survey verbs (after the word_flags fix)

With glosses such as "he killed" and "come!, he came" now flagged, the survey lects have verbs
(Marwari 70 train pairs, Malvi 83, Gujari 44, Mewari 35, Hadauti 27, …). I rebaselined them
(`r6/before5/`) and gave them the past-participle citation.

- `%verb-menu +itaka` (-ita + -aka > -iyo/-yo; placebo +iraka) beats the placebo in Malvi,
  Marwari, Gujari, Mewari, Hadauti, Bagri, Shekhawati, Godwari, Dhundari and Rj.
  - The placebo is 0.0000 in six of them and −0.002 … −0.022 on verb train in Marwari,
    Mewari, Hadauti and Godwari, always well below the real item.
  - No effect in Nimadi, Mewati, Shahawadi or Ahirwadi, which have few or no -yo citations.
- Surfacing -yɔ needs the i to drop after a consonant. With the -iyo glide already present
  (Marwari, Mewari, Hadauti, Gujari): `i > ∅ / C _ y o #` (māryɔ, dekʰyɔ, pʰūṭyo). In Malvi,
  which has no glide: `i > y / C _ o #`.
  - In Bagri, Shekhawati, Godwari and Mewati the i-loss makes ext test worse (+0.001 … +0.002).
    There the citation is -iyo, so only the menu item is kept.

| lect | verb train | verb test | ext train | ext test | plain train | plain test |
|---|---|---|---|---|---|---|
| Marw | 0.2918→0.2680 | 0.2152→0.2152 | 0.3507→0.3474 | 0.3729→0.3723 | 0.4881→0.4884 | 0.4923→0.4923 |
| mewari_dholpura | 0.4351→0.3444 | 0.3905→0.3262 | 0.3949→0.3891 | 0.4303→0.4268 | 0.5102→0.5106 | 0.5528→0.5528 |
| had | 0.3685→0.3364 | 0.4583→0.3000 | 0.3621→0.3592 | 0.3891→0.3861 | 0.5119→0.5119 | 0.5382→0.5382 |
| Goj | 0.3542→0.3455 | 0.0000→0.0000 | 0.3876→0.3864 | 0.3534→0.3534 | 0.4834→0.4834 | 0.4692→0.4692 |
| mewari_basad | 0.4461→0.3860 | 0.3333→0.2600 | 0.3902→0.3825 | 0.4385→0.4343 | 0.5040→0.5040 | 0.5471→0.5471 |
| bagri_fatehabad | 0.3842→0.3585 | 0.3750→0.3750 | 0.3625→0.3609 | 0.3520→0.3520 | 0.4821→0.4821 | 0.4676→0.4676 |
| dhundari_badagaon | 0.3730→0.3491 | 0.3125→0.3125 | 0.3721→0.3714 | 0.3523→0.3523 | 0.5030→0.5030 | 0.4810→0.4810 |
| marwari_bagra | 0.4580→0.3824 | 0.2500→0.2500 | 0.3620→0.3576 | 0.4025→0.4025 | 0.4948→0.4948 | 0.5519→0.5519 |
| dhundari_bamore | 0.4181→0.3875 | 0.0000→0.0000 | 0.3608→0.3599 | 0.3603→0.3603 | 0.4811→0.4811 | 0.4968→0.4968 |
| Rj | 0.4557→0.4290 | 0.2500→0.2500 | 0.3823→0.3813 | 0.3352→0.3352 | 0.4924→0.4924 | 0.4633→0.4633 |

Test verb pairs are 0–11 per lect, so the test verb column is noisy. Plain NED changes by at most
+0.0004.

## Round 7

**Scoring.** Vowel qualities are now scored as distinct: ɔ and ɛ no longer count as o and e (ε is
still ɛ). All accent and tone marks are dropped from the headline scores.

**Before and after.** Every number below uses the current engine, including the 08:20, 09:10, 09:25
and 10:35 fixes. "Before" is the pre-round-7 snapshot (`tmp/sound_changes/snapshots/pre-round7/rules/`)
scored with that same engine.

**Tools.** They are in `tmp/sound_changes/rajasthani-gujarati/r7/`:
- `mkcand.py`: builds candidates;
- `qd.sh`: sequential gate queue;
- `diffw.py`: word diffs, with source tails;
- `vsubs.py`: aligned vowel substitutions;
- `crossmiss.py`: group rules a lect lacks;
- `backup/`: every file before each install.

**Accent.** ext_acc and ext_tone are unchanged in every lect.

### Scores (pre-round-7 snapshot → now, same engine)

| lect | ext train | ext test | plain train | plain test | verb train | verb test | ext-exact test |
|---|---|---|---|---|---|---|---|
| G | 0.2587→0.2263 | 0.2615→0.2301 | 0.3720→0.3738 | 0.3723→0.3728 | 0.1928→0.1910 | 0.1872→0.1853 | 28.5%→34.7% |
| OG | 0.2253→0.2242 | 0.2336→0.2313 | 0.3085→0.3134 | 0.3078→0.3196 | 0.1463→0.1463 | 0.1581→0.1581 | 28.0%→28.0% |
| OMarw | 0.2612→0.2612 | 0.2516→0.2516 | 0.3211→0.3265 | 0.3151→0.3286 | 0.2168→0.2168 | 0.1459→0.1459 | 29.5%→29.5% |
| WK | 0.3804→0.3804 | 0.5318→0.5318 | 0.4877→0.5221 | 0.6082→0.6366 | 0.4313→0.4313 | 0.6410→0.6410 | 7.3%→7.3% |
| Marw | 0.4023→0.3576 | 0.4032→0.3657 | 0.5081→0.5125 | 0.5010→0.5009 | 0.3188→0.2788 | 0.2894→0.2333 | 12.4%→18.8% |
| Goj | 0.3927→0.3787 | 0.3554→0.3459 | 0.4893→0.4875 | 0.4714→0.4714 | 0.3653→0.3417 | — | 14.3%→15.3% |
| Rj | 0.4054→0.3785 | 0.3426→0.3300 | 0.5012→0.4949 | 0.4633→0.4633 | 0.4409→0.4139 | 0.2500→0.2000 | 23.3%→23.3% |
| mewari_basad | 0.3914→0.3727 | 0.4421→0.4214 | 0.5073→0.5131 | 0.5490→0.5508 | 0.4011→0.3725 | 0.2773→0.2727 | 6.2%→9.1% |
| mewari_dholpura | 0.4349→0.3801 | 0.4619→0.4270 | 0.5270→0.5287 | 0.5623→0.5618 | 0.4233→0.3416 | 0.4690→0.3548 | 4.6%→6.7% |
| marwari_bagra | 0.4311→0.3609 | 0.4550→0.3980 | 0.5241→0.5086 | 0.5609→0.5553 | 0.5005→0.4093 | 0.2500→0.2500 | 6.4%→10.3% |
| bagri_fatehabad | 0.4123→0.3790 | 0.3920→0.3639 | 0.5080→0.5011 | 0.4860→0.4840 | 0.4322→0.3310 | 0.3750→0.3500 | 14.9%→16.8% |
| dhundari_badagaon | 0.4182→0.3790 | 0.3956→0.3638 | 0.5224→0.5154 | 0.5017→0.4943 | 0.4083→0.3600 | 0.3125→0.3125 | 14.2%→16.7% |
| dhundari_bamore | 0.4195→0.3480 | 0.4321→0.3588 | 0.5022→0.4961 | 0.5126→0.5091 | 0.4667→0.3069 | — | 8.0%→16.1% |
| had | 0.4126→0.3629 | 0.4271→0.3893 | 0.5234→0.5166 | 0.5494→0.5388 | 0.3957→0.3398 | 0.4250→0.3000 | 7.7%→11.9% |
| hadothi_kelwada | 0.3609→0.3205 | 0.3607→0.3473 | 0.4286→0.4255 | 0.4515→0.4474 | 0.3933→0.3762 | — | 19.5%→24.4% |
| mewati_akera | 0.3638→0.3346 | 0.3669→0.3555 | 0.4565→0.4522 | 0.4509→0.4509 | 0.3127→0.2849 | 0.5000→0.5000 | 16.9%→18.5% |
| mewati_jhambaus | 0.3209→0.3202 | 0.3597→0.3597 | 0.3893→0.3893 | 0.4448→0.4448 | 0.1726→0.1726 | 0.5000→0.5000 | 12.2%→12.2% |
| Nimadi | 0.3771→0.3668 | 0.4188→0.3982 | 0.4853→0.4817 | 0.4914→0.4976 | 0.4645→0.4412 | 0.6528→0.6528 | 7.7%→9.7% |

**Plain losses.** They all have one of two causes:
- **The removed citation `@morph` rules** (item 6; see below). OMarw and WK change only through these
  removals (OG also gains two menu items), so their plain rise is pure bare-vs-extended.
- **Step 0's apocope of -ā/-i/-u** (Marw, Malvi, Nimadi). The bare headword loses a vowel that the
  -ikā/-aka/-akā variant restores.

### Step 0: the interrupted round-7 edits, re-gated

The previous agent left edits in 15 files. I tested them in two steps:
1. Each file, snapshot → current, as a whole.
2. Each diff hunk left out in turn (current vs current minus that hunk).

A hunk is kept when leaving it out makes ext train worse and does not make test better.

All survived except one. Godwari `hiatus.aaaa.cond` made test worse by 0.0004 and was reverted. It
came back later from the cross-lect pass (ext −0.0029/0, after the -ɔ and -ī changes), so it is in the
final file.

The kept hunks, with ext train/test when the hunk is left out:
- **`a {d y} a > e / _ C`** (MIA -aya-/-ada- > e: kadala > keḷɔ, gata- > gelɔ): 12 lects,
  +0.0001 … +0.0040. It also helps the verbs (-ayati): Marw verb +0.014 without it.
- **Group form of the -aḍa-/-ala-/-ara-/-aṇa- syncope** (adds r, ṇ): Marw, Gujari, Bagri, Mewati.
- **`hiatus.aaaa.cond`**: Gujari, Malvi, Ahirwadi.
- **Apocope of -ā/-i/-u** (Marw, Malvi, Dhundari-Bamore, Nimadi): ext +0.0045 … +0.0068 train and
  +0.012 test when left out. The previous agent's unrestricted form also deleted the vowel of
  monosyllables (gāva > *g, bā > *b). It is now `final-vowel-loss.aaiu.final-polysyllabic`
  (V C* _ #), which passes in all four; plain −0.0009 … −0.0031.
- **G's `{i u} > a / V C* _ C #`**: ext −0.0005/−0.0001. Rejected in round 6 on effect size;
  there is no threshold now.
- **The rest**: Rj `syncope.i.cyo`, Malvi -Cī syncope, `s-stop-aspirate.sk` (Bagri, Shekhawati),
  `assim-sonorant.nnmlvy` (Shekhawati), `assim-sonorant.sson` (Nimadi), and `glide.io` (Dhundari,
  Nimadi).

Bare `use` lines got mechanism comments.

### Focus 0: ɔ and ɛ

**Final -ɔ.** `o > ɔ / _ #` runs at the end of the main chain. In these varieties the masculine -o
(< -aka > -ao; past participle -iyo < -ita-ka; -āo) is open ɔ: gʰoṛɔ, keḷɔ, dekʰyɔ.
- Marwari: two of the three sources write ɔ. A `%source CDIAL` notation rule `ɔ > o / _ #` covers
  CDIAL, which writes 110 -o and no -ɔ.

| lect | ext train | ext test |
|---|---|---|
| Rj | −0.0217 | −0.0068 |
| Mewari (Dholpura) | −0.0273 | −0.0146 |
| Godwari | −0.0579 | −0.0399 |
| Bagri | −0.0283 | −0.0254 |
| Shekhawati | −0.0325 | −0.0281 |
| Dhundari | −0.0492 | −0.0579 |
| Hadauti | −0.0466 | −0.0342 |
| Shahawadi | −0.0365 | −0.0134 |
| Mewati | −0.0282 | −0.0107 |
| Marwari | −0.0338 | −0.0190 |

Plain and verb scores also improve.

Rejected:
- **Malvi**: the Mewari survey at Basad writes 80 -o against 41 -ɔ, and Varghese writes o. Even with
  a Varghese notation block the result is +0.0057/+0.0046.
- **Nimadi**: both sources are mixed (+0.0041 train). A Bareli-only notation block fails on verb
  (+0.0082).
- **Ahirwadi**: masculine -ā; verb +0.024.
- **Gujari**: writes o.

**Gujarati.** The source is CDIAL, which writes ɔ ε.
- **G1** `o > ɔ / _ #`: -aka > -ɔ; 966 -ɔ against 42 -o. Ext −0.0255/−0.0254; 728 reflexes better,
  21 worse.
- **G2** `a {o u} > ɔ`: replaces `hiatus.aou`. MIA a + u after consonant loss gives ɔ (catura > cɔrɔ,
  catuṣka > cɔk, *kaṭṭakūṭa > kāṭɔṛɔ; 50 ɔ against 9 o). Ext −0.0032/−0.0032.
- **G3** `a h {i ī} > ɛ / _ C`: replaces `h-loss.ahii`. Examples: abʰijñāna > ɛdʰāṇ, grahila > gʰɛlũ,
  mahilā > mɛḷā. Ext −0.0005/−0.0002.
- **Joint**: −0.0274/−0.0265.
- **Kept as e/o**: OIA ai/au > G e/o (tel, sel, motī), and the medial i + a contraction (kaṇer,
  divel; 25 e against 11 ɛ).
- **Rejected**:
  - aha > ahɛ (`a > ɛ / a h _ C`: gʰarɛṇ, lɛhɛr): train −0.0001, test +0.0005.
  - -avi- > ɛ (bɛsvũ, pɛsvũ): train +0.0002.
- **Initial ava-/apa-/upa-**: 48 o against 27 ɔ, so left as o.
- **Also for G**:
  - `repeat: a ũ > ũ / _ #`: an -aka stem + -akam gives -aaũ (āmalaka > āmḷũ, avaśīrṣaka > osīsũ).
    Ext −0.0005/−0.0015.
  - Polysyllabic apocope: −0.0002/−0.0000.
- **G test ext overall**: 0.2615 → 0.2301, including the neuter menu items below.

**Survey medial ɛ/ɔ is sporadic** (pɛṭ, rɛt, ɛk; cɔkkɔ, sɔnɔ), so it is not ruled:
- ē/ō before a cluster gives ɛ/ɔ in a minority of cases;
- -aya- gives e in most gold forms (keḷā, gelɔ);
- there is no o…ɔ harmony (o-ɔ and ɔ-ɔ are about equally common).

Other quality notes:
- Nimadi ə is Vunnamatla's reduced suffix vowel (lākəḍi beside lākaḍɔ, 17 against 25 a), so it is
  left alone.
- Gujari (Rensch) has e 46 against ɛ 36, with no conditioning found.

### Other changes kept

- **`aka.iaa.final`** (-ikā/-ika > -ia > -ī) in the lects that lacked it:
  - Hadauti: −0.0015/0, verb −0.012.
  - Godwari: −0.0012/0.
  - Bagri: −0.0005/−0.0002.
  - Rejected in Rj and Shekhawati (0 train).
  - Mewari (Dholpura) failed at first (test +0.0048: its uncontracted -īā stood in for the plural).
    After the +akā plural went in it passes: −0.0110/−0.0070, verb −0.0048.
- **Masculine plural -akāḥ.** Mewari (Dholpura) gets `%ext-menu +akā` and `a ā > ā / _ #` (-aā >
  -ā after apocope). Its apocope (`a > ∅ / C _ #`) leaves -aā, so the plural surfaces as -ā. Real item
  −0.0146/−0.0124 against placebo +arā −0.0025/−0.0027.
- **The same contraction in Gujari, Malvi and Nimadi.** It works only if apocope spares -ā after a:
  `{a i u} > ∅ / V C* _ #`, then `ā > ∅ / V C* _ # // a _`, then `a ā > ā / _ #`.
  - Gujari −0.0098/−0.0095; Malvi −0.0085/−0.0070; Nimadi −0.0032/−0.0046.
  - In the eight other +akā lects, verbs got worse (+0.013 … +0.08): Marw, Rj, Godwari, Shekhawati,
    Dhundari, Hadauti, Shahawadi, Mewati.
  - The Bagri word diff shows the reason: these sources cite the plural/oblique with short -a (gʰara,
    tāra, sona), which the unprotected -akā > -a matched.
  - A variant that also protected monosyllables (`final-vowel-loss.aaiu.vcc`) hurt verbs everywhere.
- **Group rules a lect lacked** (`crossmiss.py`: every `use` rule in ≥ 8 of the 14 modern lects,
  inserted after its predecessor; 67 candidates). Kept, ext train/test:
  - Hadauti `hiatus.a.postvocalic` −0.0003/−0.0036.
  - Shahawadi `vowel-shift.u.vccv.except` −0.0034/0.
  - Shekhawati `aphaeresis.a.cv` −0.0005/0.
  - Marwari `int-aspirate-h` −0.0001/0.
  - Godwari `aphaeresis.a.cv` + `hiatus.a.postvocalic` + `hiatus.aaaa.cond`, jointly
    −0.0086/−0.0139.
  - Nimadi `aphaeresis.a.cv` + `int-p-v`, jointly −0.0008/−0.0037.
  - The first batch's X-numbers were scrambled by set ordering and rerun deterministically, so only
    the Y results count.

- **First-syllable o > ɔ before a final -ɔ** (`o > ɔ / # C* _ C C* ɔ #`: sɔnɔ, cʰɔrɔ, tʰɔḍɔ). This is
  harmony with the new final ɔ, tested in the ten -ɔ lects.
  - Kept in Shekhawati only: −0.0027/−0.0015 (ɔ…ɔ 13 against o…ɔ 6).
  - Elsewhere test got worse (+0.001 … +0.007) or train did not improve.
- **Drop-one sweep of every rule in Marwari and Hadauti** (95 candidates): no removal passes. Three
  rules are inert (0.0000): `assim-sonorant.jn` and `initial-degem.cc` (Marw); `initial-degem.cc`
  (Hadauti). They were left, since they are not @morph.

- **Neuter `%ext-menu +aḍakam` and `+allakam`** (G -ṛũ, -lũ: jʰupṛũ, cavṛũ, bokʰlũ, dʰāplũ; OG
  -aḍauṁ, -lauṁ). This is the top G near-miss pair ɔ → ũ (71): the masculine -aḍaka/-allaka was the
  only option. Placebo +arakam in each lect:

  | lect | +aḍakam | +allakam | placebo +arakam | joint |
  |---|---|---|---|---|
  | G | −0.0011/−0.0014 | −0.0018/−0.0007 | −0.0006/−0.0006 | −0.0026/−0.0019 |
  | OG | −0.0007/−0.0016 | −0.0008/−0.0017 | −0.0004/−0.0010 | −0.0011/−0.0023 |

  Plain and verb are flat.

### Item 6: legacy `@morph` rules (leave-one-out, then jointly)

**Removed:**
- **Survey verb-citation rule** `{a ā ē ō} {i ē} > o / _ #`: Malvi, Mewari, Godwari, Hadauti, Bagri,
  Shekhawati, Shahawadi, Dhundari.
  - Neutral removal (ext/verb 0) with plain +0.003 … +0.012.
  - Mewari: ext −0.0002.
  - Verbs get the -ɔ from the `+itaka` menu item and the sound changes.
- **Marwari**: the same rule plus `if:root ∅ > o`, jointly ext −0.0002/0, plain +0.0119/+0.0047.
- **OMarw and OG** `ē > i / a t _ #` (middle -atē): neutral or −0.0001; plain +0.005/+0.012.
- **WK** -ati > -vũ: neutral; plain +0.034/+0.028.
- **G** `if:root ∅ > v ũ`: ext −0.0008/−0.0011; plain +0.006 (bare roots such as *aḍḍ scored against
  -vũ infinitives).

**Retagged as sound changes** (-aka > -ao > -o is regular):
- G `a a > o / _ #`, jointly with the removal above: ext −0.0012/−0.0015, verb −0.0016/−0.0020.
- WK: neutral.

**Kept**, with the gate that fails without them:
- -in > -ī in G, OG, WK, OMarw and Marw: ext train +0.0006 … +0.0016.
- G -ati > -vũ: ext +0.0003/+0.0009, because some -ati headwords are not verb-flagged. Retried after
  the 10:35 flag fix; still fails.
- Ahirwadi -ā verb citation: ext train +0.0022, retried after the fix.

### Rejected (other)

- **Polysyllabic apocope in the unrestricted-apocope lects**: plain better, but ext train flat, so it
  fails the gate.
  - Rj, WK, Godwari, Bagri, Shekhawati, Hadauti, Shahawadi, Mewati, Ahirwadi.
  - Mewari (Dholpura), replacing `a > ∅ / C _ #`: +0.0010.
- **Ahirwadi +akā**: ties its placebo on test (−0.0033 both).
- **OG/OMarw medial a + a > e** (naverau, nesāla): +0.0009/+0.0022 and +0.0003/+0.0009.
- **OMarw fem. -ikā > -ī** (motī, kaviḷī): train −0.0095 but test +0.0036 (vetoed).
- **Malvi compensatory lengthening**: sources split (Mewari survey 33 long against 17 short, Varghese
  40 against 89), so it is left as a source difference.

### Not done / open

- Survey citation variation: short -a plural/oblique citations (gʰara, sona) in the unprotected
  lects; -iyɔ vs -ɔ; stray -k of the -akka stand-in.
- G first-syllable ā̃/ã (after a cluster, before a long syllable) and ā/a in -āvvũ/-avvũ (77 against
  43): lexical or orthographic, not ruled.
- Survey medial ɛ/ɔ; initial ava- ɔ/o in G.
- Survey consonant classes (l/ḷ, ḍ/ṛ, s/h) are unchanged since round 6.
