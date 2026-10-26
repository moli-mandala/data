# dardic-south — Kohistani, Kunar, Pashai, Tirahi, Wotapuri

Files: `rules/OIA-{Mai,Gowro,bhatr,Chil,Tor,Bshk,Wot,Tir,Pas,Dm,Gaw,Shum}.sc`,
`results/OIA-*.hand.frontier.csv`. Scratch tools (A/B tester, swap and move tester,
rule-library search, per-source diagnostics, chain-transfer matrix) are in
`tmp/sound_changes/dardic-south/`.

## 1. Summary table

Train/test NED: null → final. "auto" = test NED of the greedy baseline at the closest rule count it
reaches; `induce` stops at 13–27 rules, so for most lects this is its final point. "+n" = rules
in `%source` notation blocks, which are not chain rules.

| lect | name | #etyma | #rules (+block) | train NED null → final | test NED null → final | test exact | auto test NED @n | source blocks |
|---|---|---:|---:|---|---|---:|---|---|
| Mai | Indus Kohistani | 1530 | 29 (+4) | 0.719 → 0.577 | 0.735 → 0.612 | 3.0% | 0.673 @13 | RDH1992; CDIAL RDH1992 liljegren; zoller2005 |
| Pas | Pashai | 1094 | 27 | 0.680 → 0.528 | 0.688 → 0.548 | 6.5% | 0.589 @18 | — |
| Bshk | Bashkarik | 693 | 34 (+3) | 0.687 → 0.474 | 0.688 → 0.499 | 7.8% | 0.554 @14 | RDH1992; gawri |
| Dm | Dameli | 676 | 29 (+2) | 0.685 → 0.511 | 0.656 → 0.500 | 11.7% | 0.582 @18 | CDIAL; perder2013dameli decker1992 liljegren |
| Gaw | Gawarbati | 536 | 26 | 0.674 → 0.524 | 0.650 → 0.501 | 7.1% | 0.599 @15 | — |
| Gowro | Gowro | 487 | 28 (+2) | 0.708 → 0.547 | 0.745 → 0.605 | 4.6% | 0.681 @13 | RDH1992 |
| Tor | Torwali | 460 | 32 (+2) | 0.730 → 0.504 | 0.722 → 0.555 | 9.8% | 0.623 @18 | RDH1992 |
| bhatr | Bhateri | 356 | 32 (+3) | 0.731 → 0.527 | 0.740 → 0.595 | 3.6% | 0.674 @16 | RDH1992; zoller2005 |
| Shum | Shumashti | 352 | 33 | 0.708 → 0.528 | 0.738 → 0.610 | 7.0% | 0.660 @20 | — |
| Wot | Wotapuri | 266 | 33 | 0.694 → 0.423 | 0.676 → 0.456 | 8.5% | 0.485 @27 | — |
| Tir | Tirahi | 217 | 32 | 0.660 → 0.467 | 0.728 → 0.519 | 9.8% | 0.602 @18 | — |
| Chil | Chilisso | 124 | 29 (+2) | 0.731 → 0.478 | 0.756 → 0.475 | 20.7% | 0.735 @19 | RDH1992 |

The hand chains beat the automatic baseline on the test split in every lect, by 0.03–0.26 NED.
The smallest margin is for Wotapuri (0.456 against 0.485 at 27 auto rules). The largest is for
Chilisso, where the auto chain over-fits 137 training pairs.

**The noise floor is high.** Some etyma are linked to several distinct forms in one lect. The
mean NED between those forms is 0.42–0.59 in every lect (Mai 0.59, Pas 0.56, Bshk 0.48, Dm 0.50),
which is as large as the residual error. The causes:
- CDIAL and Zoller link derivatives, compounds and phrases (`kǎr-kar`, `rǎŋg kʰēlī̂`,
  `vǎṇḍᵃ karǎṽ`).
- 6–13 % of all pairs have NED ≥ 0.9 under the final chains: suppletion, mislinks and
  onomatopoeia.

Final NEDs of 0.42–0.58 therefore capture most of the regularity that can be recovered. Test
exact-match rates stay low (3–21 %), mainly because of suffixes and, in Zoller's data, tone marks.

**Per-source train NED** (sources with ≥ 20 train pairs, final chain with its blocks):

| lect | sources (n, NED) |
|---|---|
| Mai | Zoller 1684 0.587; CDIAL 235 0.563; RDH 143 0.490 |
| Pas | CDIAL 1521 0.529; Strand 162 0.549; Lehr 117 0.491 |
| Bshk | CDIAL 535 0.461; Gawri 390 0.473; RDH 220 0.516 |
| Dm | CDIAL 598 0.483; Perder 436 0.555; Decker 92 0.497; Liljegren 36 0.541 |
| Gaw | CDIAL 532 0.523; Decker 52 0.529 |
| Gowro | Zoller 476 0.552; RDH 62 0.449; CDIAL 42 0.664 |
| Tor | CDIAL 424 0.512; RDH 70 0.487 |
| bhatr | Zoller 244 0.559; Strand 125 0.510; RDH 55 0.439 |
| Shum | CDIAL 303 0.523; Buddruss 37 0.591 |
| Chil | RDH 94 0.443; CDIAL 43 0.560 |

Short elicited wordlists (RDH, Strand, Lehr, Decker) fit best. Dictionaries with derivatives,
phrases and inflected forms (Zoller, Perder) fit worst.

**Frontier.** Backward elimination (`results/OIA-*.hand.frontier.csv`) reaches 90 % of the train
gain with 17–24 of the 26–34 rules. In every lect the last rules to go, i.e. the most valuable,
are:
- accent removal and apocope;
- the verb-ending @morph rule;
- deaspiration (where present);
- degemination and v- > b-;
- the palatal affrication;
- in Kohistani proper, the Stop+r affricates.

No source was excluded: `%exclude-source` is unused. Notation differences are handled with
`%source` blocks (§5).

## 2. Base chain (group order) and deviations

All twelve files share one skeleton, in the order below (tags in brackets).

1. **Accent and verb morphology.**
   - `V > $1-acc` [@acc-loss].
   - Verb citation morphology [@morph]: `-ati` and `-ayati` → `a`, which is then removed with the
     final vowels. The stem-final consonant is therefore treated as intervocalic; this beat
     deleting the ending outright in every lect.
   - Mai uses Zoller's infinitive `-ǎṽ` and Gowro uses `-ō̃`.
2. **Final consonants, diphthongs, r̩.**
   - `{s ḣ t n m} > ∅ / V C (V) _ #` [@final-cons-loss].
   - `{ai au} > {ē ō}` [@ai-au]. Tir first has `ō > ọ` (Morgenstierne's close o) for OIA ō
     only, not for ō from au.
   - `r̩ > i` [@r-vocalic]. Tir has `r̩ > a`; Dm applies `r̩ > u` after labials first.
3. **kṣ.** `kṣ > ʦ̣ʰ` [@ks]. Bshk has `ʦ̣ʰ-` initially and `-ʦ̣-` medially; Pas and Tor have
   medial `kṣ > ṣ`.
4. **Stop + r**, the main sub-grouping change (§4):
   - Retroflex affricate [@x-dardic-south-tr-affricate] in Mai, Gowro, bhatr and Chil
     (`tr pr > ʦ̣`, `dr bʰr > ẓ`); Tor has only `tr > ʦ̣`.
   - Lateral [@x-dardic-south-Cr-lateral]:
     - Bshk: `tr kr > ɬ`, `dr gr > l`, plus `pr > c`, `bʰr > j`.
     - Gaw and Shum: voiceless Stop+r `> ɬ`, `dr gr > l`; Gaw also `pr bʰr > pl bl`.
     - Pas: `pr br bʰr gr dr > l`.
     - Wot: `dr > l` only.
   - Kept in Dm and Tir.
5. **ṣṭ, st.** `ṣṭ, st > ṭʰ, tʰ` without gemination [@cluster-simplify @s-stop-aspirate], in
   Mai, Gowro, bhatr, Chil, Tor, Bshk and Wot only. Tir, Pas, Dm, Gaw and Shum keep the cluster;
   adding the rule costs them +0.006 to +0.011 NED.
6. **Onsets and palatals.**
   - `{d dʰ j} y > z` or `dy > j` [@palatal-Cy].
   - `c cʰ j > ʦ ʦʰ z` [@affrication]; Tir `cʰ > ʦ`.
   - `y- > j/ź` [@y-j].
   - Initial `s + C` [@initial-degem].
7. **Deaspiration.** `{bʰ dʰ gʰ jʰ ḍʰ} > {b d g j ḍ}` [@deaspiration] everywhere except Mai,
   Gowro and bhatr (Zoller writes bʰ dʰ gʰ). Pas and bhatr also deaspirate voiceless stops.
8. **Lenition block.**
   - Contraction `ava > ō` [@aya-e] in Bshk, Tor, Wot, Shum and bhatr (yáva > zō, náva > nō);
     `aya > ē` in Shum and Chil.
   - `{ṭ ḍ} > ṛ / V _ V` [@int-retroflex @flap]; Bshk has `r`, Dm `r̆`.
   - `{t d} > l` [@x-dardic-south-t-l] in Mai and Chil only; `> ∅` [@int-stop-loss] elsewhere.
     This was tested in every lect.
   - `{k g} > ∅`. Tor instead voices: `{k p} > {g b}` [@int-voicing].
   - `p > v` [@int-p-v].
   - `{v y} > ∅` [@v-loss]; `h > ∅` [@h-loss]; Wot also loses `p`.
   - Hiatus absorption `{a i} > ∅ / V _` [@hiatus].
9. **Cluster-simplification block** (after lenition).
   - r + dental `> ṭ/ṛ/n` [@assim-r].
   - `C r > C`, `C v > C`, `C y > C`, and degemination `C $1 > $1` [@cluster-simplify].
   - Late `r > ∅ / _ C`.
   - Nasal clusters: `nt > n` and `ṇṭ > ṇḍ` [@nasal-voicing] in Kohistani proper; `nt > nd` in
     Pas and Tir; `N + stop > N` in Tor, Wot and Dm.
10. **Apocope and late edge rules.**
    - Apocope `{a ā i u} > ∅ / C _ #` [@final-vowel-loss].
    - Variants: Gaw keeps -u; bhatr, Chil and Tir also delete a final vowel after a vowel.
    - In Mai, Gowro, Wot, Pas, Gaw and Shum, apocope comes *before* hiatus contraction (§3).
    - Then `ṁ > ∅` [@x-dardic-south-anusvara-loss], `ṇ > n` [@n-dental] and `v- > b-` [@v-b].
    - Lect-specific vowel-length changes [@x-dardic-south-length-loss,
      @x-dardic-south-final-lengthening], spirantisation, and glide prothesis `∅ > y / # _ e`
      [@prothesis] in Shum and Dm.

Lect-specific additions:
- Mai and Chil: final voiceless stops aspirated.
- Tor: final `r > ∅` (áŋgāra > aŋā, kṣīrá > ʦ̣ʰī), final `y > ∅`, `ṛ > r`, and `dy > j`
  before the `-ya-` passive suffix is dropped.
- Wot, Gowro and Chil: a second flapping, `ḍ > ṛ / V _`, applies to ḍ from degeminated ḍḍ.
  Chil has only this late flapping.
- Shum: `kʰ > x`; `ś > s` except word-initially; late `h > ∅`.
- Tir: `ś > s`, `ṣ > x`, `ṣṭ > ṣt`, initial `h > ∅`; flapping ordered before deaspiration.
- Wot: `ṣ > ś`, `dr > l`.
- Dm: `s + aspirate > aspirate`, `s > ś / _ t`, `y j > ź`, intervocalic `v y h > ∅`.
- Bshk and bhatr: `ē > i`.
- Tor, Bshk and Wot: word-final high vowels lengthened (`{i u} > {ī ū} / _ #`); Wot shortens
  them before consonants.
- bhatr: `ṭʰ tʰ > ṭ t`.
- Vowel-length loss: all vowels in Gowro and Chil; high vowels in Mai, bhatr and Wot; ū ē in
  Shum and Dm.

Tried and rejected across all lects (each raised train NED): umlaut, vowel nasalisation before
NC, compensatory lengthening, OIA-accent-conditioned apocope, and any tone rule (§5). Gaw
r-metathesis into the first syllable, feeding ɬ (kárman > *kraman > ɬam), fit a few words but did
not survive pruning.

## 3. Relative chronology

Evidence comes from swap tests on train; Δ is the NED increase when the order is reversed.

- **Intervocalic lenition precedes degemination in all 12 lects.** The geminate escapes
  lenition: kukkuṭá > Mai kukī̃̂, not *kuī. Moving `C $1 > $1` above the lenition block costs:

  | Mai | Pas | Bshk | Dm | Gaw | Gowro | Tor | bhatr | Shum | Wot | Tir | Chil |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | +0.011 | +0.007 | +0.019 | +0.010 | +0.014 | +0.006 | +0.012 | +0.011 | +0.007 | +0.009 | +0.006 | +0.007 |

  This is the Dardic analogue of the MIA rule that geminates protect, but *without* the MIA
  gemination stage.
- **The other cluster simplifications also follow lenition.** This covers `C y`, `C v`,
  r + dental, `C r`, and the new late `r > ∅ / _ C`.
  - Moving the first four below the lenition block improved 10 of 12 lects (Shum −0.005,
    Gaw −0.003, Pas −0.002, Gowro −0.0017, Bshk −0.0016, Tor −0.0013, …). The cost was at most
    +0.0008, in bhatr.
  - Swapping the late r-loss with lenition costs Bshk +0.008 to +0.010, Shum +0.006 and
    Pas +0.006.
  - So consonants from former clusters (kárdama, *krapyati) were never intervocalic while
    lenition was active.
- **Deaspiration precedes intervocalic loss** in every lect that deaspirates: dʰ patterns with
  d (gōdʰā́ > Pas gō). Moving deaspiration after the lenition block costs Pas +0.0015,
  Bshk +0.0018, Dm +0.0012, Gaw +0.0015, Tor +0.0035, Wot +0.0036, Shum +0.0012 and Tir +0.0009.
- **Lenition precedes apocope everywhere.** Apocope first costs +0.007 (Gowro) to +0.032 (Wot):
  Mai +0.009, Pas +0.017, Bshk +0.021, Dm +0.017, Gaw +0.018, Tor +0.017, Tir +0.027.
- **Apocope and hiatus contraction split the group.**
  - Apocope first, then hiatus contraction: Mai, Pas, Gaw and Shum. Reversing costs +0.0018,
    +0.0060, +0.0041 and +0.0022. Gowro and Wot have the same order.
  - Hiatus first: Bshk, Tor and bhatr. Putting contraction right after apocope costs +0.0012,
    +0.0013 and +0.0015. Dm is neutral.
  - Mechanism: `-aka > -aa` keeps a final vowel only when apocope applies first. So
    kaṭaka > Pas kaṛa and Mai kaṛa (Zoller kaṛâh, with the masculine -âh), but Bshk/Tor lose
    the vowel.
  - This isogloss separates the Swat lects and Bhateri from Indus Kohistani and Gowro.
- **Stop+r > ɬ bleeds the general `C r > C`.** Swapping costs Bshk +0.006 and Gaw +0.007.
- **The order of -p- > -v- and the loss of old -v- differs by branch.**
  - Gaw, Dm, Pas and Mai lost old -v- *before* `p > v`, so the new v survives. Swapping costs
    Gaw +0.0013, Dm +0.0015, Pas +0.0015 and Mai +0.0006.
  - In bhatr and Shum, `p > v` *feeds* the loss of v, so p is lost. Swapping costs bhatr
    +0.0015; the feeding order gains −0.0031 in Shum.
- **Smaller orderings.**
  - Tor: apocope feeds final r-loss (+0.009 if swapped).
  - Mai and Chil: apocope feeds final aspiration (+0.003 and +0.0075).
  - Shum and Wot: `y- > j` feeds `j > z` (+0.004 and +0.007).
  - Tir: flapping precedes deaspiration, so ḍʰ > ḍ is not flapped (−0.0032 against the reverse).
  - Wot and Gowro: apocope precedes the cluster simplifications (−0.0025 and −0.0043).

## 4. Shared vs lect-specific changes; sub-grouping

Initial reflexes (counts of reflex onsets, all sources):

| OIA | Mai | Gowro | bhatr | Chil | Tor | Bshk | Wot | Tir | Pas | Dm | Gaw | Shum |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tr- | ʦ̣ 10/16 | ʦ̣ 2, c 2 | ʦ̣ 4, c 3 | ʦ̣ 2/2 | ʦ̣ 2/2 | ɬ 5/5 | t/ṭ/ɬ | tr 3/3 | tr 6, ɬ 6 | tr 4/4 | ɬ 4/4 | ɬ/λ/sl |
| pr- | ʦ̣ 15/23 | p 4, c 3 | ʦ̣ 3/6 | c 3/4 | p 4/6 | c 13/19 | p | pr | l 10, ɬ 9, pl 7 | pr 32/35 | pl 9, ɬ 8 | ɬ 6/10 |
| dr- | ẓ 8/13 | ẓ 6/11 | — | — | d/ź | l 6/6 | l 3/4 | — | dr/l/d 4 each | dr 2/2 | l 4/4 | l 2/3 |
| gr- | g 7/8 | g 5/7 | g 3/3 | g | g 2/2 | l 3/5 | g 5/5 | — | l 11/13 | gr 10/11 | l 2/3 | l 2/3 |
| bʰr- | ẓh 8/13 | ẓh 2/4 | h/ź | ẓ/ź | b 3/4 | j 5/6 | — | br 2/2 | l 16/20 | br 9/10 | bl 4/6 | l 2/3 |
| c- | ʦ 64, c 34 | ʦ 7/14 | ʦ 9/14 | ʦ 3/4 | c 18/22 | c 36/37 | ʦ 7/10 | ʦ 3/4 | c 50/56 | c 36/45 | ʦ 8/10 | ʦ 12/14 |
| j- | z 26/48 | z 7/12 | z 9, ž 5 | z 2/2 | j 8/10 | j 17/17 | z/zy | j 2/2 | j 39/47 | ź 20/31 | z 9/11 | z 6/7 |
| y- | y 18, ź/z 13 | y 7/17 | ź 3, y 3 | — | y 4/10 | y 19/27 | z 4/6 | z/t | ź 16, y 14 | ź 17/22 | ź 9/16 | z/ź 5/6 |
| v- | b 97/132 | b 30/42 | b 25/30 | b 7/8 | b 26/27 | b 56/60 | b 12/15 | b 12/14 | v 120/159 | b 51/70 | v 11, b 7, ∅ 5 | v 10, ∅ 6 |
| h- | h 57/58 | h 15/17 | h 16/16 | h 5/5 | h 14/14 | h 28/30 | ∅/y | ∅ 3/4 | ∅ 29/48 | ∅ 10, ź/z/d | h 19/19 | ∅ 11/13 |
| gʰ- | gʰ 34/36 | gʰ 6/7 | gʰ 8/9 | gʰ | gʰ 4, g 3 | g 15/15 | g 10/10 | g 4/5 | g 27/38 | g 14/15 | g 9/13 | g 7/8 |

**Chain-transfer matrix.** Each lect's pairs (all splits, no source blocks) were scored with every
other lect's chain. The row shows the best non-own chains:

| data | own | best other chains |
|---|---|---|
| Mai | 0.595 | bhatr 0.616, Gowro 0.617 |
| Gowro | 0.572 | bhatr 0.584, Mai 0.592 |
| bhatr | 0.569 | Gowro 0.600, Mai 0.604 |
| Chil | 0.595 | Wot 0.595, bhatr 0.595, Mai 0.598 (its RDH notation is in a block, so its own chain looks weak here) |
| Tor | 0.533 | Bshk 0.560, Wot 0.571 |
| Bshk | 0.493 | Tor 0.535, Dm 0.536 |
| Wot | 0.434 | Shum 0.490, Gaw 0.494, bhatr 0.498, Bshk 0.498 |
| Tir | 0.477 | Dm 0.542, Wot 0.549 |
| Pas | 0.532 | Gaw 0.563, Shum 0.572 |
| Dm | 0.516 | Bshk 0.578, Wot 0.579 |
| Gaw | 0.519 | Shum 0.550, Wot 0.553 |
| Shum | 0.547 | Gaw 0.582 |

Groupings suggested by the accepted rules and the transfer matrix:

- **Kohistani proper: Mai, Gowro, bhatr, Chil.**
  - Stop+r > retroflex affricate or fricative (including the unusual `pr > ʦ̣`).
  - `ṇṭ > ṇḍ`, `nt > n`, c j > ʦ z, ṣṭ > ṭʰ.
  - Voiced aspirates kept (not in Chil).
  - Vowel-length reduction.
  - The three Zoller lects are each other's nearest chains. Mai and Chil share `t d > l` and
    final aspiration.
  - Mai and Gowro keep a final vowel from *-aka (apocope before hiatus contraction), as Kunar and
    Pashai do; bhatr does not.
- **Swat Kohistani: Tor and Bshk, mutual nearest neighbours.**
  - Shared with Kohistani proper: ṣṭ > ṭʰ, v- > b-, h kept.
  - Shared with each other: **no palatal affrication** (c j kept), deaspiration, and dy > j.
  - Torwali still has `tr > ʦ̣`, the Kohistani outcome.
  - Also shared: `ava > ō` and lengthened word-final high vowels. Both have hiatus contraction
    before apocope, so *-aka loses its vowel.
  - Bashkarik has the lateral reflexes (tr kr > ɬ, dr gr > l) of Kunar and Pashai, plus its own
    `pr > c` and `bʰr > j`.
- **Kunar: Gaw and Shum, mutual nearest.**
  - Laterals from Stop+r, ṣṭ kept, v- kept or lost, c j > ʦ z.
  - Apocope before hiatus contraction, so *-aka keeps a final vowel.
  - **Dameli** is the archaic member. It keeps Stop+r and ṣṭ and has no affrication of c. Its
    nearest chains are Bshk and Wot, not Gaw or Shum. Its Kunar traits are retentions, ź, and two
    things it shares with Shumashti: y-prothesis before initial e and shortening of ū ē.
- **Pashai**: nearest chains are Gaw and Shum.
  - Laterals from voiced Stop+r (`bʰrāturjāyā > lājā́`); voiceless Stop+r splits tr/ɬ by dialect.
  - Shared with Kunar: v- kept, ṣṭ kept, apocope before hiatus contraction.
  - Shared with Swat Kohistani: c j kept.
  - Also `nt > nd`, h- lost, medial kṣ > ṣ.
- **Tirahi** is **not Kohistani** on this evidence. Its nearest chains are Dm and Wot.
  - Keeps Stop+r and ṣṭ, like Dameli.
  - Has `nt > nd` and h- loss, like Pashai.
  - Has its own ś > s, ṣ > x (Pashto-like) and r̩ > a.
- **Wotapuri** is transitional. Its nearest chains are Shum and Gaw, but Bshk and bhatr are
  close.
  - Kohistani-like: ṣṭ > ṭʰ, gr > g, c j > ʦ z.
  - Kunar- and Pashai-like: dr > l, y- > z, h- loss, deaspiration.
  - Its own: ṣ > ś.

The laterals from Stop+r (Bshk, Gaw, Shum, Pas, partly Wot) and the retention of c j (Tor, Bshk,
Pas, Dm) both cut across the traditional Kohistani / Kunar / Pashai division. Both look like areal
isoglosses; the Kohistani-proper affricates are the only Stop+r outcome confined to one clade.

## 5. Novel, surprising or data-related findings

- **Zoller's tone cannot be predicted from OIA.**
  - About 70 % of Zoller's Indus Kohistani forms carry â (falling) or ǎ (rising). Falling tone
    is 55 % overall.
  - Rising tone is favoured after voiced aspirates (falling share: bʰ 35 %, dʰ 31 %, ḍʰ 20 %) and
    in the infinitive -ǎṽ (19 % falling). This is a trace of the known correlation between
    voiced aspirates and low or rising tone.
  - No OIA feature predicts the tone: not accent position, syllable weight, vowel length or the
    final vowel.
  - Every tone rule tried raised NED, including a Zoller-only `%source` block. The tone often
    sits on derivational suffixes (-âh, -ī̂, -ū̌) that the chain cannot generate.
  - Gowro marks tone on only about 40 % of forms, 80 % of them rising. A Gowro tone block was
    much worse (+0.045 on Zoller's pairs).
- **Morphology in the reflexes.**
  - Zoller cites Mai verbs as `-ǎṽ` and Gowro verbs as `-ō̃`.
  - Gawri (Bashkarik) gives many verbs as `-ánt` participles: 55 of 500 pairs.
  - Dameli (Perder) gives inflected forms with hyphens (kur-isan, gan-ni).
  - Only the two Zoller infinitives could be modelled. A CDIAL/RDH block removes Mai's `-ǎṽ`,
    because those sources cite stems.
- **RDH 1992 marks vowel length almost throughout**, writing ā ī ū for OIA short vowels. The
  block `%source rensch-decker-hallberg1992 {a i u} > {ā ī ū}` lowers that source's NED by
  0.06–0.12 in all six lects that use it. RDH also never writes aspiration on ʦ̣, so the same
  block has `ʦ̣ʰ > ʦ̣`.
- **Other notation blocks and differences.**
  - Dameli: Morgenstierne (CDIAL) writes /ā/ as ɔ, so a CDIAL block has `ā > o`; Perder writes ā.
    Morgenstierne's flap `r̆` is written `r` by Perder and Decker.
  - The Gawri dictionary writes /a/ as æ, and Zoller's Bhateri also writes æ for /a/.
  - Zoller's Bhateri has extra palatalised letters (ď ť ľ ň ř ž, ȳ, ø); these are not mapped.
  - Zoller writes a superscript release vowel after a final nasal + voiced stop (kâṇḍᵃ).
- **Pashai dialect mixture.**
  - 44 % of Pashai etyma have two or more distinct CDIAL forms; the mean NED between them is 0.56.
  - tr/ɬ, ā/ō, a/ō and kṣ as ʦ̣ʰ/ṣ/x alternate by dialect, so one chain cannot fit them all.
  - Strand and Lehr are no further from CDIAL than CDIAL's own variants (0.34 min-NED).
- **Retroflex affricates from *pr.**
  - Indus Kohistani and Bhateri have `pr > ʦ̣` (pralabʰatē > ʦ̣ālǎṽ, *prēratē > ʦ̣ēlǎṽ);
    Bashkarik and Chilisso have `pr > c`.
  - Both pattern with `tr > ʦ̣`, which implies a general *Cr > retroflex affricate before the
    labial was lost.
- **Torwali voices** intervocalic k p (niṣkālayati > nigāl) but loses its dentals. It is the only
  lect with @int-voicing.
- **Late loss of preconsonantal r** is shared by 10 lects and always follows lenition.
- **Data problems.**
  - Zoller POS abbreviations leaked into Mai reflexes (`ǒh perṣproṇpl`, `kū inṭproṇ`; 7 pairs).
  - A few HTML and ellipsis artefacts: `pī<sup>ṇ</sup>`, `trē<sup>æ</sup>`, `ʦâgar ... tōrū̌`.
  - Obvious mislinks: *lakka > biyǎṽ, *cʰapp > limbǎṽ, tuvám > ʦǎṽ.

## 6. Tags defined here

| tag | meaning |
|---|---|
| `@x-dardic-south-tr-affricate` | Stop + r (and śr) > retroflex or palatal affricate/fricative: tr pr > ʦ̣, dr bʰr > ẓ (Kohistani); pr > c, bʰr > j (Bashkarik) |
| `@x-dardic-south-Cr-lateral` | Stop + r > lateral: voiceless > ɬ, voiced > l (Bashkarik, Gawarbati, Shumashti, Pashai; Wotapuri only dr) |
| `@x-dardic-south-t-l` | intervocalic t d > l (Indus Kohistani, Chilisso) |
| `@x-dardic-south-anusvara-loss` | anusvāra lost without nasalising the vowel |
| `@x-dardic-south-final-aspiration` | word-final voiceless stops aspirated |
| `@x-dardic-south-final-r-loss` | word-final r lost (Torwali) |
| `@x-dardic-south-length-loss` | inherited vowel length neutralised (all vowels, or high vowels only) |
| `@x-dardic-south-kh-spirant` | kʰ > x (Shumashti) |
| `@x-dardic-south-s-x` | ṣ > x (Tirahi) |
| `@x-dardic-south-s-merger` | ś > s except word-initially (Shumashti) |
| `@x-dardic-south-deretroflex` | ṣṭ > ṣt (Tirahi) |
| `@x-dardic-south-final-lengthening` | word-final high vowels lengthened (Torwali, Bashkarik, Wotapuri) |
| `@x-dardic-south-tonic-lengthening` | round 2: Vedic-accented short vowel of a CV́C(V) disyllable lengthened (Pashai) |

## 7. Engine features wished for

- `%source` blocks that can see the etymon, or a marker left by the chain (for example "this was
  a verb"). That would allow per-source citation morphology (Gawri `-ánt`, Perder's inflected
  forms) without guessing.
- A way to score a reflex against a *stem*, ignoring trailing suffix material, or a cap on the
  length penalty. Derivational suffixes dominate the residual error in Zoller's data.
- Optional tone-insensitive scoring that strips ̂ ̌ ᷄ on both sides, so tonal and non-tonal sources
  can be compared on segmental changes.


## 8. Round 2: vowels and the Vedic accent

Scratch: `tmp/sound_changes/dardic-south/` (`accbat2.sh` accent battery, `acctab.py` outcome tables
by accent/position/weight, `tone_acc.py` Zoller tone vs accent, `r2-*.txt` logs; round-1 files in
`r2bak/`). Frontiers were rerun for all 12 lects.

### 8.1 Audit of the partial round-2 edits (06:33–06:39)
Kept, all with positive LOO: `%gold-drop tone` (Mai, Gowro, bhatr, Bshk) plus Mai's infinitive
rule rewritten `-ǎṽ > -aṽ` to match the stripped gold; Bshk `ō > o / _ C C` (+0.0020) and `-ī` in
apocope (+0.0015); Dm/Wot causative `@morph` rule (+0.0022, +0.0040). Removed three zero-LOO rules
(Mai causative -ǎṽ line, Chil causative line, Tor `{ai au} > {ē ō}` which never fires).

### 8.2 Scores (train / test NED)

| lect | round 1 (summary) | round 2 final | test exact |
|---|---|---|---|
| Mai | 0.577 / 0.612 | 0.521 / 0.542 (* gold-drop) | 8.4% |
| Pas | 0.528 / 0.548 | 0.519 / 0.547 | 6.7% |
| Bshk | 0.474 / 0.499 | 0.459 / 0.489 (*) | 8.5% |
| Dm | 0.511 / 0.500 | 0.504 / 0.476 | 13.6% |
| Gaw | 0.524 / 0.501 | 0.520 / 0.501 | 7.1% |
| Gowro | 0.547 / 0.605 | 0.510 / 0.573 (*) | 6.6% |
| Tor | 0.504 / 0.555 | 0.492 / 0.542 | 12.1% |
| bhatr | 0.527 / 0.595 | 0.506 / 0.574 (*) | 5.4% |
| Shum | 0.528 / 0.610 | 0.522 / 0.603 | 7.0% |
| Wot | 0.423 / 0.456 | 0.414 / 0.445 | 8.5% |
| Tir | 0.467 / 0.519 | 0.459 / 0.500 | 12.2% |
| Chil | 0.478 / 0.475 | 0.472 / 0.475 | 20.7% |

No lect's test NED rose. Round-1 chains rescored under the current engine (tokenisation fixes)
already move Shum 0.610 → 0.601, Tir 0.519 → 0.500, Wot 0.456 → 0.449; the rest by ≤ 0.002.

**`%gold-drop tone` is a change of scoring, not a sound change.** Decomposition (test NED):

| lect | round-1 chain, tones scored | round-1 chain + gold-drop | round-2 chain + gold-drop | round-2 chain, tones scored |
|---|---|---|---|---|
| Mai | 0.612 | 0.552 | 0.542 | 0.611 |
| Gowro | 0.605 | 0.574 | 0.573 | 0.604 |
| bhatr | 0.593 | 0.579 | 0.574 | 0.588 |
| Bshk | 0.499 | 0.497 | 0.489 | 0.491 |

So Mai's 0.612 → 0.552 drop in the partial edits was entirely the directive (−0.060); round-2 sound
changes add −0.010 on top (with tones scored: 0.612 → 0.611). For Gowro, bhatr, Bshk the directive is
worth −0.031, −0.014, −0.002 and the rules −0.001, −0.005, −0.008.

### 8.3 Adopted vowel/accent rules (LOO on train; examples)
- **Apocope restricted to polysyllables, all 12 lects**: `… > ∅ / V C* C _ #` (or `V C* _ #`).
  Monosyllables keep their vowel (ma, kím > ki). Gain per lect −0.001 (Dm) to −0.008 (Bshk, Tor,
  Chil). A positional (word-size) condition, not accentual.
- **Apocope also after a vowel** (Mai, Pas, Bshk, Dm, Gaw, Tor: −0.0005 to −0.0016): -ikā > -iā > -i.
  It weakens the §3 apocope/hiatus ordering evidence: Mai now prefers apocope-first by only +0.0004,
  Pas is neutral (−0.0002); Gaw +0.0017, Shum +0.0010, Bshk +0.0090, Tor +0.0034 keep their orders.
- **Apocope vowel set generalised**: + ē ō in Tor (−0.0019) and Pas (−0.0007); + ē in Tir, Chil;
  + ē ī in Gowro.
- **Pas tonic lengthening** `{á í ú} > {ā ī ū} / # C* _ C (V) #` [@stress], LOO +0.0024, 23 etyma:
  yáva > yā, mála > māl, kʰála > kāl, náva > nā, káṭa > kāṛa. The same rule on *unaccented* vowels
  costs +0.0024, on all vowels +0.0008 — **the accent is the better conditioner**. Extending it to
  closed syllables/final C gains a further −0.0013 on train but was not adopted (few items).
  Tested in all other lects: only Tor is marginally positive (−0.0008, driven by i/u); the rest
  +0.001 to +0.005.
- **Medial long-vowel shortening** `if:!verb {ā ī ū ē ō} > {a i u e o} / V C* _ C* V` in Dm
  (+0.0022, 65 etyma; *mēdapālikā > mēvalī́) and Wot (+0.0031). Accent-independent: letting tonic
  vowels keep length scores the same (Wot) or worse (Dm −0.0020 vs −0.0022); fails in the other ten.
- **Hiatus**: Tor `{a i u} $1 > {ā ī ū}` (+0.0030; katamá > kām, saptádaśa > sattāś);
  `i {a ā} > ī / _ #` in Pas (+0.0023; kárṇikā > kanī́), Gowro (+0.0012), Shum (+0.0011);
  Shum `a > æ / _ V` (+0.0036; mahiṣá > mǽiṣa, cáturdaśa > ʦæudas).
- **bhatr** `{a á} > ā / # _ C C` (+0.0036; aṣṭā́ > āṭ, adyá > āz); accented-only version −0.0000.
- **Citation morphology**: `if:verb,!numeral` on the -ati rule in Tir (+0.0073) and Gaw (+0.0017),
  on the causative rule in Pas (+0.0030). Elsewhere `if:verb` hurts (+0.001 to +0.006: the verb flag
  misses unglossed verbs) and `if:!numeral` alone hurts or is neutral (the -ati rule happens to
  model numeral -ti loss too).

### 8.4 Hypotheses tested and rejected (ΔNED on train when added; + = worse)
- Tonic lengthening, all tonic vowels `{á í ú} > {ā ī ū}`: +0.002 (bhatr) to +0.031 (Wot);
  first-syllable only +0.002 to +0.024; open-syllable only +0.001 to +0.020.
- Pretonic long-vowel shortening `_ !#* Acc`: +0.000 to +0.010; posttonic: ≈0 everywhere except Dm
  −0.0004, Wot −0.0017 (both beaten by the positional medial rule above).
- Pretonic medial syncope `V C _ C !#* Acc`: +0.001 to +0.009; immediately posttonic syncope
  `Acc C _ C V`: +0.003 to +0.020; general medial syncope: +0.010 to +0.043.
- **Accent-conditioned apocope** (posttonic final vowels reduced before @acc-loss, tonic finals
  kept): +0.004 to +0.024 relative to a matched control in Mai, Pas, Bshk, Dm, Gaw, Tor, Shum, Wot.
  Raw rates do differ (Mai final -a lost 54 % tonic vs 64 % posttonic; Dm 41 % vs 57 %), but loss
  stays the majority outcome in every cell.
- Weight-conditioned apocope (kept after CC, after a long vowel, or in light disyllables):
  +0.002 to +0.064.
- Accent-conditioned compensatory lengthening in closed first syllables `á > ā / # C* _ C C`:
  +0.003 to +0.007 (all vowels +0.015 to +0.024). Mai tonic closed first-syllable a is ā/ā̃ in 39 %
  vs 18–22 % pretonic/unaccented, Pas 25 % vs 10–12 %: a real accent effect, but a minority outcome.
- Stress rounding `á > o`, `ā́ > ō`: +0.004 to +0.017. Umlaut (a > e/æ/i before i, a > o before u,
  accented or not): mostly +0.002 to +0.037; bhatr's −0.0012 is catur- contraction, not umlaut.
- Monosyllable lengthening, first-syllable lengthening, two-mora law (except Bshk ō): all worse.
- Vocalic r̩: splits by labial/velar neighbour, next-vowel harmony, or accent (ŕ̩ > a/u) are all
  within ±0.001 except Tir tonic ŕ̩ > u (−0.0012, 2 items). Dameli's apparent "tonic r̩ > u 68 %" is
  one etymon (mŕ̩ṣā) with ~50 inflected Perder forms — a pair-count artefact.
- Final -ī: accent-conditioned loss (unaccented -ī lost, tonic -ī́ kept) fits only Tor (−0.0007,
  no tonic item actually fires) — rejected.

### 8.5 Kohistani length and Zoller's tone
- **Length reduction is not accent-conditioned.** Shortening only unaccented long vowels (accented
  ones kept) is worse than unconditional shortening in every lect with a length-loss rule: Mai
  +0.0011 (accented-only +0.0024), bhatr +0.0009, Gowro +0.0044, Chil +0.0148, Wot +0.0027,
  Shum +0.0067, Dm +0.0061.
- **Zoller's tone (Mai, 2118 Zoller pairs, train)**. Falling share: overall 58 %; after an initial
  voiced aspirate 48 % vs 58.5 % (rising favoured, n = 138); any voiced aspirate 53 % vs 58 %.
  By Vedic accent (nouns only, verbs are dominated by the rising -ǎṽ infinitive): initial accent
  68 % falling (n = 225), final accent 57 % (n = 271), unaccented 60 % (n = 909); medial accent
  71 % (n = 34). So initial accent mildly favours falling tone (≈ 2.4 s.e.), but the tone-bearing
  syllable does not track the accent (initially-accented words carry tone on σ2 in 39 % of cases).
  Tone-assignment rules scored against the unstripped gold all hurt: first-syllable falling +0.049,
  rising +0.057, rising after voiced aspirates +0.0024 (+0.0027 non-initial).

### 8.6 What the accent does and does not condition in dardic-south
- Does: Pashai lengthening of the accented vowel of disyllables (the one accent rule that beats its
  positional control); weak, sub-majority tendencies to lengthen tonic closed first syllables
  (Mai, Pas) and to retain final tonic vowels (Mai, Dm).
- Does not: apocope, syncope, pretonic/posttonic reduction, vocalic r̩, Kohistani length loss, or
  (beyond a small initial-accent bias) Zoller's tone. The successful vowel conditioners are
  positional: word size (monosyllables keep final vowels, all 12 lects), medial syllable (Dm, Wot
  shortening), and hiatus contraction.
- Caveat: only ~30 % of etyma are accented and Turner's starred forms never are, so the accent
  tests rest on 200–700 etyma per lect; pair counts are inflated by inflected forms (Perder, Gawri).

## 9. Accent score (round 3)

Tools: `tmp/sound_changes/dardic-south/abacc.py` (per-op Δ train accent NED, flags any movement of
the plain NED), `accbat3.sh` (placement battery), `accpos.py` (where the attested accent sits).
All placement rules sit at the end of the chain, before the `%source` blocks. Plain train and
test NED are unchanged in every file (checked). None of our rules or `%source` blocks write a
visible vowel accent (only ś ź), so no notation hurts the score. Frontiers rerun for the ten
changed lects.

### 9.1 Test accent NED: no accent / Vedic carried / final chain (train in brackets)

| lect | acc. pairs tr/te | none | Vedic | final | adopted placement (train accent LOO) |
|---|---|---|---|---|---|
| Mai | 1523/393 | 0.566 (0.543) | 0.563 (0.538) | **0.543** (0.523) | last syllable, +0.0204 |
| Pas | 649/180 | 0.587 (0.558) | 0.591 (0.559) | 0.603 (0.553) | last syllable, +0.0052 |
| Bshk | 433/122 | 0.580 (0.517) | 0.572 (0.505) | **0.549** (0.466) | long final vowel +0.0034; else first syllable +0.0190 |
| Gowro | 274/70 | 0.592 (0.528) | 0.593 (0.527) | 0.592 (0.522) | last syllable, +0.0060 |
| bhatr | 208/54 | 0.598 (0.543) | 0.588 (0.536) | **0.578** (0.509) | last syllable, +0.0340 |
| Wot | 63/21 | 0.456 (0.473) | 0.447 (0.473) | **0.434** (0.450) | last syllable of polysyllables, +0.0222 |
| Shum | 75/31 | 0.516 (0.546) | 0.513 (0.547) | **0.501** (0.533) | first long vowel +0.0103; else first syllable +0.0027 |
| Gaw | 174/50 | 0.519 (0.575) | 0.526 (0.578) | **0.509** (0.567) | last syllable of polysyllables, +0.0082 |
| Tor | 31/10 | 0.653 (0.534) | 0.679 (0.514) | 0.679 (0.500) | last syllable, +0.0338 |
| Tir | 40/4 | 0.560 (0.538) | 0.560 (0.551) | 0.650 (0.525) | last syllable, +0.0133 |
| Dm | 141/44 | 0.579 (0.542) | 0.581 (0.550) | 0.579 (0.542) | none adopted |
| Chil | 3/0 | — | — | — | not attempted |

Test accent NED improves in Mai, Bshk, bhatr, Wot, Shum and Gaw. It is flat in Gowro. It is
worse in Pas (+0.017 on 180 pairs; the train gain is only 0.005) and in the tiny Tor and Tir sets
(10 and 4 test pairs). Rules were chosen on train only.

### 9.2 Where the attested accent sits
Zoller's tone marks (Mai, Gowro, bhatr) and Gawri's marks (Bshk) count as accent locations. In Mai,
84 % of accent-marked Zoller forms carry the tone on the last syllable of the attested form (1510
pairs); in disyllables it is 80 % last, in trisyllables 62 % last and 30 % penult. Pas attested
final long vowels carry the accent in 96–97 % of cases. Dameli is weight-sensitive in the
attested forms: the last syllable if it is long (92 %), otherwise mostly the penult (52–68 %).
None of these weight rules survives on the *predicted* forms, because the chain predicts forms
without the suffixes (-âh, -ī, -a) that carry the attested accent. The mismatch in word shape is
the main error left in the accent score.

### 9.3 Zoller's tone against the positional rules (Mai, Gowro, bhatr)
Train accent NED, Mai: none 0.5429; Vedic carried 0.5384; first syllable 0.5442; penult at the end
of the chain 0.5515; old penult before apocope 0.5308; old last before apocope 0.5369; **last
syllable 0.5225**. Also tested (Δ against last syllable): the last long vowel, else the penult,
+0.0229; the first long vowel, else the last syllable, +0.0009; the rightmost long vowel,
+0.0026; verbs on the first syllable, +0.0075; the penult for words ending in an open vowel in
three or more syllables, +0.0001. Gowro: last syllable −0.0060 (old penult plus monosyllables
−0.0062, not adopted: two rules for 0.0002). bhatr: last syllable −0.0340 (old penult −0.0315,
first syllable −0.0259). **Zoller's tone is a last-syllable prominence of the modern word**; it
does not continue the Vedic accent (§8.5).

### 9.4 Rejected hypotheses (Δ train accent NED against the file without placement unless noted)
- Pas: first syllable +0.0215; old penult +0.0025; old last −0.0047; last long vowel, else the
  penult, −0.0037; closed last syllable, else the penult, +0.0089; rightmost long vowel, else the
  last syllable, −0.0005 more than the last syllable alone (not adopted).
- Pas: the accented-vowel lengthening in disyllables (§8.3) also helps the accent score. Removing
  it with the last-syllable rule in place costs +0.0024, the same as its plain-NED value. Under the
  Vedic model its removal costs +0.0010.
- Bshk: last syllable −0.0429; old penult −0.0357; the first long vowel, else the first syllable,
  −0.0508 (the adopted pair is −0.0518); nouns on the first syllable and verbs on the last,
  −0.0376 under the Vedic model.
- Wot and Gaw: last syllable including monosyllables +0.0001 / +0.0016, against −0.0222 /
  −0.0082 without them. The sources (Morgenstierne in CDIAL, Buddruss) never mark accent on
  monosyllables: Wot has 1 accent-marked monosyllable and Gaw 0. Adding a monosyllable rule on
  top costs +0.022 (Wot) and +0.010 (Gaw). Old penult: Wot −0.0034, Gaw +0.0053.
- Dm: every placement is worse than none or within noise. First syllable +0.0176, old penult
  +0.0071, last syllable +0.0029, last syllable of polysyllables +0.0019, monosyllables +0.0011,
  last long vowel else penult +0.0106. Old last syllable of polysyllables −0.0012 was rejected: it
  marks almost nothing that survives apocope.
- Shum: last syllable +0.0022; the first long vowel, else the last syllable, +0.0011; first
  syllable alone −0.0033.
- Tor: first long vowel, else last syllable, −0.0370 (last syllable alone −0.0338; the simpler
  rule was kept on 31 pairs). Tir: first long vowel, else last syllable, −0.0158 against −0.0133.

### 9.5 The Vedic accent, given its best chance
- **Vedic carried alone**: better than no accent on train in Mai (−0.0045), Bshk (−0.0125), bhatr
  (−0.0074), Gowro (−0.0009) and Tor. It is worse in Pas, Wot, Shum, Dm, Gaw and Tir.
- **Vedic for accented etyma, positional rule only for unaccented (starred) etyma** (`%accent
  vedic` + `if:!accented`), against the positional rule for all words:

  | lect | Vedic + positional | positional only |
  |---|---|---|
  | Mai | 0.5294 | 0.5225 |
  | Pas | 0.5611 | 0.5530 |
  | Bshk | 0.4844 | 0.4665 (first syllable) |
  | Gowro | 0.5251 | 0.5222 |
  | bhatr | 0.5201 | 0.5093 |
  | Wot | 0.4715 | 0.4503 |
  | Dm | 0.5535 | 0.5423 (none) |
  | Gaw | 0.5815 | 0.5749 (none) |

- **Positional rule for nouns, Vedic for verbs**: Pas 0.5528 (−0.0002 against the positional rule
  alone, noise); Mai and bhatr no better.
- **Conclusion**: in no dardic-south lect does the Vedic accent beat a positional rule, even for the
  accented etyma alone.
  - The modern accent or tone is a word-level prominence.
  - Kohistani (Zoller's tone), Pashai, Torwali and Tirahi put it on the last syllable of the modern
    form; Wotapuri and Gawarbati do the same, but only polysyllables carry a mark.
  - Bashkarik (Gawri) and Shumashti put it on the first syllable, or on the first long vowel.
  - Dameli shows no placement the chain can predict.
  - This matches §8: the accent conditions no segmental change except the Pashai lengthening.
- Engine wish: a way for a placement rule to test for "no accent yet" (or a rule-placed accent
  that replaces the others). "Keep the Vedic accent if its vowel survived, else the last syllable"
  cannot be expressed now.

## 9b. Aspiration-conditioned accent placement (round 3 follow-up)

Tools: `tmp/sound_changes/dardic-south/aspacc.py` counts the attested accent position by the
aspiration of the etymon's σ1 onset. `asptest.py` gives train accent NED overall and split into
"σ1-aspirate" etyma (σ1 onset contains bʰ dʰ gʰ jʰ ḍʰ h) and all other etyma. Plain NED was
unchanged in every variant. **No rule was adopted** and the rule files are unchanged since §9, so
no frontier was rerun.

### Attested forms (train, accent-marked reflexes of two syllables): share of accent on σ2

| lect | σ1 onset voiced aspirate/h | other | caron share after aspirate vs other (all lengths) |
|---|---|---|---|
| Mai | 78 % (n = 77) | 81 % (725) | 52 % vs 42 % |
| bhatr | 22 % (9) | 73 % (51) | 35 % vs 21 % |
| Gowro | 92 % (13) | 76 % (148) | 88 % vs 78 % |
| Bshk | 84 % (19) | 72 % (244) | 3 % vs 4 % |
| Pas | 84 % (32) | 87 % (321) | 8 % vs 1 % |

The Palula/Shina rightward shift does not appear in Mai or Pas: those two go slightly the other
way. Gowro and Bshk agree in direction on 13 and 19 forms; bhatr contradicts it on 9. The rising
contour is more frequent after aspirates in the three Zoller lects (Mai, bhatr, Gowro), but the
accent score does not distinguish contour types.

### Engine tests (Δ train accent NED; "asp1" = the σ1-aspirate subset)
- **Mai, bhatr, Gowro, Pas: a shift on top of the adopted last-syllable rule is vacuous** (Δ 0.0000
  in all four). The accent is already on the last syllable, and the aspirate-initial σ1 of a
  disyllable leaves nothing to its right.
- **First-syllable placement + shift** (`V > $1+stress / ASP Stress C* _`,
  `Stress > $1-stress / ASP _ C* V`):
  - Mai 0.5443 → 0.5430 (asp1 0.5415 → 0.5295); second-mora variant 0.5431. Both remain far
    behind the last-syllable rule (0.5225).
  - bhatr 0.5174 → 0.5164 (last syllable 0.5093).
  - Gowro 0.5369 → 0.5365 (last syllable 0.5222).
- **Bshk (first syllable, or a long final vowel).** σ2 placed before deaspiration (the next
  syllable, the second mora, or lost aspirates only), with the rightmost accent kept at the end:
  0.4655 → 0.4657 in all three variants (asp1 0.5229 → 0.5198, others 0.4609 → 0.4613). Without
  the keep-rightmost rule the result equals the base exactly. The chain reduces these words to
  monosyllables (bʰaginī > ban, hásta > hatʰ, gʰōṭaka > gōr), so the shifted accent is deleted
  with σ2 and the shift cannot be realised on the predicted forms.
- **Lost vs kept aspirate.** Bshk and Pas lose the voiced aspirates and Bshk keeps h. The Zoller
  lects keep bʰ dʰ gʰ and h. So "only where the aspirate is lost" equals the voiced-aspirate
  variant in Bshk (identical numbers) and is empty for Mai, bhatr and Gowro.

### Vedic vs positional, with aspiration controlled (train accent NED)

| lect | subset | positional (adopted) | Vedic carried | Vedic + aspiration shift |
|---|---|---|---|---|
| Mai | asp1 (159) | 0.5266 | 0.5375 | 0.5357 |
| Mai | other (1364) | 0.5220 | 0.5385 | 0.5385 |
| bhatr | asp1 (23) | 0.5197 | 0.5378 | 0.5378 |
| bhatr | other (185) | 0.5080 | 0.5356 | 0.5356 |
| Gowro | asp1 (17) | 0.6034 | 0.5676 | 0.5676 |
| Gowro | other (257) | 0.5168 | 0.5246 | 0.5246 |
| Bshk | asp1 (32) | 0.5229 | 0.5414 | 0.5388 |
| Bshk | other (401) | 0.4609 | 0.5018 | 0.5018 |
| Pas | asp1 (51) | 0.4958 | 0.5044 | 0.5052 |
| Pas | other (598) | 0.5579 | 0.5634 | 0.5634 |

Mixed models:
- **Positional rule, but Vedic in σ1-aspirate words** (the last-syllable rule with exception
  `// # C* ASP !#* _`, under `%accent vedic`): Mai 0.5236 and bhatr 0.5113, both worse than the
  last-syllable rule for all words. Gowro 0.5200 (−0.0022) is better, but only on 17 pairs, and it
  keeps the Vedic accent rather than shifting it rightward. Not adopted.
- **Positional rule, but first syllable in σ1-aspirate words**: Mai 0.5240, bhatr 0.5098,
  Gowro 0.5226. All worse.

**Conclusion.** With aspiration held constant, the positional rule beats the Vedic accent in both
subsets in Mai, bhatr, Bshk and Pas. The only exception is Gowro's 17 σ1-aspirate pairs. An
aspiration shift improves the Vedic model by at most 0.0026 on the aspirate subset, and it
changes almost nothing overall: Mai −0.0002, Bshk −0.0002, Pas and Gowro 0. The Palula/Shina
pattern, an accent pushed rightward by a lost aspirate, is not recoverable in dardic-south. What
remains in Kohistani is the higher rate of Zoller's rising contour after aspirates, a tone-type
effect that the accent score does not measure.

### 9b.1 Rescore after the accent-score revision (12:50)

The coordinator changed the accent distance: a missing accent and a misplaced accent now both cost
0.5, on top of the accent-blind distance. The accent NEDs in §9 and in the tables above use the
old scoring. Plain NED is unchanged. Rescored train/test accent NED:

| lect | none (old → new) | Vedic (old → new) | final chain, new scoring | placement now |
|---|---|---|---|---|
| Mai | 0.543/0.566 → 0.608/0.630 | 0.538/0.563 → 0.595/0.616 | **0.546/0.562** | last σ |
| Pas | 0.558/0.587 → 0.632/0.677 | 0.559/0.591 → 0.622/0.661 | **0.585/0.640** | last σ |
| Bshk | 0.517/0.580 → 0.577/0.649 | 0.505/0.572 → 0.560/0.633 | **0.506/0.594** | long final vowel, else first σ |
| Gowro | 0.528/0.592 → 0.607/0.674 | 0.527/0.593 → 0.591/0.668 | **0.544/0.617** | last σ |
| bhatr | 0.543/0.598 → 0.636/0.698 | 0.536/0.588 → 0.611/0.674 | **0.539/0.617** | last σ |
| Wot | 0.473/0.456 → 0.548/0.527 | 0.473/0.447 → 0.540/0.507 | **0.498/0.480** | last σ (changed) |
| Shum | 0.546/0.516 → 0.602/0.576 | 0.547/0.513 → 0.582/0.564 | **0.551/0.528** | first long vowel, else first σ |
| Gaw | 0.575/0.519 → 0.638/0.596 | 0.578/0.526 → 0.627/0.591 | **0.601/0.545** | last σ (changed) |
| Dm | 0.542/0.579 → 0.614/0.649 | 0.550/0.581 → 0.608/0.645 | **0.571/0.592** | last σ (new) |
| Tor | 0.534/0.653 → 0.587/0.746 | 0.514/0.679 → 0.559/0.746 | **0.518/0.692** | last σ |
| Tir | 0.538/0.560 → 0.602/0.666 | 0.551/0.560 → 0.593/0.635 | **0.536/0.666** | last σ |

What changed:
- **Vedic now beats "no accent" in every lect.** Under the old scoring it lost in Pas, Wot, Shum, Dm,
  Gaw and Tir, because a misplaced accent cost more than a missing one. The positional chain still
  beats Vedic in every lect, by 0.017 (Wot/Tor, test) to 0.072 (bhatr, train).
- **Placement choices under the new scoring** (Δ train accent NED against the final chain):
  - Wot: last σ including monosyllables is now better than polysyllables-only by −0.0121, so the
    rule was changed. Old penult +0.0005, first σ +0.0261.
  - Gaw: last σ including monosyllables −0.0114 against polysyllables-only; rule changed. Old penult
    −0.0053 against polysyllables-only; first σ +0.0082.
  - Dm: last σ −0.0431 against no accent; adopted. Old penult −0.0365, first σ −0.0325, first long
    vowel −0.0325, last σ of polysyllables −0.0176.
  - Mai, Pas, Gowro, bhatr, Tor, Tir: last σ is still best. The runners-up cost +0.0067 to +0.0366:
    old penult, first long vowel, first σ, and last long vowel else first σ.
  - Bshk: a long final vowel, else the first σ, is still best. First long vowel +0.0002, last σ
    +0.0011, first σ +0.0012; the ranking is now nearly flat.
  - Shum: the first long vowel, else the first σ, is still best. Last long vowel else first σ
    +0.0037, first σ +0.0040, last σ +0.0080.
  - `accent_hypotheses.py` reads the live rule file, so its baselines now sit on top of my placement
    rules. For Mai it reproduces the final chain exactly (0.5461/0.5620).
- **Test accent NED improves over both "none" and Vedic in every lect.** The only exception is Tir,
  which ties "none" on 4 test pairs. The Pas regression reported in §9 was an artefact of the old
  cost scheme.
- **Aspiration, rescored** (train accent NED, all / σ1-aspirate subset):

  | lect | last σ (Bshk: first σ) | Vedic | Vedic + shift | last σ, Vedic in σ1-aspirate words | Vedic for accented etyma, last σ otherwise |
  |---|---|---|---|---|---|
  | Mai | 0.5461 / 0.5452 | 0.5948 / 0.6001 | 0.5946 / 0.5993 | 0.5518 | 0.5594 |
  | bhatr | 0.5394 / 0.5342 | 0.6114 / 0.5996 | 0.6114 | 0.5455 | 0.5698 |
  | Gowro | 0.5442 / 0.6230 | 0.5909 / 0.6525 | 0.5909 | 0.5456 | 0.5600 |
  | Pas | 0.5853 / 0.5202 | 0.6215 / 0.5719 | 0.6215 | 0.5860 | 0.6004 |
  | Bshk | 0.5061 / 0.5666 | 0.5599 / 0.6138 | 0.5602 | — | — |

  Gowro's small Vedic advantage on σ1-aspirate words (§9b) disappears under the new scoring
  (0.6525 vs 0.6230). The aspiration shift changes the Vedic model by at most −0.0008 on the subset
  and −0.0002 overall. **The conclusions of §9 and §9b stand**: a word-final (Kohistani, Pashai,
  Kunar, Torwali, Tirahi) or word-initial (Bashkarik, Shumashti) positional accent, no
  continuation of the Vedic accent, and no aspiration-driven shift.

## 10. Extension-aware score (round 4)

Tools, all in `tmp/sound_changes/dardic-south/`:
- `abext.py`: Δ plain, extension-aware and accent NED per edit.
- `extshow.py`: a word's output bare and with each extension.
- `extres.py`: residual ending mismatches under the extension-aware choice.
- `extba.py`: before/after table.
- `menutest.py`: adding or dropping menu items.

All numbers use the current menu (aka, akka, ikā, uka, aḍaka, allaka, araka). The "before"
column is the file with the round-4 rules removed.

### 10.1 Scores (train/test)

| lect | plain NED before → after | ext-aware NED before → after | accent NED train | ending choice, train (after) |
|---|---|---|---|---|
| Mai | 0.5210/0.5422 (unchanged) | 0.4662/0.4826 (unchanged) | 0.5461 | bare 71, aka 12, ikā 5, akka 4, uka 2, aḍaka 2, araka 2, allaka 2 |
| Pas | 0.5185/0.5473 → 0.5184/0.5473 | 0.4651/0.4723 → 0.4642/0.4726 | 0.5853 → 0.5849 | bare 72, akka 8, aka 8, ikā 5, aḍaka 3, uka 2, allaka 1, araka 1 |
| Bshk | 0.4585/0.4890 (unchanged) | 0.4327/0.4623 → 0.4304/0.4606 | 0.5061 | bare 85, akka 5, allaka 3, ikā 3, aḍaka 2, uka 2, aka 1 |
| Dm | 0.5038/0.4758 (unchanged) | 0.4448/0.4114 (unchanged) | 0.5708 | bare 69, aka 12, ikā 8, aḍaka 4, uka 3, akka 2, allaka 1, araka 1 |
| Gaw | 0.5196/0.5006 → 0.5196/0.4990 | 0.4655/0.4290 → **0.4528/0.4204** | 0.6006 → 0.5989 | bare 66, aka 13, ikā 8, akka 5, aḍaka 3, allaka 2, uka 1, araka 1 |
| Gowro | 0.5102/0.5726 (unchanged) | 0.4754/0.5413 (unchanged) | 0.5442 | bare 79, aka 9, uka 3, akka 3, allaka 2, araka 2, ikā 1, aḍaka 1 |
| Tor | 0.4916/0.5424 → **0.4885/0.5377** | 0.4766/0.5187 → **0.4583/0.4996** | 0.5180 → 0.5156 | bare 82, ikā 5, uka 4, akka 3, aka 3, aḍaka 2, allaka 1, araka 1 |
| bhatr | 0.5059/0.5739 (unchanged) | 0.4732/0.5307 (unchanged) | 0.5394 | bare 82, ikā 6, akka 4, uka 4, allaka 2, araka 1, aḍaka 1, aka 0 |
| Shum | 0.5220/0.6033 (unchanged) | 0.4594/0.5155 (unchanged) | 0.5513 | bare 69, akka 10, aka 10, aḍaka 4, ikā 3, araka 2, uka 1, allaka 1 |
| Wot | 0.4136/0.4453 → 0.4146/0.4453 | 0.3799/0.4152 → 0.3774/0.4226 | 0.4980 | bare 81, aka 5, araka 4, aḍaka 3, ikā 2, akka 2, uka 2, allaka 1 |
| Tir | 0.4593/0.4997 → 0.4561/0.4997 | 0.3998/0.4426 → 0.3990/0.4474 | 0.5355 | bare 71, aka 19, aḍaka 3, ikā 3, uka 2, akka 2, araka 1 |
| Chil | 0.4719/0.4753 → 0.4707/0.4753 | 0.4316/0.4559 → **0.4078/0.4559** | — (3 pairs) | bare 72, aka 15, uka 8, ikā 4, akka 1 |

Accent placements are untouched. The segmental rules changed the accent score only in Pas, Gaw
and Tor, and only for the better (−0.0004, −0.0017, −0.0024). Wot's extension-aware test NED
rose (+0.0074 on 71 pairs), and so did Tir's (+0.0048 on 41 pairs). Both were chosen on train.

### 10.2 Rules adopted (LOO on train extension-aware NED; plain LOO)
- **Tor** `k > ∅ / V _ V #` before the voicing rule [@int-stop-loss], ext +0.0179, plain +0.0030.
  Torwali voices -k-/-p- in general (niṣkālayati > nigāl), but the -k- of the -aka/-ika/-uka suffix
  is lost. Without the rule every extension came out as -ag/-ig/-ug. With it, e.g.
  udaká > ū, kukkuṭá > kugū.
- **Tor** `u a > ū / _ #` [@hiatus], ext +0.0068, plain +0.0007.
- **Chil** `a a > o / _ #` right after velar loss [@aka @hiatus], ext +0.0237, plain +0.0012:
  masc. -aka > -o (mahallaka > mʰālo, *lēkka > leko, pā́ṇḍara > pʌnaro). Placed after apocope it
  cost plain +0.0012; placed right after velar loss it gains plain −0.0012.
- **Gaw** `i {a ā} > ī / _ #` before apocope [@hiatus], ext +0.0127, plain 0.0000: fem. -ikā > -ī.
- **Wot** (ext +0.0025, plain −0.0010) and **Tir** (ext +0.0008, plain +0.0033, 3 bare items):
  the same -iā > -ī contraction, right after velar loss.
- **Pas** (ext +0.0007) and **Bshk** (ext +0.0025, plain 0): `u a > ū / _ #`, -uka > -ū.

### 10.3 Tried and rejected (Δ train ext / plain / accent)
- **-aka contraction to a long or rounded vowel** (`a a > ā / o / ū / u`, before apocope or right
  after velar loss). Rejected in every lect except Chil.
  - Worse on the extension-aware score in Pas, Dm, Gaw, Gowro, Shum, Wot and Tir: +0.002 to
    +0.030. In Dm, Gaw and Shum the -aka variant already keeps its -a, and the attested endings
    split between -a, -i and ∅.
  - Mai −0.0014: the contracted -ā is lost to apocope, so it changes little.
  - bhatr `aa > ā` −0.0035, but accent +0.0024.
- **Mai -ikā > -ī** placed after the high-vowel shortening: ext −0.0063, plain +0.0006, but
  accent +0.0013. Rejected under the accent constraint. Mai's -ī forms (8 % of attested endings)
  stay unmodelled because of the chain's ī > i shortening.
- **Gowro apocope without -ī** (the round-2 generalisation): ext −0.0035, plain +0.0013,
  accent +0.0014. Rejected. It is the reason Gowro's -ikā variant comes out bare.
- **Dm `i {a ā} > ī`**: ext +0.0068. **bhatr/Bshk `i {a ā} > ī`**: worse. **Mai `u a > ū`**: ext +0.0034.
- **@morph citation rules**: these fire only on -ati/-ayati verbs, and verbs are never extended,
  so there is no double-ending problem in this group.

### 10.4 What the ending choices say
- **-aka.** It is the second choice (8–19 %) in Kunar, Pashai, Tirahi, Chilisso and Indus
  Kohistani, where the chain turns it into a retained final vowel: -a in Mai, Pas, Dm, Gaw, Shum
  and Tir; -o in Chil. It is nearly unused in Bashkarik (1 %) and Bhateri (0 %), where -aka > ∅,
  like the bare stem. That fits §3: those two lects contract the hiatus before apocope.
- **-akka.** It is used as much as -aka in Pashai and Shumashti (8–10 %) and in Bashkarik (5 %).
- **-ḍa, -la and -ra extensions** are productive but minor: 3–6 % combined in most lects. They
  are highest in Wotapuri (-araka 4 %, -aḍaka 3 %) and Bashkarik (-allaka 3 %, the Gawri -l
  diminutives).
- **Feminine -ikā** is used in every lect (3–8 %). It is largest in Dameli (8 %), Gawarbati (8 %)
  and Bhateri (6 %). Masculine -uka is significant in Chilisso (8 %), which has -ū̃/-u endings.

### 10.5 Should the menu change? (evidence from `menutest.py`: Δ train ext-aware NED from adding one item, number of reflexes choosing it)
Worth adding:
- **-āka** (the headword's final vowel replaced by -āka). Pas −0.0112 (110), bhatr −0.0076 (13),
  Mai −0.0050 (62), Tir −0.0045, Gaw −0.0025, Shum −0.0014; 0 in Bshk and Dm.
  - OIA/MIA -āka is real (Pk. -āa). In Pashai it mostly stands in for an attested -ā that the
    chain's -aka > -a cannot produce, so it is partly a surface fix.
  - It is the strongest candidate for Pashai and Kohistani.
- **-āla / -ālu**. Pas −0.0070 (131), Mai −0.0044 (78), bhatr −0.0046, Tir −0.0046, Gaw −0.0033,
  Tor −0.0028. These are real adjectival and agentive suffixes (Pk. -āla, -ālu; Pashai -āl,
  Kohistani -âl). Recommended.
- **-ikka**. Shum −0.0152 (37), Gaw −0.0047, Gowro −0.0031, Pas −0.0030 (46), Bshk −0.0027. It is
  the feminine counterpart of -akka (Kunar -ik). Recommended if -akka stays.
- **Plain feminine -ī** (replacing the final vowel). Only Dm gains much: −0.0062 (39), Dameli
  fem. -ī. Tor −0.0013; the rest ≈ 0, because -ikā already covers it.

Minor gains:
- **-ḍa, -ra, -la without -ka**. −0.001 to −0.003 everywhere, 2–35 reflexes each. They are
  surface variants of -aḍaka, -araka and -allaka. Low priority.
- **-ukka**. Wot −0.0038, Bshk −0.0029, Pas −0.0022.

Not worth adding:
- **-iya**. Only Tir −0.0077 (8); 0 elsewhere.
- **Dardic -u** (replacing the final vowel). ≤ −0.001.

Drops. Removing an item costs:
- **-aḍaka**: 0 in Bshk and Chil, +0.0006 in Gowro.
- **-araka**: 0 in Bshk and Chil.
- **-allaka**: 0 in Chil and Tir.
- **-akka**: +0.0003 in Chil, +0.0007 in bhatr.

With the chain's retroflex flapping, -aḍaka and -araka often give the same output (Bshk
gōrar), so one of the two is redundant for the Kohistani lects. Every item is used by at least
one large lect, so **nothing should be dropped**.

## 11. Verb citation forms (round 5)

Tools, both in `tmp/sound_changes/dardic-south/`:
- `abverb.py`: Δ train verb-menu, plain, extension-aware and accent NED per edit, flagging the
  four acceptance conditions.
- `verbshow.py`: attested verb citations next to every verb-menu output.

The rule files now `use` shared rules from `changes.sc`; round-5 edits are local lines marked
"round 5".

### 11.1 Scores (verbs only: plain NED with citation rules / verb-menu NED without them)

| lect | test verbs | test plain | test verb-menu before → after | train verb-menu before → after | test winners (after) |
|---|---|---|---|---|---|
| Pas | 63 | 0.442 | 0.399 → **0.384** | 0.4303 → 0.4162 | bare 45, root 10, ita 3, itum 2 |
| Mai | 44 | 0.478 | 0.486 → **0.417** | 0.4712 → 0.3933 | anaka 25, bare 9, root 6, tavya 3 |
| Dm | 34 | 0.433 | 0.375 → 0.375 | 0.4167 → 0.4145 | bare 22, root 5, aṇa 4, anaka 3 |
| Bshk | 11 | 0.345 | 0.315 (unchanged) | 0.4305 | bare 9, anaka 2 |
| Gowro | 10 | 0.335 | 0.409 → **0.322** | 0.5671 → 0.4571 | anaka 5, bare 4, root 1 |
| Gaw | 9 | 0.466 | 0.384 (unchanged) | 0.5045 | bare 6, itum 1, root 1, ita 1 |
| Wot | 9 | 0.496 | 0.430 → 0.422 | 0.3539 → 0.3430 | bare 7, ati 1, root 1 |
| Tor | 6 | 0.668 | 0.616 (unchanged) | 0.4620 | bare 3, root 2, ati 1 |
| Shum | 6 | 0.667 | 0.550 → 0.583 | 0.4508 → 0.4484 | bare 5, ita 1 |
| Tir | 4 | 0.492 | 0.400 (unchanged) | 0.3824 (plain 0.3602) | root 3, bare 1 |
| bhatr, Chil | 1 each | — | — | — | — |

The verb-menu score now beats the plain one in every lect with more than one test verb. On
train, Tir is the exception (0.382 against 0.360). Shumashti's test verb-menu NED rose on 6
pairs; the change was chosen on train.

### 11.2 Formations behind the citation forms
- **Indus Kohistani, Zoller's infinitive -ǎṽ** (−aṽ once tone is dropped): it continues the verbal
  noun **-anaka**. The chain is -anaka > -anaa (velar loss) > -aṽaa (n > ṽ between a-vowels) >
  -aṽa (hiatus) > -aṽ (apocope). Examples: kālayati → kālǎṽ, gālayati → galǎṽ. After the change,
  -anaka wins for 104 of 189 train verbs.
- **Gowro -ō̃** (Zoller -ō̃̌): also **-anaka**. The post-lenition -anaa contracts to -ō̃
  (kalpáyati → kapō̃̌, ālapati → alō̃̌).
- **Pashai, Dameli, Wotapuri, Shumashti**: CDIAL and the modern sources cite the **bare stem**.
  The best derivation is the present **-ati** with the ending lost, which keeps the stem-final
  consonant lenited (*avalōḍayati → oroṛ), or the plain root. Dameli also uses **-aṇa** (Perder's
  -an/-ani inflected forms) and **-ita**.
- **Bashkarik**: bare headword; Gawri's -ánt participles partly match **-anaka**.
- **Gawarbati**: bare headword or root, plus some **-itum**.

### 11.3 Rules added
All four acceptance conditions hold for each rule:
- **Mai** `n > ṽ / a _ a` right after velar loss [@x-dardic-south-n-nasal-v], with the hiatus rule
  moved before apocope. Δ train: verb-menu −0.0779, plain −0.0002, extension-aware −0.0061,
  accent −0.0008.
  - The unrestricted `n > ṽ / V _ V` also qualifies but is marginally worse.
  - Moving the hiatus rule alone costs verb-menu +0.0075.
- **Gowro** `a n a a > ō̃ / V C* _ #` [@hiatus]. Δ verb-menu −0.1101, plain 0.0000,
  extension-aware −0.0098, accent 0.0000.
  - The unrestricted version (`a n a (a) > ō̃ / _ #`) also turns the noun *tanaka into tō̃, costing
    plain +0.0013 and accent +0.0027. The `V C*` condition (at least one syllable before -anaa)
    avoids that.
- **Pas, Dm, Wot, Shum** `{i ē} > ∅ / a _ #` before apocope [@hiatus]. Final -ai/-aē (from
  -ati/-atē) loses its second vowel, so apocope then removes the -a and leaves the lenited stem.
  Δ verb-menu: Pas −0.0141, Dm −0.0022, Wot −0.0109, Shum −0.0024. Plain 0 and accent 0 in all
  four; extension-aware −0.0003 to −0.0022.
- **Pas** `{ṭ ḍ} > ṛ / V _ #` [@flap]. Δ verb-menu −0.0027, plain 0, extension-aware −0.0004,
  accent 0. It flaps root-final retroflexes.

### 11.4 Rejected
- **Moving hiatus contraction before apocope across the board.** Pas: verb-menu −0.0155 but
  extension-aware +0.0035 and accent +0.0033. Dm, Gaw and bhatr are worse on verbs. Wot and Shum
  are worse on extension-aware and accent.
- **The -ai rule in Gaw and Tir.** Gaw is worse on all scores. Tir gains verb-menu −0.0556 but
  loses plain +0.0040 to +0.0047 and accent +0.0012 to +0.0050. So Tir still does worse under the
  verb menu than plain on train.
- **Pas: all postvocalic ṭ ḍ > ṛ** (plain +0.0034), and the same rule at the end of the chain
  (plain +0.0038).
- **Gowro: `n > ṽ / a _ a` alone**: verb-menu −0.0037, but plain +0.0018 and accent +0.0016.
- **Not attempted**: Bashkarik's -ánt participle (not in the verb menu), and Torwali and
  Gawarbati beyond the -ai test (too few verbs). The @morph citation rules stay in place for the
  plain score, as the brief requires.

## Round 6 audit

Tools:
- `sound_changes/residuals.py` (outputs in `tmp/sound_changes/dardic-south/r6/<lect>.txt`).
- `tmp/sound_changes/dardic-south/g6.py`: `gates.py`-style verdicts for a single edit, with a
  better/worse word diff.
- `r6before.py`: runs `gates.py` against the file with every "round 6" line removed.

All installed lines are tagged "round 6". Every lect passes `gates.py` against its pre-round-6
file (`r6/gates-all.txt`; the final gates for Mai, Bshk, Gowro and Tir were rerun after the mb
rule).

### Error profile (shared by all 12 lects)
- **Ending mismatches: 46–59 % of the error.** Mostly lexical suffixes and inflection:
  - Zoller's -âh/-ǎh/-ǎy/-ī̂ (Mai, Gowro, bhatr);
  - Perder's inflected forms -i/-ē/-n/-numa (Dm);
  - Gawri's -ant participles (Bshk);
  - Pashai dialect endings -ā/-e/-o.
- **Consonant insertion/deletion: 11–17 %.**
- **Consonant quality: 7–10 %.**
- **Syncope missing: 6–9 %.** Lexical: Tor būmel, pin; Mai nihā̌r.
- **Vowel quality and length: 3–8 %.**

The systematic errors found were mostly **unsimplified stop clusters** (kt, pt, ts, tk, mb) and a
few **hiatus leftovers** (-aī, the -akka ending).

### Kept changes

**All 12 lects: `{k p g b} {t tʰ d dʰ} > $2`**, before degemination [@cluster-simplify
@assim-stop]. saptá > sat, taptá > tat, bʰaktá > bʰā̌tʰ, *gʰr̩pta > git.
- Alone it passed every gate in all 12 lects.
- Train extension-aware NED Δ: −0.0008 (Pas, Dm) to −0.0037 (bhatr, Wot).
- The fully general stop+stop version failed in 9 lects.

**`t s > ʦ` or `> c`**, whichever scored better on train [@cluster-simplify @affrication].
útsa > ūʦ, mátsya > maʦ-.
- ʦ: Mai, Bshk, Dm, Gaw, bhatr, Tir.
- c: Pas, Tor, Shum, Wot.
- Gains −0.0002 to −0.0045. Neutral in Gowro and Chil (not installed).

**Mai, Pas, Dm, Tor: `{t d} {k kʰ g gʰ p pʰ b bʰ} > $2`**, dental + labial/velar stop > the second.
utpaṭati > ōpuṛ. Gains −0.0002 to −0.0017.

**Mai, Bshk, Gowro, Tir: `m {b bʰ} > m`.** Gains −0.0004 to −0.0006. Gaw was also accepted, but
at −0.0002 it was not installed. Rejected in Pas and bhatr.

**Mai: `kʰ > h / V C* a _ #`**, after final aspiration. -akka > -akʰ > -ah, Zoller's -âh
(kṣīrá+akka > ʦ̣ʰirâh, kōkila > kōlâh).
- Δ extension-aware −0.0097/−0.0116, plain +0.00003, accent +0.00004.
- The strict gate rejected it by one word: śalakka, attested both śalā̌kʰ and ʦasakk > caskâh. It
  was kept as justified: it fixes 33 -âh reflexes.

**Gowro: `k > h / V C* a _ #`**, the same -akka > -ah (kárṇa+akka > kanǎh), Δ −0.0133/−0.0116.
Rejected for bhatr (test accent +0.0037) and a no-op in Chil.

**Gaw and Gowro: `a ī > ī / _ #`**, before apocope [@hiatus]. tapyatí > tapī́,
*kʰasarati > kʰasarī́.
- Gaw: Δ −0.0011, 5 better / 0 worse.
- Gowro: Δ −0.0023; its gains come from -aī numerals and verbs.

**Tor: `{ṣ s} > ∅ / V _ C`**, before the intervocalic voicing, so it feeds it.
niṣkālayati > nigāl, tīkṣṇá > tīn; 7 better / 1 worse (kr̩ṣṇá).
- Δ −0.0019/−0.0016.
- Placed after voicing it gains only −0.0013 on train.
- In the other 11 lects it fails (+0.0007 to +0.0075 on train extension-aware, or a worse
  test/accent score).

### Before → after (train/test)

| lect | ext-aware NED | plain NED | verb NED | accent NED |
|---|---|---|---|---|
| Mai | 0.4591/0.4769 → **0.4461/0.4629** | 0.5208/0.5423 → 0.5180/0.5404 | 0.3893/0.4163 → 0.3865/0.4066 | 0.5453 → 0.5421 |
| Pas | 0.4507/0.4556 → 0.4481/0.4528 | 0.5184/0.5473 → 0.5153/0.5436 | 0.4053/0.3693 → 0.4041/0.3648 | 0.5849 → 0.5815 |
| Dm | 0.4338/0.4024 → 0.4319/0.3982 | 0.5038/0.4758 → 0.5014/0.4709 | 0.4003 → 0.4007 | 0.5708 → 0.5635 |
| Tor | 0.4521/0.4915 → 0.4458/0.4849 | 0.4885/0.5377 → 0.4805/0.5318 | 0.4459 → 0.4412 | 0.5156 → 0.5148 |
| Gaw | 0.4346/0.4044 → 0.4288/0.4020 | 0.5196/0.4990 → 0.5151/0.4979 | 0.4966 → 0.4942 | 0.5989 → 0.5957 |
| Bshk | 0.4216/0.4563 → 0.4163/0.4521 | 0.4585/0.4890 → 0.4536/0.4843 | = | 0.5061 → 0.5037 |
| Gowro | 0.4637/0.5273 → **0.4458/0.5115** | 0.5102/0.5704 → 0.5077/0.5692 | 0.4607 → 0.4551 | 0.5442 → 0.5406 |
| bhatr | 0.4719/0.5307 → 0.4635/0.5277 | 0.5059/0.5739 → 0.4970/0.5709 | = | 0.5394 → 0.5285 |
| Shum | 0.4341/0.4880 → 0.4301/0.4828 | 0.5220/0.6033 → 0.5170/0.5974 | 0.4423 → 0.4407 | 0.5513 → 0.5486 |
| Wot | 0.3694/0.4057 → 0.3640/0.3953 | 0.4146/0.4453 → 0.4082/0.4350 | 0.3430 → 0.3419 | 0.4980 → 0.4884 |
| Tir | 0.3967/0.4239 → 0.3917/0.4153 | 0.4561/0.4997 → 0.4516/0.4888 | = | 0.5355 → 0.5230 |
| Chil | 0.4030/0.4559 → 0.3998/0.4507 | 0.4707/0.4753 → 0.4688/0.4695 | = | — |

Plain, verb and accent NED improve or stay equal everywhere. The one exception is Dm's train verb
NED, +0.0004, within the gate.

### Rejected candidates
- **Pas: final `{d ḍ b} > ∅` after a nasal, and -st > -s.** Extension-aware −0.0017 but accent
  +0.0037 and verb +0.0031. This is Pashai dialect variation.
- **Pas and Mai: aphaeresis of initial a-/u- before v.** Mai +0.0004; Pas worse on test and
  accent.
- **Mai: keeping ī ū long in (future) monosyllables or initial syllables.** Four variants; all
  worsen the accent score (+0.0016 to +0.0035), and plain rises by up to +0.0033.
- **ava > ō** in Gaw, Mai, Pas, Dm and Gowro. Gaw gains −0.0016/−0.0022 but loses accent +0.0029;
  the others are worse.
- **Dm: CDIAL block `ā > ɔ` instead of `ā > o`**: +0.0100.
- **Tor: `{ai au} > {ē ō}` restored**: train 0.0000 (it only helps test tailá > tel), so the
  primary gate fails.
- **Tor: medial -aa-/-ōu- contractions**: they pass the gates but fix only 2 words.
- **Tir: final hiatus `a > ∅ / V _ #`**: train −0.0114, but test +0.0049.
- **Compensatory lengthening in future monosyllables** (`a > ā / # C* _ C C V #`, as in kárṇa >
  kān):
  - Accent or plain worse in Mai, Tor, Pas and Bshk.
  - bhatr: extension-aware −0.0028/−0.0132, but plain +0.0034 on train. That is not the
    bare-vs-extended artifact, so it was not justified.
- **Fully general stop + stop > second stop**: 9 lects fail on test or accent.
- **-aka > -ah via `k > h / a _ a #`**: it destroys the -anaka infinitives (Mai and Gowro verb
  NED +0.07/+0.11).

### What remains
Mostly lexical and source variation, left alone:
- Zoller's gender and diminutive suffixes (-âh vs -ǎy vs -ī̂ on the same stem) and unpredictable
  vowel length.
- Perder's and Gawri's inflected and participial forms. Bashkarik -ánt would need a present
  participle -ant in `VERB_MENU`; there was no time for a placebo test.
- Pashai dialect doublets (ā/ō, tr/ɬ, ṣ/ś).
- Suppletion and mislinks.

No lect was left undone.

### Round 6 second pass

Method:
- `tmp/sound_changes/dardic-south/pat6.py` and `pat6x.py` tabulate every non-ending edit operation
  in misses with NED < 0.6, by (predicted → attested) and by C/V context, with up to 40 examples.
- Each candidate went through `g6.py` (gates plus a better/worse diff).
- `r6before.py` then re-gated every lect against its pre-round-6 file (`r6/gates-all2.txt`).
  All 12 pass.

**Kept (second pass):**
- **Mai: intervocalic t now splits by the following vowel.** `t > ∅ / V _ {a ā}` comes first,
  then `{t d} > l / V _ V` for the rest; this replaces the shared `use t-l`.
  - Lost before a: catúraḣ > ʦaur, cētáyati > ʦē̂r, and -ati verbs become bare stems.
  - Kept as l before high vowels: gīti > gī̂l, cíti > ʦîl.
  - d > l everywhere.
  - 71 better / 14 worse. Δ train/test: ext-aware −0.0067/−0.0053, verb +0.0009/−0.0045,
    accent −0.0011/−0.0006.
  - Plain "t and d both lost" fixes more words but fails the test verb gate (+0.0083). So does "t
    lost only word-finally".
- **Late penult syncope** `{a i u} > ∅ / V C _ C V C* #` after apocope (modern trisyllables):
  - Mai: pitala+akka > pitlâh, *kōppara > kōprâh, vasantá > basnâh, *aṭṭakkalā > aṭkǎl. It is
    blocked before the -aṽ infinitive, which keeps its stem vowel. 53 better / 17 worse,
    Δ −0.0022/−0.0022. Without the infinitive exception the test verb NED is +0.0016.
  - Also accepted in Bshk (−0.0009/−0.0016) and Wot (−0.0018/−0.0017).
  - Rejected in Pas, Dm, Gaw and Tir (worse extension-aware or accent) and in Tor (train verb
    +0.0021). Gowro gains −0.0053, but accent +0.0004, so it was not installed.
- **Long vowels shortened outside the final (stressed) syllable** `{ā ī ū ē ō} > {a i u e o} / _ C
  C* V` at the end of the chain:
  - **Pas**: *ākṣapāṭa > aṣpā́ṛ, āṣāḍʰīya > aṣaṛī́, kāśmīra > kaśmīr. 94 better / 48 worse,
    Δ −0.0044/−0.0070, verb −0.0081/−0.0119, accent −0.0032/−0.0061. This fits the round-3
    finding that the Pashai accent sits on the last syllable.
  - Also accepted in **Bshk** (−0.0031/−0.0045) and **Tir** (−0.0013/−0.0071).
  - Tor gains −0.0079 on train but its test accent rises by +0.0143 (10 pairs). Mai, Dm, Gaw,
    bhatr and Wot fail.

**Rejected (second pass):**
- **Mai: anticipatory ŷ/y epenthesis before C + final -i** (kali > kaŷli). 15 better / 35 worse:
  Zoller's forms vary within the same words.
- **Mai: jj > ź.** 4 better / 5 worse.
- **Mai: syncope restricted to a, or to -ah/-akʰ endings.** Smaller gains than the version kept.
- **Dm: r-metathesis into the first syllable** (takrá > traka, tántra > tran, kárman > kram,
  *tarpati > trap). 7 better / 6 worse; test extension-aware +0.0100. It fails before sibilants
  and after long a (kárṣati, gā́tra, pārśvá). Partial and lexical.
- **Pas: a > e before C + final i** (umlaut; also Gaw), **u > o before a**, **h- loss** and
  **s-C > C**. All fail the gates. Pashai dialect doublets.
- **Bshk: dropping the round-2 `ō > o / _ C C`.** +0.0013.

**Final before → after, round 6 total (train/test):**

| lect | ext-aware NED | plain NED | verb NED | accent NED |
|---|---|---|---|---|
| Mai | 0.4591/0.4769 → **0.4373/0.4554** | 0.5208/0.5423 → 0.5169/0.5386 | 0.3893/0.4163 → 0.3855/0.4021 | 0.5453/0.5618 → 0.5402/0.5576 |
| Pas | 0.4507/0.4556 → **0.4436/0.4458** | 0.5184/0.5473 → 0.5156/0.5398 | 0.4053/0.3693 → 0.3960/0.3529 | 0.5849/0.6399 → 0.5783/0.6313 |
| Dm | 0.4338/0.4024 → 0.4319/0.3982 | 0.5038/0.4758 → 0.5014/0.4709 | 0.4003 → 0.4007 | 0.5708 → 0.5635 |
| Tor | 0.4521/0.4915 → 0.4458/0.4849 | 0.4885/0.5377 → 0.4805/0.5318 | 0.4459 → 0.4412 | 0.5156 → 0.5148 |
| Gaw | 0.4346/0.4044 → 0.4288/0.4020 | 0.5196/0.4990 → 0.5151/0.4979 | 0.4966 → 0.4942 | 0.5989 → 0.5957 |
| Bshk | 0.4216/0.4563 → **0.4123/0.4460** | 0.4585/0.4890 → 0.4529/0.4819 | 0.4171/0.3152 → 0.4159/0.2545 | 0.5061 → 0.5009 |
| Gowro | 0.4637/0.5273 → **0.4458/0.5115** | 0.5102/0.5704 → 0.5077/0.5692 | 0.4607 → 0.4551 | 0.5442 → 0.5406 |
| bhatr | 0.4719/0.5307 → 0.4635/0.5277 | 0.5059/0.5739 → 0.4970/0.5709 | = | 0.5394 → 0.5285 |
| Shum | 0.4341/0.4880 → 0.4301/0.4828 | 0.5220/0.6033 → 0.5170/0.5974 | 0.4423 → 0.4407 | 0.5513 → 0.5486 |
| Wot | 0.3694/0.4057 → 0.3623/0.3936 | 0.4146/0.4453 → 0.4075/0.4350 | 0.3430 → 0.3375 | 0.4980 → 0.4874 |
| Tir | 0.3967/0.4239 → 0.3904/0.4082 | 0.4561/0.4997 → 0.4514/0.4888 | 0.3639 → 0.3537 | 0.5355 → 0.5155 |
| Chil | 0.4030/0.4559 → 0.3998/0.4507 | 0.4707/0.4753 → 0.4688/0.4695 | = | — |

Every lect passes `gates.py` against its pre-round-6 file.

What remains in the non-ending classes is lexical or source variation:
- Zoller's vowel length and ŷ;
- Perder's inflection;
- Pashai dialect doublets (ā/ō, a/e, h-);
- reduplicated and compound headwords (gʰǎr-gʰar, … karǎṽ).

### Round 6 third pass

Tools, all in `tmp/sound_changes/dardic-south/`:
- `vmiss.py`: train verb misses, with the winning menu form and the pred→gold ending pairs.
- `near6.py`: single-edit near misses (NED ≤ 0.25), grouped by segment pair, position and
  context.
- `score --ext-loo` for all 12 lects, to find rules with ext-aware LOO ≤ 0.

**(1) Verbs.**
- **Mai.** 76 of 154 verb misses already end in the right -aṽ. The rest are stem-internal:
  - Zoller shortens the stem ā before -ǎṽ (gālayati > galǎṽ). A pretonic-shortening rule for Mai
    still fails on accent.
  - Also -iǎṽ against -iṽ, and y against ∅.
- **Pas, Dm, Bshk.** The misses are:
  - Pashai -ayati stems keep a final -a (cara for car). `a > ∅ / a _ #` before apocope fixes them
    but breaks the -aka nouns: ext-aware +0.0020/+0.0067.
  - Dameli: Perder's inflected forms (kur-an, kur-um, -numa).
  - Bashkarik: Gawri's -ánt participles (karant).
- **Wot: `a > ∅ / a _ #` before apocope** (final -aa from -aya-/-aka- contracts).
  - Δ ext-aware −0.0001/−0.0154, verb −0.0072/−0.0188, accent −0.0032.
  - Kept. It is rejected in Pas, Dm and Shum.

**(2) Near misses.**
- **Mai `i {a ā} > ī / V C* _ #`** before apocope: fem. -ikā > -ī in polysyllables
  (*guṭṭʰikā > guṭī̌, paṭṭikā > paṭī̌).
  - Δ ext-aware −0.0088/−0.0076, plain −0.0001.
  - Accent +0.00002, from one word, mŕ̩ttikā. Kept as justified, like the round-6 -âh rule.
  - Without the polysyllable condition, síkatā and jiyā́ also changed and accent rose +0.0003.
- **Mai `{k ṭ t p} > {g ḍ d b} / {ŋ ṇ n m} _`**: voiceless stops voiced after a nasal
  (káŋkata > kaŋgǎh, *ḍʰiŋka > ḍʰǐŋgi). It generalises the old ṇṭ > ṇḍ, which is now redundant
  (ext LOO 0). Δ −0.0008/−0.0003.

Near-miss groups left as lexical or source variation:
- Mai: the -ī / -ū / -ā / -o gender endings on bare stems, -ah against -ǎy against -ǎṽ after
  -akka, ŷ epenthesis, and h after sonorant onsets (zhǎr, lhǎṭʰ).
- Pas: fem. -ā, and ā/ō.
- Dm: Perder's inflection.
- Gowro, Gaw, Bshk: vowel length.
- Bshk: final -ī. Deleting it after the final-lengthening rule passed the gates
  (−0.0031/−0.0044) but was not installed. Inserting the lengthening and then deleting it is not
  a coherent history, and the coherent version (dropping -i before the lengthening) fails with
  0 better / 15 worse.

**(3) Rules with ext-aware LOO ≤ 0.**
- Apart from the @morph citation rules and the accent placements, which carry no ext-aware
  value by design, the candidates were:
  - Dm final-consonant loss (−0.0022);
  - Gaw pr/br > pl/bl (−0.0015);
  - Shum aya > ē (−0.0008);
  - Mai ṇṭ > ṇḍ (0, redundant).
- Dropping them fails the gates:
  - Dm: plain +0.0030, accent +0.0015.
  - Gaw: test ext-aware +0.0082, test verb +0.0370.
  - Shum: plain +0.0018, verb +0.0085.
  - Mai: no change at all. It stays for now.

**Final round-6 totals (train/test, against the pre-round-6 files, current engine):**

| lect | ext-aware NED | plain NED | verb NED | accent NED |
|---|---|---|---|---|
| Mai | 0.4587/0.4758 → **0.4273/0.4464** | 0.5208/0.5423 → 0.5160/0.5383 | 0.3846/0.4027 → 0.3808/0.3897 | 0.5453 → 0.539 |
| Pas | 0.4510/0.4556 → 0.4439/0.4458 | 0.5184/0.5473 → 0.5156/0.5398 | 0.4069/0.3693 → 0.3973/0.3529 | 0.5849 → 0.578 |
| Dm | 0.4332/0.4024 → 0.4313/0.3982 | 0.5038/0.4758 → 0.5014/0.4709 | 0.3953 → 0.3957 | 0.5708 → 0.563 |
| Tor | 0.4521/0.4915 → 0.4458/0.4849 | 0.4885/0.5377 → 0.4805/0.5318 | 0.4459 → 0.4412 | 0.5156 → 0.514 |
| Gaw | 0.4343/0.4044 → 0.4286/0.4020 | 0.5196/0.4990 → 0.5151/0.4979 | 0.4947 → 0.4923 | 0.5989 → 0.595 |
| Bshk | 0.4216/0.4563 → 0.4123/0.4460 | 0.4585/0.4890 → 0.4529/0.4819 | 0.4171/0.3152 → 0.4159/0.2545 | 0.5060 → 0.500 |
| Gowro | 0.4628/0.5243 → 0.4449/0.5085 | 0.5102/0.5704 → 0.5077/0.5692 | 0.4488 → 0.4433 | 0.5442 → 0.540 |
| bhatr | 0.4719/0.5307 → 0.4635/0.5277 | 0.5059/0.5739 → 0.4970/0.5709 | = | 0.5394 → 0.528 |
| Shum | 0.4358/0.4880 → 0.4318/0.4828 | 0.5220/0.6033 → 0.5170/0.5974 | 0.4565 → 0.4549 | 0.5513 → 0.548 |
| Wot | 0.3694/0.4057 → 0.3622/0.3782 | 0.4146/0.4453 → 0.4072/0.4315 | 0.3430/0.3923 → 0.3302/0.3630 | 0.4980 → 0.484 |
| Tir | 0.3967/0.4239 → 0.3904/0.4082 | 0.4561/0.4997 → 0.4514/0.4888 | 0.3639 → 0.3537 | 0.5355 → 0.515 |
| Chil | 0.4030/0.4559 → 0.3998/0.4507 | 0.4707/0.4753 → 0.4688/0.4695 | = | — |

All 12 lects pass `gates.py` against their pre-round-6 files (`r6/gates-all3.txt`). The "before"
column is rescored under today's engine, so it differs slightly from the second-pass table.

### Round 6 fourth pass

This pass went through Gowro, bhatr, Tor, Shum, Dm and Mai with `pat6.py`, `near6.py` and
`vmiss.py`, plus a placebo test of `%verb-menu`.

**Kept (both Gowro):**
- **`r > b / m _`** before the Stop+r rules: mr > mb (āmrá+akka > ambǎh, tāmrá > tāmbâh).
  - Δ ext-aware −0.0011, plain −0.0011, accent −0.0003.
  - Mai and Pas also pass, but only at −0.0002 (a handful of words), so they were not installed.
  - Rejected in bhatr, Shum, Dm and Gaw; a no-op in Tor and Bshk.
- **`{a i u} > ∅ / V C _ C V C #`**: syncope of an open penult before a closed final syllable
  (kambalá+akka > kamlǎh, *gaḍa-baḍa > gǎḍbaḍ).
  - Δ ext-aware −0.0027, plain −0.0005, accent −0.0003.
  - The Mai-style version, which allows an open final syllable, also produced mahallaka > mahla
    and drōṇadʰānuka > …dʰnu and cost accent +0.0004.
  - A version blocked before final -a scored ext-aware −0.0009, verb −0.0113. It was not chosen,
    because ext-aware on train is the primary gate.

**Gowro, round 6 total**, train/test: ext-aware 0.4628/0.5243 → **0.4410/0.5085**, plain
0.5102/0.5704 → 0.5061/0.5692, verb 0.4488/0.2400 → 0.4433/0.2400, accent 0.5442/0.6173 →
0.5400/0.6129. ACCEPT.

**Rejected:**
- **Shum: narrowing the round-2 `a > æ / _ V`** (before i/u only, or not before a/ā), and adding
  an æā > ā contraction. The contraction scores ext-aware −0.0022 on train, but test ext-aware
  is +0.0017 and test verb +0.0278 (1 of 6 test verbs).
- **bhatr and Tor near misses**: length (u/ū, a/ā), Zoller's palatal letters (ť ď) and æ. All
  are lexical or notational; each involves only 1–3 pairs.
- **Bshk `%verb-menu +ant`** (Gawri's -ánt participle), placebo `+alk`. Both give Δ 0.0000,
  and so do +anta and +alka. The chain regularly simplifies nt > n and drops final -t after a
  vowel (dánta > dân), so a Gawri -ánt form cannot surface from any OIA formation through this
  chain. It is a Gawri morphological formation, not a sound change. Not installed.
- **Dm `%verb-menu`**: Perder's -nun/-num/-an forms are inflection of the modern stem. Not
  attempted.

The other lects are unchanged since the third pass (see its table). Most of what remains is
lexical.

### Round 6: re-gate under the ext_aned accent gate

`gates.py` now gates on accent scored on the variant that the extension-aware score picks
(`ext_aned`), not on the bare headword. Script: `tmp/sound_changes/dardic-south/regate.py`.

**The two Mai rules kept earlier with a written accent justification now pass cleanly:**

| rule | Δ ext-aware train/test | Δ ext_aned train/test | Δ bare accent | verdict |
|---|---|---|---|---|
| `kʰ > h / V C* a _ #` (-akka > -âh) | −0.0108 / −0.0108 | −0.0136 / −0.0128 | 0.0000 | ACCEPT |
| `i {a ā} > ī / V C* _ #` (-ikā > -ī) | −0.0087 / −0.0076 | −0.0186 / −0.0164 | 0.0000 | ACCEPT |

**Candidates rejected earlier only on accent, re-gated:**
- **Gowro: syncope before an open final syllable too** (`{a i u} > ∅ / V C _ C V C* #`, the Mai
  form). Now ACCEPT: ext-aware −0.0027/−0.0033, verb −0.0113/−0.0071, ext_aned −0.0008/−0.0073.
  The bare accent is still +0.0007. It **replaces** the narrower closed-final version from the
  fourth pass.
- **bhatr: `k > h / V C* a _ #`** (-akka > -ah, as in Mai and Gowro). Now ACCEPT: ext-aware
  −0.0037/−0.0072, ext_aned −0.0029/−0.0046. The bare accent on test is still +0.0037, and test
  plain +0.0018; the plain gate is on train only. **Installed.**
- **Mai: -iā > -ī without the polysyllable condition.** ACCEPT, but only −0.0002 better than the
  installed version, so the narrower rule was kept.
- **Still rejected:**
  - **Gaw ava > ō**: ext_aned +0.0004.
  - **Tor: shortening outside the final syllable**: ext_aned +0.0145/+0.0143.
  - **Mai: high-vowel length kept in monosyllables**: ext-aware +0.0071, ext_aned +0.0172.

**Re-gated round-6 totals for the changed lects** (train/test, pre-round-6 → now):
- **Mai**: ext-aware 0.4587/0.4758 → 0.4274/0.4464; ext_aned 0.4660/0.4842 → 0.4187/0.4442.
- **Gowro**: ext-aware 0.4628/0.5243 → **0.4383/0.5052**; ext_aned 0.4870/0.5555 →
  0.4477/0.5152.
- **bhatr**: ext-aware 0.4719/0.5307 → **0.4598/0.5205**; ext_aned 0.4910/0.5432 →
  0.4772/0.5332.

All three pass (ACCEPT).

## Round 7

Tools are in `tmp/sound_changes/dardic-south/r7/`:
- `tonetab.py LECT [split]` traces every tone-marked reflex back to its OIA vowel. It uses the
  latent accent tier as a tracer: the installed chain minus its placement rules, plus one marker
  `$1+stress` rule after `V > $1-acc`, for each feature (Vedic accent, OIA long vowel, preceding or
  following voiced aspirate/h, voiceless aspirate). Output: `tone-<lect>-<split>.tsv` and the tables.
- `predfeat.py` / `goldfeat.py` give tone class by modern-form features.
- `mk.py`, `latent.py`, `bshk.py` build candidates; `vs.sh` gates one candidate against another.
- `place.py LECT` is the placement battery.
- `vq.py`, `vsub.py` handle vowel quality.
- Pre-round-7 files are `OIA-<lect>.pre-r7.sc`.

All installed tone rules are **tone-gated**: gates.py returned "ACCEPT (prosody-only)". Plain, ext and
verb scores are exactly unchanged, ext_tone improves on train, and neither ext_tone nor ext_acc gets
worse on train or test. The rules sit at the end of the chain, after every segmental rule.

Sources that mark stress but not tone type get an `@notation` strip rule at the head of their
`%source` block: RDH and CDIAL in Mai, Gowro, bhatr and Bshk; Strand in bhatr; CDIAL and Strand in Pas.
- The strip rule turns ǎ/â back into the vowel with the generic accent.
- Without it, their acute-marked forms would be charged for Zoller's tone type.
- Tails that match plain vowels still fire (RDH `{a i u} > {ā ī ū}`, Zoller-bhatr/Gawri `a > æ`,
  which gets a tone-marked twin).

`%gold-drop tone` was removed from Mai, Gowro, bhatr and Bshk.
- When removed, before the 09:10 engine change, this was a no-op.
- Since 09:10 the line would hide Zoller's and Gawri's marks from the accent and tone scores. The
  "before" numbers below are therefore the pre-round-7 file *without* that line, so they score the
  same gold as "after".

### The tone question: does the Vedic accent predict tone type?

**Mostly no.** Tone type is predicted by the onset of the tone-bearing syllable (a depressor
effect) and by a final nasal, not by the Vedic accent.

The Vedic accent gives a small bias towards falling on the vowel it survives on. The bias appears
in all three Zoller lects. No Vedic-conditioned rule passes, and the bias shrinks on test.

**Indus Kohistani (Zoller, train, nouns, n = 1484).** Overall: falling 57 %, rising 37 %, high 6 %.
Verbs (n = 144) are 74 % rising: the -ǎṽ infinitive.

| factor (tone-bearing = last syllable) | falling | rising | n |
|---|---|---|---|
| Vedic accent on the tone vowel (accented etyma) | 66 % | 34 % | 145 |
| other vowel accented | 57 % | 35 % | 380 |
| … Vedic + OIA long vowel | 76 % | 24 % | 49 |
| … Vedic accent, test split | 59 % | 41 % | 39 (vs 55 % of 119) |
| paroxytone / oxytone / unaccented etymon | 65 / 54 / 56 % | 31 / 38 / 39 % | 210 / 270 / 906 |
| OIA long vs short vowel | 63 vs 56 % | 37 vs 36 % | 375 / 947 |
| attested closed vs open syllable | 60 vs 52 % | 40 vs 34 % | 911 / 573 |
| **attested onset: voiceless stop/affricate** | **68 %** | 32 % | 342 |
| onset fricative / sonorant | 66 / 56 % | 34 / 43 % | 76 / 507 |
| onset voiced stop | 49 % | 51 % | 269 |
| **onset voiced aspirate / h** | 34 / 46 % | **66 / 54 %** | 67 / 52 |
| OIA voiced aspirate/h before the vowel (traced) | 42 % | **57 %** | 111 |
| predicted final -ṽ (from -ana-) | 15 % | **79 %** | 157 |
| predicted final -h (from -akka) | **75 %** | 18 % | 260 |

- **Gowro** (Zoller, train n = 290 nouns): rising 74 %, verbs 94 %. The Vedic-accented tone vowel
  is 50 % falling (n = 28) against 24 % otherwise; Vedic + long is 6 of 8 falling.
- **bhatr** (Zoller, n = 115): rising 58 %. The Vedic-accented vowel is 60 % falling (n = 10)
  against 31 %.
- **Bshk** (Gawri, n = 357): high 67 %, low 14 %, rising 16 %, falling 4 %.
  - The Vedic accent does nothing: Vedic 65/14/16 % against other 69/11/14 % (high/low/rise).
  - After a kept voiceless aspirate onset the tone is low 46 %, rising 38 %, high 8 % (n = 26).
  - After a lost OIA voiced aspirate it is high 39 %, low 42 % (n = 31).
- **Tirahi** (CDIAL): all 5 grave (low) marks follow an OIA voiced aspirate (ḍʰāla > dàl,
  bʰū́mi > bʰùm, dʰūmá > dʰùng).
- **Pashai** (Lehr): 5 of 5 carons follow a lost aspirate/h (gʰāsá > gā̌ns, hásta > ā̌st,
  hā́rdi > ā̌ṛ).

Vedic-conditioned rules, gated (Δ ext_tone train/test against the lect's best non-Vedic tone rules):
- **Mai**:
  - Vedic final vowel → falling: +0.0001/+0.0004. REJECT.
  - Vedic long final vowel → falling: 0.0000/0.0000. REJECT.
- **Gowro**: Vedic long vowel → falling (else rising): +0.0003/0.0000. REJECT.
- **bhatr**: Vedic final vowel → falling: −0.0039/+0.0006. Rejected by the test veto.
- The Vedic vowel is read with the new `Vedic` feature before placement (`Vedic > $1+acc / _ C* #`,
  then `{á …} > {â …}`).

Positional and weight alternatives also fail to predict type:
- In Mai, closed/open, long/short (OIA or modern), word length and vowel quality all leave falling
  as the majority in every cell.
- In Gowro and bhatr, rising is the majority in every cell.

**Placement (where the tone sits) was re-audited with `ext_acc`** (`place.py`: first σ, last σ,
penult, first long vowel, long final else first σ, long final else penult):
- Last σ still wins in Dm, Tor, Gaw and Wot.
- First long vowel, else first σ, still wins in Shum.
- **Bshk changes to the last σ**: ext_acc −0.0035/−0.0069 against round 3's "long final vowel,
  else first σ". That rule was chosen on bare headwords, which lack Gawri's accented suffixes.
  Every Kohistani and Kunar lect with a positional accent now puts it on the last syllable,
  except Shumashti.

### Installed tone rules (mechanisms)

- **Mai** (three rules):
  - The accented last syllable is **falling** by default (kaṇṭa > kâṇḍ, dʰā́sas > dʰā̂s,
    kṣīrá+akka > ʦ̣ʰirâh).
  - It is **rising before a final nasal** -ṽ/-m (kālayati > kālǎṽ, gáḍati > gaṛǎṽ, ḍōmba > ḍū̌m).
  - It is **rising after a depressor onset** bʰ dʰ gʰ ḍʰ h b g (gʰāsá > gʰā̌, gʰr̩tá > gʰī̌,
    káŋkata > kaŋgǎh, *gāva > gā̌).
  - Δ ext_tone train/test: all-falling −0.0225/−0.0219. Adding the nasal and breathy rules brings it
    to −0.0262/−0.0254; adding b g gives −0.0004/−0.0002 more.
  - Rejected variants:
    - all voiced obstruents (+0.0004 on train against the breathy set);
    - adding z (test +0.0013), or z v b g together (test +0.0010); b g alone pass;
    - rising before ŋ (+0.0008);
    - high on vowel-initial syllables (+0.0015);
    - breathy anywhere before the tone vowel (+0.0006);
    - breathy + (C) (+0.0002).
  - The OIA-traced breathy onset (latent tier) ties the modern onset (−0.0002/+0.0001, vetoed).
- **Gowro, bhatr**: the accented last syllable is **rising** (adyá > ā̌z, ágra > agǎr,
  kárṇa+akka > kanǎh; bhatr ālu > alō̌).
  - Gowro −0.0270/−0.0258; bhatr −0.0123/−0.0176.
  - bhatr: high vowels falling fails the test veto (−0.0005/+0.0080).
- **Bshk**: the last syllable carries the accent (see above). Gawri's tone is **rising after a
  voiceless aspirate onset** (pʰā́la > pʰā̌l, kʰā́dati > kʰā̌nt, *cʰindati > cʰǐn).
  - Rising is −0.0013/−0.0001. The version adapted to the new placement is −0.0013 on top of the
    placement change.
  - Low instead of rising: −0.0005/+0.0010, vetoed.
  - Low after a lost voiced aspirate (latent tracer): +0.0006 train, rejected.
  - Low after h: 0.0000/+0.0003, rejected.
- **Tir**: **low tone after an OIA voiced aspirate** (ḍʰāla > dàl, bʰū́mi > bʰùm), −0.0019/0.0000.
  - The aspirates are lost in the chain, so the vowel is marked on the latent tier early and the
    mark is made visible just before placement: `V > $1-stress` (clears the unused Vedic latent),
    `V > $1+stress / ASP _`, then `Stress > $1+acc`, `V > $1-stress`, placement, and `{á …} > {à …} / _ C* #`.
  - All voiced onsets instead: −0.0010.
- **Pas**: **rising after an OIA aspirate/h** (Lehr gʰāsá > gā̌ns, hásta > ā̌st, kʰáṭvā > kā̌ṭ),
  built the same way. −0.0006/−0.0833 (the test tone set is 3 Lehr pairs). Voiced aspirates and h
  only: −0.0004. Without stripping CDIAL/Strand the rule loses (+0.0039): those sources mark stress
  only.

### Segmental: Mai monosyllable lengthening

**`i > ī / # C* _ C #`** at the end of the chain, after apocope and the round-2 high-vowel shortening:
- It lengthens i in a closed monosyllable of the modern word: íṣṭakā > ī̂ṭʰ, kiri > kī̂r,
  kīla > kī̂l, ŕ̩kṣa > ī̂ʦ̣ʰ.
- Zoller writes 31 long vs 12 short.
- Δ ext −0.0038/−0.0050, verb −0.0015/−0.0076, ext_acc −0.0041/−0.0019, plain +0.0009.
  53 better / 24 worse. ACCEPT.

Variants and other lects:
- u (41 long / 41 short) and a (48/82) are not lengthened. With u included: ext −0.0053/−0.0076, but
  plain +0.0079 (ACCEPT-IF-JUSTIFIED, not taken).
- Including open syllables and clusters: ext −0.0039/−0.0047 (weaker).
- Gowro fails (plain +0.0046, verb test +0.025). bhatr fails (ext +0.0004).
- Zoller's u → ō (24 of ~300) and a/ā length remain lexical.
- bhatr: keeping tone-bearing short a as a rather than Zoller's æ is REJECT (plain +0.0031,
  verb +0.042).

### Consistency: Mai -akka > -ah

Mai's local `kʰ > h / V C* a _ #`, which ran after final-stop aspiration, is replaced by the shared
`use south-final-aspiration.k.vca` (`k > h / V C* a _ #`) of Gowro and bhatr, placed before the
aspiration.
- Every metric on both splits is identical (gates.py says REJECT only because nothing improves).
- Installed as a consistency change, so centralize.py can merge the three lects.
- Revert if score-neutral edits are unwanted.

### Vowel quality (scoring change)

- **Dm: CDIAL notation `ā > o` → `ā > ɔ`.** Morgenstierne writes Dameli /ā/ as ɔ (adyá > źɔ,
  aṣṭā́daśa > aṣṭɔś, *āḍu > ɔru). The old `o` was a fold for the merged scoring.
  - ext −0.0100/−0.0089; plain −0.0098/−0.0088; verb −0.0073/0.
  - ACCEPT. Round 6 had rejected `ā > ɔ` at +0.0100 under merged scoring.
- **Checked and left alone.** In each case the letter is scattered, or the rules fail:
  - Pas CDIAL ə (20 of ~1000 predicted a). `i > ə / # v _` (vənd, vəl) is 10 better / 17 worse, REJECT.
  - Pas æ; Shum CDIAL æ/ə:
    - `a > æ` in the first syllable: +0.017 ext.
    - `a > æ` before a final closed syllable: +0.004.
    - final `a > ə`: train −0.0017 but test +0.0016.
  - Tir ə, Wot ə, Tor ə/ʌ; Bshk CDIAL ʌ/æ/ə and RDH ɛ/ɔ; Gowro Zoller ɛ (7 scattered forms);
    Gaw CDIAL ɔ (5); Dm CDIAL æ/ε.
- **Dm: `%gold-drop circumflex`** (a scoring declaration, not gated).
  - CDIAL's Dameli â is a vowel letter: 20 forms, usually with a stress acute on the same vowel
    (anấr, tấlu, nấya). It was being read as a falling tone.
  - After the change: train ext_acc 0.4464 → 0.4441 and ext_tone 0.4492 → 0.4444. The test set loses
    its only tone-scored forms (ext_tone 0.4806 → none), and test ext_acc changes 0.4757 → 0.4867
    because circumflex-only forms (pã̂c) are no longer accent-scored.

### Before → after (train/test; before = pre-round-7 file without `%gold-drop tone`, current engine)

| lect | ext | tone | ext_tone | ext_acc | what changed |
|---|---|---|---|---|---|
| Mai | 0.4277/0.4464 → **0.4239/0.4414** | 0.5899/0.6077 → **0.5702/0.5866** | 0.4811/0.5055 → **0.4502/0.4782** | 0.4420/0.4640 → 0.4379/0.4620 | 3 tone rules; i > ī in closed monosyllables |
| Gowro | 0.4393/0.5074 = | 0.5779/0.6558 → 0.5550/0.6355 | 0.4943/0.5700 → **0.4673/0.5442** | = | rising |
| bhatr | 0.4675/0.5258 = | 0.5717/0.6801 → 0.5603/0.6653 | 0.5204/0.5798 → **0.5081/0.5622** | = | rising; Strand strip |
| Bshk | 0.4180/0.4596 = | 0.5206/0.5883 → 0.5173/0.5846 | 0.4674/0.5404 → 0.4632/0.5338 | 0.4522/0.5213 → 0.4487/0.5136 | last-σ accent; rising after aspirate |
| Pas | 0.4366/0.4343 = | 0.5817/0.7500 → 0.5811/0.6667 | 0.4501/0.7500 → 0.4495/0.6667 | = | rising after lost aspirate |
| Tir | 0.4106/0.4349 = | 0.5549/0.6659 → 0.5508/0.6659 | 0.4751/0.6128 → 0.4731/0.6128 | = | low after lost voiced aspirate |
| Dm | 0.4406/0.4066 → **0.4306/0.3977** | 0.5729/0.6014 → 0.5683/0.5986 | 0.4547/0.4863 → 0.4492/0.4806 | 0.4518/0.4814 → 0.4464/0.4757 | CDIAL ā > ɔ (gated); then `%gold-drop circumflex` (declaration: ext_acc 0.4441/0.4867, ext_tone 0.4444/–) |

- Unchanged: Tor, Gaw, Shum, Wot, Chil. Placement, vowel-quality and tone probes were all rejected
  or vacuous there (Gaw, Shum and Wot are not tone-scored).
- Every changed lect passes `gates.py` against its pre-round-7 file (`r7/before-after-drop.txt`,
  `r7/gates-all.txt`):
  - ACCEPT (prosody-only): Gowro, bhatr, Bshk, Pas, Tir;
  - ACCEPT: Mai (with the i-lengthening; `r7/gates-Mai-final.txt`) and Dm (the ā > ɔ notation,
    gated before the circumflex declaration).

### Not done / open
- **Strand's ǐ in bhatr is a glide letter**, not a rising tone (gʰǐū, bʰǐõ, hǐū; 4 train forms).
  It makes Strand tone-scored in bhatr. `%gold-drop caron` cannot be used, because Zoller's caron in
  the same lect is a tone. This would need a per-source gold-drop in the engine.
- **Zoller's -ǎy/-âh/-ī̂ suffix tones** follow from the ending chosen by the ext menu. Tone type on
  non-final syllables, Zoller's acute (≈ 6 %), and Gawri's low tone (14 %) are not predicted.
- **Pashai test tone scores rest on 3 Lehr pairs.** Tirahi's rest on 5 graves (train), and its test
  has none after an aspirate.
- **Segmental leftovers stay lexical**, as in round 6: Zoller's length and u/ō, Pashai a/ā/ō/ə,
  Perder's inflection.

### Round 7, second part (after 09:30)

Engine changes since the first part: per-source `%gold-drop <mark> @SOURCE`; CDIAL ɔ̌ is no longer a
tone; canon splits ãĩ ãũ āi āu. All lects were re-baselined before any candidate was gated.

**1. bhatr, Strand's glide ǐ.**
- Added `%gold-drop caron @strand` (a scoring declaration): gʰr̩tá > gʰǐū, bʰaginī > bʰǐõ, hŕ̩daya > hǐū.
- Strand is now no longer tone-scored, so the round-7 `%source strand` strip block was redundant
  (removing it leaves every metric identical) and has been removed.
- Against the pre-round-7 file with the same declaration, bhatr's rising default now gains ext_tone
  0.5666/0.5745 → 0.5413/0.5416 (ACCEPT, prosody-only).
- Vedic final vowel → falling was re-tested on the new gold: −0.0081 on train, +0.0010 on test.
  It is vetoed again.

**3. Bshk, Gawri low tone (last-σ placement).**
- Low is the majority in only one cell, an OIA voiced aspirate or h before the tone vowel: low 42 %,
  high 39 % (n = 31). In every other cell high dominates. Voiced onsets are low 22 % (n = 77); a long
  open monosyllable is 35 % low / 35 % high (n = 23).
- Low tone after a lost voiced aspirate/h (latent tracer, with or without h, with or without the
  CDIAL strip) is +0.0009 to +0.0012 on train, or 0.0000 with the strip. REJECT.
- **Gawri's low tone is not predictable from OIA or from the modern form.**

**6. Legacy `@morph` citation rules.** Leave-one-out, then jointly (`r7/morph.py`,
`r7/morph-{1,2}.txt`). A rule is removed when ext, verb, ext_acc and ext_tone are unchanged or better
on both splits ("neutral removal"; the plain delta is the bare-headword artifact).

| lect | removed (plain Δ train/test) | kept (why) |
|---|---|---|
| Pas | causative `if:verb a y a t {i ē} > a` + -ati `a t {i ē} > a` (+0.0041/+0.0011) | — |
| Dm | -ati (0/0) | causative: ext +0.0025 |
| Tor | -ati (+0.0016/+0.0008) | -yati: ext +0.0003 |
| Gaw | -ati (+0.0015/−0.0014) | — |
| Gowro | both -ō̃ infinitive rules (+0.0076/+0.0070; ext_acc −0.0001) | — |
| Shum | -ati (+0.0008/0) | — |
| Wot | causative + -ati (+0.0013/0) | — |
| Tir | `if:verb` -ati (+0.0068/+0.0102) | causative: ext +0.0008 |
| Mai | — | -ati > -aṽ: ext +0.0025/+0.0016, ext_acc +0.0035 |
| bhatr | — | -ati: ext +0.0031 |
| Chil | — | -ati: ext +0.0035 (test −0.0069) |

The kept rules still help the extension-aware score through nouns cited in -ati/-aya-.

**2, 4. Near-misses and consistency** (`near6.py`; `r7/batch.py` tests one rule in many lects).

- **`{k ṭ t p} > {g ḍ d b} / {ŋ ṇ n m} _`** (Mai's round-6 nasal voicing) is now also in **Gowro,
  bhatr, Tor, Bshk, Wot and Tir**. Each passed gates.py: 1–6 words better, none worse.
  - Examples: káŋkata > kǎŋg (bhatr), kyæŋg (Tor), kēŋg (Bshk); *riŋk > riŋg; *kaṇṭa > mar-kaṇḍḗ (Tir).
  - Rejected in Pas (ext_acc test +0.0011), Dm (ext test +0.0032) and Gaw (ext +0.0013).
  - Vacuous in Shum and Chil.
- **Gaw `use prothesis.ee`** (y- before initial ē, as Dm and Shum): *ēkka > yak, *ēkkala > yēkulɔ́,
  ēḍa > yēṛa.
  - ext −0.0010/−0.0035; 6 better / 3 worse. ACCEPT.
  - Rejected in Tor (+0.0019), Tir (+0.0015), Chil (+0.0034) and Pas (ext_acc test +0.0009).
    Vacuous in Wot and Bshk.
- **Wot `ś > ∅ / V _ C`** (late): preconsonantal ś from ṣ is lost (gōṣṭʰá > goṭ, dr̩ṣṭá > diṭ,
  śúṣka > śuk, piṣṭá > pīṭ).
  - ext −0.0041/−0.0066; 6 better / 0 worse. ACCEPT.
  - The narrower `/ V _ {ṭ k}` scored −0.0033. The version with all sibilants before voiceless stops
    was no better.
  - In other lects it is vacuous (Gaw, Shum, Tir, Chil) or fails (Dm +0.0012, Pas, Bshk test).
    Tor passes on train at −0.0001, but that is the single word *paśca, so it was not installed.
- **Rejected near-miss candidates:**
  - Mai: second-syllable a > u after u before a final liquid (kuṭûl, mûsul). −0.0007 on train, but
    test +0.0004.
  - Mai: i > ē before r. +0.0016.
  - Mai: ndʰ > nd (bandâh). +0.0011.
  - Mai: rising before ŋ (above).
  - Gowro: mb/mr order, i.e. mr > mb after the mb > m rule, or dropping mb > m. +0.0005/+0.0012.
  - Gowro and bhatr: i > ī in closed monosyllables (above).
  - Wot: final ə for Morgenstierne (CDIAL). +0.10.
- **What is left is lexical:**
  - Zoller's -ǎy/-ai/-ah after -akka, -ī/-ū gender endings, h after sonorant onsets (zhǎr, lhǎṭʰ);
  - Tor u/ū; Shum æā contraction (round 6); Gaw a/ā.

**Round-7 totals after the second part.** Train/test. "Before" is the pre-round-7 file under the
current engine, without `%gold-drop tone`; for bhatr it also has `%gold-drop caron @strand`, so that
both sides score the same gold.

| lect | ext | plain | ext_acc | ext_tone |
|---|---|---|---|---|
| Mai | 0.4277/0.4464 → **0.4239/0.4414** | 0.5163/0.5383 → 0.5172/0.5380 | 0.4420/0.4640 → 0.4379/0.4620 | 0.4811/0.5055 → **0.4502/0.4782** |
| Gowro | 0.4393/0.5074 → 0.4382/0.5074 | 0.5067/0.5714 → 0.5133/0.5784 (@morph removal) | 0.4579/0.5348 → 0.4559/0.5348 | 0.4943/0.5700 → **0.4654/0.5442** |
| bhatr | 0.4675/0.5258 → 0.4660/0.5258 | 0.5039/0.5787 → 0.5024/0.5787 | 0.4999/0.5615 → 0.4981/0.5615 | 0.5666/0.5745 → **0.5377/0.5416** |
| Bshk | 0.4180/0.4596 → 0.4174/0.4584 | 0.4582/0.4892 → 0.4576/0.4881 | 0.4522/0.5213 → 0.4484/0.5136 | 0.4674/0.5404 → 0.4629/0.5338 |
| Pas | 0.4366/0.4343 = | 0.5169/0.5412 → 0.5209/0.5423 (@morph removal) | = | 0.4501/0.7500 → 0.4495/0.6667 |
| Dm | 0.4406/0.4066 → **0.4306/0.3977** | 0.5125/0.4787 → 0.5027/0.4698 | 0.4518/0.4814 → 0.4441/0.4867 (declaration) | 0.4547/0.4863 → 0.4444/– (declaration) |
| Tir | 0.4106/0.4349 → 0.4099/0.4349 | 0.4716/0.5073 → 0.4779/0.5174 (@morph removal) | 0.4707/0.6128 → 0.4678/0.6128 | 0.4751/0.6128 → 0.4702/0.6128 |
| Tor | 0.4533/0.4854 → 0.4516/0.4835 | 0.4870/0.5358 → 0.4869/0.5347 | = | = |
| Gaw | 0.4292/0.4059 → 0.4282/0.4024 | 0.5152/0.4990 → 0.5158/0.4942 | 0.4622/0.4232 → 0.4605/0.4232 | – |
| Wot | 0.3638/0.3833 → **0.3587/0.3768** | 0.4090/0.4366 → 0.4053/0.4343 | 0.4263/0.4159 → 0.4225/0.4159 | – |
| Shum | = | 0.5197/0.6019 → 0.5205/0.6019 (@morph removal) | = | – |
| Chil | unchanged | | | |

**Rules that fire rarely** (`evaluate.py --ext-loo`, `r7/loo/`). In Tor, Gaw, Shum, Wot and Chil every
segmental rule has a positive ext LOO except two:
- Gaw pr/br > pl/bl: −0.0015. Dropping it again fails the test (ext +0.0082, verb +0.0333,
  ext_acc +0.0068), so it stays.
- Chil r-assimilation: +0.0002.

The placement rules show 0 by design.
- Also rejected: Chil, RDH short vowels in a final closed syllable of polysyllables (+0.0159 plain).
- Mai's `use nasal-voicing.nt` (ṇṭ > ṇḍ) remains redundant with the general nasal-voicing rule.

**Final whole-file verdicts** against the pre-round-7 files (`r7/gates-final.txt`):
- ACCEPT: Mai, bhatr, Bshk, Tor, Gaw, Wot.
- ACCEPT-IF-JUSTIFIED: Gowro and Tir. The plain loss comes from the neutral @morph removals, the
  bare-headword artifact; ext, ext_acc and ext_tone are equal or better.
- "REJECT", as expected:
  - **Pas**: the tone rule plus a neutral @morph removal. ext is unchanged, and the plain change
    stops gates.py treating the edit as prosody-only. Each part was gated separately above.
  - **Dm**: the `%gold-drop circumflex` declaration changes the gold set. `ā > ɔ` was gated on its
    own (ACCEPT).
  - **Shum**: only a neutral @morph removal (plain +0.0008, all else equal).

**Kept @morph rules re-gated after the 10:35 verb-flag change** (leave-one-out against the new
baseline, `r7/morph3-*.txt`; Δ train/test):
- **Removed, neutral on every gated score:**
  - Tor causative/-yati `y a t {i ē} > a` (ext 0/0; plain +0.0066/−0.0007);
  - Tir causative `a y a t {i ē} > a` (ext 0/0; plain −0.0003/0).
  - Neither lect now has an @morph chain rule.
- **Still kept (removing them makes ext worse on train):**
  - Mai -ati > -aṽ: ext +0.0009/+0.0007, ext_acc +0.0014.
  - bhatr -ati: ext +0.0028, ext_acc +0.0049.
  - Chil -ati: ext +0.0029 on train (−0.0069 on test).
  - Dm causative: ext +0.0007.
