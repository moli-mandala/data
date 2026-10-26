# dardic-north: OIA → Kashmiric, Shinaic and Chitrali Dardic

Lects: K (Kashmiri), pog (Poguli), dod (Dodi), Sh (Shina), Ush (Ushojo), Phal (Palula),
Sv (Sauji), Kalk (Kalkoti), Kund (Kundal Shahi), Kho (Khowar), Kal (Kalasha).
- Rule files: `sound_changes/rules/OIA-<lect>.sc`.
- Frontiers: `sound_changes/results/OIA-<lect>.hand.frontier.csv`.
- Scratch tools in `tmp/sound_changes/dardic-north/`:
  - `src.py`: per-source NED
  - `cross.py`: tries one rule in every lect
  - `chrono.py`: swap tests
  - `isogloss.py`: presence/absence matrix
  - `tailsweep.py`: per-source %source tests
  - `table.py`: builds the table below

## 1. Results

| lect | etyma | rules (+tail) | train NED null → final | test NED null → final | test exact | test NED w/o %source | auto test NED @≈n | sources in %source blocks |
|---|---|---|---|---|---|---|---|---|
| K | 2363 | 35 (+0) | 0.664 → 0.399 | 0.670 → 0.402 | 16.3% | 0.402 | 0.507 @16 | — |
| pog | 112 | 18 (+0) | 0.695 → 0.434 | 0.719 → 0.506 | 9.7% | 0.506 | 0.694 @11 | — |
| dod | 130 | 17 (+0) | 0.627 → 0.414 | 0.643 → 0.486 | 3.8% | 0.486 | 0.547 @10 | — |
| Sh | 1350 | 32 (+11) | 0.699 → 0.525 | 0.698 → 0.560 | 6.5% | 0.583 | 0.627 @15 | CDIAL, backstrom1992, buddruss-shina1996, degener-shina2008, grierson-lsi1928, liljegren-hindukush, rajapurohit2012, schmidt |
| Ush | 155 | 26 (+2) | 0.671 → 0.447 | 0.657 → 0.483 | 13.3% | 0.543 | 0.681 @19 | liljegren-hindukush, rensch-decker-hallberg1992 |
| Phal | 827 | 34 (+9) | 0.672 → 0.509 | 0.663 → 0.539 | 6.9% | 0.556 | 0.604 @12 | CDIAL, decker1992, kalkoti, liljegren, liljegren-hindukush, strand, zoller2023 |
| Sv | 330 | 28 (+8) | 0.680 → 0.499 | 0.656 → 0.515 | 8.8% | 0.550 | 0.597 @22 | CDIAL, decker1992, knobloch2020sauji, liljegren-hindukush |
| Kalk | 270 | 26 (+6) | 0.704 → 0.483 | 0.712 → 0.581 | 4.5% | 0.599 | 0.707 @26 | hultman2023kalkoti, kalkoti, liljegren-hindukush, rensch-decker-hallberg1992 |
| Kund | 143 | 23 (+0) | 0.673 → 0.443 | 0.638 → 0.479 | 10.0% | 0.479 | 0.559 @15 | — |
| Kho | 1147 | 33 (+8) | 0.716 → 0.434 | 0.726 → 0.465 | 12.7% | 0.479 | 0.558 @20 | bashir2023, decker1992, strand |
| Kal | 850 | 29 (+4) | 0.687 → 0.501 | 0.676 → 0.513 | 6.0% | 0.523 | 0.580 @17 | CDIAL, decker1992, trail-cooper1999, zoller2023 |

"rules (+tail)" is the number of chain rules, plus the rules inside `%source` blocks (which the
engine does not count). "test NED w/o %source" scores the same chain with the source blocks
switched off, for a like-for-like comparison with the auto baseline, which has none. "auto" is
the greedy baseline's test NED at the closest prefix length. Test was never used for a decision.

Frontier summary (greedy backward elimination on train; test NED at a given number of rules;
hand frontiers include the %source blocks, auto has none):

| lect | hand test NED @5 / @10 / full | auto test NED @5 / @10 / full | hand rules for 80% of its test gain |
|---|---|---|---|
| K | 0.535 / 0.485 / 0.402 (35) | 0.573 / 0.527 / 0.507 (16) | 13 |
| pog | 0.582 / 0.538 / 0.506 (18) | 0.681 / 0.694 / 0.694 (11) | 8 |
| dod | 0.515 / 0.501 / 0.486 (17) | 0.522 / 0.547 / 0.547 (10) | 4 |
| Sh | 0.606 / 0.587 / 0.560 (32) | 0.643 / 0.624 / 0.627 (15) | 10 |
| Ush | 0.513 / 0.501 / 0.483 (26) | 0.649 / 0.635 / 0.681 (19) | 5 |
| Phal | 0.582 / 0.561 / 0.539 (34) | 0.612 / 0.606 / 0.604 (12) | 8 |
| Sv | 0.580 / 0.553 / 0.515 (28) | 0.613 / 0.594 / 0.597 (22) | 15 |
| Kalk | 0.652 / 0.608 / 0.581 (26) | 0.707 / 0.706 / 0.707 (26) | 11 |
| Kund | 0.536 / 0.523 / 0.479 (23) | 0.542 / 0.570 / 0.559 (15) | 8 |
| Kho | 0.586 / 0.511 / 0.465 (33) | 0.625 / 0.581 / 0.558 (20) | 10 |
| Kal | 0.565 / 0.546 / 0.513 (29) | 0.617 / 0.598 / 0.580 (17) | 11 |

**No source is excluded.** Differently transcribed sources get `%source` notation/citation
blocks instead:
- **Survey transcriptions write length on almost every vowel.** The blocks map short vowels to
  long:
  - Decker 1992 (Kho, Kal: {a i u e o}; Phal, Sv: {a i u})
  - Rensch, Decker & Hallberg 1992 (Kalk {a i u}; Ush {i u})
  - Backstrom 1992 (Sh {a i u})
- **Tone marks** (circumflex/caron), which `gold_tokens` does not strip: Schmidt (Sh) and
  Liljegren (Phal) put tone on long vowels. A block maps every long vowel to the circumflex
  (falling) form:
  - Schmidt: 0.583 → 0.534
  - Liljegren: 0.577 → 0.545

  Carons in Strand/Hultman/Bashir did not respond to any simple rule and are left.
- **Digraph aspirates:** Strand (Kho, Phal) and Decker (Kho) write ʦ̣h, ch, ʦh.
- **Liljegren's Hindukush wordlist** writes palatal c/j as ʦ̣/ʣ̣ in every Shinaic lect (Sh,
  Ush, Sv, Kalk). This is a transcription or a genuine c > ʦ̣ among the speakers recorded.
- **Kalkoti æ:** the Kalkoti dictionary / Hultman write æ for a between consonants. The survey
  lists write a.
- **Kalasha ḷ:** Trail & Cooper and Zoller write ḷ (velarised l) where CDIAL and Decker write l.
- **Per-source citation forms of verbs** (`@morph` inside blocks):
  - Sh: CDIAL -oiki̯, Schmidt -ō̌n(u), Rajapurohit -ṓno, Liljegren -ōn, and stems in
    Degener, Buddruss, Backstrom and LSI.
  - Kho: Strand stems vs -ik.
  - Kal: CDIAL 1sg -im vs Trail & Cooper -ik.
  - Sv: CDIAL stems, Knobloch -ān-, Decker/Liljegren -ilo.
  - Kalk: Rensch stems, Liljegren -il.

Data problems left in place:
- **CDIAL K.** has 11 forms with raw `<sup>…</sup>` HTML (dat<sup>ə</sup>rⁱvǝtras), which
  tokenize as `< s u p >`. It also uses ǝ (U+01DD), which is not folded like ə.
- **Liljegren's wordlists** use creaky/devoiced diacritics (ḭ̄, l̥).
- **Knobloch's Sauji** forms are inflected (moril-ō). The hyphen is dropped.
- **Backstrom (Sh)** often gives conjugated phrases (tū beī).
- **Many worst misses are compounds or mislinks:**
  - dʰāpayati > Kho hardi deik "remind"
  - nára > Kal birẹ̃-kakavák (= kukkuṭá)
  - vātiŋgaṇa > Sh kīṇo bālūgun
  - numeral compounds

## 2. Group base chain (in order) and deviations

Order is identical across files except where noted (see §3 for the data behind the key
orderings). Every accepted lect-specific specialisation has train LOO > 0 in the final files.

| # | stage / tag | typical rule | lects (deviations) |
|---|---|---|---|
| 0 | pre-citation vowel rules | Kho `{a á} > o / #(C)(C)_C(C){a á u ú ō ṓ}` (`@a-rounding @umlaut`) and i > e before i; Kal a > o / _NC a; Phal `ā́ > ō / _CV` (`@vowel-shift @stress`), `@x-…-aspiration-shift`, `@x-…-monosyllable-lengthening` | Kho, Phal |
| 1 | `@morph` | -ati/-atē → citation form; per-source variants in `%source` blocks | K -un, pog -nu, dod -ṇō, Sh -oik, Ush -ōn, Sv -il, Kalk -ūn, Kund ∅, Kho -ik (causative -ik), Kal -ik; none in Phal. Sh and Phal also -aka > -o |
| 2 | `@acc-loss`, `@ai-au`, `@final-cons-loss` (ḣ), `@anusvara-vowel` (ṁ > n in K/Kho, ∅ elsewhere), `@n-dental` in some | | all |
| 3 | `@x-dardic-north-length-loss` | {ā ī ū ē ō} > short | Kho, Kal, Sv, Kund. In Kund it precedes accent loss, so Vedic-accented long vowels stay long |
| 4 | `@r-vocalic` / `@r-vocalic-ri` | r̩ > i; r̩ > ri after C (Kal, Phal, Kalk); r̩ > u after labials (Kho) | all |
| 5 | `@vowel-shift` | K: i u > e o / _CC, i > e / #(C)(C)_Ca, ō > ū, ē > i / _C(C)V; Phal, Ush: ē > ī | K, Phal, Ush |
| 6 | `@ks` | kṣ > ʦ̣ʰ | all. ʦ̣ in Sv and Kalk; ṣ in Ush; cʰ in K and dod, placed after the affrication |
| 7 | `@x-dardic-north-ts-affricate` | ts > ʦ (K ʦʰ) | all but dod |
| 8 | tr | `@x-…-tr-affricate` tr dr > ʦ̣ ʣ̣ (Sh, Ush); `@x-…-tr-lateral` > ɬ l (Sv) | others keep tr |
| 9 | `@x-dardic-north-sv-sp` | śv sv > śp sp | Kho, Sh (medial), Ush |
| 10 | initial s + C | `@s-stop-aspirate` (Sh, Ush, Phal, Sv initial; K and dod everywhere); pog `Sib Stop > h Stop`; `@prothesis` i- (Kho, Kal); Kalk s > ∅ | |
| 11 | `@nasal-voicing` then `@x-dardic-north-nd-n` | N + {k c ṭ t p} > N + voiced; then N + voiced dental/retroflex > N | voicing: K Kho Kal Phal Sv Kalk Kund Ush; nd > n: Kho, Kalk. Sh: all NC > N, ŋg > g late |
| 12 | `@deaspiration` | bʰ dʰ gʰ jʰ ḍʰ > b d g j ḍ | all but dod. Kal and Phal only non-initially |
| 13 | `@int-p-v`, `@flap` | p > v; ṭ ḍ > ṛ (Sh Phal Sv Kund dod), r (K pog Kalk), ḷ (Kho). Kho also `@x-…-t-r`, `@int-spirant` k g > ɣ | |
| 14 | `@int-stop-loss` (+`@v-loss @h-loss`) | {k g t d c j y (v h)} > ∅ / V_V | all. **k survives in Kal, Phal, Sv and Kalk**: the chain does better without k in the loss set, but survival is only partial. A crude count of first-syllable single -k- with any k/g/ɣ in the reflex gives Kal 58%, Phal 39%, Kalk 36%, Sv 22%, against Sh 28% and K 39%; Kho has k > ɣ (69%) |
| 15 | `@hiatus` | {a i} > ∅ / _V (Sh, Ush and Sv keep word-final VV sequences) | all but Kho, Kal |
| 16 | `@degem-plain @cluster-simplify` | Stop Stop > $2, C $1 > $1 | all |
| 17 | `@x-dardic-north-s-stop-loss` | {s ṣ} > ∅ / _Stop | Sh, Ush, Kund |
| 18 | `@x-dardic-north-r-retroflex`, `@metathesis` | r {t d} > {ṭ ḍ}; Stop V r C > Stop r V C | r-retroflex: K Phal Kalk Kund; metathesis: Phal, Kalk |
| 19 | `@cluster-simplify @assim-r`; `@x-dardic-north-cr-loss` | r > ∅ / V_C; C r > C | r-loss: all but Kho, dod. **Cr > C: Sh, Ush, Sv only** |
| 20 | `@palatal-Cy`, C{y v} > C | {t d dʰ} y > {c j j} (K Sh Phal Kal Kund); {d dʰ} y > ź (Sv Kalk Ush) | all |
| 21 | Kashmiric block | `@affrication` c cʰ j > ʦ ʦʰ z; kṣ > cʰ; ś ṣ > h except before front vowels (`@s-h`); ṇ ŋ > n | K; pog and dod partly |
| 22 | `@v-b`, `@y-j` | v- > b- | all but K and pog |
| 23 | `@final-vowel-loss` | V > ∅ / V(C)C_# (not in monosyllables) | all. **-i/-u survive in Kho, Kal, Sv and Ush** (only a ā ē ō are lost); dod: after a single C only, then -a > -ō; pog keeps the -u of C-nu; Sh and Phal spare the -o of -aka |
| 24 | late `@hiatus`, `@aya-e` | a > ∅ / V_; Kho a > e / C_i; Kal a > ∅ / C_i | |
| 25 | K only | `@prothesis` v- before u o; `@umlaut` a > u in a final non-initial syllable | K |
| — | late lect-specific | word-final ʦ̣ʰ > ʦ̣ (`@deaspiration`) in Kho, Kal, Phal; Sh s > z / V_V; Kho y- j- > ź-; Kho/Kal s + aspirate > aspirate before the i-prothesis; Phal d g b > breathy before r (dr > dʰr); Kund ṇ > n, final ś ṣ s > y y z; Ush a > ō / _n (after r + C loss); pog h- > ∅ (including h < s before stops), ṁ > ∅; Kal p > v after v-loss | |

## 3. Relative chronology

Swap tests (`chrono.py`, train NED increase when the rule is moved to the other side;
positive = the file order is better):

| lect | current order | Δ train NED if reversed | rule A | rule B |
|---|---|---|---|---|
| K | B before A | +0.0051 | `r > ∅ / V _ C` | intervocalic stop loss |
| Sh | B before A | +0.0059 | `r > ∅ / V _ C` | intervocalic stop loss |
| Phal | B before A | +0.0058 | `r > ∅ / V _ C` | intervocalic stop loss |
| Kal | B before A | +0.0026 | `r > ∅ / V _ C` | intervocalic stop loss |
| K | A before B | +0.0005 | deaspiration | intervocalic stop loss |
| Sh | A before B | +0.0011 | deaspiration | intervocalic stop loss |
| Kho | A before B | +0.0038 | deaspiration | intervocalic stop loss |
| K | B before A | +0.0132 | `Stop Stop > $2` | intervocalic stop loss |
| Sh | B before A | +0.0119 | `Stop Stop > $2` | intervocalic stop loss |
| Kho | B before A | +0.0009 | `Stop Stop > $2` | intervocalic stop loss |
| Phal | B before A | +0.0133 | `Stop Stop > $2` | intervocalic stop loss |
| Kal | B before A | +0.0057 | `Stop Stop > $2` | intervocalic stop loss |
| K | B before A | +0.0075 | `C {y v} > $1` | intervocalic stop loss |
| Sh | B before A | +0.0051 | `C {y v} > $1` | intervocalic stop loss |
| Sh | B before A | +0.0039 | `{s ṣ} > ∅ / _ Stop` | intervocalic stop loss |
| Ush | B before A | +0.0062 | `{s ṣ} > ∅ / _ Stop` | intervocalic stop loss |
| Kund | B before A | +0.0071 | `{s ṣ} > ∅ / _ Stop` | intervocalic stop loss |
| Sh | B before A | +0.0009 | `C r > $1` | intervocalic stop loss |
| Sh | B before A | +0.0058 | `ŋ {k g gʰ} > g` | intervocalic stop loss |
| Phal | B before A | +0.0009 | r-metathesis | intervocalic stop loss |
| Kalk | B before A | +0.0030 | r-metathesis | intervocalic stop loss |
| Kho | B before A | +0.0004 | `{n ṇ} {d ḍ dʰ} > n` | post-nasal voicing |
| Kalk | B before A | +0.0021 | `{n ṇ} {d ḍ dʰ ḍʰ} > $1` | post-nasal voicing |
| K | A before B | +0.0062 | `{c cʰ j} > {ʦ ʦʰ z}` | `k ṣ > cʰ` |
| K | B before A | +0.0134 | `{ś ṣ} > h` | `Sib Stop > $2+asp` |
| K | A before B | +0.0040 | `{i u} > {e o} / _ C C` | `Stop Stop > $2` |
| Kho | A before B | +0.0248 | Kho a-harmony | length loss |
| Kho | A before B | +0.0061 | Kho a-harmony | -ati citation @morph |
| Kal | B before A | +0.0017 | `p > v / V _ V` | intervocalic stop loss |
| Ush | B before A | +0.0041 | `{a á ā ā́} > ō / _ n` | `r > ∅ / V _ C` |
| Phal | A before B | +0.0014 | `{g d b j ḍ} > $1+asp` | deaspiration |
| Kund | A before B | +0.0035 | length loss | accent loss |
| pog | B before A | +0.0028 | `h > ∅ / # _` | sibilant + stop > h + stop |
| Kho | A before B | +0.0000 | `t > r / V _ V` | intervocalic stop loss |
| K | A before B | +0.0000 | post-nasal voicing | deaspiration |

Robustness check (`chrono_boot.py`). Nine key swaps were recomputed on five random halves of
the train etyma, and every one stayed positive in all five halves:
- stop + stop after lenition: K, Sh, Phal (+0.009 to +0.016)
- r + C after lenition: K, Sh, Kal (+0.0007 to +0.007)
- Kho deaspiration before loss
- K affrication before kṣ
- Kho harmony before the citation rule (+0.002 to +0.009)

What this implies:
- **Clusters shielded stops from lenition and were reduced later.** This is the strongest and
  most general result. Moving the reduction before intervocalic lenition costs:
  - stop + stop / geminates: +0.011 to +0.013 in K, Sh and Phal
  - C + y/v: +0.005 to +0.007
  - r + C: +0.0025 to +0.005 in K, Sh, Phal and Kal
  - s + stop: +0.004 to +0.007 in Sh, Ush and Kund

  Dardic thus shares the MIA ordering: single stops lenite first, clusters reduce afterwards.
  But it reduces without gemination or compensatory lengthening (karpaṭa > K kapur, not
  *kavur; hásta > Ush hāt, not *hā).

  The "no compensatory lengthening" part was tested directly. Adding MIA-style
  degemination-with-lengthening (V C C > V̄ C) raises train NED in every lect: K +0.008,
  Phal +0.010, Kho +0.008, Kal +0.007, Sh +0.002 to +0.006.
- **Palula and Kalkoti r-metathesis also follows lenition** (durgá > drūg keeps the g that the
  rg cluster protected). So does Shina's ŋg > g (+0.005).
- **Kalasha p > v follows the loss of inherited intervocalic v** (+0.0018). The new v
  survives.
- **Ushojo a > ō before n follows r + C loss** (+0.0041): kárṇa > kan > kōn.
- **Kundal Shahi shortens length only in unaccented syllables** (+0.0035 if after accent loss).
  With the (sparse) Vedic accent this is weak evidence for stress-conditioned length.
- **Deaspiration precedes intervocalic loss** (Kho +0.0036, Sh +0.0014). Medial dʰ gʰ were
  lost like d g (gōdʰū́ma > Sh gūm).
- **Post-nasal voicing precedes the loss of the stop in nd > n** (Kalkoti +0.0019, Kho +0.0004).
  The chain was nt > nd > n, not nt > n directly.
- **Kashmiri:**
  - c > ʦ precedes kṣ > cʰ (+0.0062). The new cʰ from kṣ escaped the affrication, hence the
    contrast K. ʦʰ (< cʰ) vs cʰ (< kṣ).
  - ś, ṣ > h follows s + stop > aspirated stop (+0.0135). Otherwise gōṣṭʰa > *guhṭʰ.
  - High-vowel lowering is conditioned by the *original* closed syllable (+0.0040 if moved
    after degemination).
- **Khowar a-harmony must see the OIA verb ending** (+0.0055 if applied after the -ik citation
  rule): rákṣati > roʦ̣ʰík, kásati > kosík.
- **Palula aspiration shift precedes medial deaspiration** (+0.0011): gōdʰū́ma > gʰōm.
- **Caveat for cross-file order comparisons.** `@cluster-simplify` is a catch-all tag here.
  It covers initial sC, NC > N, Stop + Stop, C + y/v and C + r, which sit at different points
  of the chain. Compare the specific rules or the `@x-dardic-north-*` co-tags instead. The
  Kashmiric files put kṣ > cʰ and ts > ʦʰ *after* the affrication c > ʦ (data-driven, see
  above), so `@ks` comes late there and early elsewhere.
- **Unordered pairs** (disjoint inputs), which the synthesis should not read as chronology: Kho
  t > r vs stop loss, Kalk/Sh tr rule vs initial sC, and K nasal voicing vs deaspiration.

## 4. Shared vs lect-specific; sub-grouping

Pan-group changes:
- accent loss, ai/au > e/o, r̩ > i/ri
- ts > ʦ; kṣ > a retroflex affricate (Kashmiric: cʰ)
- **cluster reduction without gemination, after lenition**
- intervocalic loss of stops (incl. v, h)
- C + y/v simplification
- final-vowel loss sparing monosyllables. This helped in 9/9 lects where tested (Ush −0.015,
  Sv −0.006); a monosyllable keeps its vowel (tráyaḣ > ʦ̣ē).

Isogloss matrix (`isogloss.py`). Each cell reports one of two things. **P** = the rule is in
that lect's chain, and the cell gives its LOO (NED increase when removed). **A** = the rule is
absent, and the cell gives the NED change if it is inserted (negative would mean the lect
wants it).

```
change                   K       pog       dod        Sh       Ush      Phal        Sv      Kalk      Kund       Kho       Kal
tr>ʦ̣              A+0.010   A+0.009   A+0.005   P+0.003   P+0.013   A+0.014   A+0.000   A+0.021   A+0.018   A+0.008   A+0.014
NC>N               A+0.006   A-0.000   A+0.006   P+0.003   A-0.001   A+0.000   A+0.002   P+0.004   A+0.006   P+0.001   A+0.004
nasal-voicing      P+0.003   A+0.005   A+0.007   A+0.000   P+0.003   P+0.006   P+0.005   P+0.003   P+0.010   P+0.003   P+0.001
sC>C late          A+0.000   A+0.000   A+0.000   P+0.002   P+0.008   A+0.001   A-0.001   A+0.003   P+0.006   A+0.011   A+0.007
rC>C late          P+0.005   P+0.002   A+0.008   P+0.007   P+0.010   P+0.001   P+0.004   P+0.005   P+0.009   A+0.001   P+0.007
deaspiration       P+0.017   P+0.019   A+0.010   P+0.012   P+0.011   P+0.004   P+0.009   P+0.020   P+0.013   P+0.022   P+0.007
int. k loss        P+0.007   P+0.003   P+0.001   P+0.002   P+0.010   A+0.005   A+0.003   A+0.004   P+0.012   A+0.000   A+0.001
v>b                A+0.010   A+0.006   P+0.012   P+0.014   P+0.004   P+0.010   P+0.007   P+0.013   P+0.012   P+0.013   P+0.008
c>ʦ j>z            P+0.021   P+0.013   P+0.004   A+0.014   A+0.009   A+0.023   A+0.015   A+0.021   A+0.011   A+0.009   A+0.013
sv>śp              A+0.001   A+0.002   A+0.000   P+0.001   P+0.002   A+0.000   A+0.000   A+0.000   A+0.004   P+0.003   A-0.000
prothesis i-       A+0.002   A+0.002   A+0.000   A+0.005   A+0.006   A+0.003   A+0.002   A-0.000   A+0.004   P+0.003   P+0.001
length loss        A+0.028   A+0.041   A+0.079   A+0.008   A+0.026   A+0.021   P+0.014   A+0.026   P+0.021   P+0.057   P+0.033
Kho a-harmony      A+0.026   A+0.035   A+0.047   A+0.026   A+0.024   A+0.008   A+0.027   A+0.012   A+0.039   P+0.014   A+0.010
Cr>C late          A+0.005   A+0.001   A+0.003   P+0.002   P+0.002   A+0.011   P+0.002   A+0.007   A+0.002   A+0.009   A+0.012
r+dental>retrofl   P+0.001   A+0.000   A+0.000   A-0.000   A+0.002   P+0.002   A+0.001   P+0.003   A-0.002   A+0.004   A+0.002
final ʦ̣ʰ>ʦ̣       A+0.000   A+0.000   A+0.000   A-0.001   A+0.000   P+0.002   A+0.000   A+0.000   A+0.000   P+0.002   P+0.002
ś ṣ > h            P+0.009   P+0.016   A+0.015   A+0.015   A+0.041   A+0.024   A+0.034   A+0.026   A+0.028   A+0.018   A+0.019
```

Reading:
1. **Kashmiric (K, pog, dod) is marked by an innovation the others reject.** That innovation is
   the affrication c j > ʦ z (P in all three; A +0.009 to +0.022 in all others). Kashmiric also
   has its own treatment of s + stop:
   - K: aspirated stop
   - pog: h + stop
   - dod: aspirated stop

   K and pog keep v- (adding v > b costs +0.010 and +0.006). Every other lect, dod included,
   has v- > b-. Dodi otherwise looks Western-Pahari (voiced aspirates kept, -ṇō, masc. -ō,
   ṭḍ > ṛ). Its Kashmiric features are the affrication and kṣ > cʰ.
2. **Shina proper (Sh, Ush) is defined by a bundle of innovations:**
   - tr > ʦ̣ (P in Sh and Ush; A +0.008 to +0.021 everywhere else)
   - s + stop > stop after lenition (also in Kundal Shahi)
   - C + r > C (after lenition). This is shared with **Sauji** (Sv) and refused by every
     other lect (A +0.004 to +0.012): K, Kho, Kal, Phal and Kalk keep Cr.
   - Sh: NC > N (Ush has nd ~ n variation)
3. **The southern Shinaic lects (Palula, Sauji, Kalkoti, Kundal Shahi) lack the Shina-proper
   bundle.** They keep tr (Sauji makes it ɬ), voice after nasals, and (Palula, Sauji, Kalkoti)
   keep intervocalic k. Palula keeps initial voiced aspirates. These are **retentions**, so they
   do not prove a southern clade. They do show that the Shina-proper innovations are younger
   than the Shinaic split, or were never shared across it. The innovations cross the southern
   group rather than defining it:
   - Sauji shares Cr > C with Shina proper.
   - Kundal Shahi shares s + stop > stop with Shina proper.
   - Kalkoti independently has nd > n.

   Palula and Kalkoti share r-metathesis after lenition (durgá > drūg, dīrgʰá > dríg). This is
   a candidate Palula–Kalkoti innovation.
4. **Chitrali (Kho, Kal)** shares the prothetic i- before s + C (P in both, A elsewhere), loss
   of length and the retention of final -i/-u. Final -i/-u retention is also in Sauji and
   Ushojo; the other lects prefer losing every final vowel. Word-final ʦ̣ʰ > ʦ̣ is shared by
   Kho, Kal and Palula. Kalasha shares with the southern Shinaic lects retained initial voiced aspirates
   and intervocalic k. Khowar has its own innovations:
   - t > r and k g > ɣ
   - a-harmony (a > o before a following a)
   - it alone keeps r + C (A +0.001)
5. **Which of Kashmiric and Chitrali does Shinaic side with?**
   - **With Khowar, by innovations:** Sh and Ush share sv > śp and v- > b- with Khowar. Sh
     also shares nd > n with Khowar (nd > n is also in Kalkoti).
   - **With Kashmiri, by types of reduction:** deaspiration, s + stop and r + C reduction, and
     loss of intervocalic k. These are also natural lenitions and are shared with the Chitrali
     lects in part (rC > C in Kalasha).
   - So the innovation-based links of Shina proper point to Khowar (the Gilgit–Chitral contact
     zone), and the Kashmiri links look like parallel drift. The southern Shinaic lects share
     most with Kalasha, but by retention.
6. **Survey sources behave alike across lects.** Decker, Rensch et al. and Backstrom mark
   phonetic length and use digraph aspirates in every lect. Liljegren's wordlist writes ʦ̣ for c
   in every Shinaic lect. Such differences are source effects, not dialect evidence.

Largest-effect rules per lect (train LOO). Besides accent loss, citation morphology and
final-vowel loss, the biggest single contributors are:
- intervocalic stop loss everywhere
- the affrication in Kashmiri
- length loss in Chitrali
- deaspiration in Kalkoti

| lect | five largest per-rule LOO values (train ΔNED if removed) |
|---|---|
| K | `V > ∅ / V (C) C _ # \| V C C C _ #` +0.064; `V t {i ē} > u n / _ #` +0.046; `V > $1-acc` +0.026; `{k g t d c j y} > ∅ / V _ V` +0.021; `{c cʰ j} > {ʦ ʦʰ z}` +0.021 |
| pog | `V t {i ē} > n u / _ #` +0.062; `V > ∅ / C _ # // C n _` +0.060; `{k g t d c j y v h b} > ∅ / V _ V` +0.029; `{a i} > ∅ / _ V` +0.021; `V > $1-acc` +0.021 |
| dod | `V > $1-acc` +0.054; `V t {i ē} > ṇ ō / _ #` +0.034; `V > ∅ / V C _ #` +0.030; `{k g t d c j y v p} > ∅ / V _ V` +0.015; `v > b / # _` +0.012 |
| Sh | `V > $1-acc` +0.049; `{a i u e ā ī ū ē ō} > ∅ / V (C) C _ # \| V C C C _ #` +0.027; `V t {i ē} > o i k / _ #` +0.027; `{k g t d c j y v h} > ∅ / V _ V` +0.022; `v > b` +0.014 |
| Ush | `V > $1-acc` +0.071; `{k g t d c j y v h ḍ b} > ∅ / V _ V` +0.053; `{a ā ē ō} > ∅ / V (C) C _ # \| V C C C _ #` +0.052; `a > ∅ / V _` +0.029; `{t d} r > {ʦ̣ ʣ̣}` +0.013 |
| Phal | `V > $1-acc` +0.034; `{a i u ā ī ū ē ō} > ∅ / V (C) C _ # \| V C C C _ #` +0.034; `{g t d c j y v} > ∅ / V _ V` +0.020; `k ṣ > ʦ̣ʰ` +0.013; `v > b / # _` +0.010 |
| Sv | `V > $1-acc` +0.056; `{g t d c j y v h} > ∅ / V _ V` +0.047; `V t {i ē} > i l / _ #` +0.036; `{a ā ē ō} > ∅ / V (C) C _ # \| V C C C _ #` +0.024; `{a i} > ∅ / _ V // _ V #` +0.018 |
| Kalk | `V > ∅ / V (C) C _ # \| V C C C _ #` +0.057; `{g t d c j y v h b} > ∅ / V _ V` +0.048; `V > $1-acc` +0.032; `V t {i ē} > ū n / _ #` +0.027; `{bʰ dʰ gʰ jʰ ḍʰ} > {b d g j ḍ}` +0.020 |
| Kund | `V > ∅ / V (C) C _ # \| V C C C _ #` +0.090; `V > $1-acc` +0.036; `{k g t c j y v h} > ∅ / V _ V` +0.024; `{ā ī ū ē ō} > {a i u e o}` +0.021; `{a i} > ∅ / _ V` +0.019 |
| Kho | `{ā ī ū ē ō} > {a i u e o}` +0.057; `V t {i ē} > i k / _ #` +0.051; `V > $1-acc` +0.040; `a > ∅ / V (C) C _ # \| V C C C _ #` +0.040; `{p b d v y h c j} > ∅ / V _ V` +0.024 |
| Kal | `V > $1-acc` +0.053; `{ā ī ū ē ō} > {a i u e o}` +0.033; `{g t d c j y v h b} > ∅ / V _ V` +0.029; `V t {i ē} > i k / _ #` +0.026; `{a ā ē ō} > ∅ / V (C) C _ # \| V C C C _ #` +0.016 |

## 5. Notable findings

- **Khowar vowel harmony (coordinator's probe, tested as rules on train).**
  - Test 1, rounding harmony generalised. The final rule is
    `{a á} > o / # (C)(C) _ C (C) {a á u ú ō ṓ}` (LOO +0.0139, 271 etyma): first-syllable a
    rounds before a following short a (kárman > korum, śatá > śor, rákṣati > roʦ̣ʰík). It also
    rounds before a following u or ō (caturtʰá > ocorti, kapṓta > kovór).
    - Extending the set from {a} to {a u ō} gains −0.0011.
    - Adding ū: no gain. Adding "labial + i" as a trigger: no gain.
    - Before ā and front vowels it stays a.
    - The rule replaced the inherited "stressed á > o" (Δ −0.004); Vedic accent plays no role.
  - Test 2, i > u before u: 4 etyma, LOO 0.0000, rejected. The labial + ī variant: no gain,
    rejected.
  - Test 3, front dissimilation `{i í} > e / #(C)(C)_C(C){i í}`: kept, but weak (LOO +0.0002, 11
    etyma; ríṣṭi > reṣṭ, kilima > kelel). Including ī as a trigger makes it worse (+0.0005).
  - Test 4, stress/weight hypothesis:
    - Pure weight (round whenever the next syllable is light, any vowel quality) is worse than
      the quality condition (+0.0024).
    - Quality ∩ light next syllable is marginally better (−0.0003).
    - Quality before a *closed* next syllable is much worse (+0.0137).
    - So both matter. Heavy (closed) second syllables block the rounding, which supports the
      weight-sensitive-stress explanation. But a long ō still triggers it and a light i does
      not, so vowel quality cannot be reduced to weight. The simpler quality rule is kept.
  - Test 5, weaker changes:
    - ō > u in closed first syllables: +0.0007, rejected.
    - ū > i before ś/ṣ: −0.0007 but only 4 etyma, rejected as item-specific.
  - Ordering:
    - The harmony must precede the Kho length neutralisation (+0.0259 if after it: ā would
      then trigger).
    - It must precede the -ik citation rule (+0.0058). The trigger is the OIA vowel of the
      second syllable, so this is an early change.
    - It fails in every other lect (+0.001 to +0.006 for the round-trigger part), so the
      rounding harmony is **Khowar-only**.
  - **Kashmiri has a parallel lowering** of first-syllable i before a following a (siddʰa >
    sɛd; LOO +0.001). Both look like umlaut of the first vowel by the (later lost)
    second-syllable vowel.
- **Kalasha (coordinator's probe).**
  - `a > o` before a nasal cluster + a is kept but weak (áŋga > oŋg, gaṇḍa > goṇ; LOO +0.0003,
    24 etyma). Without the "+ a" condition, and in Khowar (where the harmony already covers
    these words), it does not help. So splitting Khowar's harmony into a shared NC part adds
    nothing.
  - **i-umlaut** (first-syllable a/ā > e before a following front vowel, closed or not,
    final-i only or not): every variant *raises* train NED, in Kalasha (+0.0006 to +0.006) and
    in Khowar (+0.0006 to +0.0033). The pattern exists (ákṣi > ec, yaṣṭí > jeṣṭ), but at a 20–25%
    rate a categorical rule costs more than it gains. It is therefore **not** supported as a
    shared Chitrali innovation at this data size.
- **Khowar keeps r + C** (tʰurt, nohortik). It is the only lect in the group to do so.
- **Palula:**
  - aspiration shift (gōdʰū́ma > gʰōm, dōháyati > dʰov-)
  - monosyllable lengthening (hásta > hātʰ, kárṇa > kāṇ)
  - stressed open ā > ō
  - r-retroflexion followed by metathesis (śūrtá > ṣūṛ, durgá > drūg)
- **Sauji tr > ɬ, dr > l** (gā́tra > gāɬ, drā́kṣā > lāʦ̣). This is an innovation of its own.
- **Kalkoti:**
  - a > æ between consonants in the Kalkoti dictionary.
  - Kalkoti otherwise has the most "Kashmiri-like" flap (ṭ ḍ > r).
- **Kalasha retroflex vowels.** Trail & Cooper's ạ ọ ụ (106 train forms) come from lost ḍ, ṛ, l,
  ṇ, r (gala > gạ, kárṇa > kọ̃, nāḍī́ > nạ). They are too sporadic to model.
- **Kashmiri:**
  - ś, ṣ > h except before front vowels (har, kuhun, but śīn); s is kept.
  - Prothetic v- (vɔzul, vuṭʰ).
  - Final-syllable a > u, the trace of the masc. -u (acʰur, kɔkur).
  - ts > ʦʰ (aspirated, unlike elsewhere).
- **Poguli:**
  - sibilant + stop > h + stop (aṣṭā́ > āhṭ).
  - Initial h lost (halá > al).
- **Dodi:** final vowels are lost only after a single consonant; after clusters the -a
  survives as masc. -ō (vaḍra > baḍḍō).
- **Verb citation:** the -ati → infinitive @morph rule also hits -ti nouns and numerals (saptati,
  aśīti). This is the main systematic cost of form-based citation modelling.

## 6. New tags and engine wishes

New `@x-dardic-north-…` tags:
- `tr-affricate`: tr, dr > ʦ̣, ʣ̣ (Shina proper).
- `tr-lateral`: tr, dr > ɬ, l (Sauji).
- `ts-affricate`: ts > ʦ (K ʦʰ).
- `sv-sp`: śv/sv > śp/sp.
- `nd-n`: nasal + (voiced) dental/retroflex stop > nasal. Co-tagged `@cluster-simplify`.
- `s-stop-loss`: s/ṣ + stop > stop, ordered after lenition.
- `cr-loss`: consonant + r > consonant, ordered after lenition (Shina proper, Sauji).
- `final-sibilant-lenition`: word-final ś ṣ > y, s > z (Kundal Shahi: ākāśá > ay, dvā́daśa > bay, divasá > dīz).
- `r-retroflex`: r + t/d > ṭ/ḍ.
- `length-loss`: loss of the vowel-length contrast (Kho, Kal; Sv and Kund may be notational).
- `t-r`: intervocalic t > r (Khowar).
- `aspiration-shift`: breathiness of a medial voiced aspirate/h moves to an initial voiced stop
  (Palula).
- `monosyllable-lengthening`: a > ā in words that become monosyllables by final-vowel loss
  (Palula).
- `retroflex-sibilant-loss`: ṣ > ś before consonants (Sauji).

Engine wishes:
- **Verb-only (POS/gloss-filtered) rules**, so citation @morph stops hitting -ti nouns.
- **A replacement modifier for retroflex (dot-below) vowels and vowel nasalisation**, for
  Kalasha ạ < aḍ/al and anusvāra vowels.
- **`%source` blocks that can see the OIA input**, or a verb flag. Palula's Liljegren -e
  citation cannot be added after the chain, because verbs are no longer identifiable there.
- **Built-in per-source NED in `score`.**

## 7. Round 2: vowels and accent

New scratch tools in `tmp/sound_changes/dardic-north/`:
- `tr.py`: tries insert/replace/delete edits on train and shows the etyma that change.
- `cmp2.py`: compares two rule files etymon by etymon.
- `tonetab.py`, `tonepos.py`: cross-tabulate tone/stress marks in a source against the Vedic accent.

"Δ" below is the change in train NED when the edit is applied; negative is better. "LOO" is the
value from the final file.

### 7.1 Scores (round 1 → round 2)

Round 1 is `summary-round1.csv`. Test was never used for a decision.

| lect | rules r1 → r2 | train NED | test NED | test exact |
|---|---|---|---|---|
| K | 35 → 39 | 0.3995 → 0.3925 | 0.4016 → 0.3951 | 16.3 → 16.9% |
| Sh | 32 → 37 | 0.5246 → 0.5160 | 0.5596 → 0.5531 | 6.5 → 7.3% |
| Kho | 33 → 34 | 0.4337 → 0.4313 | 0.4651 → 0.4679 | 12.7 → 12.0% |
| Kal | 29 → 30 | 0.5008 → 0.4994 | 0.5127 → 0.5118 | 6.0 → 6.3% |
| Phal | 34 → 38 | 0.5090 → 0.5020 | 0.5389 → 0.5331 | 6.9 → 6.9% |
| Sv | 28 → 29 | 0.4987 → 0.4971 | 0.5146 → 0.5157 | 8.8 → 8.8% |
| Kalk | 26 → 28 | 0.4828 → 0.4719 | 0.5813 → 0.5723 | 4.5 → 6.3% |
| Ush | 26 → 28 | 0.4469 → 0.4306 | 0.4833 → 0.4704 | 13.3 → 15.6% |
| Kund | 23 → 23 | 0.4433 → 0.4433 | 0.4794 → 0.4794 | 10.0 → 10.0% (unchanged) |
| dod | 17 → 18 | 0.4138 → 0.4128 | 0.4858 → 0.4858 | 3.8 → 3.8% |
| pog | 18 → 18 | 0.4340 → 0.4301 | 0.5059 → 0.5140 | 9.7 → 9.7% |

Frontiers were rerun for all eleven lects.

**Audit of the partial round-2 edits** (the snapshot equals the files as they were at 11:00).
Every partial edit has positive LOO, so all were kept:
- `{a á} {s n} > ∅ / V C* _ #` for -as/-an stems, in K, Sh, Kal, Phal, Sv, Kalk, Ush and dod
  (LOO +0.0009 to +0.0066).
- K and Kal r̩ > ri after C (+0.0005, +0.0018).
- Sh final -ā shortening (+0.0015) and ā > a before ā (+0.0010).
- Phal accented open á > ā (+0.0012).
- Kho harmony blocked in CaCaCa words (+0.0012 over the plain rule).
- `if:verb` on the Phal and pog citation rules (+0.0005 and +0.0010 over unconditional).

**Why Poguli test NED rose (+0.0081).** The whole rise is one test etymon out of 31. The
`if:verb` guard stopped the -nu citation rule from firing on *áiti* 'comes near' > yēun. That
word gets no `verb` flag: its gloss does not start with "to", and -iti is not in the engine's
`VERB_ENDING` list. On train the same guard gains 0.0010, so it stays. This is an engine
heuristic gap (-éti/-aiti verbs), not a sound-change regression.

The same `if:verb` guard hurts every other lect (+0.002 to +0.033). The gloss-based flag misses
too many verbs. `if:!numeral` is neutral everywhere except Kalkoti, where it gains 0.0011 and was
adopted.

**`%gold-drop tone`** is a change of scoring and was not adopted. Scored with the directive (and
with the Schmidt/Liljegren circumflex blocks removed), train / test change by:
- Sh −0.0019 / −0.0015
- Phal −0.0071 / −0.0059
- Kalk −0.0088 / −0.0113
- Kho ±0

### 7.2 Vowel and accent rules adopted

| lect | rule (position) | LOO | examples |
|---|---|---|---|
| Kho | `{í ī́ ú ū́} > ∅ / V C* _ # // t _` (before @acc-loss) `@final-vowel-loss @stress` | +0.0009 | āṇí > aṇ, rāśí > raś, lagʰú > loʦ. The same rule on unaccented final i/u costs +0.0021 and +0.0008. |
| Phal | `{ō ṓ} > ū`, then `ā́ > ō` (any syllable), then `ā > ō / # C* _ C V C* #` `@vowel-shift @stress` | +0.0020, +0.0026, +0.0015 | gōla > gūli, rōṣa > rūṣ; tā́raka > tōro; śāka > śō, gʰrāṇá > gʰrōṇ |
| Phal | `á > ā / # C* _ C V` (partial edit) `@stress` | +0.0012 | ná > nā-, bʰárati > bʰār- |
| Sh | `if:verb {ā ī ū} > short` (before @acc-loss) `@morph @stress` | +0.0027 | vā́śyatē > baśóiki, sī́kṣatē > siʦ̣óiki, māráyati > maróiki |
| Sh | `{u ú} > o / V C* _ C* #` `@vowel-shift` | +0.0009 | rájju > rajo, tā́lu > tālo, sáktu > sāto |
| Sh | `{ai au} > {ē ō}` (long instead of short) | Δ −0.0004 | tailá > tēl, paútra > pōʦ̣o |
| K | `{ō ū} > u / V C* _ C* V` `@vowel-shift` | +0.0007 | masū́ra > mɔsur, trayōdaśī > truvāh |
| K | first-syllable i > e before a **or ā** (ā added) `@umlaut` | Δ −0.0003 (rule LOO +0.0014) | siddʰa > sɛd |
| K | `n > ñ / {i ī e} _ {i ī} #` `@palatalisation @umlaut` | +0.0008 | garbʰíṇī > gabiñ, hariṇī > hariñ, bʰaginī > biñ |
| Kalk | `{a á} > ā / # (C)(C) _ C C {a ā} #` `@x-dardic-north-monosyllable-lengthening @degem-lengthen` | +0.0066 | dánta > dān, rakta > rāt, páñca > pānj |
| Ush | the same lengthening rule | +0.0098 | aṣṭā́ > āṭ, adyá > āź, taptá > tāto. Only 11 etyma, so thin evidence. |
| Kalk | `if:!numeral` on the -ūn citation rule `@morph` | Δ −0.0011 | |

Notes on the adopted rules:
- **Kho, accented final -i/-u.** The `// t _` clause keeps the -tí nouns (jñātí > iźnāri,
  viṁśatí > biśir), whose -i survives as -ri. Accented final -i/-u are lost; unaccented ones
  survive.
- **Palula ō > ū / ā > ō is a drag chain.** ō > ū must precede ā > ō: putting it after costs
  +0.0087. It is the back partner of the existing ē > ī. Neither raising helps any other lect
  (Sh +0.004, Kho +0.011, Kal +0.007, Sv +0.005).
- **Palula ā > ō and the accent.** The accent rule and the positional rule are complementary:
  - the positional rule alone (open first syllable of a disyllable) gives −0.0010;
  - accent alone gives 0;
  - both together give −0.0014.
  Unaccented ā > ō in any open syllable costs +0.0022. Stressed ā went to ō; the Vedic accent
  marks the stress where it survived, and disyllables default to first-syllable stress.
- **Palula á > ā.** Accented-only is best. Extending it to all a costs +0.0022, to unaccented a
  +0.0034.
- **Shina verb-root shortening** is the vowel reflex of the Shina infinitive stress on
  -óiki/-ōnu (the root is pretonic). The shortening does not generalise to Shina nouns in
  closed syllables. There, accented long vowels shorten (−0.0011, 16 etyma, mostly verbs) and
  unaccented ones do not (+0.0009). It fails in K (+0.0037), Kalk (+0.011) and Phal (+0.0009).
- **Kalkoti/Ushojo monosyllable lengthening.** The Palula rule, restricted to closed
  syllables. It is accent-independent: accented a alone gives −0.0012, unaccented a −0.0019,
  both −0.0054. It is rejected by Sh (+0.010), K (+0.010), Kund (+0.007), dod and pog, so it is
  another Palula–Kalkoti (± Ushojo) link, next to r-metathesis.

### 7.3 Hypotheses tested and rejected (Δ train NED when added)

**Pretonic or medial syncope.** Rejected in every lect, whether conditioned by accent or by
position, and whether placed early or late:
- K: `u > ∅ / # C _ C Acc` −0.0001 (6 etyma, noise); `{u i}` +0.0007; heavy-next variant −0.0005
  (14 etyma, su-v- words).
- Sh: `a > ∅ / V C _ C V` +0.007 (early) or +0.0023–0.0028 (after lenition); accent-conditioned
  +0.0017.
- Kal: +0.0016 to +0.0067 in every position.
- Kho and Phal: +0.001 to +0.007.
- Ush, Kalk, Sv and Kund: +0.002 to +0.023.

**Accent-protected length** (length loss before @acc-loss, the Kundal Shahi order). Rejected
everywhere except Kundal Shahi: Sv +0.0015, Kho +0.0152, Kal +0.0087.

**Tonic lengthening** (Palula `á > ā`) elsewhere: Sh +0.0044, Kalk +0.0066, Ush +0.0086,
K +0.0050 — Palula only.

**a > o before a nasal, accented only.**
- Kal: −0.0003 on its own, but the existing harmonic rule is better. As a replacement +0.0010;
  as an addition +0.0004.
- Everywhere else +0.001 to +0.005.

**r̩.** The outcome is accent-independent throughout:
- ŕ̩ > a: K −0.0002 (noise); Sh, Kho, Kal and Phal +0.0004 to +0.0008.
- r̩ > a before a: +0.0007 to +0.0027.
- Khowar: the probe's "r̩ lost in unaccented headwords" is an alignment artefact. Unaccented r̩
  is really r̩ > ir/ri/or with metathesis (tr̩tī́ya > ortiri, *vr̩mra > birmoɣ).
  - `if:!accented r̩ > ∅` +0.0006
  - r̩ > ∅ +0.0035
  - r̩ > ri/ir/ur/or +0.0013 to +0.0028
- Final kin-term r̩ > -ar (svásr̩ > ispusār): −0.0009, but on only 4 etyma, so not adopted.

**Khowar rounding harmony** is accent-independent:
- target a only +0.0060; á only +0.0093
- dropping á as a trigger +0.0023
- "accented a does not round before u" −0.0008, but only 5 etyma (kádru > kadur, áśru > aśru),
  so not adopted.

**Kashmiri umlaut:**
- a > e before final i +0.0011; a > o before final u ±0
- u > o before a +0.0004
- lowering only in the first syllable +0.0001
- palatalisation of l/t/d before final -i +0.0006 to +0.0010
- ā > a in closed syllables +0.0032

**Closed-syllable shortening of mid vowels** (the MIA two-mora law): ē > i / _CC is Sh −0.0004,
elsewhere +0.0002 to +0.0020; ō > u is positive (worse) everywhere. This confirms §3: the group
has no MIA closed-syllable law.

**Final -á > -o (oxytone retention)** as a rule: Sh −0.0001 (a wash); Phal, Sv and Ush
+0.0065 to +0.0075.

**Shinaic-internal:**
- Sv ā́ > ō +0.0038
- Kalk and Sv i/u lengthening, rejected in Ush (+0.0036), Kalk (+0.0039) and Kund (+0.0107)
- the Kho accented final-vowel rule in Sh, Phal, Sv, Ush and Kal: 0 to +0.004

### 7.4 What the Vedic accent does and does not condition in this group

**It does condition four things:**
1. **Palula ā > ō.** The accented rule is worth +0.0026 LOO, and the accent matters beyond
   position.
2. **Palula á > ā** in open first syllables (accented only).
3. **Khowar loss of accented final -i/-u.** Unaccented ones survive.
4. **Kundal Shahi length loss in unaccented syllables** (round 1).

There is also a statistical effect below rule strength. Oxytone final -á survives as a final
vowel (Sh/Phal -o, Kho -u) more often than post-tonic -a in nouns with the accent elsewhere
(train reflexes, nouns only):

| lect | final vowel, oxytone -á | final vowel, post-tonic -a |
|---|---|---|
| Kho | 39% | 22% |
| Phal | 51% | 38% |
| Sv | 63% | 45% |
| Ush | 52% | 37% |
| Sh | 63% | 64% |

In Sh, -o specifically is 40% vs 32%.

**It does not condition:**
- Khowar harmony
- Kashmiri vowel shortening
- r̩ outcomes
- syncope
- length retention in Kho, Kal or Sv
- a > o before nasals

**Modern stress and tone placement do not follow the Vedic accent** (`tonepos.py`). The table
gives the share of reflexes with the modern stress or tone mark on the first syllable,
restricted to OIA disyllabic nouns (verbs excluded), by the position of the Vedic accent:

| lect / source | Vedic accent σ1 | Vedic accent σ2 |
|---|---|---|
| Sh / Schmidt | 82% (n=22) | 87% (n=46) |
| Sh / CDIAL | 31% | 34% |
| Phal / Liljegren | 52% | 64% |
| Phal / Strand | 56% | 54% |
| Kho / Bashir | 24% | 36% |
| Kal / Trail & Cooper | 33% | 57% |

For both accent positions the modern placement is the same or reversed. Modern Shina/Palula
pitch-accent placement is a new system, set by word shape and morphology, e.g. the stressed
Shina infinitive -óiki. Its only trace of the Vedic accent is in vowel quality and length
(Palula ō, Khowar final vowels).

**Engine wishes (round 2):**
- `VERB_ENDING` should include -éti/-aiti/-īti forms (the *áiti* case).
- Gloss-based `verb` detection misses most verbs whose reflex glosses lack "to".
- A way for `%source` blocks to read the Vedic accent, so that Schmidt/Liljegren tone type could
  be tested as a function of it.

## 9. Accent score (round 3)

Tools, in `tmp/sound_changes/dardic-north/`:
- `tra.py`: shows the Δ in plain and accent NED for each edit (`ACC=vedic` switches the
  model); `LOO=1` gives the accent LOO of every `stress` rule.
- `accsrc.py`: accent NED per source, with a sample of the misses that are wrong only in the
  accent.

Plain NED is unchanged by every adopted edit, on train and on test. All decisions used train.

### 9.1 Test accent NED

Train is in brackets. "Final" is the adopted chain, model `chain`.

| lect (acc. pairs tr/te) | none | Vedic carried | final | adopted rules (accent LOO) |
|---|---|---|---|---|
| Sh (1346/304) | 0.5509 (0.5096) | 0.5452 (0.5080) | **0.5313** (0.4801) | `V > $1+stress / _ C* Short C* #` +0.0286; `Long > $1+stress / _ C* #` +0.0008. Both sit before hiatus contraction and final-vowel loss. |
| Phal (1126/315) | 0.5733 (0.5258) | 0.5763 (0.5232) | **0.5630** (0.5113) | `V > $1+stress / _ C* V C* #` (before hiatus) +0.0113; monosyllables `V > $1+stress / # C* _ C* #` +0.0006 |
| Kho (720/146) | 0.5238 (0.4835) | 0.5255 (0.4846) | **0.5025** (0.4629) | `V > $1+stress / _ C* #` at the end of the chain +0.0206 |
| Kal (601/153) | 0.5279 (0.5046) | 0.5219 (0.5079) | **0.5110** (0.4973) | `V > $1+stress / _ C* V C* #` just after @acc-loss +0.0073 |
| Kalk (56/13) | 0.5810 (0.4720) | 0.5617 (0.4635) | **0.5049** (0.4353) | `V > $1+stress / _ C* #` at the end of the chain +0.0368 |
| Sv (58/14) | 0.5091 (0.5300) | 0.5393 (0.5312) | unchanged | every placement is worse on train (penult +0.018, last σ +0.010, first σ +0.031), so nothing adopted |

Two patterns cover everything:
- **The old penult, placed before apocope:** Shina, Palula, Kalasha.
- **The last syllable after apocope:** Khowar, Kalkoti.

These are largely the same syllable. In Shina the penult rule placed before hiatus contraction
(−0.0290) and before apocope (−0.0288) are both marginally better than the last syllable at
the end of the chain (−0.0272). Khowar keeps final -i/-u, which is why it prefers the final
syllable.

### 9.2 Hypotheses rejected (Δ train accent NED against the adopted chain)

**Shina:**
- Final -o or any final vowel stressed, with the penult unstressed: +0.0024 / +0.0025.
- Antepenult: +0.0126.
- `if:verb` stress one syllable further left: +0.0037.
- Placement only for non-verbs: +0.0050 (−0.0240 vs −0.0290).
- Monosyllable rule: +0.0001.
- Exception for a CC-closed final: ±0.
- Earlier placement (before @acc-loss): −0.0238, i.e. worse than the adopted −0.0290.
- The -aka nouns in -o and the verb infinitives are already right under the penult rule. Before
  apocope, maróik has its penult on -ó-, and hastaka > hasto is stressed on the root.
- Most remaining misses are gender and number endings that the chain does not generate
  (banó, barí, kiṣī́).

**Schmidt circumflex block.** It writes a mark on every long vowel, but it still pays. Deleting
it costs +0.0014 accent and +0.0055 plain. Restricting it to the first or last long vowel gives
+0.0001 / ±0. Schmidt forms mostly carry exactly one mark (261 of 450 train forms have one long
vowel and one mark; 115 have one mark and no long vowel), so multiple predicted marks are rare after the chain's shortenings.

**Palula:**
- Weight-sensitive placement (long final vowel, else the penult): ±0.
- Placement before apocope instead of before hiatus: +0.0010; before @acc-loss: +0.0028.
- Stress attracted to a first-syllable ō (< stressed ā): −0.0004 for the pair, LOO +0.0001
  each. Dropped as noise.
- Deleting the Liljegren circumflex block: +0.0122 accent and +0.0120 plain.

**Khowar:**
- Penult for words ending in -i/-u: +0.0004 to +0.0011.
- Penult for vowel-final words: +0.0019.
- Placement before apocope: +0.0110.
- Last syllable only if closed, else the penult: +0.0345.

Khowar final-vowel loss (plain NED), accent vs position. The Vedic-accent rule
`{í ī́ ú ū́} > ∅` beats every positional rewrite (Δ plain / accent):
- after a single C: +0.0035 / +0.0055
- after CC: +0.0002 / +0.0008
- in disyllables: +0.0024 / +0.0052
- after a long first vowel: +0.0017 / +0.0038
- after a short first vowel: +0.0025 / +0.0022

The round-2 finding stands: the loss is conditioned by the Vedic accent, not by position.

**Kalasha:**
- Placement before length loss or before apocope: ±0 / +0.0002.
- Weight-sensitive placement: ±0.
- Verbs stressed on the first syllable (Trail & Cooper kásik): +0.0033.
- Monosyllables: +0.0001.
- Penult blocked before a CC-closed final syllable: −0.0003 (noise, not adopted).
- Engine note: a `+stress` placed *before* @acc-loss scores −0.0011 better, but only because
  `$1-acc` turns a rule accent on a Vedic-accented vowel back into a latent Vedic accent. The
  rule is placed after @acc-loss instead.

**Kalkoti:** the penult at the end of the chain is +0.0411 worse than the last syllable.

### 9.3 The Vedic accent, given its best chance

Two tests were run:
1. `%accent vedic` on top of the final chains, so that the Vedic accent fills whatever the
   rules leave unplaced.
2. "Vedic where the headword is accented, positional rule elsewhere" (`if:!accented` placement
   with `%accent vedic`).

| lect | 1: Vedic fills gaps (train) | 2: Vedic where accented (train) |
|---|---|---|
| Sh | 0.4803 vs 0.4801 (test 0.5289 vs 0.5313) | +0.0146 |
| Phal | 0.5109 vs 0.5113, ≈ the monosyllable rule; not adopted, the model stays `chain` | +0.0062 |
| Kho | 0.4627 vs 0.4629 | +0.0155 |
| Kal | 0.4983 vs 0.4973 | Vedic, else penult: 0.5021 vs 0.4973 |

With no rules at all, Vedic carried is better than no accent only in Sh (−0.0016), Phal
(−0.0026) and Kalk (−0.0085). It is worse in Kho, Kal and Sv.

**Conclusion.** The modern accent position is *not* the Vedic accent in any lect of the group.
It is a new positional accent: the old penult (Shinaic, Kalasha), which surfaces as the final
syllable where apocope left the word shorter (Khowar, Kalkoti). This matches §7.4 (the
`tonepos.py` counts). The Vedic accent's only visible effects are segmental and pre-date the
new accent:
- Palula ā́ > ō and á > ā;
- Khowar loss of accented final -i/-u;
- Kundal Shahi length.

## 9b. Aspiration-conditioned accent shift, and the Vedic accent with aspiration controlled

New scratch tool: `tmp/sound_changes/dardic-north/accdiff.py` lists the etyma whose accent cost
changes under an edit.

Δ is train accent NED against the §9 chains; plain NED is unchanged in every test.

**The shift test.** The shift is the coordinator's pair of rules:
- `V > $1+stress / {bʰ dʰ gʰ jʰ ḍʰ h} Stress C* _`
- `Stress > $1-stress / {bʰ dʰ gʰ jʰ ḍʰ h} _ C* V`

`Stress` also matches the latent Vedic accent. So where a shift rule runs after @acc-loss, it
also fires next to Vedic-accented syllables.

| lect | variant | Δ acc |
|---|---|---|
| Sh | penult moved before deaspiration (±0), + shift from any aspirate/h onset | +0.0028 |
| Sh | same, σ1 onset only (`#` anchor) | +0.0017 |
| Sh | same, voiced aspirates only | +0.0015 |
| Sh | same, intervocalic (lost) aspirate only | +0.0006 |
| Sh | same, σ2 onset (accent moves onto the syllable the aspirate opens) | +0.0005 |
| Sh | same, σ2 onset, final syllable only | +0.0002 |
| Sh | late shift after initial h- (the only aspirate left after apocope) | +0.0001 |
| Phal | shift after the penult rule, from any initial aspirate/h | +0.0001 (10 etyma) |
| Phal | same, with the Cr- onset | +0.0003 |
| Phal | same, voiced aspirates only / h only | +0.0001 each |
| Phal | late shift, after apocope | +0.0004 (5 etyma, all verbs) |
| Kal | shift from any aspirate/h onset | −0.0008 |
| Kal | same, σ1 onset only | −0.0003 |
| Kal | same, σ2 onset | −0.0006 |
| Kal | same, voiced aspirates only | −0.0008 |
| Kal | same, h only | +0.0001 |
| Kho | final-syllable stress already covers σ2 | ±0 |

**Why the shift fails here.** The coordinator's effect (σ2 accent 55–56% after an aspirate vs
36–45% without) is measured on reflexes that *kept two syllables*. The chains apocopate almost
every final vowel. What survives is mostly gender/number endings (-o, -i, -u) that the chains
do not generate, and verb citation forms. So a stress shifted onto σ2 usually lands on a vowel
that is deleted later, and the word ends up with no predicted accent at all.

In Kalasha the whole gain comes from exactly that: gōdʰū́ma goúm → goum, bʰakti bʰáti → bʰati,
bʰitta bʰít → bʰit. A missing accent costs less than a wrong one; the shift predicts no correct σ2
accents. That is an artefact, not a prosodic change, so **nothing was adopted**.

A fair test needs chains that keep the inflectional final vowel (Sh/Phal masc. -o, fem. -i).

**Vedic vs positional, aspiration controlled.** The setup:
- `%accent vedic`;
- positional penult (placed before deaspiration) for unaccented headwords and for every word
  containing a voiced aspirate or h;
- the Vedic accent shows only in accented, aspirate-free words.

| lect | Δ acc, Vedic where accented (no control) | Δ acc, Vedic only in aspirate-free words |
|---|---|---|
| Sh | +0.0149 | +0.0123 |
| Phal | +0.0061 | +0.0049 |
| Kal | +0.0038 | +0.0042 |

**Conclusion.** Controlling for aspiration narrows the Vedic deficit slightly in Sh and Phal,
but the Vedic accent still loses clearly to the positional penult in every lect. This confirms
§9.3 and the coordinator's Palula count: accent on the old σ1 in 69% of barytones and 73% of
oxytones. The Vedic accent survives only in segmental effects (Palula ā́ > ō; Khowar accented
final-vowel loss), not in the position of the modern accent.

No rule files changed in this follow-up, so the §9 frontiers stand.

## 10. Extension-aware score (round 4)

All numbers use the menu with -akka:
`EXT_MENU = aka, akka, ikā, uka, aḍaka, allaka, araka`. "Before" is the round-3 file (the
round-4 lines stripped: `tmp/sound_changes/dardic-north/r3/`).

Scratch tools:
- `trx.py`: shows Δplain, Δaccent and Δext for each edit.
- `extshow.py`: prints a chain's output for each extended variant.
- `menu.py`: monkeypatches `EXT_MENU` in its own process to score candidate items; the engine is
  untouched.

### 10.1 Scores

| lect | plain train | plain test | ext train | ext test | chosen endings, train (after) |
|---|---|---|---|---|---|
| K | 0.3925 = | 0.3951 = | 0.3702 = | 0.3757 = | bare 86, akka 6, aka 4, aḍaka 2, allaka 1 |
| Sh | 0.5160 = | 0.5531 = | 0.4263 = | 0.4529 = | bare 59, aka 14, ikā 11, uka 9, akka 3, allaka 2, aḍaka 1, araka 1 |
| Kho | 0.4313 → 0.4312 | 0.4679 = | 0.3938 → **0.3861** | 0.4253 → 0.4165 | bare 76, uka 8, ikā 7, aka 5, akka 2, araka 1, aḍaka 1 |
| Kal | 0.4994 → 0.4988 | 0.5118 → 0.5112 | 0.4479 → **0.4412** | 0.4508 → 0.4478 | bare 66, aka 16, ikā 7, uka 4, aḍaka 3, allaka 2, araka 2 |
| Phal | 0.5020 = | 0.5331 = | 0.4501 = | 0.4756 = | bare 66, aka 9, ikā 9, uka 7, akka 5, aḍaka 2, allaka 1, araka 1 |
| Sv | 0.4971 → 0.4960 | 0.5157 → 0.5089 | 0.4607 → **0.4155** | 0.4774 → 0.4089 | bare 59, ikā 15, aka 10, akka 6, uka 5, allaka 3, aḍaka 2, araka 1 |
| Kalk | 0.4719 = | 0.5723 → 0.5700 | 0.4417 → **0.4288** | 0.5244 → 0.5142 | bare 75, ikā 7, allaka 4, akka 4, aka 3, uka 3, aḍaka 3 |
| Ush | 0.4306 → 0.4302 | 0.4704 = | 0.3831 → **0.3642** | 0.4221 → 0.3981 | bare 72, aka 11, ikā 7, uka 6, allaka 2, araka 1, akka 1 |
| Kund | 0.4433 = | 0.4794 = | 0.4177 → **0.4094** | 0.4539 → 0.4515 | bare 83, allaka 5, uka 4, akka 4, aka 1, ikā 1, araka 1 |
| dod | 0.4128 = | 0.4858 = | 0.3952 → **0.3792** | 0.4179 → 0.4403 (test worse: 26 pairs; the -aka > -o rule displaced -uka/-akka picks) | bare 78, aka 6, akka 6, uka 5, allaka 2, ikā 2, araka 2 |
| pog | 0.4301 = | 0.5140 = | 0.4090 → **0.4050** | 0.4914 = | bare 83, akka 9, uka 4, allaka 1, aka 1, aḍaka 1 |

**Accent score** (the engine's accent scoring changed after round 3, so these differ from §9):
- Kal 0.5175 → 0.5162
- Kho 0.4813 → 0.4811
- Sv 0.6023 → 0.6004
- Kalk unchanged (0.4518)
- Sh, Phal: rule files unchanged

No edit worsens the accent score, and no placement rule was touched.

### 10.2 Rules added (train extension-aware LOO)

| lect | rule | ext LOO | Δplain / Δacc when added |
|---|---|---|---|
| Sv | `{a á} k {a á} > o / _ #` (@morph @aka: -aka > -o, as in Sh) | +0.0120 | −0.0003 / 0 |
| Sv | `k > ∅ / V _ {a ā} #` (@int-stop-loss @aka: suffix k lost, stem k kept) | +0.0192 | −0.0008 / −0.0019 |
| Ush | `{a á} k {a á} > o / _ #` | +0.0120 | −0.0003 / — |
| Ush | `ā > ∅ / {i u} _ #` (@hiatus: -ikā/-ukā > -i/-u; Ush otherwise keeps final VV) | +0.0066 | 0 |
| Kalk | `{a á} k {a á} > o / _ #` | +0.0129 | 0 / 0 |
| Kund | `{a á} k {a á} > o / _ #` | +0.0062 | 0 |
| Kund | `{a ā ō} > ∅ / u _ #` (-uka > -u) | +0.0021 | 0 |
| dod | `{a á} k {a á} > o / _ #` | +0.0111 | 0 |
| dod | `{a ā ō} > ∅ / u _ #` | +0.0049 | 0 |
| pog | `{a ā ō} > ∅ / u _ #` | +0.0041 | 0 |
| Kal | `k > ∅ / {i ī u ū} _ {a ā} #` (-ika/-uka > -i/-u; stem k survives) | +0.0067 | −0.0006 / −0.0013 |
| Kho | `{k g} > ∅ / V C* u _ {a ā} #` (suffixal -uka > -u, beside the existing -ika > -i; root k stays ɣ: yū́kā > źoɣ) | +0.0077 | −0.0002 / −0.0002 |

**Rejected:**
- *Kal -aka > -a/-o/-u/∅*: ext +0.007 to +0.021, plain +0.0013 to +0.0025. Kalasha keeps -ak;
  its "aka" share of 16% is -ak.
- *Kal k > ∅ before any final -a*: plain +0.0013.
- *Kho -aka > -o/-u/-a*: plain +0.0003 to +0.0014.
- *Kho k-loss after any vowel*: plain +0.0033.
- *Palula -ikā/-uka > -ī/-ū (or k loss)*: ext −0.007 but plain +0.0019 to +0.0023 and accent
  +0.002 to +0.003. Palula bare headwords in -ika keep -ik, so it stays open.
- *Kalk suffix k loss*: plain +0.0011, accent +0.0067.
- *Ush -aka > -u*: ext +0.0165. The -u is the Kashmiric/Kalasha-type ending, not Shinaic.

**The Shina -aka rule** (`{a á} k {a á} > o / _ #`) and the Palula copy behave correctly on the
variants: gʰōṭa + aka > gōṛo, + ikā > gōṛi, + uka > gōṛu, + aḍaka > gōṛaṛo, + allaka > gōṛalo.
It doubles nothing, because the menu strips the final vowel first. On headwords already in
-aka, the extra -aka variant (hastaka > hastakaka > hatako) is just a worse candidate and is
never chosen. Kashmiri loses every extension ending to apocope (the masc. -u is an unscored
matra), so its aka/ikā/uka variants coincide with the bare form. Only -aḍaka/-araka > -ur and
-allaka > -ul are distinct, plus -akka > -uk/-ok.

### 10.3 What the choices say

**-aka.** Clearly productive in Shina (14% + akka 3), Kalasha (16%, as -ak), Ushojo (11%),
Sauji (10%) and Palula (9%).

**-ikā and -uka.** These are about as frequent as -aka in Shinaic and Khowar: the fem. -i and
the -u stems.

**-akka (-k endings).** Chosen by the Kashmiric lects: K 6%, pog 9%, dod 6%. This matches
K. -uk/-ok and Poguli -k. Also chosen by Palula (5%) and Sauji (6%).

**-ḍa, -la, -ra extensions.** Rare everywhere (1–5%):
- -aḍaka is productive only in K. In K, and in every lect with ṭ/ḍ > r (Kalk, Ush, Kund, dod,
  pog), it is indistinguishable from -araka: dropping either one costs 0.0000.
- -allaka is the most productive of the three in the small lects (Kalk 4%, Kund 5%).

### 10.4 Should the menu change?

`menu.py` adds each candidate to the old six-item menu on train. Under a min-over-variants
score, any extra variant lowers NED a little by chance. So each candidate is compared with two
nonsense controls, -ampa and -oṣa, which lower train ext-NED by:

| Sh | Kho | Kal | Phal | K | Sv | Kalk |
|---|---|---|---|---|---|---|
| −0.0017 to −0.0019 | −0.0007 to −0.0024 | −0.0010 to −0.0029 | −0.0010 | −0.0027 to −0.0031 | −0.0006 to −0.0009 | −0.0006 to −0.0008 |

Only gains well beyond the lect's control band are evidence. Counts in brackets are train
reflexes that choose the item.

**Recommend adding:**

**-iya/-īya (-īya ≈ -iya).** Real MIA/NIA morphology (the -iya adjective/noun suffix).
- Palula −0.0087 (73 reflexes; control −0.0010)
- Shina −0.0049 (81; control −0.0019)
- The small lects agree: Ush −0.0095 (5), dod −0.0082 (10), pog −0.0083 (5)
- Kho, Kal and Sv never choose it.

**Plain final -ī / -i / -u (no -ka).** The Dardic fem. -i / masc. -u endings.
- Clear only in Palula: -ī −0.0051 (44), -i −0.0043 (50), -u −0.0037 (40), against a control of
  −0.0010.
- Elsewhere at or below noise, e.g. Sh -ī −0.0022 (53).
- It is real morphology, but largely a surface fix: the chains apocopate the stem vowel, and
  these endings restore it.

**-āla/-ālu.**
- Shina −0.0041 (129 reflexes, the largest count of any candidate; control −0.0019)
- Palula −0.0040 (53)
- Kalkoti −0.0036 (13)
- Sauji −0.0031 (26)

It is real, well attested (-āla/-ālu adjectival and agentive), and worth adding as one item.

**Borderline:**
- **-ikka.** Kho −0.0043 (57) against a control of up to −0.0024; Kal −0.0031, at noise. The
  Khowar gain may reflect its k > ɣ, which -ikka escapes.
- **Bare -ka.** Kho −0.0040 (59) and Kal −0.0034 (45) are near their controls (−0.0024,
  −0.0029). Mostly a surface fix for the survival of intervocalic k in Chitrali.

**Not recommended:**
- -aḍa, -ala, -ara without -ka are at noise in the large lects. They only matter where the -ka
  ending is lost anyway (Sv, Kalk ≈ −0.003 to −0.006, i.e. partly noise). Ushojo -aḍa −0.0136
  rests on 13 reflexes.
- -illa/-ulla are at noise.
- -ita (participles) helps Palula (−0.0052), but these are verbal forms, not extensions.

**Menu items never or hardly chosen:**
- -aḍaka in Kund, dod and pog (0–1%), and redundant with -araka wherever ḍ > r.
- -araka in Kalk.
- -ikā in K (0%) and pog (0%).

None of these is worth dropping family-wide; they are simply redundant in Kashmiric.

## 11. Verb citation forms (round 5, cut short at 18:27)

Verb-menu NED is train/test; "plain" (with the citation rules) is unchanged in every lect.
All four acceptance conditions were checked on train:
- plain train change ≤ +0.0004 (Phal), otherwise 0;
- noun extension-aware score better: Sh −0.0041, Phal −0.0043, dod −0.0353, pog −0.0508;
- accent unchanged or better.

| lect | verb-menu NED before | verb-menu NED after | plain (unchanged) | citation form, OIA source | change |
|---|---|---|---|---|---|
| Sh | 0.5589 / 0.5168 | **0.5260 / 0.4652** | 0.4591 / 0.4294 | -ōno/-ōnu/-ōn < -ánaka | `{a á} > ō / _ n {a á} k` (@vowel-shift); -aka > -o retagged from morph to @hiatus @aka |
| Phal | 0.4785 / 0.4245 | **0.4508 / 0.3986** | 0.5085 / 0.4839 | Liljegren -e < -ati (-ai > -e) | `a i > e / _ #` (@hiatus); -aka > -o retagged @hiatus |
| dod | 0.4928 / 0.6000 | **0.3839 / 0.5250** | 0.3529 / 0.4167 | -ṇō < -anaka | `{a á} > ∅ / V C* _ C {a á} k {a á} #` (@syncope); `n > ṇ / _ {a á} k {a á} #` (@n-retroflex); -aka > -o retagged @hiatus |
| pog | 0.4981 / 0.6183 | **0.2807 / 0.4611** | 0.3240 / 0.4889 | -nu < -anaka | the same syncope; `{a á} k {a á} > u / _ #` (@hiatus @aka) |

In the dod row, "before" is on 23 train / 4 test verb pairs.

**Not done:**
- **Kho (0.4918/0.4759), Kal (0.4440/0.4936).** The -ik infinitive has no source in the menu.
  A monkeypatched test (`vmenu.py`, engine untouched) shows that adding **-itaka** gives
  Kalasha −0.077 train (103 reflexes pick it: -itaka > -iak > -ik). **I recommend adding
  -itaka.** Khowar gains nothing from it, because Khowar t > r, and from -ika only −0.0017.
- **K (already 0.2559/0.2142, -anaka > -un).**
- **Kalk, Sv; Ush and Kund** (no verbs).

## Round 6 audit

Method:
- `residuals.py` for error classes and ending pairs.
- A contextual confusion table on the best ext/verb variant: `tmp/sound_changes/dardic-north/conf.py`.
- Train-only trials: `trv.py` (Δ plain/accent/ext/verb).
- `gates.py` for every install. The candidate is built with `mk.py`; the base is backed up in
  `tmp/sound_changes/dardic-north/r6base/`.

Per-lect before→after is from `gates.py r6base/OIA-X.sc rules/OIA-X.sc`. Every lect's
cumulative verdict is ACCEPT.

### Summary (ext NED train / test; also verb, plain, exact)

| lect | ext train | ext test | verb train / test | plain train / test | exact train / test |
|---|---|---|---|---|---|
| Sh | 0.4256 → **0.4132** | 0.4475 → 0.4322 | 0.4680 → 0.4396 / 0.3959 → 0.3344 | 0.5160 → 0.5101 / 0.5531 → 0.5459 | 6.5 → 6.8% / 7.3 → 7.3% |
| Phal | 0.4374 → **0.4074** | 0.4598 → 0.4190 | 0.4455 → 0.4357 / 0.3928 → 0.3867 | 0.5024 → 0.5006 / 0.5331 → 0.5241 | 7.7 → 7.9% / 6.9 → 7.1% |
| Kho | 0.3794 → **0.3749** | 0.4034 → 0.4003 | 0.3726 → 0.3606 / 0.3539 → 0.3489 | 0.4312 → 0.4306 / 0.4679 → 0.4667 | 13.2 → 13.2% / 12.0 → 12.2% |
| Kal | 0.4272 → **0.4249** | 0.4402 → 0.4364 | 0.3561 → 0.3483 / 0.4208 = | 0.4988 → 0.4970 / 0.5112 → 0.5074 | 7.0% = / 6.3% = |
| Kalk | 0.4260 → **0.4222** | 0.5097 = | 0.4821 → 0.4574 / 0.3889 = | = | = |
| dod | 0.3730 → **0.3708** | 0.4318 → 0.4279 | 0.3820 → 0.3715 / 0.5071 → 0.4821 | = | = |

The accent score is better or equal everywhere:
- Sh 0.5107 → 0.5031
- Phal 0.5523 → 0.5508
- Kho 0.4811 → 0.4800
- Kal 0.5162 → 0.5132

| Ush | 0.3642 → **0.3557** | 0.3981 → 0.3936 | — | 0.4302 = / 0.4704 → 0.4815 (see below) | 13.6 → 14.2% / 15.6% = |
| pog | 0.3844 → **0.3756** | 0.4699 → 0.4645 | = | = | = |
| Kund | 0.4067 → **0.4026** | 0.4482 → 0.4432 | — | = | = |

**Audited, nothing passed:** K, Sv. See the rejected lists below.

**Ushojo test plain rose by +0.011** (45 test pairs). The gate checks only train plain; the
rise comes from bare test headwords with a-a hiatus, which the round-6 contraction now turns
into ā.

### Error profiles

The ending class dominates everywhere:

| lect | ending share of error | other top classes |
|---|---|---|
| Sh | 51% | C ins/del 14%, C quality 8%, V missing 7%, V quality 6.5%, V length 6.7% |
| Kho | 42% | C ins/del 19%, C quality 12%, V quality 10% |
| K | 50% | C ins/del 17%, C quality 10%, V quality 8% |
| Phal | 52% | C ins/del 13%, V quality 8% |
| Kal | 47% | C ins/del 15%, C quality 12% |
| small lects | 49–61% | — |

Most ending misses are gender/number endings and case forms the chains do not generate:
- Sh -ī/-e/-ā;
- K oblique -as and the unscored matras;
- Kal -ak vs -a.

They also include mislinks, compounds and phrases (tūs kʰāʰ, nāl-mut, val-moc). These are
lexical or source variation and are left alone.

### Kept changes (all `(round 6)` in the rule files)

**Sh:**
- `{a ā} {u ū} > ō / _ C` and `ā > ∅ / _ ō` (@hiatus leftovers: māuli > mōli, kʰāōno > kʰōno).
- Word-final ʦ̣ʰ > ʦ̣ (as already in Kho, Kal, Phal).
- `{ṭʰ tʰ} > {ṭ t} / {s ṣ} _ V`. Aspiration is lost after a sibilant before a vowel
  (aŋguṣṭʰá > agúṭo, ṓṣṭʰa > ōṭo); finally it stays (hásta > hʌtʰ). Plain −0.0022.
- Pre-pretonic long-vowel shortening `{ā ī ū ē ō} > short / _ C* V C* V C* V`, before apocope.
  - The broader pretonic version (`_ C* V C* V`) is much better on train: ext −0.010,
    verb −0.026.
  - It fails the test-accent gate (+0.0011), because it shortens long vowels that Schmidt's
    block marks with a circumflex.
- `dʰ r > ʣ̣`: dʰr joins the tr/dr affrication (vīdʰrá > béʣ̣u).
- `o > u / ō n _ #`: -ōno > Gilgiti -ōnu. Verb −0.023 train, −0.038 test.
  - The general dissimilation `o > u / {ō o} C _ #` is equal on train but fails test accent
    (+0.0008).
- `{ś s} r > ṣ` (below).

**Kho:**
- Final long -ī is lost regardless of accent: `{ī ī́} > ∅ / V C* _ #`.
- Late medial syncope `{a i u} > ∅ / V C _ C V C`. Late placement is ext −0.0015; the
  round-2 *early* syncope was rejected.
- The a + i > ei rule (`a > e / C _ i`, dʰāpayati > deik) was tagged `@morph`, so it was
  switched off for verbs. Retagged `@aya-e @hiatus`: verb −0.0085.

**Phal:**
- `k > ∅ / V C* V _ #`: a final k left by apocope in polysyllables is lost
  (áṇu-uka > āṇu, taptá-uka > tātu). Ext −0.025 train, −0.027 test.
- `a k a m > ū m / _ #`: neuter -akam > -ūm (cʰōr-akam > ʦ̣ʰorū́m).
- `{ś s} r > ṣ`.
- **Two single-rule gate overrides, noted in the rule comments.** Each rule alone raised
  test-verb NED by +0.0013 / +0.0014, above the 0.001 limit. `vdiff.py` traced each rise to a
  single mislinked pair:
  - *vunāti > sil (the -ik candidate matched sil by chance);
  - upapatí > vī́vay.

  The cumulative Palula verdict is ACCEPT, with test verb −0.0061.

**Kal:**
- `{ś s} r > ṣ` (úpaśrayati > uṣík, *avaśrayati > uṣím). Plain −0.0017, accent −0.0030,
  test ext −0.0039.
- The same rule passes in Sh (test accent −0.0106) and Phal (test ext −0.0078). It is neutral
  or worse in Kho, K, Sv, Kalk and Kund.

**Kalk:**
- `a > ū / V C* _ n #`: the -aṇa infinitive > -ūn (dūn, būn). Verb −0.018.
- -aka > -o retagged from morph to `@hiatus @aka`. Verb −0.0067.

**Ush:**
- `a a > ā` before the hiatus rule: two a's in hiatus contract to ā instead of losing one
  (mahallaka > mālo/-ī/-u). Ext −0.0085.
- The same rule is rejected in Sh (verb +0.0020), Phal (ext +0.0016), Kho (+0.005), Sv and
  Kalk (ext worse).

**pog, Kund:**
- `k > ∅ / V C* V _ #`, the Palula final-k loss. Ext −0.0089 (pog), −0.0041 (Kund); test
  also better.
- Rejected elsewhere:
  - dod: test ext +0.0019;
  - Kalk and Sv: accent worse;
  - Kal and Kho, which keep final k/ɣ: +0.010 to +0.033.

**dod:**
- Final -a > -ō (after clusters) was a `@morph` rule. Retagged as @vowel-shift: verb −0.011
  train, −0.025 test.

### Rejected (Δ train, or the gate that failed)

**Sh:**
- a > o/ō before a nasal cluster or rṇ: ext +0.003 to +0.008. The a→o/ō confusions (≈160)
  are real but too inconsistent across dialects.
- Keeping ṇ (dropping ṇ > n): +0.0025.
- Final -m loss: plain +0.0045.
- ava > ō: plain +0.0007 to +0.0012.
- Medial ai > ē: +0.0007.
- -ṣ/-ś and -z/-s alternations: source dialects, not tried.

**Kho:**
- l > ḷ after back vowels: +0.0056 / +0.0078.
- ṣ > ś generally: +0.0026. Bashir's ś is a notation or dialect split, already partly handled
  in a %source block.
- oo > o, ei > e: 0 / +0.003.

**K**, all rejected:
- final t ṭ k p > aspirated after a former geminate: plain +0.005 to +0.010;
- m > n finally: plain +0.0058, though verb −0.020;
- final-syllable ā > ɔ: +0.0026;
- u > o before final l: plain +0.0012.

K is otherwise saturated. Its remaining a→o/u/∅ confusions follow the unscored matras and
are lexical.

**Phal:**
- First-syllable a > ē before a final -i (umlaut). Train ext −0.005, but it breaks the -ij-
  verbs (tapíja, lajíja): test verb +0.0061.
- Length before -u/-o: accepted on train, dropped with the umlaut package.
- Dropping r̩ > ri: +0.0008.
- a > ū before a final m: test accent worse.

**Kal:**
- Final -ak > -a, the biggest single ending class (155 k→∅): plain +0.0004 to +0.0009,
  accent +0.0002 to +0.0021.
- aya > e: test ext and verb worse.
- l > v finally: +0.0045.
- ṇ kept before retroflex stops: +0.0004.
- j- > ź-: accent +0.0027.

**Small lects:**
- dod: -aka > -ō (ext +0.0044); final m loss (test ext +0.0128).
- pog: final m loss (plain +0.0022).
- Kund: final Cr > C, m loss (accent worse).
- Sv: ao > o (±0).
- -aka retag in Sv, Ush and Kund: neutral (they have no verbs), so it fails the
  must-improve gate.

### Engine bug found, and menu notes (reported, not fixed)

**`evaluate.verb_form` turns ś into s.** It builds the root with `sca._deacc`, which strips the
acute of ś as if it were an accent (`_deacc('śumbʰáti') == 'sumbʰati'`). Every verb-menu form
of a ś-verb therefore starts from s. This shows in Kho s→ś [#_V] (25 pairs: śúcyati >
sucik | śicik) and in every lect with ś. The fix belongs in the engine: strip only vowel
accents.

**Sh -oiki (CDIAL/Kohistani).** 109 missing -i̯ and 102 missing -o- still come from verbs, which
the chain can only reach as -ik (from -ikka). No menu form gives -oiki. The causative -y-
(cʌpyōnu, unyóiki) is also unreachable, because `verb_form` strips -ay-: 75 ∅→y. A causative
-ayanaka form would be the item to test, with a placebo.

### What remains

Endings and mislinks, as above. Then:
- Sh vowel length: ā→a 148, a→ā 73. This is dialect/source length marking that no single
  shortening rule captures without breaking the Schmidt tone block.
- The Khowar harmony residue: a→o 132.
- Kal -ak vs -a.

Not done beyond the batch tests: Sv (no candidate passed), K (saturated).

Other rejected candidates:
- Sh: changes to Backstrom's length block (removing it +0.016; open syllables only +0.006).
- Kho: harmony extended to non-initial syllables (+0.004).
- pog: ś- > h- (+0.009).
- Sv: s + stop loss (+0.003).

### Round 6 second pass

**Re-baseline after the engine fixes** (`verb_form` keeps ś; ˈ ˌ stripped). Verb-menu NED,
train / test:

| lect | verb-menu NED train / test |
|---|---|
| Sh | 0.4315 / 0.3322 |
| Kho | 0.3565 / 0.3489 |
| Phal | 0.4284 / 0.3867 |
| Kal | 0.3376 / 0.4183 |
| K | 0.2400 / 0.2020 |
| Kalk | 0.4408 / 0.3889 |
| dod | 0.3541 / 0.4821 |
| pog | 0.2636 / 0.4611 |
| Sv | 0.2669 (10 pairs) |

The Kho s→ś class (25 pairs) is gone.

**1. Kashmiri, now improved.** The -anaka infinitive -un missed after a cluster (níndati >
nɛndun, trúmpati > trɔmbun), because the final-syllable umlaut `a > u / V C _ C #` required a
single consonant.
- Widened to `a > u / V C (C) _ C #`.
- Gates vs the round-6 base:
  - ext 0.3561 → 0.3537 train, 0.3624 → 0.3612 test;
  - verb 0.2400 → 0.2290 train, 0.2020 → 0.1919 test;
  - plain −0.0008 / −0.0007;
  - exact 16.1 → 16.4%.
  ACCEPT.
- Rejected: `ō > ū` blocked before CV, for verb -ōrun (verb −0.0026 but plain +0.0012).

**Sauji:** still nothing passes. It has only 10 train verb pairs. aa > ā gives ext +0.0019;
final-k loss makes accent worse.

**2. Ushojo +0.011 test plain.** This was not the bare-vs-extended artifact. It came from one
test etymon, ajá (a → ā | aī): a word-final aa contracted in a now-monosyllabic word. On train,
náva got worse the same way. Narrowed to `a a > ā / _ C`, which only contracts before a
consonant. Against the round-6 base:
- ext train −0.0085, test −0.0046 (unchanged from pass 1);
- plain train −0.0062 (now a gain);
- plain test ±0.

**3. Second pass on Sh, Kho, K, Phal.** Verbs: winning menu form vs attested
(`tmp/sound_changes/dardic-north/vwin.py`).
- **Sh.** The residue is -ik (from -ikka) vs CDIAL -oiki(i̯) (≈50 pairs), and -ōnu vs causative
  -yōnu (14). Neither is reachable by sound change: there is no menu form with -oi-, and
  `verb_form` strips -ay-.
- **Kho.** 43 near-misses are verb-root a vs o (karik | korik, kasik | kosik): the a-harmony
  sees OIA -ati, but the -ikka form has no following a.
  - `a > o / # (C)(C) _ C (C) i k k a #` gives verb −0.0165 and ext −0.0044, plain ±0.
  - Rejected as a morphological condition in disguise: it names the suffix, and harmony across
    an intervening i is not a plausible sound change.
  - Front dissimilation extended to ī / -ik: ±0.
- **Phal.** The -e/-ō̂/-ū̂/-ī̂ variation of Liljegren's forms is person/gender morphology, left
  alone.

Near-misses, NED ≤ 0.25 (`near.py`):

**Sh**
- ∅→r (19: Cr kept or metathesised, e.g. cakrá > cʌrko): dialectal.
- a↔ā (34): length marking by source.
- n→ṇ (9): dropping ṇ > n was already +0.0025.
- ṣ→ẓ (7): `ṣ > ẓ / V _ V` gives +0.0027.
- t→tʰ (7: *uttʰala > utʰalo). The cause is an engine quirk: `C $1 > $1` ignores aspiration, so
  it turns ttʰ into t.
  - Putting `Stop {kʰ cʰ ṭʰ tʰ pʰ} > $2` first is worse: Sh +0.0006 plain; Sv +0.0025 plain,
    +0.0135 accent.
  - So the data prefers geminate-aspirate deaspiration, and the quirk stays as a de-facto
    sound change.

**Phal**
- Final -o vs ∅ (19) and ∅→u (17): gender endings.
- a↔ā/ā̂ (27): length and tone.

**Kho**
- ∅→e before -ik (15): this comes from causatives, whose -ay- the menu strips.
- i→e (14): see above.

Kept this pass:
- K: the umlaut widening.
- Ush: the narrowed aa > ā.

Not done: Sv (no candidate passes); Kal, Kalk, Kund, dod and pog had no second pass.

### Round 6 third pass (Kal, Kalk, Kund, dod, pog; causative formations)

Near-misses, verbs and non-ending classes (`near.py`, `vwin.py`, `trv.py`, `gates.py`).

**Kept: Poguli.** `a > u / V C (C) _ C #`, the Kashmiri final-syllable umlaut (paśy-anaka >
paśun, zapun). Gates vs pass 2:
- ext −0.0057 train, −0.0032 test;
- verb −0.0283 train, ±0 test;
- plain ±0.

ACCEPT.

**Rejected:**
- **Kal**
  - Final nd > n: train ext −0.0007, but test verb +0.0044 and test ext +0.0006.
  - l > ∅ / a _ u#: plain +0.0003.
  - {l ḷ} > v / a _ #: +0.0024.
  - Remaining classes, lexical or source-bound: -ak vs -a (35), length (17), -ék causatives
    (9, see below).
- **Kalk**: final Cr > C, plain +0.0012.
- **dod**: the Poguli umlaut, ext +0.0045. The -ō vs ∅ near-misses (dantō | dant) are lexical.
- **Kund**: each near-miss pair occurs once; nothing systematic.

**Causative formations (placebo test).** `vplacebo.py` monkeypatches `evaluate.VERB_MENU` in a
scratch process; the engine is untouched. Each candidate is compared with a same-shape suffix
whose consonant (r) survives. Δ is verb-menu NED, with the number of pairs that pick the item:

| lect | -ayikka (caus. -aya- + -ikka) | placebo -arikka | -āpayikka (caus. -āpaya- + -ikka) | placebo -āparikka |
|---|---|---|---|---|
| Kho | train −0.0207 (105 picks), test −0.0223 (27) | −0.0105 (70), −0.0150 (22) | −0.0140 (55), −0.0254 (23) | −0.0034 (17), −0.0104 (10) |
| Sh | 0 (0) | −0.0082 (28), −0.0075 (4) | −0.0122 (46), −0.0117 (6) | −0.0079 (27), −0.0075 (4) |
| Kal | −0.0028 (3), 0 | −0.0052 (9), −0.0022 (1) | −0.0035 (7), −0.0022 (1) | −0.0026 (6), 0 |

- **Khowar -eik = causative -aya- + -ikka.** The causative items beat their placebos by about
  2× (train) and 1.5–2.5× (test), so this is a real formation. Recommended VERB_MENU additions:
  **-ayikka**, and **-āpayikka** for the -āpaya- causatives.
- **Shina.** -āpayikka beats its placebo modestly (−0.0122 vs −0.0079). It gives a partial
  route to the -oiki of causatives (-āvik > -oik). Plain -ayikka does nothing, because Shina
  loses intervocalic y.
- **Shina -yōnu.** Adding -ayanaka picks nothing: the chain's y-loss makes it identical to
  -anaka. So Shina -yōnu (cʌpyōnu) would need a y-retention rule that the noun data do not
  support.
- **Kalasha -ék** does not beat the placebo: no evidence.

Not done: nothing further in these five lects.

### Round 6 third pass (Sh, Kho, K, Phal, Kal): rule audit and lect-specific verb formations

**Rule audit** (`tmp/sound_changes/dardic-north/loo4.py`). Each chain line is dropped in turn,
and the change in train plain/accent/ext/verb is recorded (output in `loo4.out`).

- **Kal, Phal, Kho**: no rule whose removal passes the gates. Every rule has positive ext or
  plain/accent value.
  - Rules with near-zero ext value survive on accent or plain: the Kal NC+a rounding
    (ext +0.0015, plain +0.0002) and the accent-placement lines (ext 0, accent +0.036).
- **Sh**: one drop candidate, `ā > ∅ / _ ō` from pass 1. After the later `o > u / ō n _ #`,
  removing it gives verb −0.0012 train and −0.0032 test, but test ext +0.0002. REJECT; kept.
- **K**: not run. One K pass takes more than an hour.

Near-misses were grouped by single differing pair in passes 2 and 3 (`near.py`). The conditioned
candidates found there are listed above. None of the remaining single-pair groups in these five
lects has a phonological conditioning that passes the gates.

**Lect-specific verb formations** (`%verb-menu`). Each is placebo-tested with a same-shape suffix
whose consonant survives, via `gates.py`. Admitted only where real beats placebo by more than
0.002 on train and is also better on test.

| lect | item | real Δverb train / test | placebo | placebo Δverb train / test | installed |
|---|---|---|---|---|---|
| Kho | +ayikka +āpayikka (causative -aya-/-āpaya- + -ikka > -eik) | **−0.0226 / −0.0320**; ext −0.0062 / −0.0080 | +arikka +āparikka | −0.0105 / −0.0150 | ✓ |
| Sh | +āpayikka (caus. -āpaya- + -ikka, partial source of -oiki) | **−0.0122 / −0.0117** | +āparikka | −0.0079 / −0.0075 | ✓ |
| Phal | +aka (root + -aka; coordinator's run) | **−0.0044 / −0.0032** | +ara | ±0 / ±0 | ✓ |
| Kal | -ayikka / -āpayikka | — | — | does not beat the placebo (pass 3 table) | ✗ |

Plain and accent are unchanged by all three directives.

**Final verb-menu NED** (train / test):
- Kho 0.3565 → 0.3339 / 0.3489 → 0.3169
- Sh 0.4315 → 0.4193 / 0.3322 → 0.3206
- Phal 0.4284 → 0.4240 / 0.3867 → 0.3835

### Round 6 fourth pass (Kashmiri audit, Kashmiri near-misses, Sauji)

**Kashmiri near-misses** (NED ≤ 0.25, by differing segment pair; `near.py`).

Kept, gated together as ACCEPT:
- `ṁ > ∅ / _ {ś ṣ s h}`. Anusvāra is lost before a sibilant instead of becoming n
  (pāṁśu > pāh, māṁsá > mās, and the -viṁśati numerals).
- The first version also had `{ś ṣ} y > ś` (náśyati > naśun). Its ṣy half broke pakṣyà >
  pocʰ, which put test plain at +0.0008, so it was narrowed to `ś y > ś`. The audit then showed
  that rule had zero effect, and it was dropped.
- Net vs the pass-3 file:
  - plain −0.0010 train, +0.0001 test;
  - ext −0.0014 / −0.0001;
  - verb −0.0005 / 0;
  - accent −0.0056 train (K has no test accent pairs).

Rejected:
- Final nd/mb > n/m: accent +0.083 train, plain +0.0018.
- nʦ > nz: no effect.
- Cr > C after velars: +0.0009.
- Dropping r̩ > ri: +0.0003.

Left as lexical or morphological:
- The u→a / a→o / ∅→a/i endings. The final-syllable umlaut over- and under-applies noun by
  noun (andur | andar vs kɔkur).
- The ō/ū alternation in verb roots.

**Kashmiri rule audit** (`loo4.py K`, drop one rule at a time on train; `loo4K.out`). Two
candidates came out:
- `use hiatus.a.postvocalic`: ext −0.0025, but plain +0.0016 train, +0.0038 test, and test verb
  +0.0029. REJECT.
- The no-op `ś y > ś`, removed.

Every other rule has positive value.

Two further observations from the audit, not acted on:
- **K accent.** It is scored only on a few train pairs (0.5417). Removing `V > $1-acc` or the
  citation rule "improves" it (−0.028 / −0.040), because visible Vedic acutes then match stress
  marks. This is the carried-accent artifact, not a signal.
- **K affrication and s + stop aspiration** carry accent value (+0.083 / −0.111) only through
  those same few pairs.

**Sauji, one more look.**
- `a > ā / # C* _ C {u o} #` (gʰaná-uka > gānu, cʰagalá > cʰālu): ext −0.0046 train, −0.0058
  test, but test accent +0.032 (14 pairs). REJECT.
- The broader `_ C V #`: train accent +0.0062.
- Final nd > n: plain +0.0014.

Sauji stays unchanged.

### Round 6: re-gating on the extension-aware accent (ext_acc)

`gates.py` now gates on ext_acc: the accent scored on the variant the extension-aware score
picks. Changes previously rejected only (or mainly) on the bare-headword accent were re-gated.

**Now installed:**
- **Sh.** The broad pretonic shortening `{ā ī ū ē ō} > short / _ C* V C* V`, before apocope,
  replaces the narrower pre-pretonic rule of pass 1.
  - ext −0.0051 train, −0.0025 test.
  - verb −0.0095 / −0.0058.
  - ext_acc −0.0092 / −0.0026.
  - plain −0.0015 train, +0.0009 test.
  - The bare accent is +0.0011 on test: long vowels that Schmidt's block marks lose their
    circumflex on bare headwords.

  ACCEPT.
- **Kalk.** `k > ∅ / V _ {a ā} #`: suffixal -aka/-ikā/-uka lose their k.
  - ext −0.0033 train, −0.0009 test.
  - ext_acc ±0.
  - plain +0.0011 train (within the 0.002 limit), +0.0030 test.
  - The bare accent is +0.0067: bare -ka headwords lose k. This is the bare-vs-extended
    artifact, since the attested forms are suffixed.

  ACCEPT.

**Still rejected:**
- Sv first-syllable lengthening before -u/-o: ext_acc +0.0029 train, +0.0179 test.
- Sv lengthening before any final vowel: ext_acc +0.0007 / +0.0071, plain +0.0012.
- Kal final -ak > -a in 3+ syllables: test ext +0.0039, test verb +0.0027.
- Kho śr/sr > ṣ: ext +0.0001.
- Sh generalised -o > -u after an o-syllable: ext +0.0004, ext_acc +0.0020.
- K final nd/mb > n/m: still fails plain (+0.0018), not re-gated.

## Round 7

Scratch: `tmp/sound_changes/dardic-north/r7/`.
- Snapshot at the start of this pass: `r7start/`.
- Tools:
  - `vq.py` / `vx.py`: vowel confusions and identity cross-tabs on the ext-chosen variant, with the
    final mātrā and the source.
  - `tonex.py` / `tonex2.py`: attested tone class against the OIA features.
  - `tonediag.py`: tone position and class errors per source.
  - `tonenet.py`: per-pair tone deltas.
  - `mktone.py`: appends per-source tone blocks.
- Every number below is on the current engine, after the fixes of 08:20–09:40.

### Summary (ext NED train / test; ext_tone train / test)

"Before" is `r7start/` re-scored on the engine as of 10:40; "after" is the installed file.

| lect | ext train | ext test | verb train / test | ext_tone train | ext_tone test |
|---|---|---|---|---|---|
| K | 0.3774 → **0.3211** | 0.3868 → **0.3284** | 0.2501 → 0.2356 / 0.2138 → 0.1855 | — | — |
| Sh | 0.4232 = | 0.4392 = | = | 0.4592 → **0.4534** | 0.4700 → 0.4622 |
| Phal | 0.3978 → 0.3976 | 0.4124 → 0.4117 | = | 0.4574 → **0.4556** | 0.4830 → 0.4823 |
| Kho | 0.3718 → **0.3708** | 0.3944 → 0.3941 | 0.3408 → 0.3381 / = | 0.4121 → 0.4117 | 0.4443 = |
| Kal (from `r7base/`) | 0.4278 → 0.4273 | 0.4404 = | 0.3537 → 0.3514 / = | 0.4659 → 0.4642 | 0.5218 = |
| Kalk | 0.4176 → 0.4166 | 0.4919 = | = | 0.4816 → **0.4657** | 0.5819 → 0.5722 |
| Kund | 0.4097 = | 0.4365 = | = | 0.5350 → **0.5258** | 0.7812 = |
| Sv | 0.4262 → **0.4233** | 0.4153 = | 0.4810 → 0.4589 / = | — | — |
| Ush | 0.3531 → 0.3512 | 0.3880 = | 0.4042 → 0.3884 / — | — | — |
| dod, pog | = | = | = | — | — |

Notes on the table:
- K plain NED: 0.4147 → 0.4034 train, 0.4189 → 0.4035 test; exact 13.6 → 15.2% / 13.7 → 17.5%.
- The Kal ext_acc is higher (0.4282 → 0.4363 train, 0.4547 → 0.4669 test) because the morning
  stress change was reverted; see §0. Against `r7base/` it is unchanged.
- Plain rises in Phal, Kund, dod, pog and Ush from the neutral removals of @morph rules; see
  the legacy @morph section.
- Every installed change passed `gates.py` when installed: ACCEPT, or ACCEPT (prosody-only) for
  the tone blocks. Two exceptions:
  - the K umlaut conditioning, ACCEPT-IF-JUSTIFIED (below);
  - the neutral @morph removals, which only change plain.

### 0. Changes installed earlier this round (07:46–08:11, before the scoring change)

They were re-gated on the current scoring, from `r7base/` to `r7start/`:

| lect | change | Δ | verdict |
|---|---|---|---|
| Phal | ṭʰ/tʰ deaspiration after a sibilant; stress on the (last) long vowel | ext −0.0005 / −0.0018; ext_acc −0.0072 / −0.0138 | ACCEPT |
| Sv | stress on the last syllable | ext_acc −0.042 / −0.048 | ACCEPT |
| Kund | stress on the last syllable | ext_acc −0.056 / −0.063 | ACCEPT |
| Kalk | śr > ṣ | ext test −0.0063 | ACCEPT |
| Ush | ṭʰ deaspiration | ext −0.0031 / −0.0056 | ACCEPT |
| Kal | first-long-vowel/else first-syllable stress | ext_acc −0.0091 / −0.0122, but test ext_tone +0.0053 | REJECT, reverted |

The Kal stress change was reverted to the round-3 penult. It lost on the tone-scored CDIAL Kal.
subset (50 test pairs): 12 worse, 9 better, e.g. *matsiya maʦī́, pastíya pastī́, *pipʰīla
pʰīvilík. In these the attested stress is on a long vowel from contraction, which the
first-syllable rule misses.

The variant "first long vowel, else penult" fails on train (ext_tone +0.0006).

### (a) Vowel qualities

**Kashmiri (all installed; each step gated in turn, then cumulatively).** The CDIAL K. vowel
system is regular once the mātrās are modelled. The mātrās are unscored, but they condition the
vowel before them.

1. **Short high vowels lower to open-mid in closed syllables.** `{i u} > {ɛ ɔ} / _ C C`
   replaces {e o}.
   - Examples: níndati > nɛndun, kukṣí > kɔcʰ, kuṇḍá > kɔnḍ.
   - Counts: e→ɛ 77 against e→i 31 and e→e 7; o→ɔ 72 against o→o 11 outside the -ᵘ words.
   - ext −0.0074 / −0.0096.
2. **The first-syllable umlaut i > ɛ before a.** siddʰa > sɛd-, kiráṇa > kɛran. ext −0.0028 / −0.0014.
3. **The -aka mātrā.** The masculine -aka, its k lost, gives -au > -ᵘ (`a > u / C a _ #`, before
   the hiatus contraction). Before this latent u:
   - root ā > ɔ;
   - a first-syllable a > o;
   - ɛ > yo.

   Examples: kāla-ka > kɔlᵘ, accʰa > oʦʰᵘ, kʰaḍaka > kʰorᵘ, cʰinná > cʰyonᵘ, nímba > nyombᵘ.
   - The a > o umlaut is first-syllable only. With a non-initial a it gave the -anaka infinitive
     -on (verb +0.0041), so the non-initial a goes to u instead (point 6).
   - ext −0.0145 / −0.0147, then ɛ > yo −0.0014 / −0.0022.
   - ɔ > o before -ᵘ was rejected: plain +0.0014, test ext +0.0029.
4. **The -ikā mātrā.** Feminine -ikā > -iā > -ᵘ̈ (`i ā > ɨ / C _ #`). Before it:
   - ā > ə̄;
   - a > ü (scored u);
   - ɔ > ə̄̆.

   Examples: ā́rā > ə̄rᵘ̈, kapā́la > kvə̄jᵘ̈, paṭṭa > püṭᵘ̈, tumba > tə̄̆mbᵘ̈.
   - ext −0.0080 / −0.0061 (82 better, 0 worse), then ɔ > ə̄̆ −0.0009 / −0.0010.
5. **v-prothesis before ɔ.** Its trigger set lacked the new ɔ (újjāyatē > vuzun,
   ujjvala > vɔzul). ext −0.0019 / −0.0022, verb −0.0047 / −0.0110.
6. **The final-syllable umlaut a > u is now conditioned on the -ᵘ.** It becomes
   `a > u / V C (C) _ C (C) u #`, before apocope, and replaces the unconditioned `umlaut.vccc`
   after apocope.
   - Bare -a nouns keep a: ā́sana > āsan, ántara > andar, mārjana > māzan, sāhasa > sāhas.
   - -aka forms and -anaka infinitives get u: akṣára-ka > acʰur, cal-anaka > ʦalun.
   - ext −0.0134 / −0.0137, verb −0.0008 / −0.0007.
   - Plain is +0.0033 / +0.0021: **ACCEPT-IF-JUSTIFIED.** This is the bare-vs-extended artifact.
     - The 146 bare headwords that got worse are almost all attested with the -ᵘ/-ᵘ̈ mātrā that
       the -aka/-ikā variant supplies (kɔtʰulᵘ, zɔzurᵘ, kanguvᵘ, ōrukᵘ, muhüjᵘ̈).
     - Their ext score improves: 304 reflexes better, 30 worse.
7. **Prefix contractions.**
   - ava- > ɔ- (+ v-): *avastarati > vɔtʰarun, *avaśapati > vɔhavun. ext −0.0005 / ±0.
   - upa- > ɔ- and upā- > vā-: *upadʰāra > vɔrᵘ, upākʰyāna > vakʰonᵘ. ext −0.0005 / −0.0013,
     verb test −0.0027.
8. **The v-prothesis moved before the umlaut rules.** An umlaut o- (accʰa > oʦʰᵘ, agryá > ogᵘ,
   aruja > orᵘ) no longer gets v-. ext −0.0010 / −0.0010.
9. **i > yu before -ᵘ**, the partner of ɛ > yo: antimá-ka > andyumᵘ, dákṣiṇa-ka > dacʰyunᵘ.
   ext −0.0012 / −0.0016.
10. **Consonants before the -ᵘ̈ mātrā.**
   - n, l palatalise: tanikā > tüñᵘ̈, jā́la-ikā > zə̄jᵘ̈, kīla > kījᵘ̈. ext −0.0019 / −0.0013.
   - ś/ṣ stay ś (the ś > h rule treats ɨ as front): kaṣá-ikā > küśᵘ̈. ext −0.0009 / −0.0023.

Rejected in K:
- ND > N (nd ndʰ mb mbʰ > n m before the NT > ND voicing: lamba > lam, klíndati > kɛnun, but
  jambú > zumb, ḍimba > ḍɛmb): 47 better, 55 worse; ext +0.0007 / +0.0012. See (c).
- Non-initial a > o before -ᵘ except before n (anzolᵘ, arosᵘ; 40 pairs): ext +0.0068 / +0.0085.
  u is the majority there.
- Medial syncope `a > ∅ / V C _ C V` (the reduced ᵃ of ǝndᵃra): before apocope, test verb
  +0.0099; after it, ext +0.0020.
- Palula-style ā́ > ō in Sh (nā́man > nōm): ext +0.0024 / +0.0064.
- The Poguli analogues of points 3 and 6: pog keeps its -u and -nu. Unconditioned umlaut: verb
  +0.020. First-syllable a > o: +0.009.

**Sh.** The lax-vowel error is CDIAL's ʌ against a.
- CDIAL writes both: σ1 a 192 against ʌ 153; closed monosyllables a 45 against ʌ 40.
- ʌ is commoner in the unmarked, i.e. unaccented, Gilgiti-type entries (193 against 148) and
  rarer in the stress-marked ones (75 against 83). So it is a sub-source split inside the CDIAL
  key, not a conditioned change.
- A CDIAL-block `a > ʌ / # C* _` makes test ext +0.0047. Rejected.
- The ɛ of Backstrom and CDIAL is scattered: -ɛ endings, gr̩hín > gɛn. Nothing systematic.

**Kho, Kal.** Lax vowels are ≤ 3% of the error; nothing systematic.
- Kho: CDIAL ʌ 45, ɛ 25, ə 21.
- Kal: CDIAL ʌ 24, ɔ 14, æ 12.

### (b) Tone: does the Vedic accent predict tone type?

Cross-tabs are on train reflexes with a tone mark, for the long vowel of reflex σ1. "rel" is
where the Vedic accent was relative to the toned syllable (`tonex2.py`).

**Shina, Schmidt (ˆ falling, ˇ rising, ´ high).**

| condition | falling | rising |
|---|---|---|
| Vedic on the syllable (same) | 75% | — |
| accent one syllable later (+1) | 74% | — |
| accent two syllables later (+2) | 60% | — |
| unaccented headword | 59% | — |
| OIA long vowel before a single kept consonant or a cluster | 91–93% | — |
| long from hiatus contraction or secondary lengthening | — | 48% (against 16–22%) |

So in Schmidt the tone type follows the **origin of length**, not the Vedic accent: inherited
long vowels are falling, contracted ones are rising. The one case where the Vedic accent helps
(hiatus, rel = same: 39% rising against 65% for unaccented headwords) is small.

**Shina, Degener and Buddruss (` against ´).** Their grave is Schmidt's falling on stressed long
vowels (65%).
- It depends on the Vedic accent: 85% grave when the Vedic accent was on that syllable, 44%
  when it was on the next.
- The +1 case with a single consonant is 92% acute (kālá-type). Oxytone apocope usually deletes
  that Vedic vowel, though.
- A chain flag that keeps it (below) raised acc on Sh (+0.0022) and was not pursued.

**Shina, Backstrom.** A caron on 93% of accented long vowels: a notation.

**Palula, Liljegren (ˆ against ˇ).** Falling is the default: 83% on long σ1 vowels.
- Rising grows with the distance of the Vedic accent to the right: same 10%, +1 23–25%, +2 64%,
  unaccented headword 20%.
- In monosyllabic reflexes of oxytones (kālá > kā̌l, adyá > ā̌j, kṣurá > ʦ̣ʰū̌r) rising reaches
  30%.
- Strand (´ against ˇ): same 2%, +1 19%, +2 29%.
- Aspirates matter little: rising 16% with an initial aspirate against 11% without.

So in Palula **the Vedic accent does predict tone type**. Where the old accent stood to the
right of the modern tonic syllable (retracted stress), rising tone is 2–6× commoner than on the
old accented syllable. It is the majority only at +2 (n = 11).

**Kho (Bashir).** The caron occurs only on long vowels, and every accented long vowel has it.
It marks length plus accent, not a tone type. The chain predicts no long tonic vowels there, so
a Bashir caron block has no effect.

**Installed (prosody-only; NED, verb and ext_acc unchanged; ext_tone better on train and not
worse on test).** The tone blocks are appended as `%source` blocks after every segmental rule:
`Stress > $1+acc`, `Vedic > $1-acc`, then a map on acute long vowels.

| lect | block | ext_tone train / test |
|---|---|---|
| Sh | Degener/Buddruss: stressed long → grave | —, see below |
| Sh | Backstrom: stressed long → caron | —, see below |
| Sh | Schmidt: ō̂ > ō̌ before final -n(V), the contracted -ōn(u) (kʰā́dati > kʰō̌n) | −0.0010 / −0.0009 |
| Sh | all three together | −0.0058 / −0.0079 |
| Phal | Liljegren: falling on every stressed vowel (the chain's short tonic vowels are long falling there: aṣṭā́ > ā̂ṣṭ, udaká > vī̂) | −0.0016 / −0.0005 |
| Phal | **Vedic-conditioned rising**, below | −0.0002 / −0.0002 |
| Kalk | Hultman: stressed long → caron (10 of 11) | −0.0164 / −0.0097 |
| Kund | source kund: stressed → caron | −0.0092 / ±0 |

The Phal Vedic-conditioned rising works like this:
1. Before apocope, a stressed vowel with the Vedic accent on a later syllable is flagged:
   `Stress > $1+acc / _ !#* Vedic`; the other Vedic latents are dropped; then `Acc > $1-acc`
   leaves the flagged vowel with a latent Vedic mark.
2. Liljegren's block writes the flagged vowel rising (caron); the other sources' block restores
   it as the stress.
3. Segmentally neutral; acc/ext_acc unchanged.
4. Variants, against this: Vedic vowel word-final only ±0; +2 only +0.0002; adjacent syllable
   only ±0.

**Rejected tone candidates:**
- Sh: Schmidt stressed-only falling instead of all long vowels: +0.0002.
- Sh: Schmidt rising after a preceding Vedic vowel: +0.0004.
- Sh: Degener with the Vedic flag (+1 → acute): acc +0.0022. The flag was not segmentally clean
  in the Sh chain.
- Sh: Degener "not before a surviving Vedic vowel": +0.0001 against the plain block.
- Kho: Bashir caron block: no effect.
- Kund: long only: no effect.

**Finding.** The Vedic accent predicts tone type in Palula: rising where it stood after the
modern tonic syllable. It predicts it weakly in Degener's Shina: grave on the Vedic syllable.
It does not predict it in Schmidt's Shina, whose rising/falling split follows the origin of
length (contraction → rising). Only in Palula does a Vedic-conditioned rule pass the gates, by
−0.0002. The bulk of the tone gain is source notation of the default tone on long vowels.

### (c) Open items

- **K nd/mb.** The split is not by voicing alone (see (a)). It stays lexical or dialectal.
- **Kal -ak.** The -k → ∅ class (35 pairs: átra-ka > atra) is unchanged. Earlier rejections
  stand (round 6); not re-tried beyond the residual check.
- **Sh -oiki / -yōnu.** Still unreachable. `verb_form` strips -ay-, and no menu form carries
  -oi-. The -oiki of non-causatives has no OIA source in the menu.
- **Sauji.** The residuals are endings (48%: -ō/-e/-ē gender forms, -ān verbs) and a→ā length
  (53 pairs). Round-6 lengthening candidates fail the accent gate. Nothing new passed.

### Consistency (focus 3)

- The ava- contraction is now a local rule in four lects. Kho, Kal and Phal share the identical
  pair `{a á} v {a á} > u / # _ {n m s ś ṣ}` and `> o / # _`; K has `> ɔ / # _`.
  - Kho: avarati > oréik, *avasnāyati > usnéik. ext −0.0009 / −0.0004, verb −0.0023.
  - Kal: ext −0.0005, verb −0.0022.
  - Phal: ext −0.0002 / −0.0007.
  - K: see (a).
  - Rejected in Sh (test verb +0.0018), Kalk (+0.0014 verb), Sv, Ush, Kund and dod (±0).
- **Kho: word-final mb > m.** ḍimba > ḍim, *kṣumbati > ʦ̣ʰum. ext −0.0007 / ±0, ACCEPT.
  - Kho ii > ī (dʰīyatē > dī̌k) was rejected: ext +0.0008, verb +0.0029.
  - The same rule in Kalk (ext −0.0010 / ±0), Ush (−0.0010 / ±0) and Kal (−0.0001 / ±0): installed.
    Rejected in Sv, Kund, dod, pog and Sh (no gain). In Phal it is
    rejected (test plain +0.0012).
- upa- > o-/vā- passes only in K. Rejected in Kho (+0.0002), Kal (ext_tone ±0, no ext gain),
  Phal (+0.0005) and Sh (+0.0006).

### Legacy @morph rules (focus 6)

Each chain-level @morph rule was dropped in turn (`r7/morph/loo.txt`).

**Removed (neutral removal: ext, verb, ext_acc and ext_tone ±0 on train and test):**

| lect | rule | plain Δ train / test |
|---|---|---|
| Sh | `if:verb` root-vowel shortening | −0.0005 / −0.0011 |
| Kund | `V t {i ē} > ∅` | +0.0065 / ±0 |
| Phal | `if:verb V t {i ē} > e` | +0.0033 / +0.0035 |
| dod | `V t {i ē} > ṇ ō` | +0.034 / +0.035 |
| pog | `if:verb V t {i ē} > n u` | +0.060 / +0.051 |
| Ush | `V t {i ē} > ō n` (after the 10:35 verb-flag fix; ext was +0.0040 before it) | +0.0054 / ±0 |

These are bare-headword citation rules only; the verb menu derives the citation forms.

**Kept (removal fails a gate; train ext Δ re-run after the 10:35 verb-flag fix, `r7/morph2/loo.txt`):**

| lect | rule | failing gate |
|---|---|---|
| Kho | causative -ayati > -ik | ext +0.0003 |
| Kho | -ati > -ik | ext +0.0009 |
| Sh | -ati > -oik | ext +0.0015 |
| Kalk | -ati > -ūn | ext +0.0019 |
| Kund | -aka > -o | ext +0.0037 |
| K | -ati > -un | ext +0.0002 |
| Kal | -ati > -ik | ext +0.0001; test ext_tone +0.0044 |
| Ush | -aka > -o | ext +0.0149 |
| Sv | -ati > -il | ext +0.0075 |
| Sv | -aka > -o | ext +0.0139 |
| dod | causative -ṇō | ext +0.0037 |

The -ati rules still fire on non-verb -ti/-tē headwords, which the ext score keeps.

**Retagged, not removed.** Ush, Kund and Sv each have `{a á} k {a á} > o / _ #`. Removing it costs
ext +0.004 to +0.015, so it stays. It is a sound change (-aka > -ao > -o, as `aka.aakaa.final` in
Sh), so it is retagged `@hiatus @aka`. Since it is no longer switched off for verbs, the verb
menu's -anaka now gives -ano (Sv dádāti > dāno, kʰā́dati > kʰāno; Ush krīṇā́ti > kīnōno):

| lect | ext train | verb train | test | verdict |
|---|---|---|---|---|
| Sv | −0.0029 | −0.0221 | ±0 | ACCEPT (the first Sauji change that passes) |
| Ush | −0.0009 | −0.0159 | ±0 | ACCEPT |
| Kund | ±0 | ±0 | ±0 | neutral retag |

(A first gate run showed ±0 everywhere only because the comment still contained the old tag
word, which the tag regex reads. Comments must not mention a tag they do not mean.)

Also tested for Phal: removing the %source stem rule `e > ∅ / C _ #` together with the chain
rule gives ext +0.0002. Kept.

### Not done / open after round 7

- **Kal stress placement.** The morning rule (first long vowel, else first syllable) is better
  for Trail & Cooper. It is worse on the tone-scored CDIAL Kal. subset, so it fails the tone
  gate. It needs a treatment of contraction-long final vowels (-iya > -ī́, -ík).
- **Sh CDIAL ʌ/a.** A sub-source split, not modellable per source key.
- **Kho long tonic vowels.** Bashir's ē̌/ō̌ from -aya-/-ava- contraction are not produced, so tone
  cannot be scored right there.
- **Sh -oiki.** Unreachable.
- **Kal -ak, K nd/mb, K ō/ū, K ś-/h-.** Lexical or dialectal.
- **The Vedic-flag mechanism in Sh.** It was not segmentally clean. Debugging it might show a
  Degener effect: 85% grave on the Vedic syllable.

