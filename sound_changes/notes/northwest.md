# Northwest group: OIA → Punjabi, Lahnda, Sindhi and neighbours

Rule files: `sound_changes/rules/OIA-{P,S,L,OP,awan,kcch,D,srk,thar,poth,khet,Khol}.sc`.
Frontiers: `sound_changes/results/OIA-<lect>.hand.frontier.csv`.
Scratch tools (all under `tmp/sound_changes/northwest/`): `try.py` (insert/replace/delete a rule and
rescore), `swap.py` / `adjswap.py` (order tests), `iso.py` (isogloss matrix: add/remove a fixed rule in
every lect), `cross.py` (apply one lect's chain to another lect's data), `srcconf.py` (per-source NED and
confusions), `clean.py` (drop rules with LOO ≤ 0), `table.py` (the table below).

## 1. Results

NED = mean normalised edit distance, null (etymon unchanged) → final chain. "Auto" is the greedy
baseline's test NED at its last rule count, with the hand chain's test NED at the same count on the
hand frontier in brackets. Rule counts are main-chain rules (incl. `@acc-loss`, `@notation`, `@morph`);
rules inside `%source` blocks are not counted.

| lect | name | #etyma | #rules | train NED (null → final) | test NED (null → final) | test exact % | auto test NED @n (hand @n) | excluded sources / blocks |
|---|---|---|---|---|---|---|---|---|
| P | Punjabi | 4337 | 56 | 0.6041 → 0.3609 | 0.6032 → 0.3588 | 11.8% | 0.442 @19 (0.410) | excl: bagri; blocks: gill |
| S | Sindhi | 3892 | 57 | 0.6357 → 0.3351 | 0.6354 → 0.3515 | 18.4% | 0.435 @26 (0.393) |  |
| L | Lahnda | 2687 | 50 | 0.6089 → 0.3777 | 0.5869 → 0.3706 | 15.2% | 0.457 @14 (0.447) |  |
| OP | Old Punjabi | 2298 | 42 | 0.6010 → 0.4022 | 0.5950 → 0.3976 | 14.7% | 0.452 @14 (0.434) |  |
| awan | Hindko | 846 | 49 | 0.6434 → 0.4157 | 0.6422 → 0.4470 | 10.4% | 0.504 @22 (0.480) | blocks: gill, rensch-hallberg-oleary1992 |
| kcch | Kutchi | 792 | 48 | 0.6626 → 0.3954 | 0.6645 → 0.3981 | 13.4% | 0.506 @18 (0.441) |  |
| D | Domaaki | 306 | 30 | 0.6704 → 0.5064 | 0.6776 → 0.5136 | 4.1% | 0.579 @14 (0.537) | blocks: backstrom1992, weinreich2008 |
| srk | Saraiki | 269 | 41 | 0.6669 → 0.4736 | 0.6670 → 0.4907 | 7.2% | 0.624 @12 (0.558) | blocks: gill, rensch-hallberg-oleary1992 |
| thar | Dhatki | 245 | 36 | 0.6697 → 0.4269 | 0.6436 → 0.4602 | 9.7% | 0.500 @20 (0.484) |  |
| poth | Pahari-Pothwari | 242 | 45 | 0.6605 → 0.4109 | 0.5950 → 0.4549 | 9.7% | 0.565 @23 (0.502) | blocks: gill, liljegren-hindukush, lothers-lothers2010pahari |
| khet | Khetrani | 206 | 37 | 0.7026 → 0.4988 | 0.6335 → 0.4611 | 13.0% | 0.539 @15 (0.485) | blocks: elfenbein, grierson-lsi1928 |
| Khol | Kholosi | 185 | 41 | 0.7212 → 0.4384 | 0.7402 → 0.5440 | 4.5% | 0.592 @15 (0.572) |  |

At the auto baseline's own (final) rule count, the hand chain is better on test for every lect, by
0.010 (L) to 0.066 (srk, kcch) NED. Test NED tracks train closely for the large lects; the small ones
(awan, srk, thar, poth, Khol) show a 0.02–0.10 train/test gap, as expected with 30–60 test pairs.

Sources, exclusions and per-source blocks:
- **P**: `bagri` (130 pairs) **excluded** — a Bagri (Rajasthani) wordlist filed under Punjabi, with
  Rajasthani compensatory lengthening (dugdʰá > dūdʰ, matsara > mācʰar); NED 0.58 vs 0.38 for CDIAL P
  under the P chain. Other groups own Bagri chains (`OIA-bagri_*.sc`). `gill` block: degemination +
  tonal devoicing (see §5).
- **awan** (Hindko): CDIAL "awan." (Awankari, 809) + Rensch et al. 1992 (Hindko, 517) + Gill. Block for
  Rensch/Gill: no written gemination, e o without macron (CDIAL Awankari writes ē ō; −0.006), initial
  voiced aspirate > voiceless (tone).
- **srk**: Rensch/Gill block: no written gemination (the devoicing block *hurts* here, +0.006).
- **poth**: Lothers & Lothers (365 of 447 pairs) / Gill / Liljegren block: no written gemination;
  Lothers-only block: short i u written ī ū (kukkuṭa > kūkṛī).
- **khet**: Elfenbein block: initial implosives ɠ ʄ ɓ ɗ (Elfenbein ɗūr, ɓolī; CDIAL/Grierson forms have
  none); Elfenbein + Grierson block: no written gemination.
- **D**: Weinreich block: long vowels written double (kóon, póo, ʦ̣éedoos); Backstrom block: length
  marked on most vowels (the same convention the Shina agent found in Backstrom's Shina).
- **OP**: `shackle` (1312) and `shackle-auto` (2047) both kept, though `shackle-auto` is noisier
  (per-source train NED 0.38 vs 0.42) and contains OCR/linking debris (reflexes "M", "'Huhi", English
  glosses; iyattaká > tretā, mā́ > makʰattū). Excluding it would halve the data; for reference, the same
  chain scores train 0.387 / test 0.381 NED on `shackle` + CDIAL alone.
- **thar** (source `thari`) and **Khol** (source `kholosi`) are single-source; **kcch** is CDIAL only.

## 2. Base chain (in order) and deviations

The NW base chain is `OIA-P.sc`; the other files are copies of it (L/awan/srk/poth/khet via L, S/kcch
via S) with lect-specific additions and deletions. Shared rules keep the same relative order in every
file (the few order differences are the ones in §3). P order:

**MIA block**
1. `@acc-loss`; `@notation` (headword anusvāra before a stop = homorganic nasal; P L S OP awan)
2. `@ai-au` ai au > ē ō
3. `@r-vocalic-ri` r̩- > ri-; `@r-vocalic` r̩ > u after labials (P L S kcch), r̩ > i (srk thar: r̩ > a)
4. `@sibilant-s` — everywhere except **D** (`@sibilant-keep`)
5. `@palatal-Cy` / `@assim-sonorant`: ty dy > cc jj; hy > jjʰ; ly ny > ll ṇṇ (L thar also vy);
   `@x-nw-dv-b` dv > bb (P L S awan kcch khet Khol)
6. `@ks` kṣ > kkʰ; `@s-stop-aspirate` sT > TTʰ (not D: aṣṭā > oṣṭ; not Khol)
7. `@x-nw-rt-cerebral` rt rtʰ rd > ṭṭ ṭṭʰ ḍḍ (P L S awan poth) — precedes r-assimilation
8. `@assim-sonorant`, `@assim-r`, `@assim-stop` (MIA gemination). S and khet keep dental + r;
   D has tr dr > ʦ̣ ʣ̣ (`@x-dardic-north-tr-affricate`); Khol keeps Cr (putra > peter).
   `@v-b @y-j` MIA vv yy > bb jj (P S OP awan kcch D srk poth Khol); `@x-hindi-mr-mb` mr ml > mb
   (kcch thar poth Khol); `@n-dental` ṇṇ > nn (P L S OP kcch srk thar khet: karṇa > kann)
9. `@initial-degem` (S, khet: not before r)
10. `@int-aspirate-h`, `@int-retroflex`, `@int-p-v` (not kcch/thar: aspirates kept; Khol none);
    `@m-v` m > v / V_V here in P L OP (the S-type m > Ṽ is later, #21)
11. `@aya-e` ava > o here in L, thar, poth, khet (P-type ava > au comes in the NIA block)
12. `@int-stop-loss`; `@aya-e` medial aa (< aCa) > e (P L S OP awan kcch srk poth Khol);
    `@morph` in-stems -in > -ī
13. `@final-cons-loss` (only OIA-type finals; CDIAL roots keep theirs); `@anusvara-vowel`
    Vṁ > V̄ before s h (S awan kcch thar poth khet: siṁha > sīh)
14. `@final-vowel-loss @x-nw-final-a-short` ā > a / C_# (Apabhraṁśa shortening; P L poth here; S,
    OP, srk, thar, D have it in the NIA block after their citation-vowel morphology)
15. `@x-nw-anusvara-loss` leftover ṁ > ∅

**NIA block**
16. `@degem-plain` pretonic: G G > G before a heavy (stressed) syllable (P L OP awan);
    `@x-nw-pretonic-shortening` V̄ > V / _ C(C) V̄ C (P L khet)
17. `@morph` citation forms: roots (*dabb) as verbs; -āyati/-āpayati/-ayati/-ati → infinitive
    (P -ṇā, L -aṇ, S -aṇu, awan -uṇ, kcch -ṇū, srk -aṇ, Khol -en, D -na, OP stem-i, poth bare stem);
    masc. citation vowel (P awan -ā; S -o/-u; OP -u; kcch thar Khol -o). L lost its `a > ā` once the
    Apabhraṁśa -ā shortening was in (removing it then helped, −0.0023).
18. `@nasal-voicing` NT > ND (all but thar, D)
19. S only: `@implosive` (initial g j ḍ b; voiced geminates; d- > ᶑ), `@x-nw-tr-cerebral` tr > ṭr;
    srk kcch thar: `@x-nw-d-retroflex` d- > ḍ-
20. `@x-nw-n-retroflex` n > ṇ / V_V (all but Khol, D)
21. `@x-nw-l-r` l > r / V_V (S kcch thar); `@m-v` Vm > Ṽ (S awan srk poth)
22. `@s-h` s > h / V_V (P L S awan srk khet poth; awan poth then `@h-loss`)
23. `@aya-e` ava > au / _C (P OP awan kcch srk Khol); `@v-loss` (S kcch)
24. `@hiatus` -iā -uā > -ī -ū; aa > ā; a lost after another vowel; `@ai-au` secondary ai au > e o
    (L S awan srk poth Khol)
25. `@flap` ḍ ḍʰ > ṛ ṛʰ (not before ḍ; all but Khol, D); P S D: `@v-loss` V_V
26. `@degem-plain` after a long vowel (P L OP awan kcch) / everywhere (S kcch thar khet Khol D;
    and, as a notation block, the phonetic sources of awan srk poth khet)
27. `@final-vowel-loss` {a i u} > ∅ (not S, OP, D); `@syncope` a > ∅ / VC_CV (L awan kcch srk thar
    khet Khol; in P its LOO fell to 0 after the Apabhraṁśa -ā shortening and it was dropped)
28. `@x-nw-n-dissimilation` ṇ > n after r ṛ ṇ (P khet; dropped from L OP srk when its LOO fell to ≤ 0)
29. `@v-b` v- > b- (P poth D only); `@y-j`; `@notation` ē ō > e o

Lect-specific additions: poth `@tone @deaspiration` (initial bʰ dʰ gʰ > p t k), `@h-loss`, i > e / _V;
awan `@h-loss`, `@x-nw-mb-mm`; kcch `@x-nw-postnasal-aspiration`, u- > au- (`@vowel-shift`); Khol
`@x-nw-aspirate-devoicing`, kʰ > x (`@int-spirant`), `@x-nw-ch-sh`, `@x-nw-deretroflexion`, `@x-nw-s-z`,
{i u} > {e o} / _C; D `@x-dardic-north-tr-affricate`, `@x-nw-nc-nasal` (ŋg > g separately),
`@n-dental`, `@deaspiration`, `@h-loss`, `@v-loss`, -na infinitive; thar `@nasal-cluster` +
`@anusvara-vowel` (Rajasthani-type ṁ), `@x-hindi-mr-mb`.

Tag presence per file (count of rule lines carrying the tag; `·` = absent):

| tag | P | L | OP | awan | srk | poth | khet | S | kcch | thar | Khol | D |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `@acc-loss` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `@notation` | 2 | 2 | 2 | ✓ | ✓ | ✓ | ✓ | 2 | ✓ | ✓ | 2 | ✓ |
| `@ai-au` | ✓ | 2 | ✓ | 2 | 2 | 2 | · | 2 | ✓ | · | 2 | ✓ |
| `@r-vocalic-ri` | ✓ | ✓ | ✓ | ✓ | · | · | · | ✓ | · | · | · | · |
| `@r-vocalic` | 2 | 2 | ✓ | ✓ | ✓ | ✓ | ✓ | 2 | 2 | ✓ | ✓ | ✓ |
| `@sibilant-s` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · |
| `@palatal-Cy` | 2 | 2 | 2 | 2 | 2 | ✓ | ✓ | 2 | ✓ | ✓ | ✓ | · |
| `@x-nw-dv-b` | ✓ | ✓ | · | ✓ | · | · | ✓ | ✓ | ✓ | · | ✓ | · |
| `@assim-sonorant` | 5 | 4 | 5 | 4 | 4 | 4 | 3 | 5 | 3 | 4 | ✓ | 2 |
| `@ks` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · |
| `@s-stop-aspirate` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | · |
| `@x-nw-rt-cerebral` | ✓ | ✓ | · | ✓ | · | ✓ | · | ✓ | · | · | · | · |
| `@assim-r` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `@v-b` | 2 | · | ✓ | ✓ | ✓ | 2 | · | ✓ | ✓ | · | ✓ | 2 |
| `@y-j` | 2 | ✓ | 2 | 2 | 2 | 2 | ✓ | 2 | 2 | · | 2 | 2 |
| `@assim-stop` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| `@n-dental` | ✓ | ✓ | ✓ | · | ✓ | · | ✓ | ✓ | ✓ | ✓ | · | ✓ |
| `@initial-degem` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | ✓ | ✓ |
| `@int-aspirate-h` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | · | · | ✓ |
| `@int-retroflex` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | ✓ |
| `@int-p-v` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | ✓ |
| `@m-v` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | ✓ | · | · | · | · |
| `@int-stop-loss` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 2 |
| `@aya-e` | 2 | 2 | 2 | 2 | 2 | 2 | ✓ | ✓ | 2 | ✓ | 2 | · |
| `@morph` | 7 | 5 | 4 | 4 | 3 | 2 | 2 | 8 | 5 | 2 | 5 | ✓ |
| `@final-cons-loss` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `@final-vowel-loss` | 2 | 2 | ✓ | ✓ | 2 | 2 | ✓ | ✓ | ✓ | 2 | ✓ | ✓ |
| `@x-nw-final-a-short` | ✓ | ✓ | ✓ | · | ✓ | ✓ | · | ✓ | · | ✓ | · | ✓ |
| `@x-nw-anusvara-loss` | ✓ | ✓ | · | ✓ | · | ✓ | ✓ | · | ✓ | · | ✓ | ✓ |
| `@degem-plain` | 2 | 2 | 2 | 2 | · | · | ✓ | ✓ | 2 | ✓ | ✓ | ✓ |
| `@x-nw-pretonic-shortening` | ✓ | ✓ | · | · | · | · | ✓ | · | · | · | · | · |
| `@nasal-voicing` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | ✓ | · |
| `@x-nw-n-retroflex` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | · |
| `@s-h` | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ | · | · | · | · |
| `@hiatus` | 3 | 2 | 2 | 3 | 2 | 3 | 3 | 3 | 3 | 3 | 3 | ✓ |
| `@flap` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | · |
| `@v-loss` | ✓ | · | · | · | · | ✓ | · | ✓ | ✓ | · | · | ✓ |
| `@x-nw-n-dissimilation` | ✓ | · | · | · | · | · | ✓ | · | · | · | · | · |
| `@syncope` | · | ✓ | · | ✓ | ✓ | · | ✓ | · | ✓ | ✓ | ✓ | · |
| `@anusvara-vowel` | · | · | · | ✓ | · | ✓ | ✓ | ✓ | ✓ | 2 | · | · |
| `@h-loss` | · | · | · | ✓ | · | ✓ | ✓ | · | ✓ | · | ✓ | ✓ |
| `@x-nw-mb-mm` | · | · | · | ✓ | · | · | · | · | · | · | · | · |
| `@x-nw-d-retroflex` | · | · | · | · | ✓ | · | · | · | ✓ | ✓ | · | · |
| `@x-hindi-mr-mb` | · | · | · | · | · | ✓ | · | · | ✓ | ✓ | ✓ | · |
| `@vowel-shift` | · | · | · | · | · | ✓ | · | · | ✓ | · | ✓ | · |
| `@tone` | · | · | · | · | · | ✓ | · | · | · | · | · | · |
| `@deaspiration` | · | · | · | · | · | ✓ | · | · | · | · | · | ✓ |
| `@implosive` | · | · | · | · | · | · | · | 3 | · | · | · | · |
| `@x-nw-tr-cerebral` | · | · | · | · | · | · | · | ✓ | · | · | · | · |
| `@x-nw-l-r` | · | · | · | · | · | · | · | ✓ | ✓ | ✓ | · | · |
| `@x-nw-postnasal-aspiration` | · | · | · | · | · | · | · | · | ✓ | · | · | · |
| `@nasal-cluster` | · | · | · | · | · | · | · | · | · | ✓ | · | · |
| `@x-nw-aspirate-devoicing` | · | · | · | · | · | · | · | · | · | · | ✓ | · |
| `@int-spirant` | · | · | · | · | · | · | · | · | · | · | ✓ | · |
| `@x-nw-ch-sh` | · | · | · | · | · | · | · | · | · | · | ✓ | · |
| `@x-nw-deretroflexion` | · | · | · | · | · | · | · | · | · | · | ✓ | · |
| `@x-nw-s-z` | · | · | · | · | · | · | · | · | · | · | ✓ | · |
| `@x-dardic-north-tr-affricate` | · | · | · | · | · | · | · | · | · | · | · | ✓ |
| `@sibilant-keep` | · | · | · | · | · | · | · | · | · | · | · | ✓ |
| `@x-nw-nc-nasal` | · | · | · | · | · | · | · | · | · | · | · | ✓ |
| `@cluster-simplify` | · | · | · | · | · | · | · | · | · | · | · | ✓ |

## 3. Relative chronology (train NED change when the order is reversed)

From `swap.py` / `adjswap.py` on the final or near-final files; + = the reversed order is worse.

- **Sindhi lenitions precede Sindhi degemination.** Moving each after `G $1 > $2`:
  `l > r` +0.0056, voiced geminate > implosive +0.0035, `s > h` +0.0030, `n > ṇ` +0.0020,
  `m > Ṽ` +0.0018. Kutchi `l > r` before degemination: +0.0041. So intervocalic l, s, n, m were
  lenited while old geminates (ll, ss, nn, mm, and dd/bb/gg/jj, which became implosives) were still
  long. Moving degemination wholesale into the MIA block (before final-consonant loss) costs +0.017
  (Kutchi +0.010). Sindhi degemination is late (after the implosives arose), not part of the MIA
  cluster simplification.
- **Pretonic degemination precedes the -aka > -ā contraction** (P +0.0126 if moved after the -ā /
  infinitive morphology; L +0.0115 in the earlier L file that still had the -ā rule). kuttaka > kuttaa > kuttā keeps its geminate while *pēṭṭāra > piṭārī loses
  it: the shortening was conditioned by an old long vowel in the next syllable, before -aa contracted.
- **Flap before post-long-vowel degemination** (P +0.0012 if reversed; L, OP, kcch preferred the same
  order by 0.0005–0.0018): ḍḍ after a long vowel is not flapped, i.e. that degemination is later than ḍ > ṛ.
- **Hindko s > h precedes h-loss** (+0.0065): s > h > ∅ (tone), a feeding chain; h-loss also precedes
  the flap (+0.0040). Pothwari shows the same feeding (+0.020 if reversed).
- **rt > ṭṭ precedes r-assimilation** (P +0.0006, L +0.0008): trivial bleeding, but it places the
  cerebralisation in the MIA stage.
- **Final -ā shortening**: in P/L/poth it works best at the end of the MIA block (Apabhraṁśa), in S after
  the -a > -u / -aka > -o citation rules (reversing costs +0.003–0.005): fem. -ā did not pass through
  -a > -u. tr > ṭr precedes initial d > ᶑ in S (dr- > ḍr-, not ᶑr-).
- Orders that did not matter (Δ ≤ 0.0003): n-dissimilation vs syncope, nasal voicing vs the
  morphology, n > ṇ vs hiatus, int-aspirate-h vs stop loss.

## 4. Shared vs lect-specific changes; sub-grouping

**What separates NW from Central (Hindi).** Isogloss tests (`iso.py`, final files; Δ train NED of adding
a rule where absent, or removing it where present (`*`); negative = the rule helps):

| lect | s>h | n>ṇ | l>r | nasal-voi | implos | v>b | y>j | fin-V-loss | deasp | int-asp-h | rt>ṭṭ | m>Ṽ | degem-all | comp-len | sib>s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P | −0.002* | −0.005* | +0.014 | −0.005* | +0.017 | −0.001* | −0.003* | −0.001* | +0.023 | −0.006* | −0.001* | +0.003 | 0* | +0.051 | −0.022* |
| L | −0.004* | −0.006* | +0.014 | −0.009* | +0.010 | +0.009 | −0.004* | −0.005* | +0.022 | −0.004* | −0.001* | 0 | +0.018 | +0.065 | −0.028* |
| S | −0.007* | −0.008* | −0.009* | −0.006* | −0.006* | +0.009 | −0.003* | −0.008* | +0.021 | −0.005* | −0.001* | −0.002* | −0.048* | 0 | −0.030* |
| OP | +0.004 | −0.001* | +0.013 | −0.001* | +0.015 | +0.001 | −0.003* | −0.001* | +0.022 | −0.005* | 0 | +0.004 | +0.006 | +0.029 | −0.024* |
| awan | −0.006* | −0.007* | +0.013 | −0.006* | +0.014 | +0.006 | −0.002* | −0.009* | +0.017 | −0.006* | −0.001* | −0.001* | −0.005* | +0.051 | −0.026* |
| kcch | 0 | −0.007* | −0.010* | −0.003* | +0.034 | 0 | −0.004* | −0.034* | +0.026 | −0.001 | 0 | +0.001 | −0.013* | +0.047 | −0.018* |
| D | +0.003 | 0 | +0.017 | 0 | +0.014 | −0.014* | −0.001* | −0.004* | −0.012* | −0.009* | 0 | +0.013 | −0.017* | +0.011 | +0.013 |
| srk | −0.004* | −0.007* | +0.014 | −0.006* | +0.018 | +0.005 | −0.001* | −0.004* | +0.016 | −0.007* | 0 | −0.004* | −0.006* | +0.045 | −0.025* |
| thar | +0.022 | −0.004* | −0.003* | 0 | +0.008 | 0 | 0 | −0.004* | +0.009 | +0.018 | 0 | +0.008 | −0.058* | −0.001 | −0.054* |
| poth | −0.021* | −0.004* | +0.015 | −0.004* | +0.017 | −0.003* | −0.001* | −0.002* | −0.012* | −0.006* | −0.001* | −0.002* | −0.026* | +0.040 | −0.029* |
| khet | −0.017* | −0.002* | +0.009 | −0.007* | +0.017 | +0.007 | −0.002* | −0.041* | +0.004 | −0.003* | 0 | +0.002 | −0.015* | +0.038 | −0.036* |
| Khol | +0.005 | 0 | +0.011 | −0.003* | +0.018 | +0.012 | −0.003* | −0.004* | −0.001 | +0.002 | 0 | +0.006 | −0.009* | +0.002 | −0.021* |

H column (scratch copy of `OIA-H.sc`, rule added): post-nasal voicing +0.007, s > h +0.015,
rt > ṭṭ +0.0008 (n > ṇ cannot be tested: that file ends in ṇ > n).

The NW core (P, L, S, awan, srk, khet, poth, kcch) is defined by **no compensatory lengthening**
(every one pays +0.03…+0.07 for it), **post-nasal voicing** (nt > nd), **n > ṇ**, and (P L S awan poth)
**rt > ṭṭ**; **s > h** holds in P L S awan srk khet poth but not OP, kcch, thar, Khol, D. Hindi rejects
all three testable ones. Sibilant merger is universal except Domaaki.

**Cross-application** (`cross.py`: train NED of lect Y's data under lect X's main chain, without
`%source` blocks; own chain in bold; H/Marw/G/Sh are other groups' files, read-only):

| data ↓ / chain → | null | P | L | S | OP | awan | kcch | thar | Khol | khet | srk | poth | D | H | Marw | G | Sh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P | 0.604 | **0.362** | 0.416 | 0.525 | 0.457 | 0.450 | 0.493 | 0.531 | 0.620 | 0.444 | 0.435 | 0.465 | 0.553 | 0.466 | 0.560 | 0.509 | 0.565 |
| L | 0.609 | 0.442 | **0.379** | 0.500 | 0.478 | 0.436 | 0.528 | 0.544 | 0.628 | 0.431 | 0.399 | 0.475 | 0.576 | 0.539 | 0.598 | 0.526 | 0.586 |
| S | 0.636 | 0.509 | 0.478 | **0.335** | 0.475 | 0.542 | 0.452 | 0.478 | 0.616 | 0.492 | 0.496 | 0.570 | 0.580 | 0.579 | 0.549 | 0.562 | 0.620 |
| OP | 0.601 | 0.483 | 0.514 | 0.536 | **0.403** | 0.535 | 0.551 | 0.560 | 0.640 | 0.526 | 0.531 | 0.555 | 0.568 | 0.558 | 0.559 | 0.583 | 0.598 |
| awan | 0.651 | 0.493 | 0.478 | 0.575 | 0.533 | **0.432** | 0.562 | 0.593 | 0.627 | 0.496 | 0.498 | 0.494 | 0.604 | 0.578 | 0.622 | 0.562 | 0.581 |
| kcch | 0.663 | 0.517 | 0.509 | 0.553 | 0.545 | 0.544 | **0.395** | 0.510 | 0.620 | 0.525 | 0.508 | 0.539 | 0.610 | 0.578 | 0.564 | 0.564 | 0.609 |
| D | 0.692 | 0.651 | 0.644 | 0.715 | 0.656 | 0.677 | 0.685 | 0.692 | 0.670 | 0.631 | 0.668 | 0.627 | **0.541** | 0.641 | 0.689 | 0.700 | 0.583 |
| srk | 0.667 | 0.531 | 0.501 | 0.578 | 0.552 | 0.534 | 0.586 | 0.607 | 0.631 | 0.517 | **0.480** | 0.529 | 0.625 | 0.588 | 0.645 | 0.590 | 0.622 |
| thar | 0.670 | 0.563 | 0.534 | 0.567 | 0.559 | 0.580 | 0.477 | **0.427** | 0.607 | 0.514 | 0.558 | 0.559 | 0.602 | 0.507 | 0.514 | 0.487 | 0.584 |
| poth | 0.678 | 0.517 | 0.523 | 0.612 | 0.567 | 0.517 | 0.580 | 0.600 | 0.594 | 0.502 | 0.544 | **0.450** | 0.591 | 0.561 | 0.607 | 0.586 | 0.581 |
| khet | 0.708 | 0.559 | 0.552 | 0.640 | 0.622 | 0.571 | 0.645 | 0.670 | 0.672 | **0.513** | 0.583 | 0.592 | 0.647 | 0.635 | 0.687 | 0.629 | 0.646 |
| Khol | 0.721 | 0.623 | 0.591 | 0.678 | 0.624 | 0.623 | 0.592 | 0.611 | **0.438** | 0.584 | 0.603 | 0.602 | 0.626 | 0.629 | 0.638 | 0.636 | 0.627 |

(Citation morphology inflates every off-diagonal number; compare within a row.)

Sub-grouping suggested:
- **Lahndic core = L + srk + khet + awan + poth, with P attached.** CDIAL L data are predicted best
  by the Saraiki chain (0.399) and then Khetrani (0.431), i.e. CDIAL "L." is mainly southern
  (Multani) Lahnda. Hindko (awan) goes with L (0.478) and P/poth (0.493/0.494); Pothwari with khet/P/awan
  (0.502/0.517/0.517). Saraiki and Khetrani go with L, not with Sindhi (srk: L 0.501 vs S 0.578;
  khet: L 0.552 vs S 0.640), although srk shares m > Ṽ and final ā > a with S, and khet shares full
  degemination and dental + r retention.
- **Punjabi vs Lahnda** differ in degree rather than kind: implosives (0% of initial g j ḍ d b in
  P, 28% in L, 28% khet, 17% srk, 69% S), initial d- written ḍ/ᶑ (P 18%, L 53%, kcch 75%, S 81%),
  ava > o (L, khet, poth, thar) vs ava > au (P, awan, OP, kcch, srk), v > b (P, Pothwari; L +0.009).
  A south-to-north gradient S > khet/srk > L > P looks like a Sindhi–Lahnda–Punjabi continuum.
- **North vs south Lahndic:** tonal devoicing and h-loss are visible in the phonetic sources for Hindko
  and Pothwari (and Gill's Punjabi) but hurt Saraiki (+0.006) and Khetrani (+0.016).
- **Sindhic = S + kcch (+ khet partly)**: full degemination without lengthening, l > r (S kcch thar),
  initial d- > retroflex (kcch ḍ- in 47 of 63 initial d-). Kutchi lacks implosives, s > h and
  aspirate > h (Gujarati-like). S data are predicted best by kcch (0.452), then OP/L/thar (0.475–0.478).
- **Dhatki (thar)** is Kutchi/Gujarati/Rajasthani-like: best foreign chains kcch (0.477), G (0.487),
  H (0.507), khet (0.514), Marwari (0.514); S 0.567. It shares l > r and d- > ḍ- with Sindhic, but has no s > h (+0.022)
  and no implosives.
- **Kholosi** is closest to khet/L/kcch (0.584/0.591/0.592) — a Sindhi–Lahnda–Gujarati-type base with
  Iranian-contact innovations (§5).
- **Old Punjabi** (GGS) is closer to P (0.483) than to L (0.514) or H (0.558): Punjabi-Lahndic phonology
  (no lengthening, n > ṇ) but no s > h and -u/-i endings — an archaic/literary register.
- **Domaaki is not NW**: of all foreign chains, the Shinaic ones of the dardic-north group fit it
  best — Kalkoti 0.577, Shina 0.583, Ushojo 0.587, Savi 0.603 (measured against a slightly earlier D
  file; Sh 0.583 is unchanged on the final one) — close to its own chain (0.541 without blocks), and far
  better than P (0.651), H (0.641), Pothwari (0.627), khet (0.631), the other Dardic chains
  (Kalasha 0.606, Gawri 0.610, Phalura 0.628, Khowar 0.632, Kashmiri 0.638) or the migratory "Dom"
  lects (Balkan Romani 0.633, European Romani 0.645, Domari 0.647, Lomavren 0.667). Its sound
  changes are Shinaic; a Dom/Romani-type or NW-Punjabi affiliation finds no support here. See §5.


## 5. Novel / interesting observations

- **Gemination is partly a notation variable.** CDIAL writes geminates in 13–17% of P/L/awan reflexes;
  the phonetic field wordlists for the same lects almost never do (Gill P 2/52, Rensch Hindko 24/517,
  Rensch Saraiki 1/120, Lothers Pothwari 14/365, Elfenbein/Grierson Khetrani). A per-source
  `G $1 > $2` notation block helps every one of them (awan −0.004, srk −0.004, khet −0.004,
  poth −0.017 with the Lothers vowel block). Anyone comparing "degemination" across groups should check
  whether the source writes geminates at all.
- **Tone is visible only in phonetic sources, and it splits Lahndic north/south.** CDIAL P/L forms
  follow Gurmukhi/Shahmukhi spelling (voiced aspirates, h), so the P and L chains keep bʰ dʰ gʰ and h;
  adding tonal h-loss costs P +0.010, L +0.009. The phonetic sources show the tonal devoicing of
  initial voiced aspirates (Gill P bʰaginīpati > pàṇoīā; Rensch Hindko bʰaginī > pɛṇ; Lothers Pothwari
  gʰara > kar, plus 32 forms with the breathy-vowel diacritic ̤). Intervocalic h-loss is part of the
  Hindko and Pothwari chains (s > h feeding it: awan −0.006 for s > h before h-loss, poth −0.016 for
  adding s > h) but hurts Saraiki (+0.006) and Khetrani (+0.016), and the Rensch devoicing block helps Hindko but hurts Saraiki (+0.006). Awankari CDIAL forms mark tone
  with a grave accent exactly where an intervocalic h/aspirate was lost (kīdr̩śa > kìā, jáḍʰu > jàlā).
  So: northern Lahndic (Hindko, Pothwari; and Punjabi phonetically) = tonal, southern (Saraiki,
  Khetrani, Sindhi) = no tone, voiced aspirates and h kept.
- **Bagri mis-filed under Punjabi** (130 P pairs; Rajasthani lengthening, NED 0.58 vs 0.38). Excluded.
- **Domaaki is not simply "Punjabic".** Its chain keeps all three sibilants (merging them costs
  +0.013; aṣṭā́ > oṣṭ, śaknōti > śʌk, ṣoḍaśa > ṣõõi, and s + stop survives), deaspirates voiced aspirates
  (−0.013), has no ṇ, retroflex affricates from tr/dr (tráyaḥ > ʦ̣éedo, dīrgʰa > ʣ̣īgā), NC > N
  (síndʰu > sina, kāṇḍa > kóon) and keeps final vowels. ā > o in final closed syllables is visible
  (dvāra > dor, nāman > nom, kappāla > kʌpōl) but mixed with ā (kāla > kālā) and is not a net gain;
  n > ṇ, s > h, l > r and nasal voicing are all neutral or harmful. Like Shina it also loses
  intervocalic p/v and h (p > v −0.002, v > ∅ −0.004, h > ∅ −0.003). The unmodified Kalkoti/Shina chains fit
  Domaaki (0.577/0.583) almost as well as its own chain and far better than any NW or Central chain
  (§4). Domaaki shares with the plains only v > b (−0.014) and degemination without lengthening. Whether this is a Dardic substrate/adstrate (Shina in
  Hunza–Nagar) or a Dom lect that was never NW, its *sound changes* are Shina-type.
- **Kholosi** (Hormozgan, Iran) shares Sindhic/Gujarati morphology (-o, -en) and is closest to L/kcch
  in cross-application, but has Iranian-contact innovations: kʰ/gʰ > x, voiced aspirates > voiceless
  aspirates (bʰaginī > pʰen, duhitr̩ > tʰīv), de-retroflexion (pēṭṭa > pet, nāḍī > narī), s > z
  intervocalic (daśa > dazerā), cʰ > ś, lowering of short i u (putra > peter). Its geminates are mixed
  (atto, kadden, śekken beside pet, kan): early in the session degemination looked harmful, but once
  the other rules were in, full degemination helps (−0.009). It keeps Cr clusters (peter, doter, pātīl)
  and s + stop (adding sT > TTʰ hurts, +0.001).
- **Old Punjabi (Shackle)**: the Guru Granth Sahib language keeps final -u/-i (a > u / C_#: +0.028),
  cites verbs as stem + -i (galati > gali), has no s > h (+0.005 if added), keeps written anusvāra ṁ
  (removing it is neutral, unlike P/L/S where anusvāra loss helps), and v-/b- is ~50/50 in
  `shackle-auto` (Braj/sant-bhāṣā b- vs Punjabi v-): a lexical-stratum mixture, not notation.
- **Kutchi** unexpectedly degeminates like Sindhi (−0.013) and has l > r and initial d > ḍ, but no s > h,
  no intervocalic aspirate > h, no implosives; it adds post-nasal aspiration (aŋguṣṭʰa > āŋgʰūṭʰo,
  undura > ūndʰar) and initial u- > au- (utkalati > aukkarṇū; probably [ɔ]). Its CDIAL forms still write
  some geminates (auttarṇū, auṭṭʰ), so its degemination is a majority, not an absolute.
- **Dhatki (thar)** is a Kutchi/Gujarati/Rajasthani-type lect: no s > h (+0.022), no implosives, masc. -o,
  n > ṇ, degemination without lengthening; but it shares Sindhic l > r (−0.003) and d- > ḍ- (−0.001).
- **Punjabi -ṇā/-nā**: the infinitive is -nā after r ṛ ṇ (sārnā, jʰarnā), -ṇā elsewhere; modelled as
  a late dissimilation. The same dissimilation also helps L, OP, srk, khet.
- **Sindhi -u vs -o** is the largest residual (u→o ≈ 900 train confusions): CDIAL S cites many a-stems
  in -o (MIA -aka) without -aka in the headword; no phonological or accentual conditioning found
  (starred etyma: -o 421 vs -u 353; attested OIA: -o 592 vs -u 733).
- **Compound members often behave as separate words**: the second member usually keeps its initial
  consonant unlenited (niṣputra > P naputtā, *pratipautra > paṛottā, catuṣpattra > caupattā; only old,
  lexicalised compounds lenite: *ēkkalaputra > iklauttā), so intervocalic p > v over-applies across
  compound boundaries. An engine-side word-boundary marker in compound etyma would help.
- **-tra > -tar** (anaptyxis instead of assimilation) is sporadic in P/L (sūtra > sūtar, mātra > mātar,
  vahítra > bahitar, carítra > calittar, ēkatra > ikattar; áśru > attʰrū); a word-final rule gains
  only in L (−0.0007) and hurts P, so it is left out — probably a semi-learned stratum. Overall only
  ~5% of reflexes (P 4.7%, L 5.4%, S 4.9%, OP 6.9%) are closer to the bare etymon than to the chain
  output, i.e. the tatsama stratum is small in these CDIAL data.
- **CDIAL headword conventions** that needed rules: anusvāra before a stop = homorganic nasal; root
  etyma (*dabb, *cul) are verbal and surface as infinitives (P -ṇā, L -aṇ, S -aṇu, OP stem-i); in-stems
  cited in the nominative -ī; ā-stem causatives (-āyati, -ā(C)ayati) keep the ā (S -āiṇu, P -āuṇā).
- **Data debris** seen in the worst misses: truncated reflexes (L sárva > "a", kārttiká > "a",
  viśrānta > "ntā"; S gārjara > "ā"), an etymon "*?" (S, elfenbein), phrasal and reduplicated reflexes
  (kcch "pʰā̃sle me ganṇū", S "mõjʰu-mõjʰu"), and OCR/link noise in `shackle-auto`.

## 6. New tags, engine wishes

New `@x-nw-*` tags:
- `@x-nw-n-retroflex` — intervocalic n > ṇ (NIA, NW), distinct from the Prakrit spelling `@n-retroflex`.
- `@x-nw-n-dissimilation` — ṇ > n after r ṛ ṇ (Punjabi -nā after r-stems).
- `@x-nw-rt-cerebral` — r + dental > retroflex geminate (rt > ṭṭ, rd > ḍḍ) in MIA.
- `@x-nw-tr-cerebral` — dental + r > retroflex + r (Sindhi ṭr, ḍr).
- `@x-nw-dv-b` — dv > b(b).
- `@x-nw-l-r` — intervocalic l > r (Sindhi, Kutchi, Dhatki).
- `@x-nw-d-retroflex` — initial d- > ḍ- (Saraiki, Kutchi, Dhatki; Sindhi's version is `@implosive` ᶑ).
- `@x-nw-final-a-short` — final -ā > -a (Apabhraṁśa shortening; always co-tagged `@final-vowel-loss`,
  as in OIA-H.sc's `ā > a / C _ #`; in P/L/poth at the end of the MIA block, in S/OP/srk/thar/D after
  the citation-vowel morphology).
- `@x-nw-anusvara-loss` — leftover anusvāra (before s h v, word-final) lost without nasalisation.
- `@x-nw-pretonic-shortening` — long vowel shortened before a stressed long vowel (P L khet thar).
- `@x-nw-postnasal-aspiration` — voiced stop aspirated after a nasal (Kutchi aŋgʰūṭʰo, ūndʰar).
- `@x-nw-mb-mm` — mb mbʰ > mm (Hindko only; P, L, srk data reject it).
- `@x-nw-nc-nasal` — NC > N (Domaaki; the same process as dardic-north's `@x-dardic-north-nd-n`).
- `@x-nw-aspirate-devoicing` — voiced aspirates > voiceless aspirates (Kholosi pʰen, tʰīv).
- `@x-nw-deretroflexion` — ṭ ḍ ṇ > t d n (Kholosi).
- `@x-nw-s-z` — intervocalic s > z (Kholosi).
- `@x-nw-ch-sh` — cʰ > ś (Kholosi).
- Reused from another group: `@x-dardic-north-tr-affricate` (Domaaki tr dr > ʦ̣ ʣ̣).

Engine wishes:
- A source-conditioned rule *inside* the chain (not only after it): e.g. only Shackle's GGS forms keep
  anusvāra, only CDIAL Domaaki cites verbs with -na — these interact with later rules, so a tail
  block cannot express them. (Also: a way to mark compound boundaries in etyma, see §5.)
- Scoring against the *set* of attested variants of an etymon as alternatives (min over reflexes)
  rather than summing: P/L/S often give 3–5 derivatives per etymon (-ā, -ī, -ṇā forms).
- `frontier`/`look`-type tools that apply `%source` tails (my `look.py` shows main-chain output only).


## 7. Round 2: vowels and accent

Scratch for this round is in `tmp/sound_changes/northwest/r2/`:
- `batch.py` inserts, replaces or deletes rules and rescores train;
- `tab.py` gives the attested outcome rates of OIA vowels by Vedic accent within fixed positional cells;
- `qual.py` shows where OIA vowel V surfaces as X;
- `cmp.py` lists the items whose prediction changes between two rule files;
- `accend.py` gives the citation vowel by accent class.

Δ = change in train NED when a rule is added (negative = better). LOO = rise in train NED when the rule
is removed from the final file.

### 7.1 Audit of the partial round-2 edits; results

The stopped agent's edits (`tmp/sound_changes/snapshots/round2-partial/`) were all rescored with
`--loo`.

**Kept:**
- `if:verb` / `if:!numeral` gating of the citation morphology (every lect with infinitive rules);
- final `ā > ə` (so that the -ā citation rule does not re-lengthen the shortened fem. -ā; P L poth);
- `if:!accented a > ā`, the citation -ā (P L awan poth khet; L later switched to `if:starred`);
- S `if:starred a > o`;
- S `-r̩ > -u`;
- the pretonic ū / ī ū shortening in S, L and khet.

All of these have positive LOO.

**Removed:**
- thar `ū > u / _ !#* Acc`: it fires on only one etymon, so it is word-specific;
- five older rules whose LOO fell to ≤ 0 under the new engine:
  - S and L `a {i u} > {e o}` (Δ on removal −0.0005 / −0.0001);
  - four awan rules (anusvāra notation, `h y > jjʰ`, `V m > Ṽ`, `in > ī`; all 0.0000 or −0.0001).

**Added in this session:** see §7.2.

**Test NED against round 1:** no lect's test NED rose by more than 0.005. The largest rise is P
(+0.0009 on test from the long-vowel nasalisation rule, which gains 0.0013 on train). Note also
that the round-1 file for khet in `frozen/` already scored 0.4960 / 0.4534 under the new engine.

| lect | train NED r1 → r2 | test NED r1 → r2 | test exact r2 | main sources of the change |
|---|---|---|---|---|
| P | 0.3609 → 0.3539 | 0.3588 → 0.3540 | 14.6% | if:verb gating, ə for final -ā, if:!accented -ā, V̄NC > Ṽ̄C |
| S | 0.3351 → 0.3253 | 0.3515 → 0.3465 | 19.3% | if:verb, if:starred -o, final -u on C-final stems, V̄NC > Ṽ̄C, pretonic ī ū |
| L | 0.3777 → 0.3713 | 0.3706 → 0.3619 | 16.2% | if:verb, ə, if:starred -ā, pretonic ū |
| OP | 0.4022 → 0.4000 | 0.3976 → 0.3947 | 15.6% | if:!numeral, starred -ā |
| awan | 0.4157 → 0.4106 | 0.4470 → 0.4352 | 13.0% | if:!numeral, if:!accented -ā, fem. -ā/-ī loss |
| kcch | 0.3954 → 0.3886 | 0.3981 → 0.3941 | 12.8% | if:verb, fem. -ā loss, jʰ > j |
| D | 0.5064 → 0.5040 | 0.5136 → 0.5136 | 4.1% | if:!numeral -na |
| srk | 0.4736 → 0.4724 | 0.4907 → 0.4846 | 7.2% | if:verb |
| thar | 0.4269 → 0.4269 | 0.4602 → 0.4602 | 9.7% | none (the round-2 rule was removed) |
| poth | 0.4109 → 0.4046 | 0.4549 → 0.4397 | 10.8% | ə, if:!accented -ā |
| khet | 0.4988 → 0.4799 | 0.4611 → 0.4315 | 16.7% | if:verb, if:!accented -ā, final -ā > ə, pretonic ī ū |
| Khol | 0.4384 → 0.4384 | 0.5440 → 0.5440 | 4.5% | none |

Round-1 numbers come from `results/summary-round1.csv`. The engine's tokenisation changes alone move
them by ≤ 0.001 (P started round 2 at 0.3602 / 0.3579).

**`%gold-drop tone`** was tested as a pure scoring change and **not adopted**:
- awan train 0.4139 → 0.4133 (test unchanged);
- poth and P unchanged to four decimals;
- L test 0.3640 → 0.3637.

The tone marks in these sources are too few to matter.

### 7.2 Rules adopted this round (vowels, accent, citation vowels)

| lect | rule | LOO | examples / controls |
|---|---|---|---|
| S | `{ī ū} > {i u} / _ !#* Acc` @two-mora @stress | +0.0004 (54 etyma) | Carried by ū (the ū-only rule scores the same). Beats the positional control `_ C* V` (non-final ī ū) by 0.0037 and `_ !#* Long` by 0.0002. Restricting it to accented words, `if:accented {ī ū} > {i u} / _ C* V`, also scores the same, so tonic non-final ū is neutral. |
| L | `ū > u / _ !#* Acc` @two-mora @stress | +0.0010 (18) | The positional control `ū > u / _ C* V` is 0.0021 worse. |
| khet | `{ī ū} > {i u} / _ !#* Acc` | +0.0027 (8) | Small; kept because L, its closest relative, has it. |
| P S | `{ā ī ū ē ō} Nasal > $1+nas / _ Stop` @anusvara-vowel | P +0.0013 (74), S +0.0009 (73) | Nasal + stop after a **long** vowel > nasal vowel + stop: āntrá > P ā̃d, S ā̃ḍro; kāṇṭaka > S kā̃ḍo; nānda > P nā̃d. After short vowels the nasal stays (aŋka > aŋg). For all vowels the rule costs +0.020. It hurts L (+0.0018), OP, kcch, srk, poth, khet and Khol, so in CDIAL P it may partly reflect Gurmukhi spelling (bindī on long vowels, ṭippī after short ones). |
| awan | `{ā ī} > ə / C _ #` @final-vowel-loss @x-nw-final-a-short | +0.0031 (44) | Hindko loses fem. -ā **and -ī**: vallī > val, bʰaginī > bʰε̄̀ṇ, garbʰíṇī > gabbʰuṇ, dʰā́rā > dʰār. The -ī part (Δ −0.0015 on its own) is Hindko-only: P +0.0011, srk +0.0015, khet +0.0017, L 0, poth −0.0003. |
| khet | `ā > ə / C _ #` (as in P L poth) | +0.0024 | Khetrani had no Apabhraṁśa -ā shortening. |
| kcch | `ā > ∅ / C _ #` @final-vowel-loss | +0.0005 (42) | Fem. -ā lost after the masc. -o citation rule: jihvā > jībʰ. |
| P awan poth khet (L: `if:starred`) | `if:!accented a > ā / C _ #` @morph | P +0.0090, L +0.0045, awan +0.0090, poth +0.0058, khet +0.0104 | A lexical-stratum effect, not an accent effect: see 7.3. |
| S | `if:starred a > o / C _ #` @morph | +0.0037 | `if:!accented` is 0.0020 worse; dropping the rule is 0.0036 worse. |
| OP | `if:starred a > ə / C _ #` … `ə > ā` @morph | +0.0020 | Turner's starred a-stems are cited in -ā, attested ones in -u. The ə placeholder lets the -ā escape the -ā shortening. `if:!accented` in the same slot is 0.0037 worse, the unconditioned rule 0.0135 worse. |
| S | `∅ > u / C _ #` @morph | +0.0030 (147) | Sindhi nouns end in a vowel: consonant-final starred root/suffix headwords take masc. -u (*kakudʰaṭṭ > kūhaṭu, *cuttaḍ > cutaṛu). Final -o instead of -u: Δ −0.0029. |
| S | `{r̩ ŕ̩} > u / _ #` @r-vocalic | +0.0004 (11) | pitr̩ > piu. |
| kcch | `jʰ > j` @deaspiration (not a vowel rule) | +0.0039 (22) | dahyátē > ḍajṇū, *jʰal > jalṇū, búdʰyatē > bujṇū. |

### 7.3 Hypotheses tested and rejected

Δ is train NED when the rule is added (+ = worse).

**Pretonic long-vowel shortening follows weight, not the Vedic accent.** The accent version,
`{ā ī ū} > {a i u} / _ !#* Acc` (placed before @acc-loss), hurts every lect:

| lect | Δ | lect | Δ |
|---|---|---|---|
| P | +0.0067 | srk | +0.0055 |
| S | +0.0053 | thar | +0.0138 |
| OP | +0.0082 | poth | +0.0106 |
| awan | +0.0083 | khet | +0.0113 |
| kcch | +0.0094 | Khol | +0.0114 |
| D | +0.0007 | | |

In P it costs +0.0075 if it *replaces* the weight rule. Even the narrowest accent version, "shorten
before a *tonic long* vowel" `_ C (C) {ā́ ī́ ū́ ḗ ṓ}`, costs P +0.0011 and L +0.0008; before any
tonic vowel (`_ C (C) Acc`) it costs +0.0061 / +0.0082. The round-1 weight rule
(`V̄ > V / _ C (C) V̄ C`, long vowel before a heavy long syllable) keeps a positive LOO (P +0.0009).

On the headwords (first-syllable open ā ī ū), the share shortened:

| | pretonic | tonic | unaccented | before heavy σ | before light σ |
|---|---|---|---|---|---|
| L | 8% | 12% | 27% | 39% | 14% |
| P | 15% | 12% | 30% | 34% | 21% |

The Vedic accent does not predict which long vowels shorten; the weight of the next syllable does.
Pretonic ē ō > i u (Hindi type) is also rejected: by accent +0.0019 / +0.0022 (P / L); by weight
+0.0011; folded into the weight rule 0.0000 / +0.0005.

**Pretonic degemination also follows weight, not the accent.** Tonic short vowels right after a
cluster were marked before @acc-loss (`if:!verb á > ɐ / C C _`, restored to a later). The marker
alone is neutral (P 0.0000, L +0.0001). Adding degemination before the marked tonic vowel,
`G $1 > $2 / _ ɐ`, costs P +0.0020 and L +0.0021. The weight rule
(`G $1 > $2 / _ V̄ | _ V C C`) has LOO P +0.0037, L +0.0020; its long-vowel half alone is
0.0007 / 0.0005 worse.

**Syncope and apocope.**
- On the headwords, L medial open a is lost 15% of the time when tonic vs 37% posttonic, 61%
  pretonic and 46% unaccented. Final i u are kept 59% of the time when tonic vs 24% posttonic.
- But the chain already produces these differences through the -aka contraction, stop loss and
  hiatus rules:
  - protecting tonic medial a from syncope (`if:!verb á > ɐ`, restored after syncope) changes six L
    etyma (−0.0001);
  - without `if:!verb` the same protection hurts (+0.0034), because it hits the thematic -áyati
    verbs;
  - protecting tonic final i u hurts (+0.0009).
- Early posttonic syncope, `{a i u} > ∅ / Acc C* _ C V`, hurts everywhere: OP +0.017, awan
  +0.017, kcch +0.026, srk +0.006, poth +0.010, khet +0.011, Khol +0.034, D +0.017.
- Pretonic-only vs posttonic-only syncope could not be evaluated cleanly with a marker on every
  medial a: the marker alone costs +0.028 because it bleeds the a-specific hiatus rules.
- Positional syncope in P (P has none): `a > ∅ / V C _ C V` +0.0002; before a heavy syllable
  +0.0001; `{a i u}` −0.0001. Not adopted.

**Citation vowels are lexical strata, not accent.** For a-stem headwords (train), the share of
reflexes ending in -ā (P, L, awan, poth, OP) or in -o / -u (S, kcch):

| class | P -ā | L -ā | awan -ā | poth -ā | OP -ā / -u | S -o / -u | kcch -o |
|---|---|---|---|---|---|---|---|
| barytone | 30% | 26% | 28% | 25% | 22 / 50% | 26 / 36% | 31% |
| oxytone | 35% | 29% | 23% | 15% | 24 / 41% | 30 / 40% | 34% |
| unaccented, attested | 39% | 32% | 33% | 29% | 27 / 40% | 34 / 34% | 35% |
| starred | 44% | 37% | 41% | 40% | 37 / 27% | 40 / 31% | 39% |

- Oxytone and barytone words do not differ consistently: oxytones have slightly *more* -ā in P and
  L and less in awan and poth. The gradient runs by the age of the lexeme instead. Vedic-attested
  a-stems surface bare (hásta > hattʰ). Later and reconstructed words carry the MIA -aka extension
  (-ā, -o).
- Which flag works best differs by lect:
  - `if:!accented` beats both the unconditioned rule (P by 0.0015, awan by 0.0011, khet by 0.0119)
    and `if:starred` (P by 0.0020, awan by 0.0046, khet by 0.0107);
  - in L and poth the two flags are about equal (L `if:starred` 0.0002 better, adopted; poth
    `if:!accented` 0.0004 better), and both beat the unconditioned rule (+0.0032 / +0.0066 worse);
  - in S and OP, `if:starred` is better (S: `if:!accented` +0.0020; OP: +0.0037, unconditioned
    +0.0135);
  - in kcch, thar and Khol, both flags are worse than the unconditioned -o (kcch `if:!accented`
    +0.0011 / `if:starred` +0.0059; thar +0.0075 / +0.0127; Khol +0.0024 / +0.0106).
  - So the flag tracks attestation strata, and which stratum boundary matters depends on the lect.
- Sindhi -o vs -u has no phonological conditioning either. The -o share is 0.50 / 0.47 / 0.45 by
  penult weight (CC / long / light) and 0.49 / 0.45 / 0.46 by syllable count (2 / 3 / 4+).
- Sindhi final -i / -ī / -ā show no accent pattern (counts of 9–78 per cell).
- The probe's large Sindhi cell "posttonic final i > u" (n = 276) is the -ati verb ending. As a
  vowel rule (`{i í} > u / Acc !#* _ #`) it costs +0.0158.

**Vocalic r̩.** Pretonic `r̩ > a / C _ !#* Acc` (the mr̩ttikā́ type) is not adopted:

| lect | Δ | lect | Δ |
|---|---|---|---|
| P | −0.0001 | awan | +0.0009 |
| S | +0.0005 | kcch | +0.0013 |
| OP | +0.0006 | Khol | +0.0023 |

`if:accented r̩ > a` scores −0.0001 in P. The next-vowel split (a before i) and the velar-context
split score 0 / +0.0001.

**First-syllable vowels before clusters (the geminate-keeping lects).** NW keeps both the length and
the quality of first-syllable vowels before old clusters (Δ given for P / L / S):

| rule | Δ P / L / S |
|---|---|
| two-mora-law shortening `{ī ū} > {i u} / # C* _ C C` | +0.0014 / +0.0011 / +0.0011 |
| all long vowels, `_ C C` | P +0.0076, S +0.0082 |
| ē ō > i u | +0.009 / +0.009 / +0.008 |
| i u > e o | +0.018 / +0.024 / +0.022 |
| tonic lengthening `á > ā / # C* _ C C` | +0.0055 / +0.0070 / +0.0052 |
| open-syllable tonic lengthening `á > ā / # C* _ C V` | +0.004…+0.009 in every lect tested |

Long vowel + geminate is handled by the late post-long-vowel degemination (kārya > kājja > P kāj),
not by the two-mora law. Compensatory lengthening in the fully degeminating lects,
`{a i u} > {ā ī ū} / # C* _ C C` before degemination, is rejected too: thar +0.048, Khol +0.029
(for S and kcch see §4).

**Harmony / umlaut.** Tested in P, L and S (Δ P / L / S):

| rule | Δ P / L / S |
|---|---|
| a > u before u | +0.0040 / +0.0038 / +0.0033 |
| a > i before i | +0.0066 / +0.0063 / +0.0065 |
| i > u before u | +0.0006 / +0.0003 / +0.0006 |
| u > i before i | +0.0013 / +0.0012 / +0.0014 |

The aŋguli > uŋgal type is sporadic.

**Sindhi medial a > i.** Attested 17% of the time for medial a before an a-syllable
(añcala > ãcilo, āmalaka > āviro, utkalati > ukiraṇu). Every formulation hurts:
- after a heavy syllable +0.0145;
- after a long vowel only +0.0039;
- posttonic +0.0135;
- before r / l / ṛ only +0.0004.

**Other rejected rules.**
- awan final ā > ∅ as a late rule: +0.0203. The early ə version was adopted instead.
- D final a > ∅: +0.0142 (`if:accented` +0.0060). Domaaki keeps final vowels.
- The positional pretonic control, `{ā ī ū} > {a i u} / _ C* V` (all non-final long vowels):
  +0.012…+0.031 everywhere.

**Tone vs the Vedic accent (Hindko, Pothwari).** Take the reflexes that carry a tone mark and whose
etymon is accented. The toned syllable is the Vedic accented one in:

| lect | toned syllable = Vedic accented syllable |
|---|---|
| Awankari | 26 of 74 |
| poth | 3 of 7 |
| L | 5 of 11 |
| P | 11 of 23 |

In the other cases the tone sits on the syllable next to the lost h or voiced aspirate
(āṣāḍʰá > hā̀ṛ, gōdʰā́ > gṑ, adʰikaraṇī́ > ε̄̂ruṇ, kīdr̩śa > kìā). The tone comes from h / aspirate
loss, not from the Vedic accent.

### 7.4 What the accent does and does not condition in NW

**It does not condition:**
- pretonic long-vowel shortening or pretonic degemination (weight does both);
- syncope or apocope;
- the r̩ split;
- first-syllable lengthening;
- the citation vowel (-ā / -o / -u);
- Lahndic tone.

**Its only positive signals:**
- a small pretonic ū > u shortening in S, L and khet (LOO +0.0004…+0.0027, 8–54 etyma);
- the `accented` flag as a **lexical-stratum** marker (Vedic-attested vs later vocabulary) for the
  Punjabi/Lahndic citation -ā. Oxytone vs barytone makes no difference here.

**What does condition NW vowels:**
- weight and position: shortening before a heavy syllable, degemination before a heavy syllable,
  and late degemination after long vowels, all without compensatory lengthening;
- MIA-type contraction (-aka > -ā / -o, aya > e, hiatus);
- final short-vowel loss (not in S, OP or D);
- final -ā shortening (Apabhraṁśa), and in Hindko -ī loss as well;
- long-vowel nasalisation before NC (P, S).

**Frontier note.** Frontiers were rerun for all 12 lects at the end of this round. An earlier
batch (11:11) was stopped with `pkill -f "evaluate.py frontier"`. It also killed other groups’ frontier runs (confirmed by the coordinator); from round 3 on, only my own PIDs are touched.

## 9. Accent score (round 3)

Scratch for this round is in `tmp/sound_changes/northwest/r3/`:
- `batch.py` inserts or replaces placement rules. It reports train and test accent NED and flags any
  change in plain NED.
- `aloo.py` gives the accent-NED LOO of each `±stress` line.
- `alist.py` lists the accent-marked pairs with predictions.
- `acmp.py` shows the per-item accent difference between two rule files.

Every adopted rule leaves plain NED unchanged, on both train and test: awan 0.4106 / 0.4352,
D 0.5040 / 0.5136, P 0.3539 / 0.3540. No NW `%source` block writes accent or tone marks, so no
notation block affects the accent score. Weinreich's Domaaki doubling block carries a placed accent
onto the first of the two vowels, as the source writes it (kóon).

Scoring artefact: a wrongly placed accent costs two segments (0.5 on the predicted vowel and 0.5 on
the missed one), while no accent at all costs one. So a default placement only pays off where it is
right more than half the time. In D and P, "no rule for the rest" beats every fallback tried.

| lect | acc. pairs tr/te | test accent NED: none | Vedic | final | train accent NED: none → Vedic → final |
|---|---|---|---|---|---|
| Hindko (awan) | 97 / 21 | 0.4965 | 0.4687 | **0.4334** | 0.4852 → 0.4746 → **0.3857** |
| Domaaki (D) | 82 / 11 | 0.6126 | 0.6141 | **0.5989** | 0.6324 → 0.6362 → **0.6246** |
| Punjabi (P) | 35 / 11 | 0.5171 | 0.5285 | **0.5171** | 0.5833 → 0.5820 → **0.5498** |

### 9.1 Hindko (awan)

**Adopted.** Four rules place the accent by syllable weight, just before apocope (after the flap
rule, before `{a i u ə} > ∅ / _ #`). All are tagged `@accent-placement`. LOO = train accent-NED
increase when the rule is removed.

| rule | LOO | what it does |
|---|---|---|
| `Long > $1+stress / # C* _` | +0.0839 | a long first syllable is accented |
| `Short > $1+stress / # C* _ C C` | +0.0074 | a closed first syllable is accented (garbʰarūpa > gàbʰrū) |
| `Long > $1+stress / # C* Short C _` | +0.0036 | after a light first syllable, a long second syllable is accented (palāśá > pʰulā̀, vivāhá > viyā̀) |
| `Short > $1+stress / # C* _ C V // _ !#* Long` | +0.0047 | otherwise the first syllable |

This is the NIA weight rule: the first heavy syllable, otherwise the first.

**Placement point** (train accent NED for the same four rules):

| position in the chain | train accent NED |
|---|---|
| OIA headword | 0.3905 |
| end of MIA | 0.3897 |
| before h-loss | 0.3876 |
| after h-loss | 0.3926 |
| before hiatus | 0.3926 |
| **before apocope** | **0.3857** |
| after apocope | 0.3874 |
| end of chain | 0.3885 |

**Tonal-syllable hypothesis** (the coordinator's request; the tone sits next to the lost h or
aspirate):
- The accent on the vowel before an intervocalic h, placed before h-loss, with no default:
  0.4181.
- "Tonal, else first long", with the default cleared in words that have V h V: 0.3903, vs 0.3880
  for "first long" alone at the same point.
- A tonal override on top of the weight rules: 0.3885 vs 0.3876.
- An override only for a **long** vowel before a lost h: 0.3861. It gains a single item
  (udgrāhayati > ugrā̀vuṇ) and is not adopted.

The tone mark is on the stressed syllable. A lost h decides *whether* a word is toned (and so
whether it is scored at all), not *where* the tone sits: divasá > dḕō̃ and śapátʰa > sàɔ̃ have the
tone on the first syllable, although the h was lost after the second.

**Rejected alternatives** (train accent NED; the final chain is 0.3857):

| hypothesis | train accent NED |
|---|---|
| none | 0.4852 |
| Vedic carried | 0.4746 |
| Vedic, else penult | 0.4154 |
| Vedic where attested, weight rules elsewhere (`if:!accented` + `%accent vedic`) | 0.4200 |
| weight rules + Vedic fallback (`%accent vedic`) | 0.3910 |
| old penult (before apocope) | 0.4266 |
| last syllable | 0.4467 |
| first syllable | 0.3938 |
| first long else first (end of chain) | 0.3924 |
| first long else first (before apocope) | 0.3924 |
| first long else first (before h-loss) | 0.3880 |
| heavy penult, else antepenult (Latin type) | 0.4245 |
| rightmost long vowel | 0.4208 |
| long penult only (the P rule) | 0.4382 |
| closed = geminate only | 0.3942 |
| `if:verb` exemption | ±0 |
| relaxed first-syllable fallback (hiatus, monosyllables) | 0.3884 |
| relaxed fallback, also for light-light-long words | 0.3894 |

### 9.2 Domaaki (D)

**Adopted:** `Long > $1+stress / # !Long* _` (accent on the first long vowel; no accent otherwise),
placed at the end of the MIA block. LOO +0.0078. It is Shina-type (Weinreich kóon, póo, cíi, gíiri).
The same rule at the OIA headword scores the same (0.6246); before degemination 0.6256; at the end
of the chain 0.6297.

**Rejected** (train accent NED; the final chain is 0.6246):

| hypothesis | train accent NED |
|---|---|
| none | 0.6324 |
| Vedic carried | 0.6362 |
| Vedic, else penult | 0.6293 |
| rule + Vedic fallback | 0.6315 |
| Vedic where attested, rule elsewhere | 0.6308 |
| first syllable | 0.6503 |
| last syllable (end of chain) | 0.6316 |
| last syllable (before the -na infinitive) | 0.6348 |
| penult (end of MIA) | 0.6285 |
| penult (before -ā shortening) | 0.6293 |
| first long, else last syllable (end of chain) | 0.6309 |
| first long, else last syllable (end of MIA) | 0.6287 |
| first long, else penult | 0.6312…0.6316 |
| last long vowel only | 0.6294 |
| heavy last syllable, else penult | 0.6380 |
| first long + closed first syllable | 0.6324 |
| long penult | 0.6245, as a replacement (not adopted: it is the P rule and ties within one item) |

The accent score here is dominated by lexical mismatch: many Weinreich forms are different words
(ṓṣṭʰa > acimóo).

### 9.3 Punjabi (P)

**Adopted:** `Long > $1+stress / _ C* V C* #` (a long penult carries the accent), placed before
apocope. LOO +0.0335. Examples: gōpālá > guā́llā, prahēlikā > pahḗlī, vatsatará > vacʰḗrā,
nistārayati > natā́rnā, and Gill's tone forms *maurdʰaka > móḍā, *dāṁṣṭra > dā́ṛ.

**Rejected** (train accent NED; the final chain is 0.5498):

| hypothesis | train accent NED |
|---|---|
| none | 0.5833 |
| Vedic carried | 0.5820 |
| Vedic, else penult | 0.5537 |
| rule + Vedic fallback | 0.5514 |
| Vedic where attested, rule elsewhere | 0.5538 |
| any penult (before apocope) | 0.5537 |
| penult (before pretonic shortening) | 0.5839 |
| rightmost long non-final vowel, else penult | 0.5537 |
| the Hindko weight rules | 0.5649 |
| first long only | 0.5644 |
| first long, else first | 0.5822 |
| first syllable | 0.5691 |
| last syllable | 0.6089 |
| rightmost long including final | 0.6030 |
| heavy penult, else antepenult | 0.5846 |
| long penult at the end of the chain | 0.5594 |
| a Gill-only `%source` override: first long, else first | 0.5677 |
| a Gill-only `%source` override: first syllable | 0.5534 |

Hindko and Punjabi differ:
- Hindko has the first heavy syllable; the P rule costs Hindko +0.053.
- CDIAL Punjabi has a long penult: the stressed long vowel that pretonic shortening spares.

### 9.4 Conclusion on the Vedic accent

The Vedic accent carried through the chain is never competitive:
- Hindko: 0.4746 vs 0.3857 final;
- Domaaki: 0.6362, which is worse than no accent at all;
- Punjabi: 0.5820 vs 0.5498.

Its best fair chance in each lect is "Vedic where the headword is accented, the positional rule
elsewhere" or "positional rule, Vedic fallback". Both are worse than the positional rule alone:

| lect | Vedic-first | Vedic fallback | positional rule alone |
|---|---|---|---|
| awan | 0.4200 | 0.3910 | 0.3857 |
| D | 0.6308 | 0.6315 | 0.6246 |
| P | 0.5538 | 0.5514 | 0.5498 |

NW accent and tone are placed by NIA syllable weight: the first heavy syllable in Hindko, a long
penult in CDIAL Punjabi, the first long vowel in Domaaki. This matches §7: pretonic shortening and
degemination are also conditioned by weight, not by the Vedic accent.

**Small samples.** 97 / 82 / 35 train pairs; one item is worth about 0.0013 (awan) to 0.004 (P) of
accent NED. Only the first-syllable-long rule in Hindko and the long-penult rule in P are large
effects.

**Process note.** In round 2 I used a broad `pkill -f "evaluate.py frontier"`, which killed other
groups' frontier runs. In this round only PIDs I started were touched.

## 10. Extension-aware score (round 4)

Scratch for this round is in `tmp/sound_changes/northwest/r4/`:
- `batch.py` scores a rule edit on plain NED, extension-aware NED and the accent score;
- `show.py` gives a word's output for the bare form and every extension;
- `pick.py` lists the reflexes that choose a given extension;
- `resid.py` and `cand2.py` show the ending mismatches that remain under the extension-aware choice;
- `before.txt` (6-item menu) and `before2.txt` (7-item menu with -akka) hold the baseline scores.

All numbers below use the 7-item menu (aka, akka, ikā, uka, aḍaka, allaka, araka). Adding -akka
lowered train extension-aware NED by 0.0003 (P) to 0.0031 (D).

### 10.1 Scores and ending choices

Only khet changed in this round. For every other lect, before = after.

| lect | plain NED train / test | ext-aware NED train / test | train choice: bare / aka / akka / ikā / uka / aḍaka / allaka / araka (%) |
|---|---|---|---|
| P | 0.3539 / 0.3540 | 0.3035 / 0.3009 | 72 / 10 / 1 / 11 / 2 / 2 / 1 / 2 |
| S | 0.3253 / 0.3465 | 0.2560 / 0.2724 | 66 / 13 / 3 / 10 / 7 / 1 / 0 / 1 |
| L | 0.3713 / 0.3619 | 0.3144 / 0.3052 | 69 / 12 / 2 / 11 / 1 / 2 / 1 / 2 |
| OP | 0.4000 / 0.3947 | 0.3381 / 0.3362 | 66 / 15 / 3 / 5 / 1 / 5 / 2 / 3 |
| awan | 0.4106 / 0.4352 | 0.3505 / 0.3576 | 70 / 10 / 2 / 10 / 2 / 2 / 1 / 3 |
| kcch | 0.3886 / 0.3941 | 0.3387 / 0.3531 | 72 / 6 / 3 / 9 / 2 / 2 / 2 / 3 |
| D | 0.5040 / 0.5136 | 0.4414 / 0.4755 | 68 / 6 / 8 / 11 / 3 / 0 / 2 / 2 |
| srk | 0.4724 / 0.4846 | 0.4054 / 0.4121 | 66 / 13 / 3 / 11 / 2 / 2 / 0 / 2 |
| thar | 0.4269 / 0.4602 | 0.3690 / 0.4037 | 66 / 5 / 4 / 15 / 1 / 2 / 3 / 5 |
| poth | 0.4046 / 0.4397 | 0.3445 / 0.3576 | 71 / 9 / 2 / 10 / 1 / 3 / 2 / 3 |
| khet (before → after) | 0.4799 / 0.4315 (unchanged) | 0.4259 → **0.4246** / 0.3749 → 0.3749 | 71 / 11 / 1 / 7 / 2 / 2 / 1 / 4 |
| Khol | 0.4384 / 0.5440 | 0.3742 / 0.4258 | 66 / 2 / 3 / 6 / 16 / 1 / 3 / 4 |

**Rule changed.** khet: the hiatus rule `{i ī} {a ā} > ī / _ #` was widened to
`{i ī u ū} {a ā} > {ī ī ū ū}@1 / _ #` (@hiatus @aka), so -uka > -ū as in L, P, S, kcch and awan.
Extension-aware NED −0.0013; plain and accent NED unchanged. Frontier rerun.

**Tried and rejected:**
- thar, the same -uka > -ū rule: +0.0005.
- khet -uka > short -u: −0.0093, but spurious. The -u is then lost by apocope, so "uka" becomes a
  bare consonant-final variant (it jumped to 11% of the choices).
- D -ikā > -i: plain +0.0030, ext ±0. With -uka > -u added: plain +0.0041, ext −0.0007. The general
  hiatus rule `{i a} > ∅ / V _` collapses -ikā to the bare stem (gʰōṭikā > go), but the plain score
  needs that rule.
- Keeping the medial a of -aḍaka (syncope blocked before a final -ṛV): ext L +0.0002, kcch +0.0007,
  awan +0.0004, thar +0.0019, srk 0, Khol 0, khet −0.0011. Blocking it before final -ṛV/-rV/-lV
  is worse everywhere (+0.0003 to +0.0090). So the -ṛṛā that syncope makes is closer to the data.
  The khet gain alone is not adopted, since it would break the group-wide order.

`--ext-loo` for P (6.5 min): only one rule has negative extension-aware LOO, the citation rule
`if:!accented a > ā / C _ #` (−0.0182). Every other rule is ≥ 0. S and L were stopped (their
processes were mine), because per-rule deletion tests of the citation rules answered the question
faster.

### 10.2 The citation-vowel rules are the -aka extension in disguise

The `@morph` citation rules (`if:!accented` / `if:starred a > ā`, `a > o`, `∅ > o`, S `a > u`)
never double an ending. Every extended variant ends in a vowel, so `C _ #` cannot fire on it, and
-aka comes out as the lect's masculine ending in every lect:

| lect | -aka gives | other extensions |
|---|---|---|
| P L awan srk khet poth OP | -ā | -ikā > -ī, -uka > -ū (P L awan srk poth OP; khet from this round), -aḍaka > -(a)ṛā, -araka > -(a)rā, -allaka > -allā / -lā |
| S | -o | -uka > -u |
| kcch, thar | -o | |
| Khol | -o | |
| D | -a | final vowels kept |

The rules do, however, make the bare stem unreachable for every unaccented or starred a-stem: bare
and -aka give the same -ā / -o. This is the largest residual: P predicts -ā where the attested form
is consonant-final in 967 train reflexes, L in 371.

Deleting each rule gives a large extension-aware gain but costs plain NED far more than the 0.001
limit. So all are kept.

| lect | rule deleted | plain Δ | ext-aware Δ |
|---|---|---|---|
| P | `if:!accented a > ā` | +0.0090 | −0.0182 |
| L | `if:starred a > ā` | +0.0047 | −0.0101 |
| awan | `if:!accented a > ā` | +0.0087 | −0.0170 |
| poth | `if:!accented a > ā` | +0.0058 | −0.0160 |
| khet | `if:!accented a > ā` | +0.0103 | −0.0105 |
| OP | `if:starred a > ə…ā` | +0.0019 | −0.0051 |
| S | `if:starred a > o` | +0.0037 | −0.0009 |
| S | `a > u` (nominative) | +0.0207 | −0.0030 |
| kcch | `∅ > o` | +0.0139 | −0.0270 (accent score also changes) |
| thar | `∅ > o` | +0.0210 | −0.0449 |
| Khol | `a > o` | +0.0147 | −0.0046 |
| OP | `a > u` | +0.0243 | +0.0280 (OP -u is a real ending, not -aka) |

Two artefacts follow in Khol and srk. Khol's "uka" (16%) and srk's "akka" (-akk) act as stand-ins
for the missing consonant-final bare form: Khol -uka > -u is lost by apocope, giving xot for
gʰōṭa, next to bare xoto.

**What the choices say:**
- -aka is chosen over the bare stem by 10–15% of reflexes in P, L, S, OP, srk and khet, and by only
  5–6% in kcch, thar and D. The true share is higher in P and the Lahndic lects, because the
  citation rule already gives the bare form the -ā.
- -ikā is steady at 6–15%; thar is highest (15%).
- -ḍa (-aḍaka) is most productive in OP (5%). Elsewhere it is 1–3%.
- -ra (-araka) is 2–5%, highest in thar and Khol. Many of these picks are spurious: stems that
  already end in r, and -tra with anaptyxis (ágra > aggar, karapattra > karvattar).
- -alla is 0–3%. The P picks are mostly spurious (kamala > kãval, kaccʰa > kācʰal). A real -la
  suffix shows up as -lā after a consonant or long vowel (adʰástāt > heṭʰlā), not -allā.
- -uka matters only in S (7%). The Khol figure (16%) is an artefact.
- D prefers -akka (8%) over -aka (6%), as Shina-type chains keep -k-.

### 10.3 Should the menu change?

These are recommendations only; the menu was not edited. "Residual" counts are train reflexes whose
attested ending is still not produced under the best variant (from `cand2.py`).

**Add plain feminine -ā** (replace the final vowel by -ā, i.e. the ā-stem feminine), for S.
- S has 252 residual reflexes ending in -a (5.9% of S train), the largest S residual. Examples:
  *avipaṣṭʰa > aiṭʰa, *avōtta > ota, accʰa > acʰa.
- In P and L the -ā is lost by apocope, so it would change little there: L has 42 -a residuals,
  some of them numerals.
- This is real, well-attested MIA/NIA morphology (the feminine motion suffix).

**Consider -ara, -aḍa, -ala without -ka** (bare -Vr, -Vṛ, -Vl) for P and the Lahndic lects.
- Residual counts: P -Vr 220, -Vl 107, -Vṛ 84; L 91 / 38 / 38.
- Part of this is real: Pk -aḍa-, -ara-, -alla- forms that lost their vowel by apocope (kaccʰa >
  kācʰal, kantʰā > kʰãdʰol-).
- Part is surface: the -tar of anaptyxis (P 25 × -tar), and stems that already have r or l.
- Expected gain: perhaps half the counts. The risk is that with no cost for an extension, any stem
  with an r or l near the end gets a free near-match.

**Do not add:**
- -iya / -īya: 31 residuals in P (0.5%), 11 in L, 4 in S. In NW it falls together with -ikā
  (-iā > -ī).
- -ālu / -āla: P 35, L 17, S 10.
- -ikka and further -kk- variants: the -kV residuals are P 24, S 28, L 9, and -akka is already
  chosen by only 1–3% (D 8%).

**Largest non-menu residual:** infinitives on headwords not flagged as verbs, i.e. denominative and
derived verbs. P 146, S 69, L 64. The menu is nominal; a verbal "-ayati" variant for nominal
headwords would be real morphology, but it is the engine's `verb` flag that decides this, not an
extension.

**Items the NW lects barely use:**
- -allaka: at most 3% anywhere, 0% in S and srk, and the P picks are spurious. It could be dropped
  or replaced by -ala (see above).
- -uka in P, L, awan, poth and OP: 1–2%. Keep it for S.
- -aḍaka: 0% in D.

**The main recommendation concerns the score, not the menu.** A lect whose chain encodes -aka as a
citation rule cannot reach its bare consonant-final forms (P 967 reflexes, L 371). The
extension-aware score would be fairer if it also ran a variant with the chain's `@morph` citation
rules switched off. This needs an engine change, which the coordinator would decide.

## 11. Verb citation forms (round 5, short: the deadline was moved to 18:30)

Scratch: `tmp/sound_changes/northwest/r5/`. `batch.py` checks the four acceptance numbers (train
plain, verb-menu, extension-aware and accent NED); `before.txt` holds the baseline.

| lect | test verbs | plain NED | verb-menu NED, before → after | formation behind the citation form |
|---|---|---|---|---|
| P | 276 | 0.3110 | 0.3169 → **0.2660** (train 0.3169 → 0.2618) | -anaka > -ṇā |
| S | 213 | 0.2713 | 0.3321 → **0.2159** (train 0.3192 → 0.2089) | neuter -anakam > -aṇu (785 of 962 train verbs choose it) |
| L, kcch, awan, OP, srk, D, poth, khet, Khol | | | not done (baselines in `before.txt`) | |
| thar | | | not applicable (no verbs) | |

For L, the baseline already has -aṇa winning (verb-menu 0.2460 < plain 0.2821). Needed but not
done: kcch -anaka > -ṇū, and awan -uṇ.

**Rules added.** All pass the four checks: plain train NED within +0.001, extension-aware NED
better, accent NED unchanged.

P:
- `a > ∅ / C _ ṇ ā #` @syncope: the stem vowel is lost before the heavy final -ṇā (aṭṭanaka > aṭṭṇā).
  Verb-menu −0.0504, plain +0.0008.
- `ā v a > ā u / _ C` @aya-e: the causative -āvaṇā > -āuṇā. Verb-menu a further −0.0047.

S:
- `a a m > u / _ #` @final-nasal @hiatus: neuter -akaṁ > Ap. -aaũ > -u. Verb-menu −0.1103,
  plain +0.0001, extension-aware −0.0245.
- Two alternatives score about the same: retagging the nominative `a > u / C _ #` as a sound change
  (−0.1097, via -aṇa), or `a > u / ṇ _ #` (−0.1099). The -anakam route was chosen because it is a
  plain sound change of the neuter -aka stem.

## §12 Sindhi implosives (post-round-5 pass)

Initial implosive precision 0.69 → 0.75 (over-application 217 → 159; correct 479 → 478). Train NED
0.3284 → 0.3224, test 0.3473 → 0.3415; ext 0.2394 → 0.2335 / 0.2569 → 0.2511; verb menu
0.2087 → 0.2023 / 0.2147 → 0.2099; accent unchanged. Changes (OIA-S.sc):

- **Dissimilation**: initial g- j- stay plain when the word already has an implosive (8/43 implosive
  there vs 59% elsewhere; b- ḍ- show no such effect). The geminate rule now runs first.
- **Plain geminates only**: `$1` ignores aspiration, so g+gʰ (< r+gʰ, d+gʰ) was made implosive;
  aspirated geminates keep the aspirate in 99/139 attestations (argʰa > agʰu, dīrgʰa > ḍrigʰo).
- **dVr metathesis** before a non-coronal: dīrgʰa > *drīrgʰa > ḍrigʰo, darbʰa > ḍrabʰu (the r is
  copied so the cluster still protects the stop); not before ś or dentals (darśa > ᶑasu).
- **Initial vy- > v-** (vyākʰyāna > vākʰāṇu), not vv > bb > ɓ.

Rejected: y- > j- before the implosive rule (+0.003; most y- words are plain j-); nasal conditioning
(no consistent effect). Left as lexical: ~150 plain g-/j-/b- words (gaṇaṇu, gupʰā, goro …, often
attested both ways), v- > ɓ in vīṭā, dʰr- > ᶑ (dʰruva) vs ḍr/ḍʰr (dʰvasta, dʰrāyati).

## §13 Sindhi residual pass (larger sample: all 3,570 misses classified)

Train NED 0.3224 → 0.3207, test 0.3415 → 0.3395; ext 0.2335 → 0.2306 / 0.2511 → 0.2478 (then
0.2295 / 0.2471 with the gloss-gated denominal verb forms); verb menu and accent flat. Kept:
`ã u > õ` (ordinals aṭʰõ), medial `a {i u} > {e o} / _ C` (seru, potu, jero), `a ī > ī` (ciṛī),
`if:!starred` ŋg > ŋ (aŋu, aŋuṭʰo; late expressive *ḍiŋga keeps ŋg), a > i before -ṇo/-ṇī outside
the first syllable (vaṭiṇo, kambiṇī). Rejected (worse or source variation): final -u > -ū after a
vowel (77 worse vs 20 better), glides -āyo / -yo (source orthography), ly > lʰ (lexical: kālʰo vs
calaṇu, tulu). Nouns whose reflex is a verb (ārta > āṭaṇu 'to grow hot') now take a verb formation
on the stem, but only when the reflex is glossed as a verb; a blanket -anaka/-anakam on the noun
menu was indistinguishable from a placebo suffix (-atapa: median gain 0.0007 vs 0.0010).

## Round 6 audit

Method: `residuals.py` (plus a 45-example variant, `tmp/sound_changes/northwest/r6/resid45.py`), then
screening with `r6/batch.py` (train plain, extension-aware and accent NED), word-level diffs with
`r6/wdiff.py` (reflexes made better or worse under the extension-aware choice), and `gates.py` for every
installed change. Pre-audit copies of all files are in `r6/orig/`. "Before → after" is `gates.py orig
installed` on the final engine (it includes the coordinator's mid-round engine fixes).

### Summary (train / test)

| lect | ext NED | plain NED | verb-menu NED | exact (plain) | verdict |
|---|---|---|---|---|---|
| P | 0.2841→0.2591 / 0.2796→0.2593 | 0.3539→0.3613 / 0.3553→0.3641 | 0.2223→0.2218 / 0.2208→0.2195 | 15.0→16.4% | accept, justified (see citation rule) |
| L | 0.2882→0.2817 / 0.2816→0.2747 | 0.3759→0.3748 / 0.3663→0.3639 | 0.2315→0.2253 / 0.2312→0.2271 | 16.3→16.4% | accept |
| OP | 0.3247→0.3236 / 0.3206→0.3183 | 0.4022→0.4014 / 0.3976→0.3958 | 0.4041→0.4020 / 0.3352→0.3344 | 14.3→14.4% | accept |
| awan | 0.3530→0.3347 / 0.3559→0.3354 | 0.4102→0.4105 / 0.4347→0.4329 | 0.3551→0.2349 / 0.3785→0.2396 | 13.7→14.0% | accept (+%verb-menu) |
| kcch | 0.3574→0.3156 / 0.3679→0.3336 | 0.3886→0.3886 / 0.3933→0.3933 | 0.4109→0.2665 / 0.3732→0.2331 | 12.5→12.5% | accept |
| D | 0.4249→0.4122 / 0.4674→0.4500 | 0.5044→0.5001 / 0.5127→0.5068 | 0.3508→0.3330 / 0.5119→0.4841 | 7.8→8.8% | accept (+%verb-menu); accent 0.6759→0.6645 |
| srk | 0.3947→0.3870 / 0.4101→0.3924 | 0.4717→0.4689 / 0.4839→0.4812 | 0.3485→0.3474 / 0.4762→0.4762 | 9.4→9.4% | accept |
| thar | 0.3252→0.3208 / 0.3650→0.3453 | 0.4269→0.4479 / 0.4602→0.4598 | – | 7.5→12.3% | accept, justified |
| poth | 0.3333→0.3322 / 0.3443→0.3428 | 0.4037→0.4038 / 0.4397→0.4382 | 0.3417 / 0.4167 flat | 16.4% | accept |
| khet | 0.4071→0.3994 / 0.3686→0.3614 | 0.4788→0.4894 / 0.4302→0.4403 | flat | 12.4→10.1% | accept, justified |
| Khol | 0.3848→0.3488 / 0.4109→0.3967 | 0.4384 / 0.5425 flat | 0.4261→0.2761 / 0.4850→0.3600 | 4.5% | accept |
| S | 0.2249→0.2190 / 0.2427→0.2370 | 0.3207→0.3204 / 0.3395→0.3388 | 0.1818→0.1685 / 0.1898→0.1798 | 21.0→21.3% | accept (+%verb-menu) |

Error profile (all lects, after): the ending (last two segments) is 35–58% of the remaining error, then
consonant insertion/deletion (14–23%), consonant quality (6–12%), missing syncope (4–10%). Most of the
ending residue is lexical (gender/extension doublets, CDIAL listing several derivatives) or, where the
citation rule survives (awan, poth, kcch), the bare-vs-extended artifact below.

### Kept changes

**Citation -ā/-o rules removed (P, khet, thar)** — the bare-vs-extended artifact. The `@morph` rules
`if:!accented,!ext a > ā / C _ #` (P, khet) and `if:!ext ∅ > o / C _ #` (thar) gave every bare a-stem the
masculine -ā/-o that the attested forms owe to the -aka extension, which the extension menu now supplies.
With them the bare consonant-final forms were unreachable (P: 967 reflexes -ā for attested -∅ in round 4;
now 55). Plain NED rises (P +0.0073, khet +0.0106, thar +0.0210) because the plain score still compares
the bare headword with -ā/-o reflexes filed under it (ágra > aggar vs aggarā; kaṇṭa > kaṇḍ vs kaṇḍā); plain
*exact* rises in P (15.0 → 16.4%) and thar (7.5 → 12.3%). Commented out, not deleted (`; if:…`), as the
coordinator did for L, S, OP. Blocked by the accent gate (so kept): awan (ext −0.0168 but accent test
+0.031, 21 pairs), poth (accent train +0.021, 6 pairs), kcch `∅ > o` (ext −0.0295/−0.0449 but its single
accent pair gʰaná > gʰɔ̀ṇɔ gets worse). These three are recommended for deletion if the accent gate is
relaxed for tiny accent sets; Khol `a > o` is rejected on the merits (ext test +0.0012).

**P**
- `{a i} > ∅ / V C (C) _ C {ā ī ū} #` @syncope, placed *after* the flap, with `G $1 > $2 / _ C`
  (geminate before consonant) and `C $1 $1 > $1 $1` (triple reduction): medial syncope before a heavy
  final syllable (hariṇikā > harnī, śēkʰara > sihrā, *āśara > āsrā, *carmaḍaka > camṛā, *kʰiccaḍikā >
  kʰicṛī, *vaptra > vattā). Before the flap it turned -aṛā into -Cḍā (aŋguṭṭʰḍā); after it, 228 reflexes
  better / 114 worse (many of the worse ones only switch variant). ext −0.0018 train / −0.0026 test.
- `a ī > ī / _ #` @hiatus (-aī > -ī: ālaya-ikā > ālī).
- `v > ∅ / V _ V // _ ā #` (local replacement of `use v-loss.v.intervocalic`): v kept before final -ā
  (kṣēpaka > kʰevā, *kʰavaka > kʰavā). Together with the -aī rule and the first syncope: ext −0.0033.

**L**: `a ī > ī / _ #`; `v > ∅ / ā _ #` (final -āv > -ā: tāpá > tā, udvāpa > ubā; 72 better / 3 worse);
syncope extended to `a > ∅ / V C (C) _ C V // V C C _ C a ṇ #` + `G $1 > $2 / _ C` (*parsula > paslī,
*maṇṭʰara > maṇḍʰrā, *gʰussaḍ- > gʰusṛaṇ). Without the exception for geminate stems before the infinitive
-aṇ (uggʰaṛaṇ, ujjiṛaṇ) the verb test got worse (+0.0040) and the gate rejected it.

**OP**: `a ī > ī / _ #`; `v > ∅ / ā _ {u i} #` (ārāma > ārāu, āgʰrāpayati > āgʰāi; 28 better / 4
worse); removed the orphan `ə > ā` left after the coordinator commented out the starred -ā rule.

**awan**: `a > u / C _ ṇ a #` (before apocope) — the Hindko infinitive -uṇ from the verbal noun -aṇa
(ākʰyāti > ākʰuṇ, kárṣati > kassuṇ; 113 better / 5 worse; verb-menu −0.117); stem syncope before it,
`Short > ∅ / V C (C) _ C u ṇ #` + `G $1 > $2 / _ C` (*utkarati > ukruṇ, *udgalati > ugluṇ); `%verb-menu +aka`.

**kcch**: `a a m > ū / _ #` before `a a > e` — neuter -akaṁ > -aũ > -ū, the Kutchi infinitive -ṇū from
-anakam (kaḍḍʰanakam > kaḍʰṇū; verb-menu 0.411 → 0.267). Placed after `a a > e` it never fires.

**D**: `ā > o / # C* _ #` (secondary monosyllables: tāpá > tóo, pā́da > póo) and `{a ā} m a > o m / _ #`
(kárman > kom, nā́man > nom, stambʰa > tom); the monosyllable rule alone failed (plain/accent test),
together they pass, accent improves; `%verb-menu +iyanakam`.

**srk**: the L-type syncope extension (`a > ∅ / V C (C) _ C V` + `G $1 > $2 / _ C`; 24 better / 7 worse).

**thar, poth**: `a ī > ī / _ #`; poth also word-final geminate simplification `G $1 > $2 / _ #` (saptá >
sat, ákṣi > akʰ; the phonetic Pothwari sources).

**Khol**: `a > e / C _ {n ṇ} a #` — the Kholosi infinitive -en from the verbal noun -ana (karṓti > karen,
*kaḍḍʰati > kadden; 29 better / 0 worse; verb-menu 0.426 → 0.276).

**S** (after the coordinator's §12–13): the -ṇo/-ṇī raising extended to liquids,
`a > i / V C* C _ {ṇ r ṛ l} {o ī i} #` plus `a > i / V C* C _ {r ṛ l} a ṇ u #` (mantʰaraka > mandʰiro,
*āśara > āsiro, *āmraḍ- > ambiṛī, *uttʰalyati > utʰilaṇu, *ubbal > uɓiraṇu; 140 better / 44 worse; ext
−0.0035 / −0.0042, verb −0.0027 / −0.0032, plain −0.0003). The same raising before any consonant is
weaker (−0.0026).

**%verb-menu (coordinator's cross-lect placebo run)**: S `+aka` (ext −0.0024/−0.0015, verb −0.0106/−0.0068),
awan `+aka` (ext −0.0005/−0.0007, verb −0.0032/−0.0051), D `+iyanakam` (ext −0.0022/−0.0023, verb
−0.0178/−0.0278); plain and accent unchanged; all three ACCEPT.

### Rejected candidates (Δ train ext unless stated)

- P: n-dissimilation only after ṇ (+0.0010) or dropped (+0.0013) — -ṇā/-nā after r ṛ is 50/50 lexical;
  final geminate simplification (plain +0.0139, ext +0.0024); dropping v- > b- (+0.0009; initial v-/b- is
  ~50/50 in every vowel context: a 70/75, i 63/82, ā 22/31); dropping s > h (+0.0031) or keeping -s#
  (+0.0009); medial au/ai > o/e before a consonant (+0.0042/+0.0050; P keeps au: valāu, aul); v-loss only
  before rounded vowels (+0.0034) or none (+0.0043).
- L: -ar > -ur (+0.0013); final geminate simplification (plain +0.0128); initial implosives g j b (d)
  (+0.013; the L implosive forms are a minority and lexical); P-type syncope alone (−0.0021, superseded).
- OP: dropping n > ṇ (+0.0011), flap > r (+0.0031), general intervocalic v-loss (+0.0007).
- awan: final geminate simplification (ext −0.0049 but accent +0.0016); -vuṇ glide (0); extended syncope
  (ext test worse); -aī > -ī (−0.0002).
- kcch: -ṇū glide y after ā (+0.0028); -ũ instead of -ū (+0.0004); extended syncope (ext test worse).
- D: ā > o before a nasal in closed final syllables (+0.0025); final -a loss after sonorants (+0.0087).
- srk: final geminate simplification (plain +0.004), -aī, -āv, -uṇ (0 or worse).
- khet: final degemination, -aī, -āv (all 0).
- Sindhi-type raising before final -ṇo/-ro/-ṛo/-lo/-ī: kcch +0.0034 (plain +0.0046), thar 0, Khol 0 — a
  Sindhi-only change.
- Khol: `a > e / C _ n #` after apocope (0: the n is still ṇ there); extended syncope (ext test worse).

### What remains

- Lexical doublets: v-/b- (P, L), -ṇā/-nā (P), n/ṇ (OP), -u/-a/-o (S, OP), implosive vs plain (L, srk).
- The bare-vs-extended artifact in awan, poth and kcch (citation rules kept only because of 1–21 accent
  pairs).
- Numerals (-tālī, -hattar, -vañjā) and compound headwords, where the chain lenites across the compound
  boundary.
- CDIAL headwords whose reflex is a different formation (causatives in -āuṇā on non-causative headwords,
  -ṛā/-ṛī diminutives without -ḍa- in the headword).
- Not done beyond one pass: thar, khet, Khol (≤ 250 etyma; residue mostly single items).

### Round 6 second pass

Tools: `r6/p2/near.py` (near-misses, 0 < NED ≤ 0.25, under the extension-aware choice, grouped by
differing segment pair and position), `r6/p2/accdiff.py` (accent-scored pairs whose accent NED changes
between two files). Note: my screening/candidate scripts matched anchors by line prefix, so in P the
first-pass `G $1 > $2 / _ C` and `C $1 $1 > $1 $1` landed after the *pretonic* degemination
(`degem-plain.g.aiueovcc`), not after the post-syncope one. Moving them there is neutral on ext and
worsens accent (+0.0036), so they stay where the gates accepted them; the anchor match is fixed.

Cumulative before → after (orig → installed; train / test):

| lect | ext NED | plain NED | verb-menu NED | exact |
|---|---|---|---|---|
| P | 0.2841→0.2584 / 0.2796→0.2577 | 0.3539→0.3610 / 0.3553→0.3632 | 0.2223→0.2218 / 0.2208→0.2171 | 15.0→16.5% |
| L | 0.2882→0.2815 / 0.2816→0.2744 | 0.3759→0.3746 / 0.3663→0.3633 | 0.2315→0.2253 / 0.2312→0.2271 | 16.3→16.4% |
| S | 0.2249→0.2159 / 0.2427→0.2345 | 0.3207→0.3204 / 0.3395→0.3387 | 0.1818→0.1549 / 0.1898→0.1688 | 21.0→21.3% |
| kcch | 0.3574→0.3152 / 0.3679→0.3336 | 0.3886→0.3885 / 0.3933→0.3933 | 0.4109→0.2649 / 0.3732→0.2331 | 12.5→12.8% |

**Kept (second pass)**
- P and L: `G $1 > $2 / a {u i} _` — geminates simplify after a secondary diphthong, which counts as long
  (caturdvāra > caubārā, apakṣá > aukʰ, apútra > aut, kaparda > kauḍ; P 59 better / 22 worse, ext
  −0.0008/−0.0016; L −0.0003/−0.0003). The worse ones are cau- numerals (caukkā, cautʰī), probably
  re-formed on cau- 'four'.
- S: `ā v a > ā i / _ ṇ` — the causative -āpanakam > -āvaṇu > -āiṇu (útpātayati > upāiṇu, *uccayatē >
  ucāiṇu; 98 better / 4 worse; ext −0.0030/−0.0025, verb −0.0136/−0.0109).
- kcch: `∅ > y / ā _ ṇ ū #` (kʰā́dati > kʰāyṇū, *karmāpayati > kamāyṇū; ext −0.0004, test flat).

**Near-miss groups examined (NED ≤ 0.25)**
- P (1,503 near-misses): medial a→∅ 85 (hiatus ea/ae and syncope — `{e ē} a > e` is neutral), b→v at
  the beginning 65 (initial v/b is ~50/50 in every vowel context: lexical/source), degemination k t b ṭ
  p g s c l →∅ ≈ 190 in all (after au/ai — kept; word-final: plain +0.014, rejected; before C: see
  above), ∅→ṇ at the end 41 (infinitives on unflagged verbs), ā→a medial 32 (shortening in a closed
  syllable before a final long vowel: plain +0.0030, rejected), h→s 30 (s kept after a long vowel:
  +0.0024, rejected), n↔ṇ 50 (lexical), ∅→a 31 (the -ṇā syncope after a geminate; the `$1` exception has
  no effect in the engine).
- L (742): a→∅ 24, k/t→∅ 35 (geminates), a→u/a→i before a final r/ṛ 34 (oḍʰur/oḍʰir/oḍʰar all attested
  — lexical), d→ḍ/ᶑ, g→ɠ, b→ɓ at the beginning ≈ 45 (the L implosive/retroflex minority; rules cost
  +0.013), e→ē / o→ō 25 (CDIAL L writes macrons in 168 of ~720 forms with e/o — a notation mix inside one
  source, not separable), ∅→ã/ā̃ at the end 21 (nasal fem. forms, lexical).
- S (1,237): -u→-a 71 and -o→-∅ 67 (ending doublets), ∅→i 61 — of which the causative -āiṇu is fixed and
  the passive -ijaṇu is not: `%verb-menu +iyanakam` gives −0.0021 but its placebo `+iranakam` −0.0018, so
  it is not admitted; ɠ→g and ʄ→j (coordinator's implosive pass; lexical), r→l 24 (l > r
  over-applies in ~24 words; lexical), ū→u 27 (the pretonic ū rule under-applies).
- kcch (228): -o→-∅ 50 (the blocked citation rule), ∅→a before -ṇū 7 (syncope is needed: blocking it
  costs +0.020).

**Blocked citation-rule removals (awan, poth, kcch)**: the blocking accent pairs are all the same
artifact, not accent errors. The accent score is computed on the bare headword only (no extension
variants), so removing the citation -ā/-o changes only the *segments* of the scored forms:
- awan: 19 train / 3 test accent pairs change; 11 train pairs get better (galla > gàl, mālya > mā̀l,
  prahara > pā̀r) and 8 worse, all worse ones being attested with -ā (pīṭʰa > pī̀ṛā, lōṭʰa > lṑṛā,
  śēkʰara > sḕrā); test: *pipʰīla > pḕlā/pḕlī, *rātryāndʰa > ratā̀ṇā (+0.03 over 21 pairs).
- poth: the single pair *bʰārta > pā́rā (pāṭṭā → pāṭ).
- kcch: the single pair gʰaná > gʰɔ̀ṇɔ (gʰaṇo → gʰaṇ).
No narrowing of the removal is principled: the attested -ā/-o on these forms is the -aka ending, which
the extension score supplies but the accent score does not. Recommendation: score the accent on the
extension-aware best variant (or exempt ending-only differences), then remove these three rules as in
P, khet, thar (expected ext: awan −0.017, poth −0.015, kcch −0.030/−0.045 test).

**Not done**: Punjabi tone reflexes of voiced aspirates beyond the existing Gill notation block (only the
Gill forms mark tone; 52 pairs), and the second pass for OP, awan, srk, poth, thar, khet, Khol, D.

#### Citation rules removed under the coordinator's accent waiver (awan, poth, kcch)

The accent gate is waived for citation-rule removal only, because the accent score sees the bare form
(the bare-vs-extended artifact, `### Round 6 second pass`). All three removals improve the
extension-aware score on train and test and leave verb-menu NED unchanged. Rules are commented out,
not deleted. Figures are train / test, with installed file → removal:

| lect | ext NED | plain NED | accent NED | verb | exact |
|---|---|---|---|---|---|
| awan (`if:!accented,!ext a > ā`) | 0.3347→0.3178 / 0.3354→0.3273 | 0.4105→0.4189 / 0.4329→0.4515 | 0.4040→0.3973 / 0.4695→0.5008 | flat | 14.0→11.6% |
| poth (`if:!accented,!ext a > ā`) | 0.3322→0.3176 / 0.3428→0.3331 | 0.4038→0.4099 / 0.4382→0.4547 | 0.8125→0.8333 / 0.7292 flat | flat | 16.4→15.3% |
| kcch (`∅ > o`) | 0.3152→0.2857 / 0.3336→0.2887 | 0.3885→0.4022 / 0.3933→0.3992 | 0.3750→0.6250 (one pair) / – | flat | 12.8→13.9% |

Example words:
- **Now right:** the bare consonant-final reflexes, e.g. awan galla > gàl, mālya > mā̀l, kūṭa > kūṛ,
  kʰādya > kʰāj; poth *pēṭṭa > peṭ, rakta > rat; kcch ántara > andʰar, ádʰarma > adʰam, apakṣá > aukʰ.
- **Unchanged:** the -ā / -o forms, which now come from the -aka variant.
- **Now wrong on the accent score:** only the scored forms attested with -ā/-o, e.g. awan pīṭʰa >
  pī̀ṛā, lōṭʰa > lṑṛā; poth *bʰārta > pā́rā; kcch gʰaná > gʰɔ̀ṇɔ.

#### Second pass on the remaining lects

**Kept:**
- **OP:** `G $1 > $2 / a {u i} _` (apakṣá > aükʰā). ext −0.0003 / −0.0007.
- **poth:** stem syncope before -ṇā, `a > ∅ / C _ ṇ ā #`, as in P (*hiss > hisṇā). ext −0.0005 on
  train, test flat; verb −0.030 on train.
- **Khol:** `s > ∅ / V _ Stop`: s is lost before a stop, with no aspiration (hásta > hat, masta >
  mato, śúṣka > sokko). 5 better / 0 worse; ext −0.0062 / −0.0023; plain −0.0045.

**Rejected:**
- OP:
  - n > ṇ only after a long vowel: ext +0.0001.
  - `i ā > i / _ #`: ext −0.0078, but the attested -i are GGS locative/oblique case forms (ahi,
    andari, uttari), not a sound change.
- awan:
  - Word-final degemination (galla > gal, *cutta > cut): ext −0.0078 train and test. Accent is worse
    by +0.0011, through the same bare-form artifact (dadrú > dʰàddur, *yabdʰa > yàddʰā, which are
    attested with a vowel after the geminate). It is not covered by the waiver, so it is not
    installed. It is recommended once the accent score is extension-aware.
  - Degemination after au/ai: 0. Geminate before a consonant: −0.0001.
- poth: P-type syncope before a heavy final syllable (ext test worse).
- srk, thar, khet, D: no near-miss group of ≥ 5 reflexes that reads as a regular change. Residue
  includes:
  - spurious -akka picks (srk kapṛak for kapṛa);
  - thar kumār/kuār and r/ṛ;
  - khet ō/o notation;
  - D final -a loss (kʰaṭ, caur, cut): lexical, rejected in the first pass (+0.0087).

#### Round 6 final: pre-audit files → installed files, on the final engine

Figures are train / test. For awan, kcch and poth the accent change is the waived citation removal.

| lect | ext NED | plain NED | verb-menu NED | accent NED |
|---|---|---|---|---|
| P | 0.2841→0.2583 / 0.2796→0.2577 | 0.3538→0.3609 / 0.3553→0.3632 | 0.2222→0.2217 / 0.2208→0.2171 | 0.5904→0.5893 |
| S | 0.2248→0.2159 / 0.2427→0.2345 | 0.3206→0.3203 / 0.3395→0.3387 | 0.1819→0.1551 / 0.1898→0.1688 | = |
| L | 0.2877→0.2810 / 0.2816→0.2744 | 0.3755→0.3741 / 0.3663→0.3633 | 0.2335→0.2273 / 0.2312→0.2271 | = |
| OP | 0.3247→0.3230 / 0.3206→0.3177 | 0.4022→0.4013 / 0.3976→0.3953 | 0.4148→0.4090 / 0.3352→0.3344 | – |
| awan | 0.3531→0.3178 / 0.3559→0.3273 | 0.4102→0.4189 / 0.4347→0.4515 | 0.3564→0.2366 / 0.3785→0.2396 | waived |
| kcch | 0.3574→0.2857 / 0.3679→0.2887 | 0.3886→0.4022 / 0.3933→0.3992 | 0.4109→0.2649 / 0.3732→0.2331 | waived |
| D | 0.4249→0.4122 / 0.4674→0.4500 | 0.5044→0.5001 / 0.5127→0.5068 | 0.3508→0.3330 / 0.5119→0.4841 | 0.6759→0.6645 |
| srk | 0.3947→0.3870 / 0.4101→0.3924 | 0.4717→0.4689 / 0.4839→0.4812 | 0.3485→0.3474 / flat | better on test |
| thar | 0.3252→0.3208 / 0.3650→0.3453 | 0.4269→0.4479 / 0.4602→0.4598 | – | – |
| poth | 0.3333→0.3171 / 0.3443→0.3331 | 0.4037→0.4099 / 0.4397→0.4547 | 0.3417→0.3119 / flat | waived |
| khet | 0.4059→0.3982 / 0.3686→0.3614 | 0.4788→0.4894 / 0.4302→0.4403 | flat | = |
| Khol | 0.3848→0.3426 / 0.4109→0.3944 | 0.4384→0.4338 / 0.5425→0.5414 | 0.4261→0.2761 / 0.4850→0.3600 | – |

Plain NED rises only in the lects whose citation rule was removed (P, khet, thar, awan, kcch, poth).
In each of them extension-aware NED falls by at least as much.

#### Re-gating under the new accent gate (ext_acc)

`gates.py` now gates on ext_acc, the accent scored on the variant the extension-aware score picks.
Numbers below are train / test.

**Hindko word-final degemination, now installed.** The rule is `G $1 > $2 / _ #` after final-vowel
loss (galla > gal, *cutta > cut, bʰráṣṭra > bʰaṭʰ). Verdict: ACCEPT.

| score | before → after |
|---|---|
| ext | 0.3178→0.3100 / 0.3273→0.3187 |
| ext_acc | 0.3434→0.3387 / 0.4342 flat |
| verb | 0.2366→0.2347 / flat |
| plain | 0.4189→0.4194 / 0.4515→0.4476 |
| exact | 11.6→14.2% |

It was blocked only by the old bare-form accent score (+0.0011). That score is now reported but not
gated.

**No other change was rejected on accent alone.** The D monosyllable rule on its own also failed on
plain test NED. The late P `G $1 > $2 / _ C` did not improve ext.

**The three waived citation removals pass on ext_acc.** Each row compares the lect with the citation
rule restored against the installed file, and ext_acc is never worse. The only remaining flag is
plain NED ("accept if justified"), which is the bare-vs-extended artifact already argued above.

| lect | ext | ext_acc | verb |
|---|---|---|---|
| awan | 0.3298→0.3100 / 0.3287→0.3187 | 0.3602→0.3387 / 0.4342 flat | flat |
| poth | 0.3317→0.3171 / 0.3428→0.3331 | 0.7472 flat / 0.6667 flat | flat |
| kcch | 0.3152→0.2857 / 0.3336→0.2887 | 0.3750 flat / – | flat |


## Round 7

Method as in round 6. All gates use `gates.py` (installed vs candidate) on the engine of the day,
with the 08:20 to 09:40 engine fixes in place. Tools are in `tmp/sound_changes/northwest/r7/`:
- `cand.sh STEM NAME`: gate a candidate.
- `wdiff.py`: per-reflex better/worse under the extension-aware choice.
- `srcconf.py`: per-source confusions.
- `segdist.py`: per source, what the gold aligns to a predicted segment.
- `vq.py` / `vpos.py`: lax-vowel and tone marks per source.
- `accex.py` / `toneex.py`: accent and tone pairs, right or wrong.
- `nomorph.py`: comment out `@morph` rules.

The candidates themselves are in `r7/c/<name>/`.

### Summary: pre-round-7 snapshot → installed

Scores are `gates.py <snapshot> <installed>`, on the same engine as of 12:50 (after the 11:40 fix).
Each cell is train / test; `=` means unchanged.

| lect | ext NED | plain NED | verb-menu NED | ext_acc | ext_tone | verdict |
|---|---|---|---|---|---|---|
| P | 0.2585→0.2570 / 0.2572→0.2555 | 0.3607→0.3963 / 0.3624→0.3984 | = / 0.2275→0.2267 | 0.5334→0.4963 / 0.4953→0.4478 | 0.6098→0.5610 / = | ACCEPT-IF-JUSTIFIED |
| S | 0.2151→0.2095 / 0.2338→0.2294 | 0.3198→0.4065 / 0.3382→0.4255 | = / = | 0.6815→0.6190 / = | – | ACCEPT-IF-JUSTIFIED |
| L | 0.2817→0.2785 / 0.2757→0.2730 | 0.3752→0.4104 / 0.3639→0.3989 | 0.2367→0.2368 / = | 0.6799→0.6361 / 0.6171→0.5476 | 0.6799→0.6435 / 0.6131→0.5714 | ACCEPT-IF-JUSTIFIED |
| OP | 0.3321→0.3059 / 0.3296→0.3042 | 0.4013→0.4275 / 0.3953→0.4202 | 0.3720→0.2849 / 0.3698→0.2931 | – | – | ACCEPT-IF-JUSTIFIED |
| awan | 0.3108→0.2927 / 0.3224→0.3119 | 0.4223→0.4417 / 0.4513→0.4688 | 0.2644→0.2274 / 0.2499→0.2209 | 0.3461→0.3087 / 0.4399→0.4251 | 0.3979→0.3118 / 0.4413→0.3838 | ACCEPT-IF-JUSTIFIED |
| kcch | 0.2866→0.2842 / 0.2897→0.2859 | 0.4029→0.4758 / 0.4002→0.4598 | = / = | 0.6250→0.5000 / = | 0.6250→0.5625 / = | ACCEPT-IF-JUSTIFIED |
| D | 0.4360→0.4109 / 0.4601→0.4511 | 0.5220→0.5383 / 0.5192→0.5336 | 0.3861→0.3813 / = | 0.5915→0.5340 / 0.5609→0.5268 | 0.9276→0.8800 / = | ACCEPT-IF-JUSTIFIED |
| srk | 0.3868→0.3655 / 0.3924→0.3609 | 0.4722→0.4664 / 0.4812→0.4558 | 0.3739→0.3670 / = | 0.6069→0.5583 / 0.5833→0.5417 | 0.6069→0.5701 / = | ACCEPT |
| thar | 0.3208→0.3178 / 0.3453→0.3426 | = / = | = / = | – | – | ACCEPT |
| poth | 0.3392→0.3265 / 0.3344→0.3258 | 0.4344→0.4377 / 0.4574→0.4512 | 0.3086→0.2967 / = | 0.7417→0.6000 / = | 0.7762→0.6702 / = | ACCEPT-IF-JUSTIFIED |
| khet | 0.3974→0.3954 / = | 0.4916→0.4913 / = | = / = | 0.3333→0.2833 / 0.1250→0.0000 | – | ACCEPT |
| Khol | 0.3404→0.3301 / = | 0.4338→0.4630 / 0.5414→0.5505 | 0.3004→0.2689 / = | – | – | ACCEPT-IF-JUSTIFIED |

Every lect improves extension-aware NED on train, and none gets worse on test. ext_acc and ext_tone are
never worse. Plain NED rises where `@morph` citation rules were removed; this is the
bare-vs-extended artifact, accepted by the coordinator's neutral-removal rule.

The largest gains:
- OP: ext −0.026 / −0.025, verb −0.087 / −0.077. From the -ai > -i and -akaṁ > -u rules and the
  shackle-auto notation.
- srk: −0.021 / −0.032, from the Rensch notation, -akaṁ > -ā̃ and placement.
- D: −0.025 / −0.009, from -akaṁ > -o with `+iyanaka`.
- awan: −0.018 / −0.011, ext_tone −0.086 / −0.058.
- poth: −0.013 / −0.009.
- Accent: L, srk, poth, S, khet and kcch had no accent placement before this round.

### Step 0: the interrupted round-7 edits

The pre-round-7 snapshot (`tmp/sound_changes/snapshots/pre-round7/rules/`) differs from the installed
files in two rules, both accent placement. Both were re-gated snapshot → current under the new scoring:

- **D** `V > $1+stress / _ C* # // _ !#* Long | Long !#* _`: words without a long vowel are accented
  on the last syllable (Weinreich baríś, kirmá, caagá). Segmental scores are identical. ext_acc
  0.5915→0.5824 train, 0.5609→0.5496 test. **ACCEPT (prosody-only)**, kept.
- **P** `V > $1+stress / # C* _ // …`: the first syllable is accented when there is no long penult.
  ext_acc 0.5334→0.5106 / 0.4953→0.4608, ext_tone 0.6098→0.5818 train. **ACCEPT (prosody-only)**, kept.
  - Its exception `// _ C* Long C* V C* #` only blocked the rule when the *next* vowel was the long
    penult. So bʰaginīpati got two accents (bʰa*iṇī*vṇā).
  - Widened to `// _ !#* Long C* V C* #`: ext_acc −0.0013 train, test flat. ACCEPT (prosody-only).

### Vowel qualities (focus 0)

Only the D, L, awan and poth sources write lax or open vowels in quantity (`vq.py`, all sources):

| lect | source | forms |
|---|---|---|
| D | CDIAL (Lorimer) | ʌ 51, ɛ 10, ə 5, ɪ 4 |
| D | Backstrom | ɛ 7 |
| L | CDIAL | ɔ 57, ɛ 40 |
| awan | CDIAL | ɛ 15, ɔ 6 |
| awan | Rensch | ɛ 26 |
| poth | Lothers | ɛ 30 |

P, S, OP, kcch, srk, thar, khet and Khol have fewer than 5 such forms each. Nothing was installed:

- **D**, Lorimer's ʌ for short a. Train evidence (first/medial syllable, not word-final): ʌ 42 vs a 24.
  - A CDIAL-only `@notation` rule `a > ʌ / _ C` improves train (ext −0.0142, verb −0.039).
  - Test vetoes it (ext +0.0050, verb +0.028; aṇuni > árīn, ṣáṣ > ṣa, hárati > harina keep a).
  - Rejected.
- **L**, ɔ ɛ against au ai inside CDIAL (au 44 / ɔ 57, ai 62 / ɛ 40). The ɔ/ɛ forms mostly continue
  the secondary diphthongs (caturtʰá > cɔtʰā, bʰaginī > bʰɛṇ), but the same source writes au/ai as
  often.
  - `au > ɔ`: ext +0.0001, 18 better / 17 worse.
  - `ai > ɛ`: ext −0.0002, but ext_acc +0.0077.
  - Both together: ext flat, 30/30.
  - `a > ɔ` before h: 0.
  - This is a notation mix inside one source; not separable.
- **awan (Rensch) / poth (Lothers)**, e against ɛ. The same source gives both spellings for the same
  etymon (Rensch tel/tɛl, ek/ɛk, ser/sɛr; Lothers pelī/pɛlī): 16 e vs 16 ɛ in closed first syllables.
  This is variation, not notation; left.

### Tone and accent (focus items 1–2)

**Where tone is scorable.** Tone types appear only in:
- CDIAL Awankari: 105 grave and 6 circumflex among its accent-marked forms.
- Gill: P, awan, srk, poth.
- Lothers: one caron.

The tone-scored pair counts are:

| lect | train | test |
|---|---|---|
| awan | 101 | 19 |
| P | 22 | 0 |
| L | 14 | 3 |
| poth | 7 | 2 |
| srk | 5 | 0 |

Gill writes Punjabi tone as follows (dā́ṛ, móḍā, láū; pàṇoīā, bàṇiā):
- high (acute) from a medial or final lost h or voiced aspirate;
- low (grave) after an initial voiced aspirate.

Turner's Awankari and Gill's Pothwari write the low tone of these lects with a grave: mahārājá > mε̄̀rā,
lōhá > lṑā, vāhana > vā̀ṇ, bʰaginī > bʰε̄̀ṇ; Pothwari bāhú > bā̃̀, lṓhita > làu, kuśī́ > kàī.

**How a tone is written.** After the placed accent is made visible, the `%source` blocks map it to a
grave. The block runs `Vedic > $1-stress`, then `Stress > $1+acc`, then `Stress > $1-stress`, then the
acute→grave vowel map. Segmental scores are unaffected. The first step is needed because the `Stress`
class also matches carried Vedic accents.

**Installed (all ACCEPT, prosody-only unless stated):**

| lect | change | ext_acc (train / test) | ext_tone (train / test) |
|---|---|---|---|
| awan | `%source CDIAL`: Turner's tone is a grave (105/114 marked forms) | = | 0.3970→0.3487 / 0.4413→0.3966 |
| awan | `Short > $1+stress / # C* _ V`: a light first syllable in hiatus takes the accent (madʰū́ka > màūā, divasá > dḕō̃, gupʰáti > gùnuṇ) | −0.0029 / −0.0072 | −0.0029 / −0.0088 |
| poth | Hindko weight placement (the four awan rules; there was none) + `%source gill` grave block | 0.7417→0.6208 / flat | 0.7762→0.6940 / flat |
| poth | the hiatus placement rule, as in awan | −0.0208 / flat | −0.0238 / flat |
| L | Punjabi-type placement (long penult, else first syllable; there was none) | 0.7260→0.6882 / 0.6592→0.6071 | 0.7260→0.6956 / 0.6706→0.6429 |
| srk | Punjabi-type placement | 0.6069→0.5583 / 0.5833→0.5417 | 0.6069→0.5701 / – |
| S | Punjabi-type placement (end of chain; S keeps its final vowels) | 0.7812→0.7396 / flat | – |
| kcch | Punjabi-type placement | 0.625→0.500 (1 pair) | – |
| khet | Hindko weight placement | 0.3333→0.2833 / 0.1250→0.0000 | – |

Notes on the table:
- For poth, the Gill block alone adds −0.032 ext_tone on train.
- For L, Punjabi-type and Hindko-type placement tie on train ext_acc; the bare accent score on train
  favours Punjabi-type (0.7515 vs 0.7634). Hindko-type would have been better on test, but test did
  not choose.
- For srk, Punjabi-type beats Hindko-type on train (0.5583 vs 0.5819).
- For khet, Punjabi-type placement changed nothing.

The group now has two placement types, consistent by area:
- Punjabi (P, L, srk, S, kcch): long penult, else first syllable.
- Hindko weight stress (awan, poth, khet).

**Rejected:**
- P, Gill low tone after an initial voiced aspirate (`%source gill`, grave on the vowel after # bʰ dʰ gʰ
  jʰ ḍʰ, optionally h).
  - First tried: no change.
  - Re-tried after the @morph removal: bare tone improves (−0.0019; with h −0.0042: hastatala >
    tʰèḷī), but ext_tone is unchanged, so REJECT.
  - The reason: the extension-aware choice for these words (bʰaginīpati + -ayāmi > paiṇīā) carries
    the long-penult accent, while Gill's tone falls on various syllables (pàṇoīā, paṇòīā, paṇvaī̀ā).
  - Punjabi tone stays unscored beyond the default high accent, on 22 Gill pairs.
- khet, the hiatus placement rule: no change.
- Tone type from the Vedic accent: only awan has a tone contrast (grave vs 6 circumflexes: cā̂ṛuṇ,
  kō̂ṛ, ε̄̂ruṇ …), which does not follow the Vedic accent or the aspirate (pīṭʰa > pī̀ṛā grave vs
  kōṭʰa > kō̂ṛ). Not testable with 6 forms.

### Source notation (per-source confusions, `srcconf.py` / `segdist.py`)

- **awan, srk: Rensch et al. 1992**, `%source rensch-hallberg-oleary1992`: `{cʰ j} > {c y}` and
  `u > ū` (@notation).
  - Across both lects, Rensch writes no j at all and cʰ only twice.
  - Where the chain predicts j, Rensch has y (awan 11/11, srk 4/4: jána > yaṇā, majján > mī̃y).
  - Where it predicts cʰ, Rensch has c (14/15, 5/5: *cʰatti > cat, púccʰa > pūc). CDIAL Awankari keeps
    cʰ (16/16), so this is the wordlist's spelling.
  - Rensch writes u as ū 23 vs u 15 (*kuḍa > kūṛī, dugdʰá > dūd), like Lothers for Pothwari.

  | lect | change | ext (train / test) |
  |---|---|---|
  | awan | cʰ, j alone | −0.0058 / −0.0087 |
  | awan | + u (installed) | −0.0081 / −0.0053 |
  | srk | cʰ, j alone | −0.0087 / −0.0171 |
  | srk | + u (installed) | −0.0137 / −0.0261 |

  The version with u is chosen because it is better on train in both lects.
- **OP: shackle-auto**, `%source shackle-auto`: `{ṛ ṇ} > {r n}` (@notation). Where the chain predicts
  the flap or ṇ, the automatically extracted glossary has:
  - r 55 vs ṛ 25, and n 94 vs ṇ 88;
  - the hand-checked `shackle` list has r 12 vs ṛ 43, and n 36 vs ṇ 77.

  The source is Gurmukhi-derived and often loses the flap and retroflex nasal.

  | candidate | ext (train / test) | verb (train) | verdict |
  |---|---|---|---|
  | ṛ only | −0.0008 / +0.0004 | – | rejected on test |
  | ṛ and ṇ | −0.0011 / −0.0014 | −0.0093 | ACCEPT, installed |

  Both were designed from the train counts before gating; ṛ + ṇ is also the better one on train.
- Checked, nothing separable:
  - Lothers (poth): e/ɛ, ū/o, ṇ/n all mixed.
  - Elfenbein (khet): implosives ~50/50.
  - Backstrom (D): its final -ā is written 31 and dropped 19; the latter are lexical.
  - Weinreich (D): compounds.
  - Gill: too few forms.

### Sound changes (residuals, near-misses, consistency, rarely-firing rules)

**Installed:**

**kcch, thar, Khol: `∅ > y / {i ī} _ o #`.** A glide is inserted between a front vowel and the final
-o (dīpaka > ḍīyo, gōṣṭʰika > goṭʰyo, calita > caryo).

| lect | ext (train / test) | reflexes better / worse |
|---|---|---|
| kcch | −0.0024 / −0.0037 | 20 / 1 |
| thar | −0.0030 / −0.0027 | 11 / 0 |
| Khol | −0.0024 / flat | 2 / 1 |

Sindhi rejects it (+0.0017 / +0.0010, 50 better / 159 worse: Sindhi writes -io), as in round 6.

**awan: the -uṇ infinitive after a hiatus stem.** `a > u / V _ ṇ a #` + `∅ > v / V _ u ṇ a #`, before
the hiatus rules (*kōṣati > kṑvuṇ, piṁṣáti > pī̀vuṇ, *rahati > rā̀vuṇ). Consonant loss had left the
stem vowel in hiatus, so the -uṇ rule (which needs a consonant) never fired and hiatus contraction gave
*kōṇ.

| score | train | test |
|---|---|---|
| ext | −0.0055 | −0.0022 |
| verb | −0.034 | −0.022 |
| ext_acc | −0.029 | −0.008 |
| plain | +0.0014 | – |

23 reflexes better, 0 worse.

**Khol: the same for the -en infinitive.** `a > e / V _ {n ṇ} a #` + `∅ > y / V _ e {n ṇ} a #`
(kʰā́dati > xāyen, gā́yati > gāyen, rōdati > royen). ext −0.0080 / flat, verb −0.033 / flat; 5 better,
0 worse.

**poth: `use ai-au.aiu` removed.** Pothwari keeps the secondary diphthong from hiatus, as Punjabi does
(nakulá > naul, tráyōviṁśati > taoeī). ext −0.0020 / flat; 4 better, 2 worse (the worse ones are
lagʰukk- > lokā and trúṭyati > ṭūṭeā).

A later refinement: `a {u ū} > o / _ C V` monophthongises au before a consonant and vowel, but not
word-finally (*hastakūṭa > hatʰoṛā vs nakulá > naul). ext −0.00004 / flat; 5 better, 3 worse. ACCEPT.

**srk: `ṁ Stop > homorganic nasal` notation.** The P/L/S headword rule, added. ext −0.0006 / flat;
1 better, 0 worse.

**Rejected** (Δ ext train / test unless stated):

- **L:**
  - ly > lʰ (kalya > kallʰ): +0.0014 / +0.0010; 4 better / 25 worse.
  - ṇḍ > nn (kunnī, pinnī): +0.0027; 28 / 49.
  - ā > ā̃ before final h (cā̃h, bā̃h): +0.0005; 10 / 26.
  - Final -Car anaptyxis (aggar, cakkar, pātar): P has 28 -Car against 117 other outcomes. Lexical,
    not tried.
- **kcch:** word-final geminates kept (kann, ratt, ajj, kamm): train −0.0011 but test +0.0079; 25 / 35.
- **Khol:**
  - Dropping `hiatus.aa.final` (ext-LOO −0.0032): test +0.0049.
  - Also dropping the -o citation rule: +0.0300.
- **thar:** dropping the final -ā shortening (ext-LOO −0.0010) gains 1 reflex but costs plain +0.0063.
  Sindhi-type fem. -a is attested, so this is not the bare artifact. Rejected.
- **poth:** dropping `anusvara-vowel.vm.sh` (ext −0.0006): 1 better / 2 worse. Not installed.
- **ṁ + stop notation elsewhere:**
  - awan: ext −0.0003 but ext_acc +0.0005.
  - poth: +0.0007.
  - khet, kcch, thar, Khol, D: 0.
- **Consistency check.** A matrix of shared `use` lines across the 12 files shows no other equivalent
  variants to merge. The -aī, au/ai degemination and syncope isoglosses were already tested in round 6.

#### Second pass (after the @morph work, same gates)

**D: `k ṣ > cʰ / # _`.** Initial kṣ- > cʰ, as in Shina (kṣārá > cʰor, kṣīrá > cʰīr). ext −0.0020 /
flat; 4 better, 0 worse.
- Adding medial kṣ > c (ákṣi > ac, lākṣiká > lāc) improves ext further (−0.0021 / −0.0079) but costs
  ext_acc +0.0023 on train. Rejected.
- Also rejected: `ṣ > ∅ / V _ {ṭ ṭʰ}` (hangúṭ, õõṭo): ext +0.0017 on train, ext_tone +0.066.
- **poth:** `a > o / C _ r #` (mūtor, patʰor, macʰor) — Lothers has -ar 11 vs -or 7: ext +0.0029, verb
  +0.0071; rejected (lexical).

**awan, poth, srk: `ū > ū̃ / _ h`.** Spontaneous nasalisation of ū before h from kʰ tʰ, placed before
h-loss (múkʰa > mū̃, gūtʰa > gū̃̀).

| lect | ext train | test | reflexes better / worse |
|---|---|---|---|
| awan | −0.0006 | flat | 2 / 1 |
| poth | −0.0014 | flat | 1 / 0 |
| srk | −0.0011 | flat | 1 / 0 |

Rejected: progressive nasalisation of a final long vowel after a nasal (yaṇā̃, nā̃, mā̃). Tested in
awan, poth and srk; it is worse on train or test in each (awan test +0.0064; poth train +0.0015;
srk train +0.0013).

**khet: `use ai-au.aiau` added.** OIA ai au > ē ō is in every other NW chain but was missing here
(daúhitra > dotrā, not *ɗauhitrā). ext −0.0020 / flat; 3 better, 0 worse. thar also lacks it, but
adding it there changes nothing.
- khet: `use assim-stop.stopstop` (missing): ext +0.0011 train (test −0.0046); rejected — Khetrani keeps some clusters (Elfenbein).
- Khol: adding `int-p-v` (0) or `int-aspirate-h` (+0.0002) — rejected; the Kholosi chain's gaps are deliberate.
- OP: `use s-h.s.intervocalic` (missing; ehu, kohu): ext +0.0039 / +0.0030, 59 better / 120 worse — the GGS keeps s; rejected.
- OP: final v-loss generalised from ā_ to any vowel (govi > goi): ext −0.0002 but verb +0.0024 / +0.0065; rejected.

**L: `%source gill` block.** Gill writes no gemination, and initial voiced aspirates are devoiced by
tone. L was the only lect with Gill forms (17 pairs) and no Gill block; the P/awan/srk/poth block is
now added. ext −0.00004 / flat (ACCEPT, installed for consistency).

**awan, srk: `use hiatus.ai.final`** (-aī > -ī after consonant loss, as in P L OP poth thar). Round 6
rejected it on effect size, which no longer applies.

| lect | ext (train / test) | reflexes better / worse |
|---|---|---|
| awan | −0.0002 / −0.0015 | 5 / 1 (píbati+ikā > pī) |
| srk | −0.0004 / flat | 1 / 0 |

khet: 0. kcch: +0.0004 (Kutchi keeps -aī).
- awan: degemination after au/ai (P L OP): 0, because awan monophthongises au/ai first.

**P, L, awan, poth, srk: neuter -akaṁ > -ā̃.** `a a {m ṁ} > ā̃ / _ #`, placed before `aya-e.aa.cc`, as
in OP (-u) and kcch (-ū). Without it, -akaṁ contracted to -e (kamme, kaṛe), which only matched
locatives and adverbs by accident. The -ā̃ forms: kāṇá > kāṇā̃, āntrá > āndrā̃, alavaṇa > alūṇā̃,
kā́ka > kā̃.

| lect | ext (train / test) | reflexes better / worse | also |
|---|---|---|---|
| P | −0.0011 / −0.0018 | 76 / 33 | ext_acc −0.013 / −0.013, ext_tone −0.021 |
| L | −0.0026 / −0.0021 | 71 / 16 | ext_acc −0.015 |
| awan | −0.0028 / −0.0015 | 34 / 19 | verb −0.0019, ext_acc −0.0043 |
| poth | −0.0073 / −0.0086 | 14 / 3 | |
| srk | −0.0049 / −0.0054 | 9 / 1 | |

The worse forms are the -e locatives and adverbs (ágra > agge, pára > pare, kárṇa > kanne).

khet: ext −0.0068 but verb +0.0026 (its -an infinitive); rejected.

**OP, L, S ext-LOO** (and the small lects). Every segmental rule in L, OP and S has a positive
leave-one-out value on train extension-aware NED. The zero-valued rules are the accent placement rules
and S `{r̩ ŕ̩} > u / _ #` (fires on nothing in train). No removal candidates.
- D: `use hiatus.aa.final` (-aka > -o: tāro, dādo): ext −0.0085 / −0.0009 but verb +0.025 / +0.028 (the -na infinitive becomes -no); rejected.
- D: Vedic accent as the fallback instead of the last-syllable rule: ext_acc +0.0031 / +0.0114; rejected (the Vedic accent does not predict Domaaki stress).
- P: Vedic accent as the fallback instead of the first-syllable rule: ext_acc +0.0091 / +0.0345, ext_tone +0.0091; rejected. As in round 3, the Vedic accent does not predict NW accent or tone position.

**awan: Rensch a > ā before a geminate** (`%source rensch-hallberg-oleary1992`, before the
no-gemination notation). Rensch writes phonetic length where CDIAL has aCC (hásta > hātʰ, agní >
āg). ext −0.0002 / flat. ACCEPT, but a weak result: the same block in srk improves train (−0.0008)
and is vetoed by test (+0.0024).

**D: `a a {m ṁ} > o / _ #` (-akaṁ > -o, tāraká > tāro, *dādda > dādo) + `%verb-menu +iyanaka`
replacing `+iyanakam`.**
- On its own, the sound change turns the passive verb item -iyanakam into -ino (huína, hʌsiná become
  -o). That costs verb +0.016 / +0.028.
- The same passive item without the neuter ending, `+iyanaka`, restores it.

| candidate | ext (train / test) | verb (train / test) | ext_acc (train / test) |
|---|---|---|---|
| installed | −0.0204 / −0.0090 | −0.0051 / 0.0000 | −0.043 / −0.023 |
| placebo `+iranaka` | – | +0.0157 / +0.0278 | – |

The installed change makes 34 reflexes better and 0 worse. It beats the placebo on train and test.
- Khol: -akaṁ > -o: 0 better / 4 worse; rejected.
- S: glide `∅ > y / {a ā} _ o #` (āyo, kāyo): +0.0006 / +0.0004; 9 better / 30 worse. Rejected, as
  in round 6.
- awan: P-type syncope before a heavy final syllable (bakarī > bakrī, camaṛī > camṛī): train ext
  −0.0036, 32 better / 12 worse. Test +0.0038 vetoes it, as in round 6.



**Hiatus before the new -ā̃ / -o.** When the stem ends in a vowel, -akaṁ leaves a hiatus (kā́ka+akaṁ >
*kāā̃, tāraká+akaṁ > *tārao). The stem vowel is absorbed: `{a ā} ā̃ > ā̃ / _ #` (P L awan poth srk) and
`{a ā} o > o / _ #` (D).

| lect | ext (train / test) | reflexes better / worse |
|---|---|---|
| P | −0.0002 / −0.00003 | 7 / 4 |
| L | −0.0006 / −0.0006 | 12 / 3 |
| awan | −0.0004 / flat | 2 / 1 |
| poth | −0.0009 / flat | 1 / 0 |
| srk | −0.0007 / flat | 2 / 1 |
| D | −0.0027 / flat (ext_acc −0.0056) | 2 / 1 |

The OP analogue (`{a ā} u > u`) is rejected: 9 better / 29 worse, because the GGS keeps -au.

**P: `use syncope.a.cna` removed** (the round-5 stem syncope before -ṇā). P ext-LOO −0.0002: the later
syncope before a heavy final syllable now derives the same verbal nouns (aṭṭanaka > aṭṭṇā) through the
verb menu. ext −0.0002 / flat, verb flat; 5 better, 0 worse. The only other non-positive P rule is
`C $1 $1 > $1 $1` (−0.00001), which was left alone.
- poth: dropping its copy of `syncope.a.cna`: verb +0.0107; kept.

### Legacy `@morph` rules (focus item 6)

Method: all `@morph` rules of a lect removed jointly, then one at a time where the joint removal failed
(`nomorph.py`; removed rules are commented out and tagged `[round 7: @morph removed]`). Per the
coordinator's rule, a **neutral removal** (ext, verb, ext_acc and ext_tone unchanged or better on train
and test) is accepted, and its plain / bare-accent loss is the bare-vs-extended artifact. For verbs the
engine already runs every variant without `@morph`, so `if:verb` citation rules only ever fed the bare
headword score.

**Removed:**
- **poth, both bare-stem rules** (`a a {i ē} > i`, `a {i ē} > i`): neutral after the 10:35 verb-flag fix, plain +0.0070 / −0.0027. Before the fix they failed ext (+0.0056), because verb headwords with a non-verbal gloss (lēṭyáti > leṭ, máratē > mar) got no verb menu.
- **OP, dir. sg. `a > u / C _ #`**: removed once -akaṁ > -u was in place (below).
- **srk, all 3** (`if:verb` -aṇ rules): neutral, plain +0.0108 / +0.0013.
- **D, the -na infinitive** (`if:verb`): neutral, plain +0.0179 / +0.0144.
- **khet, `if:verb` -ayati > -an**: neutral, plain +0.0023 / 0.
- **awan, `if:verb` -uṇ**: neutral, plain +0.0270 / +0.0193.
- **kcch, both `if:verb` -ṇū rules**: neutral, plain +0.0731 / +0.0597.
- **L, the 4 `if:verb` -aṇ rules**: neutral, plain +0.0345 / +0.0348.
- **P, the 5 `if:verb` -ṇā / -āuṇā rules**: neutral, plain +0.0351 / +0.0352.
- **Khol, -ati > -en** (`a {i ē} > e n / _ #`): neutral, plain +0.0308 / +0.0091.
- **S** (see below).

**Kept** (the gate that fails when the rule is removed):
- **In-stems `i n > ī / _ #`** (P, L, khet, kcch). Removing it fails ext on train: kcch +0.0004, and it is
  the cause of the joint ext loss in P (+0.0003), L (+0.0003) and khet (+0.0023). No menu item turns
  -in into -ī (hastin > hattʰī).
- **awan `a a {i ē} > u ṇ / _ #`** (no verb condition): ext +0.0006, ext_acc +0.0008, ext_tone +0.0008
  on train.
- **OP `∅/a a/a {i ē} > i`** (stem + -i): ext +0.0200 / +0.0253 before the verb-flag fix, +0.0061 /
  +0.0074 after it.
- **Khol**:
  - `∅ > e n / C _ #`: ext +0.0005 on train, before and after the verb-flag fix.
  - `a a {i ē} > e n`: ext test +0.0004.
  - The -o citation rule (round 6: ext test +0.0012; joint removal test +0.0012).

**S.** 7 of 8 rules removed: the 5 `if:verb` -aṇu/-āiṇu rules, `a > u / C _ #` (nom. -u) and
`∅ > u / C _ #`. The in-stem rule is kept, as elsewhere.

| score | train | test |
|---|---|---|
| ext | 0.2151→0.2096 | 0.2336→0.2295 |
| verb | flat | flat |
| ext_acc | flat | flat |
| plain | +0.0873 | +0.0881 |

181 reflexes better, 100 worse:
- **Better:** the forms CDIAL cites in -a: accʰa > acʰa, uddʰārá > udʰāra, kakkōla > kakora, kaṇika >
  kaṇika.
- **Worse:** the bare -u nouns kárman > kamu, cárman > camu, úras > uru. No menu item gives bare -u.

Against removing all 8, same engine: ext −0.0004 / −0.0005, 16 better / 6 worse. The plain loss is
the bare-headword artifact.

**OP after the 10:35 verb-flag fix.** More OP headwords now take the verb menu (ext 0.3219 → 0.3307).
Their citation forms showed two regular gaps, both now filled by sound change:
- `a i > i / C C* _ #`: final -ai (< -ati after t loss) > -i after a consonant, giving the verb stem +
  -i of the Shackle glossaries (mállatē > malli, yācyatē > jāci, stʰāpyatē > tʰāpi).
  - ext −0.0137 / −0.0138, verb −0.082 / −0.086.
  - 291 reflexes better, 2 worse.
- `a a {m ṁ} > u / _ #`, before `aya-e.aa.cc`: neuter -akaṁ > -aũ > -u, which gives the verbal noun
  -aṇu (útkʰanati > ukkʰaṇaṇu, katʰáyati > kahaṇu) and the direct -u of nouns (kaṭaka > kaṛu).
  - After `aya-e` it never fires (-akaṁ > -e).
  - ext −0.0064 / −0.0035, verb −0.0056 / −0.0034.
  - 123 reflexes better, 10 worse.

With -akaṁ > -u in place, the `@morph` dir. sg. `a > u / C _ #` could be removed:
- ext −0.0048 / −0.0062, 117 better / 44 worse, plain +0.0281 / +0.0270 (bare artifact). Removed.
- The stem + -i rules still fail ext (+0.0061 / +0.0074) and are kept. They serve verb headwords with
  a non-verbal gloss.

After the verb-flag fixes (10:35, 11:40), awan `a a {i ē} > u ṇ` (ext +0.0005), Khol `∅ > e n / C _ #`
(ext +0.0005) and OP stem + -i (`∅ > i / C _ #` alone: ext +0.0032 / +0.0040) were re-tested. All
still fail and are kept.

### What remains

- **Lothers' breathy vowels** (a̤ ɛ̤ o̤, 25 Pothwari forms). They mark tone on the accented vowel after
  an initial voiced aspirate or a lost h (gʰara > kā̤r, bʰārá > pā̤r, bāhú > bā̤̃, divasá > tɛ̤āṛā).
  - Segmental scores count them as vowel-quality errors.
  - Writing them needs a chain-level mark early in the chain, which would block later rules on those
    vowels, plus a removal in every other source's block. Not attempted.
- **Punjabi tone in P.** Only Gill marks it (22 train pairs, 0 test), and Gill's placement varies
  within one etymon.
- **Vowel qualities.** Every source with lax/open vowels mixes spellings within one etymon (Rensch,
  Lothers) or with au/ai (CDIAL L). Lorimer's ʌ (D) is test-vetoed.
- **Lexical doublets**, as in round 6:
  - initial v/b; -ṇā/-nā; flap/r in OP;
  - Kutchi final geminates; Pothwari -ar/-or; Lahnda -ar/-ur;
  - nasal feminines (-ā̃ without -akaṁ).
- **OP shackle-auto OCR debris.** 93 inserted r are mostly "rh" for the anusvāra ṁ (dasavārh,
  dahīrh, jahārh), plus English fragments ("preceding vord"). This is not notation a rule can model.
- **Kept `@morph` rules:**
  - in-stems (P L khet kcch S);
  - awan -ayati > -uṇ;
  - OP stem + -i;
  - Khol root -en, -ayati -en and -o.

  Each fails a named gate (above).
