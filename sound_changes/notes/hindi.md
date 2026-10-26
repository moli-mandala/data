# Hindi group: OIA → Hindi-Urdu, Old Awadhi, Awadhi, Braj, Kannauji, Bundeli, Haryanvi, Bagheli, Chhattisgarhi, Surgujia

Rule files: `sound_changes/rules/OIA-{H,OAw,Aw,Brj,Kannauji-Central,bundeli_atarra,kaithal,bagheli_lakshman,NDu,SSr}.sc`.
Frontiers: `sound_changes/results/OIA-<lect>.hand.frontier.csv`. Scratch tools (candidate tester, adjacent-swap /
move chronology test, diagnostic toggle battery, pruning, strata counts): `tmp/sound_changes/hindi/`.

Lect abbreviations used below: H = Hindi-Urdu, Kn = Kannauji, Brj = Braj, Bu = Bundeli, Hr = Haryanvi (kaithal),
OAw = Old Awadhi, Aw = Awadhi, Bg = Bagheli, Ch = Chhattisgarhi (NDu), Sg = Surgujia (SSr).

## 1. Results

| lect | name | #etyma | #rules | train NED null → final | test NED null → final | test exact % | auto test NED @n (hand @n) | excluded / per-source blocks |
|---|---|---|---|---|---|---|---|---|
| H | Hindi-Urdu | 5801 | 49 | 0.627 → 0.360 | 0.634 → 0.362 | 18.3 | 0.461 @16 (hand 0.433) | excl: kannauji; blocks: — |
| OAw | Old Awadhi | 880 | 46 | 0.539 → 0.245 | 0.576 → 0.260 | 33.9 | 0.337 @24 (hand 0.300) | excl: —; blocks: — |
| Aw | Awadhi | 751 | 49 | 0.676 → 0.361 | 0.674 → 0.309 | 29.3 | 0.480 @27 (hand 0.381) | excl: —; blocks: — |
| bundeli_atarra | Bundeli | 322 | 39 | 0.646 → 0.513 | 0.646 → 0.524 | 4.3 | 0.562 @18 (hand 0.553) | excl: —; blocks: bundeli, kannauji |
| Brj | Braj | 313 | 47 | 0.640 → 0.401 | 0.668 → 0.436 | 8.3 | 0.557 @21 (hand 0.479) | excl: —; blocks: bundeli |
| bagheli_lakshman | Bagheli | 276 | 33 | 0.626 → 0.471 | 0.636 → 0.511 | 6.0 | 0.573 @22 (hand 0.515) | excl: —; blocks: bundeli, koshy2022bagheli |
| kaithal | Haryanvi | 251 | 39 | 0.655 → 0.473 | 0.641 → 0.489 | 12.8 | 0.588 @22 (hand 0.503) | excl: —; blocks: webster2024haryanvi |
| NDu | Chhattisgarhi | 248 | 40 | 0.682 → 0.492 | 0.676 → 0.503 | 7.7 | 0.611 @17 (hand 0.546) | excl: —; blocks: chattisgarhi |
| SSr | Surgujia | 164 | 37 | 0.681 → 0.467 | 0.674 → 0.484 | 16.2 | 0.553 @13 (hand 0.554) | excl: —; blocks: chattisgarhi |
| Kannauji-Central | Kannauji | 151 | 38 | 0.629 → 0.404 | 0.633 → 0.426 | 13.5 | 0.627 @14 (hand 0.476) | excl: —; blocks: kannauji |

"auto test NED @n" is the greedy baseline at its largest rule count n; "hand @n" is my chain's test NED at the same
n on the hand frontier. The hand chain wins at equal size everywhere except Sg, where they tie (0.554 vs 0.553 at n=13;
Sg has 138 train pairs).

Notes on the table:
- Rule counts include `@acc-loss`, `@morph` and chain-level `@notation` lines, but not the per-source `%source` blocks.
- The auto baseline scored *all* sources with no notation handling. Where I excluded a source (H) or
  added per-source notation blocks (the survey lects), hand and auto numbers are not on exactly the same pairs.
- H excludes the `kannauji` source (404 H pairs): those are Kannauji dialect forms (-o adjectives, `kh`
  digraphs, imperatives) filed under Hindi, i.e. dialect mixture, not a notation difference.
- Only CDIAL-based lects (H, OAw, Aw, most of Brj) reach 0.25–0.37; the survey-list lects stay near 0.4–0.52.
  Their word lists cite inflected/derived forms (`bʰaĩsiyā`, `kiravā`, imperatives, past tenses), so a large share
  of the residual is morphology, not sound change.

## 2. Base chain and deviations

The group template is the H chain plus the central/eastern `l > r` rule (row 28a) and, for Aw/Brj, a post-nasal
flap (39a) and final -v > -u (48a, also Kn, Bg). Shared changes are in the same
relative order in every file. ✓ = present, · = absent or pruned (LOO ≤ 0 or never fires in that lect), ≈ = a
lect-specific variant of the same change (given in the list below the table).

| # | rule (H text) | tags | H | Kn | Brj | Bu | Hr | OAw | Aw | Bg | Ch | Sg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `V > $1-acc` | @acc-loss | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 2 | `∅ > Ø / {k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ tʰ d dʰ p pʰ b bʰ l ḷ r} _ #` | @morph | ✓ | ✓ | · | · | ✓ | ✓ | ✓ | · | · | · |
| 3 | `{ai au} > {ē ō}` | @ai-au | ✓ | · | ✓ | ✓ | · | ✓ | ✓ | · | · | · |
| 4 | `r̩ > r i / # _` | @r-vocalic-ri | ✓ | · | · | · | · | ✓ | ✓ | · | · | · |
| 5 | `r̩ > i` | @r-vocalic | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 6 | `Sib > s` | @sibilant-s | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 7 | `V ṁ > $1+long / _ s` | @anusvara-vowel | ✓ | ✓ | ✓ | ✓ | · | · | ✓ | ✓ | ✓ | · |
| 8 | `{t tʰ d dʰ} y > {c c j j} {c cʰ j jʰ}@1` | @palatal-Cy | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 9 | `{k t p} s > {k c c} {kʰ cʰ cʰ}@1` | @ks | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 10 | `s Stop > $2-asp $2+asp` | @s-stop-aspirate | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 11 | `h {y v} > {j b}@2 {jʰ bʰ}@2` | @x-hindi-h-cluster | ✓ | ✓ | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| 12 | `m {r l} > m b` | @x-hindi-mr-mb | ✓ | ✓ | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | · |
| 13 | `s Son > s s` | @assim-sonorant | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 14 | `Nasal Stop > ṁ $2` | @nasal-cluster | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | ✓ | ✓ |
| 15 | `r t > ṭ ṭ` | @assim-r @x-hindi-rt-retroflex | ✓ | · | ✓ | ✓ | ✓ | · | · | ✓ | · | · |
| 16 | `{r ḣ} C > $2-asp $2` | @assim-r | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 17 | `Stop {r y v l Nasal} > $1-asp $1` | @assim-sonorant | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 18 | `Nasal Nasal > $2 $2` | @assim-sonorant | ✓ | · | · | · | · | ✓ | · | · | · | · |
| 19 | `Stop Stop > $2-asp $2` | @assim-stop | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | ✓ | ✓ |
| 20 | `{n ṇ l v} {y v} > $1 $1` | @assim-sonorant | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 21 | `{v y} {v y} > {b j} {b j}@1` | @v-b @y-j | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 22 | `G $1 $1 > $1-asp $2` | @cluster-simplify | ✓ | ✓ | ✓ | · | · | ✓ | ✓ | · | ✓ | ✓ |
| 23 | `C C > $2 / # _` | @initial-degem | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 24 | `{kʰ gʰ tʰ dʰ pʰ bʰ} > h / V _ V` | @int-aspirate-h | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 25 | `{ṭ ṭʰ} > {ḍ ḍʰ} / V _ V` | @int-retroflex | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 26 | `p > v / V _ V` | @int-p-v | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 27 | `V m > $1+nas v / _ V` | @m-v | ✓ | · | ✓ | · | · | ✓ | ✓ | · | ✓ | ✓ |
| 28 | `{k g c j t d y} > ∅ / V _ V` | @int-stop-loss | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 28a | `l > r / V _ V` | @x-hindi-l-r | · | · | ✓ | ✓ | · | ✓ | ✓ | · | ✓ | ✓ |
| 29 | `i n > ī / _ #` | @morph | ✓ | · | · | · | · | ✓ | ✓ | · | ✓ | · |
| 30 | `C > ∅ / _ #` | @final-cons-loss | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 31 | `a v a > o // _ #` | @aya-e | ✓ | ✓ | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| 32 | `ā > a / C _ #` | @final-vowel-loss | ✓ | · | · | · | ✓ | ✓ | ✓ | · | ✓ | ✓ |
| 33 | `a a {i ē} > n ā / _ #` | @morph | ✓ | · | ≈ | ≈ | ≈ | ≈ | ≈ | · | · | · |
| 34 | `a {i ē} > n ā / _ #` | @morph | ✓ | · | ≈ | ≈ | ≈ | · | ≈ | · | · | · |
| 35 | `ā {i ē} > ā n ā / _ #` | @morph | ✓ | · | ≈ | · | · | · | ≈ | · | · | · |
| 36 | `V G $2 > $1+long $3 / _ V #` | @degem-lengthen | ✓ | ✓ | ✓ | ✓ | ✓ | ≈ | ≈ | ✓ | ✓ | ✓ |
| 37 | `G $1 > $2` | @degem-plain | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 38 | `{ḍ ḍʰ} > {ṛ ṛʰ} / V _` | @flap | ✓ | ✓ | ≈ r, pre-degem | ✓ | ✓ | ≈ r (ḍ only), pre-degem | ≈ r | ✓ | ✓ | ✓ |
| 39 | `V ṁ > $1+long+nas / _ C` | @anusvara-vowel | ✓ | ✓ | ✓ | ✓ | ✓ | ≈ penult only | ✓ | ✓ | ✓ | ✓ |
| 39a | `{ḍ ḍʰ} > {ṛ ṛʰ} / V _` (post-nasal, after 39) | @flap | · | · | ✓ | · | · | · | ✓ | · | · | · |
| 40 | `{i ī} {a ā} > ī / _ #` | @hiatus @morph | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | ✓ | ✓ |
| 41 | `a a > e / C _ C` | @aya-e | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 42 | `{a ā} {a ā} > ā` | @hiatus | ✓ | ✓ | ✓ | · | ✓ | · | ✓ | · | ✓ | ✓ |
| 43 | `{a i u} > ∅ / _ #` | @final-vowel-loss | ✓ | ✓ | ✓ | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ |
| 44 | `a > ∅ / V C _ C V` | @syncope | ✓ | · | ✓ | · | · | · | · | · | · | · |
| 45 | `ṇ > n` | @n-dental | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 46 | `v > b / # _` | @v-b | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 47 | `y > j / # _` | @y-j | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 48 | `{ē ō ē̃ ō̃} > {e o ẽ õ}` | @notation | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 48a | `v > u / V _ #` | @x-hindi-v-vocalise | · | ✓ | ✓ | · | · | · | ✓ | ✓ | · | · |
| 49 | `Ø > n ā` | @morph | ✓ | ≈ | · | · | ≈ | ≈ | ≈ | · | · | · |

Lect-specific variants and additions (all in the same slot as the H rule they replace):
- **Citation morphology (@morph).** Verbs: H `-nā`, Brj `-no`, Aw `-ab` (`piṁṣati > pīsab`), OAw 3sg `-ai`
  (`-ati > -ai`, `-ayati > -ai`, `-atē > -ai` via `ē > i / a _ #`), survey lists (Bu, Hr) bare stems/imperatives,
  Kn imperative `-o` for roots. Starred verbal roots (`*aṭṭ`, `*ḍag`) get a protected stem-final consonant (`Ø`)
  that is spelled out as the lect's citation ending at the very end (H `nā`, Aw `ab`, OAw `ai`, Kn `o`).
- **Brj:** `{a ā} a > o / _ #` — the Western Hindi masc. `-o` for `-aka` (`gʰōṭaka > gʰoro`); flap `ḍ > r`, ordered
  before degemination; post-nasal flap to ṛ (`kuṇḍá > kū̃ṛī`) and final `-v > -u` as in Aw; final -ā kept (the
  `ā > a / C _ #` shortening prunes).
- **OAw:** keeps final short vowels (Tulsīdās/Jāyasī spelling), keeps vowels in hiatus (`kāyara`), no syncope; CL and
  anusvāra-lengthening only in the stressed penult, counting `-ai` as one syllable (`V G $2 > $1+long $3 / _ V (V) #`:
  `gárjati > gājai`), elsewhere short V + ṁ is written (`maṇḍapa > māṁḍava` but `kuṁḍara`); nasal vowels written V+ṁ
  (`@notation`); flap `ḍ > r`.
- **Aw:** flap `ḍ > r`; a second, post-nasal flap after nasal-vowel formation (`daṇḍá > ḍā̃ṛā`, `bʰāṇḍa > bʰā̃ṛā`,
  −0.004), which OAw does not have (OAw keeps `ṁḍ`, +0.003 if added) — a post-OAw change; final `-v > -u` (`nāũ`); CL also before the `-ab` infinitive (`várdʰatē > bāṛʰab`); no syncope; the `{ē ō} > {e o}`
  notation rule sits before CL, because Saksena marks ē ō exactly where CL lengthened the vowel.
- **Hr:** `n > ṇ / V _ V` (`bʰaginī > bahaṇ`), `h > ∅ / V _` (tone in the source: `bāhú > bā́`); no `m > ṽ`, no
  `ava > o`, no `m {r l} > m b`.
- **Bg:** the Koshy list writes N+C clusters and rarely marks nasality, so `Nasal Stop > ṁ` and the anusvāra rule prune
  away; CL has LOO slightly < 0 but is kept (see §5.13).

Per-source `@notation` blocks (not chain rules):
- `bundeli` (in Bu, Brj, Bg): `ṛ` written `ḍ`; `u` written long by default (ū 372 : u 75); nasal vowel + stop written
  V + homorganic nasal (`aŋgūṭī`). Its dental `t` appears as `t ` (a stripped IPA diacritic leaves a space, harmless
  after tokenising) and dental `tʰ` as `t ʰ` (30 of 34; a block rule `tʰ > t ʰ` reproduces the two segments).
- `kannauji` (in Kn, Bu): voiceless aspirates as digraphs `kh ch ṭh th ph` (voiced ones use ʰ), `ṛ` as `ḍ`, long `u`,
  N+C clusters.
- `koshy2022bagheli`: dentals written with the retroflex dot (`saptá > saṭey`), schwa written `e` (`samāná > seman`),
  a/u length unmarked, `i` written `ī`, `ṛ` as `ḍ`, nasality rarely marked.
- `chattisgarhi` (Ch, Sg): high vowels always long (ī 256 : i 12, ū 157 : u 1); N+C clusters.
- `webster2024haryanvi` (Hr): high vowels short (u 86 : ū 10), `ṛ` as `ḍ`, tone accents (stripped by the scorer).

## 3. Relative chronology

Δ = train-NED increase when the order is reversed (swap of adjacent rules, or moving the first rule after the second).

| earlier change | later change | lect(s) | Δ if reversed | reading |
|---|---|---|---|---|
| `{k g c j t d y} > ∅ / V _ V` (int-stop-loss) | `G $1 > $2` (degemination) | H | +0.062 | stop loss is MIA; NIA degeminated stops survive (`bʰakta > bʰāt`) |
| `{kʰ …} > h / V _ V` | `G $1 > $2` | H | +0.010 | same for aspirates (`hásta > hātʰ`) |
| `l > r / V _ V` | `G $1 > $2` | OAw, Aw, Ch, Brj, Bu, Sg | +0.006, +0.007, +0.006, +0.009, +0.004, +0.008 | r < l predates degemination: -ll- stays l |
| `V G $2 > $1+long $3` (CL) | `G $1 > $2` | H / OAw / Aw | +0.004 / +0.009 / +0.013 | lengthening feeds on the geminate |
| CL | `{a i u} > ∅ / _ #` | H | +0.005 | CL needs the final vowel (penult stress) |
| `ā > a / C _ #` | `{a i u} > ∅ / _ #` | H | +0.032 | feminine -ā is shortened, then lost like -a (`jihvā > jībʰ`) |
| `ā > a / C _ #` | `-ati > -nā` (@morph) | H | +0.008 | bookkeeping: citation morphology is added to the NIA stem |
| `a a > e / C _ C` | `{a ā} {a ā} > ā` | H | +0.0009 | medial a(y)a > e must precede general hiatus contraction |
| `{a ā} {a ā} > ā` (hiatus) | `{a i u} > ∅ / _ #` | H | +0.003 | -aka > -aa > -ā contracts before final -a is lost |
| `{a i u} > ∅ / _ #` | `a > ∅ / V C _ C V` (syncope) | H | +0.003 | schwa syncope is later than apocope |
| `G $1 > $2` | syncope | H | +0.001 | syncope is NIA |
| `{ḍ ḍʰ} > {ṛ ṛʰ}` (flap) | `V ṁ > $1+long+nas` | H | +0.0015 | ṁḍ (> ṇḍ) is not flapped in H: flapping precedes nasal-vowel formation |
| `V ṁ > $1+long+nas` | second flap `{ḍ ḍʰ} > {ṛ ṛʰ} / V _` | Aw, Brj | +0.004, +0.001 (LOO of the second flap) | Aw/Brj flap ḍ again after nasal vowels (`daṇḍa > ḍā̃ṛā`); OAw does not (+0.003 if added): post-OAw |
| flap | `G $1 > $2` | Brj (OAw ≈) | +0.0022 | ḍḍ > ḍ stays unflapped in Braj; H, Aw prefer the reverse (+0.0004/+0.0005), Ch clearly (−0.006) |
| `i n > ī / _ #` | `C > ∅ / _ #` | H, Aw | +0.0013, +0.0020 | -in stems take -ī before final-consonant loss (bleeding) |
| `Sib > s` | `V ṁ > $1+long / _ s` | Aw, H | +0.0025, +0.0005 | ṁś and ṁṣ lengthen too, so the sibilant merger is earlier |
| `r t > ṭ ṭ` | `{r ḣ} C > $2-asp $2` | H | +0.0005 | retroflexion must bleed the general r-assimilation |

A full reorder search on H (every rule tried at every position, 2 256 variants) finds no move that improves train
NED by more than 0.0004, so the H order is a local optimum; on OAw and Aw the best single moves gain ≤ 0.0015 and are
not linguistically meaningful (I adopted one: Aw's `ē ō > e o` notation placed before CL, see §2).

The MIA block (assimilations → initial degemination → intervocalic lenition/loss → final-consonant loss) and the NIA
block (CL → degemination → flap → anusvāra vowel → hiatus → apocope → syncope) are the same in all ten files; the
only order that differs by lect is flap vs degemination (Brj/OAw: before; the rest: after). Aw and Brj add a second
flap after nasal-vowel formation.

## 4. Shared vs lect-specific changes; sub-grouping

**Shared by all ten lects** (all carry positive LOO wherever the data can test them): accent loss; ś ṣ > s; kṣ > kkʰ
(never ccʰ: +0.2 to +0.6 points of NED everywhere when ccʰ is forced); Cy palatalisation; s+stop > aspirated geminate;
all cluster assimilations; vv > bb, yy > jj; initial degemination; intervocalic aspirate > h; ṭ > ḍ; p > v; loss of
intervocalic k g c j t d y; final consonant loss; penultimate compensatory lengthening + plain degemination; flapping
of single ḍ; anusvāra > long nasal vowel; medial a(y)a > e (`kadalī > kelā`, helps all ten); ṇ > n; initial v > b and y > j (CDIAL writes *no* initial v- or y- for any lect:
0/453 H, 0/83 OAw, 1/25 Hr); loss of final short vowels (except OAw spelling).

Diagnostic toggles (battery; Δ train NED × 100 for *adding* the change to the lect's chain — negative = the lect's data
supports the change; each column uses the same insertion anchor):

| change added | H | Kn | Brj | Bu | Hr | OAw | Aw | Bg | Ch | Sg |
|---|---|---|---|---|---|---|---|---|---|---|
| main flap ḍ>ṛ | -1.09 | +0.00 | +0.00 | +0.00 | +0.00 | +0.24 | +0.00 | +0.00 | -0.87 | -0.57 |
| main flap ḍ>r | -0.22 | +0.46 | -0.05 | +0.39 | +1.03 | -0.61 | -0.39 | +0.33 | -0.54 | -0.66 |
| post-nasal flap ṇḍ>ṛ | +0.15 | +0.00 | -0.14 | +0.00 | +0.00 | +0.26 | -0.38 | +0.00 | +0.15 | +0.00 |
| final -v > -u | +0.06 | -0.07 | -0.13 | +0.12 | +0.12 | +0.00 | -0.35 | -0.21 | +0.63 | +0.60 |
| l>r /V_V (pre-degem) | +1.18 | +0.13 | -0.66 | -0.55 | +1.31 | -0.61 | -0.89 | +0.20 | -0.80 | -0.83 |
| ava>o (not word-final) | -0.19 | -0.29 | -0.79 | -0.61 | +0.11 | -0.10 | -0.43 | -0.35 | -0.48 | -0.88 |
| m>ṽ | -0.16 | +0.64 | -0.35 | +0.08 | +0.62 | -0.34 | -0.37 | +0.16 | -0.13 | +0.00 |
| CL (penult) | -0.54 | -0.88 | -1.33 | +0.20 | -0.01 | -0.93 | -1.09 | +0.24 | -1.23 | -2.35 |
| anusvāra > V̄̃ | -2.00 | -0.94 | -1.47 | -1.04 | -0.78 | +0.20 | -2.63 | +0.00 | -1.18 | -0.91 |
| final a i u loss | -4.03 | -6.02 | -4.91 | -0.96 | -3.42 | +11.28 | -5.89 | -2.31 | -3.06 | -2.67 |
| final -ā > -a | -0.24 | +0.10 | +0.18 | +0.40 | -0.13 | -0.46 | -0.73 | +0.10 | -0.25 | -0.45 |
| -aka > -o | +0.58 | +0.00 | -0.29 | +0.04 | +0.59 | +0.57 | +0.29 | +0.10 | +0.01 | +0.00 |
| syncope | -0.18 | +0.13 | -0.13 | +0.05 | +0.05 | +1.30 | -0.05 | +0.05 | +0.02 | +0.10 |
| hiatus aa>ā | -0.46 | -0.25 | -0.11 | -0.17 | -0.51 | +0.12 | -0.14 | -0.00 | -0.76 | -0.88 |
| ṇ>n | -0.67 | -0.79 | -0.66 | -0.51 | -0.22 | -0.72 | -0.73 | -0.50 | -0.67 | -1.10 |
| n>ṇ /V_V | +0.79 | +0.70 | +0.51 | +0.18 | -0.38 | +2.06 | +0.79 | +0.26 | +0.56 | +0.48 |
| v>b initial | -1.08 | -1.36 | -0.51 | -0.99 | -1.19 | -1.83 | -1.27 | -1.08 | -1.28 | -0.70 |
| y>j initial | -0.27 | -0.35 | -0.11 | -0.37 | -0.12 | -0.38 | -0.43 | -0.37 | -0.43 | -0.36 |
| rt>ṭṭ | -0.05 | +0.00 | -0.11 | +0.00 | -0.05 | +0.06 | +0.03 | -0.03 | +0.00 | +0.00 |
| ṁs>V̄s | -0.04 | -0.20 | -0.15 | -0.06 | -0.06 | +0.01 | -0.25 | -0.11 | -0.11 | +0.00 |
| r̩>a (instead of i) | +0.11 | +0.00 | +0.20 | +0.05 | -0.05 | +0.14 | -0.06 | +0.02 | +0.00 | +0.14 |
| kṣ>ccʰ (instead of kkʰ) | +0.35 | +0.55 | +0.16 | +0.16 | +0.18 | +0.41 | +0.40 | +0.33 | +0.35 | +0.42 |

(Kn, Bu, Hr, Bg write ṛ as ḍ, so both flap rows are ~0 or positive there by notation, not phonology. In Aw and Brj
the "main flap" rows are measured with their post-nasal ṛ flap still in place, which then flaps every ḍ, so only
the r row is informative there. OAw's +11.3 for
apocope and +1.3 for syncope are orthographic: its sources write final and medial a.)

Readings:
1. **`l > r` is the sharpest isogloss, and it is not West vs East.** Adding intervocalic `l > r` (before degemination)
   helps Brj, Bu, OAw, Aw, Ch, Sg by 0.6–0.9 points and hurts H (+1.2) and Hr (+1.3). Raw counts agree: an etymon's
   intervocalic l survives as l in 77% of H reflexes and 98% of Haryanvi ones, but in 22–35% of Aw, OAw, Brj, Bu, Ch, Sg.
   Braj `kāro`, `pīro` belong with Awadhi here; Khari Boli (standard H) and Haryanvi form the l-keeping north-western
   corner. (Kn is neutral on 16 items; Bg-Koshy rejects it, but Koshy's list also writes `ḍ` for many r.)
2. **Flap outcome:** `ṛ` in H and Ch, `r` in OAw, Aw (and Sg, Brj about equally good). The survey lists (Kn, Bu, Hr,
   Bg) do not distinguish ṛ from ḍ at all, so they cannot vote.
3. **Western -o:** `-aka > -o` is supported only by Brj (−0.3); it hurts H, Hr, OAw, Aw (+0.3 to +0.6). Kn and Bu
   lists do end many adjectives in -o (`kā̃ṭo`, `lambo`, `sūkho`), but on etyma without -ka, so no rule can fire.
4. **Haryanvi** alone takes intervocalic `n > ṇ` (−0.4; every other lect +0.2 to +2.1) and postvocalic h loss with
   tone (`bāhú > bā́`; every other lect rejects medial h loss), keeps ava (ava > o hurts),
   keeps -m- (m > ṽ hurts +0.6) and keeps some geminates (46 Webster forms, e.g. `tāttā`, `baḍḍā`). These are
   Panjabi-side features; Hr is the most divergent Western lect.
5. **Syncope** (medial schwa deletion) is supported in H, Brj (weakly), and nowhere else; OAw rejects it strongly
   (+1.3) because OAw spelling writes every medial a.
6. **Post-nasal flap and final -v > -u group Braj with Awadhi.** Aw and Brj flap ḍ a second time after nasal-vowel
   formation (`daṇḍá > ḍā̃ṛā`, `kuṇḍá > kū̃ṛī`; −0.38 / −0.14) and vocalise final -v (`nāũ`, `dā̃u`; −0.35 / −0.13);
   H rejects both (+0.15, +0.06), Ch/Sg reject -v > -u strongly (+0.6).
7. **OAw sits with Aw, as its older stage.** It shares every central/eastern choice (l > r before degemination, flap
   r, ava > o except finally, final -ā > -a, m > ṽ, ṇ > n) in the same relative order. It lacks exactly the change that
   the chronology puts last in Aw: the post-nasal flap (OAw writes `kuṁḍara`, `māṁḍava`; adding the flap costs
   +0.003). The rest of what separates OAw from Aw is orthography (final -a written, V+ṁ, hiatus kept, medial a kept)
   and verb morphology (3sg -ai vs infinitive -ab). No change was found that OAw shares with the Western lects
   against Aw.
8. **E. Hindi survey lists (Bg, Ch, Sg)**: Ch and Sg (same `chattisgarhi` source) share l > r, flap ṛ, hiatus
   contraction, CL, and both reject -v > -u. Bg (Koshy) is too noisy to place (its notation block has five rules).

Suggested sub-grouping from these files: {H, Hr} (l kept; Hr further n > ṇ, h-loss with tone, geminates) vs a
central/eastern block {Brj, Bu, Kn?, OAw, Aw, Ch, Sg} with l > r. Inside that block Brj and Aw share the post-nasal
flap and -v > -u (so Braj is closer to Awadhi than to Khari Boli in its sound changes), and OAw is Aw minus the
latest change. The handbook Western vs Eastern Hindi split shows up mainly in morphology (-o vs -ā; verb endings),
not in the sound changes the data can test.

## 5. Novel / surprising / data issues

1. **`l > r` precedes degemination.** In every lect that has it (Brj, Bu, OAw, Aw, Ch, Sg), moving `l > r / V _ V`
   after `G $1 > $2` costs +0.004 to +0.009 train NED: single -l- becomes r (`mū́la > mūra`, `tala > tara`,
   `kāla > kariā`, Brj `kāro`), while -ll- survives as l (`*navalla > navala`, `*pilla > pilavā`). Placed late, the rule
   is worthless (OAw −0.0001 vs −0.006 early). So r < l is older than the NIA degemination.
2. **Compensatory lengthening is penult-only.** Extending CL to any initial syllable costs +0.013 in H. Aw and OAw
   extend the penult to verb stems before the citation ending (`V G $2 > $1+long $3 / _ V (V) #` in OAw, `/ _ V (b) #`
   in Aw: `gárjati > gājai`, `várdʰatē > bāṛʰab`), i.e. stress stays on the stem. CL still hurts 228 H words
   (helps 297): most are H forms with a geminate kept before an -ā/-ī suffix (`pakkā`, `billā`, `laggā`, `ḍabbā`) —
   the expressive/suffixed stratum where degemination did not happen. No phonological conditioning (vowel quality,
   geminate type) separates them.
3. **Anusvāra vs written nasal consonant (H).** The anusvāra rule helps 487 H words and hurts 152. The hurt ones keep a
   written nasal consonant and a short vowel (`añjal`, `kusumbʰ`, `gandʰ`, `dambʰ`, `hindol`) — a learned stratum.
   Lengthening only in the penult with a homorganic nasal elsewhere is worse (+0.005). OAw spelling does the
   opposite of H: anusvāra lengthens only in the stressed penult (`maṇḍa > māṁḍa`), elsewhere short V + ṁ.
4. **ava > o except word-finally** (`a v a > o // _ #`) beats plain `ava > o` in 9 of 10 lects (`náva > nau`, not
   `no`); Hr keeps ava.
5. **MIA geminate vv > bb, yy > jj** (`sarva > sabba > sab`, `kārya > kajja > kāj`) helps all ten lects; so do the
   triple-geminate reduction (`uṣṭra > uṭṭʰṭa > ūṭ`) and initial r̩ > ri (`r̩kṣa > rīcʰ`, `r̩ddʰa > rījʰnā`).
6. **kṣ > kkʰ, never ccʰ**, in all ten lects, although H has cʰ in 33 of ~250 kṣ words (`r̩kṣa > rīcʰ`,
   `kakṣyā > kācʰ`, `*kṣaṇati > cʰannā`): the cʰ reflexes are a lexical minority everywhere, not a regional isogloss here.
7. **Tatsama stratum.** 11% of H reflexes keep an OIA single intervocalic k g c j t d (`agaru > agar`); 6–19% in the
   other lects. Those H reflexes score NED 0.42 against 0.32 for the ones that lost the stop (274 vs 2 200 train
   pairs), i.e. the learned stratum is a visible share of the residual. No lect ever keeps initial v- or y- (0/453 v- in H).
8. **-ā / -ī / -o extensions are unpredictable from the etymon.** For H nouns whose predicted form ends in a
   consonant, the attested form ends in C 43%, -ā 38%, -ī 14%. Accent, syllable count, etymon cluster and final
   consonant barely move this (52–59% vowel-final). A blanket `@morph ∅ > ā / C _ #` would gain only 0.003 (H), 0.005
   (Bu), 0.004 (Ch) and was not adopted. Braj/Kannauji/Bundeli `-o` (`kā̃ṭo`, `lambo`, `sūkho`) is the same
   phenomenon; only CDIAL etyma with -aka (`gʰōṭaka > gʰoro`) can be predicted.
9. **Flap vs geminate.** Brj prefers flapping before degemination (ḍḍ stays ḍ: +0.0022 if reversed); H and Aw prefer
   after (+0.0004/+0.0005); NDu after (−0.006 for "after"). In the survey lists with ṛ written ḍ the order is moot.
10. **Haryanvi** keeps some geminates (46 Webster forms: `tāttā`, `baḍḍā`, `haḍḍi`), but removing degemination still
    costs +0.011, so degemination stays; `n > ṇ` and h > ∅ with tone (`bā́`) are its own.
11. **Sporadic, not modelled:** `ṣ- > cʰ-` in the 'six' family (H `cʰah`, Aw `cʰabbīs`; −0.0013 in Aw, a handful of
    words); OAw masc. nom. -u (`ūtaru`, `gālu`, `nāhu`, 29 items vs 236 with -a); Apabhraṃśa `puhupa`, `puhumi`.
12. **Data problems.** (a) 404 `kannauji` pairs filed under H are Kannauji dialect forms (excluded). (b) Survey-list
    notation (see §2): Koshy's dentals-as-retroflex and schwa-as-e; Chhattisgarhi long high vowels; Webster short
    high vowels; bundeli's detached aspiration `t ʰ`. (c) Numerals in -ati (`viṁśati`, `saptati`) get verb endings
    because the engine cannot tell nouns from verbs. (d) Etyma with a trailing hyphen (`*vaḍrakk-`, 309 in H) are
    nominal stems but look like consonant-final roots to the rules, so they get the verb ending. (e) Aw writes
    whispered final vowels as superscripts; `ᵘ ⁱ` are dropped by the scorer but `ᵃ` (`upᵃjab`, 5 forms) is not.
13. **Survey lists vs CDIAL.** Several changes that CDIAL clearly supports prune away on survey data: Bu and Bg lose CL
    by NED (their suffixed citation forms `kappaṛā`, `hatarīyā` have no CL), although exact-match rises when CL is
    kept (Bu 5.1 → 5.5%, Bg 6.7 → 7.5%); I kept CL there. Bg (Koshy) also loses the anusvāra rules because the list
    writes N+C clusters or no nasality at all (`páñca > pac`).

## 6. New tags, engine wishes

New tags:
- `@x-hindi-l-r` — intervocalic l > r (single l only; ordered before degemination so -ll- stays l).
- `@x-hindi-rt-retroflex` — r + t > ṭṭ (dental retroflexed by a preceding r: `vartate > vaṭṭai`, `āvartana > auṭan`);
  co-tagged `@assim-r`.
- `@x-hindi-mr-mb` — m + liquid > mb (`āmra > amba > ām`).
- `@x-hindi-h-cluster` — h + y/v > jjʰ/bbʰ (`guhya > gujjʰa`, `jihvā > jibbʰā > jībʰ`).
- `@x-hindi-v-vocalise` — word-final -v (from -m-/-p-) vocalises to -u (`nā́man > nāũ`, Brj `dā̃u`).
- `Ø` is a placeholder segment (not a tag): `∅ > Ø / {non-OIA-final C} _ #` protects the stem-final consonant of starred
  verbal roots from final-consonant loss and from lenition; the lect's citation ending replaces it at the end.

Engine features I would have used:
- A way to mark a word class (verb / noun) from the etymon or gloss, so infinitive morphology is not guessed from `-ati`
  (numerals in `-ati`, `viṁśati`, get `-nā`/`-ab` now).
- Class-internal place agreement in replacements (`ṁ > homorganic nasal`), which now needs five rules.
- `{` `}` inside CDIAL etyma (`*ga{l}lati`) tokenise as segments and cannot be matched by a rule.

## 7. Round 2: vowels and accent

Scratch: `tmp/sound_changes/hindi/{try.py,cmp2.py,sc2.py,acctab.py,clacc.py,finu.py}`; LOO dumps `loo_r2_<lect>.txt`;
round-1 files `r2before/`, final round-2 copies `r2final_OIA-<lect>.sc`. No `%gold-drop` directive is used in any
hindi-group file, so every change below is a change of rules, not of scoring.

### 7.1 Train / test NED, round 1 → round 2

Round-1 numbers are the round-1 files re-scored with the current engine (differences from `summary-round1.csv` ≤ 0.0006).

| lect | train r1 → r2 | test r1 → r2 | test exact r1 → r2 | #rules r1 → r2 |
|---|---|---|---|---|
| H | 0.3595 → 0.3555 | 0.3618 → 0.3592 | 18.3 → 18.6 | 49 → 53 |
| OAw | 0.2449 → 0.2377 | 0.2596 → 0.2570 | 33.9 → 33.3 | 46 → 48 |
| Aw | 0.3598 → 0.3487 | 0.3093 → 0.3085 | 29.3 → 28.7 | 49 → 54 |
| Brj | 0.4002 → 0.3923 | 0.4360 → 0.4200 | 8.3 → 8.3 | 47 → 50 |
| bundeli_atarra | 0.5128 → 0.5105 | 0.5242 → 0.5226 | 4.3 → 4.3 | 39 → 41 |
| bagheli_lakshman | 0.4698 → 0.4667 | 0.5114 → 0.5042 | 6.0 → 6.8 | 33 → 35 |
| kaithal | 0.4721 → 0.4666 | 0.4872 → 0.4792 | 12.8 → 13.8 | 39 → 41 |
| NDu | 0.4923 → 0.4902 | 0.5031 → 0.4937 | 7.7 → 7.7 | 40 → 42 |
| SSr | 0.4673 → 0.4628 | 0.4838 → 0.4883 | 16.2 → 16.2 | 37 → 39 |
| Kannauji-Central | 0.4042 → 0.3970 | 0.4264 → 0.3963 | 13.5 → 16.2 | 38 → 41 |

No test NED rose by more than 0.005. SSr rose by 0.0045 (37 test pairs); all of it comes from the anusvāra rule
(7.2 e), which helps SSr train by 0.0014.

### 7.2 Audit of the stopped round-2 edits, and rules adopted

The first round-2 run left four kinds of edit. I re-tested each with `score --loo` and with swaps (Δ = train NED
of the alternative minus the adopted version):
- **CL restricted to disyllables** (`V G $2 > $1+long $3 / # C* _ V #`, replacing the penult-only `_ V #`): kept in
  H (+0.0011 if reverted), OAw (+0.0033), Aw (+0.0047), Brj (+0.0011), Bu (+0.0008), Bg (+0.0012), Kn (+0.0010).
  Rejected in Hr (+0.0008 if adopted), Ch (+0.0001) and Sg (+0.0014), which keep the penult version.
- **H anusvāra split** (long nasal vowel only in the first syllable, short elsewhere): kept (+0.0003 if reverted).
  It makes no difference in the other lects, so I did not copy it.
- **Aw syncope before a long vowel** (`a > ∅ / V C _ C Long`): kept, LOO +0.0013. The general `V C _ C V` form is
  +0.0005 worse. The same restriction hurts H (+0.0006) and Brj (+0.0008).
- **`if:!numeral` on the -ati → -nā/-ab/-no/-ai morphology.** The group lead asked for `if:verb,!numeral`, and
  I tested it. It helps Aw (−0.0009) and Brj (−0.0012), so both now use it. It hurts H (+0.0006), OAw (+0.0033),
  Bu (+0.0012) and Hr (+0.0019), which keep `if:!numeral` (or no flag). The `verb` flag misses real verbs whose
  CDIAL gloss is not "to …" (`váhati` > `bahnā`, `úpaviśati` > `baisnā`, `*duhati` > `duhnā`). `!numeral` already
  removes the numerals (`viṁśati`); only the -ti nouns (`kṣati`, `yuvatí`) still misfire.

New rules adopted (LOO = train-NED increase when the rule is removed; fires = etyma changed):

a. **Positional shortening of long vowels, before @acc-loss:**
   `{ā ī ū} > {a i u} / _ C* V C* V C* V C* V` applies to a long vowel with three or more OIA syllables after it.
   Tags: `@stress @x-hindi-pretonic-shortening`.
   - LOO: H +0.0005 (45 etyma), OAw +0.0005 (5), Aw +0.0007 (4).
   - H examples: `pānīyāmalaka` > `paniyālā`, `ānāyayati` > `anānā`, `āgʰrāpayati` > `agʰānā`,
     `*pūrayati` > `purānā`, `*ārdrakavaṭikā` > `adauṛī`.
   - Brj takes the wider version, with two or more syllables after the vowel (`_ C* V C* V C* V`): LOO +0.0027
     (10 etyma), e.g. `dyūtakāra` > `juār`, `*grāmadāra` > `gãvār`. That version hurts H (+0.0017), OAw (+0.0085)
     and the survey lects (+0.0007 to +0.0072).
   - The Vedic-accent version `_ !#* Acc` is rejected everywhere (7.3).
b. **Final -u > -ū, except oxytone -ú,** before @acc-loss: `u > ū / C _ #`. The literal `u` does not match `ú`.
   Tag `@stress @x-hindi-final-u`.
   - LOO: H +0.0008 (59 etyma), Hr +0.0033 (3), Kn +0.0026 (1), Bg +0.0016 (2), Brj +0.0009 (1), Bu +0.0003 (1).
   - Examples: barytone `áśru` > `ā̃sū`, `sáktu` > `sattū`, `mánu` > `manū`, `harídru` > `haldū`; oxytone
     `bāhú` > `bāh`, `vēṇú` > `ben`, `mr̩tyú` > `mīc`, `dadrú` > `dād`.
   - H counts (train reflexes): -ú → -ū 19% (n=31); post-tonic -u → -ū 48% (n=27); unaccented -u → -ū 36% (n=76).
   - Aw, Ch and Sg prefer the accent-blind `{u ú} > ū` (LOO +0.0010, +0.0008, +0.0030; there are only 2–9 accented
     items). OAw rejects both versions (+0.0004), because its spelling keeps final short vowels.
c. **`{ai au} > {ē ō}` added to Kn, Hr, Ch and Bg** (LOO +0.0026, +0.0022, +0.0013, +0.0002; 1–2 etyma each).
   Round 1 had pruned it there. It now scores because accented diphthongs are tokenised as one segment. Sg still
   rejects it (+0.0014). The rule is long-standing in H (LOO +0.0022, 67 etyma).
d. **OAw final long vowels shorten:** `ā > a / C _ #` extended to `{ā ī ū} > {a i u} / C _ #`, LOO +0.0058.
   Adding -ī and -ū gains 0.0009, e.g. `-ī` > `-i` in `sō̃ṭʰi`. This is Apabhraṃśa-style spelling. The same
   extension hurts H (+0.0021), Aw (+0.0013) and Brj, where -ī survives.
   Aw instead shortens only unaccented and post-tonic final -ī, before @acc-loss: `ī > i / C _ #`, LOO +0.0010
   (9 etyma). These are the whispered final vowels of the Aw source (`bʰaginī` > `bahinⁱ`, `sapátnī` > `savatⁱ`).
   Including tonic -ī́ costs +0.0012, so in Aw a tonic -ī́ keeps its length. The effect rests on about 15 reflexes.
e. **No anusvāra lengthening outside the NIA stressed syllable.** Placed before the lengthening rule:
   `V ṁ > $1+nas / _ C Long | _ C V C C | _ C V C V` (`@anusvara-vowel @stress`).
   - LOO: H +0.0010 (255 etyma), Aw +0.0010, Bu +0.0012, OAw +0.0006, Sg +0.0014, Kn +0.0010.
   - The rule fires before a heavy syllable (`áŋgāra` > `ãgār`, `aŋkūrá` > `ãkūr`, `śŕ̩ŋgāra` > `sĩgār`,
     `*ciŋgʰāṭa` > `cĩgʰāṛ`) or when two or more syllables follow (`máṇḍala` > `mãḍal`, `maṇḍapa` > `mãṛvā`).
     Lengthening remains in `píṇḍa` > `pī̃ḍ`, `ántara` > `ā̃tar`, `aŋgula` > `ā̃gul`.
   - Brj uses only the weight clauses (LOO +0.0014), since the ≥2-syllable clause costs it +0.0010.
   - Ch is +0.0004 if the rule is added; Hr and Bg are unaffected (their lists write N+C).
   - Together with the disyllable-only CL, this gives one NIA stress rule: a vowel lengthens by CL or anusvāra only
     in the stressed syllable of the NIA word, i.e. before a single light final syllable.

### 7.3 Hypotheses tested and rejected

Δ is the train-NED change when the rule is added. Unless stated otherwise, the rule was placed before @acc-loss.
H numbers first.

| hypothesis | rule | Δ H | Δ elsewhere |
|---|---|---|---|
| pretonic (Vedic) shortening of long vowels | `{ā ī ū} > {a i u} / _ !#* Acc` | +0.0062 | OAw +0.0118, Aw +0.0073, Brj +0.0094, Bu +0.0030, Bg +0.0079, Kn +0.0046, Hr +0.0044, Ch +0.0019, Sg +0.0024 |
| … first syllable only | `# C* _ !#* Acc` | +0.0045 | |
| weight: before a closed syllable | `_ C* V C C` | +0.0004 | |
| weight: before a long vowel (P/L-style) | `_ C* Long` | +0.0012 | OAw +0.0010, Aw +0.0020 |
| ū only, before a closed syllable | `{ū ū́} > u / _ C* V C C` | −0.0002 | too small; not adopted |
| CL only at the tonic vowel (accented words) | early `{á í ú} > long / # C* _ C C V #` + `if:!accented` CL | +0.0032 (control with every vowel: +0.0007) | OAw +0.0078 (ctrl +0.0037), Aw +0.0106 (ctrl +0.0033) |
| anusvāra lengthening only at the tonic | the same design with `_ Nasal Stop` | +0.0018 (ctrl +0.0001) | |
| tonic final -ā́ resists shortening and loss | `ā > a / C _ #` moved before @acc-loss (skips ā́) | +0.0005 (ctrl −0.0001) | OAw +0.0016 (ctrl +0.0008), Aw +0.0029 (ctrl +0.0005) |
| tonic -ī́ kept, other final -ī shortened | `ī > i / C _ #` | +0.0008 (all -ī: +0.0012) | Brj +0.0020 (all: +0.0063); adopted in Aw only (7.2 d) |
| syncope blocked in pretonic medial a | marker `a > ə / _ !#* Acc`, restored after vs before syncope | +0.0001 | |
| syncope blocked in tonic medial a | marker `á > ə` | +0.0008 | |
| syncope blocked in post-tonic medial a | marker `a > ə / Acc !#* _` | **−0.0011** | would need the accent carried past 45 rules; not adoptable |
| syncope: position / weight | `# C* V C _ C V` (2nd syllable only) | ±0.0000 (= general) | excluding the 2nd syllable: +0.0018; after a long syllable only: +0.0014; after a short one only: +0.0006; before a long V: +0.0006 |
| Haryanvi pretonic a loss (probe: 54% ∅, n=48) | `a > ∅ / V C _ C !#* Acc` | Hr +0.0018 | first-syllable version Hr +0.0203 |
| initial a- aphaeresis | `a > ∅ / # _ !#* Acc`; `{a á} > ∅ / # _ C V` | +0.0018; +0.0023 | |
| r̩ > u next to labials / before u | `r̩ > u / {p pʰ b bʰ m v} _` | +0.0003 | |
| r̩ > a when tonic / pretonic / before a | `ŕ̩ > a`, `r̩ > a / _ !#* Acc`, `r̩ > a / _ C* {a ā}` | +0.0005, +0.0006, +0.0012 | r̩ > i is unconditioned in H |
| two-mora law (long V before a geminate) | `{ā ī ū} > {a i u} / _ G $1`; `_ C C` early | +0.0008; +0.0011 | |
| CL blocked before a long final vowel (`pakkā`) | `V G $2 > … / # C* _ Short #` | −0.0001 | too small |
| ā shortened before ā in the next syllable | `ā > a / # C* _ C* ā` | +0.0002 | OAw −0.0008 on 6 etyma only; Aw, Brj +0.0003 / +0.0013 |
| CL restricted to non-starred etyma | `if:!starred` | +0.0015 | |

### 7.4 What the Vedic accent does and does not condition in the Hindi group

- **Syllable position and weight in the NIA word, not the Vedic accent, condition the vowel changes.**
  - CL and anusvāra lengthening happen only in the NIA stressed syllable: the first syllable of a disyllable,
    before a light final syllable (7.2 e).
  - Long vowels shorten three or more syllables from the end, two or more in Braj.
  - Schwa syncope targets the second syllable.
  - Every accent-based version of these changes is worse, by +0.002 to +0.012.
- **CL exceptions are not pretonic.** Splitting the words that CL helps or hurts by where the Vedic accent sits
  (`clacc.py`, H):
  - CL site pretonic (oxytone disyllables, `bʰaktá` > `bʰāt`): 78% helped (68/87);
  - CL site tonic: 70% (33/47);
  - unaccented attested etyma: 61%; Turner's starred etyma: 56%.
  So the exceptions are concentrated in the unaccented *-stratum (expressive words that keep their geminate,
  `*gaḍḍa`, `pakkā`). They are not pretonic. OAw and Aw show the same ranking.
- **Where the accent does matter:**
  - Final -u: oxytone -ú is lost like -a, while barytone and unaccented -u survive as -ū (H 19% vs 48% -ū;
    Hr 0/9 vs kept). Adopted in six lects (7.2 b).
  - In Aw, a tonic final -ī́ keeps its length while other final -ī is whispered or shortened (small n).
  - A weak marker test in H suggests that Vedic post-tonic medial a resists syncope (−0.0011). The engine cannot
    use this without carrying the accent through the chain.
  - Final -ā shows no accent effect.
- **The descriptive tables** (`acctab.py`, H train) agree:
  - Medial open a is lost in 38% of pretonic, 56% of tonic, 43% of post-tonic and 49% of unaccented cases.
  - Non-final long vowels are shortened in 20% of pretonic first syllables, 17% of tonic ones and 31–35% of
    unaccented ones.
  - First-syllable closed short vowels are lengthened in 58% of pretonic cases, 51% of tonic ones and 28–32% of
    unaccented ones.
  None of these puts the pretonic syllable on the weak side.

New tags: `@x-hindi-pretonic-shortening`: a long vowel shortens when it stands three or more syllables from the end
(two or more in Braj); this is positional, NIA pretonic, not Vedic. `@x-hindi-final-u`: OIA final -u > -ū
(barytone and unaccented only in the western lects).
Engine wish: a way to keep the accent visible to late rules (e.g. an accent tier that survives `@acc-loss`), so that
accent-conditioned syncope can be tested without marker vowels. The marker vowels disturb every intermediate rule
(+0.02 NED of noise).

## 10. Extension-aware score

All numbers use the 13:29 menu (aka, akka, ikā, uka, aḍaka, allaka, araka). "Before" is the file state at the
start of round 4, copied to `tmp/sound_changes/hindi/r4before/`. The rule files had been lightly retagged since
round 2 (`@mr-mb`, `@rt-retroflex`; H's notation line moved). Scratch tools in `tmp/sound_changes/hindi/`:
- `tryx.py`: variant tester reporting plain and ext NED;
- `scx.py`: before/after table;
- `extend_tab.py`: what the chain makes of each suffix;
- `ext_resid.py`: residual ending mismatches;
- `menu_cand.py`: candidate menu items;
- `extloo_<lect>.txt`: `--ext-loo` output.

### 10.1 Plain / extension-aware NED, before → after (train and test)

| lect | train plain | train ext | test plain | test ext | train ending choices after (%) |
|---|---|---|---|---|---|
| H | 0.3555 → 0.3555 | 0.2608 → 0.2573 | 0.3592 → 0.3588 | 0.2654 → 0.2610 | bare 59, aka 20, ikā 11, aḍaka 3, akka 3, araka 1, allaka 1, uka 1 |
| OAw | 0.2377 → 0.2362 | 0.1929 → 0.1924 | 0.2570 → 0.2560 | 0.2068 → 0.2067 | bare 75, aka 10, ikā 6, uka 6, akka 2, aḍaka 1, allaka 0 |
| Aw | 0.3487 → 0.3479 | 0.2802 → 0.2773 | 0.3085 → 0.3061 | 0.2535 → 0.2499 | bare 68, aka 14, ikā 8, aḍaka 5, uka 3, akka 2, allaka 0 |
| Brj | 0.3923 → 0.3930 | 0.3148 → 0.3101 | 0.4200 → 0.4200 | 0.3383 → 0.3383 | bare 62, aka 20, ikā 10, aḍaka 3, akka 2, allaka 2, uka 1 |
| bundeli_atarra | 0.5105 → 0.5109 | 0.4356 → 0.4125 | 0.5226 → 0.5215 | 0.4488 → 0.4238 | bare 51, aka 22, ikā 8, akka 7, araka 5, aḍaka 4, uka 2, allaka 1 |
| bagheli_lakshman | 0.4667 → 0.4656 | 0.3842 → 0.3638 | 0.5042 → 0.5006 | 0.4036 → 0.3773 | bare 49, aka 18, ikā 13, akka 9, aḍaka 4, araka 3, uka 3, allaka 1 |
| kaithal | 0.4666 → 0.4666 | 0.3600 → 0.3581 | 0.4792 → 0.4792 | 0.3579 → 0.3579 | bare 52, aka 22, ikā 16, aḍaka 4, uka 3, akka 1, allaka 1, araka 1 |
| NDu | 0.4902 → 0.4887 | 0.3847 → 0.3803 | 0.4937 → 0.4937 | 0.3709 → 0.3659 | bare 48, aka 21, ikā 18, aḍaka 5, uka 3, akka 2, araka 2, allaka 1 |
| SSr | 0.4628 → 0.4610 | 0.3633 → 0.3531 | 0.4883 → 0.4883 | 0.3536 → 0.3400 | bare 49, ikā 21, aka 17, aḍaka 7, akka 2, uka 2, araka 1, allaka 1 |
| Kannauji-Central | 0.3970 → 0.3927 | 0.3223 → 0.3138 | 0.3963 → 0.3918 | 0.3100 → 0.3114 | bare 65, aka 13, ikā 11, aḍaka 4, araka 3, uka 2, allaka 2 (akka 0) |

- Plain train NED rises by at most 0.0007 (Brj) and 0.0004 (Bu). Every other lect improves or is unchanged.
- The accent scores are unchanged in H, OAw, Aw and kaithal, the lects that have one.
  - A first version of the H CL edit moved H's accent NED on its 4 accent items (0.6652 → 0.7009): it lengthened
    `utsr̩ta` > `ūcʰī` against the tatsama "reflex" `úccʰrita`.
  - I replaced it with the explicit extension-hiatus version below, which restores the accent score and is
    0.0001 better on plain NED.

### 10.2 Does the chain derive the extensions cleanly?

Output of `extend_tab.py`, before the fixes:
- **-aka.** H, Aw, Hr, Ch, Sg and Kn gave -ā (≈90%) and Brj gave -o (86%).
  - Bundeli and Bagheli gave bare -a: they had no hiatus contraction, so -aa lost its second a.
  - Kannauji gave -ā, although its list writes the Western -o (20% of final vowels, against 14% -ā).
- **-ikā** gave -ī everywhere except Bagheli (-iā, because its -iā > -ī rule had been pruned). OAw gave -ī.
- **-uka** gave -u (apocope of the hiatus -a), and OAw gave -ua. H wants -ū (`bālū`, `ālū`).
- **-aḍaka / -allaka / -araka** gave -ṛā / -lā / -rā in H, Hr, Ch, Sg and Kn; -rā (flap r) in OAw and Aw; -ro / -lo
  in Brj. After the fixes Bundeli and Bagheli give -ṛā / -lā / -rā.
- **Remaining leftovers.** About 5–10% of each suffix still surfaces as V + suffix vowel after stems ending in a
  vowel (`áṁsiyaka` > `āsiā`, H `-iyā`). The ya-śruti rule below spells some of these as -iyā.
- **The CL restriction starved the -aka variants.** CL was limited to disyllables (§7), so `*akkʰaka`
  (a·kkʰa·a) got no lengthening and gave `akʰā` against attested `ākʰā`. Allowing the hiatus vowel fixes this in
  H, Aw, Brj, Kn, Ch and Sg.

### 10.3 Rules changed

"ext LOO" is the `--ext-loo` value; Δ is the change when the edit is adopted, measured with `tryx.py`.

| lect | change | Δ plain / Δ ext (train) | ext LOO |
|---|---|---|---|
| H | `{u ū} {a ā} > ū / _ #` (@hiatus @morph: -uka > -ū) | −0.0002 / −0.0004 | +0.0003 |
| H | `∅ > y / {i ī} _ {a ā}` (@ya-sruti: -iyā) | ±0 / −0.0008 | +0.0002 |
| H | CL also before the hiatus left by -aka / -ikā / -uka: `/ # C* _ V # \| # C* _ {a ā} a # \| # C* _ i ā # \| # C* _ u a #` | ±0 / −0.0025 | (generic `(V)` version: +0.0100) |
| Aw, Brj, NDu, SSr, Kn | CL `_ V #` → `_ V (V) #` (the same idea) | Aw −0.0002/−0.0019; Brj +0.0007/−0.0047; Ch −0.0015/−0.0044; Sg −0.0018/−0.0102; Kn ±0/−0.0028 | +0.0243, +0.0163, +0.0165, +0.0255, +0.0143 |
| Kn | `{a ā} a > o / _ #` (@hiatus @morph: -aka > -o, as in Brj) | ±0 / −0.0023 | +0.0030 |
| Bu, Bg | general hiatus `{a ā} {a ā} > ā` restored (it had been pruned) | Bu +0.0004/−0.0202; Bg −0.0013/−0.0054 | +0.0231, +0.0227 |
| Bg | `{i ī} {a ā} > ī / _ #` restored (-ikā > -ī) | +0.0001 / −0.0104 | +0.0104 |
| Hr | `{u ū} {a ā} > ū / _ #` | ±0 / −0.0020 | +0.0020 |
| OAw, Aw | `∅ > y / {i ī} _ {a ā}` | OAw −0.0006/−0.0002; Aw −0.0002/−0.0009 | +0.0008, +0.0009 |
| Kn, OAw, Aw | root-protection `∅ > Ø / C _ #` restricted to `if:starred` (the citation ending belongs to Turner's starred roots, not to attested C-final nouns) | Kn −0.0043/−0.0039; OAw −0.0009/−0.0003; Aw −0.0004/−0.0002 | (H: ±0/+0.0002, not changed) |

Tested and not adopted:
- **Bu `{a ā} a > o`** (−0.0094 ext): half as good as -ā. The Bundeli list has -ā 22% against -o 11%.
- **Aw -uka > -uā** (`{u ū} a > u ā / _ #`): plain −0.0014 but ext +0.0012.
- **-uka > -ū in Aw:** +0.0051 ext; Aw keeps -uā / -u.
- **Ya-śruti elsewhere:** it hurts Kn, Bu, Bg, Brj, Hr, Ch and Sg (+0.0005 to +0.0083 ext).
- **CL `(V)` in Bg** (plain +0.0011), **Bu** and **Hr**: rejected.
- **Rules kept for plain NED despite negative ext LOO:**
  - `@x-hindi-final-u` (H −0.0005, Sg −0.0012, Ch −0.0003): under the extension-aware score, -uka supplies the
    same -ū;
  - Aw syncope before a long vowel (−0.0015);
  - OAw `a a > e / C _ C` (−0.0005).
- **@morph interactions.** Verbs are never extended, so the -nā / -ab / -no / -ai citation rules cannot fire on
  variants. The only harmful interaction was the root-protection `Ø`, fixed above.

Correction to §7: the pretonic-shortening environment `_ C* V C* V C* V C* V` requires *four* following vowels
(the Braj version three), not "≥3 / ≥2 syllables" as written there.

### 10.4 What the choices say

- **-aka is the main extension everywhere** (13–22% of train reflexes). Hindi, Hr, Bu and Ch-Sg choose it about
  as often as H (20–22%). Awadhi and Kannauji choose it less (13–14%), and OAw least (10%): its verse forms are
  mostly bare stems (75% bare).
- **-ikā is strongest in E. Hindi.** Chhattisgarhi and Surgujia take it for 18–21% of reflexes (Sg even more than
  -aka). Hr takes it for 16%, the rest for 6–13%. These are the feminine/diminutive -ī lists.
- **-ḍa (-aḍaka > -ṛā) is productive in the east and south.** It takes 7% in Sg, 5% in Ch and Aw, and 4% in Bu,
  Bg, Hr and Kn, but 3% in H and Brj and 1% in OAw. Examples: Aw `kiravā`-type forms, Chhattisgarhi -ṛā nouns.
- **-ra (-araka) is productive only in Bundeli and Bagheli** (5% and 3%) and marginal elsewhere (≤3%).
- **-la (-allaka) is marginal everywhere** (0–2%).
- **-akka** is chosen 7–9% in Bu and Bg, but there it only yields a -Vk ending (`agnakka` > `agak`). I read that as a
  surface fit to consonant-final survey forms, not as -kk- morphology. It is 1–3% elsewhere and 0% in Kn.

### 10.5 Should the menu change? (evidence for the coordinator)

Method: `menu_cand.py` adds one candidate suffix to the current menu and counts the train reflexes whose closest
variant becomes strictly better, and the resulting ext-NED gain. It also includes a control variant `-a`, which
replaces the headword's final vowel with -a and thereby lets -i/-u stems drop their vowel. The control alone
gains H −0.0044 ext. Many -ra / -la / -ā "fixes" are that effect, not the suffix, so the counts below are over
menu + control.

| candidate | H | Aw | Bu | Bg | Ch | Hr | verdict |
|---|---|---|---|---|---|---|---|
| **-iyaka** (> -iyā diminutive) | 133 refl. (1.7%), −0.0034 | 24 (3.8%), −0.0089 | 84 (7.1%), −0.0066 | 22 (2.9%), −0.0027 | 39 (7.6%), −0.0098 | 0 | **add.** Real NIA diminutive/adjectival -iyā (H `aṭiyā`, `ḍibiyā`; Aw `ãbiyā`; Bu, Ch -iyā); the biggest gain in the eastern and Bundeli lists |
| -illaka (> -ilā) | 112 (1.4%), −0.0018 | 23 (3.6%), −0.0047 | 43 (3.6%), −0.0039 | 12, −0.0017 | 32 (6.2%), −0.0063 | 0 | real MIA -illa, but it largely overlaps -iyaka (the same -iyā reflexes win with either); second priority |
| -ṭṭaka (> -ṭā) | 88 (1.1%), −0.0014 | 8, −0.0014 | 32 (2.7%), −0.0027 | 9, −0.0015 | 11, −0.0024 | 5, −0.0021 | real Pk -ṭṭa-, but modest |
| plain feminine -ī | 12 (0.2%), −0.0002 | 3, −0.0010 | 13 (1.1%), −0.0020 | 6, −0.0023 | 0 | 1 | no. Once the stem-final vowel can be replaced, -ikā covers it |
| -ḍa without -ka | 66 (0.8%), −0.0010 | 29 (4.5%), −0.0062 | 26, −0.0025 | 14, −0.0027 | 5 | 3 | no. The Aw "fixes" are surface (`avaśyā` → `osḍ` = `os`); -aḍaka covers the real -ṛā |
| -ra / -la without -ka | 149 / 89 (1.9 / 1.1%) | | | | | | no. The gains come from truncating the stem (`akṣōṭa` → `akʰoṭ` = `akʰroṭ`), not from a suffix |
| -ikka, -ullaka, -ālaka | ≤ 0.8%, −0.0006 to −0.0010 | ≤ 1.4% | -ikka 5% (surface: `aŋgúli` → `ãgurik` ≈ `aŋgūli`) | ≤ 2% | ≤ 2% | 0 | no |

- **Remaining ending mismatches** under the extension-aware choice (`ext_resid.py`) are dominated by ∅ ↔ -ā:
  H C→ā 316 and ā→C 306 (per reflex, where no variant fits), plus C→ī 100 (`agní` > `āgī`).
  - The C→ī and C→ā cases are i-/u-stems and consonant stems that take a gender vowel without any -ka. A "replace
    the stem-final vowel" option (the `-a` control, −0.0044 in H) or plain thematisation would catch more than any
    new suffix. I recommend it as a menu option ("thematise": final V → -a, then the chain) if the coordinator wants
    one surface-neutral addition.
  - The survey lects' remaining mismatches are Bundeli -o vs -ā (C→o 87, ā→o 45), which is dialect mixture in the
    lists rather than a missing suffix, and Chhattisgarhi -ɛ (oblique/verbal), which is morphology outside the menu.
- **Never or almost never chosen:** -allaka (0% OAw, Aw; ≤2% elsewhere) and -uka (1–3%, except OAw 6%). Both are
  cheap to keep. -akka is 0% in Kannauji and 1–3% in the CDIAL lects; in Bu and Bg its 7–9% is a surface fit.
  I would not drop anything, but -akka should not be read as evidence of -kk- morphology in this group.

## 11. Verb citation forms

Round 5 was cut short (hard stop at 18:27). Only H and Aw were changed. The baseline files are in
`tmp/sound_changes/hindi/r5before/`, and the tools are `scv.py`, `tryv.py` and `vshow.py`.

In the tables, "plain" is verbs scored with the @morph citation rules; "menu" is the verb menu, with every @morph
rule switched off.

**Hindi-Urdu (H)**

| | test plain | test menu | train menu |
|---|---|---|---|
| before | 0.2566 | 0.2219 | 0.2311 |
| after | 0.2567 | 0.2116 | 0.2207 |

- The -nā infinitive comes from -anaka.
- Syncope now runs right to left. The rule is `a > ∅ / V C _ C V // _ C a C V`: an a is not dropped when the next a
  is itself syncopable (`kaṣakk-anaka` > `kasaknā`, not `kasknā`; `anuharnā`, `ujʰalnā`).
- The rule replaces `use syncope.a.vccv`. Measured deltas: plain train +0.0001, ext train −0.0024, verb menu
  −0.0104, accent unchanged.

**Awadhi (Aw)**

| | test plain | test menu | train menu |
|---|---|---|---|
| before | 0.1860 | 0.2768 | 0.3106 |
| after | 0.1860 | 0.1791 | 0.1807 |

The -ab infinitive comes from the gerundive -tavya, through the MIA -iavva. Rules added:
- `∅ > i / C _ t a v y a #` (@anaptyxis): the thematic -i- of -itavya, generalised
  (`car-tavya` > `cariavva`). Its conditioning is segmental, but it encodes the set-root formation, so it borders on
  morphology.
- `i > ∅ / _ a C #` and `{ā a} a > ā / _ C #` (@hiatus): `cariab` > `carab`, `kʰāiab` > `kʰāb`. These also help
  plain NED (−0.0005).
- CL environment extended with `| # C* _ (i) a b b a #` (`kaḍḍʰiabba` > `kāṛʰab`).

Train effect of the Aw changes, all measured before adoption: ext 0.2833 → 0.2632 and plain 0.3479 → 0.3474;
the accent score is unchanged.

**Not done:**
- OAw: verb menu already better than plain (test 0.161 vs 0.172; 3sg -ai from -ati).
- Brj: -no from -anaka; menu 0.329 vs plain 0.439 test, 11 test pairs.
- kaithal: 2 test verbs.
- Bundeli, Bagheli, Chhattisgarhi, Surgujia and Kannauji: 0–2 verb pairs each.

## §12 Hindi residual pass (all 6,632 misses classified)

Error share: endings 33% (mostly citation variants: bare vs -ā, -ī vs ∅, -nā vs bare stems, and
regional -r/-ṛ forms Turner lists under H.), consonant loss/insertion 19%, consonant quality 10%,
syncope 15%, vowel length 7%. Changes:

- **verb_form bug (all lects):** verb headwords that are not -ti presents (participles jita, bhagna;
  starred roots *cammakka; -yatēaya causatives) kept their final vowel, so jita + -anaka became
  *jitenā. Now the final vowel drops (and -tē/-ti + aya is recognised). Verb-menu NED better in 54
  lects, worse in 10; H 0.2196 → 0.2090 train.
- **Causative -āpanaka/-āpanakam on the verb menu (all lects):** 285 of 1,917 H verb pairs are
  -ānā/-vānā causatives. Against a placebo whose consonant survives (-āranaka), the real item wins
  in 16 lects vs 5 (P, B, N ≈ −0.02); -ātanaka was a bad placebo (intervocalic t loss mimics -ānā).
- **H `v > ∅ / ā _ n ā #`:** -āvnā > -ānā (aṭānā, ãtarānā): verb menu −0.016 / −0.017, ext −0.004.
- **`a ī > ī / C _ #`** in H, Bagheli, Chhattisgarhi, Haryanvi (rejected in Braj, Bundeli, Surgujia,
  Kannauji, where it hurt).

Rejected: ṇḍ > Ṽṛ, final l > r, final ṛ > r (all worse: regional variants within CDIAL's H.), final
v > u (plain worse). Remaining verb gaps: -ak- (29), -al-/-ar- (18) extensions, reduplication (24).

## Round 6 audit

Method: the brief's `residuals.py` + `gates.py`. Scratch tools are in `tmp/sound_changes/hindi/`: `gatex.py` (the
same gates, with the baseline scored once) and `xdiff.py` (words made better / worse, per reflex, on the
extension-aware choice). Reports are in `tmp/sound_changes/hindi/r6/`, and the pre-audit files in `r6before/`.
Every installed change passed the gates on its own and again in the final combined check; the final `gates.py` run
of `r6before` against the installed files gives ACCEPT for all ten lects.

### Before → after

| lect | ext train | ext test | verb train | plain train | ext-exact train |
|---|---|---|---|---|---|
| H | 0.2363 → 0.2336 | 0.2392 → 0.2374 | 0.1855 → 0.1763 | 0.3552 → 0.3547 | 33.0 → 33.6% |
| OAw | 0.1894 → 0.1787 | 0.1985 → 0.1921 | 0.1332 → 0.1312 | 0.2464 → 0.2425 | 39.9 → 42.6% |
| Aw | 0.2603 → 0.2569 | 0.2403 → 0.2372 | 0.1598 → 0.1585 | 0.3533 → 0.3525 | 33.0 → 33.4% |
| Brj | 0.3010 → 0.2958 | 0.3227 → 0.3192 | 0.3577 → 0.3337 | 0.3930 → 0.3881 | 22.0 → 22.5% |
| bundeli_atarra | 0.4056 → 0.3996 | 0.4132 → 0.4081 | – | 0.5109 → 0.5092 | 9.4 → 9.7% |
| bagheli_lakshman | 0.3591 → 0.3566 | 0.3702 → 0.3635 | – | 0.4658 → 0.4656 | 13.5 → 13.5% |
| kaithal | 0.3543 → 0.3485 | 0.3511 → 0.3424 | 0.5151 → 0.5040 | 0.4666 → 0.4647 | 18.1 → 19.2% |
| NDu | 0.3678 → 0.3633 | 0.3511 → 0.3485 | – | 0.4891 → 0.4861 | 14.2 → 15.0% |
| SSr | 0.3443 → 0.3397 | 0.3400 → 0.3400 | – | 0.4610 → 0.4595 | 17.4 → 18.1% |
| Kannauji-Central | 0.3130 → 0.3070 | 0.3069 → 0.2979 | – | 0.3970 → 0.3915 | 26.8 → 28.3% |

Accent NED is unchanged everywhere.

### Error profiles

`residuals.py` classes, share of error:
- **Endings: 34% (H) to 54% (Hr).** These are mostly the bare vs -ā / -ī / -iyā / -o choice and lexical suffixes
  (-rīyā, -aīyā in Bundeli).
- **Consonant insertion / deletion: 13–21%.**
- **Consonant quality: 5–12%.** In the survey lists this is largely notation: Koshy's retroflex dentals; ṛ written ḍ.
- **Syncope and vowel-missing classes: 4–8% each.**
- **Vowel length: 1–7%.**
- **Bundeli and Bagheli are dominated by list noise.** Bundeli mixes -o and -ā in one list. Koshy writes schwa as e
  and dentals as retroflex.

### Changes kept (Δ = train ext, from the individual gate runs)

1. **uva / upa > o except word-finally** (`u v a > o // _ #`, @aya-e). This parallels ava > o: `suvarṇa` > `sonā`,
   `upālabʰatē` > `olahnā`, `yuvatí` > `joī`.
   - Kept in H (−0.0006; 34 words better, 20 worse), NDu (−0.0028), Aw (−0.0011), Brj (−0.0021), Bu (−0.0011),
     SSr (−0.0038), Kn (−0.0035).
   - Rejected in OAw (verb +0.0018), Bg (test worse) and Hr (no effect).
2. **ō/ē + a/i hiatus contraction** (`{ō ē (o e)} {a i} > … / _ C`, @hiatus): `*cʰōkara` > `cʰorā`,
   `jyautiṣika` > `josī`.
   - Kept in H (−0.0004), Brj (−0.0014), Aw (−0.0008), Bu (−0.0011), Bg (−0.0001; test −0.0059), Hr (−0.0016),
     Kn (−0.0025), OAw (−0.0003).
   - No effect in Ch and Sg.
3. **Long-vowel shortening before the diminutive -i(y)ā** (`{ā ī ū} > {a i u} / _ C i (y) ā #`, @stress
   @x-hindi-pretonic-shortening): `stʰālī` > `tʰariyā`, `kāla` > `kariyā`.
   - Aw uses the `y` version (−0.0012; 6 better, 0 worse).
   - The others use `(y)`: Bu −0.0038, Bg −0.0024, Ch −0.0017, Sg −0.0008, Hr −0.0002 (test −0.0021),
     H −0.0003.
   - Brj and Kn: no effect.
4. **OAw hiatus** (@hiatus, together −0.0068; 52 better, 27 worse):
   - `{u ū} a > u / _ #`: -uka > -ua > -u, the OAw masc. -u (`gālu`, `lāhu`, `bʰālu`);
   - `{a ā} ā > ā` (`piyārā`, `cirānā`);
   - `i ī > ī` (`dukʰī`);
   - `∅ > y / # (C) a _ ā` (@ya-sruti: `payāna`, `ayāna`, `bayāri`; −0.0020, 6 better, 1 worse).
5. **OAw -m- in the final syllable:** `V m V > $1 v $3+nas / _ #` (@m-v). The nasality goes onto the final vowel
   (`daśamá` > `dasavaṁ` / `dasauṁ`, `dāḍima` > `dārivaṁ`, `dʰūmá` > `dʰūvaṁ`). Elsewhere `V m > V+nas v` as before.
   −0.0017; 9 better, 1 worse.
6. **Aw `a ī > ī / C _ #`** (@hiatus, as H already had): `pōstaka+ikā` > `potʰī`. −0.0003, test −0.0015.
7. **Haryanvi syncope** (@syncope, right to left, not before the -ṇā/-nā infinitive):
   `a > ∅ / V C _ C V // _ {n ṇ} {ā a} # | _ C a C V`.
   - Examples: `bakri`, `cʰokri`, `camḍi`, `musli`, `mogrā`, `gillā`.
   - −0.0040 ext, verb −0.0111; 33 better, 18 worse.
   - Without the -ṇā exception it fails the verb gate (+0.048): Hr keeps the a of -aṇā.
8. **CL before the verbal noun -ana(k)a** (the CL environment gains `| # C* _ a n a a #`).
   - H: −0.0014 ext, verb −0.0067 train / −0.0080 test. Examples: `búdʰyatē` > `būjʰnā`, `bʰagna` > `bʰāgnā`,
     `vartáyati` > `bāṭnā`.
   - Braj: `-no`, −0.0017 ext, verb −0.0240.
   - H is 120 better, 81 worse. The losers are short-vowel intransitives (`ḍubnā`, `tʰaknā`, `uṛnā`, `rusnā`), i.e.
     the Hindi intransitive / causative vowel alternation, which no single sound change captures.
   - Hr: no effect.

### Rejected

- **Bundeli:**
  - m > ṽ (ext +0.0005);
  - syncope in any form;
  - l > r blocked before a final vowel: plain +0.0065 for −0.0002 ext. Bu `pʰal` / `bāl` vs `kāro` is lexical.
- **Bagheli:**
  - syncope: test ext +0.0097;
  - l > r: test worse. Koshy writes ḍ for many r.
- **Aw:**
  - Dropping syncope before a long vowel passes the gates (ext −0.0013, plain +0.0017), but it is 28 better vs 34
    worse. Syncope is real in stems (`tircʰā`, `kabrā`, `janvās`); only the -araka / -allaka variants suffer. Not
    kept. Blocking it between like consonants makes test worse.
  - `{a ā} ī > ī`: plain +0.0008.
- **OAw:** `ā a > ā / _ C` (ext +0.0001); plain `∅ > y / a _ ā` (verb +0.0013).
- **Kannauji:**
  - m > v: 1 word only;
  - `V m > Ṽv`: +0.0061;
  - l > r: plain +0.0013.
- **Braj:** `i a > ī`: test worse.
- **Ch:** l > r before a final vowel: plain +0.0059.

### What remains (lexical or source variation, left alone)

- **Endings:** bare vs -ā / -o / -ī, the Bundeli -o / -ā mixture, and Chhattisgarhi -ɛ / -s / -tʰɛ verb forms (the
  survey lists do not flag verbs, so these get noun extensions).
- **Numerals:** -saptati > -hattar, -catvāriṁśat > -tālīs (Aw).
- **Spontaneous nasalisation:** H 4.3%, e.g. `bʰaktikā` > `bʰā̃tī`.
- **H intransitive short vowels** (see 8 above).
- **Regional l / r and ṛ / ḍ doublets** within one lect's sources.
- **Expressive / learned strata:** geminates kept, OIA clusters written.

**Not done:** nothing is left undone at the lect level (all ten lects audited). H was only checked for the
residue classes (vowel length and syncope); its §12 changes were not redone.

### Round 6 second pass

All numbers are re-baselined with the mid-round engine fixes (residuals now apply the `%source` tails; IPA stress
marks are stripped; `verb_form` keeps ś). The table compares the pre-round-6 files (`r6before/`) with the installed
files; `gates.py` gives ACCEPT for every lect.

| lect | ext train | ext test | verb train | verb test | plain train |
|---|---|---|---|---|---|
| H | 0.2363 → 0.2297 | 0.2392 → 0.2336 | 0.1855 → 0.1621 | 0.1785 → 0.1570 | 0.3552 → 0.3537 |
| OAw | 0.1894 → 0.1779 | 0.1985 → 0.1921 | 0.1332 → 0.1317 | 0.1420 → 0.1420 | 0.2464 → 0.2420 |
| Aw | 0.2603 → 0.2558 | 0.2403 → 0.2364 | 0.1598 → 0.1569 | 0.1763 → 0.1763 | 0.3533 → 0.3516 |
| Brj | 0.3010 → 0.2958 | 0.3227 → 0.3192 | 0.3577 → 0.3337 | 0.3106 → 0.3106 | 0.3930 → 0.3881 |
| bundeli_atarra | 0.4056 → 0.3996 | 0.4132 → 0.4081 | – | – | 0.5109 → 0.5092 |
| bagheli_lakshman | 0.3591 → 0.3554 | 0.3702 → 0.3627 | – | – | 0.4658 → 0.4648 |
| kaithal | 0.3543 → 0.3474 | 0.3511 → 0.3410 | 0.5151 → 0.5040 | 0.3304 → 0.3304 | 0.4666 → 0.4647 |
| NDu | 0.3678 → 0.3633 | 0.3511 → 0.3485 | – | – | 0.4891 → 0.4861 |
| SSr | 0.3443 → 0.3388 | 0.3400 → 0.3400 | – | – | 0.4610 → 0.4595 |
| Kannauji-Central | 0.3130 → 0.3070 | 0.3069 → 0.2979 | – | – | 0.3970 → 0.3915 |

**(1) H vowel length and syncope.**

The vowel-length misses were tabulated by direction × syllable position × open/closed × next-vowel weight
(`vlen.py`). The largest cells are:
- **short → long in the first syllable of NIA disyllables that were OIA trisyllables** (105 + 69 pairs): `sajjana` >
  `sājan`, `piñjara` > `pī̃jar`, `garbʰiṇī` > `gābʰin`.
  - Extending CL to `# C* _ V C {a i u} #` is rejected: plain +0.0031, ext +0.0003. Letting the anusvāra
    lengthening into these words gains ext only −0.0001 against plain +0.0006, so it is not kept.
  - The trisyllabic CL seen in `sājan` is lexical in H: it helps as many words as it hurts.
- **Verbs with a nasal cluster** (54): `mī̃jnā`, `bā̃dʰnā`. Adopted: the anusvāra no-lengthening rule gets the
  exception `// _ C a n a a #` (the root syllable before -ana(k)a is stressed).
  - Δ ext −0.0014 train / −0.0010 test; verb −0.0068 / −0.0053.
  - 55 better, 4 worse.

Syncope:
- `a` syncope extended to `i` (`{a i} > ∅ / V C _ C V // _ C a C V`: `laṛikā` > `laṛkā`, `parinī` > `parnī`,
  `mukʰiyā` > `mukʰyā`). This also works for Haryanvi (ext −0.0011).
- Also including u is worse on plain.
- Blocked in Braj: test ext +0.0033.

Other H changes:
- **v-loss after a rounded vowel** (`v > ∅ / {u ū ō} _ V`, @v-loss): `kūā`, `pūā`, `goāl`, `duār`. Together with
  i-syncope: ext −0.0009 / −0.0012; 142 better, 70 worse. Also adopted in Aw (−0.0011), Bg (−0.0012),
  OAw (−0.0008) and Sg (−0.0009). Rejected in Ch and Hr (plain / no effect); Brj and Bu are ≈ 0, so not added.
- **Near-miss pairs in H verbs.** Two regular fixes were found (together ext −0.0008, verb −0.0030; 37 better,
  5 worse):
  - `m y > m m` (ramyati > `rāmnā`, *viśamyatē > `bisamnā`);
  - `d v > b b / V _` (udvartman > `ubaṭ`, udvijatē > `ūbnā`).
  Neither fires in OAw, Aw or Brj.
- **`%verb-menu +aka`** (coordinator item), installed in H:
  - verb −0.0042 / −0.0052, ext −0.0008 / −0.0010;
  - placebo `+ara`: −0.0003 / −0.0002.

**(2) Verbs in other lects.**
- **OAw** (3sg -ai from -ati, 40 of 53 verb misses): the residue is kṣ > cʰ (4), causative -āvai / -āi (4), and CL
  in -ya- passives (`sakai`, `pacai`).
  - An `∅ > v / {ā u} _ a i #` glide (`cʰāvai`, `lāvai`) passes the gates but is 4 better / 3 worse, so it is not
    kept.
- **Aw:** the -itavya item now on the menu wins 41 / 49, so the round-5 anaptyxis rule is commented out (done by
  the coordinator).
  - The remaining Aw verb misses are flap r / ṛ doublets and single words.
- **H verb gaps** that remain:
  - ā→∅ (74): the -āpanaka item beats -anaka because CL-before-anaka lengthens -ya- passive / intransitive stems
    (`rakʰnā`, `kʰipnā`, `paknā`). The short-vowel intransitive class is not predictable from the segments.
  - l / r and o / u doublets.
  - kṣ > cʰ (11).
  - -ak- / -al- / -ar- extensions and reduplication: not given a menu item, since no same-shape placebo test was run.

**(3) Near-misses (NED ≤ 0.25), grouped by segment pair.**
- **H:**
  - a↔ā (43) and i/u↔ī/ū (63): the trisyllabic CL above, lexical.
  - -ā↔∅ (43): citation variant.
  - l↔r (32) and ṛ↔r (20): CDIAL regional doublets, rejected in §12.
  - v↔∅ (53): fixed above.
  - i/u↔∅ (62): i-syncope fixed; u rejected.
  - ã↔a / ā̃ (both directions, ≈20 each): lexical nasality.
- **OAw:**
  - -a↔-ai / -i (23): locative / oblique citations.
  - ṁ↔∅ in m > v (9 each way): the doublets cancel, and plain `m > v` is +0.0048.
  - CL in trisyllables `sājana`: plain +0.0017, test ext worse.
- **Aw:**
  - the -allaka variant (8);
  - v↔∅ (5): fixed;
  - CL in trisyllables (4).
- **Brj, Ch, Hr:**
  - bare vs -ā / -o / -ī;
  - initial cʰ→c in Ch and Hr (4 each): too few words, and probably the survey lists' spelling.

**Not done:** Bundeli, Bagheli, Kannauji and Surgujia got no second-pass near-miss reading. For them only the shared
checks were run: v-loss, i-syncope and the -iyā shortening.

### Round 6 third pass

This pass re-baselines the survey lects after the new `word_flags` rule, which now marks clause glosses such as
"he killed" as verbs. The survey lists now have verbs:

| lect | train verbs | test verbs |
|---|---|---|
| Bg | 63 | 7 |
| Bu | 26 | 0 |
| Ch | 25 | 4 |
| Kn | 10 | 0 |
| Hr | 5 | 2 |
| Sg | 4 | 1 |

All numbers below are train / test NED from `gates.py`: the re-baselined `r6c/` files against the installed files.

| lect | ext | verb | kept |
|---|---|---|---|
| NDu | 0.3613 → 0.3607 / 0.3474 → 0.3469 | 0.3760 → 0.3640 / 0.4167 → 0.4000 | `%verb-menu +itasya` |
| SSr | 0.3383 → 0.3378 / 0.3400 → 0.3400 | 0.4583 → 0.4417 / 0.6000 → 0.6000 | `%verb-menu +itasya` |
| bagheli_lakshman | 0.3517 → 0.3516 / 0.3627 → 0.3621 | 0.3201 → 0.3180 / 0.3976 → 0.3786 | `%verb-menu +itasya` |
| Kannauji-Central | 0.3127 → 0.3118 / 0.2979 → 0.2979 | 0.3283 → 0.3167 / – | `%verb-menu +itaka` |
| bundeli_atarra, kaithal | unchanged | unchanged | – |

Plain NED is unchanged everywhere.

**Verb citation forms in the survey lists**
- **E. Hindi 3sg past -is** (Ch `dekʰīs`, `marīs`, `jarīs`; Bg `marīs`): participle -ita + enclitic -sya.
  - `+itasya` beats its same-shape placebo `+itapya` (Δ exactly 0) in Ch, Sg and Bg.
  - It does nothing in Bu, Kn and Hr: the -is past is an eastern isogloss.
- **W. Hindi past -(y)o** (Kn `pīyo`, `khāo`): participle -itaka.
  - `+itaka` beats the placebo `+iraka` (Δ 0) in Kannauji.
  - No effect in Bundeli, whose verbs are mostly bare stems and imperatives.
- **Bagheli -na infinitive** (`celna`, `cebna`): needs syncope of the -ana- vowel. `a > ∅ / V C _ n {a ā} #` gains
  only −0.0002 ext (3 words), so it is not kept.
- **Bagheli / Ch present participle -at(a)** (`celeṭa`, `uḍeṭa`): `+anta` and its placebo `+anka` both have no effect.
  This formation still has no working menu item: `+anta` never wins.

**Near-miss reading (NED ≤ 0.25; Bu, Bg, Sg, Kn)**
- **Bundeli:**
  - -k↔∅ (28): an artefact of the -akka / -ikka variants against bare -i / -a forms;
  - ∅↔-o (21: `sāro`, `goḍo`, `būro`) and -ā↔-o: the list's -o / -ā mixture;
  - ḍ↔r (7: `kīrā`, `makarī`): the list writes the flap both ways;
  - l↔r doublets.
  None is regular.
- **Bagheli:**
  - Koshy's a→e over- and under-applies (`kanṭa`, `ṭaŋga`);
  - l↔r and ḍ↔r doublets;
  - -īa / -iyā.
- **Initial cʰ > c** (Sg `cālī`, `cānī`; Bg `caṭ`, `cenī`; Hr `cokrā`; Ch `cerī`).
  - It passes the gates in Sg, Bg and Hr but fails the test split in Ch, although Ch and Sg share one source.
  - Word by word it is 13 better / 7 worse in Bg, 6 / 5 in Hr and 6 / 4 in Ch, with the *same etyma* on both sides
    (`cʰōkara` > `corā` and `cʰorā`).
  - It is source-internal variation, not a sound change. Not kept.
- **Sg and Kn:** only 111 and 89 misses. No pair covers ≥ 5 words; the residue is single-word lexical variation.

**Not done:** nothing is left undone.

## Round 7

Method as in Round 6 (`residuals.py`, `gates.py`), with the round-7 gates: ext train better, ext test not
worse, verb within 0.001, ext_acc and ext_tone not worse, no size threshold. Scratch tools are in `tmp/sound_changes/hindi/`,
and reports and snapshots in `tmp/sound_changes/hindi/r7b/`:
- `g8.py` (gates.py verdict for BASE vs CAND, baseline scored once) and `g9.py` (candidates as line edits);
- `vq.py` (lax-vowel substitutions per source, and a per-reflex dump), `syn.py` / `syn2.py` (syncope by skeleton),
  `nm.py` (near-miss edit pairs), `fire.py` (how often each rule fires and how the words it touches score);
- `log.md` is the running decision log.

All numbers use the current engine: separate vowel qualities, accent marks dropped from plain/ext, verb flags
widened, and the tag parser reading only the tags before the em dash.

### Before → after (pre-round-7 snapshot → installed files, same engine)

| lect | ext train | ext test | verb train | verb test | plain train | ext-exact train | better / worse | gates.py |
|---|---|---|---|---|---|---|---|---|
| H | 0.2290 → 0.2248 | 0.2331 → 0.2283 | 0.1676 → 0.1660 | 0.1651 → 0.1641 | 0.3544 → 0.3691 | 34.7 → 35.5% | 333/141 | IF-JUST |
| Aw | 0.2534 → 0.2494 | 0.2353 → 0.2306 | 0.1582 → 0.1582 | 0.1692 → 0.1692 | 0.3506 → 0.3793 | 33.8 → 35.2% | 24/4 | IF-JUST |
| OAw | 0.1740 → 0.1706 | 0.1883 → 0.1870 | 0.1325 → 0.1278 | 0.1518 → 0.1518 | 0.2420 → 0.2598 | 43.6 → 45.2% | 23/10 | IF-JUST |
| Brj | 0.2975 → 0.2956 | 0.3188 → 0.3174 | 0.3476 → 0.3476 | 0.3607 → 0.3512 | 0.3897 → 0.4027 | 22.5 → 22.8% | 5/0 | IF-JUST |
| bundeli_atarra | 0.4022 → 0.3918 | 0.4143 → 0.4016 | 0.5007 → 0.4856 | 0.6997 → 0.6997 | 0.5111 → 0.5094 | 9.3 → 9.4% | 105/32 | ACCEPT |
| bagheli_lakshman | 0.3569 → 0.3516 | 0.3632 → 0.3591 | 0.3441 → 0.3361 | 0.4750 → 0.4517 | 0.4692 → 0.4698 | 13.4 → 13.9% | 45/20 | ACCEPT |
| kaithal (Hr) | 0.3479 → 0.3425 | 0.3401 → 0.3401 | 0.4101 → 0.4101 | 0.5193 → 0.5193 | 0.4633 → 0.4567 | 19.5 → 20.4% | 15/6 | ACCEPT |
| NDu (Ch) | 0.3639 → 0.3567 | 0.3539 → 0.3519 | 0.4420 → 0.4021 | 0.4952 → 0.4762 | 0.4931 → 0.4916 | 14.0 → 14.6% | 33/16 | ACCEPT |
| SSr (Sg) | 0.3485 → 0.3375 | 0.3400 → 0.3400 | 0.4667 → 0.4367 | 0.6000 → 0.6000 | 0.4713 → 0.4610 | 15.9 → 18.1% | 1/0 | ACCEPT |
| Kannauji-Central | 0.3196 → 0.3196 | 0.3024 → 0.3024 | 0.3597 → 0.3597 | 0.6667 → 0.6667 | 0.3961 → 0.3961 | 25.2 → 25.2% | 0/0 | no change (only a no-op @morph removal) |

`IF-JUST` = ACCEPT-IF-JUSTIFIED. Each one comes only from removing legacy `@morph` citation rules. That plain
loss is the bare-vs-extended artifact: the bare headword no longer gets a pasted -nā / -ab / -ai / -no, and the verb
menu supplies it. ext_acc and ext_tone are unchanged everywhere.

### Step 0: the interrupted round-7 edits

The previous agent left five edits. Each was re-gated alone against the snapshot:

| lect | edit | train / test ext | words better / worse | kept? |
|---|---|---|---|---|
| H | CL also in OIA trisyllables with a light 2nd syllable and a final short vowel (`# C* _ Short C {a i u} #`: kajjala > kājal, úttara > ūtar; not uttāra > utār) | −0.0008 / −0.0003 | 74 / 46 | yes |
| Aw | same CL extension | −0.0018 / −0.0037 | 9 / 4 | yes |
| OAw | same CL extension, with `use degem-lengthen.vg.cvv` written out locally | −0.0018 / ±0 (verb −0.0009) | 11 / 4 | yes |
| Aw | `{ā ī ū} > {a i u} / _ C i y ā #` → `use pretonic-shortening.aiu.ciya` | identical scores | – | yes (unification) |
| Brj | `use hiatus.oeai.preconsonantal` → `use hiatus.oeoeai.preconsonantal` | identical scores | – | yes (unification) |

The same CL extension was then tried in every other lect. It fails everywhere: Brj, Bu, Bg, Hr, Ch, Sg and Kn lose
+0.004 to +0.013 plain (Bg only reaches ACCEPT-IF-JUSTIFIED, −0.0001 ext against +0.0052 plain, so it is not kept).
The survey lists keep the short vowel (kajal, utar).

### Vowel qualities (focus 0)

Only the survey lists write lax vowels; CDIAL H./Aw./OAw./Brj. do not. The counts:
- chattisgarhi: ɛ 90 against e 70 (NDu); ɛ 22 against e 11 (SSr);
- bundeli: ɛ 126, e 103;
- kannauji: ɛ 24, ɔ 17;
- Koshy (Bg): ɔ 27, ɛ 8;
- Webster (Hr): ɛ 7.

Kept:
- **NDu `{a ā} {e ē i} > ɛ / _ #`** (@hiatus @vowel-shift). The E. Hindi 3sg -ai/-ae (< -ati/-atē after t-loss)
  monophthongises to -ɛ: máratē > marɛ, bʰajyátē > bʰājɛ. Train ext −0.0014, 4 better / 0 worse, test ±0. SSr: no effect.
- **SSr `{e ẽ} > {ɛ ɛ̃} / _ C`** (`@notation` in `%source chattisgarhi`). The Surgujia part of the list writes /e/ as ɛ
  before a consonant (kɛrā, cɛrī, dɛh, pɛṭ, lɛṭgā, bɛrʰ); e is kept only in ek, dekʰ- and yeh-. Train ext −0.0103,
  test ±0.
  - The same line in NDu is worse (+0.0010, verb +0.0113). Its Chhattisgarhi part writes e and ɛ about equally
    (kerā ~ kɛrā, cerī ~ cʰɛrī), so this is transcription variation, not a phoneme.

Rejected:
- Chain `{a ā} {i ī} > ɛ`, `{a ā} {u ū} > ɔ` and `a > ɛ / _ h` in Bu, Bg, Kn, Hr, Ch and Sg. All are worse on
  train or vetoed by test. Hr ai > ɛ improves ext_acc by 0.05 but test ext is +0.0062.
  - The lists write aī / ai as often as ɛ.
- Bundeli ɛ for e: +0.0046.
- Kannauji final o > ɔ: +0.0234.
- Koshy final o > ɔ passes (−0.0012), but it rests on two words and Koshy writes final o 10× against ɔ 8×.
  Transcription variation, not kept.
- The remaining ɛ / ɔ are endings: the oblique -ɛ, the 3sg -ɛ, the W. Hindi past -yɔ. Their doublets inside one
  list (ek ~ ɛk, bɔkerī ~ bekerī) are not regular.

### H vowel length and syncope

Syncope, from the per-skeleton table (`syn2.py`):
- The largest "syncope missing" class is extra medial vowels, VC_CV# (150 train pairs). Its biggest members are
  short **nasal** vowels, which the `{a i}` syncope did not include (31 + 18 pairs).
- Kept: **`{a i ã ĩ} > ∅ / V C _ C V // _ C a C V`** (kaḍambá + -ikā > kaṛbī, palyaŋka + -aka > palkā,
  daśamá > dasvā̃). Train ext −0.0011, test −0.0008, ext_acc −0.0067; 116 better / 9 worse. Also adopted in:
  - Aw (`{a ã} > ∅ / V C _ C Long`): −0.0005, 7 / 0;
  - Brj (`{a ã} > ∅ / V C _ C V`): −0.0001, test −0.0014.
  - Hr: no effect.
- Rejected (u syncope): `{a i u}` (69 / 57, test +0.0002) and `u > ∅ / V C _ C ī #` (16 / 14).

Length, from `vlen.py`:
- Kept: **anusvāra lengthening in the same trisyllables as the round-7 CL.** The no-lengthening rule gets the
  exception `# C* _ C Short C {a i u} #` (ántara > ā̃tar, *daŋgara > dā̃gar). Train −0.0003, test −0.0006; 21 / 10.
- Kept: **pretonic `ō > u / # C* _ C Long C V`.** A first-syllable ō, mostly from ava- / apa- / upa-, shortens to u
  before a stressed long second syllable: apasāra > usārā, avasārayati > usārnā. This matches the ā ī ū > a i u
  pretonic shortening. Train −0.0002, verb −0.0009, test −0.0003; 22 / 9.
  - ē does not do this (e 16× vs i 9× in that cell), so ē is not included.
- Kept: **`{ā ī ū} > {a i u} / # C* _ C Long C #`**. A first-syllable long vowel shortens before a stressed final
  CV̄C syllable (the NIA oxytone disyllable): āhāra > ahār, ābʰīra > ahīr, *kācahāra > kahār. Train −0.0003,
  test −0.0005; 17 / 5. Also in Aw (−0.0006, test −0.0010; 2 / 0). No effect or worse in Brj, Bu and Hr.
  - Versions restricted to h (`ā > a / # C* _ h Long …`) do worse.
- Rejected:
  - CL before -aka in quadrisyllables (`# C* _ Short C a a #`: 45 / 68);
  - `ū > u / _ {ā a}` (15 / 11, test worse);
  - ṁ before s kept nasal in the first syllable (14 / 15). The oral outcomes are mostly numerals (-bīs, -tīs).

Hiatus and final v (all kept):
- **`a ū > a u`**: H. writes au 325× against aū 16× (*hastakūṭa > hatʰauṛā, *padūna > paunā). Train −0.0003,
  test −0.0002; 13 / 5. Brj: −0.0010.
- **`i ī > ī`** (aheṛiī > aheṛī, kapāsiī > kapāsī): 15 / 1, −0.0003 / −0.0001. Also Aw (−0.0011, 6 / 0),
  Brj (−0.0004) and NDu (−0.0002).
- **`use v-vocalise`**: final -v > -u, as in Aw and Bg (*gāva > gāu, jīvá > jīu; H. -Vu 63× vs -Vv 29×).
  −0.0004 / −0.0001; 24 / 17. Round 6 rejected this on plain NED alone. No effect or worse in Bu, Hr, Ch, Sg, OAw.

Rejected:
- `i i > ī`, `u u > ū` (test vetoes);
- v loss after a consonant (`v > ∅ / C _ ā`: 44 / 51; koṭvāl-type forms keep it);
- removing mr > mb (test better, but train +0.0001).

### Survey-lect verbs

- **-ita + -sya > -iasa > -īs** (E. Hindi 3sg past). The participle vowel and the enclitic's a stayed in hiatus
  (predicted jarīās, marīes). The fix is `{i ī} {a ā} > ī / _ C #` (@hiatus).
  - NDu: −0.0019, verb −0.0400; 7 / 0.
  - SSr: −0.0010, verb −0.0333.
  - Bg: jointly with the -na rule below, −0.0011, verb −0.0106 / test −0.0333.
  - Bu: −0.0008, test −0.0015. These are noun -ias forms; Bu has no -is past.
  - Hr: −0.0015. Brj: −0.0004. H: −0.0002, 23 / 20, kept for consistency.
  - Kn: no effect. Aw already drops the i (`i > ∅ / _ a C #`).
- **Bagheli / Bundeli -na infinitive**: `a > ∅ / V C _ n ā #` (cárvati > cebna, cálati > celna). Installed in Bg with
  the -īs rule, and in Bu (−0.00003, 2 / 1).
  - In NDu and SSr it fails the verb gate (+0.04): the Chhattisgarhi list keeps -anā (ūṛanā).
- **OAw glides** (the 3sg -ai verbs). Kept jointly: `∅ > v / {ā u o} _ a i #` (cʰādáyati > cʰāvai,
  *cyutati > cuvai, dyṓtatē > jovai) and `∅ > y / ā _ a` (pā́da > pāya, rā́jan > rāya, kātara > kāyara).
  Train −0.0016, verb −0.0044, test −0.0013; 12 / 6.
  - The v glide showed zero effect in one gate run. The cause was the text "@morph" in its comment, which was then
    parsed as a tag. The coordinator has since fixed the parser.
- **ya-śruti in the survey lists.** Every list writes -iyā (Bundeli 87× vs -iā 1×).
  - `use ya-sruti.iiaa` kept in NDu (−0.0021 / −0.0011) and Bg (−0.0042 / −0.0031).
  - Bu uses `∅ > y / {i ī} _ ā` (−0.0093 / −0.0112; 95 / 30). The full version fails the verb gate (+0.0012).
  - Rejected in SSr, Kn, Hr (test +0.0037) and Brj.
- **Bu `%verb-menu +itaka`**: it and its placebo +iraka have no effect, so it is not added.

### Legacy @morph rules (focus 6)

Each rule was tested leave-one-out, then jointly. The H, Aw and OAw runs predate the widened verb flags (10:08). The
blocked removals were retried after both flag fixes.

| lect | removed | kept (removal fails) |
|---|---|---|
| H | starred-root marker ∅ > Ø + `Ø > n ā`; if:verb -ayati > -nā; if:verb ā-root > -ānā. Jointly ext −0.0004 / −0.0019, plain +0.0136 (IF-JUST) | in-stem `i n > ī` (ext +0.0007); `a {i ē} > n ā` (ext +0.0005; +0.0002 after the flag fixes) |
| Aw | if:verb -ab ×3, `Ø > a b` (neutral; plain +0.0299 jointly) | in-stem (ext +0.0018) |
| OAw | in-stem, `ē > i / a _ #`, -ayati > -ai, `Ø > a i` (neutral; plain +0.0184) | – |
| Brj | if:verb -no ×3 (neutral; plain +0.0142) | – |
| Bu | -ayati bare stem (neutral) | -ati bare stem (ext +0.0007, also after the flag fixes) |
| Hr | -ayati bare stem (neutral) | -ati bare stem (ext +0.0004), starred-root pair (ext +0.0006 → +0.0007 after the flag fixes) |
| NDu | in-stem (neutral, plain +0.0002) | – |
| Kn | `Ø > o` (never fires) | – |

Mewati (one @morph rule) belongs to rajasthani-gujarati, so it was not touched.

### Other consistency and near-miss work

- **Hr `{i u} > a / V C _ C`.** Unstressed medial high vowels centralise (*lakkuṭa > lakaḍ, vidyullatā > bijali,
  láśuna > lasaṇ).
  - u alone: −0.0037, test ±0; 12 / 4. Adding i: a further −0.0002.
  - Worse in every neighbour (Ch, Sg, Bu, Bg, Kn, Brj), and i > a in Bu fails test (+0.0115). Kept in Hr only.
- **OAw `%verb-menu +āpayati`** (3sg causative -āvai / -āi) beats its placebo +ārayati by only 0.0016 verb NED on
  train. That is below the 0.002 margin, so it is not added.
- **Near-miss reading** (NED ≤ 0.34) for H, OAw, Aw, Brj, Bu, Ch, Sg, Kn and Hr. The residue is mostly lexical:
  - regional doublets: l / r, ṛ / r, ṁ / ∅;
  - the menu's -allaka picked over the attested -ā;
  - numerals;
  - the Bundeli -o / -ā mixture;
  - Chhattisgarhi -tʰɛ / -gɛ verb forms;
  - initial cʰ ~ c inside one list.
- **kṣ > cʰ vs kʰ** is lexical, not positional:
  - H: initial kʰ 50 / cʰ 17, medial kʰ 129 / cʰ 21;
  - OAw: initial cʰ 4 / kʰ 2, medial kʰ 22 / cʰ 4.
- **d > ḍ before a later retroflex** (Aw 5 / 1) is H 29 ḍ vs 46 d, so it was not tried.
- **Rule-firing audit** (`fire.py`): in H, the rules with fewest hits and lowest exact rates are mr > mb,
  uva > o, ṁ + s and the new -īs contraction. Only the first two were re-tested (above).

### Not done

- Haryanvi tone: gill's acute / grave on 8 forms, e.g. bā́ < bāhú, bɛ̀ṇoī < bʰaginīpati with low tone after a lost
  initial aspirate. Too few forms to fit a tone rule.
- LOO pruning of rarely firing rules in the survey lects. Their exact rates are uniformly low (Bu 9%), so this
  would be list noise.
- Kannauji: no regular pattern ≥ 3 words. Its only change is a neutral @morph removal.
