# Brief, round 3: hill-climb the accent score

Same constraints as before:
- write only your lects' rule files, frontier CSVs, your notes file and `tmp/sound_changes/<group>/`;
- do not edit engine files; ask the coordinator if you need an engine change;
- run one Python process at a time;
- tune on **train** only.
- **Never use a broad `pkill`.** Kill only processes you started, by PID. A `pkill -f "evaluate.py frontier"` in round 2 killed other groups' runs.

## The new score

**Accent NED** is scored only on attested forms whose source marks an accent: acute, grave,
circumflex, caron and contour tones all count as "this vowel carries the accent". The score is
the accent-blind edit distance plus 0.5 once when the accent is missing or misplaced
(revised 12:55; earlier a misplaced accent cost 1.0 and a missing one 0.5). `score` and `frontier` now print it next to the plain NED.

In the engine the accent lives on a separate **latent accent tier**:
- `V > $1-acc` (your existing accent-loss rule) no longer deletes the Vedic accent. It hides it,
  and the hidden accent moves with its vowel through later rules: onto the result of a merger
  or quality change, and lost when the vowel is deleted.
- `$1+stress` places an accent on the tier (`V > $1+stress / _ C* V C* #` = penult). `$1-stress`
  removes one.
- No rule can see the tier, so **placement rules never change the plain NED**. Verify this for
  every edit (train and test plain NED identical).
- When the accent score is computed, rule-placed accents win. With `%accent vedic` in the rule
  file (or `--accent-model vedic`), the carried Vedic accent shows wherever no rule placed one.
  Without it, only rule-placed accents show.
- Visible accent or tone marks that a rule or a `%source` block writes also count as predicted
  accents. Example: Shina's Schmidt block writes a circumflex on every long vowel. Check whether
  such notation now hurts the accent score, and report it.

## Tools
- `python3 sound_changes/evaluate.py score RULES --accent-model vedic` prints both scores.
- `python3 sound_changes/accent_hypotheses.py FILE` scores six baselines on a scratch copy of the
  chain: none, Vedic carried, old penult before apocope, last syllable, first syllable, first
  long vowel.

## Preliminary results (train accent NED; the coordinator's single-rule baselines)
Across all 22 lects the Vedic accent is never the best placement. The best is the last syllable
after apocope (10 lects) or the old penult placed before final-vowel loss (5). These are often
the same syllable, so they are one pattern.

| group | lect (file) | acc. pairs tr/te | none | Vedic | old penult | last σ | first σ | first long | best |
|---|---|---|---|---|---|---|---|---|---|
| dardic-north | Shina (Sh) | 1346/304 | 0.5096 | 0.5080 | 0.4808 | 0.4824 | 0.5078 | 0.4991 | penult |
| dardic-north | Khowar (Kho) | 720/146 | 0.4835 | 0.4846 | 0.4805 | 0.4629 | 0.5034 | 0.5034 | last syllable |
| dardic-north | Indo-Aryan Kalasha (Kal) | 601/153 | 0.5046 | 0.5079 | 0.4974 | 0.5100 | 0.5117 | 0.5117 | penult |
| dardic-north | Palula (Phal) | 1126/315 | 0.5258 | 0.5232 | 0.5130 | 0.5147 | 0.5233 | 0.5200 | penult |
| dardic-south | Indus Kohistani (Mai) | 1523/393 | 0.5429 | 0.5384 | 0.5308 | 0.5225 | 0.5443 | 0.5404 | last syllable |
| dardic-south | Pashai (Pas) | 649/180 | 0.5582 | 0.5588 | 0.5607 | 0.5530 | 0.5797 | 0.5666 | last syllable |
| dardic-south | Bashkarik (Bshk) | 433/122 | 0.5173 | 0.5048 | 0.4816 | 0.4744 | 0.4665 | 0.4665 | first syllable |
| dardic-south | Dameli (Dm) | 141/44 | 0.5423 | 0.5500 | 0.5494 | 0.5453 | 0.5599 | 0.5573 | none |
| dardic-south | Gawarbati (Gaw) | 174/50 | 0.5749 | 0.5781 | 0.5801 | 0.5765 | 0.6020 | 0.5903 | none |
| dardic-south | Gowro (Gowro) | 274/70 | 0.5282 | 0.5273 | 0.5253 | 0.5222 | 0.5369 | 0.5349 | last syllable |
| dardic-south | Bhateri (bhatr) | 208/54 | 0.5433 | 0.5359 | 0.5118 | 0.5093 | 0.5174 | 0.5167 | last syllable |
| dardic-south | Shumashti (Shum) | 75/31 | 0.5458 | 0.5473 | 0.5490 | 0.5480 | 0.5426 | 0.5328 | first long vowel |
| dardic-south | Wotapuri (Wot) | 63/21 | 0.4725 | 0.4728 | 0.4691 | 0.4726 | 0.5293 | 0.5222 | penult |
| nuristani | Nuristani Kalasha (Wg) | 260/68 | 0.5597 | 0.5651 | 0.5660 | 0.5547 | 0.5948 | 0.5889 | last syllable |
| nuristani | Kamviri (Kam) | 694/171 | 0.5270 | 0.5216 | 0.5238 | 0.4956 | 0.5370 | 0.5369 | last syllable |
| nuristani | Katë (Kt) | 634/154 | 0.5367 | 0.5316 | 0.5392 | 0.5062 | 0.5441 | 0.5441 | last syllable |
| nuristani | Ashkun (Ash) | 578/128 | 0.5706 | 0.5706 | 0.5710 | 0.5655 | 0.5847 | 0.5820 | last syllable |
| nuristani | Prasun (Pr) | 174/38 | 0.6217 | 0.6163 | 0.6253 | 0.6139 | 0.6351 | 0.6351 | last syllable |
| northwest | Hindko (awan) | 97/21 | 0.4852 | 0.4746 | 0.4266 | 0.4467 | 0.3938 | 0.3924 | first long vowel |
| west-pahari | Kotgarhi (ktg) | 804/191 | 0.5011 | 0.4970 | 0.4617 | 0.5357 | 0.4506 | 0.4506 | first syllable |
| west-pahari | Bhalesi (bhal) | 67/27 | 0.5097 | 0.5053 | 0.4656 | 0.4887 | 0.4614 | 0.4659 | first syllable |
| central-east-pahari-romani | Domari (as-ll) | 152/37 | 0.4339 | 0.4409 | 0.4022 | 0.4879 | 0.4025 | 0.4025 | penult |

Smaller accent-marked sets (optional): Sauji, Kalkoti, Torwali, Tirahi, Domaaki, Punjabi, Dogri, Khashali, Mandeali, Balkan/European Romani.

## What to do (until 13:45)
1. For each of your lects with accent-marked forms, largest first, start from the best baseline
   and hill-climb train accent NED. Candidates:
   - weight-sensitive placement (a heavy last syllable, else the penult);
   - the position relative to apocope, syncope or hiatus (try placing the rule at different
     points in the chain);
   - morphological classes: verbs or infinitives (`if:verb`), -aka nouns, monosyllables;
   - the Vedic accent for some classes and a positional rule elsewhere (`%accent vedic`, with
     placement rules only where they apply);
   - source conventions, e.g. whether a source marks tone only on long vowels.
2. Give the Vedic accent its best fair chance and report the numbers, either way.
3. Adopt only rules with a positive train LOO on accent NED that read as prosodic history. Tag
   them `@accent-placement`, adding `@stress` if they read the Vedic accent.

## Reporting
Stop editing at **13:45**, then rerun `frontier` for every lect you changed (it now records
`train_aned` and `test_aned`). Add a section **"§9 Accent score"** to your notes. It should give,
per lect:
- test accent NED for none, Vedic and your final chain;
- the adopted rules with their LOO values;
- every hypothesis you rejected, with its numbers;
- your conclusion on the Vedic accent.

Be done by **14:00**. Your final message should be at most 150 words.
