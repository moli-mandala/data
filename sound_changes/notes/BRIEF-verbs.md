# Brief, round 5: derive verb citation forms by regular sound change

The constraints are the same as before:
- write only your lects' rule files, your notes file and `tmp/sound_changes/<group>/`;
- do not edit the engine or `changes.sc`; the coordinator re-centralises afterwards;
- run one Python process at a time;
- tune on **train** only;
- never use a broad `pkill`;
- **do not run `frontier`**: it is retired, and summary scores are computed directly.

## The setup
The extension-aware score now has a **verb-form menu** (`evaluate.VERB_MENU`). Each verb headword is
reduced to its root (the headword minus -ati/-atē/-ayati/-ōti/-ti). The chain is then run on these forms:
- the bare root;
- the present `-ati`;
- the verbal nouns `-anaka`, **neuter `-anakam`**, `-aṇa` and `-anīya`;
- the gerundive `-tavya`, **neuter `-tavyakam`**;
- the participle `-ita`;
- the infinitive `-itum`.

For verbs **every `@morph` rule is switched off**: the bare headword and all the verb forms run
without them. So a lect's citation form (-nā, -ṇẽ, -vũ, -ab, -ibā, -ṇu, -un …) has to come from
the right formation **through regular sound changes**, not from a rule like `a {i ē} > n ā / _ #`.
The plain score keeps using your citation rules, so leave them in place.

`python3 sound_changes/evaluate.py score RULES --verbs` prints, for verbs only, the plain NED (with
your citation rules), the verb-menu NED (without them) and which forms win.

## Where each lect stands (test verbs; lower is better)
| group | lect | test verb pairs | plain | verb menu | most chosen |
|---|---|---|---|---|---|
| bihari-tharu | Maithili | 84 | 0.233 | 0.359 | ati:62 ita:9 anakam:5 |
| bihari-tharu | Bhojpuri | 39 | 0.229 | 0.351 | ati:29 aṇa:4 bare:2 |
| bihari-tharu | Bihari | 23 | 0.226 | 0.368 | ati:18 anaka:3 ita:2 |
| bihari-tharu | Chitwania Tharu | 11 | 0.342 | 0.244 | root:4 ati:2 ita:2 |
| bihari-tharu | Morangia Tharu | 9 | 0.341 | 0.218 | root:4 bare:2 ati:2 |
| central-east-pahari-romani | Nepali | 251 | 0.302 | 0.350 | anaka:160 itum:38 anīya:29 |
| central-east-pahari-romani | Kumaoni | 164 | 0.283 | 0.326 | anaka:143 bare:6 anīya:4 |
| central-east-pahari-romani | European Romani | 45 | 0.419 | 0.437 | ati:23 root:12 bare:7 |
| central-east-pahari-romani | Garhwali | 41 | 0.199 | 0.256 | anaka:35 anīya:2 tavyakam:1 |
| central-east-pahari-romani | Welsh Romani | 12 | 0.426 | 0.393 | root:4 bare:4 ati:3 |
| central-east-pahari-romani | Lomavren | 9 | 0.506 | 0.467 | ati:3 root:3 bare:2 |
| dardic-north | Kashmiri | 120 | 0.258 | 0.214 | anaka:100 ati:11 anakam:3 |
| dardic-north | Khowar | 112 | 0.387 | 0.476 | ita:52 bare:14 aṇa:12 |
| dardic-north | Palula | 76 | 0.484 | 0.424 | ati:30 anīya:12 aṇa:10 |
| dardic-north | Shina | 68 | 0.429 | 0.517 | ati:30 anaka:22 root:6 |
| dardic-north | Indo-Aryan Kalasha | 45 | 0.446 | 0.494 | ati:15 root:11 anaka:8 |
| dardic-south | Pashai | 63 | 0.442 | 0.399 | root:25 ati:16 bare:11 |
| dardic-south | Indus Kohistani | 44 | 0.478 | 0.486 | ati:19 anaka:9 bare:6 |
| dardic-south | Dameli | 34 | 0.433 | 0.375 | root:14 ati:9 aṇa:5 |
| dardic-south | Bashkarik | 11 | 0.345 | 0.315 | ati:7 anaka:2 bare:2 |
| dardic-south | Gowro | 10 | 0.335 | 0.409 | bare:3 ati:3 anaka:2 |
| dardic-south | Gawarbati | 9 | 0.466 | 0.384 | ati:3 bare:2 itum:1 |
| dardic-south | Wotapuri | 9 | 0.496 | 0.430 | ati:7 bare:1 root:1 |
| eastern | Odia | 232 | 0.262 | 0.421 | anaka:159 tavyakam:50 bare:10 |
| eastern | Bengali | 162 | 0.299 | 0.323 | anaka:131 bare:14 ati:6 |
| eastern | Assamese | 137 | 0.290 | 0.432 | ita:64 ati:37 anaka:17 |
| eastern | Kamtapuri | 13 | 0.532 | 0.372 | root:12 ati:1 |
| eastern | Hajong | 8 | 0.306 | 0.271 | anaka:4 ati:2 anīya:1 |
| hindi | Hindi-Urdu | 363 | 0.257 | 0.222 | anaka:325 bare:13 ati:10 |
| hindi | Old Awadhi | 38 | 0.172 | 0.161 | ati:34 bare:2 ita:1 |
| hindi | Awadhi | 24 | 0.186 | 0.277 | ati:13 tavya:9 aṇa:1 |
| hindi | Braj | 11 | 0.439 | 0.329 | anaka:9 ati:2 |
| marathi-bhil | Marathi | 369 | 0.293 | 0.366 | anaka:286 aṇa:29 ati:23 |
| marathi-bhil | Old Marathi | 87 | 0.401 | 0.422 | aṇa:56 bare:13 anaka:6 |
| marathi-bhil | Konkani | 26 | 0.335 | 0.445 | tavya:7 anaka:6 anīya:5 |
| mia-insular | Prakrit | 334 | 0.233 | 0.182 | ati:222 bare:62 ita:20 |
| mia-insular | Pali | 174 | 0.110 | 0.099 | ati:112 bare:55 ita:3 |
| mia-insular | Sinhala | 151 | 0.263 | 0.374 | anaka:90 tavyakam:44 ati:7 |
| mia-insular | Dhivehi | 51 | 0.389 | 0.469 | anaka:39 ati:4 aṇa:3 |
| mia-insular | Ashokan Prakrit | 35 | 0.460 | 0.388 | ati:14 bare:9 ita:3 |
| mia-insular | Niya Prakrit | 32 | 0.327 | 0.239 | ati:17 ita:5 bare:3 |
| mia-insular | Old Sinhala | 14 | 0.542 | 0.463 | ati:5 anaka:4 tavya:2 |
| mia-insular | Gandhari | 12 | 0.180 | 0.149 | ati:6 bare:3 tavya:1 |
| northwest | Punjabi | 276 | 0.310 | 0.317 | anaka:251 bare:11 ati:10 |
| northwest | Sindhi | 213 | 0.271 | 0.332 | anaka:159 itum:28 ati:9 |
| northwest | Lahnda | 149 | 0.282 | 0.246 | aṇa:133 bare:5 ati:4 |
| northwest | Kutchi | 40 | 0.306 | 0.383 | anaka:33 ati:3 anīya:2 |
| northwest | Hindko | 39 | 0.344 | 0.389 | aṇa:33 anaka:3 root:1 |
| northwest | Old Punjabi | 10 | 0.426 | 0.371 | ati:5 anaka:3 root:2 |
| nuristani | Prasun | 37 | 0.572 | 0.518 | ati:18 root:5 bare:4 |
| nuristani | Nuristani Kalasha | 29 | 0.552 | 0.447 | ati:13 anaka:9 tavyakam:3 |
| nuristani | Katë | 20 | 0.597 | 0.540 | ati:8 bare:5 aṇa:2 |
| nuristani | Ashkun | 17 | 0.497 | 0.412 | ati:8 ita:3 anakam:2 |
| nuristani | Proto-Nuristani | 12 | 0.401 | 0.324 | ati:6 bare:2 anaka:2 |
| rajasthani-gujarati | Gujarati | 264 | 0.271 | 0.320 | anakam:142 tavyakam:94 ati:9 |
| rajasthani-gujarati | Old Gujarati | 26 | 0.179 | 0.176 | ati:23 bare:2 anīya:1 |
| rajasthani-gujarati | Old Marwari | 25 | 0.223 | 0.181 | ati:22 bare:2 ita:1 |
| west-pahari | Kotgarhi | 87 | 0.362 | 0.474 | anaka:69 ita:7 bare:7 |
| west-pahari | Kiunthali | 31 | 0.244 | 0.388 | anaka:12 itum:10 ati:5 |
| west-pahari | West Pahari (unspecified) | 29 | 0.467 | 0.387 | anaka:26 itum:1 bare:1 |
| west-pahari | Bhalesi | 25 | 0.347 | 0.444 | anaka:18 aṇa:2 ita:2 |
| west-pahari | Khashali | 17 | 0.558 | 0.513 | anaka:8 itum:4 aṇa:2 |
| west-pahari | Bhadrawahi | 16 | 0.516 | 0.492 | itum:5 aṇa:3 anaka:3 |
| west-pahari | Kochi | 14 | 0.404 | 0.410 | anaka:13 anīya:1 |
| west-pahari | Jaunsari | 9 | 0.588 | 0.575 | anaka:6 root:1 aṇa:1 |
| west-pahari | Churahi | 8 | 0.393 | 0.374 | anaka:7 itum:1 |
| west-pahari | Chambeali | 8 | 0.287 | 0.279 | anaka:8 |

Where the verb-menu score is worse than plain, the chain cannot yet turn the right formation into
the citation ending. Examples:
- Marathi's neuter rule `if:ext a a {m ṁ} > ẽ / _ #` is tagged `@morph`, so it is off for verbs and
  -anakam gives lihṇā, not lihṇẽ;
- Kotgarhi has no neuter -am rule, so liṇo comes out instead of liṇõ;
- Bihari needs -tavya > -ab;
- Odia needs -tavya(kam) > -ibā.

## What to do (until 18:27 — hard deadline)
1. For each lect, largest verb count first, identify the attested citation form(s) and the OIA
   formation they continue: infinitive or verbal noun from -anaka/-anakam/-aṇa/-anīya, gerundive
   -tavya(kam), participle -ita, or the present.
2. Add or adjust **sound changes** so the chain derives them.
   - Rules must be conditioned only by segments: no `if:verb`, no `if:ext`, no `if:numeral`, no
     `@morph`.
   - A contraction such as -aaṁ > -ẽ or -aa > -ā is a sound change: tag it as one (`@hiatus`,
     `@final-nasal` …).
   - If an existing rule already does this but carries `@morph` or `if:ext`, fix its tag or
     condition.
3. Accept a change only if **all four** hold:
   - verb-menu train NED improves;
   - plain train NED does not get worse by more than 0.001;
   - the noun extension-aware score (`--ext`) does not get worse;
   - the accent score is unchanged or better.

## Reporting
Stop editing at **18:27**. Add **"§11 Verb citation forms"** to your notes. For each lect give:
- test verbs, plain vs verb-menu NED, before and after;
- the formation behind each citation form;
- the rules you added or retagged.

Be done by **18:30**. Your final message should be at most 120 words.
