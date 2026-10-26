# Andersen (1990): Ashokan spelling variants and lexical heads

Eleven exact source-defined records, twelve proposed sidecar rows, under standing autonomous authorization. Three variant targets are already linked in the compiled graph (*ti*, *svaga*, *pāna*), two in saved Andersen overlays (*saca*, *mahalaka*), and *pi* and *kʰudaka* receive new reflex links in this pass.

| # | Ashokan Prakrit | Proposed etymology | Evidence |
|---|---|---|---|
| 1 | **tī** ‘’ | **source variant of ti**, ultimately **íti, CDIAL 1562** (variant) | The printed *tī* line says “see ti”; Turner explicitly includes Ashokan *ti* under the discourse particle *íti*. |
| 2 | **śaca** ‘’ | **source variant of saca**, ultimately **satyá, CDIAL 13112** (variant) | The printed *śaca* line says “see saca”; the truth noun is already linked, and Turner explicitly cites Ashokan *saca*. |
| 3 | **śvaga** ‘’ | **source variant of svaga**, ultimately **svargá, CDIAL 13910** (variant) | The printed *śvaga* line says “see svaga”; Turner explicitly cites Ashokan *svaga* ‘heaven’. |
| 4 | **prāṇa** ‘’ | **source variant of pāna**, ultimately **prāṇá, CDIAL 8928** (variant) | The printed *prāṇa* line says “see pāna”; Turner explicitly distinguishes Ashokan *praṇa/prāṇa* from kāl. *pāna* within one article. |
| 5 | **prāna** ‘’ | **source variant of pāna**, ultimately **prāṇá, CDIAL 8928** (variant) | The printed *prāna* line also says “see pāna”; this is a second source-defined spelling, not a new breath/life etymon. |
| 6 | **pi** ‘also’ | **ápi, CDIAL 485** (reflex) | Andersen compares RV *ápi* and Pali *pi*; Turner 485 explicitly cites Ashokan *pi* with the same additive sense. |
| 7 | **pī** ‘’ | **source variant of pi**, ultimately **ápi, CDIAL 485** (variant) | The printed *pī* line says “see pi”; its target is proposal 6. |
| 8 | **velā** ‘limit of time’ | **vḗlā, CDIAL 12115**, section 1 (reflex) | Andersen compares ŚB *vḗlā* and Pali *velā*; Turner’s first section has the limit/time sense. Turner notes possible ancient Dravidian input, without identifying an immediate Ashokan donor. |
| 9 | **salila** ‘body, relics’ | **śárīra, CDIAL 12335** (reflex) | The printed entry compares RV *śárīra* and Pali *sarīra* with the body/relic sense. Turner’s article gives the same noun in Pali and Prakrit; Andersen’s comparison supports the unusual Ashokan *-l-l-* form. |
| 10 | **kʰudaka** ‘lowly person’ | **kṣudrá, CDIAL 3712** (reflex) | Andersen compares Sanskrit *kṣudraka* and Pali *khuddaka*. Turner explicitly lists Ashokan *khuda/khudaka* under this small/lowly branch. |
| 11 | **kʰudakamahalaka** ‘lowly and great person’ | **ordered components kʰudaka + mahalaka** (two component rows) | Andersen compares Sanskrit *kṣudraka* plus *mahallaka*, gives Pali *khuddaka-mallaka*, and cross-refers to both source heads. The first is proposal 10; the second is linked to CDIAL 9935 in pass082. |

The eleven proposals create **twelve assignment rows across eleven records**: four reflexes, six variants, and two ordered component rows for one composite record. Four cases are held: **kʰugdaka** because the printed cross-reference reads *khuḍaka* while the installed form has *g*; **hevaha** because “see hevam and ah-” indicates more than a spelling variant; **sa** because the installed record merges the printed numeral ‘six’ and pronoun ‘he/she/it/that’; **pipule** because its printed *vipule* cross-reference target lacks a distinct installed source head. These are research holds, not source repairs.

Previous Andersen pass082 saved eleven rows/records. Before this pass, 166 of the source’s 279 records had no effective accepted rank-1 ancestor after compiled edges and saved sidecars were combined.

Primary evidence: Andersen scan PDF pp. 145, 150, 157–158, 168–169, 172 (printed pp. 147, 152, 159–160, 170–171, 174), including the exact “see” lines. Full Turner articles in `andersen-primary012.json` through `andersen-primary014.json`; homepage search implementation output in `andersen-search011.json`. The source-defined records and target IDs were checked against `andersen-records.json`, compiled edges, and the owning sidecar.

Validation: `make check-pass`, `make save-pass`, and `make check-sources` passed; the source audit reported 76,866 accepted rows in 89 sidecars. All twelve saved rows exactly match the owning Andersen sidecar, including component positions 1 and 2. The helper's second application changed zero rows. Before/after and current SHA-256 values match for `cldf/forms.csv`, `cldf/edges.csv`, and `data/form-identities.csv`; Andersen was the sole changed sidecar. No rebuild was run.
