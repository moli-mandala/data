# Andersen (1990): printed variants, stone pillar, and gold mountain

Eight exact source-defined records, ten proposed sidecar rows, under standing autonomous authorization. Two source-labelled compounds receive two ordered component edges each; their source heads already have accepted ancestry. Three unambiguous printed “see” references become source variants.

| # | Ashokan Prakrit | Proposed etymology | Evidence |
|---|---|---|---|
| 1 | **dāni** ‘now’ | **idā́nīm, CDIAL 1567 §1** (reflex) | Andersen compares RV *idā́nīm*, Pali *idāni/dāni*, and explains lost initial vowel. Turner explicitly lists Ashokan Rupnath *dāni*. |
| 2 | **dāṇi** ‘’ | **source variant of dāni** | Printed “see dāni”; Turner’s article also lists Prakrit *dāṇi*. |
| 3 | **vacana** ‘word’ | **vacaná, CDIAL 11199** (reflex) | Andersen compares RV *vacaná* and Pali *vacana*; Turner explicitly lists Ashokan *vacana* ‘order’. |
| 4 | **vaḍʰi** ‘increase, growth, welfare, profit’ | **vṛ́ddhi, CDIAL 12076** (reflex) | Andersen compares RV *vṛ́ddhi* and Pali *vaḍḍhi*; Turner explicitly lists Ashokan *vaḍhi/vadhi/vuḍhi* ‘growth’. |
| 5 | **silāṭʰambʰa** ‘stone pillar’ | **ordered components silā + tʰambʰa** | Andersen compares Sanskrit *stambha*, Pali *silātthambha*, and cross-refers to both source heads. Their accepted ancestors are CDIAL 12459 and 13682, each with Ashokan attestations. |
| 6 | **silātʰambʰa** ‘’ | **source variant of silāṭʰambʰa** | Printed dental spelling says “see silāṭhambha”. |
| 7 | **suvaṁṇagiri** ‘gold mountain’ | **ordered components suvaṁṇa + giri** | Andersen compares Sanskrit *suvarṇagiri* and refers to *suvaṁṇa* and *giri*; their accepted ancestors are CDIAL 13519 and 4161. |
| 8 | **suaga** ‘’ | **source variant of svaga** | Printed “see svaga”; its head also lists the Sahasram *suag[e]* form, and Turner 13910 explicitly has Ashokan *svaga*. |

These proposals create **ten assignment rows across eight records**: three reflexes, three variants, and four ordered component edges across two compounds. Three cases are held: **sātireka**, pending lexical segmentation against related adjectives; **savapaṭʰavi**, whose first lexical component has no separate source head; **asokarāja**, pending an etymology for the Aśoka name. None requires changing source parsing or transcription.

Primary evidence: Andersen scan PDF pp. 151, 166, 173–174 (printed pp. 153, 168, 175–176). Full Turner articles in `andersen-primary018.json`, extracted from the installed CDIAL source snapshot, with their original page URLs recorded. Exact source IDs, accepted component ancestors, and unlinked target status were checked against `andersen-records.json`, compiled edges, and the owning sidecar. Before this pass, 145 of Andersen’s 279 source records lacked effective accepted rank-1 ancestry after saved sidecars and compiled edges were combined.

Validation: `make check-pass`, `make save-pass`, and `make check-sources` passed. The ten saved rows match exactly the ten rows in the owning Andersen sidecar; both component records have positions 1 and 2. The second application made zero changes, protected forms/edges/identity hashes stayed unchanged, and only the Andersen sidecar changed. The final source check counted 76,920 assignment rows in 90 sidecars. No rebuild.
