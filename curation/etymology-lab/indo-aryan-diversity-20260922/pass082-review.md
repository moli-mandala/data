# Andersen (1990): Ashokan glossary heads

Eleven source-defined records prepared under standing autonomous authorization. These have been checked against the printed Andersen pages and complete Turner articles. The printed “see” cross-references point to Andersen heads; *garu* and *guna* are already linked in the compiled graph, while *māṇusa* is assigned in this pass.

| # | Ashokan Prakrit | Proposed etymology | Evidence |
|---|---|---|---|
| 1 | **guru** ‘’ | **source variant of garu** (variant) | The printed *guru* line says “see garu”; the *garu* head is already linked to CDIAL 4209. |
| 2 | **guṇa** ‘’ | **source variant of guna** (variant) | The printed *guṇa* line says “see guna”; the *guna* ‘virtue’ head is already linked to section 2 of CDIAL 4190. |
| 3 | **tiya** ‘third’ | **tr̥tī́ya, CDIAL 5912**, section 1 (reflex) | Andersen compares RV *tr̥tī́ya* and Pali *tatiya*; Turner includes Prakrit *tiiya* in the ordinal branch. |
| 4 | **di** ‘two’ | **dva, CDIAL 6648** (reflex) | Andersen compares RV *dvi* and Pali *dvi/di*, with inscriptional *duve*; Turner explicitly cites Ashokan *duve* under this numeral. |
| 5 | **budʰa** ‘Buddha’ | **buddha, CDIAL 9276**, section 1 (reflex) | Andersen’s printed *budha* compares RV *buddhá* and Pali *buddha*; Turner explicitly cites Ashokan *Budhe*. |
| 6 | **māṇusa** ‘man’ | **mā́nuṣa, CDIAL 10049** (reflex) | Andersen compares RV *mā́nuṣa* and Pali *mānusa*; Turner specifically has Ashokan *mānusa* and Prakrit *māṇusa*. The separate *manusa* is already linked to CDIAL 9828. |
| 7 | **mānusa** ‘man’ | **source variant of māṇusa** (variant) | Andersen’s separate printed *mānusa* line says “see māṇusa”; its target is proposal 6, ultimately CDIAL 10049. |
| 8 | **saca** ‘truth’ | **satyá, CDIAL 13112** (reflex) | Andersen compares RV *satyá* and Pali *sacca*; Turner explicitly gives Ashokan *saca* under this truth noun. |
| 9 | **mahalaka** ‘noble, old’ | **mahallaka, CDIAL 9935** (reflex) | Andersen compares Pali *mahallaka*; Turner explicitly gives Ashokan *mahalaka/mahālaka* ‘great, old’. |
| 10 | **bamhana** ‘’ | **brāhmaṇá, CDIAL 9327**, section 1 (reflex) | The printed head means ‘Brahman’ and compares RV *brāhmaṇá*; Turner lists Ashokan *bamaṇa/baṁbhana/bāmhaṇa*. The compiled source gloss is blank and has not been repaired. |
| 11 | **raṭʰika** ‘’ | **rāṣṭrika, CDIAL 10724** (reflex) | Andersen’s printed head means ‘official of the kingdom’ and compares Sanskrit *rāṣṭrika* and Pali *raṭṭhika*; Turner 10724 lists the same Pali form and ‘governor’. |

Eleven proposed assignment rows affect eleven records: eight reflexes and three source variants. Four unlinked forms were held: **maha** ‘large, great’ because Turner 9946 does not explicitly document Andersen’s independent short head; **tatʰā** ‘thus’ because Turner lacks the independent head and 5647 is the compound *tathāpi*; **likʰ** ‘write’ because the source gives a broad root while Turner 11048 is a particular *likháti* formation; **rajuka** because its installed p. 164 record is only a “see rājuka” line, while the actual rājuka head starts on p. 165 and has no distinct installed record. A related pre-existing compiled edge places **munisa** ‘man’ directly under 9828, although Andersen calls that form a contamination of *purisa* and *manuṣyà*. This pass leaves that existing edge unchanged; correcting it would require a separate graph review.

Primary evidence: Andersen (1990) scan PDF pp. 146, 150–151, 159–162, 164–165, 170 (printed pp. 148, 152–153, 161–164, 166–167, 172); complete Turner articles in `andersen-primary008.json` through `andersen-primary011.json`; candidate discovery by the homepage search implementation in `andersen-search009.json` and `andersen-search010.json`. Source-defined compiled records were checked against `andersen-records.json` and the existing Andersen sidecar.

Validation: shared dry-run and save checks passed. All eleven rows exactly match the owning Andersen sidecar, and `make check-sources` passed (76,833 accepted rows in 89 sidecars). The helper's second application changed zero rows; before/after and current SHA-256 values match for `cldf/forms.csv`, `cldf/edges.csv`, and `data/form-identities.csv`. The Andersen sidecar was the sole changed sidecar. No rebuild was run.
