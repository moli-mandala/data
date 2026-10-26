# Andersen (1990): Ashokan compounds and demonstrative

Ten exact source-defined records, fifteen proposed sidecar rows, under standing autonomous authorization. Five of the records are transparent source-labelled compounds and receive two ordered component edges each. The attested lexical components already have accepted ancestry, except *aya* and *amisa*, which are established in this pass.

| # | Ashokan Prakrit | Proposed etymology | Evidence |
|---|---|---|---|
| 1 | **aya** ‘worthy, honourable’ | **ā́rya, CDIAL 1347**, section 1 (reflex) | Andersen compares RV *ā́rya* and Pali *ayya*; Turner gives Pali *ayya* ‘worthy’ and explicitly cites Ashokan *aya-puta*. |
| 2 | **ayaputa** ‘prince’ | **ordered components aya + puta** (two component rows) | Andersen compares Sanskrit *āryaputra* and Pali *ayyaputta*, then cross-refers to both heads. Turner 1347 explicitly cites Ashokan *aya-puta*; 8265 lists Ashokan *puta* from *putrá*. |
| 3 | **amisa** ‘unmixed, unmingled’ | **ámiśra, CDIAL 567** (reflex) | Andersen compares ŚB *ámiśra* and Pali *a-missa*; Turner 567 explicitly lists Ashokan *amisā* nominative plural. |
| 4 | **ammisa** ‘’ | **source variant of amisa** (variant) | The printed Sa spelling *ammisa* says “see amisa”; the latter is proposal 3. |
| 5 | **mātāpitu** ‘mother and father’ | **ordered components mātā + pitu** (two component rows) | Andersen compares Sanskrit *mātāpitr* and Pali *mātāpitar* and refers to both heads. Turner 10016 explicitly has Ashokan *matapituṣu*; both local components have accepted links to 10016 and 8179. |
| 6 | **dʰammaguna** ‘Dharma-virtue’ | **ordered components dʰamma + guna** (two component rows) | Andersen compares Sanskrit *dharma-guṇa* and Pali *dhamma-guṇa* and refers to both heads, linked to Turner 6753 and the quality section of 4190. |
| 7 | **dʰammayuta** ‘engaged in Dharma’ | **ordered components dʰamma + yuta** (two component rows) | Andersen compares Sanskrit *dharmayukta* and Pali *dhamma-yutta* and refers to both heads. Turner 10479 explicitly gives Ashokan *yuta*; the first component is the linked source *dʰamma*. |
| 8 | **misamdeva** ‘god associated (with)’ | **ordered components misa + deva** (two component rows) | Andersen prints Sanskrit *miśra-deva* and refers to both heads. Turner 10135 explicitly lists Ashokan *misa* ‘mixed’ and 6523 gives Ashokan *devā*. |
| 9 | **iyam** ‘this here’ | **ayám/iyám, CDIAL 587** (reflex) | Andersen compares RV *iyám/ayám/idám* and Pali *ayaṁ*; Turner explicitly lists Ashokan *iyaṁ* alongside *ayaṁ* under this demonstrative paradigm. |
| 10 | **ayam** ‘’ | **source variant of iyam** (variant) | Andersen’s separate *ayam* line says “see iyam”; Turner 587 contains both case/gender forms, so this preserves the printed source cross-reference. |

The ten proposals create **fifteen assignment rows across ten records**: three reflexes, two variants, and ten ordered component edges across five composite records. Three cases are held: **amisamdevā**, because Andersen writes fused and spaced forms by locality without glossing the composite; **mahatata**, because its Sanskrit/Pali formation does not exactly match Turner’s *mahattva* 9933; **apacāy**, because its printed verb comparison lacks a specific Turner article. These holds avoid changing source parsing or flattening distinct morphological formations.

The previous Andersen passes082 and086 saved 22 records and 23 sidecar rows. Immediately before this pass, 155 of Andersen’s 279 source records remained without effective accepted rank-1 ancestry after compiled edges and saved sidecars were combined.

Primary evidence: Andersen scan PDF pp. 136–137, 140, 153, 161–162 (printed pp. 138–139, 142, 155, 163–164). Complete Turner articles in `andersen-primary015.json` through `andersen-primary017.json`; candidate discovery used the homepage search implementation in `andersen-search012.json` and `andersen-search013.json`. Source IDs and accepted ancestors were checked against `andersen-records.json`, compiled edges and the owning sidecar.

Validation: `make check-pass`, `make save-pass`, and `make check-sources` passed. The fifteen saved rows match exactly the fifteen rows in the owning Andersen sidecar; all five component records have positions 1 and 2. The second application made zero changes, protected forms/edges/identity hashes stayed unchanged, and only the Andersen sidecar changed. Source checks found 223 source files with settings, 215 citation keys, and 76,895 assignment rows in 89 sidecars. No rebuild.
