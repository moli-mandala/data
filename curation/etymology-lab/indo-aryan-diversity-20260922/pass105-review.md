# Andersen (1990): cross-referenced forms and a lasting compound

Five exact source-defined records, six proposed sidecar rows, under standing autonomous authorization. The glossary explicitly refers to the already linked source heads, and Turner supplies either the source word family or its Ashokan compound form.

| # | Ashokan Prakrit | Proposed etymology | Evidence |
|---|---|---|---|
| 1 | **ciraṭʰitika** ‘perpetual, lasting long’ | **ordered components cira + tʰitika** | Andersen compares Sanskrit *cirasthitika* and Pali *ciratthitika* and explicitly refers to both heads. Turner 4824 gives Ashokan *cira* in compounds; 13770 gives Ashokan *cira/cila-thitika* forms. |
| 2 | **caka** (source gloss blank; “see saka”) | **source variant of adjective saka** | Printed “see saka”; this refers to the adjective ‘able, possible’ linked to CDIAL 12253. The same-spelled Buddhist noun is a separate installed record. |
| 3 | **cʰavacʰara** (source gloss blank; “see samvacʰara”) | **source variant of samvacʰara** | Printed Rupnath spelling says “see saṃvachara”, the already linked ‘year’ head (CDIAL 13011). |
| 4 | **sāvāna** (source gloss blank; “see sāvana”) | **source variant of sāvana** | Printed Sahasram spelling says “see sāvana”, the already linked ‘announcement, edict’ head (CDIAL 12698). |
| 5 | **jānapada** ‘belonging to the country’ | **derived from source janapada** | Andersen compares ŚB *jānapada* and Pali *janapada*, cross-referring to the source's noun. Turner 5101 explicitly distinguishes Kalsi *janapade* from Girnar *jānapadasa* ‘living in the country’. |

These proposals create **six assignment rows across five records**: two ordered component edges, three printed variants, and one derived adjective. Three cases are held: **udala** and **udālaka**, whose installed dental spellings differ from the printed retroflex forms and Turner's matching retroflex forms under CDIAL 1935; **hīyam**, whose printed cross-reference points to both *hi* and *iyam*. These discrepancies require editorial review rather than silent source correction.

Primary evidence: Andersen scan PDF pp. 141, 147–149, 173, 176 (printed pp. 143, 149–151, 175, 178). Complete Turner articles in `andersen-primary020.json`, extracted from the installed CDIAL source snapshot with original page URLs. The source forms and accepted ancestors were checked against `andersen-records.json`, compiled edges, and the owning sidecar. Before this pass, 128 of Andersen's 279 source records lacked effective accepted rank-1 ancestry.

Validation: `make check-pass`, `make save-pass`, and `make check-sources` passed. The six saved rows match exactly the six rows in the owning Andersen sidecar, with the compound's two positions in order. The second application made zero changes, protected forms/edges/identity hashes stayed unchanged, and only the Andersen sidecar changed. Source checks counted 76,969 assignment rows in 90 sidecars. No rebuild.
