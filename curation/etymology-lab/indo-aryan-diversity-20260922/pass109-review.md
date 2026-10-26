# Andersen (1990): documented variants and an Ashokan relative adjective

Four exact source-defined records, four proposed sidecar rows, under standing autonomous authorization. The reflex matches an explicit Turner Ashokan series. The three variants follow printed “see” lines to already linked source heads.

| # | Ashokan Prakrit | Proposed etymology | Evidence |
|---|---|---|---|
| 1 | **yādisa** ‘which like, what like, whichever’ | **yādrśa, CDIAL 10458** (reflex) | Andersen compares RV *yādṛś* and Pali *yādisa*, with *yārisa* in the regional continuation. Turner lists Pali *yādisa* and Ashokan *yadiśaṁ/ādise/yārise*. |
| 2 | **hacʰāroha** (source gloss blank; “see hatʰāroha”) | **source variant of hatʰāroha** | Printed Nittur line says “see hatthāroha”; the linked head is CDIAL 14047 *hastyārōha* ‘elephant rider’. |
| 3 | **hatʰiyāroha** (source gloss blank; “see hatʰāroha”) | **source variant of hatʰāroha** | Printed Erragudi line says “see hatthāroha”. |
| 4 | **hadʰa** (source gloss blank; “see hida”) | **source variant of hida** | Printed Rupnath line says “see hida”; that head is linked to CDIAL 1605 *ihá* ‘here’, which explicitly lists Ashokan *hida*. |

These proposals create **four assignment rows across four records**: one reflex and three source variants. Three cases are held: **budʰaśake** is explicitly marked “for upāsake”; **śaka** only cross-refers to that uncertain reading; and **bumhanāni** says “see bambhana” while the installed head is spelled *bamhana*. The last case needs source transcription review before a graph edge.

**Adjudication finding, outside this pass.** The installed Andersen *saka* ‘a Buddhist?’ record `f_6g47ypsql2pfo` has a compiled accepted reflex edge to CDIAL **12376** `śā́khā` ‘branch’ (`cldf/edges.csv`, `Child_ID=f_6g47ypsql2pfo`, `Parent_ID=12376`, `Kind=reflex`, `Rank=1`). Andersen's printed p. 172 says: “saka- m. ‘a Buddhist?’ Perhaps merely a mistake for upāsaka-.” The printed p. 161 *budhaśake* line says “for upāsake”. Turner **2316** is *upāsaka* ‘worshipper’, citing Ashokan *upāsaka* ‘lay devotee’; Turner 12376 instead treats *śā́khā* ‘branch’. The current branch edge is semantically and historically unsupported. This note flags the bad compiled edge for separate correction; it does not silently replace it or assert that the doubtful surface spelling itself is a regular reflex of *upāsaka*.

Primary evidence: Andersen scan PDF pp. 159, 163–164, 169–170, 175 (printed pp. 161, 165–166, 171–172, 177). Full Turner articles in `andersen-primary021.json`, extracted from the installed CDIAL source snapshot with original page URLs. The source IDs, existing head links and compiled contamination were checked against `andersen-records.json`, the owning sidecar and `cldf/edges.csv`. Before this pass, about 123 of Andersen's 279 source records lacked effective accepted rank-1 ancestry.

Validation: `make check-pass`, `make save-pass`, and `make check-sources` passed. The four saved rows match exactly the four rows in the owning Andersen sidecar. The second application made zero changes, protected forms/edges/identity hashes stayed unchanged, and only the Andersen sidecar changed. Source checks counted 76,993 assignment rows in 90 sidecars. No rebuild.
