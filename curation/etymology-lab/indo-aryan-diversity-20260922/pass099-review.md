# Andersen (1990): printed participle and rule variants

Nine exact source-defined records, nine proposed sidecar rows, under standing autonomous authorization. Turner directly cites the Ashokan forms or their locally documented inflections; Andersen prints three explicit cross-references to the heads proposed in this pass.

| # | Ashokan Prakrit | Proposed etymology | Evidence |
|---|---|---|---|
| 1 | **kata** ‘made’ | **kṛtá subentry of CDIAL 2814** (reflex) | Andersen compares RV *kṛtá*, Pali *kata*; Turner explicitly lists Girnar *kata* and regional *kaṭa/kiṭa* in the participial subentry, distinct from finite *karóti*. |
| 2 | **kaṭa** ‘’ | **source variant of kata** | The separate printed line says “see kata”; Turner lists the regional *kaṭa* participle. |
| 3 | **kitā** ‘’ | **source variant of kata** | The separate printed line says “see kata”, and Andersen records Panguraria *kitā* in its head entry. |
| 4 | **no** ‘nor, but not’ | **nṓ, CDIAL 7607** (reflex) | Andersen gives RV/Pali *no* and explicitly analyses Sanskrit *no* from *na-u*; Turner cites Ashokan *no* ‘not’. |
| 5 | **pakiti** ‘rule, usage’ | **prakṛti, CDIAL 8446** (reflex) | Andersen compares GB *prakṛti* and Pali *pakati*; Turner directly cites Ashokan *pakiti* ‘rule, standard’. |
| 6 | **pakati** ‘’ | **source variant of pakiti** | The printed Udegolam spelling says “see pakiti”. |
| 7 | **saka** ‘able, possible’ | **śákya, CDIAL 12253** (reflex) | Andersen compares Sanskrit *śakya* and Pali *sakka*; Turner lists Girnar *saka* ‘possible’. The separate same-spelled Buddhist noun is a different installed record and already linked. |
| 8 | **nātika** ‘relative’ | **jñātí, CDIAL 5276**, with -ka (reflex) | Andersen compares Sanskrit *jñātika* and Pali *ñāti-ka*; Turner explicitly lists Ashokan Girnar *ñātika* and Kalsi *nātikye*. |
| 9 | **dīgʰāvus** ‘long life’ | **dīrghā́yus, CDIAL 6371** (reflex) | Andersen compares RV *dīrghā́yus* and Pali *dīghāyu/dīghāvu*; Turner directly cites Ashokan *dīghāvuse*, allowing nom. ‘long-lived’ or dat. ‘for a long life’, and the source labels its form dative. |

These proposals create **nine assignment rows across nine records**: six reflexes and three source variants. Three cases are held: **śaka** refers to a Buddhist expression rather than the adjective; **kiti** is “for kiṃti” but its head is unlinked; **kiṇti** combines *kim + iti* without a precise Turner compound article. They are not attached to superficially similar parents.

Primary evidence: Andersen scan PDF pp. 143, 145, 152, 154–155, 169–170 (printed pp. 145, 147, 154, 156–157, 171–172). Full Turner articles in `andersen-primary019.json`, extracted from the installed CDIAL source snapshot with original page URLs. Source records and the distinction between the two *saka* homographs were checked against `andersen-records.json`; accepted edges and the owning sidecar were combined to establish still-unlinked targets. Before this pass, 137 of Andersen’s 279 source records lacked effective accepted rank-1 ancestry.

Validation: `make check-pass`, `make save-pass`, and `make check-sources` passed. The nine saved rows match exactly the nine rows in the owning Andersen sidecar. The second application made zero changes, protected forms/edges/identity hashes stayed unchanged, and only the Andersen sidecar changed. Source checks counted 76,944 assignment rows in 90 sidecars. No rebuild.
