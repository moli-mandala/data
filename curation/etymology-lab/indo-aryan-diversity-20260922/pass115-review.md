# Andersen (1990): adverb, plural pronoun, and hundred

Four exact source-defined records, four proposed sidecar rows, under standing autonomous authorization. The printed glossary and Turner agree on the adverb and pronoun paradigms. A fresh scan check confirms that *sata* ‘hundred’ has its own bounded entry, resolving the cached OCR boundary concern noted in pass065.

| # | Ashokan Prakrit | Proposed etymology | Evidence |
|---|---|---|---|
| 1 | **tatʰā** ‘so, thus’ | **táthā, CDIAL 5646** (reflex) | Andersen compares RV/Pali *tathā*. Turner lists Ashokan *tatha* with variable final vowel length under *táthā*. |
| 2 | **tatʰa** (source gloss blank; printed “see tatha”) | **source variant of tatʰā** | The Nittur cross-reference precedes the source's ‘so, thus’ head; Turner's Ashokan entry covers the short/long ending. |
| 3 | **ve** (source gloss blank; acc./dat. plural “you”) | **vaḥ, CDIAL 11183** (reflex) | Andersen prints “(Skt vaḥ) see tu-” and labels the inscriptional form acc./dat. plural; Turner explicitly gives Ashokan *ve* dative. |
| 4 | **sata** ‘hundred’ | **śatá, CDIAL 12278** (reflex) | Andersen's independently bounded head compares RV *śatá* and Pali *sata*; Turner lists the Pali and Ashokan hundred series. |

These proposals create **four assignment rows across four records**: three reflexes and one printed source variant. Three cases are held: **vā** points to a *va* source record that merges particle and verb senses; **tu** itself merges the ‘but’ particle and ‘you’ pronoun; **heta** ‘here’ has a proposed *etra* formation without a precise Turner article. A single etymon would collapse distinct forms in the first two cases.

Primary evidence: Andersen scan PDF pp. 149–150, 167–168, 171, 176 (printed pp. 151–152, 169–170, 173, 178). Complete Turner articles in `andersen-primary022.json`, extracted from the installed CDIAL source snapshot with original page URLs. Source IDs and still-unlinked status were checked against `andersen-records.json`, compiled edges and the owning sidecar. Before this pass, about 119 of Andersen's 279 source records lacked effective accepted rank-1 ancestry.

Validation: `make check-pass`, `make save-pass`, and `make check-sources` passed. The four saved rows match exactly the four rows in the owning Andersen sidecar. The second application made zero changes, protected forms/edges/identity hashes stayed unchanged, and only the Andersen sidecar changed. Source checks counted 77,009 assignment rows in 90 sidecars. No rebuild.
