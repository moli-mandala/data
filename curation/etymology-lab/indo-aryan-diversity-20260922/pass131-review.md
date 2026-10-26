# Andersen (1990): pronoun and two verbs with direct Ashokan comparanda

Three still-unlinked Andersen records, three proposed reflex rows under standing autonomous authorization. The exact dictionary pages were checked on scans, and complete Turner articles 992, 9416, and 10452 are preserved in `andersen-primary131.json` from the installed CDIAL snapshot.

| Ashokan Prakrit | Proposed link | Evidence |
|---|---|---|
| **aham** (source gloss blank; nominative ‘I’) | **ahám, CDIAL 992** (reflex) | Andersen printed p. 140 says “see hakam”; its pronoun paradigm on printed p. 177 labels the Gujarra *ahaṁ* nominative singular ‘I’ and compares RV/Pali *ahaṁ*. Turner 992 explicitly lists Ashokan *ahaṁ* and *hakaṁ*. |
| **yā** ‘to go, to go on’ | **yā́ti, CDIAL 10452** (reflex) | Andersen printed p. 165 compares RV/Pali *yāti* under its separate motion-verb head. Turner 10452 lists both Pali *yāti* and Ashokan *yāti/yaṁti*. |
| **hu** ‘to be, become’ | **bhávati, CDIAL 9416** (reflex) | Andersen printed p. 178 compares RV *bhavati* and Pali *hoti*, then gives Ashokan *hoti/hota/hotu*. Turner 9416 directly lists Ashokan *hoti* under *bhávati*. |

Two holds: **as** ‘to exist’ cites Sanskrit *asti*, but the attested series printed under that head is chiefly *sumi/smi* as first-person ‘I’ and Andersen quotes a disputed analogical explanation. **sumi** itself is used in both nominative and instrumental replacement contexts under *hakam*, so a single verbal or pronominal edge would collapse its uses. *Mai* and *hamaye* are also case forms under the *hakam* paradigm, not simple nominative variants. The Buddhist *saka* contamination correction remains separate.

All three target IDs were absent from the owning Andersen sidecar and compiled edges at preparation. `make check-pass`, `make save-pass`, and `make check-sources` passed. The three saved rows match exactly three Andersen sidecar rows; a second application made zero changes. Protected forms, edges, identity, and unrelated sidecar hashes remained unchanged; only the Andersen sidecar changed. Source checks counted 77,082 assignment rows in 90 sidecars. No rebuild.
