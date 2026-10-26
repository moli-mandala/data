# Andersen (1990): half and two printed cross-references

Three still-unlinked Andersen records, three proposed rows under standing autonomous authorization. Printed dictionary pp. 136–137 were checked on scans; complete Turner articles 250, 644, and 651 are preserved in `andersen-primary123.json` from the installed CDIAL snapshot.

| Ashokan Prakrit | Proposed link | Evidence |
|---|---|---|
| **adʰa** ‘half’ | **ardhá, CDIAL 644** (reflex) | Andersen's independently bounded *adha-* m. ‘half’ compares RV *ardha-* and Pali *addha-*. Turner 644 lists Pali and Ashokan half forms, including northwestern *adha*. |
| **adʰātiya** (source gloss blank; printed “cf. aḍhatiya”) | **source variant of aḍʰatiya** | Andersen prints the cross-reference immediately after the ‘two and a half’ head. Turner 651 directly lists Ashokan *aḍhātiya-* beside *aḍhat°*. The source parent has an accepted compiled edge to 651. |
| **adʰvikāni** (source gloss blank; printed “for adhikāni, cf. adhika”) | **source variant of adʰika** | Andersen's Nittur note directs to the adjacent *adhika-* ‘exceeding’ head, which compares AVP/Pali *adhika-*. The source parent has an accepted compiled edge to Turner 250. |

One hold: **atiyo** ‘excessively’ compares Sanskrit *atīva* and Pali *atīva/ativiya*, but no precise Turner article was established. The apparent *ati-* sequence alone is insufficient for an assignment. The source's *ānapay-* command head was also checked and excluded because it already has a compiled accepted rank-1 edge to CDIAL 1096. These decisions keep the Buddhist *saka* contamination case separate.

All three proposed IDs were absent from the owning Andersen sidecar and compiled edges at preparation. `make check-pass`, `make save-pass`, and `make check-sources` passed. The saved rows match exactly three Andersen sidecar rows. A second application made zero changes; protected forms, edges, identity, and unrelated sidecar hashes remained unchanged. Only the Andersen sidecar changed. Source checks counted 77,040 assignment rows in 90 sidecars. No rebuild.
