# Andersen (1990): command, causative verbs, and printed variants

Five still-unlinked source records, five proposed sidecar rows under standing autonomous authorization. The evidence is the exact scanned glossary pages and complete Turner articles, retained in `andersen-primary119.json` from the installed CDIAL snapshot.

| Ashokan Prakrit | Proposed link | Source and Turner evidence |
|---|---|---|
| **āṇapayati** (source gloss blank; ‘command’ at referenced head) | **ājñāpayati, CDIAL 1096** (reflex) | Andersen printed p. 140 says “see āṇapay-”; printed p. 141 glosses that head ‘to give an order, to command’, compares Sanskrit *ājñāpayati* and Pali *āṇāpeti*. Turner lists the Ashokan command forms in 1096. |
| **ārādʰ** ‘to please’ | **ārādhayati, CDIAL 1316** (reflex) | Andersen printed p. 141 compares Sanskrit *ārādhnoti*, Pali *ārādheti*, and gives the causative infinitives. Turner 1316 lists Pali *ārādheti* and Ashokan causative forms under branch 1. |
| **ārodʰetave** (source gloss blank; printed “see ārādh-”) | **source variant of ārādʰ** | The Rupnath cross-reference points to the preceding verb head, which explicitly records *ārodhetave* as a Rupnath form of causative *ārādhetave*. |
| **pavat** ‘to practice [dhammaguna]’ | **právartayati, CDIAL 8788 branch 3** (reflex) | Andersen printed p. 158 compares Sanskrit *pra* + √*vart*, Pali *pavattati/pavatteti*, and prints *pavatitaviyā*. Turner 8788 branch 3 explicitly gives Ashokan *pavatitaviyā* ‘to perform’. The next source head *pavata-* ‘mountain’ is separate. |
| **pā** (source gloss blank; printed “for pi, see pi”) | **source variant of pi** | Andersen printed p. 158 directs the Rupnath form to *pi* ‘also’; that installed source head already links to CDIAL 485 *ápi*, which lists Ashokan *pi*. |

Primary source pages: Andersen PDF pp. 138–139 and 156 (printed pp. 140–141 and 158). The CDIAL article cache records the complete 485, 1096, 1316, and 8788 articles with page URLs. All five IDs were absent from the owning Andersen sidecar and prior diversity decisions when prepared. No Buddhist *saka* contamination correction is part of this pass.

Validation: `make check-pass`, `make save-pass`, and `make check-sources` passed. The five saved rows match exactly five owning sidecar rows. A second application made zero changes; protected forms, edges, identity, and unrelated sidecar hashes remained unchanged. Only the Andersen sidecar changed. Source checks counted 77,027 assignment rows in 90 sidecars. No rebuild.
