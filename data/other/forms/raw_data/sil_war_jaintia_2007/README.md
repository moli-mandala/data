# SIL ESR 2007-013 War-Jaintia wordlists

This directory contains the reproducible extraction of Appendix B.3 (printed
pp. 57–87) from Jeremy Brightbill, Amy Kim and Seung Kim's 2007 SIL Electronic
Survey Report *The War-Jaintia in Bangladesh: A Sociolinguistic Survey*.

The publisher PDF is not redistributed. `extract_war_jaintia.py` requires the
preserved official file at `/tmp/silesr2007_013-war-jaintia.pdf`, verifies its
SHA-256 and its 153-page topology, reads the embedded text layer, and converts
the report's `SAG-IPASILManuscript` private-use bytes using the checked-in subset
of SIL's official `SAGIPA2Uni.map` v1.0. This is deterministic font conversion,
not OCR. `sag_ipa_used.tsv` pins all 17 used bytes and all 2,398 occurrences.

`wordlists.tsv` retains all 1,690 printed response records and expands to 3,459
site attestations. `import_war_jaintia.py` installs 2,030 attestations from the
seven War-Jaintia lists (A–E, I, J) and records the 1,428 Pnar, Lyngngam, Khasi
War and standard Khasi controls in the audit only. The source's undefined `U`
at item 119 is retained and explicitly excluded; visual inspection shows that
the letter is printed in the PDF although Appendix B.2 defines only A–L. The
lettered group `A` at item 137 is the source's tenth similarity group after 1–9
and is retained verbatim.

Representative pages 57, 68, 70 and 87 were rendered from the preserved PDF and
checked against the extracted forms, including both printed anomalies. No
etymological relationships are asserted by the report, so no graph edges are
created.
