# SIL ESR 2008-002: Meitei (Manipuri) survey wordlists

This package installs Appendix B.3 (printed pp. 45–68) of Amy Kim and Seung Kim's
*Meitei (Manipuri) Speakers in Bangladesh: A Sociolinguistic Survey* (SIL Electronic Survey
Report 2008-002; SIL archive 9145).

The official 126-page PDF is not redistributed. Its SHA-256 is pinned in the importer and the
extractor refuses a different file. Place the downloaded PDF at `/tmp/silesr2008_002.pdf` and run:

```sh
python data/other/forms/raw_data/sil_meitei_2008/extract_meitei.py
python data/other/forms/raw_data/sil_meitei_2008/import_meitei.py
```

This source does not require OCR. The PDF wordlists have a complete text layer, but their
phonetic spans use SIL's legacy `SAG-IPASILManuscript` font and appear as U+F000–U+F0FF
private-use characters. `sag_ipa_used.tsv` records every one of the 25 used bytes and all 2,534
occurrences against SIL's official `SAGIPA2Uni.map` v1.0 converter. The extractor asserts the
PDF hash, 126-page extent, wordlist page-text hash, consecutive 1–307 prompt topology, all
1,219 printed response groups, all 2,713 expanded site attestations, and a zero count for unknown
legacy glyphs or unparsed lines.

Codes 1–6 are the Bangladesh survey communities Mukabil, Humerjan, Shivganj, Shivnagar,
Choto Dhamai, and Kunagaon. Codes 7 and 8 are same-language comparison lists from Lilong
Bazaar and Imphal in Manipur and are retained as Meitei attestations. Code 0 is the Standard
Dhaka Bangla comparison list and remains audit-only. The printed lexical-similarity group numbers
are preserved as evidence but are not interpreted as historical cognacy.
