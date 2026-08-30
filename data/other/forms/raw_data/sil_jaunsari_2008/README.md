# SIL ESR 2008-013 Jaunsari wordlists

Source: Matthews John, *Jaunsari: A Sociolinguistic Survey*, SIL Electronic Survey
Report 2008-013 (2008), SIL archive record 9074. The publisher PDF is not
redistributed. Its SHA-256 is
`e6b3b6d54c061d03614b27618f0f06d2138f07c47dc1a266d45b0fe16bd75f68`.

The report contains a 210-prompt comparative list in Appendix A.2. It explicitly
disqualifies and omits items 11 *breast*, 23 *urine*, and 24 *feces*. Pages 40–75
print 2,729 responses from seven Jaunsari lists and five comparison-language
controls. The seven targets are Chakrata (A), Bhandroli (B), Chapnu (C), Khanaad
(D), Korwa (K), Lakhamandal (L), and Maindrath (M). Hindi (h), Bangani (S),
Jaunpuri (J), Nagpuriya (N), and Sirmauri (G) remain audit-only controls.

## Legacy IPA recovery

The wordlist is text, not a scan, but the embedded `SAG-IPASILDoulos` font maps
its original single-byte encoding into PDF private-use code points U+F000–U+F0FF.
SIL Converters 5.4.1 contains the authoritative `SAGIPA2Uni.map` v1.0 converter
(full map SHA-256
`a989926e91d4b562df20758cbb613f0177fce33d1c2e9e02195087e94f1f2930`).
`sag_ipa_used.tsv` records the exact 32-byte subset that occurs on pages 40–75.
`extract_jaunsari.py` verifies the publisher PDF hash and page count, rejects any
unmapped private-use byte or unparsed line, checks every printed item against all
12 lists, and regenerates `wordlists.tsv`. The sole physical line wrap is the
Sirmauri response to item 121 and is joined deterministically.

Run extraction from the outer workspace:

```sh
cd data
uv run --with pypdf python data/other/forms/raw_data/sil_jaunsari_2008/extract_jaunsari.py
python data/other/forms/raw_data/sil_jaunsari_2008/import_jaunsari.py
```

The importer installs 1,619 target forms. The 1,110 control responses and three
source-declared item exclusions remain in the 2,732-row audit. Source forms are
already Unicode IPA after the official conversion, so `Form` and `Phonemic` are
identical and no uncertain legacy symbols remain.
