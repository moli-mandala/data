# SIL ESR 2011-025 Bangladesh Kuki-Chin wordlists

This directory contains the reproducible extraction of Appendix A.3 (printed
pp. 50–88) from Amy Kim, Palash Roy and Mridul Sangma's 2011 SIL Electronic
Survey Report *The Kuki-Chin Communities of Bangladesh: A Sociolinguistic
Survey*.

The publisher PDF is not redistributed. `extract_kuki_chin.py` requires the
public 127-page English-report/Bangla-appendix file, verifies its SHA-256, and
reads its complete embedded text layer. The `SAG-IPA-SILManuscript` font is
subsetted into private-use code points, so the extractor deterministically
decodes the embedded outlines and SIL SAGIPA2Uni mapping rather than applying
OCR. `sag_ipa_used.tsv` pins all 65 used glyphs and all 16,029 occurrences.

`wordlists.tsv` retains all 2,565 printed response records and expands to 3,875
site attestations. `import_kuki_chin.py` installs 3,235 attestations from the ten
Bangladesh sites: two each for Pangkhua, Bawm, Lushai/Mizo, Khyang/Asho Chin,
and Khumi. The 307 standard Bangla controls and 333 Myanmar Khumi comparison
attestations remain in the complete audit only. The latter include all 53
printed `no entry` records.

Similarity-group labels are preserved as source notes and do not assert
historical cognacy. No etymological relationships are asserted by the report,
so no graph edges are created.
