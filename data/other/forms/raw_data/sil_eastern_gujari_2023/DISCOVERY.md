# JLSR 2023-002 Eastern Gujari: acquisition and extraction topology

Discovery/acquisition record created 2026-08-28 with
`data/SOURCE_INGESTION_CHECKLIST.md` active. The applicable addendum is “Survey
wordlists or comparative tables.” This is **not** a completed ingestion: the
eight Indian lists are queued for exhaustive extraction and visual review.

## Pinned source

- Hugoniot, Ken, Dietmar Polster, Bashir Ahmad, and Kennedy Rajan. 2023
  [survey fieldwork 1996; report dated 1997]. *A Sociolinguistic Profile of
  Eastern Gujari*. Journal of Language Survey Reports 2023-002.
- Official SIL archive record: <https://www.sil.org/resources/archives/95899>.
- Official publisher PDF:
  <https://www.sil.org/system/files/reapdata/16/64/68/166468818346814241493507732958257420275/JLSR2023_002.pdf>.
- Acquired workspace file: `tmp/pdfs/eastern_gujari/JLSR2023_002.pdf`.
- SHA-256:
  `41352b2db97dbd059a1bc229a8ed370fed700c1726f3886a580cba586137475e`.
- File size: 9,149,165 bytes; 121 physical PDF pages.
- Rights: the JLSR publication page supplies its scholarly-research/instruction
  fair-use policy. The PDF is not redistributed in this source package.

## Appendix B topology

Appendix B, “Gujari Wordlists,” occupies physical PDF pages 41–76 (printed
pages 33–68). It prints 210 prompts for fifteen lists, arranged three prompts
per block:

### New target lists collected in India

| List | State as printed | District as printed |
|---|---|---|
| Udhampur | Jammu and Kashmir | Udhampur |
| Jammu | Jammu and Kashmir | Jammu |
| Chamba | Himachal Pradesh | Chamba |
| Rampur | Himachal Pradesh | Shimla |
| Nalagarh | Himachal Pradesh | Solan |
| Dehra Dun | Uttar Pradesh | Dehra Dun |
| Kotdwara | Uttar Pradesh | not supplied |
| Haldwani | Uttar Pradesh | Naini Tal |

The state labels are historical source labels and must be retained in source
notes; present-day administrative metadata must be resolved independently.
These eight lists contribute **8 × 210 = 1,680 conceptual target cells**.

### Republished Pakistan lists and control

| Appendix label | Original SSNP locality | Existing Jambu dialect | Existing installed rows |
|---|---|---|---:|
| Chitral | Ashriki, Shishi Koh valley | `SSNP-gojri-CHT` | 208 |
| Settled Swat | Peshmal, Swat valley | `SSNP-gojri-SSW` | 210 |
| Gilgit | Naltar Bala | `SSNP-gojri-GLT` | 209 |
| Kaghan | Mittikot above Balakot | `SSNP-gojri-KGH` | 209 |
| Northern Azad | Muzaffarabad / Subri | `SSNP-gojri-NAK` | 210 |
| Central Azad | Rawalakot / Trarkhel | `SSNP-gojri-CAK` | 208 |
| Urdu | Pakistan-survey control | control only | not installed here |

The report explicitly says that the six Pakistan Gujari lists and Urdu list
were collected by the earlier Pakistan survey and entered into the present
WordSurv database from Hallberg and O’Leary (1992). The six Gujari lists are
already installed from that primary SSNP volume in
`data/other/forms/20260725-ssnp.csv`: **1,254 attested rows across 1,260
conceptual cells**. Spot checks at both appendix ends confirm the same lexical
data, although the 2023 reprint sometimes retains similarity-group numbers,
uses slightly different diplomatic segment notation (for example `dʒ` versus
the primary import's `j`), or orders alternatives differently. They must be
audited as republished comparanda, not installed a second time or used to
overwrite the primary-source transcription. The Urdu list is likewise an
audit-only control.

The complete printed matrix therefore has **15 × 210 = 3,150 conceptual
cells**: **1,680 new Indian target cells**, **1,260 republished Pakistan Gujari
cells**, and **210 Urdu control cells**.

Numbers prefixed to forms are the report's lexical-similarity groups and must
remain notes rather than historical-etymological claims. Slashes may separate
multiple source responses, while spaces and punctuation within a response can
be phrase-internal; these distinctions must be resolved item by item rather
than by mechanical punctuation splitting.

## Transcription and review requirement

The appendix is born-digital, typeset Unicode IPA with a usable text layer, so
OCR is neither required nor appropriate. Nevertheless, every extracted cell
and every installed form must be compared visually with the rendered canonical
page. The eventual package must account for all 3,150 conceptual cells; install
only the eight new Indian lists; retain the six republished Pakistan lists and
Urdu as audit-only rows; distinguish alternatives from phrase-internal
punctuation; preserve diacritics and source uncertainty; report every blank;
and identify every ambiguous or clipped reading by physical/printed page,
prompt, and list.

Physical pages 41 and 76—the appendix topology page and the final block,
items 205–210—were rendered and visually checked during discovery. The exact
source is acquired and its matrix geometry and duplicate-source policy are
resolved; extraction, exhaustive visual review, importer/profile work, shared
metadata, audit, full build, and browser QA remain pending.
