# JLSR 2024-009 Ho dialects: acquisition and extraction topology

Discovery/acquisition record created 2026-08-28 with
`data/SOURCE_INGESTION_CHECKLIST.md` active. The applicable addenda are “Survey
wordlists or comparative tables” and “OCR-heavy source.” This is **not** a
completed ingestion: Appendix D.3 is queued for exhaustive manual
transcription and visual review.

## Pinned source

- Varenkamp, Bryan. 2024 [survey fieldwork 1989]. *A Study of Ho Dialects*.
  Journal of Language Survey Reports 2024-009.
- Official SIL archive record: <https://www.sil.org/resources/archives/100299>.
- Official publisher PDF:
  <https://www.sil.org/system/files/reapdata/57/43/56/57435603355882063345519297340287951071/JLSR2024_009.pdf>.
- SIL's current file endpoint returned an HTML bot-challenge to a command-line
  client, so the byte-identical publisher artifact was recovered from the
  Internet Archive capture `20251206143804` of that exact official URL.
- Acquired workspace file: `tmp/pdfs/ho_2024/JLSR2024_009.pdf`.
- SHA-256:
  `5ca30882dc5ed0f8480c9710e5fc0e08bf4d92e27d591582e3d953709ec1f9d1`.
- File size: 12,467,726 bytes; 142 physical PDF pages.
- Rights: the JLSR publication page supplies its scholarly-research/instruction
  fair-use policy. The PDF is not redistributed in this source package.

## Appendix D topology

Appendix D.1 gives metadata on physical PDF pages 68–69 (printed pages 59–60),
Appendix D.2 supplies the survey phonetic alphabet on physical pages 70–71,
and Appendix D.3 prints the wordlist matrix on physical pages 72–141 (printed
pages 63–132). Appendix D.3 has 210 numbered prompts, normally three prompts
per page, and twenty-seven list rows per prompt:

### New target field lists

| Code | Label/locality |
|---|---|
| HTH | Chirupada, Thakurmunda, Mayurbhanj |
| HKA | Sarudha, Karanjia, Mayurbhanj |
| HKE | Jude near Saisuathana, Keonjhar |
| HCH | Todangbasa, Chaibasa, Singhbhum |
| HCU | Kihki near Kurjatie, Cuttack |
| HSU | Tokora near Lahunipara, Sundargarh |
| HSA | Mosinta near Barakot, Sambalpur |
| HJO | Chakidi, Joshipur, Mayurbhanj |
| HDH | Kerjanga near Pal Lahara, Dhenkanal |
| HBG | Nakti, Bandgaon, Singhbhum |
| HRA | Kuchepal, Rairangpur, Mayurbhanj |
| HOP | Baigodia, Opada block, Balasore |
| HBA | Bahaldia, Baripada, Mayurbhanj |
| HNI | Hento Sura/Sura, Nilgiri, Balasore |

These fourteen 1989 field lists contribute **14 × 210 = 2,940 conceptual new
target cells**.

### Republished Ho lists

- `HO1`: John Deeney's 1975 *Ho-English Dictionary* (Chaibasa).
- `HO2`: Lionel Burrows's 1915 *A Grammar of the Ho Language* (Kolhan).
- `HO3`: Edward T. Dalton's 1872 *Tribal History of Eastern India* wordlist
  (Chota Nagpur), explicitly described as only 68 words and of questionable
  reliability.

These three rows contribute 630 conceptual cells but reproduce earlier
published sources rather than this survey's fieldwork. They must remain
audit-only comparanda pending exact duplicate/source reconciliation and must
not overwrite any primary-source Jambu transcription.

### Non-Ho comparison lists

- Bhumij: `BBG`, `BMA`, `BOP`, `BRA`, `BGH` (five lists).
- Mundari: `MU1`, `MU2` (two published-source lists).
- Santali: `SA1`, `SBA` (one published-source and one field list).
- Oriya: `OCU` (one field control).

These ten lists contribute **2,100 conceptual comparison cells** and are
audit-only for this Ho source package. The complete printed matrix therefore
has **27 × 210 = 5,670 conceptual cells**: 2,940 new target cells, 630
republished Ho cells, and 2,100 non-Ho comparison cells.

Source-prefixed numbers are lexical-similarity group judgements, not
historical cognacy. Commas and punctuation may distinguish alternatives or
belong within multiword responses, so no punctuation may be split
mechanically without item-level review.

## Transcription and review requirement

Appendix D.3 is an image scan; its wordlist pages have no usable text layer.
OCR may be retained only as a locator/comparison scaffold. Every one of the
5,670 cells must be read manually from rendered pages, and every installed form
must be visually verified. The eventual package must account for every blank
or ruled dash, keep all republished and comparison rows audit-only, preserve
the source phonetic alphabet diplomatically, and list every ambiguous,
questioned, clipped, or illegible reading by physical/printed page, prompt,
and source code. No OCR-only reading may install.

Physical pages 72, 73, and 141—the first six prompts and final three prompts—
were rendered and visually inspected during discovery, confirming the
twenty-seven-row topology and image-only transcription. Acquisition, matrix
geometry, and initial inclusion policy are resolved; exhaustive manual
transcription, source-local importer/profile work, shared metadata, audit,
full build, and browser QA remain pending.
