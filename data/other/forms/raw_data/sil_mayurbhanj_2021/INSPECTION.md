# JLSR 2021-028 Mayurbhanj: source inspection

## Decision

**No lexical forms are published in this report; there are no source records
to ingest.** The report states that eight wordlists were collected in the 1989
survey and compares them with twelve earlier or dictionary lists, but it prints
only a triangular lexical-similarity percentage matrix. It supplies no prompt
list, phonetic transcriptions, form-level similarity assignments, comparison
denominators, or lexical appendix.

This inspection applies the survey-wordlist/comparative-table addendum of
`SOURCE_INGESTION_CHECKLIST.md`. The OCR-heavy addendum is inapplicable: the
complete PDF is born-digital with a usable text layer, and there are no printed
lexical response cells to transcribe.

## Canonical source and inspected artifact

- Bryan Varenkamp. 2021. *A Summary of the 1989 Mayurbhanj Survey*.
  Journal of Language Survey Reports 2021-028. SIL International.
- SIL archive record: <https://www.sil.org/resources/archives/88872>.
- Canonical publisher PDF:
  <https://www.sil.org/system/files/reapdata/12/40/00/124000268832303749671562379320478565107/JLSR2021_028.pdf>.
- Inspected workspace file: `tmp/pdfs/mayurbhanj/JLSR2021_028.pdf`.
- PDF SHA-256:
  `34d6770e7c59edb975874a6f689b5a9b9cb7da7db19f75cae9a7ef6cdab2d247`.
- File size: 849,049 bytes; 14 PDF pages; printed extent v + 8.
- The archive record identifies the report as created in 1989 and published in
  2021. PDF p. 3 gives the JLSR scholarly-research/instruction fair-use policy
  and prohibition on republication or commercial use without consent.
- The report PDF is not redistributed by Jambu. This inspection pins metadata,
  hash, scope evidence, and the zero-form decision only.

Direct command-line access to the publisher file returned a 5,770-byte HTML
challenge. The canonical PDF was therefore downloaded through the SIL archive
page in the in-app browser, then its size, hash, metadata, text layer, page
count, and relevant rendered pages were verified locally.

## Exact evidence

The complete fourteen-page PDF was text-inspected. The contents, comparison
matrix, and final page were also rendered and visually checked.

1. **Contents, PDF p. 5 / printed p. iv.** The report runs from sections 1--9
   directly to `References`. There is no appendix, wordlist-data section, or
   form table.
2. **Survey contact, PDF p. 8 / printed p. 2.** The report says that a Mohanta
   wordlist and text were obtained. Neither is printed.
3. **Lexical similarity, PDF p. 10 / printed p. 4, §5 and Table 2.** The text
   states that eight wordlists collected during this survey were compared with
   five earlier Bhumij lists, one Mundari list, two Santali lists, two Ho lists,
   and one Oriya list. Table 2 names twenty lists in all and prints the complete
   lower triangle of pairwise percentages, but no forms or denominators.
4. **Current-survey lists.** Underlining in Table 2 marks the eight newly
   collected lists: Asthia Mundari, Dhungarisai Mundari, Baripada Mundari,
   Madhupur Bhumij, Bisoi Birhor, Dhungarisai Birhor, Bisoi Mahali/Santali, and
   Mohanta. The other twelve rows are earlier field or dictionary comparisons.
5. **Birhor discussion, PDF pp. 11--12 / printed pp. 5--6.** The report states
   that two Birhor wordlists were collected and summarizes their percentage
   relationships. It does not reproduce the lexical evidence.
6. **Recommendations, PDF p. 12 / printed p. 6.** The author calls for more
   wordlists and further survey rather than presenting the collected lists.
7. **Report terminus, PDF p. 14 / printed p. 8.** The final page is
   `References`; no lexical appendix follows it.

## Accounting and exclusions

- Published list labels in Table 2: **20**.
- Current-survey lists identified by underlining: **8**.
- Earlier/dictionary comparison lists: **12**.
- Published pairwise percentage cells: **190** (`20 × 19 / 2`), represented
  by the complete triangular matrix.
- Published lexical-form cells: **0**.
- Published prompts, IPA responses, native-script responses, form-level blanks,
  clipped forms, illegible forms, and per-pair comparison denominators: **0**.
- Manual IPA cells inspected/transcribed: **0**, because the report publishes
  no IPA response cell.
- Table 2 percentages are aggregate lexicostatistical results. They are not
  lexical attestations and cannot be reverse-engineered into forms, cognate
  sets, or historical-etymology edges.
- No installed forms CSV, importer, sound profile, bibliography entry, language
  row, dialect row, graph edge, or browser database change is justified solely
  by this report.

The unpublished 1989 wordlists remain desirable independent sources. If their
archival data sheets or a later publication containing the actual responses are
located, they should be pinned and ingested as separate sources with their own
complete cell audits.

