# ESR 2013-004 Northern Dhule Bhils: acquisition and extraction topology

Discovery/acquisition record created 2026-08-28 with
`data/SOURCE_INGESTION_CHECKLIST.md` active. The applicable addenda are
“Survey wordlists or comparative tables” and “OCR-heavy source.” This file
records the initial discovery state; exhaustive source-local transcription is
now complete, while shared installation remains deferred (see `README.md`).

## Pinned source

- Stephen Watters. 2013. *A Sociolinguistic Profile of the Bhils of Northern
  Dhule District*. SIL Electronic Survey Report 2013-004.
- Official SIL archive record: <https://www.sil.org/resources/archives/52641>.
- Official publisher PDF:
  <https://www.sil.org/system/files/reapdata/16/54/22/165422481227949765106602053433447679464/silesr2013_004.pdf>.
- The current publisher endpoint returned an HTML Cloudflare challenge to the
  command line. The canonical publisher artifact was recovered from the
  Internet Archive capture `20230606040837` of that exact URL.
- Acquired workspace file:
  `tmp/pdfs/northern_dhule_bhils_2013/silesr2013_004.pdf`.
- SHA-256:
  `edeeeda98cb76624df1a0d70c765cc816ea463d75bc79ec20883c62e6fc1c482`.
- File size: 9,214,722 bytes; 133 physical PDF pages (`vi, 125 pages` in the
  archive record).

## Appendix C topology

Appendix C begins with its key on physical PDF page 90. The response matrix is
on physical pages 91–133 (printed pages 83–125), ending with item 210. It
prints **210 prompts × 13 lists = 2,730 conceptual cells**.

The twelve Bhil-region target lists are:

- Kelpada, Dhanoura, Digiamba, and Amoda Vasave;
- Mundalwad Bhilori;
- Astamba Noiri;
- Mandvi, Bhusha, Amalwadi, Segwi, Kangai, and Shahana Pauri.

They contribute **2,520 target cells**. Toranmal Nahali is the single
audit-only control/comparison row, contributing **210 control cells**.

Several lists are republished or reused in later regional reports. The future
ingest must reconcile exact list identity with the already installed ESR
2018-011 Bareli/Pauri package and with ESR 2015-012 Noira before deciding which
cells are new. No later republication may overwrite or duplicate the earliest
authoritative transcription silently.

## Mandatory manual transcription and review

Appendix C is an image scan. Any embedded OCR/text is noisy and is not source
text. OCR may only be retained as a locator/comparison scaffold. Every one of
the **2,730 conceptual cells** must be inspected manually on rendered pages,
and every installed IPA response must be transcribed by hand and visually
verified. The audit must explicitly account for every attested cell, true
blank, dash, qualification, alternative, ambiguity, clipped cell, and
illegible reading with physical/printed page, item, and list coordinates. No
OCR-only form may install and uncertain IPA must never be guessed.

The appendix key and all response pages were rendered and visually inspected,
confirming the thirteen-list geometry and item-210 endpoint. Acquisition,
topology, exhaustive manual transcription, source-local staging, audit, and
profile proposal are resolved. Shared duplicate reconciliation, registries,
installation, full build/tests, and any user-requested browser QA remain
pending.
