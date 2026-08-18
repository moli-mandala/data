# Ghatage survey scan provenance

Checked 2026-08-17.  The Maharashtra State Board for Literature and
Culture is the original digital publisher.  Its `marathi.gov.in`
authoritative DNS server was not answering during this import, and the
Internet Archive Wayback CDX service returned HTTP 503.  The URLs below are
retained so acquisition can be resumed without rediscovery.

| Volume | Year | Publisher scan URL | Local status |
|---|---:|---|---|
| Konkani of South Kanara | 1963 | <https://sahitya.marathi.gov.in/scans/Konkani%20of%20South%20Kanara.pdf> | Awaiting publisher/archive recovery |
| Kudali | 1965 | <https://sahitya.marathi.gov.in/scans/Kudali%20%28II%29.pdf> | Awaiting publisher/archive recovery |
| Kunbi of Mahad | 1966 | <https://msblc.maharashtra.gov.in/pdf/newpdf/Kunbi%20of%20Mahad.pdf> | Awaiting publisher/archive recovery |
| Cochin | 1967 | <https://msblc.maharashtra.gov.in/pdf/pdf/cochin%20pdf.pdf> | Awaiting publisher/archive recovery |
| Konkani of Kankon | 1968 | <https://msblc.maharashtra.gov.in/pdf/newpdf/Konkani%20of%20Kankon.pdf> | Awaiting publisher/archive recovery |
| Varli of Thana | 1969 | <https://msblc.maharashtra.gov.in/pdf/pdf/Warli%20Of%20Thana.pdf> | Awaiting publisher/archive recovery |
| Marati of Kasargod | 1970 | <https://fliphtml5.com/ezwcd/ozvp/Marathi_of_Kasargod_%281%29/> | Obtained from complete public mirror; reconstructed as a 176-page PDF |
| Gawdi of Goa | 1972 | <https://msblc.maharashtra.gov.in/pdf/newpdf/Gawadi%20of%20Goa.pdf> | Awaiting publisher/archive recovery |
| Bhili of Dangs | 1976 | No verified publisher URL found | Awaiting a scan |

The reconstructed Kasargod PDF has SHA-256
`aa79dc5082f19c49c5147b29e3e3da1e7c481fda0f90a7e5b69ca532402815e8`.
Its vocabulary is on PDF pages 144--176, corresponding to printed pages
136--168.  The public FlipHTML5 text layer is not used as lexical source data
because it loses column structure and phonetic characters; the importer OCRs
the rendered page images and preserves confidence and review flags in a
separate audit CSV.

The canonical ingestion artifacts are:

- importer: `ghatage_survey.py`;
- installed rich-form input: `../20260817-ghatage-marati-kasargod.csv`;
- complete OCR/decision audit: `20260817-ghatage-marati-kasargod-audit.csv`;
- source-image-backed correction layer: `20260817-ghatage-marati-kasargod-corrections.csv`;
- deterministic seed-1970 review sample: `20260817-ghatage-marati-kasargod-sample.csv`; and
- source-specific sound profile: `../../../../conversion/ghatage.txt`.

The importer records exact printed-page and within-page entry locators. Its correction layer
contains 129 unique verified entry keys, including three rows recovered directly from the page
images after OCR omission. The final 20-record review sample passes 20/20, while preserving that
14 of those records had a material error before source-image correction. The remaining unreviewed
OCR transcriptions stay explicitly tagged `ocr-review`; structural admission is not a claim of
letter-perfect human verification.
