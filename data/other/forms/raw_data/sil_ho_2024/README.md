# SIL Ho survey (JLSR 2024-009)

This source-local package captures Appendix D.3 of Bryan Varenkamp's *A Study
of Ho Dialects*. The source topology is 210 prompts by 27 lists: fourteen new
1989 Ho field lists, three republished Ho controls, and ten non-Ho controls.
Only the fourteen new field lists are eligible for staging.

The source-ingestion checklist is active with the survey-wordlist and OCR-heavy
addenda. The scanned appendix has no authoritative text layer. Every accepted
cell must therefore be hand-keyed from a rendered source page and visually
verified. `ocr_scaffold.tsv` and `ocr_raw/` are non-authoritative evidence only.

## Guarded workflow

`manual_review.tsv` is the immutable 5,670-cell base ledger. Parallel reviewers
may supply disjoint files in `manual_chunks/`, but an admissible chunk must use
the exact OCR-blind schema enforced by `import_ho.py`, include the declaration
`hand-keyed-from-rendered-source; OCR-not-copied`, and target only currently
`unreviewed` base keys. The importer rejects OCR columns, duplicate or unknown
keys, reviewed-base overlap, coordinate drift, non-final statuses, missing
uncertainty notes, non-NFC text, and unapproved manual-review stamps.

Diagnostic validation of the base ledger:

```sh
python3 data/data/other/forms/raw_data/sil_ho_2024/import_ho.py --verify-pdf --base-only
```

Normal validation applies all admissible chunks. Staging remains locked until
all 5,670 cells have a final manual status:

```sh
python3 data/data/other/forms/raw_data/sil_ho_2024/import_ho.py --verify-pdf
python3 data/data/other/forms/raw_data/sil_ho_2024/import_ho.py --verify-pdf --stage
```

Staging never reads `OCR_Evidence_Only`; it consumes only
`Manual_Transcription`. Similarity-group numbers are stripped as source notes,
while the cell's punctuation and comma-separated alternatives remain intact.
Ambiguous and illegible target cells are audited but excluded. Republished Ho
and all non-Ho rows are always audit-only.

The full appendix is now admissibly complete: 5,670 cells reviewed, 5,270
attested, 397 blank, three ambiguous, and no illegible cells. The fourteen
target lists contain 2,900 attested cells, 38 blanks, and two ambiguous cells.
The importer stages the 2,900 attested target response cells source-locally;
the two ambiguous targets are audit-only. Shared installation remains deferred.
