# Numeral column-boundary investigation

## Verified

All 21 records flagged in pass 84 are marked **heuristic** in the original importer audit. The read-only reproduction identifies the mechanism: `split_cells` places surplus tokens in the first of three cells, preserving only one token each for the last two cells.

For the Kohistani row `bi} pànc bi} kom`, it produces `bi} pànc | bi} | kom`. The lexical evidence supports the hypothesis `bi} | pànc bi} | kom`: twenty / five scores / who. This is a boundary shift, not simply an exchange of complete gloss labels.

## Still unresolved

The intended boundaries have not been visually verified against a PDF page. The located Chitral mirror timed out. The plain-text extracts lack coordinates, so they cannot establish layout independently. No source forms, glosses, importer code, or etymology assignments were changed in this investigation.

The exact source records, raw lines, reproduced cells, and input hashes are in [the evidence file](numeral-boundary-investigation-20260915.json). The likely repair needs original-layout confirmation and the source-ingestion workflow before source changes or rebuilding.
