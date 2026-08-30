# JLSR 2024-011 Haryanvi wordlists

This package installs the six Haryanvi lists in Jeff Webster's *Sociolinguistic Survey of
Haryanvi* (Journal of Language Survey Reports 2024-011).  The official 89-page publisher PDF has
SHA-256 `53121a1b9803ba502092866080e3bdb35457bc6040dcc7f47da508eca1fef2e2`.
The included source is Appendix A.3, printed pages 21-34 (PDF pages 28-41): 210 prompts by ten
lists, or 2,100 source cells.

## Scope

The six target lists are HRT Rohtak, HJN Jind, HFT Fatehabad, HNG Dehar/Narayangarh, HTR Taoru,
and HLH Loharu.  They attach to Jambu's existing canonical Haryanvi language `kaithal` and to six
source-locality dialect records.  The 840 BPL Braj/Haryanvi, PBG Baghati Pahari, HIN Hindustani,
and PUN Punjabi comparison cells are audit-only.  Appendix A.2 omits metadata blocks for HJN, HTR,
and HLH; their names and localities come from table 1, and no speaker, date, or double-check status
is invented.  The report supplies a regional map but no source point coordinates, so all six
dialect coordinates remain blank.

The numbers preceding responses are the report's phonetic-similarity groupings.  Section 4.1.1
explicitly says these are based on phonological similarity, not cognate relationship.  They are
retained in Notes and the audit and never become `Parameter_ID`, `Cognateset`, or etymology edges.

## Manual transcription and OCR policy

Appendix A.3 is raster-only.  `extract_ocr.py`, the three Tesseract evidence files, and
`build_scaffold.py` reproducibly recover page/item/list structure, but OCR is non-authoritative.
`manual_transcription.tsv` is the frozen cell-by-cell human transcription made by inspecting the
enlarged source crops.  All 1,260 target cells were manually inspected: 1,231 contain direct
responses, seven are source cross-references, 21 are visibly blank, and one gives only an
elicitation instruction.  The direct cells expand to 1,546 printed alternatives; resolving the
seven same-list cross-references yields 1,553 installed rows.  No installed form originates from
OCR.  The excluded comparison cells were not manually transcribed; their exact locators and raw
OCR evidence are retained only so all 2,100 source cells remain auditable.

The diplomatic transcription preserves the source's Unicode IPA distinctions, including dental
underbars, retroflex underdots, superscript aspiration/labialization, nasalization, and the printed
middle-dot length mark.  A dedicated sound profile converts only the display layer to Jambu's
house transcription; `Original` and `Phonemic` retain the manual source transcription.  Slash
alternatives become stable child rows.  Parenthetical source qualifiers are separated from forms.
Cross-references are resolved within the same list and retain the referring prompt's own stable
entry keys.

## Known source defects and uncertainty

The manual ledger retains typed review notes for faint, wrapped, unlabelled, and clipped source
material.  In particular, item 64 repeats the HJN label where the invariant row is HFT; item 112
has an unlabelled `logaji` line; items 114, 177, 183, and 186 contain clipped additional HNG
responses; and item 135 has clipped additional HRT, HTR, and HLH responses.  Other faint or clipped
fragments are recorded in the per-cell audit.  No unreadable fragment is guessed into an installed
form.

Run `python data/other/forms/raw_data/sil_haryanvi_2024/import_haryanvi.py` from the `data`
repository to regenerate the installed CSV, complete 2,100-cell audit, and manifest.
