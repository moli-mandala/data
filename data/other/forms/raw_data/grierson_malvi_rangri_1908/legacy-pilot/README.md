# LSI Mālvī (Rāngrī), bounded comparative-table column

George A. Grierson, *Linguistic Survey of India*, volume IX, part II,
*Specimens of the Rājasthānī and Gujarātī* (Calcutta, 1908), printed p. 307,
“List of Standard Words and Sentences in Rājasthānī.” The original edition is
public domain; the [Wikimedia Commons scan](https://commons.wikimedia.org/wiki/File:Linguistic_Survey_of_India_Vol_9_Part_2.djvu)
is marked Public Domain Mark 1.0. The ignored local scan cache
`tmp/pdfs/lsi-v9-2/LSI-V9-2.djvu` has 494 pages, is 992×1404 pixels per page,
and has SHA256 `d6796bad8d267b776d2aec40dabb0b3b74d4f4fc87eb118f81db1b88e9dc6049`.
Printed p. 307 is DjVu page 322 (one-based). The Internet Archive's original
JP2 page image was also checked and is **the same 992×1404 resolution**; it did
not resolve the small diacritics left uncertain in this edition. The archive
lists a separate image-container PDF, but the original JP2 was used for the
resolution check. The page was manually inspected; no OCR output is installed.

The bounded scope is **all 21 cells** in the Mālvī (Rāngrī) column for numbered
lexical prompts 32–52 on p. 307. `transcription.tsv` and `audit.jsonl` have one
row per target cell, each keyed by page, named column, and printed prompt
number. The adjacent “Mālvī (when different from Rāngrī)” column is separately
audited as a control; its answers are excluded rather than substituted for
Rāngrī. The Nīmāṛī, Bagri, and English control columns, pronouns 26–31, other
numbered sections and other pages are outside this installation. Held visual
readings are provisional descriptions, not asserted transcriptions.

LSI p. 52 identifies Rāngrī as the form of Mālvī spoken by Rajputs of Malwa
proper, and p. 240 introduces separate Standard Mālvī and Rāngrī specimens
from Dewas State. Glottolog `malv1243` identifies Malvi, matching Jambu's
existing canonical `Malw` despite its legacy display name “Malwai.” The
named Rāngrī lect is registered as a source-qualified dialect below `Malw`.
No exact survey site is given, so dialect coordinates are blank. Existing
compiled `Malw` rows have only “evening” and “threshold, porch”; neither
duplicates the five installed prompt meanings.

The page yields **five secure cells and five rows** (hand, foot, nose, eye,
iron). Sixteen other target cells are held because length, nasalization,
underdots, or multiple-answer boundaries cannot be read with confidence at
the scan's resolution. The printed table asserts no etymological, borrowing,
or variant-direction analysis and no POS labels for these rows; these fields
are blank. The literal profile only lowercases table-initial capitals for
display and retains secure macrons. Source spelling remains in `Original`
after compilation. Five accepted forms have complete profile coverage.

From the Jambu data root, regenerate with
`.venv/bin/python data/other/forms/raw_data/grierson_malvi_rangri_1908/import_source.py --check-scan --install`.
The importer also previews counts without `--install` and fails clearly if the
non-committed scan is missing or changed when `--check-scan` is requested.
`sample-review-20.tsv` records a fresh seeded 20-cell raw-to-decision review;
all 21 target cells and all corresponding Standard Mālvī control cells were
visually checked, with zero material installed-reading errors and no known
systematic accepted-glyph error class. `tests/test_grierson_malvi_rangri_1908.py`
covers regeneration, counts, keys, locators, mapping, profile coverage, and
local parsing. The complete CLDF build, graph, references build, full test
suite, and browser QA remain deferred under the user's explicit no-build
instruction.
