# LSI IX(IV) Suketi column (bounded import)

The source is G. A. Grierson (ed.), *Linguistic Survey of India*, Vol. IX,
Part IV, *Specimens of the Pahari Languages and Gujuri* (Calcutta, 1916),
<https://archive.org/details/LSIV0-V11>. The original edition is public
domain. The original-resolution Internet Archive file `LSI-V9-4.pdf` has
998 pages and SHA256
`ef007663270b3a0ef5ba26804404e4db1281fb5eb239614cadf69b20b3d5395f`.
The scan is an ignored local cache at `tmp/pdfs/lsi-v9-4/LSI-V9-4.pdf`, not a
checked-in source asset. It is optional for CSV regeneration and checked by
`import_source.py --check-pdf` when visual verification is needed.

The precise scope is the Suketi column of the numbered Mandi-group standard
list, prompts 1–13 and 32–79, printed pp. 759–761 (scan pp. 775–777). The
neighboring Mandeali and Mandeali Pahari columns are controls and are not
installed. Pronoun/oblique prompts 14–31, inflected paradigms from 80 onward,
the prose specimens, and all other volumes are outside scope. All 61 target
cells have a transcription and an audit record: 53 accepted cells produce 55
installed rows after two independently listed answers each at brother and
man; five cells are held for uncertain print marks, one sister cell is held
because it mixes citation and explicitly oblique forms, and two cells are
printed ellipses. The importer preserves source prompt number, printed page,
scan page, and source-cell key.

The data come from the original-resolution scan, checked line by line.
Internet Archive OCR and a 20 MB text-layer PDF were inspected only as
comparators; they were too degraded around diacritics to drive the import.
Each retained source character, including macrons and underdots, was reviewed
against the page image. The five cells with unresolved diacritic or final
letter readings remain visible in the audit and out of the CSV. No typographic
italic or underlined contrast was silently flattened. The dedicated sound
profile preserves Grierson's historical Romanization apart from lowercasing
the table's sentence-style initial capital and writing source `w` as house
`v` while preserving `w` in `Original`. It deliberately leaves `ch`,
`chh`, `th`, and `ṭṭh` uninterpreted; none has been assigned a new
phonological value. `Original` retains the input spelling, and `Phonemic` is
blank because the source does not provide a separate IPA analysis.

Suketi uses the existing registered base language ID `suk`. The table labels
only that lect and supplies no narrower locality, consultant, or named
variety, so no new dialect or source-site coordinate is claimed. The current
Jambu registry gives Suketi the same Glottocode as Mandeali; this import does
not revise that classification. The already installed LSI comparative
vocabulary belongs to the same historical survey and is not counted as an
independent field source. The table makes no etymological claim, so all rows
remain unlinked. The two multi-answer cells are not directional variants.

Run `python3 import_source.py --install` from this directory to reproduce the
CSV and JSONL audit. `--audit-sample` uses seed 1916; the checked result is
`sample-review-20.tsv`. The 20 sampled cells were checked against the
original-resolution pages after resolving/holding uncertain readings; all
18 sampled accepted cells matched, one unresolved cell remained held, and
the printed ellipsis matched the audit. There were no material errors in
accepted sampled readings.
Focused checks are in `tests/test_grierson_suketi_1916.py`. Full CLDF and
browser-database builds are deferred under the user's instruction not to build
the database without an explicit request.
