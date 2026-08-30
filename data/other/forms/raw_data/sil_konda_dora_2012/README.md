# SIL ESR 2012-016 Konda Dora survey

Source-local package for Frank Blair and Jacob George's *Multilingualism Among
the Konda Dora*. `SOURCE_INGESTION_CHECKLIST.md` is active with both the
survey/comparative-wordlist and OCR-heavy addenda. Appendix 9.5 is an image
scan: the PDF text layer is only a locator, and every accepted cell was
transcribed and checked by hand against its rendered source image.

## Canonical source and scope

- SIL archive: <https://www.sil.org/resources/archives/49120>
- Publisher PDF: `silesr2012_016.pdf`, 31,978,201 bytes, 106 physical pages
- PDF SHA-256: `6e0a3e5522a45752938f8279753d07b4e29d7b76ca73e88f71c4e283dfd0f533`
- Report created 1987 and published as SIL Electronic Survey Reports 2012-016
- Included: Appendix 9.5, physical PDF pp. 88–106, printed pp. 83–101
- Lexical tables: physical PDF pp. 89–106, 214 prompts × four lists = 856 cells

The source prints 1–210, then supplemental 211 `chest`, 212 `liver`, a second
212 `foot`, and 213 `which?`. The duplicate number is retained as distinct
`212-liver` and `212-foot` prompt keys. The PDF says “All rights reserved”; it
is cached outside the data repository and is not redistributed here. Accessed
2026-08-28.

## Lists and installation scope

Two target lists are installed under the existing Jambu base language `Konda`
(`kond1295`): Koraput Konda/Kubi, recorded at Pansawalsa in February 1987, and
Visakh Konda/Kubi, recorded at Lakshmipuram in January 1987. Telugu and Adivasi
Oriya (Kotia Oriya) are comparison controls. Their 428 cells are manually
transcribed and retained in the audit but never installed.

The source's leading digits are lexical-similarity group labels, not historical
cognacy claims. They remain in the audit and installation notes only. A leading
source hyphen indicates that no group was assigned and is likewise structural.
It is not part of the form. No cognate, etymology, borrowing, derivation, or
variant graph edges are inferred.

## Manual transcription and review

Physical pages 89–106 were rendered at 220 dpi. All 856 cells were visually
inspected and manually typed into `reviewed_transcription.psv`: 428 target
cells and 428 control cells. The ledger stores the physical and printed page,
prompt identity, four exact source cells, review status, confidence, and note.
There are 727 attested cells and 129 confirmed `----` blanks (43 target and 86
control). No cell is clipped, illegible, ambiguous, or unresolved.

`ocr_scaffold.txt` is a reproducible extraction of the scan's highly unreliable
text layer. It is explicitly non-authoritative and never feeds accepted forms.
It is checked in only to make OCR disagreements reviewable. The accepted ledger
was typed and verified from the rendered page images, not copied from OCR.

Typewriter underdots are represented with NFC Unicode retroflex letters where
available. The visibly distinct source IPA retroflex flap is retained as `ɽ`.
Colons and literal question-mark glyphs are preserved in diplomatic
`Form`/`Phonemic`; the source-local conversion profile maps them to length `ː`
and glottal stop `ʔ`, while `ɽ` maps to Jambu display `ṛ`. This is a
conversion-layer interpretation, not a silent
change to the source transcription.

## Source-defined expansions and counts

For items 182–201 the source explicitly orders slash slots as third-person
past, imperative, and infinitive. Each nonblank slot is installed separately.
Slashes elsewhere represent separately elicited lexical alternatives and are
also expanded. Empty `----` slots remain explicit in the cell audit.

The importer emits 452 Konda forms: 231 from the Koraput list and 221 from the
Visakh list. These arise from 385 attested target cells after source-defined
expansion. The audit has exactly 856 rows; 342 attested control cells are
excluded, as are all 129 confirmed blanks.

## Reproduction

With the canonical PDF at the outer workspace path
`tmp/pdfs/konda_dora/silesr2012_016.pdf`:

```sh
python data/other/forms/raw_data/sil_konda_dora_2012/extract_scaffold.py
python data/other/forms/raw_data/sil_konda_dora_2012/import_konda_dora.py --install
UV_CACHE_DIR=/tmp/uv-cache uv run --with pytest --with segments --with pypdf \
  pytest -q tests/test_sil_konda_dora_2012.py
```

The source package deliberately defers shared bibliography, dialect registry,
profile routing, consolidated build, full test suite, generated CLDF, and
browser refresh to the coordinating task. Exact changes and expected counts
are in `INTEGRATION.md`.
