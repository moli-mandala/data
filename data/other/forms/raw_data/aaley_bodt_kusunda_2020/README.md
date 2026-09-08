# Aaley--Bodt Kusunda 250-concept wordlist

This package freezes Lexibank `aaleykusunda` v2.1 at commit
`09b1e8d0c19e4f9c69352e4abee500212830f396` (release DOI
`10.5281/zenodo.13149034`, CC BY 4.0). It derives from Uday Raj Aaley and
Timotheus A. Bodt's 2019 elicitation, published in 2020 as *New Kusunda data: A
list of 250 concepts*. The source audio/PDF deposit is DOI
`10.5281/zenodo.3377537` (CC BY-NC-SA 4.0).

## Scope and modeling

All 662 v2.1 CLDF lexemes are installed. The source has 250 prompt records and
three columns of lexical evidence: Gyani Maiya Sen Kusunda, Kamala Khatri (Sen
Kusunda), and the authors' tentative ground-form reconstruction. Speakers are
provenance, not dialects; all rows therefore use canonical `Language_ID`
`Kusunda`. The source's reconstruction rows receive a display asterisk during
the CLDF build while `Original` and `Phonemic` preserve the released unstarred
IPA.

The audit contains exactly 750 rows, one per prompt x source-lect cell. Of
these, 662 map to released forms and 88 are blank or source-marked missing.
Full upstream `Value`, selected `Form`, segments, raw source cells, comments,
speaker mapping, and source-local CLDF IDs are retained. The six form-selection
repairs declared by the release remain frozen in `snapshot/lexemes.tsv`.

The source makes no historical-etymology claims, so every installed row has a
blank `Parameter_ID` and remains an unlinked lexical node. Explicit Nepali loan
labels are structured only when the corresponding selected cell is directly
marked `< NEP`; other uncertain loan or Gorkha-source remarks stay visible in
the audit/notes and carry `uncertain` when installed.

## Transcription

The source uses Unicode IPA with syllable dots, IPA length `ː`, explicit dental
stops, dental versus palatal affricates, palatalization, nasalization, and
creaky voice. `conversion/kusunda-aaley-bodt.txt` removes syllable dots,
converts IPA length to Jambu macrons, maps dental stops to unmarked `t d`, maps
palatal affricates `ʧ ʤ` to `c j`, keeps dental affricates `ʦ ʣ`, maps IPA glide
`j` to `y`, and preserves the source's vowel-quality, uvular, nasalization, and
creaky-voice distinctions. Source IPA remains separately available in both
`Original` and `Phonemic`.

## Reproduction

Preview without installing:

```bash
uv run python data/other/forms/raw_data/aaley_bodt_kusunda_2020/import_kusunda.py
```

Install the dated CSV and checked-in audit:

```bash
uv run python data/other/forms/raw_data/aaley_bodt_kusunda_2020/import_kusunda.py --install
```

The importer verifies every frozen upstream checksum before writing output.
Its seeded 20-row sample (`20260901`) is marked in the audit; all 20 selected
rows were compared across the raw TSV, released CLDF, installed fields, and
rendered source table with zero material errors.

## Inspected but excluded

- Watters (2006), *Notes on Kusunda Grammar*, Appendix A: a separate
  all-rights-reserved vocabulary of about 850 items, requiring its own
  font-decoding and lexical-entry review.
- Aaley (2021), *Kusunda Gipan*: an all-rights-reserved pedagogical book in a
  legacy Nepali font, not a source representation of this 250-concept dataset.

Neither source was silently combined with the open Aaley--Bodt release.
