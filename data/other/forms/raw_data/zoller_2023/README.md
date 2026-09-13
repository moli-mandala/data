# Zoller 2023 — Linguistic data I–III

Claus Peter Zoller, *Indo-Aryan and the Linguistic History and Prehistory of North India*.
Wiesbaden: Harrassowitz, 2023 (Neuindische Studien 20). Print ISBN 9783447120142;
e-book ISBN 9783447393812. Canonical source key: `zoller2023`.

The input is the user's `Documents/Linguistics/Indo-European/Indo-Aryan/Zoller.pdf`.
Its SHA-256 and page assumptions are pinned in `source.json` and asserted by the
extractor. The full copyrighted PDF is not redistributed. This package preserves
extracted linguistic facts and the context needed to audit their interpretation.
No OCR was used. It is a new supplement, distinct from Zoller's 2005 Indus Kohistani
dictionary already in Jambu.

## Coverage and reconciliation

Printed pp. **519–1035**, PDF pages **548–1064**, inclusive; 517 pages, sections
18.1–18.8 (Linguistic data I, II and III). The source PDF has 1115 pages.
The earlier discussion in chapter 17, indexes and bibliography are outside this import.

- 3,372 numbered main records, 781 footnotes, 224 table-root records from the §18.6 summary table:
  **4,377 source records** in `record-audit.jsonl.gz`.
- 32,470 italic candidates, each accounted for in `audit.jsonl.gz`.
- **17,754 installed form rows**, from 16,586 resolved candidates; commas and slashes
  expand only outside parentheses. 328 canonical languages after the West Pahari split; 133 base-language registrations added. The final rows use
  71 source-owned dialect registrations, alongside existing dialects. Three registrations
  from the draft remain reserved but unused (90 added in total).
- 341 direct CDIAL links (334 reflexes, 7 borrowings), plus 70 explicit same-language
  `or` alternates linked as variants. These are source-attributed claims, not independent
  confirmation of the author's historical proposals.

The remaining candidate statuses are 4,100 reconstructions, 4,345 Sanskrit etymon
comparanda, 4,171 context/unglossed spans, 2,403 unresolved language assignments,
861 generic-family/comparison controls, 3 damaged leading-accent transcriptions and
1 unresolved ditto gloss. These statuses are mutually exclusive candidate decisions;
they are not counts of additional dictionary headwords. Reconstruction and Sanskrit
comparison evidence remain in the per-record source analysis rather than becoming
unlinked attestations. Unresolved cases retain their exact record, page and typography.
The requested section's resolved comparison languages are included, not only IA/Dravidian.
Generic `Kol`, `Kor.` (Korwa/Korku), `Koh.`, mixed `Brj.-Aw.`, bare Pahari and source
`Semnan` remain unresolved. In this source the Southeast Asian `Kui` comparison label
maps to Kuay, not the Dravidian Kui survey previously excluded by the user.

## Extraction and transcription

PyMuPDF glyph traces, font IDs, positions and styles recover the TeX Type3/TIPA fonts.
Ordinary PDF text extraction corrupts their phonetic symbols. The source has heterogeneous
transcriptions; the dedicated `zoller-2023` preservation profile makes **no blanket IPA
or phonological conversion**. Both Form and Original preserve the decoded source
transcription in NFC. Whitespace is collapsed. Seventeen Greek-script forms also populate Native;
romanized forms have no separately asserted native layer. Phonemic remains blank.

The decoder distinguishes diacritics above/below vowels, retains underdots on vowels
and consonants, and preserves raised letters (e.g. `gʰãːɖ`, `minᵃkī`) and boundaries
(e.g. `-tai`). Immediately postposed small numeric notation is retained as superscripts (e.g.
Mang `cuaŋ⁴`); larger reference counters and preposed indices remain in the positioned
raw tokens. Retaining a numeric mark does not assert that every such mark is a tone. Source homographs have distinct
record keys. Subscript lexical indices remain in source forms. A dot printed above a
boundary hyphen, as in Burushaski `man-̇`, is retained rather than linguistically emended:
those rows have `uncertain` and a typed transcription reason in the audit. Rare font
symbols are preserved diplomatically. Three forms with unattached initial accents
are withheld. Undecoded glyphs in surrounding prose display `[undecoded source glyph]`;
the raw glyph evidence remains in the snapshot/audit. Unbalanced source-quoted punctuation
is flagged with a typed gloss reason, not silently repaired. Line-end hyphens in glosses
are retained; they are not blindly deleted to guess English word boundaries.

## Language and relationship policy

The printed abbreviation table (pp. XIII–XIX) controls explicit mappings. Exact-name
Glottolog candidates were reviewed for label collisions. The dated, checksum-pinned
Glottolog snapshot is included for repeatability; modern language points are expressly
quality-C approximations, not the author's field sites. Source dialects with no reliable
locality coordinates have blank points. Kassite has no defensible point in the snapshot.
By explicit user decision on 2026-09-13, the 15 named West Pahari labels are separate
language records within the `W. Pahari` clade: Bangani, Deogari, Himachali, Khashdhari,
Khashi, Padri, Bauri, Bushahari, Barari, Outer Siraji, Shoracholi, Kotguru, Shimla Siraji,
Kotkhai, and Western West Pahari. The last name preserves the source's broad regional
attribution rather than asserting a narrower identification. The 32 unspecified West
Pahari attestations remain at `WPah`. Both attestations labelled Bangāṇī that previously
mapped to Garhwali now join Bangani. No new coordinate or Glottocode guesses were made.
`west-pahari-languages.json` records the metadata; `west-pahari-migration.json` audits all
2,680 changed rows. Source keys, spelling, glosses, locators and graph claims are unchanged.
Old dialect registry entries remain available for historical links, but these forms
no longer carry the former umbrella-qualified dialect tags. Existing Jambu
base IDs are reused even when their historical IDs look like survey site names.

The table's geographic coverage column is not a list of languages to which every
example should be assigned. Column-aware extraction keeps it separate. Language labels
inside citations cannot override the surrounding attestation. A bare etymon after `<`
does not inherit the preceding reflex language. Shared glosses stop at uncoordinated
new-language forms and closing parenthetical boundaries. Nested quoted phrases remain
within their complete lexical gloss. Historical stages and directional/named varieties
are preserved separately; unsupported stage assignments remain unresolved. Postposed attribution and Burushaski H./N./Y. are handled explicitly.

Only an unqualified numbered CDIAL derivation governing a structurally plain initial
form list receives a direct edge. Qualified, rejected, component and cross-family
comparison analyses remain explicit in record audits and Etymology. Nuristani comparisons
are not automatically made IA inheritance. Printed DEDR citations are retained as
citations without guessing an edge. Explicit same-lect `or` alternates get source-local
variant keys. Other list members are not assumed to be synonyms or variants merely
because they share a gloss. Remaining auxiliary author/year citations and unresolved
relationships are listed per record; only resolved bibliography keys enter Source.
The source's long prose is preserved as analysis, not treated as a chain of ancestry.

## Reproduction

From the data repository:

```sh
UV_CACHE_DIR=/tmp/jambu-uv-cache uv run python data/other/forms/raw_data/zoller_2023.py
UV_CACHE_DIR=/tmp/jambu-uv-cache uv run python data/other/forms/raw_data/zoller_2023.py --install
# Requires the exact original user-supplied PDF and PyMuPDF:
UV_CACHE_DIR=/tmp/jambu-uv-cache uv run --with pymupdf python data/other/forms/raw_data/zoller_2023.py --extract --pdf /path/to/Zoller.pdf
# Fresh visual audit; requires PyMuPDF and Pillow, writes crops/JSON to /tmp:
UV_CACHE_DIR=/tmp/jambu-uv-cache uv run --with pymupdf python data/other/forms/raw_data/zoller_2023/audit_sample.py 933 /path/to/Zoller.pdf
```

Ordinary replay uses checked-in compressed records and reviewed language mappings.
`--remap` creates new proposals for review and cannot be combined with `--install`.
Keys comprise source, section, printed page, source entry/footnote/table row, italic-span
position, lect and child index. Spelling/gloss corrections do not appear in identity keys.
The installed CSV is `data/other/forms/20260913-zoller-linguistic-data.csv`.

Final visual seed **933: 0/20 material errors**. Earlier seeds document fixed classes:
font accents, raised letters, bound hyphens, postposed/citation attribution, shared glosses,
implicit etymons, ambiguous language-name matching, historical stages, directional
varieties, mixed math/TIPA word fragments, tone digits, nested quotations and parenthetical
gloss scope. Claim audit seed **1926: 0/20**
source-claim errors. Source-specific regression tests cover these classes, every installed
form's profile coverage, keys, languages, dialects, citations and compiled graph survival.
See `source_checklists/20260913-zoller-review.md` for final build/test results.
Routine ingestion does not refresh the browser database under the standing checklist.

Final validation: all 12 source-specific tests passed in the 45-test affected-module
run. All generation stages completed, with 17,754 unique source nodes and zero old
IDs lost. The full suite had 33 failures: 31 baseline failures and two fixed-cohort
Nihali test assumptions, subsequently corrected and retested. Clean repository-wide
validation remains open; no ancestry was invented for the five new Nihali forms.
