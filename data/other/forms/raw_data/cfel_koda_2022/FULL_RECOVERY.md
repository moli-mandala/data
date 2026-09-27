# Complete Koda recovery, staged 26 September 2026

The full proposal accounts for all 2,450 physical lexical entries in both 2022
CFEL dictionaries, across 50 domains. English-headed PDF pages 10–371 correspond
to printed pages 7–368; native-headed XPS pages 10–366 correspond to printed pages
7–363. The different ordering and two entries absent from publisher search are
accounted individually. Front matter, illustrations, and English/Hindi/Bangla
control equivalents are excluded from lexical output. No target entry is excluded.

`prepare_full.py` reproducibly generates **3,230 print rows**, with a record for
each paired physical unit in `full-proposal-audit.jsonl`. The companion
`../cfel_koda_api_2026/prepare_full.py` generates **3,223 publisher rows** from the
**2,448 distinct publisher record IDs** recovered for the entire printed scope.
This does not claim an exhaustive crawl of unrelated website content. The two
print-only heads, Taste and Right hand, retain their native-print IPA.

Canonical CSVs now contain the reviewed 3,230 print / 3,223 publisher rows.
All 64 previous pilot keys are retained, with historical pilot snapshots preserved.
The physical-entry inventory, visual decisions, raw headers/descriptions and
publisher record evidence are durable inputs in this package. Original PDF/XPS
files are retained locally, outside installed data. No book images are committed.

## Evidence and transcription

- **English-headed edition:** Koda spellings and all explicit alternatives.
  Header IPA describes English and is never used as Koda pronunciation.
- **Native-headed edition:** Koda spellings, Koda IPA, grammar and English
  descriptions. All 2,450 descriptions were semantically reviewed; 292 concise
  factual notes preserve restrictions, local conditions and discrepancies.
- **Publisher:** literal native spelling, IPA, explicit alternative fields,
  grammar and upstream IDs remain separate attestations. Publisher descriptions
  are retained as raw audit evidence; print descriptions are not attributed to
  the publisher.

The native IPA inventory has 324 exact independent OCR/publisher agreements and
2,126 individually reviewed disagreements. Native spelling has 565 original
disagreements reviewed plus one additional same-class orthographic correction.
All 1,558 English-edition spelling disagreements were visually reviewed; 892
exact agreements complete that edition. Twenty-two grammar OCR discrepancies
were subsequently checked against original native headers. Sixteen description
layout cases and page-12 Sandal/Slipper physical ordering were repaired explicitly.

Print and publisher glyph differences are retained independently. Print review
can distinguish ordinary æ, open ɔ and modifier ʰ from visually confusable API
characters. The API characters are not silently replaced. The unusual dotless-j
glyph represented as յ, question mark and capital glottal-stop symbol remain
literal with typed uncertainty. Dental and retroflex contrasts are preserved in
raw transcription; the established profile converts only defensible house forms.

Native-only alternatives use **source-script Form**, identical to Native, with
blank Phonemic. This is an explicit display-transcription exception, not a claim
that Bengali spelling is IPA. Profiles preserve every Bengali character, including
ZWNJ; ordinary whitespace folding is the only tokenizer change to this layer.
An alternate never inherits a primary head's pronunciation. Ear ring has two
explicitly aligned printed transcriptions. Fried (potato/rice/fish) retains the
parent source expression and all three explicitly stated combinations.

Partial or conflicting IPA for Lingerie, Crest, Ninety-five, Clap and Fishing
festival is retained as source transcription evidence rather than assigned as the
full-head Form. Three native entries explicitly lack IPA. Stable variant keys
point only to the source entry's parent; no etymological claims are inferred.

## Review and checks

Run `python prepare_full.py` and the companion publisher preparation script to
regenerate proposals. Use `--install` to update the canonical source CSV after audit-hash verification.
Neither script builds a database.

An independent sample can be generated with:

```
data/.venv/bin/python data/data/other/forms/raw_data/cfel_koda_2022/sample_full.py \
  --seed INTEGER --output tmp/koda-full-census/independent-sample.json
```

The sample includes full raw/parsed print units, description review, publisher
records and all emitted rows. Rendering uses one overwritten local review sheet.

The full-preparation and existing pilot tests pass (14 tests), including canonical parity and independent-audit hashes.
Source-scoped `parse_file` converts all 6,453 proposed rows without errors or
lost keys. Corpus-wide profile checks preserve native script and every input
symbol. YAML metadata validation passes. Koda reuses the registered canonical
language `Koda` / `koda1236`; no new lect or inferred locality is introduced.

**Independent audit passed:** 20/20, seed 20260926112; both original editions and
publisher evidence checked. Canonical source-stage installation is complete.
Full CLDF build, compiled graph/ID verification,
full suite and browser/database work are deferred under the user's prohibition.
No source-stage result should be described as a full pipeline completion.

Copyright remains with Visva-Bharati. Installed lexical facts and short factual
paraphrases are attributed; full descriptive prose remains source audit evidence.
No public-release decision is made by this staging operation.
