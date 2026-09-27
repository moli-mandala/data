# Gutob and Gorum supplement to JLSR 2022-004

Source files installed: **441 rows**, from 420 cells (210 prompts each) for
Tikrapada Gutob (gu) and Kinumun Parenga Parja / Gorum (go). This supplements
the existing nine Bonda/Didayi lists, using the same canonical citation key.
Rona Desiya and Oriya remain excluded comparison lists.

Mathew and Chamberlain, *The Bonda and the Didayi from Malkangiri District,
Orissa: A Preliminary Study*, JLSR2022-004, survey1997:
https://www.sil.org/resources/archives/92608. AppendixB printed16–45/PDF21–50.
The manifest pins the reused official PDF and upstream native-text ledger.
No OCR was used. The copyrighted PDF and rendered pages are not installed;
the package contains extracted lexical facts and audit/provenance records.

## Reproduction

`import_source.py --output <scratch>` writes draft.csv, audit.jsonl and counts.
Add `--install` to copy the CSV and settings to the canonical source directory.
It never runs a data or database build. Installed file:
`data/other/forms/20260921-sil-gutob-gorum.csv`.

Source configuration lives in the owning `20260828-sil-bonda-didayi.yaml`:
gu/go select `conversion/sil-gutob-gorum.txt`; gt/re keep their existing profile.
Citation declarations remain unique. Entry-key deduplication preserves prompt
and site distinctions. Similarity codes do not become etymology/variant edges.

## Evidence and editorial decisions

- `source-cells.tsv`: all420 cells, with original numbered prompt/site/page.
- `visual-review.jsonl`: all420 cells visually checked. Disqualified headings
  were checked on full pages; other cells have reproducible crop coordinates.
- `structure-review.jsonl`:78 earlier structural checks, kept separately.
- `draft-audit.jsonl`: exhaustive parser decisions, including eight disqualified
  cells, six unanswered cells, one withheld corrupt salt cell (Gutob83), and six
  redundant same-cell segments. The five repeated-spelling cases are visually
  confirmed in `response-policy.json`; all similarity codes remain in the audit.
- 57 responses under paired grammatical prompts retain the full prompt and
  typed grammatical-scope uncertainty; no response-to-tense assignment is guessed.
  Pronouns carry the attested person/number/gender/inclusivity/register tags.
- `site-metadata.json`: two registered village dialects, sourceTables1/3 evidence,
  blank coordinates and village Glottocodes rather than invented points.
- `profile-review.json`: complete input-symbol inventory and difficult examples.
  Preserve source vowel qualities and unmarked length; explicit length uses
  macrons or consonant doubling. Conventional consonant transliteration is
  explicit. Preserve unusual q,ø,ɕ and rounding marks. Raw Form becomes Original;
  blank Phonemic means the source provides no separate phonemic analysis.
- `output-audit-2026092111.json`: fresh seeded sample of20 actual lightweight
  parser outputs, checked against scan crops; **0/20 material errors**. Two
  sampled rows retain already-recorded grammatical-scope uncertainty.
  Reproduce with `sample_output.py --csv <installed-csv> --seed 2026092111
  --output <sample.json>`. This parses one file and never invokes a full build.

16 focused supplement/original-package checks pass, including all441 converted
outputs, source-spelling preservation, profile routing, dialect registration,
repeat handling and installation equivalence. No new etymological edges are
asserted. The full ingestion checklist and survey/comparative-table addendum
remain applicable. Full compiled data checks, reference regeneration, full
suite and browser/app verification are deferred under the user's no-build
instruction. Do not call this a fully validated compiled ingestion.
