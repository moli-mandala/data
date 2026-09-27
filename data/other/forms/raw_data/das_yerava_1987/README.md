# Das (1987), The Yeravas of Kodagu: acquisition and reviewed table

Selected to extend sparse Ravula coverage (100 manual-source rows, all DravLex,
at selection). Census year **1981** is not the publication year: the official
PDF metadata gives **1987**, consistent with the foreword dated 19 November
1987. Author B. K. Das, Director of Census Operations, Karnataka; Census of
India 1981, Series 9, Karnataka, Part XI. The exact public official scan has 193
PDF pages and is pinned by SHA256 in `snapshot.py`.

Official download:
https://censusindia.gov.in/nada/index.php/catalog/32907/download/36088/28979_1981_KOD.pdf

An alternative public copy was inspected through web extraction:
https://kodavaclan.com/images/ebook/Census_of_India_1981_yeravas_of_Kodagu1.pdf
It also reports 193 pages; byte identity has not been verified. The official
copy is authoritative for this package. No open redistribution licence has
been established; the PDF remains outside the tracked package. The intended
import is of attributed lexical facts, not a republication of the monograph.

## Scope and evidence

The complete comparative table on printed pp.65–66 (PDF pp.97–98): **42 prompt
groups**, **84 language cells**, **90 comma-separated lexical responses**.
The preceding prose, other ethnographic chapters, and bibliography are not
lexical-list inputs. English is a prompt column, not a control-language form.

`reviewed.tsv` contains all table cells after full-page visual inspection at
170 dpi, with a 300-dpi kinship crop. Item numbers are immutable local physical
table positions, not printed numbers. Raw embedded OCR is preserved unchanged
in `evidence/`; do not rerun OCR unless a new review needs it. Corrected OCR
examples include Avva, Ileya, Ilevu, Mami, Pire, Kodu, Devaru, Male and Kallu.
The source supplies no phonemic transcription or symbol key. The installed
YAML explicitly disables conversion, preserving popular spelling in Form and
Original and leaving Phonemic blank. This is a documented preservation route,
not a claim that spellings such as Curry or Thee are house transcription.

The printed brace on p.65 includes **Mother's Father, Father's Father, and
Mother's Mother** beside Achcha / Chacha. Preserve all three as the source's
grouped gloss rather than silently correcting the apparently surprising third
label. The next row, Father's Mother, is separately Ithawa, Avva / Chachi.
Comma-separated co-equivalents are not automatically phonological variants;
no ancestry, donor, or variant edges are licensed by this table alone. The same
spelling under different prompts must retain distinct source keys.

## Language mapping

- Panjiri Yerava → existing `Ravula` (ravu1237). Glottolog explicitly lists
  Panjiri Yerava and Adiyan as alternative names:
  https://glottolog.org/resource/languoid/id/ravu1237
- Pani Yerava → existing `Paniya` (pani1256). The source identifies the two
  communities respectively with Adiyans and Paniyans (printed p.143); Glottolog
  independently registers Paniya/Paniyan:
  https://glottolog.org/resource/languoid/id/pani1256
- The Kodagu source varieties are registered as `ravula_panjiri_kodagu` and
  `paniya_pani_kodagu` under those base languages. No uniform village or coordinates are
  supplied for the table. Do not reuse a modern language centroid as a survey
  site or treat all Yerava labels as one language.

## Reproduce and remaining gates

From `data/`:

```
.venv/bin/python data/other/forms/raw_data/das_yerava_1987/snapshot.py
```

Default input: `../tmp/pdfs/yerava-census-1981/source.pdf`. The command validates
the complete PDF hash and page count, then preserves 13 evidence pages and
their hashes. It accepts `--pdf` and `--output` and fails if the source differs.

## Installed source inputs; full integration pending

`import_source.py` defaults to a non-installing preview and supports `--install`.
It reproduces 90 rows (48 Ravula, 42 Paniya), an audit of all 84 language cells,
and a seeded acceptance sample. Installation checks the passing audit against
current source/output hashes and rejects bibliography/dialect collisions.
Repeat installation succeeds without duplicate registry entries.

Canonical CSV/YAML: `data/other/forms/20260921-das-yerava.{csv,yaml}`;
bibliography: `das1987yerava`; append order 33. All 90 rows are unlinked, with
zero ancestry, borrowing, derivation or variant edges; six co-equivalents are
expanded without inferring relationships. No source cells are excluded and no
unreadable lexical characters remain. No POS or morphological tags are printed
in the table, so no grammar is inferred from English prompts. Raw glyph/OCR
repairs remain in the audit; the unusual kinship grouping has a source note.

Seed 2026092106: **0/20 material errors** against the rendered scan, covering
forms, gloss scope, lect, page/item and relation decisions. Full-table visual
review additionally covers the first/last rows, comma-separated responses,
multiword forms and the three-label kinship brace. Every parse_file output
retains printed spelling and separate entry keys with zero conversion errors.

Validation: **25 focused source, dialect and sound-profile tests passed**;
source metadata validation passed (198 files, 195 citation keys), and the
bibliography formats successfully through Pybtex.

The compiled-survival test was run and **fails: 0/90 keys in current CLDF**.
Full build, full suite, compiled identity/deduplication, graph, references,
concepts and error/generated-diff checks remain required on an authorized
runner under the laptop resource policy. This is not a completed ingestion.
Browser refresh is unrequested and inapplicable. No commit, push or deployment.
