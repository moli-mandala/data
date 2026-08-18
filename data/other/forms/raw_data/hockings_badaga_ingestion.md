# Hockings & Pilot-Raichoor 1992 Badaga ingestion

## Scope and source identity

The source is Paul Hockings and Christiane Pilot-Raichoor, *A
Badaga-English Dictionary* (Mouton de Gruyter, 1992), DOI
[10.1515/9783110846058](https://doi.org/10.1515/9783110846058). The source PDF
is copyrighted and is not redistributed. Jambu contains extracted lexical
facts plus exact scan locators.

The dictionary/glossary, PDF, and OCR-heavy checklist addenda apply. The
ingested scope is the Badaga-English dictionary and incorporated gazetteer,
printed pp. 1-621 (PDF pp. 21-643). PDF pp. 443-444 are two inserted blank scan
leaves between printed pp. 422-423 and are excluded from pagination. Printed
p. 1 is a title and p. 2 is blank; lexical articles begin on printed p. 3.

Explicit exclusions are the front matter, blank leaves, English-Badaga reverse
glossary (a duplicate lookup index rather than independent attestations),
appendices, references, and publisher advertisement. A much expanded later
edition is described publicly, but no complete inspectable copy was available
and it is not silently combined with the supplied 1992 edition.

## Reproducible extraction

`hockings_badaga.py` renders the included pages at 300 dpi with pypdfium2 and
runs `tesseract -l script/Latin --psm 3 tsv`. Page JSON, including word
coordinates and confidence, is cached under
`.cache/ocr/hockings-badaga/pages/`. Extraction uses the printed two-column
geometry: dictionary heads begin at the outer head indentation and continuation
lines at a distinct inset. Every audit article retains PDF page, printed page,
column, vertical position, raw OCR, raw head, confidence, parsed fields, source
citation, DEDR tokens, and a stable `p…:c…:y…` source key.

Reproduction command (with the PDF path adjusted locally):

```sh
uv run python data/other/forms/raw_data/hockings_badaga.py \
  "$JAMBU_HOCKINGS_BADAGA_PDF" --workers 4 --install
```

The supplied hidden OCR layer was evaluated as an alternate representation.
It preserves reading order imperfectly, fuses columns on some pages, and loses
contrastive retroflex dots. A 20-entry, scan-backed calibration is checked in
at `20260818-hockings-badaga-calibration.csv`: the structural parser passed
20/20, the embedded layer differed from the scan on 9 heads, and the fresh pass
differed on 4. The four fresh-OCR errors were corrected in the durable overlay;
all 20 final calibration results pass.

## Transcription and editorial policy

The source distinguishes five short and five long vowels, marks length with a
colon, and uses Dravidianist underdots for retroflex consonants. Original OCR
forms are preserved in CLDF `Original`; the source-specific sound profile
changes only vowel-length colons to display macrons. Unambiguous OCR glyph
substitutions such as cedilla `ļ` for printed `ḷ` are mechanical. Plain
`t/d/n/l` is never guessed to be retroflex when the dot has disappeared.

The 20 visually reviewed articles lose the review marker after an accepted or
corrected overlay decision. The remaining 9,973 articles are installed only
under the checklist's standing provisional-OCR policy: every emitted row has
the typed `uncertain` tag and its audit row has
`Review_State=needs_transcription_review`. The manifest in
`data/ocr-postcorrection.json` exposes the exact PDF crop for each stable key at
`/dev/ocr`; stale overlay decisions are rejected by an audit fingerprint.

Slash-separated forms are emitted as source-attested alternates with stable
variant relationships. Homographs remain distinct because their source keys
participate in compilation deduplication. POS labels, definitions, usage
notes, source analyses, and explicit DEDR citations are retained separately.
No fuzzy or form-similarity etymological links are created.

## Language, dialect, and graph decisions

All rows use the existing canonical `Badaga` language record (Glottocode
`bada1257`). The authors state that forms were collected throughout the
Nilgiris and may belong to any Badaga dialect, while the book does not
systematically identify which one. Consequently no per-entry dialect tag or
new dialect registry row is inferred. Terms such as *Gauda* in gazetteer prose
are social/community descriptions, not safe entry-level dialect assignments.

An article is linked to every valid, explicitly printed DEDR target. The audit
keeps 93 articles with unresolved printed citations, mostly `DEDR App.` items
or digit strings whose scan does not support a unique valid target; these are
preserved in source etymology prose and deliberately left unlinked. Articles
without an explicit valid DEDR citation remain first-class unlinked lexical
entries.

## Extraction accounting

- Raw dictionary articles: 9,993
- Installed rich rows: 16,706
- Structurally ingested articles: 9,993
- Corrupt/excluded lexical articles: 0
- Articles with at least one valid explicit DEDR link: 6,421
- Articles without a valid explicit DEDR link: 3,572
- Articles with printed alternates: 2,639
- Articles with unresolved printed DEDR citations: 93
- Visually accepted/corrected articles: 20 (16 accepted, 4 corrected)
- Remaining typed transcription-review queue: 9,973

The row count exceeds the article count because a source article may cite
multiple DEDR targets and/or print alternate head forms. Layout-only title and
blank pages are excluded before article accounting.
