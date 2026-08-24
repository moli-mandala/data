# Page extraction contract: Burrow & Emeneau 1972 DEN, part I

## Source and scope

- Source: T. Burrow and M. B. Emeneau, “Dravidian Etymological Notes:
  Supplement to DED, DEDS, and DBIA, Pt. I”, *Journal of the American
  Oriental Society* 92(3), 1972, pp. 397–418.
- Stable record: <https://www.jstor.org/stable/600566>
- DOI: <https://doi.org/10.2307/600566>
- The JSTOR PDF has one wrapper page followed by 22 article pages. PDF page 2
  is printed page 397; PDF page 23 is printed page 418.
- Printed pp. 397–398 contain the introduction, abbreviations, and most of the
  bibliography. Printed p. 399 finishes the bibliography and acknowledgements
  in the left column, then begins numbered lexical entries in the lower right
  column. Printed pp. 400–418 continue the numbered supplement.
- The goal is a complete, auditable representation of the paper's numbered
  DED/DEDS/DBIA additions and corrections. Do not modernize them to DEDR or
  infer an etymology from resemblance.

## Inputs

For each assigned PDF page N, inspect both files:

- `tmp/pdfs/burrow-den-1972-pt1/page-NN.png`
- `tmp/pdfs/burrow-den-1972-pt1/page-NN.txt`

The image is authoritative. The text layer is only a navigation aid: it
interleaves the two columns, loses italic/roman distinctions, and corrupts
retroflexion, vowel length, superscripts, underdots, apostrophes, and special
letters. Check every form and entry number against the image.

## Page and entry boundaries

- Return one JSON object for exactly one printed page.
- Set `page_kind` to `front_matter`, `bibliography`, or `lexical_entries`.
- On pp. 397–398, return zero lexical entries and summarize relevant source
  conventions, abbreviations, or scope in `page_notes`.
- On p. 399, set `page_kind=lexical_entries`, summarize the nonlexical material
  in `page_notes`, and extract every numbered entry segment after the
  “ADDITIONS AND CORRECTIONS” heading in the lower right column.
- On pp. 400–418, extract every numbered entry segment in both columns. The
  printed entry label is the primary boundary: examples are `8`, `451(a)`,
  `723(a)`, `S166`, and `S844`.
- If a numbered entry begins on the preceding page or continues onto the next,
  keep the physically present segment, set the relevant continuation flag, and
  identify the entry label if visible or inferable from the immediately
  adjacent page text. Never silently join across pages.
- A numbered entry may contain several language forms, corrections, deletions,
  cross-references, and comparison notes. Keep them together under one entry
  object, with one nested object per lexical form or explicitly corrected/deleted
  form.

## Required JSON shape

```json
{
  "pdf_page": 5,
  "printed_page": 400,
  "page_kind": "lexical_entries",
  "records": [
    {
      "unit_id": "p400:u001",
      "entry_label": "8",
      "series": "DED",
      "continues_from_previous_page": false,
      "continues_to_next_page": false,
      "operations": ["add_forms"],
      "forms": [
        {
          "language_abbrev": "Ma.",
          "language_name": "Malayalam",
          "dialect_or_source_label": "",
          "form_original": "aka-ppettuka",
          "gloss": "to catch, be caught, befall",
          "grammatical_information": "verb",
          "form_status": "active",
          "relation_to_entry": "reflex",
          "borrowing_source": "",
          "source_detail": ""
        }
      ],
      "links": [
        {
          "target_system": "DED",
          "target_id": "8",
          "relation": "entry_membership",
          "claim_status": "accepted",
          "editorial_action": "add"
        }
      ],
      "comparison_or_correction_text": "",
      "raw_entry_text": "",
      "source_locator": "burrow-emeneau1972den1[p. 400, entry 8]",
      "uncertainty": "",
      "extractor_notes": ""
    }
  ],
  "page_notes": "",
  "unresolved_glyphs": []
}
```

Use exactly these controlled values:

- `page_kind`: `front_matter`, `bibliography`, `lexical_entries`.
- `series`: `DED`, `DEDS`, `DBIA`, or `unknown`. Plain numeric labels are
  normally `DED`; labels beginning `S` are normally `DEDS`. Use `DBIA` only
  where the source explicitly presents a DBIA-numbered revision.
- `operations`: one or more of `add_forms`, `correct_form`, `correct_gloss`,
  `delete_form`, `delete_entry`, `move_or_merge`, `cross_reference`,
  `loan_reanalysis`, `etymological_note`, `new_group`, `source_correction`,
  `no_lexical_change`.
- `form_status`: `active`, `queried`, `corrected`, `deleted`, `comparison_only`,
  `loan`, or `reborrowed`.
- `relation_to_entry`: `reflex`, `borrowed`, `variant`, `derived`,
  `comparison_only`, or `unclear`.
- `target_system`: `DED`, `DEDS`, `DBIA`, `CDIAL`, or `none`.
- `relation`: `entry_membership`, `borrowed`, `variant`, `derived`, `compare`,
  `move`, `delete`, or `none`.
- `claim_status`: `accepted`, `probable`, `suggested`, `queried`, `rejected`,
  or `unresolved`.
- `editorial_action`: `add`, `correct`, `delete`, `move`, `retain`, or
  `context_only`.

## Linguistic and editorial policy

- Preserve the source's exact language abbreviation, dialect/source label,
  form, stem punctuation, alternants, and diacritics. Also supply a canonical
  `language_name` only when the abbreviation is unambiguous from pp. 397–399.
- Split coordinated lexical forms when each has its own language label, form,
  status, or gloss. Keep inflectional stems or parenthesized principal parts in
  a single `form_original` when the source treats them as one lexical item.
- An unqualified form under a numbered DED/DEDS entry is an accepted reflex
  addition. A leading `?`, “probably”, “perhaps”, “or with”, or similar caveat
  must remain queried/probable/suggested.
- Text in square brackets is editorially meaningful. Record explicit
  deletions, corrections, moves, and replacement cross-references rather than
  installing the superseded reading as active.
- A slash-prefixed `/Cf.` or ordinary `Cf.` is comparison evidence, not an
  inherited reflex. Put its forms in `forms` with
  `form_status=comparison_only` and its numbered target in `links`.
- Cross-family material marked `< IA`, `Dr. < IA`, `IA < Dr.`, “re-borrowing”,
  or tied to CDIAL/Turner must be represented as a borrowing/reanalysis claim,
  preserving direction and uncertainty exactly. Do not turn `Cf.` into a loan.
- When the entry says material belongs elsewhere or should be deleted from an
  earlier number, record both the printed source and destination identifiers.
- Keep page-local article wording in `comparison_or_correction_text`; do not
  replace it with a modern DEDR interpretation.
- Do not create independent records for running headers, footers, JSTOR notices,
  bibliography entries, acknowledgements, or generic prose.

## Completeness and evidence

- Unit IDs are page-local and immutable: `p<printed page>:u<three-digit ordinal>`
  in reading order (left column top-to-bottom, then right column top-to-bottom).
- `raw_entry_text` should transcribe the complete numbered entry segment on the
  assigned page, including deletion/correction brackets and comparison notes,
  but excluding running headers and footers.
- Every numbered entry segment must appear exactly once. Do not omit a record
  merely because it only corrects a glyph, deletes a form, or redirects an
  entry.
- Every form-like item inside prose must either be represented in `forms` or
  clearly retained in `comparison_or_correction_text` with a note explaining
  why it is not an independent attestation.
- Report every doubtful glyph, clipped line, uncertain entry label, or column
  boundary in `unresolved_glyphs` with enough context for image review.
- Return valid UTF-8 JSON only, with no Markdown fence or commentary.
