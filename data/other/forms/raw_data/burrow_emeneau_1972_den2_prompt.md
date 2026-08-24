# Page extraction contract: Burrow & Emeneau 1972 DEN, part II

## Source and scope

- Source: T. Burrow and M. B. Emeneau, “Dravidian Etymological Notes:
  Supplement to DED, DEDS, and DBIA, Pt. II”, *Journal of the American
  Oriental Society* 92(4), 1972, pp. 475–491.
- Stable record: <https://www.jstor.org/stable/599958>.
- The JSTOR PDF has one wrapper page followed by 17 article pages. PDF page 2
  is printed page 475; PDF page 18 is printed page 491.
- Printed pp. 475–479 contain the numbered “NEW ENTRIES” supplement. Printed
  pp. 480–491 are indexes only and must yield no lexical records.
- The goal is a complete, auditable representation of every numbered
  DEDS/DED/DBIA addition or correction on pp. 475–479. Do not modernize old
  numbers to DEDR or infer an etymology from resemblance.

## Inputs

For the assigned PDF page N, inspect both:

- `tmp/pdfs/burrow-den2/page-NN.png`
- `tmp/pdfs/burrow-den2/page-NN.txt`

The image is authoritative. The text layer is only a navigation aid and is
known to corrupt length, retroflexion, underdots, special letters, italics,
superscripts, and two-column order. Check every form and entry number against
the image.

## Page and entry boundaries

- Return one JSON object for exactly one printed page.
- On pp. 475–479 use `page_kind=lexical_entries`; extract every physically
  present numbered segment in reading order: left column top-to-bottom, then
  right column top-to-bottom.
- On pp. 480–491 use `page_kind=bibliography`, return `records=[]`, and note
  the index headings/languages in `page_notes`. Index tokens are not new
  lexical attestations.
- The scan prints DEN-part-II new-entry labels as S followed by a superscript 2 and the item
  number (for example S²1, S²9A, and S²37). The raw JSON convention writes these as `S21`,
  `S29A`, and `S237`; they are **not** historical DEDS 21, 29A, or 237. `DBIA S3` is a separate
  DBIA supplement label, while ordinary old DED/DEDS numbers can occur inside correction prose.
  The leading printed label governs `series`: S²... is recorded as `DEDS`; an explicit
  `DBIA ...` label is `DBIA`.
- If a segment begins or ends on an adjacent page, keep only the physically
  present material, set the continuation flags, and do not silently join it.
- Keep all forms, deletions, corrections, redirects, borrowing claims, and
  comparisons belonging to one numbered segment in one record.

## Required JSON shape

```json
{
  "pdf_page": 2,
  "printed_page": 475,
  "page_kind": "lexical_entries",
  "records": [
    {
      "unit_id": "p475:u001",
      "entry_label": "S21",
      "series": "DEDS",
      "continues_from_previous_page": false,
      "continues_to_next_page": false,
      "operations": ["add_forms"],
      "forms": [
        {
          "language_abbrev": "Ta.",
          "language_name": "Tamil",
          "dialect_or_source_label": "",
          "form_original": "accu",
          "gloss": "ridge in a field",
          "grammatical_information": "noun",
          "form_status": "active",
          "relation_to_entry": "reflex",
          "borrowing_source": "",
          "source_detail": ""
        }
      ],
      "links": [
        {
          "target_system": "DEDS",
          "target_id": "S21",
          "relation": "entry_membership",
          "claim_status": "accepted",
          "editorial_action": "add"
        }
      ],
      "comparison_or_correction_text": "",
      "raw_entry_text": "",
      "source_locator": "burrow-emeneau1972den2[p. 475, entry S21]",
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
- `series`: `DED`, `DEDS`, `DBIA`, `unknown`.
- `operations`: `add_forms`, `correct_form`, `correct_gloss`, `delete_form`,
  `delete_entry`, `move_or_merge`, `cross_reference`, `loan_reanalysis`,
  `etymological_note`, `new_group`, `source_correction`, `no_lexical_change`.
- `form_status`: `active`, `queried`, `corrected`, `deleted`,
  `comparison_only`, `loan`, `reborrowed`.
- `relation_to_entry`: `reflex`, `borrowed`, `variant`, `derived`,
  `comparison_only`, `unclear`.
- `target_system`: `DED`, `DEDS`, `DBIA`, `CDIAL`, `none`.
- `relation`: `entry_membership`, `borrowed`, `variant`, `derived`, `compare`,
  `move`, `delete`, `none`.
- `claim_status`: `accepted`, `probable`, `suggested`, `queried`, `rejected`,
  `unresolved`.
- `editorial_action`: `add`, `correct`, `delete`, `move`, `retain`,
  `context_only`.

## Linguistic and editorial policy

- Preserve the printed language abbreviation, dialect/source label, form,
  stem punctuation, alternants, and diacritics. Supply a canonical
  `language_name` only when the abbreviation is unambiguous.
- Split coordinated lexical forms when each has its own language, status, or
  gloss. Keep principal parts together when the paper treats them as one item.
- An unqualified form in a numbered DEDS entry is an accepted reflex addition.
  Preserve `?`, “probably”, “perhaps”, “or with”, and similar caveats as typed
  uncertainty.
- Square-bracketed corrections and deletions are editorial operations. Do not
  install a deleted or superseded reading as active.
- `Cf.` material is comparison evidence, not an inherited reflex. Represent
  its forms as `comparison_only` and its target as a comparison link.
- Material marked `< IA`, `Dr. < IA`, `IA < Dr.`, “re-borrowing”, or tied to
  Turner/CDIAL is a borrowing/reanalysis claim. Preserve direction and
  uncertainty exactly; do not turn an ordinary comparison into a loan.
- Keep page-local wording in `comparison_or_correction_text`; do not replace
  it with a later DEDR analysis.

## Completeness and evidence

- Unit IDs are immutable: `p<printed page>:u<three-digit ordinal>`.
- `raw_entry_text` transcribes the complete numbered segment physically on the
  page, excluding headers, footers, and JSTOR notices.
- Every numbered segment must appear exactly once, including pure corrections,
  deletions, and redirects.
- Every form-like item in prose must either appear in `forms` or be retained in
  comparison text with an explanation for not treating it as an attestation.
- Report every doubtful glyph, clipped line, uncertain entry label, or column
  boundary in `unresolved_glyphs` with enough context for image review.
- Return valid UTF-8 JSON only, with no Markdown fence or commentary.
