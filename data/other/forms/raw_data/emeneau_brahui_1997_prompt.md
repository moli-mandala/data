# Page extraction contract: Emeneau 1997 Brahui etymologies

## Source and scope

- Source: M. B. Emeneau, “Brahui Etymologies and Phonetic Developments: New
  Items”, *Bulletin of the School of Oriental and African Studies* 60(3), 1997,
  pp. 440–447.
- Stable record: <https://www.jstor.org/stable/619537>
- DOI: <https://doi.org/10.1017/S0041977X00032481>
- The JSTOR PDF has one wrapper page followed by eight article pages. PDF page
  2 is printed page 440; PDF page 9 is printed page 447.
- The goal is to structure lexical and etymological claims, not to summarize
  the article or infer claims from phonetic resemblance.

## Inputs

For each assigned page, inspect both files:

- `tmp/pdfs/emeneau-brahui-1997/page-N.png`
- `tmp/pdfs/emeneau-brahui-1997/page-NN.txt`

The image is authoritative. The PDF text layer is a navigation aid and is known
to lose or corrupt diacritics, especially macrons, underlining, dots, `š`, `ž`,
retroflex marks, and Emeneau's underlined `gh`. Never copy a suspicious glyph
from the text layer without checking the image.

## Unit of extraction

Return one JSON object for the assigned printed page. Extract every source unit
on that page that belongs to at least one of these categories:

1. A Brahui lexical attestation with a form and lexical meaning.
2. An explicit claim that a Brahui form is, may be, or is not a reflex of a
   numbered DEDR/DED/DEDS entry.
3. An explicit borrowing/donor claim involving a Brahui form and a numbered
   CDIAL entry or named donor language.
4. An instruction to add, move, remove, or remove a query from a dictionary
   entry.
5. A derivational, variant, or morphological relationship needed to understand
   one of those claims.
6. A sound development supported by one or more identified Brahui forms.
7. A newly proposed non-Brahui form that the paper explicitly says should be
   added to DEDR. Ordinary comparanda stay inside `supporting_comparanda` and do
   not become independent records.

Do not create independent records for bibliography, generic prose, or every
Tamil/Malayalam/Kurux/etc. comparison. Preserve those forms in
`supporting_comparanda` when they are evidence for an extracted claim.

## Required JSON shape

```json
{
  "pdf_page": 2,
  "printed_page": 440,
  "sections_present": ["1"],
  "records": [
    {
      "unit_id": "p440:s1:u01",
      "record_type": "attestation",
      "section": "1",
      "language": "Brahui",
      "form_original": "",
      "gloss": "",
      "grammatical_information": "",
      "attestation_source": "",
      "target_system": "none",
      "target_id": "",
      "previous_target_system": "none",
      "previous_target_id": "",
      "relation": "none",
      "claim_status": "unresolved",
      "editorial_action": "context_only",
      "derivational_analysis": "",
      "sound_changes": [],
      "supporting_comparanda": [
        {"language": "", "form": "", "gloss": "", "target_id": ""}
      ],
      "evidence": "",
      "source_locator": "emeneau1997brahui[p. 440, §1]",
      "cross_page": false,
      "uncertainty": "",
      "extractor_notes": ""
    }
  ],
  "page_notes": "",
  "unresolved_glyphs": []
}
```

Use exactly these controlled values:

- `record_type`: `attestation`, `reflex_claim`, `loan_claim`, `reassignment`,
  `rejection`, `derivation`, `sound_change`, `dictionary_correction`,
  `unresolved_etymology`, or `explicit_non_brahui_addition`.
- `target_system` and `previous_target_system`: `DEDR`, `CDIAL`, or `none`.
- `relation`: `reflex`, `borrowed`, `variant`, `derived`, `component`,
  `related`, or `none`.
- `claim_status`: `accepted`, `probable`, `suggested`, `queried`, `rejected`,
  or `unresolved`.
- `editorial_action`: `add_form`, `add_citation`, `reassign`, `remove_query`,
  `remove_entry`, `retain_unlinked`, `record_rule`, or `context_only`.

## Reflex and correction policy

- A printed DEDR/DED/DEDS number is an attributed source claim. Record the
  number exactly and separately from the prose.
- “Belongs here”, “is to be added/entered”, and an unqualified equation are
  `accepted`. “Probably” is `probable`; “may”, “tempting”, or “can be
  suggested” is `suggested`; a printed question or “query is required” is
  `queried`; explicit abandonment is `rejected`.
- `remove query in DEDR` is both an accepted reflex claim and
  `editorial_action=remove_query`.
- A move such as DEDR 500 -> 701 is one `reassignment` record with both targets.
  Do not silently delete or overwrite the earlier analysis.
- A source statement that an etymology is unknown remains a first-class
  `unresolved_etymology` record and must not be assigned by similarity.
- Distinguish inherited `reflex` from cross-family `borrowed`.
- If the source says only that a word came from a geographic/areal donor pool,
  record the numbered comparison and the areal caveat; do not invent an
  immediate donor language.
- Preserve source morphology such as stem hyphens, infinitive suffixes, and
  parenthesized variants in `form_original`. Do not normalize to Jambu's house
  transcription during extraction.
- If two senses of an identical spelling receive different analyses, make two
  records with distinct unit IDs.

## Completeness and evidence

- Source units are keyed by printed page + printed section + page-local ordinal;
  never by normalized form or output row number.
- Extract only evidence physically present on the assigned page. If an argument
  continues, set `cross_page=true` and describe what remains incomplete.
- `evidence` should be a short exact phrase or a close, unambiguous paraphrase,
  never a long copied paragraph.
- Report every doubtful glyph in `unresolved_glyphs` with its line/context.
- Return valid UTF-8 JSON only, with no Markdown fence or commentary.
