# Page extraction contract: Burrow's Dravidian papers (1938–1948)

## Source and scope

- Physical source: T. Burrow, *Collected Papers on Dravidian Linguistics*
  (Annamalai University, Department of Linguistics, Publication No. 13, 1968).
  Open scan: <https://archive.org/details/tdl.24130-collected-papers-on-dravidian-linguistics>
  (Tamil Virtual Academy). The reprint reproduces the articles unrevised with
  new pagination.
- The nine articles in scope, with their reprint page ranges and source keys:

  | key | article | original | reprint pp. |
  |---|---|---|---|
  | `burrow1938ds1` | Dravidian Studies I (convertibility of surds and sonants) | BSOS 9.3: 711–722 | 1–17 |
  | `burrow1940ds2` | Dravidian Studies II (short o/e with i/u) | BSOAS 10.2: 289–297 | 18–31 |
  | `burrow1943ds3` | Dravidian Studies III (two developments of initial k) | BSOAS 11.1: 122–139 | 32–64 |
  | `burrow1944ds4` | Dravidian Studies IV (the body in Dravidian and Uralian) | BSOAS 11.2: 328–356 | 65–112 |
  | `burrow1945ds5` | Dravidian Studies V (initial y and ñ) | BSOAS 11.3: 595–616 | 113–149 |
  | `burrow1947ds6` | Dravidian Studies VI (loss of initial c/s) | BSOAS 12.1: 132–147 | 150–177 |
  | `burrow1948ds7` | Dravidian Studies VII (further Dravidian words in Sanskrit) | BSOAS 12.2: 365–396 | 178–235 |
  | `burrow1945tps` | Some Dravidian words in Sanskrit | TPS 44: 79–120 | 236–284 |
  | `burrow1946tps` | Loanwords in Sanskrit | TPS 45: 1–30 | 285–318 |

- Reprint page = scan leaf − 7 (`leaf-243.png` is reprint p. 236). Reprint
  p. 1 is absent from the scan.
- The goal is to structure every **Indo-Aryan ↔ Dravidian** lexical claim the
  articles make, with the forms and wording that support it. Do not summarize
  the phonological argument and do not infer a loan from resemblance.

## Inputs

For each assigned leaf N, inspect both files:

- `tmp/pdfs/burrow-collected-1968/pages/leaf-NNN.png`
- `tmp/pdfs/burrow-collected-1968/pages/leaf-NNN.txt`

The image is authoritative. The text layer is Tesseract output and is only a
navigation aid: it drops or corrupts macrons, underdots, `ṟ`, `ṉ`, `ḻ`, `ñ`,
`ṅ`, `ś`, `ṣ`, `ṛ`, Brahui special letters, italics, and superscripts. Check
every form against the image; zoom when a diacritic is doubtful.

## Units of extraction

Return one JSON object for exactly one leaf. Extract:

1. **`item`** — in `burrow1948ds7`, `burrow1945tps`, and `burrow1946tps`, each
   numbered entry (`1`, `2`, … printed as a centred or leading number). One
   record per numbered item, even when several Sanskrit words are treated
   together. Keep the item together: Sanskrit head-word(s) with gloss and
   attestation marks, other Indo-Aryan forms, all Dravidian comparanda, any
   Munda/other forms, and Burrow's conclusion.
2. **`incidental_claim`** — anywhere (including the introductions of the three
   loanword papers and the whole of `burrow1938ds1`–`burrow1947ds6`), an
   explicit statement that a specific Indo-Aryan word (Sanskrit, Pali,
   Prakrit, NIA) is borrowed from, borrowed into, or otherwise historically
   connected with a specific Dravidian word. Typical wordings: "loanword from
   Skt.", "of Dravidian origin", "borrowed from Indo-Aryan", "Skt. X is from
   this root", "cf. Skt. X", "not connected with Skt. X". A bare list of
   Tamil loanwords from Sanskrit is one record per listed pair.
3. **`etymon_note`** — a discussion inside an item that separately attributes
   a *second* Sanskrit word to the same or a different Dravidian source (for
   example "Similarly Skt. Y is from Ta. …"). Prefer keeping it in the same
   `item` record as an additional `ia_forms` element unless Burrow treats it
   as a distinct etymology; then make it its own record with
   `parent_item` set.

Do not create records for purely intra-Dravidian comparisons, Uralian
comparisons, or general prose. Uralian and Munda forms cited inside an item go
in `other_forms`.

## Required JSON shape

```json
{
  "leaf": 243,
  "reprint_page": 236,
  "article": "burrow1945tps",
  "page_kind": "items",
  "records": [
    {
      "unit_id": "burrow1945tps:p236:u01",
      "record_type": "item",
      "item_number": "1",
      "parent_item": "",
      "continues_from_previous_page": false,
      "continues_to_next_page": true,
      "ia_forms": [
        {"language": "Skt.", "form": "aṭṭa-", "gloss": "tower or strong point on a wall serving for defence", "grammatical": "m.", "attestation": "B.R.", "notes": ""},
        {"language": "Skt.", "form": "aṭṭaka-", "gloss": "", "grammatical": "", "attestation": "", "notes": "same sense"}
      ],
      "dravidian_forms": [
        {"language": "Te.", "form": "aṭṭaḍi", "gloss": "a fortified place in front of a building", "notes": ""},
        {"language": "Ka.", "form": "aḍḍal", "gloss": "obstacle, hindrance", "notes": ""}
      ],
      "other_forms": [],
      "claim": {
        "relation": "ia_from_dravidian",
        "confidence": "high",
        "wording": "Compare Te. aṭṭaḍi … These words are members of a numerous Dravidian series",
        "dravidian_etymon": "series aṭ-, aṭṭ-, aḍ-, aḍḍ- 'obstacle, obstruction; athwart; to obstruct, close'"
      },
      "printed_references": ["B.R."],
      "prose": "Skt. aṭṭa- m. according to B. R. has the meaning of a tower … (full diplomatic transcription of the item as printed on this leaf)",
      "extractor_notes": ""
    }
  ],
  "page_notes": "",
  "unresolved_glyphs": []
}
```

Controlled values:

- `page_kind`: `front_matter`, `prose`, or `items`.
- `record_type`: `item`, `incidental_claim`, or `etymon_note`.
- `claim.relation`: `ia_from_dravidian` (Burrow derives the IA word from
  Dravidian), `dravidian_from_ia` (the Dravidian word is a loan from IA),
  `related_undetermined` (connection asserted, direction not decided),
  `rejected` (Burrow denies a previously proposed connection), `non_dravidian`
  (Burrow assigns the IA word to Munda or another non-Dravidian source), or
  `none` (no IA↔Dravidian claim on this record).
- `claim.confidence`: `high` for an unqualified derivation ("is from", "is a
  Dravidian word", "clearly"); `medium` for "probably", "no doubt … but",
  "may be regarded"; `low` for "possibly", "perhaps", "it is tempting", "?" or
  when Burrow himself signals doubt.
- `language` uses Burrow's printed abbreviations exactly (`Skt.`, `Pa.`,
  `Pkt.`, `H.`, `M.`, `Ta.`, `Ma.`, `Ka.`, `Te.`, `Tu.`, `Kod.`, `Kol.`, `Go.`,
  `Kui`, `Kur.`, `Malt.`, `Brah.`, `Sant.`, etc.). Do not expand them.

## Transcription policy

- `form` is the printed form character-for-character: keep hyphens, macrons,
  underdots, `ṟ ṉ ḻ ñ ṅ ś ṣ ṛ ḥ ṃ`, Brahui `ṛ ḷ x γ`, and Burrow's parenthesised
  alternants such as `g(r)ukku`. Use NFC Unicode. Never resolve an
  ambiguity by picking the likelier letter: put `?` in `notes` and list the
  glyph in `unresolved_glyphs`.
- `gloss` is the meaning only, without the abbreviation of the source
  dictionary or grammatical labels; those go to `attestation` and
  `grammatical`.
- `attestation` records source-work marks exactly as printed (`lex.`, `MBh.`,
  `B.R.`, `RV.`, `Suśr.`, `Tamil Lex.`, `Kittel`, `Bray`, etc.).
- `prose` is the full transcription of the record's text on this leaf. Keep
  paragraph breaks as `\n`. Transcribe footnotes belonging to the record at
  the end, prefixed `[fn N]`.
- When an item continues from the previous leaf, set
  `continues_from_previous_page=true`, repeat the `item_number` if it can be
  inferred from the immediately adjacent leaf text, and extract only what is
  physically on this leaf. Never join across leaves.
- Preserve the order of records as printed.

## What not to do

- Do not modernize to DEDR/CDIAL numbers; the articles predate both.
- Do not add comparanda or glosses that are not printed.
- Do not decide a relation from phonetic similarity; only Burrow's wording
  decides `claim.relation` and `claim.confidence`.
- Do not skip a leaf because it looks like continuous prose; `burrow1938ds1`–
  `burrow1947ds6` contain their claims inside prose.
