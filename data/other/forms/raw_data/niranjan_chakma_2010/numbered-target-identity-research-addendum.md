# Target-family identity research addendum

Reviewer: asur_finish, 2026-09-27. Research only; no canonical or registry changes. This supersedes the initial Prakrit objection in `numbered-language-mapping-proposal.json`, while preserving that first proposal as review history.

## Prakrit: 21 numbered slots can use existing `Pk`

The initial inference from the registry's `maha1305` alone was too restrictive. Jambu's actual base name is **Prakrit**, and `data/cdial/abbrevs.py` explicitly defines `Pk` as Prakrit. The dialect registry places Ardhamagadhi, Magadhi, Maharashtri, Paishachi and Shauraseni beneath that base. The general Sheth dictionary and ordinary CDIAL Prakrit entries use it. This agrees with [Turner's primary abbreviation key](https://dsal.uchicago.edu/dictionaries/turner/new_frontmatter.html), which distinguishes general Prakrit from its named varieties.

**Proposed decision:** map the 21 `প্রা` / `প্রা.` / `প্রাক.` slots to existing `Pk`. Preserve a source note, “The source labels this comparison Prakrit without specifying a variety.” Do not attach `mh`, any other Prakrit subtype, a new location, or an inferred ancestry. The inherited registry Glottocode is not a claim that Niranjan named Maharashtri. This decision does not affect the separate `প্রা. বাং.` Old Bengali witnesses.

## Chenchu: four numbered slots have a named-language identity

The independently reviewed declaration `চে = চেঞ্চু` is unambiguous. [Glottolog's Chenchu authority](https://glottolog.org/resource/languoid/id/chen1255) gives `chen1255` and ISO `cde`. The official [Linguistic Survey of India, Jharkhand, Kurukh chapter](https://censusindia.gov.in/nada/index.php/catalog/45248/download/48955/LSI_JHARKHAND.pdf), printed p. 206, includes Chenchu in its Dravidian family tree. These establish the named language rather than a generic Telugu attribution.

**Proposed decision:** prepare a new `Chenchu` base record, name Chenchu, Glottocode `chen1255`, matching existing clade vocabulary `S. Dravidian II`, with blank source-site coordinates. Root must approve registration before the four slots can route to it. Preserve the printed forms, including compact-glyph holds; identity resolution is not a transcription approval. Do not collapse them into Telugu.

## Munda: retain the group-level claim, not Mundari

Four numbered slots explicitly have `মু = মুন্ডা`. Existing `Mu` is a Munda group record (`mund1335`), distinct from `mu` Mundari (`mund1320`). [Glottolog identifies `mund1335` as Mundaic](https://glottolog.org/resource/languoid/id/mund1335), a grouping. The local CDIAL key likewise distinguishes `Mu`; this review found no raw CDIAL lexical rows beginning with that base.

**Proposed decision:** retain these as source-attributed Munda-unspecified comparisons. `Mu` is a defensible group-reference candidate, but not proof of a spoken lect or a proto-form. Do not assign `mu` or `PMu`. Whether group-only attestations should emit under `Mu` is an explicit parent editorial decision; until then the literals remain in the complete evidence inventory, with no inferred lect.

## Sawar / Shabar: evidence supports ambiguity, not automatic Sora

Four numbered `সান` witnesses expand to printed `সাওয়ার`; the later full-label comparison has `শবর`. [The Sora authority](https://glottolog.org/resource/languoid/id/sora1254) lists several related exonyms, including Sabar and Savara, so existing `so` is a candidate. But that does not identify these particular witnesses. A [primary linguistic study of eastern Indian contact speech](https://www.tandfonline.com/doi/full/10.1080/14790718.2025.2554270) specifically documents another use of “Sabar Bhasha” for speech in Purulia that is not thereby Sora. The [Sabar community organization](https://www.sabarsamity.in/preservation-of-culture-and-language) also documents its own language work.

The tempting separate “Dravidian Savara” identity is not a safe alternative: [Hammarström's language-catalogue review](https://pure.mpg.de/rest/items/item_2230850_2/component/file_2230849/content) discusses that entry as spurious and traces the confusion to earlier classification. No new Savara base should be created from it.

**Proposed decision:** keep the four Sawar and the later Shabar witness unresolved at the lect level, with `so` recorded only as a candidate. A source bibliography, locality or independently matched witness is required. The three bare `শ` labels remain separately undeclared and must not be folded into this decision.

## Kayethi: two numbered and three later witnesses remain named but unresolved

The declaration reads `কাইথী`, while later rows specify Kayethi, sometimes Bihar. [Unicode's primary script description](https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-15/) documents Kaithi use for multiple languages. Neither a script name nor Bihar uniquely selects Bhojpuri, Magahi, Maithili or another lect.

**Proposed decision:** preserve the literal Kayethi attribution and Bihar qualifier where printed. No Kaithal mapping, generic Hindi collapse, or guessed Bihar-language assignment. This is a closed evidence limitation, not an omitted source cell.

## Kakmara: genuine named fieldwork lead, no secure base equivalence yet

The two later rows explicitly name Kakmara (Dravidian); no numbered slot has the exact declared `কাক`. Bhowmik's [primary fieldwork article](https://weblibrary.in/books/Janajati_Darpan_Issue_7/viewstory.php?fn=chapter%2Fchapter-004.html&tt=Some+Songs+and+a+Short+vocabulary+of+the+Kakmaras%2C+a+Nomadic+Dravidian+Tribe+in+Midnapore++-++Suhrid+Kumar+Bhowmik) was available through the search index although direct opening failed. It describes Midnapore fieldwork in 1971–72 with later work, and characterizes the speech as close to Telugu. It therefore substantiates a real named speech tradition, not an invented label. The two Niranjan items do not provide a decisive exact match to the short indexed vocabulary.

**Proposed decision:** retain Kakmara as a named source variety with source-claimed Dravidian affiliation; do not silently substitute Telugu, invent a Glottocode, or assert a Midnapore elicitation site for Niranjan. A separately named provisional registry record is possible only by explicit parent policy; current base-equivalence remains unresolved.

## Vedic: five additional source-stage proposals

The five `বৈ` witnesses explicitly denote Vedic. Existing Sanskrit `Sk` includes source-era treatment in `tags.py`, so the base itself need not be withheld. **Proposed decision:** `Sk` with an explicit Vedic source qualifier retained in Notes (or an approved exact source-stage tag). Do not infer Early-Vedic versus Late-Vedic, a particular text, or a date. This differs from the Carya manuscript attribution, whose language/stage remains unresolved.

## Effect and remaining gates

The keyed JSON addendum proposes 21 existing-base Prakrit resolutions and five Sanskrit-base/Vedic-qualified resolutions, plus four Chenchu mappings contingent on new registry approval. It preserves the first 395-slot inventory and every source literal. Other discussed labels retain evidence-backed ambiguity or an explicit group-level decision. Parent review, registration where needed, source-specific profile/transcription review and whole-source installation remain separate gates.
