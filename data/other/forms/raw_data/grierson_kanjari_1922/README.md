# Grierson 1922 Kanjari full-source stage

Source: G. A. Grierson, *Linguistic Survey of India*, XI, *Gipsy Languages* (1922), printed96–120 and both Kanjari columns of the complete241-prompt comparative table, printed180/184/188/192/196/200/204/208/212. The original public-domain edition is retained at `tmp/pdfs/lsi-v11/LSI-V11.pdf`, SHA256 `50df2b41e31420139148e2b16b321336881f227e917ca55cfedf225315cf4574`. Printed page+12 gives the one-based PDF page. English table prompts are two printed pages earlier. Original snapshot: https://archive.org/download/LSIV0-V11/LSI-V11.pdf.

## Current state

Full reviewed source stage installed after fresh independent pass7 (0/20 material errors). The former163-row pilot has been replaced while preserving its entry keys. `prose-staged.tsv`, `specimen-staged.tsv`, and `table-staged.tsv` each underwent a complete second reading against original page images. `second-reading-corrections-20260926.json` records corrections and page-specific typography. The historic pilot transcription and20-cell report are retained as prior-stage evidence, not reused as certification of the extension.

The proposal accounts for2229units:482comparative-table cells,239prose lexical/grammar units and1508aligned specimen attestations. It produces1832forms. There are35printed table blanks,391exact same-lect/form/gloss specimen reuses with every locator retained, three attributed Kheri controls, and one bound genitive suffix retained in the inventory. All163legacy entry keys survive. Sitapur33 now has two source-distinct readings, Guṛārā and gurārā; the old pilot incorrectly collapsed them.

## Source scope and attribution

Every target specimen is included: Sitapur103–104; Aligarh108–110; Etawah111; Farrukhabad112; Belgaum114–117; and Kuchbandhi/Bahraich120. Pages113/118 are free translations of already-aligned narratives, not additional lexical witnesses. No native-script version appears here. Other comparative-table columns are separately named language controls. No inferred lemmata, silently discarded grammatical paradigms, or selective alphabet filters are used.

Kheri is a specific source-attribution conflict: pp97/105 explicitly call its specimen ordinary Hindostani rather than Kanjari, while p101 cites khamāl ‘property’ as an argot device. All16physical aligned lines of p106 are accounted in `kheri-control-scope-20260926.json`; three individual Kheri examples remain in the full audit, with their exact provenance and conflicting description, without guessed Kanjari or Hindi canonical records. Kuchbandhi is explicitly a Kanjar subdivision; its Bahraich label remains distinct. The bound genitive rō (p119) stays in the inventory. Incidental ethnographic plant names are not presented as Kanjari attestations.

Prose quotations from W. Kirkpatrick’s1911 *A Vocabulary of the Pasi Boli or Argot of the Kunchbandiya Kanjars* are attributed to Grierson’s quotation and not assigned a fictitious “Kirkpatrick” dialect. The printed bibliographic reference is Journal and Proceedings of the Asiatic Society of Bengal7, pp277ff. Tentative comparisons with Romani, Dravidian and other languages remain attributed source claims in Notes/audit; no asserted cognacy or borrowing edge is invented. The general discussion of Arabic numerals does not justify tagging every Belgaum numeral as a proven Arabic loan.

## Transcription and validation

Original preserves source case, hyphens, editorial brackets and literal vowel/consonant marks. The scoped profile lowercases, applies house w→v and ṅ→ŋ, and reads underlined d̲z̲ as the house affricate symbolʣ. It removes question marks and editorial square brackets only from normalized Form. Other vowel quality/length is not guessed. In particular, source-specific ṭipuī/ṭipūī, hū̃dō/hū̃ḍō and byādīk-mā/dusārnō-nā are retained. Locality labels have no invented coordinates.

Reproduce a proposal with `python data/other/forms/raw_data/grierson_kanjari_1922/preview_full_source.py` from the data repository. The reviewed installation was reproduced with `import_source.py --check-pdf --install`. Focused tests cover complete scope, stable keys, source-specific variants, every reused citation, controls, dialects, profile coverage and scoped parsing. No database, fullCLDF, fullsuite, graph build or browser refresh is authorized; those gates remain deferred and this must not be described as end-to-end application completion.

## Full source-stage installation

The complete 1,832-form source stage is now installed after independent pass7 (0 material errors/20), five focused canonical tests, source-meta validation and scoped parser/profile checks. All 163 legacy keys survive. Canonical CSV SHA256 is `c2818893e9e3ea2dc9f2d16048003734dd532f87847bbdc701e9b64aef63f7a4`. The 2,229 decoded canonical audit records exactly match the frozen reviewed proposal. See `SOURCE_STAGE_HANDOFF.md` for exclusions and deferred DB/full-build/browser gates.
