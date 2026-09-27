# Bounded Dravidian source discovery — 2026-09-26

Discovery only; no new lexical staging, installation, database build, remote execution, or bulk download. Recommend **Grierson 1906, complete Gōlarī/Hōliyā material in LSI IV**, with the other two candidates as fallbacks. These are complete, bounded lect packages within a public primary volume, not permission to select only convenient vocabulary pages.

## Inventory checks and counting limits

Checked canonical input filenames/YAML, raw-package names, the dated source-completeness ledger, and existing diversity notes. Literal language IDs in `data/other/forms/*.csv` are only a source-input selection aid: DEDR and source aliases mean these are NOT full lexical totals. In particular, Kudiya initially looked absent under literal `Kudiya`, but `20260813-kudiya.csv` already installs Joseph2024 through `kudiya_g1`/`kudiya_k1` aliases; it is explicitly ruled out. The earlier diversity ledger also documents existing compiled coverage for Manda539, Naiki518, Pengo1015 and Malto1511. Do not report these as zero-coverage languages.

No dedicated Grierson Holiya, Kui, or Naiki package was found in current source YAML/raw-package names or the dated partial-package ledger. DEDR-derived overlaps still require source-level reconciliation at any later ingestion.

## Candidates

| Priority | Whole package | Existing source-input evidence | Scope and effort |
|---|---|---|---|
| 1 | Grierson1906 **Gōlarī or Hōliyā**, printed385–395 | Dedicated recent source is Metry2017 Holiya numerals:39installed rows, under canonical `Holiya`/`holi1239`. No dedicated LSI package found. | Eleven chapter pages. Prose/grammar385–387 is present in the Nordic keyed-text corpus; the following specimen pages must be fully recovered from original images. Original394–395 explicitly supplies a Bhandara specimen and identifies it as Gōlarī/Hōliyā. Include every chapter example and all specimens/witnesses, with exact repeated attestations accounted; preserve historical Kanarese classification as a source claim, not an automatic canonical reassignment. |
| 2 | Grierson1906 **Naikī of Chanda**, printed570–575, plus Naikī-attributed material in shared introduction561–563 | No standalone Naiki direct source CSV found. Existing broader DEDR-derived coverage is documented; Naikri numeral source is a different named lect and must not be silently merged. | Six dedicated pages plus three shared pages to inspect for attribution. Keyed text covers570–571; original575 inspected and contains aligned dialogue followed by free translation. Include whole dedicated grammar/specimen and all explicitly Naikī shared examples; keep Kolami/Pusad comparison controls separate. Identity must distinguish Chanda Naiki from Naikri. |
| 3 | Grierson1906 **Kui/Kandhī/Khond**, printed457–471, plus complete Kui comparative-table column later in the volume | Only7literal `Kui` rows occur in the generic other-forms inventory (Burrow–Emeneau notes), but substantial DEDR-derived coverage exists. No standalone LSI package found; earlier Gondi–Kui annotated Swadesh source was explicitly excluded from a previous selected-surveys import. | Larger than the first two:15chapter pages plus all241comparative prompts. Keyed prose covers457–460 and skeleton grammar462–463; actual page images needed for missing pages/tables. Kalahandi material may represent Kuvi and needs explicit source/modern identity reconciliation rather than blanket Kui assignment. Whole Kui/Kuvi source scope must be resolved before staging. |

## Primary availability and verification

- [DSAL LSI catalogue](https://dsal.uchicago.edu/books/lsi/) explicitly lists volumeIV, *Muṇḍā and Dravidian Languages*.
- [Primary volumeIV BookReader](https://dsal.uchicago.edu/books/lsi/lsi.php?pages=701&volume=4) is public. Its HTML requires JavaScript; this is not an absence of the scanned source.
- Existing original `tmp/pdfs/LSI-V4.djvu` is already local. During this discovery, viewed local physical414=printed394 and physical415=printed395 (Holiya), and physical595=printed575 (Naiki). This copy's verified folio offset is+20 at those locations; do not borrow the Nordic/DSAL+19 page-coordinate convention without checking originals.
- Existing `tmp/pdfs/lsi-sprakbanken.xml.bz2` supplies keyed prose with named language/page metadata. It confirms chapter starts385Holiya,396Kurumba,457Kui,472Gondi,570Naiki,576Telugu. It is an aid to scope, not a substitute for original-image transcription or proof that omitted interlinear material is absent.
- Whole-volume comparative table descriptions do not name separate Holiya or Naiki columns; later ingestion must verify the original table headers and any cross-references before finalizing the census. Kui does have a dedicated comparative column, which must not be omitted.

## Other leads screened out

Joseph2024 Kudiya survey is already installed. Burrow–Bhattacharya1970 Pengo is233pages and public search results exposed only catalogue/snippet or lending access, so not a verified manageable open candidate. Reddy2009 Manda remains a catalogue lead in prior Jambu discovery notes; full primary access was not established here. CIIL Kudiya material was previously flagged for repository terms restricting systematic compilation; no material downloaded. Numeral pages for Holiya and several Kurumba varieties are already installed and are not new sources.

## Recommendation

Reserve the **whole Grierson Gōlarī/Hōliyā chapter385–395** first. It offers broader lexical and inflectional coverage than the current39numerals, an already-available public original, and a tractable11-page scope. Before ingestion, apply the full checklist, inventory every lexical-bearing span including all specimen witnesses and cross-references, verify `Holiya` identity against historical lect labels, and use DSAL grayscale for final glyph review. Do not install a grammar-only or one-specimen subset. No source extraction has started under this discovery task.
