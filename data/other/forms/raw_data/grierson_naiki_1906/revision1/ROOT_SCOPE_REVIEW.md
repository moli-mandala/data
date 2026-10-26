# Naikī of Chanda: original-source scope review, 2026-09-27

Scope inspection only; no lexical proposal or installed output yet. The full
ingestion checklist is active, with dictionary/glossary and OCR addenda where
applicable. Full compiled and browser gates remain deferred by user instruction.

Root inspected all nine original printed pages 561–563 and 570–575 from the
existing `tmp/pdfs/LSI-V4.djvu`. Physical DjVu page = printed page + 20 was
confirmed from visible folios. Readable 1500×2100 bounding-box renders are in
`tmp/naiki-scope/`; these are scope evidence, not a final glyph audit. Use richer
grayscale originals for consequential degraded glyphs before final review.

## Included source scope

- 570: all target noun/case examples and bound morphology; numerals and explicit
  uncertainty about e/o length; the complete three-column personal-pronoun
  table; pronoun prose and relative-clause example; verb forms and their
  source-qualified interpretation. Telugu, Tamil, and Kui comparison forms
  require explicit control audit dispositions.
- 571: all target verb examples, negative morphology, imperatives, infinitives,
  participles, and agency-noun examples, including both whole phrases and
  genuinely printed component forms. Preserve “seems,” “apparently,” and the
  author’s explicitly questioned reading; do not silently correct them.
- 572–573: specimen I, the complete Prodigal Son interlinear narrative, headed
  District Chanda / No.77. Preserve every physical source occurrence before
  any exact-analysis reuse, including the page break and repeated words.
- 574–575: specimen II, a popular tale, District Chanda / No.78; include the
  complete interlinear material, speaker labels where lexically glossed, and
  the milkman dialogue continuing onto 575. The free English translation on
  575 is source evidence, not additional invented word alignment.

No native-script witness is printed in these six chapter pages. Bibliographic
references to Hislop must remain references; no claim that the referenced work
has itself been exhausted.

## Shared introduction and controls

Original 561 contains introductory classification and then an explicit Kōlāmī
heading. Original 562–563 is Kōlāmī grammar, including its own pronoun and verb
tables. It must not be relabelled Naikī because the broader heading names both.
The one explicit Naikī claim on 562 states that feminine and neuter singular
take the same form; it prints no separate Naikī lexical example. Retain that
claim as contextual evidence, with no fabricated Naikī forms from surrounding
Kōlāmī examples. The classification prose on 561 is historical source analysis,
not authority to collapse present registry identities.

## Keyed transcription limitations

`snapshot_scope.py` reproducibly selects the five keyed pages 561,562,563,570,571
from the cached Norwegian XML, retaining page attributes and sentences in
`nordic-scope-snapshot.json`. Pages 572–575 are absent. The pronoun table on 570
is also absent between the introductory sentence and comparison paragraph.
Therefore the keyed text is a prose extraction aid only, not a complete source.

The keyed page metadata supplies ISO `nit` and coordinates 19.69,78.59, whereas
the current Jambu registry distinguishes `Naiki` (Chanda, `naik1250`) from
`Naikri` (`sout1549`, whose coordinates match those keyed values). Do not copy
the XML identity/coordinates into the new source without explicit adjudication.
Current source heading “Darwe Gōṇḍs in Chanda” supports reviewing the Chanda
identity; it does not make Naikī a Gondi dialect in the canonical registry.

## Remaining work before any installation

1. Check whole-volume comparative headers, index, errata, and cross-references
   for additional Naikī material. The nine inspected pages do not establish
   whole-volume exclusion by themselves.
2. Record immutable primary-source identity/hash and transcription provenance.
3. Census every lexical/morphological unit, including controls and uncertain
   readings; transcribe all four specimen pages and the omitted pronoun table.
4. Reconcile source identity, existing attestations, stable keys, grammatical
   labels, sound profile, citations, and full audit accounting.
5. Freeze actual proposed output, independently audit originals against it,
   then run focused checks before source-stage installation. Do not label
   this scope packet a completed ingestion.

## Expanded scope check, later on 2026-09-27

The reproducible whole-volume keyed-text search is now saved in
`nordic-volume-naiki-mentions.json`. It identifies nine page contexts; spelling
search is a retrieval aid and does not prove absence of OCR-missed mentions.

**Add printed 623 to the target inventory.** The original was visually checked:
the Brahui genitive discussion explicitly cites “Naikī and Kōlāmī nē.” Preserve
the literal printed `nē` as a genitive-suffix witness with this locator and
comparison provenance, without inferring a borrowing or etymological edge.
Do not exclude it simply because the surrounding chapter is Brahui. This can
later reuse an identical chapter analysis only if all citations survive.

Other retrieved contexts: 622 (dative/accusative equivalence), 626 (negative
conjugation), 474 (classification), 283–284 (classification), and the already
inspected 561,562,570. The keyed sentences at those additional locations do not
print a Naikī form; visual context checks remain to be completed where not
already inspected. Preserve contextual assertions separately from lexical rows.

Root viewed the full standard-list header spread on original 646–649. Columns
are English, Tamil (Poona), Korvi (Belgaum), Kaikadi (Sholapur), Malayalam,
Kanarese (Belgaum), Kurukh (Palamau), Malto, Kui (Khondmal), Gondi (Mandla),
Telugu, and Brahui (Kalat), with English repeated at the far edge. No Naikī
column occurs in this list; do not substitute a Gondi or other column.

Root also read original Addenda Minora IV printed17–18 in the cached alternate
scan (`tmp/holiya-scope-review/addenda-leaf706.jpg` and
`tmp/korvi-full-scope/alternate-leaf708.jpg`). The ordered corrections jump from
483 to 580, then 621 and comparative-table items. None targets 570–575, 623, or
a Naikī column. This is a scoped no-applicable-correction result, not a claim
about all subsequent scholarship. The later addendum must not be dated 1906
merely because it is appended to the volume scan.

The live Glottolog `naik1250` page was read on 2026-09-27:
https://glottolog.org/resource/languoid/id/naik1250 . It labels Naiki a dialect;
the accessible page did not establish its parent or historical Chanda identity.
Do not overstate that lookup as validation of the complete language mapping.
The existing canonical Jambu ID remains a reuse candidate pending reconciliation.

## Original cross-reference and grayscale checks

Root inspected the complete original pages 622 and 626 on 2026-09-27.
Page 622 compares dative/accusative identity across named languages, including
Naiki; the concrete forms that follow belong to Brahui, Tamil, Gondi, and
Kurukh, not Naiki. Page 626 compares the existence of negative conjugation;
its displayed paradigm belongs to Brahui, and its named negative examples
belong to other explicitly named languages. Neither page adds a Naiki lexical
witness. These are positive contextual checks, not exclusions based solely on
an automated string search.

The six DSAL grayscale originals for 570–575 are now cached locally, with
URLs, byte counts, and SHA-256 hashes in `dsal-originals-manifest.json`.
The complete originals 570 and 571 have been compared against the prose
first-reading inventory at page level. Their lexical scope agrees; exact
glyph acceptance is still pending, especially the agency-noun `t`/`ṭ`
distinction. This page-level comparison must not be substituted for the
fresh frozen-output audit. The original pronoun table is visibly present on
570, confirming that the Norwegian keyed text omits substantive material.

Root additionally inspected original pages 283, 284, and 474. Page 283 explicitly
separates Kolami and Naiki from Gondi in the survey's classification. Page 474
repeats that separation; the following statistics are Gondi statistics and
must not be assigned to Naiki. Neither page prints target lexical forms.
Page 284 groups Kolami, Naiki, and the Pusad Bhili variety in a comparative
classification paragraph and discusses gender, oblique bases, plural suffixes,
and numerals without quoting Naiki forms on this page. That paragraph continues
onto 285, which still needs inspection before closing this cross-reference.

The continuation on 285 has now been inspected. It contains a collectively
introduced pronoun `aund` ‘he’, comparing the group of lects named on 284
with Old Kanarese and Telugu. It does not separately attest the form in each
lect. The ambiguous attribution is explicitly retained in
`contextual-claims.jsonl` rather than silently excluded or emitted as a certain
Naiki form. The remainder of 285 is classification discussion and a family
diagram. This closes the retrieved cross-reference's page boundary while
leaving the collective lexical attribution visible for review.
