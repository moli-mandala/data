# LSI IV Turi: complete chapter, pp. 128–134

The original 1906 *Linguistic Survey of India*, volume IV, is a public-domain source. The 701-page Wikimedia Commons DjVu is pinned by SHA256 in `manifest.json`; printed128–134 correspond to DjVu147–153. The complete chapter includes prose examples, Ranchi/Jashpur/Sarangarh interlinear specimens, translations and lexical footnotes.

The importer accounts for352 units:24 prose,117 Ranchi,102 Jashpur and109 Sarangarh. It emits291 forms, reuses57 exact same-site/form/gloss attestations with every citation retained, and excludes four contextual controls (three prose comparisons/conjectures and one bracketed editorial insertion). Introductory names without defensible language attribution and isolated grammatical endings have separate dispositions in `prose-context-dispositions.json`. Inflected forms and aligned expressions are retained; no unattested lemmas are reconstructed.

Sambalpur, Ranchi, Jashpur and Sarangarh are registered Turi dialects with blank coordinates. Unattributed prose remains unspecified. All three pilot keys survive. The former three-row pilot's21 generic holds are superseded; its audit and manifest remain historical evidence.

A literal preservation profile covers57 source symbols without inferring IPA: apostrophes, vowel quantity, nasal signs, underdots and circumflexes remain distinct. Repository display conventions map w→v and ṅ→ŋ; Original retains the literal source. Phonemic stays blank. Prose grammatical information becomes tags. Review instructions remain in the audit; Notes retain attributed source information. The contradictory prose/specimen glosses of apan remain separate. One Sarangarh sons reading, bākūnī, is explicitly uncertain rather than silently regularized.

`prepare_full_review.py` produces the reproducible preview from four source-local reading inventories. `import_source.py --install` installs it and the full audit without building CLDF or a database. The independent20-entry audit passed with0 material errors; its post-editorial addendum verifies the final Notes/tags and hashes. Focused tests cover full unit accounting, sites, citations, legacy keys, source distinctions, uncertainty, Unicode, profile coverage and all291 rows through the scoped parser.

Full CLDF/build, compiled identity/graph/reference checks, full-suite and browser verification remain deferred under the user's no-build instruction. This is source-stage completion, not a claim that the full pipeline or application has been refreshed. No etymological edges are inferred.
