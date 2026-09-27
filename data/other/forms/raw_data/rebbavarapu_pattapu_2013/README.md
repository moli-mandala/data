# Pattapu — ISO639-3 request2013-020

> Current status: **201 source rows installed**, representing199 of210 prompts.
> Ten uncertain transcriptions withheld; one unanswered prompt excluded; four
> emitted tie-placement readings uncertain.104 prompts extend the derived selection.
> Profile, settings, bibliography and Ethamukkala locality are registered.
> 24 distinct focused checks pass; metadata validation passes. Full data/database
> builds, compiled validation, full suite and app QA remain deferred.
> Earlier sections below record preparation history, not current installation state.

Selected for completion of a sparse language's primary wordlist, not as an
independent attestation of Lindgren's101-row derived selection. Lindgren's
thesis explicitly identifies IRA(2013) as its Pattapu source (PDF28/printed27).

Ruth Rebbavarapu submitted the request on19July2013. The appended wordlist
records fieldwork on2April2013 at Ethamukkala, Andhra Pradesh, with researcher
G Sudheer. The pinned official PDF has9pages; all210 numbered prompts are on
pages6–8. See manifest for checksum, provenance and exclusions.

`extract_scaffold.py --pdf <pinned-pdf> --output <jsonl>` reproduces the210-cell
locator scaffold from column-wise native text. **The text layer loses visible
phonetic glyphs.** PDF6 was rendered and inspected; no scaffold string is an
accepted reading. Visual transcription is required before emission. No OCR,
installed CSV, profile, dialect or bibliography registration yet.

Source contact details and individual elicitation metadata are not dialects.
Use the attested village as locality and leave coordinates blank until verified.
The survey-wordlist ingestion checklist is active. Full data/database builds
and remote execution remain prohibited by the user.

## Embedded-font recovery

`recover_font.py --pdf <pinned-pdf> --output <directory>` now repairs15 false
space mappings using the embedded TrueType Unicode cmap and Identity CID-to-glyph
mapping. Ordinary spaces are preserved. Original text scaffold remains unchanged;
recovered210-cell scaffold is a separate artifact, still not installable.
No OCR or language-based spelling guesses are used.

Unencoded glyphCID1345 occurs in9cells. Enlarged crops show an overhead arc near
affricates, but its exact Unicode/attachment is unresolved; U+E000 explicitly
marks it in the scaffold. Do not treat that placeholder as a phoneme or silently
replace it. The first10cells have glyph comparisons against the rendered page;
`visual-review.jsonl` does not claim full transcription/profile acceptance.
Two focused tests pass(0.37s), including reproduction from the actual pinnedPDF.

Further local review compared items11–42 with a scale3 rendering of the left
column. The ledger now contains42 glyph comparisons; all retain the recovered
readings pending profile and overlap review. Item11 preserves both comma-separated
responses; items13/14 retain their separate arm/elbow prompts despite identical
forms. The embedded OpenType MATH horizontal/vertical variant constructions were
checked: none identifies CID1345, so its nine occurrences remain unresolved.

## Complete visual pass and overlap review

All210 prompts now have a visual comparison:199 retain recovered glyph readings,
nine remain unresolved at CID1345, item69 has a vertical-stroke notation needing
review, and item73 is visibly unanswered. These are preparation decisions, not
acceptance of a sound profile. Removed a cropped footer fragment from item87 by
locating the footer on the full page before splitting columns. The original
`text-scaffold.jsonl` remains historical raw evidence; current reproduction targets
`font-recovered-scaffold.jsonl`.

`lindgren-overlap-review.jsonl` accounts for all101 existing rows.93 have ordinary
shared-source prompt correspondences; eight need semantic/grammatical review.
The derived cold row has the primary chili form (item75); primary cold things
is item137. Two derived gendered second-person forms correspond ambiguously to
primary informal/formal prompts. The primary we (two) is not explicitly exclusive,
and generic we is not explicitly inclusive. Three third-person analyses add
remote/animate distinctions absent from the primary prompts. Existing rows remain
unchanged; this ledger must guide reconciliation before installation.

Four focused tests pass, including actual-PDF reproduction, footer exclusion,
complete review accounting and retention of the overlap disagreements. No full
pipeline or database build was run.

## Reproducible lexical draft

Run `import_source.py --output <scratch-directory>` to write201 proposed rows
and a210-prompt audit.199 prompts emit rows; breast and snake each supply two
comma-separated responses. No variant direction is asserted. Ten unresolved
transcriptions and one unanswered prompt remain audit-only.104 proposed prompts
lie outside the candidate correspondences for the existing Lindgren selection.

Pronoun tags follow primary prompts: informal/formal, explicit person/number and
gender, and first-person dual for we (two); no inclusive/exclusive or remote label
is inferred. Complete elicited clauses remain intact. Imperative prompts are tagged
without stripping the source response to a guessed stem. The sole source phonetic
layer is preserved in raw Form (future Original); no duplicate Phonemic or invented
native script is emitted. Stable keys use numbered item and reading, not form text.

The44-symbol inventory is checked in; no profile is accepted yet. The request's
non-wordlist pages were searched for a transcription key and provide none. Stress,
superscripts, dental marks, vowel distinctions and tie placement therefore remain
explicit profile-review matters. Draft output lives only in workspace scratch;
there is no install mode. Five focused tests pass(0.35s).

## Conservative profile and parser audit

The package-local profile now covers all201 proposed forms in NFC and NFD.
It converts explicit long vowels to macrons and familiar IPA symbols j/ɡ/ɖ/ʂ/ʋ
to y/g/ḍ/ṣ/v, preserving vowel qualities, dental marks, stress, superscripts,
word boundaries and the source's anomalously placed tie. Items4,76,138,139 have
an uncertain tag with a typed tie-placement reason in the audit. No assertion
about gemination or tie attachment is inferred from these marks.

`check_draft.py` exercises only the source-row parser with temporary settings:
201 conversions, no errors, all originals and glosses intact. Seed2026092202
supplied20 output comparisons against source pages:0 material errors, recorded
in `output-audit-2026092202.json`. Six focused tests pass(0.42s). Global merging
and compiled outputs are outside this check; no database was built.

Prepared a package-local bibliography record (parsed and formatted) and an
Ethamukkala dialect proposal with blank coordinates. Neither is registered yet.
The full overlap reconciliation is still required before installing primary
rows alongside the Lindgren-derived selection.

## Overlap integration decision

Primary and derivative source representations retain their own transcription and
analysis. The Lindgren rows additionally carry source-specific cognate-set claims;
primary forms must not silently inherit those claims, and stress/diacritics must
not be erased merely to force matching. Bibliography metadata explicitly records
the shared fieldwork provenance, so the two sources are not independent evidence.
The overlap ledger provides all101 correspondences, including ambiguity.

Eight existing Lindgren rows now preserve their original form, gloss, phonemic
field, citation, key and cognate set while adding an uncertainty flag and a
primary-source comparison note. The original importer reads the pinned review
sidecar and checks expected form/gloss before reapplying annotations; generated
audit records retain both upstream data and the typed review. No corrected gloss
or revised grammatical claim is silently attributed to Lindgren.17 focused tests
pass, including source-file reproduction; the compiled-data test was deliberately
excluded because the user has not requested a build.

## Installed source files

`import_source.py --install` installs CSV, YAML and profile and refreshes the
210-cell audit only. IPAʒ maps to houseź while retaining the tie's exact placement;
no affricate collapse is asserted. Source Originals remain intact. Locality metadata
has a canonical Pattapu parent and blank coordinates. Bibliography formatting passed
in memory without regenerating references. No ancestry/borrowing/variant edges are
inferred. Representative source keys: item75(chili), item137(cold things), item203/204
(informal/formal you), item208(we two). App availability is not claimed before a
requested build and browser refresh.
