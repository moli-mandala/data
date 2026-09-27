# Gtaʔ (Chatterji) in the Munda Lexical Archive

The complete source-stage extraction accounts for **2,066 lexical lines → 2,259 forms**: all 2,063 numbered records and three unnumbered lines. Every headword, witness and explicitly separate sense is represented. No record is excluded by alphabet, headword multiplicity, or missing definition. Full CLDF build, compiled graph/reference validation and full-suite gates remain deferred by the user's instruction; no browser database refresh was requested.

The canonical input is Donegan and Stampe's 2004 Munda Lexical Archive Gtaʔ (Chatterji) file, pinned to the 2021-03-23 Wayback snapshot with archived Last-Modified 2014-09-05. The original URLs and SHA-256 are in `manifest.json`. `LICENSE` reproduces the archive's required same-conditions reuse terms. No OCR was used. An alternative third-party IndianLexicon PDF was inspected previously but is not the extraction input.

## Coverage and recovery

The parser treats each physical lexical line as a source unit. The old numeric-ID parser silently joined an unnumbered `go-gu` ‘seventeen’ to the next entry; this line now has its own key. Two unnumbered `jibon-lEe-ke-ne` ‘live’ lines are recovered by fixing an unambiguous closing delimiter. The missing POS brace in entry 6172 and single closing gloss quote in entries 11031/13272 are likewise explicitly repaired. Exact raw lines and all five repairs remain in the audit.

The 2,259 forms include 243 explicitly unglossed archive stem heads and 64 forms whose definitions are only source placeholders (`G`, `?`, `??`). These **307 rows intentionally have blank glosses**, with the original evidence and typed reason retained in the audit. They are attested source heads, not inferred definitions. All other lexical definitions are preserved; consecutive grammatical senses expand separately. Cited synonyms do not become extra headword senses. The one fully glossed causative subentry embedded in entry 1112 is emitted as a distinct child attestation, not as another sense of its preceding headword.

Headwords retain uppercase letters, `@`, `~`, apostrophes, square-bracket analyses, optional parenthetical segments, equal signs, dots, spaces and hyphens literally. The explicit preservation profile changes only `w` to house `v`. It makes no unsupported phonemic claim. `Original` keeps source spelling; `Phonemic` and `Native` remain blank. There are 1,456 rows with typed transcription, grammar, provenance, or source-editor uncertainty. Source witness labels `(C)`, `(M)`, `(:)`, `M` and `CM` are retained per head in the audit, not invented as dialects or silently excluded. All forms use registered base language `gt`.

Archive POS labels are mapped where defensible; `X` remains uncategorized and unexplained `D` is flagged rather than guessed. Verb-object `VO` receives only the verb category. Negative/interrogative/prefix/vocative and explicit transitivity markers are structured; NB and NK are noun categories, with NK additionally tagged kinship. Archive `@N...`/`@S` locators are preserved within the DSGT citation as archive locators; their bibliography is not invented. Genuine `!` prose is Notes, lexical analyses and comparisons are Etymology, and raw parser/review evidence stays in the audit.

There are **109 alternate edges** for explicit `,,` lists; lists questioned as phrases and slash/backslash analyses remain separate without guessed variant links. Twelve explicit causative claims resolve to unique exact source-head targets; the explicitly glossed child in entry 1112 adds a thirteenth derivation edge. Seven source-questioned and three ambiguous causative claims remain unlinked with candidate evidence. Donor-language comparisons remain source prose; no donor form or cross-source etymon is fabricated.

## Identity and validation

Existing 733 pilot keys are retained as the first head/first sense of their source entry. Additional heads/senses use stable numbered child keys. The three unnumbered lines have source-local keys and exact body-line citations. `audit.jsonl` has one decision per raw lexical line, with emitted child records; duplicate-looking source-defined entries remain distinct. Existing Rau 2019 DSGT attestations remain separate inputs pending compiled deduplication review.

Reproduce from the data repository:

```sh
python3 data/other/forms/raw_data/gta_chatterji_mla_2004/import_source.py --install
PYTHONPATH=. uv run pytest -q tests/test_gta_chatterji_mla_2004.py tests/test_sound_profiles.py tests/test_dialects.py
uv run python source_meta.py
```

Twenty freshly seeded raw/output records pass with zero material errors (`full-scope-audit-20260926.json`, seed 2026092610). Deliberate edge checks cover malformed delimiters, all three unnumbered lines, mixed witnesses, multi-sense entries, phrase/variant ambiguity, commentary boundaries and no-definition heads. **25 focused importer/profile/dialect tests pass**, followed by four focused source tests after the final subentry repair. An independent fresh 20-record audit also found zero material errors (`independent-audit-20260926-pass1.json`, seed 2026092611); the additional entry 1112 edge is separately checked. All 2,259 rows survive the scoped `parse_file` check with unique keys and no mapping errors; source metadata validates. This is not a full data build or a claim of compiled/browser completion. Representative source-local entries: 631 (three heads × two senses), 6172 (recovered noun), 4351 (literal source analysis), 30 (explicit causative), unnumbered:1 (recovered numeral).
