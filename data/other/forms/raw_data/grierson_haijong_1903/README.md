# Grierson 1903 Haijong full source recovery

The source-stage replacement is installed: **896 rows, 385 Native values, 907 audit units**. Full CLDF/graph/reference builds and the full suite remain deferred under the user's no-build/no-remote instructions. Browser refresh was not requested. This is not a claim that compiled or browser gates passed.

George A. Grierson, *Linguistic Survey of India*, V.I, *Specimens of the Bengali and Assamese Languages* (Calcutta, 1903), public domain. Full scope is chapter214–220 and all245 Haijong comparative cells on354,358,362,366,370,374,378,382,386,390, including all22sentence prompts. Nordic XML and separate tables omitV.I; originals were therefore necessary. DSAL grayscale image hashes/URLs and the463image DjVu SHA256 are preserved in provenance files. Source-wide OCR scope review found context mentions but no additional lexical candidates outside this scope; this is an OCR-assisted census, not a claim of visual review of the entire volume.

The canonical language is Hajong (hajo1238). Mymensingh and Sylhet are separate registered district dialects with coordinates blank. Shared grammar is not assigned arbitrarily to one district. SpecimenII was supplied by A.Porteous in1900; that is provenance, not a separate dialect.

## Evidence and accounting

-70grammar groups:57target groups,12bound markers,1standard Bengali comparison control.
-245table cells:234populated,11source blanks. All former pilot typography holds were resolved against grayscale originals.
-566Roman specimen atoms in65lines:391Mymensingh and175Sylhet.
-26Bengali-script lines, reviewed independently with4corrections. All391Mymensingh Roman atoms align:385direct Native assignments and3shared native words covering6Roman atoms. Shared words remain complete in the audit and are not forcibly split.

These881Roman source units yield896rows through explicit alternatives, plus26parallel native witnesses in the audit. Repeated source attestations retain separate keys. All14legacy keys survive corrections; durable form-identities.csv is unchanged. No etymological or borrowing links are inferred; explicit alternatives use closed source-local variant edges. First readings, the former14-row pilot, and correction reports remain immutable evidence.

Literal source Roman forms survive as Original, and Native preserves the parallel Bengali witness even when spelling differs. No separate IPA is claimed. The display profile lowercases, maps conventionalṅ→ŋ andw→v, and removes terminal sentence punctuation; it preserves word spaces, hyphens, underlining, breve/caron/diaeresis/macron and raised glyphs. Four typed glyph uncertainties remain: three raised s-shaped glyphs and an extra upper mark on haurī. They are explicitly tagged uncertain rather than reconstructed.

Grammar labels are structured as tags; blank pronoun/paradigm glosses are supplied from explicit person/number/case or paradigm scope. Residual source claims remain attributed Notes. The historical classification prose is not treated as independent modern genealogy. OCR contributed to the native first reading, so reference OCR attribution is Yes.

## Validation and reproduction

Whole-source independent grammar/table/Roman reviews, independent native review/alignment, and a fresh20-row final audit (seed20260926146) pass with0material errors. Twelve focused recovery and installed-source tests pass; the normal scoped parser converts896/896rows with no errors and preserves Original, Native, citations, notes, tags and keys. No full build was run.

From the data repository, run:

```sh
.venv/bin/python data/other/forms/raw_data/grierson_haijong_1903/import_source.py --install
.venv/bin/python -m pytest -q tests/test_grierson_haijong_1903.py tests/test_grierson_haijong_full_recovery.py
```

The importer verifies the approved proposal hashes and exact regeneration before installation. `--check-scan` optionally verifies the cached original DjVu. `prepare_full_source.py --stage` prepares review files but never installs them. Any changed proposal requires renewed audit approval before installation. Compiled graph/reference checks, full suite and representative app entries are deferred; existing app data is not refreshed by this source-stage work.
