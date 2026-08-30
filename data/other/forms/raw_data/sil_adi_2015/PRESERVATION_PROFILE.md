# Source-local preservation profile

The accepted Adi layer is an exhaustive, manually reviewed diplomatic Unicode
transcription. Source-local staging therefore preserves the NFC source strings
exactly. It does not collapse the printed dental mark, map the survey's `tʃ`
or `dʒ` sequences into Jambu display conventions, remove length or
nasalization, or reinterpret question marks, commas, internal spaces, and
separately labelled responses.

`symbol_inventory.tsv` enumerates every Unicode character in the 2,770 staged
response rows and records the preservation decision. The source-local staging
profile is intentionally lossless; a future shared conversion route may map
well-understood sequences to house display conventions only while retaining
the diplomatic string as `Original`. No shared profile or route is changed by
this package.
