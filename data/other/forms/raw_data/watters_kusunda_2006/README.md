# Watters 2006 Kusunda vocabulary

This package installs Appendix A of David E. Watters, *Notes on Kusunda
Grammar: A Language Isolate of Nepal* (2006), printed and physical PDF pages
139--152.

## Scope and rights

- Source census: 877 printed dictionary entries (the introduction says “about
  850”).
- Installed census: 1,387 separately addressable attestations. Comma- and
  semicolon-delimited members of verb paradigms and other alternants are real
  printed forms, so they are separate rows linked to the first form with
  `Variant_Of_Key`. Forward-slash variants additionally carry
  `sound-variant`.
- The source PDF is copyrighted and is not checked in. Its SHA-256 is pinned in
  `manifest.json`.
- Definitions, literal explanations, cross-references, part-of-speech labels,
  form labels, and source-marked Nepali loans are retained. Running grammar
  outside Appendix A and Appendix B example sentences are out of lexical scope.

## Extraction and transcription

The PDF is born-digital, but its embedded STEDT subset has a broken Unicode
mapping. A pinned Wiktionary transcription at revision 84645357 provides the
Unicode repair layer. It is a control, not a replacement bibliographic source:
the importer fixes the row order and page/column boundaries to the PDF and the
audit preserves the printed locator for every row. The control snapshot is
CC-BY-SA 4.0 and its checksum and permanent URL are recorded in the manifest.

The PDF and control reconcile at 877 entries and at the complete printed
part-of-speech sequence. Twenty deterministic installed rows are checked
against the rendered pages. Watters's own IPA-like transcription is otherwise
preserved; the source profile converts `χ` to Jambu `x`, length to macrons, and
removes syllable dots. No speaker-specific dialect is inferred from research-
team variation that the appendix does not attribute consistently.

## Reproduction

From `data/`:

```sh
python3 data/other/forms/raw_data/watters_kusunda_2006/import_watters.py
python3 data/other/forms/raw_data/watters_kusunda_2006/import_watters.py --install
```

Passing `--pdf PATH` additionally checks the source checksum, 182-page census,
and all 876 explicit POS loci (plus the one POS-empty entry recorded in the
fixed source census). PDF verification requires `pdfplumber`; normal offline
rebuilds require only the checked-in snapshot.
