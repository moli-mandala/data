# SIL ESR 2012-015 Kurumba wordlists

This is a resumable, source-local review package for Appendix C of Frank Blair
et al., *A Sociolinguistic Profile of Kurumba Dialects* (SIL Electronic Survey
Reports 2012-015). The authoritative base ledger plus disjoint manual page-review
chunks contain all 10,450 completed cells; 4,738 manually attested forms are
staged source-locally, and no OCR-derived form has been accepted or staged.

## Pinned source and scope

- SIL archive: <https://www.sil.org/resources/archives/50805>
- canonical PDF: <https://www.sil.org/system/files/reapdata/12/35/31/123531168727431837602494229591948707669/silesr2012_015.pdf>
- workspace copy: `tmp/pdfs/kurumba_2012/silesr2012_015.pdf`
- SHA-256: `250dc3d83661227caa66bf16e390e51c2dcb7186fa435252541ed13bbfcd9137`
- 128,831,439 bytes; 436 physical pages
- included: Appendix C metadata on physical pages 214–216 and every wordlist
  cell on physical pages 217–436 (printed pages 212–431)
- expected: nineteen lists × 550 prompts = 10,450 cells; fifteen target lists
  and four comparison controls
- excluded: survey questionnaires, sociolinguistic prose/tables, bibliography,
  and other non-Appendix-C content

The SIL record does not state an extracted-data reuse licence. The source PDF
is therefore pinned by URL and checksum but is not redistributed here.

## Representations and review authority

The appendix is an image scan with a badly corrupt Adobe Paper Capture text
layer. `ocr_layer_scaffold.txt` and the `OCR_*` ledger columns are untrusted
navigation evidence only. `build_review_scaffold.py` never writes a manual
form, and `import_kurumba.py` never reads an OCR field when constructing staged
data. The renderer uses Poppler at a fixed 300 dpi.

`manual_transcription.tsv` is the immutable cell-addressed base review record.
Disjoint files under `manual_chunks/` overlay only still-pending stable cell
keys, allowing parallel reviewers to contribute pages without overwriting one
another; the importer rejects duplicate chunk keys and completed-base-row
overwrites. Each
row already has a stable key, physical/printed page, row, item, list, scope,
base-language mapping, dialect ID, and OCR comparison field. A reviewer must
visually inspect the rendered source cell and fill `Manual_Form`, `Cell_Status`,
`Confidence`, `Review_Method`, `Reviewer`, and `Notes`. True blanks and every
ambiguous or illegible cell are explicit statuses, never silent omissions.

The current accounting is:

- 220 data pages and 10,450 conceptual cells registered;
- 8,250 target cells and 2,200 comparison-control cells;
- 550 prompt glosses registered for separate visual review;
- all three Appendix C metadata pages visually reviewed and their topology
  frozen in `list_registry.tsv`;
- physical pages 217–436 complete: all 10,450 cells manually reviewed from
  450/600/900-dpi renders (with targeted 1200-dpi checks) and all 550 unique
  prompt glosses;
- physical p.415 / printed p.410 is the source-proven transition from the
  two-column Kalangal/Masinagudi lists to the one-column Maddur Betta list and
  therefore contains 25, not 50, conceptual cells; physical pp.415–436 are
  source-confirmed one-column Maddur Betta pages with 25 cells each, and p.436
  is the 436-page PDF endpoint at item 550;
- 100% of all 10,450 cells were checked directly against rendered
  source images; OCR/PDF text supplied no accepted lexical reading;
- 4,738 attested manual forms and 5,710 printed dash placeholders confirmed
  blank; one ambiguous reading at physical p.239 / printed p.234 / Pudukkottai
  item 20 and one illegible reading at physical p.261 / printed p.256 / Kotagiri
  Alu item 25; both are unresolved and excluded from the 4,738 staged forms;
- zero cells and zero prompt glosses remain pending.

## Reproduction and safe staging

```sh
sh data/other/forms/raw_data/sil_kurumba_2012/render_wordlists.sh
UV_CACHE_DIR=/tmp/uv-cache uv run --with pdfplumber python data/other/forms/raw_data/sil_kurumba_2012/build_review_scaffold.py --initialize
python3 data/other/forms/raw_data/sil_kurumba_2012/import_kurumba.py --verify-pdf
python3 data/other/forms/raw_data/sil_kurumba_2012/import_kurumba.py --verify-pdf --stage
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_sil_kurumba_2012.py
```

The scaffold initializer refuses to overwrite the ledger. The final staging
command emits only because all pages, prompts, and cells now have manual
completion evidence; ambiguous and illegible readings remain audit-only.

Shared bibliography, nineteen dialect registrations, target-only installed CSV,
and source-profile routing are complete. `install_target_forms.py` preserves the
frozen 4,738-row snapshot and deterministically installs only 3,204 target-scope
attestations; 1,534 controls and every non-attested cell remain audit-only. The
consolidated build/full suite, graph checks, opaque identity reconciliation,
browser QA, and commit remain deferred as described in `INTEGRATION.md`.
