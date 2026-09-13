# Ghatage: South Kanara Konkani (1963) and Kudali (1965)

This package extracts the vocabulary chapters only. The raw evidence is the
complete, pinned, image-only scan, not the malformed 2023 Wayback representations.

| Volume | Source ID | Included printed / PDF pages | Representation |
| --- | --- | --- | --- |
| Konkani of South Kanara | `ghatage-konkani1963` | 120–140 / 128–148 | 149-page Internet Archive scan |
| Kudali | `ghatage-kudali1965` | 95–150 / 105–160 | 161-page publisher scan, Wayback 20220528140557 |

Acquisition date: 2026-09-11. The scan hashes and page assertions are in
`import_glossaries.py` and the two generated manifests. The title-page dates are
1963 and 1965. The scan files themselves are not committed or redistributed.
Neither work has a verified publisher open licence. Only extracted lexical facts
are included. An uploader's CC label on the Konkani mirror is not represented as
publisher authorization.

URLs:

- Konkani: https://archive.org/download/KonkaniOfSouthKanara/Konkani%20of%20South%20Kanara.pdf
- Alternate Konkani OCR: https://archive.org/download/KonkaniOfSouthKanara/Konkani%20of%20South%20Kanara_djvu.txt
- Kudali: https://web.archive.org/web/20220528140557id_/https://sahitya.marathi.gov.in/scans/Kudali%20(II).pdf
- Original publisher paths: `https://sahitya.marathi.gov.in/scans/Konkani%20of%20South%20Kanara.pdf`
  and `https://sahitya.marathi.gov.in/scans/Kudali%20%28II%29.pdf`.

Ordinary PyMuPDF extraction found no usable text in either complete scan. The
Konkani IA DjVu OCR loses source diacritics and is unsuitable as the transcription
layer. Later 2023 Wayback downloads were truncated to 1 MiB and reported missing
PDF objects; they are excluded representations, not additional editions.

## Reproduction

The small checked `*-ocr.json.gz` files contain every recognized word, box,
confidence, page dimensions, and deskew angle, including material outside the
vocabulary body. They are immutable extraction evidence. `layout.py` groups the
separate form/gloss columns and attaches definition continuations. Kudali has one
pair of columns; Konkani has two independent pairs. The first-page headings,
running headers, the Kudali final-page library stamp, and printer signatures are
not lexical forms. Rejected candidate records remain in the audit.

```
python data/other/forms/raw_data/ghatage_western/import_glossaries.py
python data/other/forms/raw_data/ghatage_western/import_glossaries.py --install
```

The first command writes proposed files under `tmp/20260911-ghatage-*`; the second
writes the canonical rich CSVs, JSONL audits, and manifests. Input OCR is sufficient
to reproduce installed output; no network or PDF library is needed. Corrections
are keyed by the pinned page/column/candidate entry. No correction replaces raw
OCR. A change to the layout must explicitly reconcile existing keys before any
future re-ingestion; sorting or editing spellings must never renumber records.

To reproduce OCR, cache the scans as `SCRATCH/konkani.pdf` and `SCRATCH/kudali.pdf`,
then run `extract_pages.py SCRATCH` followed by `deskew_ocr.py SCRATCH`. These require
PyMuPDF, Pillow, numpy, and Tesseract 5.5.2 with `script/Latin`; use 300 DPI, PSM 6,
TSV output. Deskew searches −2.5°…2.5° in 0.1° steps by row-projection contrast and
uses bicubic rotation. The checked word snapshots are authoritative for repeatable
parsing; a new OCR model/pass is an explicitly reviewed new extraction.

```
python data/other/forms/raw_data/ghatage_western/audit_sample.py \
  --seed 20260913 --output /tmp/ghatage-audit --images SCRATCH
```

This selects 20 raw candidate records from each volume and optionally crops their
source images. It never marks an audit as passed automatically.

## Editorial model

South Kanara Chitrapur Sarasvat Konkani uses canonical `Ko`, with a named dialect
under that language. Kudali uses canonical `M` under the author's historical
Marathi classification, with the Vengurla locality retained as a dialect. The
individual consultants are provenance, not separately invented lects. Coordinates
and dialect Glottocodes are deliberately blank: no exact historical collection
point or independent modern dialect match was established.

Only printed POS/gender/number labels are structured. Consecutive headwords with
separately scoped genders remain separate forms. Numerals, verbs, pronouns,
adjectives, adverbs, and indeclinables retain their labels. Source parenthetic
`(pl.)` and `(sg.)` labels become number tags. Compound and optional spellings
remain in the source transcription. No inferred inherited, borrowing, derivation,
or variant relations are installed.

`Original` preserves the reviewed transcription or explicitly unreviewed OCR;
raw OCR is additionally preserved in full in the audit. `Native` and `Phonemic`
are blank because the vocabulary does not provide separate layers. Profile
`ghatage-western` changes colon length to macrons and š to ś; it retains č/c and
ǰ/j contrasts and capital morphophonemic symbols. Kudali pp. 1–3 distinguish
nasalization, nonphonemic length, aspiration, and morphophonemes. Uppercase K/C
actually occur in its vocabulary (aK ‘fire’, aC ‘today’), despite the introductory
notation using the voiced series; their case is preserved without interpretation.
Unreviewed Latin OCR diacritics are preserved, not silently guessed into phonemes.

## Quality and scope limits

These are **OCR ingests with review flags, not fully proofread transcriptions**.
Every unreviewed installed row has `ocr-review uncertain`; the audit specifies
`ocr:unreviewed`. Known damaged/unparsed heads stay audit-only. Image corrections
cover chapter boundaries, difficult layouts, and three seeded checks. See the
workspace review and `audit-review.json` for results; character-level image audits
did not achieve the target of zero material errors on fresh samples. Corrected
sample records do not establish that the rest of the OCR is error-free.

The parser's narrow left gutter and false indented head detection were fixed;
they had swallowed nose-ornament/left/pig entries and split the ‘without oil’
definition. The fresh third sample had zero observed form/gloss-row alignment
errors in each book. Letter-level OCR errors remain a separate, explicit review
backlog. Grammar examples, sentence lists, continuous texts, introductory examples,
and bibliography are excluded from this vocabulary-only ingest.
