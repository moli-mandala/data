# Thari OCR calibration with Kraken

## Result

Printed page 200 was held out from every training run. Fourteen of its eighteen
reviewed forms could be isolated automatically at the opening parenthesis of the
printed part-of-speech label.

| Recognizer | Scoring alphabet | Character accuracy | Exact words |
| --- | --- | ---: | ---: |
| Fresh Tesseract `eng` | base letters only | 67.4% | not retained |
| Thari-fine-tuned Tesseract | base letters only | 89.7% | 10/14 |
| Kraken trained from scratch | full Unicode | 67.1% | 2/14 |
| Kraken CATMuS-Print Tiny fine-tune | full Unicode | 78.5% | 7/14 |

The Kraken score is stricter: macrons, tildes, dots below, and stacked combining
marks count as characters. Its best model correctly produced `khetrī`, `khoprī`,
`khetar`, `khel`, `khes`, `khopo`, and `khor`. It remained unreliable for the
rare initial `g` glyph and the stacked `ḍ̠` sequence. A second stage at a lower
learning rate did not improve the held-out score.

## Training setup

- Recognizer: Kraken 7.1.
- Starting model: CATMuS-Print Tiny, Zenodo DOI `10.5281/zenodo.10602357`.
- Ground truth: 196 form-only crops from the existing manually reviewed Thari
  transcription; printed page 200 excluded.
- Validation: 14 form-only crops from printed page 200.
- Unicode normalization: NFD, allowing the model to learn base characters and
  combining macron, tilde, dot below, minus below, and macron below separately.
- Fine-tuning: codec union, augmented images, batch size 8, initial learning rate
  `1e-4`, 500-sample frozen backbone, early stopping.
- Best checkpoint: epoch 10; 62 correct of 79 validation characters.

The model weights remain an experiment in the temporary workspace rather than a
checked-in ingestion dependency. The source model's redistribution terms need to
be recorded before derived weights are committed.

## Comparison with Old Punjabi

The Old Punjabi pipeline in `shackle.py` did not rely on Latin OCR alone. It:

1. rendered the scan at 300 dpi;
2. OCRed the roman column with Tesseract's `script/Latin` model;
3. independently OCRed the parallel Gurmukhi column with the Punjabi model;
4. aligned the two columns by vertical position and used the Gurmukhi consonant
   skeleton to restore retroflex distinctions in the romanization;
5. treated the pre-existing manual file as authoritative gold; and
6. assigned confidence/review reasons, excluding suspicious continuation lines.

Thari lacks a parallel native-script column, so steps 3-4 cannot be reproduced.
Its analogue is the custom Unicode recognizer tested here, followed by manual
review of every non-exact or low-confidence form.
