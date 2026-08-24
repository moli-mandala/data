# Mayrhofer KEWA ingestion review — 2026-08-18

Canonical checklist SHA-256: `c0254863d2a149b3fdfd6ea9c155d957d24fb61e74c2a53c3e0e90e455da0063`.
Applicable addenda: Dictionary/glossary, OCR-heavy, Website/API, and
Etymological/comparative source.

- Source: Manfred Mayrhofer, *Kurzgefasstes etymologisches Wörterbuch des Altindischen*,
  1953–1980; samskrtam.ru per-article image edition, copyright 2021 version 1.0, snapshot
  2026-08-18. The site reports author permission for scanning but states no reuse terms.
- Scope: all 9,587 stable articles (I 4,108; II 2,811; III 2,668), including 225 supplement
  articles. The pinned index SHA-256 is
  `d0a3127ba237149713b706da0ce5b4380a1b8077ead07f854169113e4e5da234`.
- Extraction audit: every remote JPEG was decoded, dimension-checked, and checksummed
  (435,979,031 bytes total). Three malformed page labels, articles 4119/4125/4130, were repaired
  mechanically from stable anchors. OCR has no empty records or replacement characters; 9,586
  images used Tesseract PSM 6 and article 7158 used PSM 11.
- Display policy: the database contains no OCR text. Each accepted block contains only the exact
  remote scan, its stable original article link, locator, and citation. Audit OCR is neither
  displayed nor used for matching, and attaching KEWA never marks the CDIAL headword as OCR.
- Matching: authoritative index heads only; accent-sensitive exact matches precede unique
  accent-neutral matches. Main-article collisions remain unresolved unless one competitor alone
  has an accented index match. OCR glosses never disambiguate senses.
- Reconciliation: 9,587 articles = 2,400 ingested + 886 ambiguous + 6,301 unmatched; 2,432 scan
  blocks are installed on 2,376 CDIAL entries. KEWA adds no forms, languages/dialects, sound
  mappings, or graph edges, so those checklist gates are inapplicable.
- Manual audit: deterministic articles 313, 810, 1152, 1390, 2030, 3517, 4409, 4774, 4830,
  5741, 6035, 6642, 6823, 7872, 8195, 8759, 8879, 9062, 9142, and 9264 span all volumes.
  Scan boundaries and index-head/CDIAL mappings were correct in 20/20; OCR was character-perfect
  in 0/20, with 0 material structural or mapping errors.
- Validation: focused importer/identity tests, ordered full data build, full data suite, browser
  database integrity/build checks, and browser QA are recorded in the final handoff.
- Representative entries: CDIAL 594 (`araḥ`), 10063 (`mārayati`), and 14110 (`hiraṇyam`).

No commit, push, release, or deployment was performed.
