# Original-script review: explicit hasanta and disjoining

The whole-book recovery preserves the book's Bengali-script transcription. It
does not convert it to conventional Bengali spelling or infer Chakma phonemes.

During the independent Santali-chapter review, the reviewer observed explicit
lower hasanta marks on visibly separated consonants that were omitted from some
owner first-reading strings. Reported examples include printed 95 `গিন্‌তি` /
`গন্‌তি`, `গোর্‌জি`, and printed 96 `পার্‌সি`, `বাক্‌লাক`. These are review
proposals, not certified replacement spellings. Each must be reconciled with its
original page and keyed witness before installation.

This is a known systematic error class. All first-reading files remain
unaccepted for installation until the class is checked across their full scope.
Passing structural counts or a spot sample does not close this issue.

Review requirements:

1. Inspect every target and comparator form for visible hasanta, conjuncts,
   separated consonants, vowel attachments, nasal signs and detached marks.
2. Preserve an explicit vowel-suppression mark as Bengali virama U+09CD.
   Where the source visibly disjoins a consonant sequence, consider U+200C
   after the virama to preserve that display distinction. Do not insert ZWNJ
   merely because it produces a familiar spelling; record original evidence.
3. Keep source spelling, candidate normalization and phonemic interpretation
   distinct. No phonemic value is inferred from the choice of Unicode cluster.
4. Preserve frozen first readings. Record changes in keyed review/reconciliation
   artifacts with original, proposed, reason, page and reviewer. Unresolved
   clusters remain explicit transcription holds, not silently accepted strings.
5. Keep detached scan/punctuation traces in raw evidence when removing them
   from a lexical candidate; do not infer a meaningful apostrophe from a dot.
6. Before canonical installation, inventory actual Unicode clusters, confirm
   NFC without dropping join controls, and exercise the configured display
   profile on every installed form. Test the observed omission class with
   reviewed examples and show that Original retains the source distinction.

The grammar owner, reverse-wordlist owner and independent comparison reviewer
were notified on 2026-09-27. Full independent Kokborok/further-comparison review
and the Santali review are applying this check. Brokskat's existing independent
review requires an addendum for this source-wide issue. The original 742-row
thematic wordlist must also be checked for this class during whole-book
reconciliation; prior installation alone is not evidence that it is unaffected.

## Counterexample: ordinary হ and false hasanta insertion

On 2026-09-27 an independent comparison against PDF99 row17 `হাগা` showed
that ordinary হ itself carries a lower diagonal in this font. A descending
stroke alone therefore cannot distinguish হ from হ্. Root withdrew approvals
for proposed `গোহ্‌র`, `যুহ্`, `দাহ্‌গি` and `দাহ্‌ঘি`; the additional proposed
`লেহ্` / `লেহ্-লেহ্` is also unconfirmed. This affects every chapter's হ-class
proposals, including further-comparison `গোহর`, not only these examples.

Compare matching source-font controls and require an additional distinct mark
before inserting hasanta on হ. Preserve the proposed strings as review history,
withhold approval, and do not automatically restore a u-vowel either: ordinary
হ, হ্ and হু must be distinguished from the original. All three active reviewers
were notified. Clear hasanta evidence on other consonants remains independently
reviewable; this counterexample does not authorize a global reversal.

## Finite font-control resolution, 2026-09-27

The completed keyed review is
`h-font-control-review/keyed-font-control-decisions-20260927.json`.
It covers 50 named candidate records with new image controls and carries forward
13 earlier reverse-series records explicitly as prior review evidence. Those
counts must not be described as 63 newly cropped independent checks.

The same-row Bengali `হাত` versus Chakma/Brokskat `হাত`, and Santali versus
Chakma `হাগা`, show that some target characters have an **additional** lower
mark beyond the ordinary হ tail. The ordinary-tail caution above remains valid,
but does not justify deleting these additional marks. The available source
does not settle whether every additional mark is a virama, another modified
হ, or a different orthographic convention. This is a finite, typed
`marked-ha-encoding` uncertainty, not a licence to infer a phoneme.

For whole-source assembly:

- Retain existing হ্ representations provisionally, with the exact page image
  and keyed encoding caveat. Preserve earlier literal readings and proposed
  alternatives in the audit. Do not silently introduce হ্র or a u-vowel.
- Where the control comparison rejects a joined হ্ম/হ্য and shows a full
  separate following ম/য, use the keyed provisional disjoined representation
  with its encoding caveat. This fixes joining, not phonological analysis.
- Attach `uncertain` and the typed reason to affected lexical records. Clearly
  identified lexical records may proceed under the checklist's standing
  uncertain-transcription policy; damaged or headword-less material still stays
  audit-only. Do not call the uncertain Unicode interpretation certified.
- Keep unresolved non-হ readings and language identities separate. The font
  decision does not approve them or certify the entire source.

Root inspected both control atlases. These controls resolve the blanket
ordinary-tail hypothesis and distinguish known joined versus disjoined shapes;
they do not replace the full section reviews or the final corpus/profile check.
