# Appendix C extraction evidence

`snapshot_blocks.py` reads the pinned 115-page original with PyMuPDF,
retaining sorted native lines and positioned blocks for PDF93–108. It does
not emit accepted forms. `tagged-blocks-v5.json` is the current marker
inventory; earlier versions remain historical evidence, not alternative
accepted transcriptions.

`reconcile_scaffolds.py` compares the complete marker sequence with the
independent, unsorted-native-text extraction in `../scope/`. The recorded
check passes for all 637 events, including section, numbered unit and
question association. It accounts for all 126 phonetic blocks and ten
target-bearing answer blocks. Ten phonetic blocks are target questions,
not story clauses. This is structural agreement, not literal acceptance.

The script-g marker `\ɡ` in C.3 unit10 is preserved literally and recognized
as a gloss marker. C.4 unit3 has no printed free-translation marker.
Original review found that C.4 unit20 has a bare `g` without its backslash:
v5 recovers that boundary, giving 638 events. `check_bare_gloss_recovery.py`
verifies all 636 other old events unchanged and exact conservation of the
source lines split out of the contaminated phonetic block. The earlier
637-event agreement describes v4 and is not proof against shared blind spots.
C.5 free translations often cover several numbered subunits. C.2 also
contains multiple phonetic/gloss blocks within one numbered unit. Do not
zip whitespace tokens or assign one complete English sentence to every
subunit. C.5 target answers are delimited inside otherwise English answer
blocks and must not be discarded.

The source distinguishes near-close and close vowels in its phonetic key,
and its printed47 note8 describes differing vowel transcription systems.
The existing wordlist profile must not be presumed to establish a reviewed
conversion for all narrative material. A complete candidate symbol census
and explicit transcription decision remain required before installation.

No canonical CSV, registry, profile, database or durable identities are
modified by these scripts. Independent literal review is being prepared in
`../interlinear-review/`. Whole-report recovery also includes all comparison
lists and outside lexical examples; this directory is only its text evidence.
