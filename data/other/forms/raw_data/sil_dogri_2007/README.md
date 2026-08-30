# SIL ESR 2007-017 Dogri

Canonical source: Jeremy D. Brightbill and Scott D. Turner, *A Sociolinguistic Survey of the
Dogri Language, Jammu and Kashmir*, SIL Electronic Survey Reports 2007-017, 29 pages.  The
official SIL archive record is <https://www.sil.org/resources/archives/9015>.

Appendix B (printed and physical pages 26-28) publishes one 210-prompt wordlist elicited in
Batote on 11 April 2005.  Items 11 `breast`, 23 `urine`, and 24 `feces` have blank response
cells, leaving 207 lexical attestations.  The report discusses five earlier survey lists from
Reasi, Ramnagar, Udhampur, Samba, and Billawar but publishes only their similarity percentages;
those five lists therefore yield no extractable lexical rows.

No OCR is used.  The appendix has a complete text layer in the legacy
`SILManuscriptIPA93` font.  `extract_dogri.py` reconstructs the two-column layout, asserts the
PDF/page/gloss/raw-form fingerprints, and converts the 20 used legacy bytes with the checked-in
subset of SIL's official `SIL-IPA93-2001.map` v14.  The rendered pages were checked for
affricate ties, aspiration, vowel quality, retroflexion, length, and nasalization.

`wordlist_snapshot.tsv` is the frozen source representation used by `import_dogri.py`, so the
installed data can be rebuilt without keeping the publisher PDF in the data repository.  Every
prompt, including the three blanks, remains in the per-record audit.

Batote is registered as a quality-C dialect using the modern GeoNames locality point
(33.118262, 75.308893), explicitly marked as an approximation rather than a source coordinate.
