# SIL ESR 2010-012: Pahari and Pothwari

`extract_pahari_pothwari.py` freezes Michael and Laura Lothers' *Pahari and
Pothwari: A Sociolinguistic Survey* (2010) from the exact 262-page SIL
publisher PDF. Appendix B.1 occupies printed pages 147–202 (physical PDF
pages 153–208) and prints 217 prompts for sixteen lists.

No OCR is used. The response cells have a complete positioned Doulos SIL text
layer. The extractor uses the fixed grid, font identity, and content-stream
order to preserve precomposed and overstruck Indological Phonetic Script
symbols. All 3,472 cells are frozen in `wordlist_snapshot.tsv`. Fourteen cells
print the code `AUS` in the otherwise invariant ninth (OSI/Osia) row; the raw
code is retained and the row is explicitly normalized to OSI.

Fourteen Pahari, Pothwari, and Mirpuri locality lists are in scope, yielding
3,038 installed responses. The 434 Abbottabad and Mansehra Hindko comparison
cells—including eighteen blanks at items 209–217—remain audit-only. Eleven
asterisked prompts are installed as lexical attestations but retain notes that
the source excluded them from its phonetic lexical-similarity calculations.
Those calculations are not historical cognacy claims, so this import creates
no etymological edges.
