# Hindko/Gujari tongue: legacy-font evidence

On 21 September 2026, visually inspected the publisher scan reproduced by SIL's Yumpu account:

- [Printed p. 212, Hindko item 10](https://www.yumpu.com/en/document/read/11893736/hindko-and-gujari-c-sil-international/234): **ǰib**, not **yib**, at JAM, BAL, SIG, SHP, MAN, KOH, ATK, PAG, WAP, DIK and TAL. PES has a separate **zubān** response and is excluded.
- [Printed p. 260, Gujari item 10](https://www.yumpu.com/en/document/read/11893736/hindko-and-gujari-c-sil-international/282): **ǰib** at KUN, CHT, DIR, SSW, TSW, GLT, KGH, SHZ, NAK and GJW; **ǰiv** at SAK; **ǰiv / ǰiβ** at CAK. IND has **zabān** and is excluded.

The raw extraction records `jib`, `jiv`, and `jiB`. `ssnp.py` leaves `j` unchanged and decodes `B` as `b`; `conversion/ssnp.txt` maps plain `j` to `y`. Thus the current compiled forms *yīb*, *yīv*, and *yīv / yīb* cannot be used as evidence for a historical palatal-approximant onset. The scanned affricate and labial alternants support the CDIAL 5228 *jihvā* family, whose full article explicitly includes regional *jib/jībh* and Kashmiri *zev*.

This audit resolves 23 exact records in pass002. It does **not** globally reinterpret the font, repair the importer/profile, replace source spellings, or rebuild anything. The Kohistan-volume *yīb* records require their own page check and are not included in this resolution. Other `j`/`B` words are not automatically accepted from this finding.

Access note: the Zabanha PDF was readable through web text extraction, but PDF screenshot retrieval failed and direct download timed out. The successful visual verification used the Yumpu page reader, enlarged and panned to the relevant table columns. No paywall or access control was bypassed.

## Further font checks for pass003

The same legacy-font plain `j` is printed **ǰ** (affricate) across the checked columns. This supports interpretation of the source's underlying text, without claiming a sound change from imported *y*.

- [Printed Hindko p. 218, item 63](https://www.yumpu.com/en/document/read/11893736/hindko-and-gujari-c-sil-international/240): root column visibly ǰaṛ, with distinct inflected ǰaṛi/ǰaṛa also printed. Pass003 selects only the bare ǰaṛ responses.
- [Printed Gujari p. 266, item 63](https://www.yumpu.com/en/document/read/11893736/hindko-and-gujari-c-sil-international/288): ǰaṛ in the first eleven localities, ǰaṛi in GJW, no IND answer. Pass003 selects the first eleven.
- [Printed Gujari p. 273, item 123](https://www.yumpu.com/en/document/read/11893736/hindko-and-gujari-c-sil-international/294): aǰ/āǰ throughout, CAK explicitly āǰ / aǰ. Both CAK alternatives share the adya family.
- [Printed Gujari p. 274, item 138](https://www.yumpu.com/en/document/read/11893736/hindko-and-gujari-c-sil-international/296): saǰi/saǰo, soǰi / soǰo, saǰɛ / saǰo, saǰā and geminated saǰǰo / saǰǰi visible. All are the right-hand adjective, with inflection preserved.

Pass003 man/husband and run analyses use this established font mapping and the source's cached appendix text; those individual printed columns were not separately visually inspected. The exact original strings are retained in `ssnp-next-original-records.json`. The child jātak/jākat group remains held for morphological/transmission evidence even after its onset is resolved. No ingestion or profile rewrite was performed.
