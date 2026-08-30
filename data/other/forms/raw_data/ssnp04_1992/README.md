# SSNP volume 4: Pashto, Waneci, Ormuri (1992)

`extract_ssnp04.py` freezes Appendix B (printed pp. 79–146; PDF pp.
97–164) from the official SIL publisher PDF. The list inventory is printed on
p. 79 and the lexical tables occupy pp. 80–146.

The appendix has a complete positioned text layer, so no OCR is used. Its
forms use the legacy `SILDoulosNP` font. The extractor preserves the raw
keystrokes and decodes them to Unicode IPA using the existing SSNP decoder;
five volume-specific glyph behaviours were checked against the rendered pages
and the report's phonetic chart on printed p. 69.

The source prints 200 of its 210 numbered prompts for 34 Pashto locations,
one Waneci list, and one Ormuri list: 7,200 cells in total. It explicitly says
that missing numbers were excluded from the lexical-similarity count. Printed
`--` cells and the one genuinely blank cell are retained audit-only. All
lexical responses are source content, not controls. The report explicitly says
its similarity method does not identify genuine cognates, so no etymological
edges are created.
