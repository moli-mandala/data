# Census West Bengal Koda/Kora: analysis-only inventory

The official *Linguistic Survey of India—West Bengal (Part I)* includes
R. Senkuttuvan's Koda/Kora chapter on printed pp. 430–458 and a comparative
lexicon Koda column on printed pp. 731–756. The local, non-versioned PDF
cache is `tmp/pdfs/census-koda/LSI_WB_PART_I.pdf` with SHA-256
`d3d1f935612bad52cb79699c07702aa4efafd26a872163d8998782fde4639a63`.
`extract_cells.py` uses the table geometry to account for every numbered
prompt 1–500, extracting only the English prompt and Koda column into
`cells.jsonl`: 125 non-dash cells, 372 explicit dashes and three empty cells.
The 125 cells are positional candidates and have **not** received full
reading/notation review or been installed. A rendered check of printed p. 731
confirmed the first page's column alignment; text spacing still needs
per-cell visual review.

The [publisher's copyright policy](https://censusindia.gov.in/census.website/en/node/286)
says portal material may be reproduced free of charge after proper permission
by email. No such permission or open licence has been identified for this
PDF. Per the ingestion checklist's rights gate, this remains an analysis-only
inventory: there is no source CSV, YAML, bibliography registration, sound
profile or database build. The PDF and page renders stay in local cache.
The user forbids remote execution and database building, and neither occurred.
