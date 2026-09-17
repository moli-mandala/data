# Requested database rebuild — September 11, 2026

Shared CLDF and the local browser database are rebuilt, with cache version **32**. The browser database contains **603,702 lemma nodes**, 100,040,704 bytes expanded and 44,172,750 bytes compressed. No deployment, commit or push.

All **3,777 approved links over 3,731 records** and **nine new donor heads** survived compilation and browser conversion. Relations and ordered components were checked using the application's ID/alias codecs. SQLite integrity and compressed-stream checks passed; zero typed edges were dropped for unknown endpoints. Overlay row contents are unchanged from the pre-rebuild backup (byte ordering/serialization changed).

Browser QA passed at localhost:5173: Malvi badan links to Arabic badan; its donor page displays Bagheli, Malvi and Nimadi descendants with sources and locations; Malvi choṭobhai displays ordered choṭo and bhai components and their inherited ancestry. Frontend `npm run check`: zero errors, seven warnings.

All data compilation stages completed. `make all` exits 2 at the two previously reproduced manual-survey tests (Rajasthani record count and source-owned overlay rows). Earlier broader-suite failures remain documented in [ACCEPTED.md](ACCEPTED.md); the global suite is not clean.

Exact artifact paths, sizes, checksums and QA results: [browser-db-validation.json](browser-db-validation.json). Logs are saved in acceptance-logs: shared-data-build.log, shared-compiled-check.log, browser-transform.log, browser-pack.log and frontend-check.log. Pre-rebuild backups remain at `/tmp/central-db-rebuild-backup`. The old isolated validation checkout was removed after preserving its results and logs.
