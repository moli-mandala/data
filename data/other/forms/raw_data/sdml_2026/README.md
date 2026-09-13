# SDML lexical export, 2026-09-11

Official source: https://sdml.ac.in/data-analysed (Deccan College and Rajya Marathi Vikas Sanstha). Website declares CC BY-SA 4.0; see archived home.html. Snapshot hashes and exact download URLs are in snapshot.json. The export page combines lexical.csv with database.json; the latter is empty in this snapshot. data.js is archived for label comparison. Fieldwork dates and sound inventory are in methodology.html.

Run `python data/other/forms/raw_data/sdml.py` from the data repository for a proposal, then `--install`. No network needed. The importer checks all snapshot hashes. The immutable local key is source + district/taluka/village + export column + within-cell response position. Village spelling and positions are pinned to this release; future updates must reconcile against this snapshot rather than regenerate keys after reordering variants.

The methodology describes 277 surveyed villages, but the downloadable lexical snapshot exposes 271 rows; no data for the six absent sites are invented.

271 village rows × 73 response columns = 19,783 cells. The website's 69-feature map is not the same dimensionality as this export. 47,956 comma-delimited tokens yield 47,317 attestations, 606 missing-response tokens (empty/NA/dot), and 33 unresolved alternative/parenthetical/malformed tokens held outside the installed CSV. 269 villages have usable forms. Bhose (Sangli/Miraj) and Gondia (Gondia/Gondia) have no usable responses. All 271 source localities are registered beneath Marathi (`M`), with source coordinates (quality A); the survey samples Marathi as mother tongue or village contact language. Do not infer individual donor languages from this aggregated export.

Each exported column is preserved in keys and locators, including separate broom stimuli and male/female kin terms. Glosses follow export column identifiers/headers because several frontend data.js kinship labels are copy errors. Roof and bolt stimulus distinctions are retained. No POS, gender, donor, or cognacy is inferred. No graph edges are added merely for co-occurring responses. Native script is absent from this export and is not manufactured by matching to the map's variant list.

Original and Phonemic retain NFC source transcription: SDML describes it as IPA but uses Indological c/j (dental) versus č/ǰ (palatal). The sdml profile preserves these distinctions, converts g-shaped IPA ɡ to g and š to ś, composes underdots, and preserves aspiration and rare symbols. 37 forms with unusual characters retain the source reading with `uncertain` and a typed audit reason. Ambiguous alternative notation is excluded rather than expanded by guessing its scope. No OCR.

Audit contains every token, raw complete cell, original token, concept, site, frequency list, accepted count only when lists align, status and typed issues. 2,254 tokens have unavailable/unmatched frequency lists; these never get guessed counts. Repeated identical responses remain separate stable source attestations. Seeded 20-record inspection found 0 material parsing errors; sample-review.json documents edge cases.

No audio, narrative or grammatical dataset is included. Browser database refresh is a separate user-triggered step.
