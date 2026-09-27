# Nordic Digital LSI: bounded reuse discovery

Checked2026-09-26. Read-only research; no source ingestion, canonical changes, bulk downloads, remote jobs or database builds.

## Verified public resources

- [Språkbanken LSI corpus](https://sprakbanken.se/en/resources/lsi), DOI10.23695/d24m-ja14: CC-BY4.0 XML export,6.23MB, updated2020-08-25,1,193,437 tokens. This is the existing local `tmp/pdfs/lsi-sprakbanken.xml.bz2`; no second copy downloaded.
- [Project account](https://spraakbanken.gu.se/en/blog/20200901-griersons-linguistic-survey-of-india-as-open-access-digital-data-resource-for): the digitization used double keying. Its searchable/exported corpus excludes tabular material, including vocabulary and inflection tables. A separate167-item comparative vocabulary across292 varieties was still being prepared in2020. This historical statement does not establish its current publication status.
- [Developed resources](https://spraakbanken.gu.se/en/projects/digital-lsi/tools-and-resources) links Korp, CLLD grammatical-feature database, static feature maps and a semantic parser. These are not advertised as a downloadable aligned-specimen corpus.
- [lsilex](https://spraakbanken.gu.se/en/resources/lsilex), DOI10.23695/gmq4-g832, points to Karp. The [lexicon catalogue](https://sprakbanken.se/en/resources/lexicon?language=All&order=title&s=&sort=asc&t=) reports41 entries and Swedish. It is not evidence for the sought multi-language aligned-text dataset.
- [Lexibank LSI](https://github.com/lexibank/lsi) is a separate, available comparative-vocabulary dataset; Jambu already has its importer and60,533-form source. It does not supply the requested Rangri interlinear specimens.

## Actual local XML coverage

Streaming census of every corpus/page start tag, including indented tags, found2,374 page records across13 volume-parts:

|Volume-part|Page records|
|---|---:|
|3-1|319|
|3-2|201|
|3-3|172|
|4-1|305|
|5-2|63|
|6-1|39|
|7-1|102|
|8-1|150|
|8-2|337|
|9-1|129|
|9-2|117|
|9-3|83|
|9-4|357|

No volumeXI occurs. No9-2 page from238–259 occurs (nor230–265). Therefore this export cannot directly replace Rangri specimen transcription. It remains useful for lexical examples embedded in the included grammar/prose pages.

Machine-readable census and all extracted table URLs: `tmp/pdfs/lsi-sprakbanken-discovery-20260926.json`.

## Important separate-table lead

The XML itself supplies `page_tables_url` links distinct from `page_url` scan links. Example:

`http://demo.spraakdata.gu.se/lsi/tables/vol9-part2-tables.html#50`

The complete extracted set covers the same13 volume-parts, with paths `vol<volume>-part<part>-tables.html`. These HTML resources could contain precisely the material omitted by the export; their content has NOT been inspected successfully. XML absence is not proof of absence from that separate layer.

The Rangri worker independently confirmed the9-2 link and then successfully retrieved it using authorized network access. The old host redirects to `https://demo.spraakbanken.gu.se/lsi/tables/vol9-part2-tables.html` (HTTP200,278,631bytes). The earlier local DNS failures were sandbox restrictions, not evidence that the public resource was unavailable. Web-tool/archive failures likewise did not establish absence. The separate HTML content is now being checked by the Rangri worker.

## Authorized-access follow-up

The public [table directory index](https://demo.spraakbanken.gu.se/lsi/tables/) was successfully retrieved (HTTP200,3,870bytes). It lists exactly13 individual volume-part files, matching the XML coverage above, and a9.0MB aggregate `all-lsi-tables.html`. No volumeXI file is listed. The plausible volumeXI filename `vol11-part1-tables.html` returned a verified HTTP404. The aggregate was not downloaded because it is redundant with the indexed parts and outside this small-download check; absence ofXI from the directory does not certify every byte of the aggregate.

The [project directory](https://demo.spraakbanken.gu.se/lsi/) lists IE, SharedTask, clld, clld3, maps and tables. The tiny IE index lists an information-extraction PHP view and a26MB database (not downloaded). SharedTask is a2019 grammar-data-mining task page with an external ZIP link, not a documented interlinear specimen export. No additional volumeXI transcription link was located.

The Rangri worker parsed the successfully retrieved9-2 HTML:208 divs and108 distinct page IDs. Pages249–251 and254–256 are all absent; the relevant sequence jumps203→258. Grammar tables on54–58 are present in Unicode with structured cells. Thus this specific table export cannot replace the missing Rangri specimen transcription, though it can assist grammar extraction. Local copy: `tmp/rangri-specimens/nordic-vol9-part2-tables.html`.

Saved discovery-only index HTML totals approximately50KB under `tmp/pdfs/lsi-*-index-discovery.html`. No contact message was sent to project staff. No database was downloaded or built.
