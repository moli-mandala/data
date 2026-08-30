# SIL ESR 2008-003: Bishnupriya wordlists

This directory installs Appendix B.3 of Amy Kim and Seung Kim's *Bishnupriya (Manipuri)
Speakers in Bangladesh: A Sociolinguistic Survey* (SIL Electronic Survey Reports 2008-003,
official archive 9100). The appendix is a 307-item list with responses from six Bishnupriya
villages and a standard Bangla comparison list. One printed response represents every site code
inside its brackets.

The official SIL PDF renders correctly in a browser, but its legacy phonetic font is not exposed
as ordinary Unicode and the command-line endpoint currently returns a Cloudflare HTML challenge.
The extraction therefore uses the fixed-layout text transcript and page rasters in a public
Slideshare copy only as a mechanical scaffold; the SIL archive record and report remain the
bibliographic authority. The checked-in transcription is reproducible from a locally saved copy
of that HTML:

```sh
python data/other/forms/raw_data/sil_bishnupriya_2008/extract_bishnupriya.py \
  /tmp/bishnupriya-slideshare.html
python data/other/forms/raw_data/sil_bishnupriya_2008/import_bishnupriya.py
```

`slideshare_pua_used.tsv` pins all fourteen PUA glyphs and their complete 947-occurrence census.
`source_page_images.tsv` pins the eighteen rendered wordlist pages used to verify those glyphs,
superscript aspiration, ambiguous word boundaries, and the source's single lowercase `o` site-code
typo. The extractor verifies 307 headings, 746 printed response records, 2,099 expanded
attestations, 161 aspiration markers, nine headings with no responses, and zero unparsed or
unmapped symbols. Bangla controls and empty prompts remain audit-only.
