# SIL survey scan provenance

Checked 2026-08-28.  SIL Global is the authoritative publisher.  Every report below is freely
published, but `www.sil.org/system/files/...` sits behind a Cloudflare bot challenge: plain HTTP
clients receive `HTTP 403` with a `Just a moment...` interstitial, with or without a session
cookie and referer.  No copies exist on the Internet Archive (`0` hits for each report ID), and
the `silbangladesh.org` mirror serves only metadata, re-pointing at the same protected URLs.
An ordinary interactive browser passes the challenge, so acquisition is a manual download.

The report IDs below were resolved from the SIL landing pages, which are not challenged; keep them
so acquisition can be resumed without rediscovery.

## Dravidian — the Nilgiri/Wayanad/Kerala and Central Dravidian gap

| Report | Title | Coverage relevant to Jambu | URL |
|---|---|---|---|
| silesr2015-029 | Tribes of Idukki, Kerala | 15 wordlists: Muthuvan (8 locations), Mannan (7), plus one each Paliyan and Mala Pulayan | <https://www.sil.org/system/files/reapdata/72/12/64/72126465316636129311021892758240500243/silesr2015_029.pdf> |
| silesr2015-028 | Nilgiris survey | Kurumba wordlists across varieties (reported 84% lexical similarity); largest set from Irular | <https://www.sil.org/system/files/reapdata/12/30/35/123035517420667367799205409945550701297/silesr2015_028.pdf> |
| silesr2018-010 | A Sociolinguistic Survey of Nilgiri Irula | Three Irula varieties: Mele Nadu, Vette Kada, Northern | <https://www.sil.org/system/files/reapdata/63/44/66/63446659291996404202507810050484303350/silesr2018_010.pdf> |
| silesr2019-005 | A Sociolinguistic Survey of the Mudhili Gadaba People of Andhra Pradesh | Mudhili Gadaba wordlists plus recorded text testing | <https://www.sil.org/system/files/reapdata/58/58/42/58584222668822059550364150793136190163/silesr2019_005.pdf> |

Landing pages: entries `69566`, (2015-028 not resolved), `76656`, `81415`.

## Bangladesh — continuing the Kim/Ahmad/Kim/Sangma series

`kim-kim-ahmad-sangma2010santali-cluster`, `kim-ahmad-kim-sangma2011hajong` and
`kondakov2013rabha` are already installed from this series.

| Report | Title | URL |
|---|---|---|
| silesr2011-023 | The Koch of Bangladesh: A Sociolinguistic Survey | <https://www.sil.org/system/files/reapdata/11/94/41/119441625785923242746341027576530217952/silesr2011_023.pdf> |
| silesr2011-040 | The Kurux of Bangladesh: A Sociolinguistic Survey | <https://www.sil.org/system/files/reapdata/14/77/38/147738795439636144307622558968035823325/silesr2011_040.pdf> |
| silesr2011-038 | The Tripura of Bangladesh: A Sociolinguistic Survey | <https://www.sil.org/system/files/reapdata/70/33/17/70331767982612912928852534822848351408/silesr2011_038.pdf> |
| silesr2012-007 | The Garos of Bangladesh: A Sociolinguistic Survey | <https://www.sil.org/system/files/reapdata/10/03/24/10032476061994562022822003027411828102/silesr2012_007.pdf> |

Landing pages: `41580`, `41654`, `41655`, `47762` (also mirrored at
`silbangladesh.org/resources/archives/<id>`).

Do not conflate silesr2011-033 with the Bangladesh Koch report.  It is the separate data appendix
to *Koch Dialects of Meghalaya and Assam: A Sociolinguistic Survey* and its eleven India lists are
accounted for in the India table below.  Check each Bangladesh report for a genuinely matching
standalone data appendix before extracting wordlists out of report tables.

## South Asia survey census

This is the working discovery ledger for the goal, not just a list of files already downloaded.
It reconciles SIL's current archive catalog with Glottolog's `wordlist`/`comparative` source types,
the five-volume *Sociolinguistic Survey of Northern Pakistan* (SSNP), and Jambu citation keys.
`inspect` means the bibliographic source is confirmed but its lexical appendix still has to be
checked directly; a sociolinguistic report is not assumed to contain publishable forms merely from
its title.

### India

| Source | Lexical scope | Jambu state |
|---|---|---|
| ESR 2007-017 Dogri, Jammu and Kashmir | Appendix B Batote wordlist | **installed 207 forms** (210 prompts; items 11, 23, and 24 are blank and audit-only; five earlier comparison lists are reported only as similarity percentages, not published forms) |
| ESR 2008-013 Jaunsari | 210-item IPA lists; legacy-font text layer | **installed** (1,619) |
| ESR 2008-006 Indian Sign Language | final 245-item English elicitation list and item-level pairwise similarity judgments for Kolkata, Chennai, Delhi, Hyderabad, and Mumbai; the underlying videotaped signs and representative city wordlists are not published or linked | **inspected; no lexical rows to ingest** (exact 121-page publisher PDF recovered from its 2017-05-16 Wayback capture, SHA-256 `00ae89e7fcfee81dd46c6895f338dd989ed9dd7ebe99cf8604c946eaf18a426f`; all pages text-extracted and rendered; Table 2 on physical pp. 11-12 and all six Appendix G pages 58-63 visually checked; they contain English prompts plus numeric analysis only, with no sign tokens, notation, item-keyed images, or video links; 245 prompt-only rows, 2,450 pairwise judgment slots, and 245 means excluded rather than coerced into Jambu's spoken-form schema) |
| ESR 2008-018 Pardhan | questionnaire/interview survey summary; no wordlist collection reported and no lexical forms, prompts, similarity matrix, denominators, or appendix published | **inspected; no lexical rows to ingest** (complete official PDF: one title page plus four numbered report pages; all five pages text-extracted, rendered, and visually checked; report ends with References on numbered p. 4) |
| ESR 2009-010 Nagarchi | attempted wordlist collection, but the report explicitly says no consultants—including elderly speakers—could provide Nagarchi wordlists; no prompts, forms, similarity matrix, denominators, or appendix published | **inspected; no lexical rows to ingest** (complete five-page official PDF text-extracted, rendered, and visually checked; failed collection documented in §5 on p. 4; report ends with References on p. 5) |
| ESR 2009-011 Malvi | 210-item comparative wordlists: 30 target Malvi sites plus 8 published controls | **installed** (6,894 target forms; 1,891 control/missing records, 37 target `By Name` cells and 90 target disqualified-prompt cells audit-only) |
| ESR 2011-048 Bagri | comparative wordlists | **installed** (`20230521-rajasthani.csv`) |
| ESR 2011-033 Koch dialects of Meghalaya and Assam | standalone Unicode data appendix: standard Garo and Rongdani Rabha controls plus nine Margan, Harigaya, Wanang, Tintekiya and Koch-Rabha site lists, 210 prompts | **installed** (2,281 forms across eleven sites; exact page/item/site audit; no OCR or legacy-font decoding and no unresolved forms) |
| ESR 2012-002 Nimadi | 210-item comparative framework; 206 prompts printed across 13 target and 5 comparison lists | **installed** (2,826 target forms; 1,207 comparison records and 59 target omissions/no-entry cells audit-only) |
| ESR 2012-015 Kurumba dialects | Appendix C scan: nineteen 550-item lists (fifteen Kurumba/Kuruba-labelled survey lists plus four Tamil/Kannada/Badaga/Vakkaliga comparanda), including Alu, Jennu and Betta Kurumba; 10,450 conceptual cells | **shared source integration complete; consolidated build pending** (all 10,450 cells manually reviewed; 3,204 target attestations installed, 1,534 control attestations audit-only, and 5,710 printed-dash blanks audit-only; one ambiguous Pudukkottai target at physical p. 239 / printed p. 234 / item 20 and one illegible Kotagiri Alu target at physical p. 261 / printed p. 256 / item 25 remain explicitly unresolved and excluded; nineteen list/site rows, exact reference and sound-profile routing, and focused integration tests complete; OCR/PDF/prior data supplied or verified no reading) |
| ESR 2012-016 Konda Dora | Appendix 9.5 scan: two Konda/Kubi target lists (Koraput and Visakh) plus Telugu and Adivasi Oriya controls; 214 printed prompt rows per list, including visibly duplicated source item 212, for 856 conceptual cells / 428 target cells | **installed source package; shared integration complete pending consolidated build** (452 Konda forms from 385 attested target cells; 43 target blanks, 86 control blanks, and 342 attested control cells audit-only; all 856 cells manually transcribed/reviewed; no unresolved readings; OCR locator-only) |
| ESR 2012-029 Rajasthani overview + vols. 2–6 | Mewari, Hadothi, Mewati, Dhundari/Shekhawati, Marwari/Merwari/Godwari | **installed** (`20230521-rajasthani.csv`) |
| ESR 2013-014 Garasia dialects | reports 13 × 221-item collections but publishes only lexical-similarity percentages; no forms or appendix | **inspected; no lexical rows to ingest** (official PDF archive 54606, pp. 11–15; the complete Kharod Bhili list is already republished and installed from ESR 2018-011) |
| ESR 2013-015 Sambalpuri | reports four 210-item Sambalpuri collections plus one Standard Oriya comparison, but publishes only site metadata and ten lexical-similarity percentages; no prompt list, IPA responses, denominators, or lexical appendix | **inspected; no lexical rows to ingest** (complete 10-page official PDF; §2.2 and Tables 1-2 on printed pp. 5-6; report ends with References on printed p. 9) |
| ESR 2013-016 Rabha dialects of Meghalaya and Assam | Appendix B.3 prints Rongdani and Maituri phonetic tables on physical pp. 22-34 / printed pp. 18-30; the report describes a 210-item elicitation instrument but publishes 194 prompt rows after §B.1's explicit omission of problematic items, for 388 conceptual cells | **installed; census gap corrected and independent manual re-audit complete** (legacy Unicode extraction has 400 expanded rows: 205 Rongdani and 195 Maituri; source-local audit independently reviewed S001-S194 / all 388 cells from 600/1200-dpi renders: 387 attested, one explicit `no data` source blank, zero ambiguous, illegible, or unresolved readings, and 399 lexical response occurrences; exact archived SIL PDF SHA-256 `f690f404b793c601882b06940557a748e7932a9ca4afa28358fd75ca4396d02b`; existing forms were inventoried only after each manual chunk was frozen and supplied no reading) |
| ESR 2013-004 Northern Dhule Bhils | Appendix C image scan: twelve Vasave/Bhilori/Noiri/Pauri target lists plus Toranmal Nahali control; 210 prompts × 13 lists = 2,730 conceptual cells / 2,520 target cells | **shared source integration complete; consolidated build pending** (all 2,730 cells manually reviewed; 2,497 resolved target forms installed with immutable keys and exact locators; 21 target blanks, target ambiguities at item 10/KEL and item 31/MUN, and all 210 Toranmal control cells including its item 74 ambiguity remain audit-only; no illegible or unreviewed cells; exact Noira/Bareli reconciliation, parent/site registries, reference, dedicated profile route, and focused integration tests complete; OCR remained locator-only) |
| ESR 2015-012 Noira Bhils | Appendix A typeset Unicode: fourteen Noiri/Barutiya/Dungra Bhili/Nahali/Kotli/Gujari/Nihali/Korku regional lists plus Gujarati, Marathi, and Hindi controls; 210 prompts × 17 lists = 3,570 conceptual cells / 2,940 regional cells | **shared source integration complete; consolidated build pending** (all 3,570 cells manually reviewed: 3,526 attested and 44 blanks; 2,714 forms from eleven genuinely new target lists installed with unique immutable keys; 630 Dhule-republication cells / 834 responses and 630 Gujarati/Marati/Hindi control cells / 837 responses remain audit-only; KNA/KTA are provisional Noiri dialects preserving the Kotli and Adivasi Bhil-Taradi labels with blank Glottocodes/coordinates and no historical Kotali/Khandesi equation; zero ambiguous, illegible, or unresolved cells; exact reference, dedicated 54-grapheme profile route, and focused integration tests complete) |
| ESR 2015-014 Kolami | sociolinguistic questionnaires and aggregate comparison discussion; no published wordlist appendix or item-level lexical responses | **inspected; no lexical rows to ingest** (archive 69101; 35-page official PDF topology checked) |
| ESR 2015-016 Adi | Appendix B typeset Unicode: Minyong, Bori, Ramo, Milang, Pailibo, Ashing, Padam, Shimong, and Bokar; 307 prompts × 9 target lists = 2,763 conceptual target cells | **shared source integration complete; consolidated build pending** (all 2,763 cells manually reviewed: 2,670 attested and 93 explicit blanks; 2,770 forms installed with immutable keys and exact locators, including 94 double- and three triple-response cells; four parents and nine source-qualified sites registered with unsupported coordinates blank; exact reference and dedicated 42-symbol profile route complete; zero ambiguous, illegible, or unresolved readings; Unicode text remained locator-only and every retained form was checked against rendered pages) |
| ESR 2015-026 Bhumij | Appendix B.3 typeset Unicode: ten Bhumij/mixed target lists plus eight Mundari, Ho, Santali, and Oriya controls; 210 prompts × 18 lists = 3,780 conceptual cells / 2,100 target cells | **source-local ingest complete; shared integration pending** (all 3,780 cells visually reviewed: 3,690 attested, 90 explicit blanks, 3,876 expanded responses, 2,100 target candidates, zero pending or unresolved; OCR/PDF text supplied or verified no reading; one legible source-marked `(?)` qualifier retained; all 1,050 cells in five same-elicitation Ho 2024 republications reconciled with complete status parity and excluded from the later route) |
| ESR 2015-028/029 Palakkad and Idukki | 40 site lists | **installed** (4,710 + 4,630) |
| ESR 2017-013 Western Indo-Nepal Tharu | Appendix B typeset Unicode: fifteen Bhuksa/Rana/Thakur/Kathoria/Sunha/Dangora/Dang/Chitwan Tharu target lists plus Standard Hindi control; 210 prompts × 16 lists = 3,360 conceptual cells / 3,150 target cells | **shared source integration complete; consolidated build pending** (all 3,360 cells independently reviewed from source images: 3,261 attested and 99 explicit blanks; 3,560 target forms installed with immutable source entry keys; 290 Hindi-control forms remain audit-only; zero lexical ambiguities or illegibles; all 420 `RNS` locality assignments remain explicit uncertainties and 486 installed RNS alternatives carry typed uncertainty reasons; split RNS dialect metadata, exact preservation profile/routing, reference provenance, and focused integration tests are installed; no reading originated from OCR or legacy data) |
| ESR 2018-005 Vaagri Booli | publishes only site metadata and aggregate lexical-similarity results from eight Vaagri sites; no prompt list, response forms, or lexical appendix | **inspected; no lexical rows to ingest** (archive 75337; complete 22-page official PDF; SHA-256 `caba90fd15da738808913562f55a64f74120c118da62c02e4acdc2307db316dc`; distinct from the installed Vaagri Boli dictionary source) |
| ESR 2018-010 Nilgiri Irula | image-only IPA, eleven target sites plus seven neighbouring-language/comparison lists | **installed for the complete target scope** (2,054 target forms; 15 target gaps and 1,319 control records audit-only; no target reading originates from OCR; the 187-prompt comparison column `MAD` remains untranscribed/audit-only because the report never expands its language code, so no unsupported language identity was inferred) |
| ESR 2018-011 Bareli/Pauri | 210-item IPA lists from 30 regional sites plus three standard controls | **installed** (6,320 regional forms; 789 control responses, 105 explicit gaps and 33 disqualified cells audit-only) |
| ESR 2018-009 Western Arunachal / Monpa | 307 concepts across 30 Monpa, Kho-Bwa, Hruso and Miji-related lects | **installed** (9,279; complete CC-BY Lexibank v3.0 source matrices, including 1,066 forms silently absent from its generated CLDF; 142 explicit gaps audit-only) |
| ESR 2019-005 Mudhili Gadaba | image-only IPA, seven target sites | **installed** (1,538) |
| ESR 2019-006 Lahul Valley | 210 concepts; 22 newly collected lect/site lists plus 5 prior/control lists | **installed** (5,027 target forms; 1,150 prior/control responses and 29 target `no entry` records audit-only) |
| ESR 2019-007 Ladakhi–Bhoti intelligibility | recorded-text tests, RTT results, and questionnaire responses; no published wordlist appendix or item-level lexical responses | **inspected; no lexical rows to ingest** (archive 84082; complete 50-page official PDF; SHA-256 `c9d93a98fef0ff7a9d6eafd1278f823d39f897861c2fca7cf410d4311b679223`; only aggregate references to earlier Lahul/Leh wordlist comparisons, whose Lahul source is already installed) |
| Beine 1994 Gondi survey | 46 × 210 site/concept matrix | **installed** via the Rama–Çöltekin–Sofroniev digitisation (10,264) |
| Chamberlain, Chamberlain & Pavey 1998 Kinnauri manuscript | secondary metadata reports comparative wordlists from 19 Kinnaur and comparison lects; exact topology cannot be verified without the manuscript | **exact-source search exhausted; acquisition request required** (Glottolog ref 586488 / `hh:hvwsoc:Chamberlain:Kinnauri`; exact `chamberlain_kinnaur1998.pdf`, 79 pp.; absent from current SIL, OLAC, IA/Wayback, WorldCat, HathiTrust, Google Books, Open Library, LOC, MPI public resources, and SEAlang/SALA; ASJP 40-item derivative explicitly excluded) |
| JLSR 2021-009 Kullu District | Appendix C image-only IPA lists from sixteen Kullui, Inner Seraji and Outer Seraji sites | **source-local ingest complete; consolidated build pending** (2,963 manually transcribed variants from 2,753 responsive cells; 415 blanks and one layout label audit-only; all 3,168 handwritten response cells manually inspected; no installed form originates from OCR) |
| JLSR 2021-012 Kannauji | 210-item lists from fourteen target and comparison locations | **installed** (3,033 target forms in `20230526-kannauji.csv`; comparison lists excluded by that import) |
| JLSR 2021-020 Ahirani bilingualism | reports two 210-item Ahirani lists (Akkalkuva and Dhule) plus Marathi comparison, but publishes only three aggregate similarity percentages and denominators; no prompts, responses, IPA, item-level judgements, or appendix | **inspected; no lexical rows to ingest** (complete 15-page official PDF text-extracted, rendered, and visually checked; Tables 1–2 on numbered pp. 5–6; report ends with References on numbered p. 9; official SIL search found no separate Ahirani wordlist appendix) |
| JLSR 2021-028 Mayurbhanj survey summary | reports eight newly collected Mundari, Bhumij, Birhor, Mahali/Santali and Mohanta lists compared with twelve earlier/dictionary lists, but publishes only a 20-list triangular matrix of 190 lexical-similarity percentages; no prompts, responses, denominators, or lexical appendix | **inspected; no lexical rows to ingest** (complete 14-page official PDF; §5/Table 2 on printed p. 4; report ends with References on printed p. 8) |
| JLSR 2021-029 Koya dialects | Appendix E image-only comparative lists from seven Koya/Gondi/Madia sites plus Telugu and Oriya controls | **source-local ingest complete; consolidated build pending** (1,438 manually transcribed variants from 1,401 responsive target cells; 69 missing target slots and 420 control cells audit-only; all 1,840 printed cells manually inspected; no installed form originates from OCR) |
| JLSR 2021-034 Dhurwa | Appendix B typeset comparative list: 200 prompts × 5 response columns = 1,000 conceptual cells; four named Dhurwa columns plus one consistently unlabeled column | **source-local ingest complete; shared integration pending** (all 1,000 cells manually reviewed: 995 attested and 5 explicit blanks; 1,008 expanded responses; 809 staged forms from 800 known-target cells; zero ambiguous or illegible readings; the consistently blank-header fifth column remains neutral U5, with 200 cells / 199 responses retained audit-only; canonical SIL URL pinned via its exact 2024-07-06 Wayback capture after the live endpoint returned Cloudflare HTML) |
| JLSR 2021-040 Korku language area | Appendix F image-only scan: eight Korku locality lists plus one Nihali comparison list, 210 cells each / 1,890 cells total | **installed source package; shared integration complete pending consolidated build** (1,521 rows from 1,463 attested target cells; 216 target blanks and one illegible target cell excluded; all 210 Nihali controls audit-only; every cell manually transcribed/reviewed; OCR locator-only) |
| JLSR 2021-050 Amri Karbi | Appendix B.3 typeset Unicode: three Amri Karbi and twelve Karbi target lists plus Khasi and Assamese controls; 307 prompts × 17 printed lists = 5,219 conceptual cells | **installed source package; shared integration complete pending consolidated build** (5,092 installed forms after 237 exact repeated target occurrences are retained audit-only; 631 control responses and six blanks excluded; all 5,966 printed records visually checked; zero unresolved readings) |
| JLSR 2021-056 Desia | Appendix B.5 typeset Unicode: nineteen Desia/Kotia-Adivasi Oriya target lists, 210 prompts × 19 = 3,990 conceptual cells | **installed source package; shared integration complete pending consolidated build** (4,655 installed forms; 38 explicit blanks and three exact repeated readings audit-only; all 4,696 response lines visually checked; 542 combining-mark positions corrected from direct page evidence; zero unresolved readings) |
| JLSR 2021-063 Tagin and Puroik | Puroik, Bugun, Tagin and Nyishi comparative lists from sixteen sites | **installed** (4,939 target and comparison attestations in `20260813-tagin-puroik.csv`) |
| JLSR 2022-004 Bonda/Didayi | Appendix B typeset Unicode: four Bonda and five Didayi target lists plus Gutob, Parenga, Rona Desiya, and Oriya controls; 210 prompts × 13 lists = 2,730 conceptual cells / 1,890 target cells | **installed source package; shared integration complete pending consolidated build** (1,938 forms from 1,836 attested target cells; 36 cells at four disqualified prompts, 17 explicit target gaps, one physically omitted target row, and all 840 comparison cells audit-only; every cell visually checked; OCR not used) |
| JLSR 2022-005 Bonda further survey | Appendix A typeset phonetic tables: three Upper Bonda target lists plus eight Lower Bonda/Didayi/Gadaba/Parenga/Rona/Desiya/Oriya comparanda; 210 prompts × 11 lists = 2,310 conceptual cells / 630 target cells | **installed source package; shared integration complete pending consolidated build** (all 2,310 cells manually reviewed: 2,259 attested, seven blanks, 44 disqualified, 2,394 expanded responses, zero unresolved; 644 forms installed from 616 attested target cells; two target blanks and twelve disqualified target cells excluded; all 1,680 comparison cells audit-only; standalone 210-item reconciliation makes the checked 2002 Dumripada list current while preserving the superseded 1997 readings and citations in audit) |
| JLSR 2022-014 Korwa and Kodaku | Appendix B.5 typeset comparative tables: nine Korwa and nine Kodaku target lists plus seven Asuri, Birjia, Mundari, Tanmai and Sadri controls | **source-local ingest complete; consolidated build pending** (4,458 target rows from 3,730 attested target cells; 50 blank/unlisted target cells, 1,453 attested controls, 17 blank/unlisted controls, and two undefined-code responses audit-only; all 2,900 response rows visually checked and all 5,250 conceptual cells audited; no OCR-derived form installed) |
| JLSR 2022-015 Bagheli | Appendix B.4 image-only tables: eighteen Bagheli locality lists plus Standard Hindi control | **source-local ingest complete; consolidated build pending** (5,828 manually transcribed target occurrence rows, yielding 5,829 compiled forms after one printed comma-alternative expansion; 283 Hindi-control occurrences, 24 non-lexical `by name` occurrences, 47 conceptual blanks and two site-unassigned lines audit-only; all 3,990 conceptual cells manually inspected; no installed form originates from OCR) |
| JLSR 2023-002 Eastern Gujari | Appendix B typeset Unicode: eight new India Gujari lists plus six Pakistan Gujari lists republished from SSNP vol. 3 and one Urdu control; 210 prompts × 15 lists = 3,150 conceptual cells / 1,680 new target cells | **installed source package; shared integration complete pending consolidated build** (1,753 new Indian forms from 1,655 attested target cells; 25 target blanks, 1,254 attested SSNP-reprint cells, 208 Urdu responses, eight non-target blanks, and one exact duplicate target alternative audit-only; all 3,150 cells visually checked; no unresolved readings) |
| JLSR 2024-007 Kudiya | two target Kudiya lists plus a mixed comparison list | **installed** (409 target forms in `20260813-kudiya.csv`; comparison list excluded) |
| JLSR 2024-009 Ho dialects | Appendix D.3 image scan: fourteen new 1989 Ho field lists, three republished Ho lists, five Bhumij, two Mundari, two Santali, and one Oriya control; 210 prompts × 27 rows = 5,670 conceptual cells / 2,940 new target cells | **source-local ingest complete; shared integration pending** (all 5,670 cells manually reviewed; 2,900 resolved new-target forms staged; republished and comparison lists retained audit-only; three unresolved readings excluded; OCR locator-only) |
| JLSR 2024-011 Haryanvi | 210-item image-only IPA tables: six Haryanvi lists plus Braj/Haryanvi, Hindi, Punjabi and Baghati Pahari comparisons | **source-local ingest complete; consolidated build pending** (1,553 manually transcribed target variants from 1,238 responsive cells; 21 target blanks, one elicitation-only cell and 840 comparison cells audit-only; all 1,260 target cells manually inspected; no installed form originates from OCR) |
| JLSR 2025-005 Mavilan Tulu | Appendix A typeset phonetic lists: three Mavilan Tulu targets plus Malayalam Standard, Tulu, and Kodava Standard controls; 208 prompts × 6 lists = 1,248 conceptual cells / 624 target cells | **source-local ingest complete; shared integration pending** (all 1,248 cells manually reviewed: 1,230 attested and 18 explicit blanks; 615 target forms staged; controls and blanks audit-only; zero ambiguous, illegible, or unresolved; direct inspection shows physical p. 38 ends at item 208 and contains no items 209--210) |

### Bangladesh

| Source | Lexical scope | Jambu state |
|---|---|---|
| ESR 2007-013 War-Jaintia | wordlists | **installed** (2,030; official SAG-IPA mapping, no OCR) |
| ESR 2008-002 Meitei speakers (archive 9145) | wordlists | **installed** (2,406; official SAG-IPA mapping, no OCR) |
| ESR 2008-003 Bishnupriya speakers (archive 9100) | wordlists | **installed** (1,801; legacy font recovered against page images) |
| ESR 2010-006 Santali cluster | comparative wordlists | **installed** (4,882) |
| ESR 2011-023 Koch | 307-item bracketed-site-code appendix: four Bangladesh Koch sites, two Garo comparisons, and Standard Bangla | **manual review and shared source integration complete; consolidated build pending** (all 2,149 conceptual cells independently reviewed from physical pp. 43-62: 1,780 attested, 25 explicit blanks, 225 ambiguity-only legacy-modifier cells, and 119 globally not-used cells; 2,159 expanded audit rows preserve ten printed overlaps/variants; 1,017 unique target forms installed while 772 controls, 226 ambiguous rows, 25 blanks, and 119 not-used rows remain audit-only; seven sites, exact reference, and explicit 44-codepoint `sil-bangladesh` profile route registered; every retained reading came from page images, never OCR/PDF text, legacy data, or installed forms) |
| ESR 2011-025 Kuki-Chin communities | comparative wordlists | **installed** (3,235 Bangladesh attestations across five languages and ten sites; 640 controls audit-only; 65-glyph SAG-IPA recovery, no OCR) |
| ESR 2011-038 Tripura | comparative wordlists | **installed** (8,997) |
| ESR 2011-040 Kurux | 307-item bracketed-site-code appendix: five Kurux sites in Bangladesh/West Bengal plus Standard Bangla | **shared source integration complete; consolidated build pending** (exact 90-page PDF and all 19 wordlist pages pinned; all 1,842 conceptual cells reviewed: 1,661 attested, 136 blanks, 72 not-used, plus 27 retained variant rows = 1,869 rows; 1,365 target attestations installed with immutable source entry keys; 296 Bangla control forms remain audit-only; zero unresolved, ambiguous, or illegible readings; exact profile, five target-site rows plus audit-only control metadata, reference provenance, and focused integration tests pass) |
| ESR 2011-042 Hajong | comparative wordlists | **installed** (3,311) |
| ESR 2012-007 Garo | 307-item bracketed-site-code appendix: Garo, Koch, Megam and Lyngngam lists from Bangladesh and India plus Standard Bangla | **partial legacy-font recovery; manual completion active** (4,444 attested cells installed; exact 212-page PDF and all 42 wordlist pages rendered; items 1-155 independently reviewed: 2,635 conceptual cells = 2,556 ordinary attestations, one source-conflict cell with an attestation, 61 blank-only cells, and 17 not-used cells for whole-item 152; 2,728 attested response occurrences; item 12/site `p` remains the sole unresolved source conflict; items 156-307 pending; OCR/PDF text, legacy data, and installed forms supplied no reading) |
| SIL Bangladesh 2007 Chak | *The Chak of Bangladesh: A Sociolinguistic Study*, Maggard, Sangma and Ahmad, ix + 56 pp.; the official SIL Bangladesh list names the report and the official program chart marks Chak **Done**, but neither exposes a file/archive link; independent descriptive literature explicitly reports extracting IPA wordlists from it and characterizes it as providing lexical data for four Cak varieties | **missing lexical candidate, high confidence; official locator confirmed** (publisher PDF/archive record still unavailable; primary wordlist pages, four-way labels and topology remain unverified; exact acquisition audit pinned in `sil_chak_2007_discovery.json`; secondary forms are not transcription evidence) |
| SIL Bangladesh 2007 Chittagonian-speaking Community | *A Sociolinguistic Survey of the Chittagonian-speaking Community*, compiled by Loren Maggard, Mridul Sangma and Sayed Ahmad; field researchers Sayed Ahmad and Mridul Sangma; February 2007; title, responsibility and date visually verified from the primary SIL Bangladesh cover | **unclassified acquisition gap; primary cover located** (publisher PDF/archive record, extent, series ID and presence or absence of lexical/IPA wordlists remain unverified; exact acquisition audit pinned in `sil_chittagonian_2007_discovery.json`; absence of indexed lexical evidence is not evidence of absence) |
| SIL Bangladesh 2007 Chakma and Tanchangya | *A Sociolinguistic Survey among the Chakma and Tanchangya Communities*, Maggard, Sangma and Ahmad, 2007, Dhaka manuscript; title, responsibility, year and place are from a secondary bibliography, not a primary cover or title page | **unclassified acquisition gap; comparison method reported secondarily** (the official survey-program chart marks Chakma **Done** and lists Tanchangya immediately beneath it; the archive browse exposes four Chakma and four Tangchangya records, but their catalog pages are inaccessible and the report is absent from the public eleven-report list; Clifton 2013 says the study used wordlist comparison plus recorded-text tests, but this does not establish that the manuscript publishes lexical/IPA forms; primary report, publisher, extent, series/archive ID and every wordlist page/list/cell remain unverified in `sil_bangladesh_2007_unlisted_reports_discovery.json`) |
| SIL Bangladesh 2007 Marma and Rakhine | *The Marma and Rakhine Communities of Bangladesh: A Sociolinguistic Survey*, Maggard, Ahmad and Sangma, 2007, Dhaka: SIL Bangladesh; title, responsibility, year and publisher are from a secondary bibliography, not a primary cover or title page | **unclassified acquisition gap; no independent published-wordlist evidence** (the official survey-program chart marks both Marma and Rakhaine **Done** and the archive browse exposes three Marma and one Rakhine records, but the catalog pages are inaccessible and the report is absent from the public eleven-report list; the only recovered related official archive object is the distinct 2014 Davis thesis, entry 94638; primary report, extent, series/archive ID and every wordlist page/list/cell remain unverified in `sil_bangladesh_2007_unlisted_reports_discovery.json`; absence of indexed lexical evidence is not evidence of absence) |

The official SIL Bangladesh survey landing list contains eleven named reports.  Every title is now
accounted for in `sil_bangladesh_archive_candidate_audit.csv`: six installed, three partial manual
recoveries, one missing lexical candidate (Chak), and one unclassified candidate (Chittagonian).
This archive-list audit is complementary to the Glottolog bibliography audit below; neither a
secondary citation nor an archive title may supply a lexical reading.

The broader official SIL Bangladesh sociolinguistic-program chart says that 29 communities were
surveyed, including unpublished or unlisted work beyond the eleven-title publication page.  The
community-by-community reconciliation in `sil_bangladesh_survey_program_audit.csv` records 18
installed communities, five communities covered by the three partial manual-recovery reports, the
high-confidence Chak lexical gap, and five unclassified communities tied to Chittagonian,
Chakma-Tanchangya, and Marma-Rakhine report candidates.  Community coverage is not source
completion: every unclassified report still requires publisher-file acquisition and lexical-scope
inspection.

### Pakistan

The series preface identifies SSNP as five volumes, not six.  The exact volume-to-install mapping
is pinned in `sil_pakistan_ssnp_series_manifest.json`; all five numbered volumes are represented
below.

| Source | Lexical scope | Jambu state |
|---|---|---|
| SSNP vol. 1 *Languages of Kohistan* | Swat/Dir and Indus Kohistan lists, including Ushojo | **installed** in `20260725-ssnp.csv` |
| SSNP vol. 2 *Languages of Northern Areas* | Balti, Burushaski, Domaaki, Wakhi, Shina and related Northern Areas lists | **installed** in `20230416-northern.csv`: all 11,343 Lexibank v1.1 rows from 51 varieties; 507 Urdu/Pashto controls are retained in the installed source file but excluded from compiled lexical output |
| SSNP vol. 3 *Hindko and Gujari* | Hindko and Gujari lists | **installed** in `20260725-ssnp.csv` |
| SSNP vol. 4 *Pashto, Waneci, Ormuri* | 200 printed prompts × 34 Pashto locations, Waneci, and Ormuri | **installed** (7,131 responses; 68 printed gaps and one blank cell audit-only; positioned SILDoulosNP decoded against the report's phonetic chart, no OCR) |
| SSNP vol. 5 *Languages of Chitral* | Chitral lists | **installed** in `20260725-ssnp.csv` |
| ESR 2010-012 Pahari and Pothwari | 14 target survey locations, 217-item lists | **installed** (3,038 target responses; 434 Abbottabad/Mansehra Hindko control cells, including 18 blanks, audit-only; positioned Doulos SIL text, no OCR) |
| SIL EWP 2005-008 Kundal Shahi, Azad Kashmir | descriptive paper with a published lexical list | **installed** (`20220913-kundalshahi.csv`: 163 source rows; 161 compiled citation attestations after normal cross-source merging; source key `kund`) |

Discovery is still open for non-ESR SIL manuscripts and data appendices not indexed as survey
reports. New candidates must be added here before extraction so “all” remains auditable rather
than depending on memory or filename searches.

### Unclassified current-series objects

| Object | Discovery evidence | State |
|---|---|---|
| JLSR 2025-006 | *A Sociolinguistic Survey of Mandar, Pannei-Ulumanda, Pannei-Polewali, Koneq-koneq, and Dakka*, by Renhard Saupia, Stan Anonby, Tiar Simanjuntak, and Geraldy Ruwayari | **identified outside country scope** (the official PDF index exposes the report title and an abstract explicitly locating all five subject languages in western Sulawesi, Indonesia; direct PDF retrieval remains HTTP 403, so lexical scope was not inferred or inspected; the disposition evidence is pinned in `sil_jlsr_2025_006_discovery.json`) |

### 2026-08-29 Glottolog/SIL archival gap audit

The local Glottolog bibliography snapshot and SIL archive/search index were checked for India,
Pakistan, and Bangladesh lexical surveys or published wordlist appendices not represented by a
country-table row above. ESR 2013-016 Rabha was the highest-value clearly SIL, accessible,
non-duplicative census gap: its report and forms already existed elsewhere in Jambu, but the source
was mentioned only in introductory prose and had no India census row or pinned archival topology.
The source-local manual package is `sil_rabha_2013/`.

The broad explicit-country bibliography filter and its candidate-by-candidate disposition are pinned
in `sil_glottolog_candidate_audit.csv`: 32 records, of which 30 are genuine SIL sources or component
records and two are documented string-match false positives.  The exact 46,017,843-byte bibliography
snapshot, retrieval date, SHA-256, filter, and disposition counts are pinned in
`sil_glottolog_candidate_audit_manifest.json`.  Every genuine record is tied to a row above as
installed, a covered series component, a duplicate reissue, an inspected report with no published
lexical rows, or one of the three Bangladesh manual-recovery queues.

A second audit pass found that three Bangladesh-series rows had been overstated as complete.  Their
legacy-font importer's verified glyph table recovers most forms, but its per-record audits still
exclude 1,514 attested cells solely because one or more glyphs lack a verified decoding: Koch 563,
Kurux 239, and Garo 712.  These are now explicit manual-transcription queues, not accepted losses.
The render-first dispatch record is `sil_bangladesh_legacy_manual_queue.md`.
The historical audit scope is also stated explicitly as 239 legacy-excluded Kurux attestations,
563 legacy-excluded Koch attestations, and 712 legacy-excluded Garo attestations; later manual
completion does not erase those original omissions from the reconciliation trail.
The India Koch appendix ESR 2011-033 is already complete and is bibliographically distinct from the
Bangladesh Koch report ESR 2011-023; the country tables now represent them separately.

The independent bibliography pass also surfaced ESR 2008-006 on regional Indian Sign Language.
It is now represented in the census instead of silently disappearing from a spoken-language-only
search.  The exact 121-page report was recovered from the publisher URL's 2017 Wayback capture and
inspected.  It publishes 245 English prompts and item-level numeric similarity judgments, but not
the underlying sign recordings or any lexical sign representation.  The source is therefore closed
as inspected with no lexical rows; no score or gloss has been misrepresented as a form.

The same pass surfaced SIL Electronic Working Paper 2005-008 on Kundal Shahi in Azad Kashmir.  Its
163 lexical rows were already installed under source key `kund`, but the Pakistan census did not
name the source.  The row above now reconciles that existing ingest instead of treating it as a new
gap; its retrospective source checklist and 164-line per-record audit (header plus 163 records)
remain the authoritative installation evidence.

Adjacent non-SIL candidates were not promoted into this SIL lane. Ahirwal's 2003 *Dadra and Nagar
Haveli* comparative wordlist is openly archived by Census of India, but no SIL author, commission,
series, or archive relationship was found. Schmidt and Kaul's 2008 Shina--Kashmiri vocabularies are
accessible through *Acta Orientalia*, not SIL. Bhattacharya's 1976 *The Tribal Languages of South
Kerala* remains a high-value Glottolog wordlist candidate, but no verifiable accessible full scan was
found. These are discovery leads, not SIL provenance claims.

## Why these

Per the coverage audit of 2026-08-26, 23 Dravidian lects of the Nilgiri/Wayanad/Kerala tribal belt
are absent from `cldf/languages.csv` outright (Paniya, Kattunayakan, Mullu and Jenu Kurumba,
Yerava/Ravula, Muduga, Kadar, Muthuvan, Mannan, Paliyan, Malapandaram, Ullatan, Kurichiya,
Aranadan, Eravallan, Kanikkaran, Mala Malasar, Malayarayan, and others), and eight more are
effectively empty — Betta Kurumba 1 form, Belari 1, Palu Kurumba 4, Sholaga 5, Alu Kurumba 10,
Irula 169, Koraga 319, Kudiya 397 — against Toda 11410, Badaga 16886 and Kota 5051 on the same
plateau.  Central Dravidian (Naiki 512, Naikri 689, Kolami 1455, Parji 1426, Gadaba 1554) is
DEDR-derived only, with no locality resolution.

## Ingestion shape

These are survey wordlists, so `SOURCE_INGESTION_CHECKLIST.md`'s survey addendum governs:
target lects are separated from control languages and the controls excluded deliberately; each
survey site becomes a registered, language-qualified dialect beneath a canonical base language
rather than a pseudo-language; prompt/concept IDs are preserved so identical short forms under
different prompts are not collapsed; and source transcription and locality coordinates are kept
with explicit quality caveats.  This is the same model already used for the SSNP appendices in
`ssnp.py` and the SIL Nepal reports.


## Ingestion status (2026-08-29)

| Report | State | Installed |
|---|---|---|
| silesr2015-029 Idukki | **done** | 4,630 rows, 5 languages, 19 sites |
| silesr2015-028 Palakkad | **done** | 4,710 rows, 8 languages, 21 sites |
| silesr2011-033 Koch | **done** | 2,281 rows, 3 languages, 11 sites |
| silesr2011-038 Tripura | **done** | 8,997 rows, 3 languages, 24 sites |
| silesr2011-040 Kurux | **manual and shared source integration complete; consolidated build pending** | 1,365 target rows installed from the exhaustive 1,869-row audit (1,661 attested conceptual cells, 136 blanks, 72 not-used; 296 Bangla control forms audit-only), 2 languages, 6 sites |
| silesr2011-023 Koch (BD) | **manual review and shared source integration complete; consolidated build pending** | 1,017 unique target rows installed from the exhaustive 2,159-row expanded audit; 772 controls, 226 ambiguous rows, 25 blanks, and 119 not-used rows audit-only; 3 languages, 7 sites |
| silesr2012-007 Garo (BD) | **partial; manual review through item 155; full legacy-install reconciliation remains active** | 4,444 installed rows; 2,635 independently reviewed cells (2,556 ordinary attestations, one attested source conflict, 61 blank-only, 17 not-used), 5 languages, 17 sites |
| silesr2018-010 Nilgiri Irula | **done** | 2,054 rows, 1 language, 11 sites |
| silesr2019-005 Mudhili Gadaba | **done** | 1,538 rows, 1 language, 7 sites |
| silesr2008-013 Jaunsari | **done** | 1,619 rows, 1 language, 7 sites |
| silesr2008-003 Bishnupriya | **done** | 1,801 rows, 1 language, 6 sites |
| silesr2008-002 Meitei | **done** | 2,406 rows, 1 language, 8 sites |
| silesr2018-011 Bareli/Pauri | **done** | 6,320 rows, 8 languages, 30 sites |
| silesr2012-002 Nimadi | **done** | 2,826 rows, 1 language, 13 sites |
| silesr2019-006 Lahul Valley | **done** | 5,027 rows, 6 languages, 22 lect/site lists |

Importers: `sil_survey_wordlists.py` (Appendix B3 layout: Idukki, Palakkad), `sil_koch_2011.py`
(coordinate-parsed table), `sil_bracket_wordlists.py` (bracketed site codes: Tripura, Kurux,
Koch BD, Garo BD).  Sound profiles: `conversion/sil-survey.txt`, `sil-koch.txt`, `sil-tripura.txt`,
`sil-kurux.txt`, `sil-bangladesh.txt`, `sil-nimadi.txt`.

All four bracketed-code reports parse with **no unparsed lines and full item coverage**.

### 2018-011 Bareli/Pauri Unicode recovery

The exact official publisher PDF was recovered from Internet Archive capture `20240627030145`
of SIL archive file `silesr2018_011.pdf` (SHA-256
`02128358a61e175ba2a07b2862f6072167a3609cf71264e235ae21284fe2ceea`).  Contrary to the
OCR-risk label that had kept this report out of Jambu, Appendix C.3 has a complete Unicode
Charis/Doulos SIL text layer: no OCR and no legacy-font guessing are required.  Its three
landscape columns must be parsed in page reading order, because long responses sometimes continue
into the next physical column or page.

The frozen snapshot accounts for all 210 prompts and all 33 printed lists: 7,214 lexical response
records plus 33 `DISQUALIFIED` cells for item 70 *millet*.  Jambu installs 6,320 responses from the
30 regional lists under eight Glottolog-mapped base languages and distinct source-locality dialect
records.  The audit retains but excludes 789 Hindi/Gujarati/Marathi responses, 105 regional `NO
ENTRY` cells, and the 33 disqualified cells.  The source's similarity-category numbers remain
Notes rather than etymological claims; English size/location annotations are separated from forms;
and the two literal unmatched open brackets printed for item 155 Amalwadi `pats̪[` and item 157
Mandvi `hat[` are preserved with explicit uncertainty rather than silently emended.  Table 8 gives
administrative localities but no point coordinates, so site coordinates remain blank by design.

### 2009-011 Malvi CID-font recovery

The exact 280-page SIL publisher PDF was recovered from Internet Archive capture
`20150930085157` of `silesr2009_011.pdf` (SHA-256
`e67e314974ab10eb8244b08dba56d08d1ce8cbf16eaef1be022071d49032a2dd`). Appendix B is not
image-only and required no OCR, but its principal `SAG-IPASILManuscript` Type0 font has no
ToUnicode map: ordinary text extractors therefore expose its phonetic characters only as CIDs.
The report's own printed IPA chart resolves 31 of the 34 CIDs actually used in the wordlists. The
remaining three — `ɠ`, literal circumflex, and combining square below — were resolved from SIL's
official `SAGIPA2Uni.map` table and checked on rendered pages. A geometric parser preserves the
separately embedded raised aspiration and vowel symbols as Unicode modifier letters and attaches
below-line dental and square diacritics to their printed bases.

The frozen snapshot contains 8,798 printed rows from 207 prompts plus 114 explicit audit cells for
items 11 *breast*, 23 *urine*, and 24 *feces*, which the report says were disqualified and removed.
Jambu installs 6,894 responses from thirty target lists across the source's Ujjaini, Rajwadi,
Umadwadi, Sondhwadi, Gond-Malvi and Bhil-Malvi groups. The audit retains eight comparison/control
lists (two Bhili, two Nimadi, Bhopali, Hindi, Gujarati and Marathi), ten printed control `No entry`
cells, and 37 target cells printed `By Name` instead of a lexical response. Source category numbers
and letters remain Notes rather than cognacy claims. As an independent recovery check, the
Thillorkhurd list was compared with its later Unicode Malvi comparison list in ESR 2012-002: 132
response forms agree exactly within 126 concepts and cover nearly the entire used symbol inventory.
The source supplies locality and administrative metadata but no point coordinates, so all thirty
dialect rows intentionally leave coordinates blank. They attach to Jambu's existing canonical
Malvi base-language ID `mewari_basad` (`malv1243`) rather than creating a duplicate language.

### 2012-002 Nimadi Unicode recovery

The exact legacy SIL publisher PDF was recovered from Internet Archive capture `20170810011221`
of `silesr2012_002.pdf` (SHA-256
`1a7e8daaeb2b967e2f9490292689e33a188caf47dc262c942a47136bb270d0d8`). Appendix A uses a
Unicode Doulos SIL text layer rather than OCR or a lossy legacy-font encoding. Its portrait table
has parity-shifted three-column geometry; physical pages 95--99 instead use two wider columns for
paired predicate prompts. The frozen extraction accounts for all 210 standard prompt slots and all
18 lists: 4,019 printed response records, 72 explicit audit cells for the four prompts absent from
the published appendix (11 *breast*, 23 *urine*, 24 *feces*, 70 *millet*), and one unprinted
N-Son-Bal response cell. Jambu installs 2,826 nonempty responses from the thirteen Nimadi lists.
The five Parya Bhilali, Malvi, Hindi, Gujarati and Marathi comparison lists remain fully auditable
but excluded, as do five target `no entry` cells and two target cells without primary forms.

Three text-layer edge cases were checked against 250--300 dpi page renders: item 13 N-Son-Bal
fuses the category digit into the source glyph run and is preserved diplomatically as `ct̪`; item
40 N-Rup-Br has a visibly blank primary followed by category-2 `mund̪i`; and Gujarati item 98
contains a spurious extracted `(cid:1)` absent from the page image. The report explicitly treats
`ə`, `ɐ`, `ʌ`, and `ɑ` as interchangeable for its analysis, but Jambu preserves the printed source
IPA in Original/Phonemic and applies that policy only in the dedicated display profile. Source
similarity categories remain Notes, not historical cognacy. Appendix A gives locality,
administrative, speaker-community, and WordSurv-code metadata but no coordinates, so the thirteen
dialect records leave coordinates blank rather than substituting locality centroids.

### 2008-013 legacy-font recovery

Jaunsari's appendix is not OCR: its 36 wordlist pages contain a text layer whose embedded
`SAG-IPASILDoulos` font exposes the original bytes as U+F000--U+F0FF private-use characters.
SIL Converters 5.4.1 supplies the authoritative `SAGIPA2Uni.map` converter. The 32 bytes used by
the appendix account for 5,059 legacy-symbol occurrences; all map deterministically to Unicode,
with no unmapped symbol or unparsed line. The report prints 207 of the standard 210 prompts and
explicitly says items 11 *breast*, 23 *urine*, and 24 *feces* were disqualified and removed.
The complete source topology is 2,729 responses: 1,619 installed responses from seven Jaunsari
lists and 1,110 audit-only Hindi, Bangani, Jaunpuri, Nagpuriya, and Sirmauri controls.

### The two image-only reports

Neither prints a text layer for its wordlists, but they are **not** the same problem.  The decisive
measurement is the resolution of the embedded page images, not the OCR engine:

| report | native scan | wordlist pages | verdict |
|---|---|---|---|
| silesr2018-010 Nilgiri Irula | **67–74 dpi** | 29–52 (24 pp.) | recovered with typed uncertainty and full manual review |
| silesr2019-005 Mudhili Gadaba | **168 dpi** | 18–35 (18 pp.) | recovered at quarter-column zoom |

At 70 dpi a lowercase letter is 7–8 px tall and a subscript diacritic is **1–2 px**. This made the
Irula source a poor candidate for unattended OCR, but not an excuse to omit it. Enlarged crops,
structural OCR, repeated site forms, source phonotactics and explicit typed uncertainty made a
complete diplomatic recovery possible without silently pretending that every raster distinction
was certain. Every one of the 3,388 source response records and 29 layout fragments is represented
in the audit; 2,054 target forms install, 15 target gaps and 1,319 controls remain audit-only.
The failed automated approaches remain useful negative evidence:

| approach | result |
|---|---|
| `tesseract --psm 6`, whole page | merges the three columns into single lines |
| per column crop | structure correct, transcription wrong: `telu` for `ṭolu`, `elumbi` for `ɛlumbɨ`, `fattam` for `rʌṭṭʌm` |
| `--psm 4` | identical output |
| 600 dpi render | worse: `uwguréd` for `uguɾɨ`, `MAW Ll,2` for `MAV 1,2` |
| monospace cell segmentation plus clustering | 5,102 cells over three pages give 3,791 fuzzy clusters for an inventory of roughly 45, the largest sixty covering 15% |
| reading magnified crops directly | the underlying scan is 70 dpi; magnification interpolates and does not add information |

Its site key is nevertheless resolved, so a rescan could be ingested without rediscovery.  Table 1
(p. 15) gives eleven Irula sites in three dialect groups — Mele Nadu: KUN Kunjapanai, KOL Kolikarai,
CHE Chemmanarai, KIL Kilkupkad, MET Mettukal; Northern: CHO Chokkanalli, MAV Mavanalla,
ANA Anaikatty, BOO Bookapuram; Vette Kada: THA Thaliyur, NEL Nellithurai — plus CBT Coimbatore
Tamil elicited for this survey, and KAN Kannada, BAD Badaga, ALU Alu Kurumba, BET Betta Kurumba,
JEN Jenu Kurumba culled from Blair 2012.  The code MAD is never expanded in the report; its forms
are Tamil (`wuḍʌl`, `t̪ʌla`, `mugʌm`, `mu:kku`) but the expansion is unresolved, so it must not be
installed under a guessed language.

### 2019-005 transcription conventions (verified)

The Gadaba scan is a clean 168 dpi proportional serif and its wordlists are legible.  The report
distinguishes three coronal series, and the distinction was pinned down against Telugu, whose
wordlist it prints and whose phonology is known:

| printed | reading | control |
|---|---|---|
| detached box below the letter | dental `t̪ d̪ n̪` | TELUGU 'head' `t̪ələ` = తల *tala*, dental |
| tail attached to the letter, descending | retroflex `ʈ ɖ ɳ` | TELUGU 'village' `pəlːɛʈuɾu` = పల్లెటూరు *palleṭūru*; 'hair' `dʒʊʈːu` = జుట్టు *juṭṭu*; 'heart' `gunɖe` = గుండె *guṇḍe* |
| no mark | plain `t d n` | Chinachipuru 'eye' `kʌnnu` beside `kʌɳuku` elsewhere |

Length is `ː`, nasalisation a tilde.  Tesseract reads this scan's *structure* reliably (site names,
similarity-group numbers, item headers, continuation lines) while dropping every diacritic, so it is
used as a navigational scaffold exactly as in `buddruss_grangali_1979.py`, with the forms collated
against the page renders and an automatic ASCII-fold cross-check against the OCR skeleton.

Site key (Appendix A.2, p. 16): seven Mudhili Gadaba wordlists — Kothavalasa and Panukuvalasa and
Reyavanivalasa (Salur mandal), Suregadivalasa and Bobbilivalasa and Gogaduvalasa and
Chinachipuruvalasa (Pachipenta mandal), all Vizianagaram district, Andhra Pradesh — plus one Telugu
list from Srikakulam.  210 items; the appendix prints "DISQUALIFIED" for items dropped in analysis
and "0 No Entry" for gaps.
