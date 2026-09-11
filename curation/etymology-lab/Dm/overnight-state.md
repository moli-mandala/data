# Dameli overnight continuation

User explicitly asked to research all possible remaining etymologies overnight and will review them triaged by difficulty at **09:00 America/New_York, September 10, 2026**. This date is confirmed, not September 11. New non-numeral analyses remain pending for morning review; no repeated batch approval questions overnight. Numerals retain standing approval. Hold win and all gaṭ want/win variants. No DB rebuild, commit or push. Do not force weak analyses to meet a count.

## Schedule and scope

Heartbeat `dameli-overnight-etymology-research` attached to current thread `01a087ae-be6a-7bb2-a14c-2fb364393d29`, scheduled hourly at minutes 0 and 30. Prepare report before deadline; deliver at 09:00 and pause heartbeat. Only one heartbeat per thread is allowed. No separate handoff heartbeat was created.

The existing task `Etymologise 50 Gilgit Shina lemmata` (`01a08990-8f34-7232-a02e-3a49f6bb9995`) has been sent the user's queue **Gilgit → Dras Shina → Brokskat**, and confirmed deadline. It has its own heartbeat. Do not duplicate its work or modify its manifests. No sub-agents have been requested. Shared overlay can change concurrently; re-read immediately before any approved save.

## Progress

- Batches 001–016 saved. Batch 016 was saved in this turn: 117 assignment rows /109 records; prior parsed rows preserved; schema, intended temporary edges and second-application-zero checks passed.
- Pending batch 017: ten researched proposals /24 assignment rows /20 records, not approved. Four easy, five moderate, one hard proposal. Includes soc from an explicitly identified Urdu donor; kʰiṣ cultivate; ōtʰ remain; mraŋg deer/markhor; parbap; own/aunt/uncle + child compounds; tentative nanūṛa; paśūrdāri.
- `overnight-review.md` is the user-facing report to keep updated and expand. `overnight-inventory.json` snapshots remaining records, with pending IDs marked. Refresh inventory against compiled rank-1 graph and accepted overlay, and exclude pending proposals from new research selections.
- Snapshot before batch17: 775 unassigned Dameli records. Source homonyms and suffix variants must be inspected, not merged indiscriminately.

## Sources and helpers

Apply `.agents/skills/jambu-etymology/SKILL.md`; research-only does not trigger source ingestion. Existing-node overlays require focused validation, no full build.

- `/tmp/cdial-text.json`: complete cached primary CDIAL entries as `{id,page,text}`. Inspect full prose and addenda; avoid relying only on compiled comparisons.
- `data/data/cdial/cdial.pickle`: original primary snapshot.
- `tmp/pdfs/perder-dameli/all.txt`: full grammar, printed page numbers in text. Main primary URL https://www.diva-portal.org/smash/get/diva2%3A651418/FULLTEXT02.pdf
- `/tmp/prepare-dameli17.py` reproduces batch17 but **overwrites it**. Do not rerun after editing/expanding the manifest without updating the script. `/tmp/dameli17-deps.json` contains accepted parent dependencies.
- `/tmp/prepare-dameli11.py` prefix before newline `specs=` loads compiled forms and overlay. Its `blocked` uses all graph edges, so prefer explicit rank-1 inventory for rigorous new work. Avoid huge per-row set unions.
- `/tmp/save-dameli16.py`: approved-only atomic save pattern; never use on pending overnight analyses without approval.
- `/tmp/update-dm-overnight.py`: initial temp-graph checks and initial report; likewise overwrites report, so adapt before reuse.

## Research directions and cautions

1. Unresolved loan paths are the largest remaining group. Perder p126 explicitly identifies Urdu soč, now pending. Many Arabic/Persian-looking forms lack evidence for immediate donor; do not link straight to ultimate Arabic/Persian just because that node is available.
2. `abēni` is explicitly from Pashto bən in CDIAL6534; Pashto donor IDs f_g35eympyrvacm, f_zfpb7faxkio46, f_ztqpfb6r5ugik exist but unlinked. Validator requires supported ancestry; do not invent one. This is an established proposed borrowing that cannot yet be represented by the current accepted-target constraint; explain in unresolved tier.
3. Perder p88 explicitly says sava hundred and zara thousand are from Pashto. Existing donor sal hundred is also unlinked. Do not bypass immediate donor to Sanskrit. Old a'zâr four hundred is questioned by Perder; retain uncertainty.
4. `f_6v23xxakc35jq ṣāvṓ` is not a Dameli attestation: Perder p129 quotes Palula ṣaawóo. It is excluded and flagged for source-language correction; do not alter the ingested source or compiled DB during this task.
5. Remaining roots žin eat, brei girl, kōk sleep, lāk weep, sāt keep, nat enter are difficult. Some have variant edges to unlinked roots; that does not constitute an accepted etymology. Do not use variants as evidence.
6. New primary research lead: Jakob Halfmann, The Diversification of Indo-Iranian and the Position of the Nuristani Languages, https://medialibrary.reichert-verlag.de/de/file/9783752003543_ebook.pdf . Web-readable 161pp. p48 discusses secondary aspiration in Dameli before sibilants, including kʰiṣ; broader paper argues possible Nuristani affiliation, so do not equate every Nuristani connection with borrowing. p22 mentions Pashai žū eat, Kati yu and Wakhi yaw from PIE *Hi̯eu̯h₂ graze, no OIA verbal descendants. Not enough by itself to assign Dameli žin. No new nodes or source ingestion performed.
7. Morphological compounds remain promising, but only link attested components and do not interpret a whole phrase as an inherited simplex. Broad glosses can hide errors: parbap foremother excluded, mamāni mother's older brother excluded, mraŋ nail excluded from animal word.

For the morning handoff report all researched proposals by easy/moderate/hard, exact record/edge counts, and unresolved cases. Put substantive glosses and actual comparanda into every review row. No need to force batches of40 now; user's overnight instruction supersedes review pauses and wants all defensible proposals.

## Active continuation checkpoint

Batch 018 now contains 8 researched Pashto loan/kinship proposals, 14 records and 22 blocked rows. Top-level assignments intentionally empty. Existing Pashto zər donor found: f_lg6ldgrwyzv42 (with two parallel survey lects), unlinked. `overnight-unresolved.json` records nine examined groups, including Persian navāsa/navāsi loan evidence, father suppletion, sawa donor allomorph, kuč vs kōč distinction, unresolved walk/akaṭ bases and Perder’s bawi SW gloss inconsistency. `overnight-review.md` rebuilt with four-column difficulty tiers. No accepted overlay changes.

## 01:00 heartbeat checkpoint

Batch 019 adds 10 directly source-identified Urdu/Pashto loan groups across 21 records, all research-blocked by missing/unlinked donor nodes. Perder Tables 13 (p42) and 16 (p57) provide explicit immediate languages. Pashto māśūm donor f_ekfmubidrezui exists but unlinked. Also added proś/prośt/prośta bed (3 records) as a possible Nuristani loan/family comparison, donor direction unresolved; existing PNur *prost f_qmzmchb3zhjtk and Nuristani comparanda recorded. No accepted rows saved. Refresh report using /tmp/dm-refresh-review.py; do not rerun older overwrite scripts. Current future leads: Perder p125 lag probable Pashto light verb; p126 teer probable Pashto completion marker and rawan < Psht rawandal; p173 agar/agarka contact analysis; p187 laka filler and echo constructions. These have not yet been added to examined counts.

## Goal continuation checkpoint

Batch 020: four qualified contact proposals (lag, tēr, ravan, agar), nine records, donor nodes unresolved. Added unresolved agarka, laka, ultimate-English miskol/ṭelefun and tentative giāg compound; Perder primary locators retained. Batch 021: one difficult historical water+fish compound proposal, āmras (two records), 407 áp + 9758 mátsya section 1. Turner directly cites older Dm âu-mraċ and leaves r unexplained. Four proposed component rows validated; temp graph six changes, second application zero. No accepted save. Added inḍorī mortar regional comparison with Khowar aṇḍor and Gawarbati hīndūrīk, direction unresolved. Cross-language corpus lookups for pestle/sleep/girl/morning/mosquito/sneeze/comb gave no adequate new ancestry evidence; do not count those exploratory lookups as completed analyses. Remaining Decker numerals satʰaṣṭʰ seven / nõ eight / ī hundred look anomalous and need original-source verification before any automatic numeral save. No source edits performed.

## Numeral source audit

Four malformed Decker records identified and excluded: f_nuxv5esaaghrw seven+eight merged, f_r7zd42ixqhgr6 nine shifted to eight, f_dot2su2ggptj6 twenty+hundred merged, f_54cbhvurwlm3q orphan ī hundred. None has accepted overlay rows. Cached and web PDF text plus join_fragments importer logic support the diagnosis; visual PDF verification is still pending (web screenshot cache miss). See overnight-source-issues.json. No importer/source/identity repair performed.

## Kinship continuation

Batch 022 adds two proposals/six records: ẓami brother-in-law under CDIAL5200 jāmí (moderate), and tentative l-extended žamili/źāmili in-law family (hard). Explicit Savi and Kalasha/Gawri comparisons support meanings; retain sibilant and kinship-code differences. Bshk is Gawri/Bashkarik, NOT Bashgali. Female wife/woman žami excluded. Six rows validate, temp graph12changes, secondapply0. No accepted save. Numeral PDF download timed out; no live download remains.

## Verb and source discovery checkpoint

Five additional unresolved groups recorded: pʰam swelling-family lead, ṭup cover (not drip-family ṭupp), drāṣ comb (not grape drākṣā), truida ki (postposition prevents whole-phrase assignment), daś hand;ten (homonym conflation). Added research-source-leads.md documenting attempted primary-text retrieval; Kogan2007 article is only located bibliographically, not read. Older CDIAL Dameli spelling/gloss sweep yielded no additional unqualified reflexes. No accepted rows changed.

## Compound and adjective checkpoint

Batch 023 adds two transparent kinship phrases/two records/four component rows; validation passed with six temporary changes and second application zero. Batch 024 records three source-explicit adjective derivations (gaḍivēla, havavēla, xaṭāpin), blocked by missing/unlinked bases. Six additional kinship records and five coordinated phrases are documented as unresolved; the middle qualifier has explicit Ashkun/Kati/Pashai comparisons, without a asserted loan direction. Current totals: 40 pending proposals/77 records; 38 graph-valid proposed rows, 50 examined unresolved records, four source-repair records, 644 unexamined or unresolved. No accepted save or DB rebuild.

## Fever-family continuation

Batch025 adds pražar become ill as a tentative denominative of attested Dameli praźar fever (CDIAL8519). Ashkun, Waigali, Pashai and Shumashti comparisons retained; no loan direction asserted. One derived row validates with two temp changes, secondapply0. Thirsty avḍā and survey gā speak documented unresolved with explicit primary-source caveats. Totals41proposals/78records,39graph-valid rows,52examined unresolved,4source issues,641unexamined or unresolved. Noacceptedchanges.

## Further verb comparisons

Seven records in five groups examined: śām castrate (CDIAL12306 ritual-slaughter family, Waigali šāmä-istä; do not conflate with allay or attach a verb to participle), bait fold/buṭ braid (older Dm baṭyāy under11356, modern phonology unresolved), pɻeyen/pɻei squeeze (8226 comparison does not explain cluster), raṭ bury (6618 initial mismatch), prambal light (8518 unexplained mb). These are documented leads, not proposals or accepted links. Totals59examined unresolved,634unexamined or unresolved;41proposals/78records unchanged.

## Possessive paradigm checkpoint

Batch026: two proposals/eight records/eight rows under CDIAL13127 section1 santaka belonging to. Six singular sā̃/sā̃ĩ and inflection records moderate; two plural suna/suni hard because u-vowel history is not explicit. Turner directly cites older Dm-sā̃ and Kati-st(e); Perder p80paradigm confirms grouping. No Nuristani donor direction asserted. Validation16changes,second0. Counts43proposals/86records,47graph-validrows,59unresolved,626unexamined-or-unresolved. No accepted save.

## Coordinator loans checkpoint

Batch027 adds source-explicit xu from Pashto, lekin from Urdu and qualified ya from Urdu:3proposals/9records,6blockedrows,top assignments empty. Existing Pshtxō and Hindustani lēkin donors unlinked; yaword donor unresolved. Four le/lē many forms documented with Turner’s explicit caution that Dm may have original l-, not securely bahura. Totals46proposals/95records,47graph-valid rows,63unresolved,613unexamined-or-unresolved. Noacceptedchanges.

## Postposition phrase checkpoint

Batch028 adds tānu milāi with oneself and mā̃ māma ṣavāi through my uncle as two transparent phrase analyses/five components. All bases have reviewed ancestry; source uncertainty about deeper ṣavāi is retained, not strengthened. Validation7changes,second0. Taprei speculative ta+prei and misglossed mā̃-Ø his/her/its recorded unresolved. Totals48proposals/97records,52graph-validrows,65unresolved,609unexamined-or-unresolved. Noacceptedchanges.

## Source comparison audit

Verified readable author-uploaded Halfmann2022 PDF (direct URL in research-source-leads.md). Existing accepted heart note already had full qualified interpretation; preserved. Added existing-link-audit for compiled rɔk deodar→10826 roka, explicitly challenged by Halfmann p128n18. Not counted as newly researched unassigned record. No lexical changes. Causative table31 reviewed: most derived forms already have source edges and should not be duplicated as new etymology work. prambal-āi-i remains an explicit derivative of an unresolved base; potential next blocked proposal.

## Attestation and derivation checkpoint

Batch029: prambal-āi-i causative CP explicitly supported by Table31, one derived row blocked by unlinked prambal base. Additional root attestation, dāś stone in quoted rock name (genuine source gloss, not hand/ten error), and kʰuśāli joy vs short-vowel adjective intelligent examined separately. Totals49proposals/98records,52graph-validrows,68unresolved,605unexamined-or-unresolved. No accepted changes.

## Topic particle checkpoint

Batch030: six ta topic/subsequence records proposed under5612 tá as explicitly provisional family index, exact formation unresolved; hardreview. Comparative source Liljegren/Svärd2017 section4 DOI inmanifest; originalsourcekeynotinvented. Five ba records documented with possibleWaigali influence, noancestry. Validation12changes,second0. Counts50proposals/104records,58validrows,73unresolved,594unexamined-or-unresolved. Currentpending98prebatchIDs distinct and noacceptedoverlaps checked.

## Child and crop vocabulary checkpoint

Six more records examined:4zātak child forms with Iranian-versus-inherited alternatives,2Decker crop responses with documented regional Iranian mediation but noidentifiedDm donor. Totals50proposals/104records,58validrows,79examinedunresolved,588unexamined-or-unresolved. Noacceptedchanges.

## Eating family checkpoint

Batch031 adds7ži/žin/žen/ž records under10507 section2 eat as hard comparative-family proposal. Newly inspected CDIAL explicitly includes older Dmžu; Perder’s žuw feed supports connection but vowel and deeper yoke vs graze/eat roots remain unresolved. No separate10507-2 node exists; citation and notes distinguish eating sense. Validation14changes,second0. Totals51proposals/111records,65validrows,79unresolved,581unexamined-or-unresolved. Noacceptedchanges.

## Additional explicit Pashto loans

Batch032 adds zyat/ziat much (two records) and rimel kerchief (one), explicitly attributed to Pashto in Perder Table10 p41. Exact standalone donor nodes unresolved; combined Psht response not repurposed as a lexical donor. English-origin lemp lamp documented separately as unresolved immediate transmission. Totals53proposals/114records,65graph-validrows,80examinedunresolved,577unexamined-or-unresolved. Noacceptedchanges.

## Keep-verb comparison checkpoint

Batch033 adds two sāt keep records as a probable Pashto borrowing, explicitly an inference, with OPED25572 sātəl/present sāti independently verified. No donor node exists; pending moderate review, no assignments. Added ziat-a ordinary inflection to batch032 (now4records). Totals54proposals/117records,65graph-validrows,80examinedunresolved,574unexamined-or-unresolved. Noacceptedchanges.

## Suppletive first checkpoint

Batch034 records two aval first attestations, explicitly Arabic-derived through Urdu or Pashto according to Perder p92. Immediate language and donor unresolved, so no automatic numeral save. Totals55proposals/119records,65graph-validrows,80examinedunresolved,572unexamined-or-unresolved. Noacceptedchanges.

## Tentative subordinator compounds

Batch035 nitē lest (3records,6componentrows) combines ni not+tē that, explicitly probably in Perderp170; both bases reviewed. Validation9changes,second0. Batch036 kuitē because (2records,4blockedrows) combines ku why+tē that, ku unlinked; validator rejection retained. Two aaċ take examples documented as semantic/source ambiguity vs come, not automatically grouped. Totals57proposals/124records,71graph-validrows,82examinedunresolved,565unexamined-or-unresolved. Noacceptedchanges.

## After-adverb contact comparison

Batch037 bāt/bat after fourrecords compared with OPED9343 Pashto baʿd/bād after, Arabicorigin. Perderfinaldevoicing fits; immediatedonor unresolved, no assignments. Front-vowel bǣt after separately documented unresolved. Totals58proposals/128records,71graph-validrows,83examinedunresolved,560unexamined-or-unresolved. Noacceptedchanges.

## Directional phrase checkpoint

Batch038 adds two transparent pronominal phrases: mūbãĩ towardsme (2records) and yē-bãĩ inthisdirection (1record),6blockedcomponentrows. Existing bãĩ unlinked; validator rejection retained. barān outside documented with CDIAL9184/9226 regional comparisons and unresolved final-ān. Totals60proposals/131records,71graph-validrows,84examinedunresolved,556unexamined-or-unresolved. Noacceptedchanges.

## Echo formations checkpoint

Batch039 adds cay-may teaandrelatedthings and boṇḍri~moṇḍri borderandallthat as two explicit echo derivatives. Tea base existsbutunlinked; standaloneborderbase absent. Oneblockedrow, noassignments. Totals62proposals/133records,71graph-validrows,84examinedunresolved,554unexamined-or-unresolved. Noacceptedchanges.

## Unresolved truth and crying families

Two śãũ true/pure colourintensifier records examined against satya and śuddha; soundhistory unresolved. Eight lāk weep inflections grouped, superficial Marathi laḍṇẽ under10590 insufficient; nohistoricalparent selected. Totals62proposals/133records,71graph-validrows,94examinedunresolved,544unexamined-or-unresolved. NurAC1942 bibliography points only toWorldCat; no newMorgenstiernefulltext obtained. Noacceptedchanges.

## Regional body sky fruit comparisons

Batch040 adds3probable regional loan-family proposals: uźut body hard (OPED36537specificbodysense,initial/consonantissues), āsmān sky moderate (Persianfamily,routeunknown), mevā fruit moderate (OPED38825,Khowardictionaryp83,Kamviricomparison notloandirection). Noimmediatedonoredges. Totals65proposals/136records,71graph-validrows,94examinedunresolved,541unexamined-or-unresolved. Noacceptedchanges.

## Ring loan-family checkpoint

Batch041 aŋgūśterī ring: r-bearing Persianfamilycompare supportedbyCDIAL137/138Kati distinction andPalula aŋguśtē̂ri. Nuristaniintermediarypossible,notproven. Oneproposal/record,nodonoredge. Totals66proposals/137records,71graph-validrows,94examinedunresolved,540unexamined-or-unresolved. Noacceptedchanges.

## Half-family and quantity checkpoint

Batch042 kʰana/kʰani part/half two records under3792khaṇḍa, hardwithunexplainednasal/clusterhistory andloan/inheritanceopen. dū-o-kʰana twoandahalf one record/twoorderedcomponents, easystructurebutconditionalonhalfproposal. Fourrowsvalidate7tempchanges,second0. Totals68proposals/140records,75graph-validrows,94examinedunresolved,537unexamined-or-unresolved. Noacceptedchanges.

## Consistency and hold audit

Confirmed140pendingIDsunique,94unresolvedIDsunique,disjointsets,andzero pending/heldoverlapwithcurrentacceptedoverlay. Marked5gaṭrecords held_by_user, includinginflectionsandcausatives; no lexicalchanges. Addedovernight-consistency-audit.json. Reviewnowshowscross-proposal savedependencies explicitly and labels historical links proposed. Counts68proposals/140records,75validrows,94unresolved,5held,4sourceissues,532unexamined-or-unresolved.

## Automatically approved approximate numeral saved

Batch043 pā̃c-o cōr approximatefour/five (literallyfiveandfour) saved understandingnumeralapproval:one record,twoorderedcomponentrows. PrimaryPerderp165explicit. Bothbasesalreadyreviewed. Latestoverlaypreservedatomically,backup/tmp/dameli-before-batch43.csv; validation3tempchanges,second0,parsedoldrows+2verified. Pendingcountsunchanged68proposals/140records,75validpendingrows;94unresolved,5held,4sourceissues,1savedovernight,531unexamined-or-unresolved. NoDBrebuild.

## Marriage noun checkpoint

Batch044 nikā wedding three records, probableArabic-originregionalterm with OPED34605Pashtonikā/nikāh. Immediate routeunknown; masculineDmgenderpreservedvsPshtfeminine. Separatefromheldgaṭfamily. Totals69pendingproposals/143records,75validpendingrows,94unresolved,5held,4sourceissues,1savedovernight,528unexamined-or-unresolved. Noacceptedchangesafterbatch43.

## Invariable pink adjective

Batch045 gulabi pink, sourceinvariableadjective, probableborrowedregionalcolourword withOPED32639Pashtocomparison. No claimfinal-i isDmfeminine ornewrosederivation. Oneproposal/record,donorunresolved. Totals70pendingproposals/144records,75validpendingrows,94unresolved,5held,4sourceissues,1savedovernight,527unexamined-or-unresolved. Noacceptedchangesafterbatch43.

## Pure colour intensifier

Batch046 pak pure two records, probablePersian-originregionalword. PlattsreproductionverifiedmarksPersianpāk; DmTables40/44specificcolouruse preserved. Short-a andimmediate routeunresolved; existingPersiannode notusedasdirectdonor. Totals71pendingproposals/146records,75validpendingrows,94unresolved,5held,4sourceissues,1savedovernight,525unexamined-or-unresolved. Noacceptedchangesafterbatch43.

## Original, final, and finally checkpoint

Batch047 adds three probable regional loan proposals across five records: asili/asli original, axiri final, and āxir finally. Platts entries verified; Perder explicitly treats axiri as a borrowed invariable adjective. Immediate donors remain unresolved. Totals74pendingproposals/151records,75validpendingrows,94unresolved,5held,4sourceissues,1savedovernight,520unexamined-or-unresolved. No accepted changes after batch43.

## Garlic and turmeric contact comparisons

Batch048 adds probable Pashto loans ūgā garlic and kūrkāmān turmeric, each with exact compiled Hallberg survey comparanda and independent OPED lexical evidence. Original survey-page visual verification remains outstanding; direction is inferred. Both existing donor nodes fail linkability validation, so two rows remain blocked. Totals76pendingproposals/153records,75validpendingrows,94unresolved,5held,4sourceissues,1savedovernight,518unexamined-or-unresolved. No accepted changes after batch43.

## Potato, groundnut, cauliflower checkpoint

Batch049 records three qualified regional loan-family proposals. ālū potato checked against CDIAL1388 and Khowar dictionary p4; pʰalī groundnut against Platts pod and exact Pashto survey form; gulgopī cauliflower against full Pashto gval gopī. Inheritance plus semantic shift remains open for potato/groundnut; cauliflower is hard. Original Hallberg survey checks remain outstanding. Totals79pendingproposals/156records,75validpendingrows,94unresolved,5held,4sourceissues,1savedovernight,515unexamined-or-unresolved. No accepted changes after batch43.

## Kinship and title checkpoint

Batch050 adds sardār chief as probable Persian-origin regional title and sāṇḍu wife’s sister’s husband as a regional Indo-Aryan loan-family comparison under conditional CDIAL13875. Direct donors unresolved. bay elder brother and bibi elder sister separately examined and retained unresolved, with expressive/inherited versus borrowed alternatives. Totals81pendingproposals/158records,75validpendingrows,96unresolved,5held,4sourceissues,1savedovernight,511unexamined-or-unresolved. No accepted changes after batch43.

## Great-grandchildren and ritual brotherhood

Examined kaṛvāsa/kaṛvāsi against source kinship diagrams and navāsa pattern; initial kaṛ remains unexplained. Examined dram blood-brother with Perder ritual note32 and CDIAL6753 dharma; similarity insufficient for ancestry. Three records added to unresolved review. Totals81pendingproposals/158records,75validpendingrows,99unresolved,5held,4sourceissues,1savedovernight,508unexamined-or-unresolved. No accepted changes.

## Trouble and happy adjective checkpoint

Batch051 taŋ trouble compared with Persian-origin tang and the exact tang karnā light-verb pattern in Platts; immediate donor unresolved. Two kʰośan happy attestations examined with final-an/initial-aspiration problems retained unresolved. Totals82pendingproposals/159records,75validpendingrows,101unresolved,5held,4sourceissues,1savedovernight,505unexamined-or-unresolved. No accepted changes after batch43.

## Bicycle loan family and consistency audit

Batch052 groups sekal and locative sekal-a bicycle as probable English-origin regional loan, immediate donor unknown. Perder Table11 confirms ordinary locative inflection. Updated audit:161pendingIDs and101unresolvedIDs unique/disjoint, pending andheld disjointacceptedoverlay,775inventoryIDsunique. Totals83pendingproposals/161records,75validpendingrows,101unresolved,5held,4sourceissues,1savedovernight,503unexamined-or-unresolved. No accepted changes after batch43.

## Time-of-day unresolved checkpoint

Grouped four gurum morning/gurma ki tomorrow records using Perder’s explicit phrase explanation, without assigning an unsupported ancestor. mākām afternoon versus Pashto evening retained as a semantic/phonological problem. digar-a afternoon has a Persian formal lead but inspected Platts lacks that sense. Six records added to unresolved. Totals83pendingproposals/161records,75validpendingrows,107unresolved,5held,4sourceissues,1savedovernight,497unexamined-or-unresolved. No accepted changes.

## Buffalo contact comparison

Batch053 mexā buffalo: exact compiled Pashto mexā comparator, with dialect x/ś variants and independent OPED buffalo-family evidence. Direct donor unlinked; validator rejection retained. CDIAL9964 and Katir addendum do not establish direct Dameli inheritance or Nuristani mediation. Totals84pendingproposals/162records,75validpendingrows,107unresolved,5held,4sourceissues,1savedovernight,496unexamined-or-unresolved. No accepted changes.

## Guess and capture complements

Batch054 adds andaza guess and kabza capture/unknown table complement, three records. Exact Platts nominal and light-verb senses checked; immediate donors unresolved. Blank source gloss retained, example172 supplies independent capture evidence. Totals86pendingproposals/165records,75validpendingrows,107unresolved,5held,4sourceissues,1savedovernight,493unexamined-or-unresolved. No accepted changes.

## Knowledge and understanding checkpoint

Batch055 pata knowledge checked against Platts patā and patā lagānā, plus Perder independent example134. Immediate donor unresolved. pui complement in understand separately examined against Pashto poh šu/pohedəl; exact form/borrowed unit unresolved. Totals87pendingproposals/166records,75validpendingrows,108unresolved,5held,4sourceissues,1savedovernight,491unexamined-or-unresolved. No accepted changes.

## Beginning complement checkpoint

Batch056 groups two śirō beginning records, verified in Perder example40 and Table39 against Platts shurūʿ and shurūʿ honā. Immediate donor and vowel adaptation unresolved. Totals88pendingproposals/168records,75validpendingrows,108unresolved,5held,4sourceissues,1savedovernight,489unexamined-or-unresolved. No accepted changes.

## Meeting-family comparison

Examined milau meeting against CDIAL10133 explicit Dameli mili, Platts milāp/milāw comparison and Kalasha miláw hik. Native formation versus regional borrowing and final-au unresolved. Totals88pendingproposals/168records,75validpendingrows,109unresolved,5held,4sourceissues,1savedovernight,488unexamined-or-unresolved. No accepted changes.

## Fairy term phonology check

Examined perẽĩ fairy/djinn against Platts parī and compiled Khowar parí. Perder uses the form specifically to demonstrate nasal diphthongẽĩ; neither that ending nor immediate donor is explained. Added to hard unresolved review. Totals88pendingproposals/168records,75validpendingrows,110unresolved,5held,4sourceissues,1savedovernight,487unexamined-or-unresolved. No accepted changes.

## Dative-benefactive postposition checkpoint

Five ki to/for records grouped by Perder section12.1.1. CDIAL2814 kr̥té and3428 kr̥tya supply regional functional comparisons but no established Dameli sound history. Retained unresolved. Totals88pendingproposals/168records,75validpendingrows,115unresolved,5held,4sourceissues,1savedovernight,482unexamined-or-unresolved. No accepted changes.

## Why interrogative checkpoint

Two ku why records examined against CDIAL3166 kím u and3271 kútaḥ; neither contraction established for Dameli. Kept separate from ku do/or. Pending kuitē compound blocker remains. Totals88pendingproposals/168records,75validpendingrows,117unresolved,5held,4sourceissues,1savedovernight,480unexamined-or-unresolved. No accepted changes.

## Here deictic checkpoint

Three ayā here attestations grouped from Perder’s proximal/spatial paradigm; compared CDIAL1605 iha/ia/iyya and228 atra without forcing unexplained vowel or consonant history. Totals88pendingproposals/168records,75validpendingrows,120unresolved,5held,4sourceissues,1savedovernight,477unexamined-or-unresolved. No accepted changes.

## Help and remembrance checkpoint

Batch057 adds madad help and yat remembrance with Platts nominal and light-verb comparisons; immediate donors unresolved. Three ṭãu colour-intensifier attestations grouped as hard unresolved after checking Perder phonology and colour collocations. Totals90pendingproposals/170records,75validpendingrows,123unresolved,5held,4sourceissues,1savedovernight,472unexamined-or-unresolved. No accepted changes after batch43.

## Return, decision and religious school checkpoint

Batch058 adds vāpas back, faisala decision and madrasa school, six records, checked against Perder contexts and Platts. The three madrasa records are explicitly locative forms, including coalesced madrasā; ordinary inflection is not derivation. Immediate donors unresolved. Totals93pendingproposals/176records,75validpendingrows,123unresolved,5held,4sourceissues,1savedovernight,466unexamined-or-unresolved. No accepted changes after batch43.

## Bazaar, grief and parrot checkpoint

Batch059 adds bāzar and locative, ɣam and locative, plus hard toti parrot comparison; five records. Bazaar/parrot checked against Platts; grief against the actual Rekhta Dictionary entry, not misattributed to Platts. Parrot vowel correspondence remains open. Totals96pendingproposals/181records,75validpendingrows,123unresolved,5held,4sourceissues,1savedovernight,461unexamined-or-unresolved. No accepted changes after batch43.

## Book and refreshed consistency audit

Batch060 adds kitap book with Platts kitāb comparison and qualified final-devoicing account. Chair research not completed because dictionary retrieval failed; kursi remains unexamined. Audit refreshed:97pendingproposals/182unique records,123unique unresolved records, pending/unresolved/held disjoint, pending and held absent from accepted overlay,775unique inventory records. Totals75validpendingrows,5held,4sourceissues,1savedovernight,460unexamined-or-unresolved. No accepted changes after batch43.

## Journey and chain checkpoint

Batch061 adds safar journey and hard ẓanẓer chain with Platts comparisons. Chain also has compiled Palula retroflex comparanda, original pages unverified; no Nuristani mediation asserted. Totals99pendingproposals/184records,75validpendingrows,123unresolved,5held,4sourceissues,1savedovernight,458unexamined-or-unresolved. No accepted changes after batch43. Next scheduled run is the 9 a.m. handoff: refresh audit, report exact triage and remaining scope, pause heartbeat.

## 9 a.m. handoff

Research paused for user review. Final pending review:99proposals/184records; difficulty counts {'moderate': 52, 'hard': 12, 'easy': 35}. 123examined unresolved records,458unexamined-or-unresolved,5held,4sourceissues,1savedovernight. Consistency checks pass; accepted overlay unchanged since batch043. Remaining inventory is not exhausted. Main blockers: immediate donor identification, phonological correspondences, primary source access and source repairs.

## All proposals approved and valid assignments saved

User approved all99 overnight proposals and set Pashto-best-match then Urdu-Perso-Arabic donor preference. Saved 25proposals/60records/75rows to accepted overlay, with current-overlay preservation and temporary graph validation, second application zero. Remaining 74proposals/124records are approved_blocked, not awaiting review. The5win holds and123unresolved records remain unchanged. See overnight-approval-20260910.json for exact blockers and validation. NoDBrebuild.

## Sourced donor entries and approved links saved

Added 33 donor heads and reused 22; registered 33 real-build identities and appended 114 accepted assignment rows for 61 proposals / 106 Dameli records. All previous identity and overlay rows preserved. Temporary graph: 220 changes, second application zero. Cumulative approved overnight set: 86/99 proposals saved, 13 still blocked. Held win family untouched. Full ingestion validation incomplete owing to disk exhaustion and global survey test failures; browser DB remains unchanged.
