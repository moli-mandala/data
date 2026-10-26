# Latest continuation pointer

Active diversity pass: `../belari-20260922/STATE.md`, checkpoint001: five pending records,103 unexamined, canonical dry-run passed. Pattapu025 initial assessment complete:68 pending,33 holds,0 unexamined. Four passes registered in goal-pass-registry.json. Older counts below are historical; use each pass STATE and current validation. Continue research; no rebuild or accepted saving.

# Autonomous continuation handoff

The user requested eight hours of autonomous research, no rebuild, and review afterward. Work began approximately 2026-09-21 02:50 PDT. The user has superseded the original deadline: **unlimited time now**. Continue the active goal. A persistent goal now drives continuous work. The redundant heartbeat `dravidian-survey-etymology-research` is PAUSED. At the deadline, validate and deliver the review link. Do not ask the absent user to approve batches during the window.

## Current state

- 37,304 initially unlinked survey records in 16 reference groups. Broader dictionaries are inventoried but outside this survey scope.
- 11,714 proposed assignment rows for 10,958 records under **612 actual parent IDs**, in 39 languages. Printed counts now use distinct selected parent IDs.
- 2,035 explicit holds, 24,311 unexamined records. No new accepted sidecar rows have been saved. This matches deferred review; continue research without waiting for approval.
- `REVIEW.md` links 851 review-only batches, five family proposals per batch. Each manifest has exact IDs, records, citations and proposed assignments. `review-manifest-paths.json` records this pass's ownership; do not overwrite other passes.
- `initial-research-validation.json`: shared `make check-pass` dry-run passed for 3,529 rows, 7,058 changes (edges plus statuses) on first temporary application, zero on second; no unrelated edge moved. No build took place.
- Initial uv invocation failed because the sandbox cannot write its default cache. Use **`PY=.venv/bin/python`** with make targets. A subsequent dry-run failed due to another task modifying sidecars during hashing; a fresh retry passed. Do not stop other tasks. Compiled forms/edges, persistent identities and browser DB remained unchanged.

## Files and reproducibility

Run helper scripts from the workspace root unless specified:

1. `data/.venv/bin/python data/curation/etymology-lab/dravidian-surveys-20260921/prepare.py`
   - Applies manually authored `families.json` narrow form/gloss allowlists.
   - Applies manually inspected `direct-reviewed.json` rules to `direct-candidates.json`, with same-language dictionary evidence.
   - Applies `held-reviewed.json` and explicit exceptions in prepare.py.
   - Rewrites proposed-decisions, primary-evidence, selection-preview. It never writes accepted sidecars.
2. `.../render.py`: refresh owned per-language review batches, REVIEW.md, research-holds, unexamined-inventory.
3. From `data/`: `make check-pass PY=.venv/bin/python DECISIONS=curation/etymology-lab/dravidian-surveys-20260921/proposed-decisions.json PASS=<new-checkpoint> > curation/etymology-lab/dravidian-surveys-20260921/check-pass.log 2>&1`
   - This is a focused dry run against a temporary graph, not a rebuild. Parse the JSON after the first command line to a named validation JSON. Do not call save-pass before the deferred user review.

The decisions JSON uses the shared helper's `accepted` input-key but every item and the enclosing file explicitly have pending-review status. This is a validation format, not approval.

`discovery.mjs` invokes the real homepage `fetchLemmaList` via an esbuild shim against the existing read-only browser DB. Run as `node <path>/discovery.mjs REQUESTS.json RESULTS.json` from workspace root. Requests are actual ListOpts; source_ids=dedr works, though some dictionaries have dedr among multiple citations. Reflex queries are paginated up to 20 pages: counts greater than 1,000 indicate incomplete discovery. Do not treat those as exhaustive. Search logs are already present: 146 semantic queries, 59,171 result rows. Prefer reusing them and scoped additional queries.

`inspect-primary.py d123 d456 ...` prints full shortlisted DEDR articles from the local cached primary audit, with their printed page number, and saves the inspected articles. Do not scan the full source corpus to discover candidates. The attempted DSAL web tool access failed; cached primary entries were available.

`find-direct-candidates.py` creates same-language, folded-form plus compatible-gloss candidates from existing search logs. It now retains the full candidate pool and imports `research_common.py` without mutation side effects. The full pool was regenerated and reviewed direct-family rules expanded. Candidate similarity is not evidence.

`prepare.py` now re-reads all accepted/pending sidecars with read_assignments and excludes any target already represented there; `externally-assigned-records.json` records exclusions (currently zero). Family rules can restrict languages or exclude languages. `parent-holds.json` moves ambiguous parent metadata cases into explicit holds, preserving prior proposal evidence. Renderer marks any surplus old owned batches as superseded and removes them from the current index.

New research in checkpoint 003: one/nine, good/big/old/hot/cold, demonstratives, mother/kin, rope/path, eat/die, northern head/nose/mouth/back/door, fish/monkey/fly/long/forehead, new/flower/star/sand/fat, small/long/red/green/dry/wet/bad. Exact rule allowlists and complete primary articles retained. 4,094-row dry-run passed in checkpoint-002-validation.json. Checkpoint 003 passed for 4,492 rows: 8,984 first-application changes, zero second-application changes; no unrelated edges moved. See checkpoint-003-validation.json.

## Research standards and next work

- Read current sidecars through read_assignments before adding proposals; other tasks are active. The initial inventory and hashes are fixed research snapshots, not permission to override newer accepted rows.
- Keep original spellings, spaces, diacritics and glosses. The form fold is for candidate matching only; it folds several phonetic conventions, so manually inspect all new selected variants.
- `selection-preview.txt` shows family/language/form counts for basic-family rules. Direct proposals also carry exact dictionary comparator objects.
- Continue with unexamined survey rows, prioritizing source-specific groups, common verbs, kinship, remaining simple nouns and justified compound analyses. There are many survey forms with qualified or combined glosses; the initial exact-gloss pass deliberately left them aside.
- Do not blindly link higher Gadaba/Konda numerals to a remote Dravidian root: many look Telugu; immediate donor versus inheritance needs evidence. Some Toda survey forms fail expected Toda correspondences and are held for source/dialect scrutiny.
- Source-defined multiword/slash responses must be analyzed as full responses, not shortened to one stem. The current shared helper supports an ordered components array in one decision per record. Use that interface; do not duplicate a record across decisions or modify the helper to bypass validation.
- Nursery terms require the actual local role: DEDR itself says Irula/Tulu amme father, while Kannada amma means mother.

## Primary-source problems already identified

Do not silently repair these unrelated reconstruction/node problems or attach new forms despite them:

- d585 has a stored body reconstruction, but primary 585 is uṭakku ‘fit an arrow’. Correct body article is **586**, used in proposals.
- d783 has a stored eight reconstruction, but primary 783 is eṭṭu ‘reach’. Correct numeral article is **784**, used in proposals.
- d572 has a stored pestle reconstruction, but primary 572 is ‘ring’. Correct pestle article is **672**, used in proposals.
- d3485 has a stored younger-brother reconstruction, but primary 3485 is toṭṭi ‘enclosure’. No proposal uses it.
- d3690 primary water article is headed by stored *kaṇ-ṇīr ‘tears’; new water proposals are held pending parent-label resolution.
- d651 primary mortar article is headed by stored grinding-stone/pestle *uram-kkal; new mortar proposals are held.
- d1977 has distinct ear subsections; (b) is explicitly separated from (a), but one node headed *kir- conflates them. New subsection-(a) proposals are held.
- d5496 white/silver subsection (a) is under stored butter compound *veṇ-ṇey; held pending subsection resolution.
- DEDR explicitly questions Gondi near (1293) and meat (3373), Kurukh dog (2916), sun (5479), big (2177), possible IA belly origin (2244), and Parji earth (3676). Holds preserve these doubts.
- DEDR warns of possible IA loans among hair/topknot (2655) and Gondi/Kuvi/Malto seed nouns (5401). Only Kannada seed matches were proposed.
- Cached article 575 begins with an unexpanded “id.” and needs fuller primary markup verification before sand proposals.

No rebuilt artifact, importer output, donor node, commit, push or publication belongs to this task. At final handoff, report actual proposed/held/unexamined counts and validation, and link REVIEW.md. Do not claim all remaining surveys are finished if unexamined records remain.

Additional parent holds discovered in checkpoint 003: d1594 simple small under *kiṯṯ-eli mouse compound; d3821 green under *pac-Vṯ- green gram; d3045 taṇ- cold under *caḷ-; d3610 good under *nan-key wetland compound. All four now held via parent-holds.json. DEDR d3136 Gondi cow is explicitly questioned and held.

Next useful work: remaining direct candidate families (recalculate ranking against current decisions), broaden verified dry d601 and cold caḷi/kuḷir families, source-specific numeral/colour borrowing evidence and immediate donor resolution, and remaining high-frequency simple nouns (needle, door, name, year, etc.). Existing result logs already have most semantic searches; add real homepage queries for missing terms. Fully inflected Gondi verb phrases and negative adjectives require morphological/component analysis and should not be collapsed by fuzzy matching.

Checkpoint 004 adds 96 proposals: DEDR 3753 rice/paddy, 4813/142 kin, 1654 drink, 4303 village, 1769 monkey, 3283 earth/clay, 775 leaf, 3085/23 sibling, 3015 sister (Gondi plain wife held for semantic mismatch), 400 woman/wife, 1892 hair, 2030 boy/girl; plus 601 dry, 1834/2408 cold. Full primary articles read; all parent labels checked. Additional homepage search batch for missing common nouns is being logged in additional-discovery-results.json. find-direct-candidates.py now includes that file if present; compare the regenerated pool against pre-additional-direct-candidates.json before allowing expanded matches into reviewed rules.

Additional searches completed: 28 queries, 4,404 result rows, 83 new same-language candidates. Pool regenerated; only new match in an already reviewed family is Telugu anta all under d1, excluded by that rule’s gloss filter. All other delta parents need primary checking. checkpoint-004-validation.json passed for 4,588 records with zero second-application changes.

Latest after checkpoint-004 validation: added 41 proposals (5381 sky/rain, 3805 morning, 4889 face, 2599 broom, 4892 sky/cloud, 3939 boy/girl, 1692 dust); all full primary text and parent nodes checked. Four Parji evening records held because DEDR 3621 marks that comparison with ?. Current 4,629 needs next shared dry-run; last passed at 4,588. Continue primary-checking additional-direct-candidate-delta.json parents, particularly d5074/3331/3987 and compounds d990 all (may not be simple one).

Checkpoint 005: 4,834 proposals under 189 parents, 401 holds, 32,069 unexamined, 332 batches. Dry-run passed, 9,668 first changes, zero second. Added direct 5074 sky, 3331 full/many, 3987 many, 705 garlic, 4764 son, 4570 gold, 5532 roof, 990 Kurux all (explicitly attested in subsection a); then 844 all, 5153 year, 2562 spider, 1571 some, 4784 flour, 4405 morning, 4914 cloud, 5264 AluKurumba roof, 528 roof, 3332 dust, 4263 basket, 5357 comb, 1456 cultivator, 2408 Konda morning. d2807 Malto far and d4841 Tamil float held because primary explicitly questions them. Broad family allowlists expanded sky/face/broom/all, preserving compounds. New source candidates and primary evidence are cached. remaining-direct-candidates.txt lists the next exact matches to investigate.

Checkpoint 006 research: 5,085 proposals, 211 parents, 406 holds, 31,813 unexamined, 350 batches. Added source-checked exact matches for 5052 three, 3986 Kannada tooth, 2779 yesterday, 4747 buffalo, 72 foot/below, 4493 finger, 3794 see, 5033 cloud, 3805 Parji day, 1395 black, 1124 bite, 2053 give, 5116 cow, 989 yesterday, 4020 fly/bird, 4530 eye, 2528 night, 3275 bird, 3067 father, 4198 child, 334 cow, 3684 second-person singular, 4682 head, 5259 belly, 4193 Gondi belly. Held Parji pered river because primary explicitly analyses big + river (5159/4411); Brahui piḍ belly is questioned (4193). Broad allowlists expanded black/child/singular-you/give/see/Kurux yesterday. Primary article and parent metadata checked for every new family. checkpoint-006 dry run is running; inspect log. remaining-direct-candidates.txt refreshed.

While checkpoint 006 validates, fully read the primary articles 4154 bird, 4559 sun/time, 3752 Kurux snake, 2096 ant, 4567 mountain, 1478 river, 1288 bone, 3746 fat. Next-parent-inspection.json records their node labels. Gondi ant 2096 is explicitly questioned and must be held. Others have direct matches; not yet added to rules.

CRITICAL next-parent audit: d1288 primary is Kurux xōcol bone / Malto qoclu, but existing parent is *kar-umpu sugar-cane. HOLD all proposed d1288 candidates. d4559 primary time/sun article is stored as *ap-pōẓ that-time compound; HOLD simple sun links pending parent resolution. No rules for these two have yet been added.

Checkpoint-006 validation retry PASSED for 5,085 rows with zero second-application changes. Original failed only because other work changed shared inputs; failure log retained. Then added 15 direct proposals under 4154/3752/4567/1478/3746 and 9 holds under 2096/1288/4559. Current 5,100 proposals and 415 holds await next batched validation. These next rules ARE now installed; earlier handoff notes saying not yet added are superseded.

Checkpoint 007 research: 5,131 records, 5,139 assignment rows, 224 parents, 411 holds, 31,762 unexamined, 356 batches. Added 2359 snake, 1693 simple heart, 3758 yesterday, 3903 stone, 4096 Kannada milk, 4954 kin/old, 1187 door, 3647 we. Direct rules now support an optional forms allowlist. On reinspection, the CURRENT shared etymology_lab.py supports ordered components arrays (apparently changed by other workspace work); no shared helper edits made here. compound-rules.json holds explicit record IDs for Parji big+river (4411+5159, four records), Gondi heart+fruit (1693+1459, one), Kannada granddaughter (4715+4616, three). All component primary entries and parent labels checked, source order preserved. prepare.py and render.py support these research-only compound decisions, counts distinguish records from assignment rows. Rendered manifests exactly agree with shared helper rows_from_decisions for all 5,139 rows; sample compound review tables inspected. Shared checkpoint-007 dry run PASSED: 5,131 records / 5,139 rows, zero second-application changes, no unrelated edges moved.

Checkpoint 008 research: 5,307 records / 5,344 rows, 233 parents, 411 holds, 31,586 unexamined, 373 batches. Added 1263 bear, 215 rice, 4735 lie/sleep, 5430 fall, 2216 Gadaba cow (lect-specific cow/bullock distinction explicitly noted), 3516 tomorrow, 4410 name, 4300 grass, 4726 open forest. Broadened narrow form lists for name/rice/grass. Added 29 transparent comparative compounds: āṇ + piḷḷa, peṇ + piḷḷa, peṇ + makkaḷ, āṇ + makkaḷ, āṇ + paiyaṉ. Full source evidence for 399/4198/4395/4616/3939 checked; longer pembuḷa/āmpuḷa sequences excluded. Now 37 compound records. All rendered assignments agree exactly with shared helper output. Checkpoint-008 dry run PASSED for all 5,344 rows / 5,307 records, zero second-application changes and no unrelated edges moved.

## Checkpoint 010 continuation notes

Latest dry-run checkpoint-010-validation.json passes all 5,314 rows (5,277 records), second application zero; manifest/helper row agreement also passed. Remaining time extends to17:50 UTC; latest clock11:07UTC. This turn added exact same-language noun/verb matches and two Muduga forms supported by the cached Arsenault–Abraham primary paper (muduga-primary-supplement.json). Direct rules now accept optional additional_citation. Parent-holds entries can be dicts with reason and except_forms; d990 now permits only verified okk-family spellings, moving other numeral formations to holds. This reduced the proposal total while improving subsection precision.

Last inspected/processed direct groups through d2132: d5450, d5050, d4316 (held for probable loans), d1097, d811, d5270, d600, d1448, d3413, d2132. Next refresh remaining-direct-candidates.txt and work from d314 onwards. Prior remaining file is stale. d55 held for nominalization relation, d1659 expressive-family parent mismatch, d4124 pot-parent reconstruction issue, d2426 causative kill, d698 suffixed mind-parent vs house. The d990 hold catches unselected Kurux all and Kannada person forms too. d1278 only Gondi kari subsection(c) selected, Kannada karu(a) held.

## Checkpoint 013 continuation notes

Latest dry-run checkpoint-013-validation.json passes all 5,362 rows (5,325 records), second application zero; manifest/helper agreement also passed. Current clock11:14UTC; deadline17:50UTC. Current turn was substantive progress (48 additional proposals and15holds since checkpoint010).

Added relation-rules.json for explicit existing-donor or derived links, with record_ids, parent, evidence_parent, kind, citation, evidence. prepare.py compiles those after compounds and excludes any now-assigned records. Renderer now handles attested donor parents using evidence_parent for DEDR page links, explicitly labels Borrowed from/Derived from, and says parent nodes rather than DEDR articles. Two borrowed proposals: Konda f_76qulsw3rba4e pāṭa cloth → Telugu f_w4qqoh3i4j254 pāta cloth (DEDR3999 explicit loan); Gondi f_gwfiwv42eimlg rokal pestle → Telugu f_q64arbjsvuxda rō̃kali (DEDR665 explicit loan, deeper etymology questioned but donor clear). No donor nodes created. Borrowing review tables visually/textually inspected.

This turn processed remaining direct groups through d3291, including verbs, negation, numerals, kinship, smaller fauna and nouns. Remaining direct file last refreshed before d2991–d3291 processing and is stale. Next refresh and inspect from d4641 onward. New holds include questioned Toda pem, Kui miḍa, Gondi mortar/smoke; false noun-water candidate d2404 ar actually a verb drain water; d1175 meat probableIA; d2110 hair contact; d3728 stored *nūl thread contradicts primarycrushing/brokenrice. Tamil taṇṇīr compound and Gadaba maginḍsinḍ still unexamined. d3650 Konda nukuṛi held pending probable dog+young component analysis. d410 īndu today needs wholeformation analysis.

## Checkpoint 015 continuation notes

Latest dry-run checkpoint015 passed for5,390rows/5,353records, second applyzero. Current researchadds9more (5,362records/5,399rows), awaiting nextbatchedvalidation. Current manifests match helper rows. Clock11:19UTC,deadline17:50UTC. Next checkpoint016.

Processed direct candidates from d4641 through d1391. Fullprimaryread androotmetadataaudits preserved. False matches: d4979nose from nose-ornament definition held; d5221run asink correctedto d1041running family via realsearchresults, newly added narrowKannadaōḍ/ōḍī rule. d4818Kannadamār chest heldbecause dictionaryactuallyfathom; chest appearsincidentally. d3549dig heldbecauseparentgarden formation; d4243ashes heldbecauseparentmastwoodwrong; d645Brahuiust heart alternatives645/698 held; d4203Gondiporki louse explicitlyquestioned. d869Telugueṇḍudry heldpendingnounsun vsverbformation.

Remainingdirectfile staleafter d1680–d1391processing; refreshthenstartd1864onward. Note ordinary Cf.Sanskrit isnotautomaticproofloan; latestTamilbelly,Kuvi belly/blackproposalsstatecomparativefamilywithopencontacthistory. Some earlierholdsforanalogousmerecrossreferences canberevisited with thisdistinction. Explicitloan statements stillrequireimmediatedonors.

## Checkpoint 018 continuation notes

Latest dry-run018 passes5,454rows/5,417records,secondapplyzero;manifestagreementpassed. Clock11:26UTC,deadline17:50UTC. All direct-candidates now processed: remaining-direct-candidates.txt empty. Do not regenerate exactcandidatepool without auditing new matches to all existing direct rules.

Added explicit GondiKoya erra red borrowing to existing Telugu f_wpmy7hes7n65i eṟṟa red, evidenceDEDR865. Distinct homonymous Telugu earthworm f_rtred35nura4k excluded. relation-rules.json has3borrowedproposals. New wrongparentholds: d916stored dog vsrise/stand article;d864stored hill vsant article. d3016catai flesh expresslyquestioned. Added mostremainingTamilnouns/verbs andKuruxverbs;nominalizations/compoundmirrorheldpendingproperrelations. Tamilrun oṭu addeddirectd1041, alongside narrowKannadafamilyrule. Allprimarysnapshots retained.

NEXT: broaden beyond exactfold matches. next-broad-discovery.json gathers prior actualhomepage searchresults forspider,broom,firewood,garlic,needle,lightning. Lightningempty (querynotsearched exactly);needle results mostlyfalseglossmatches and mayneedfreshquerypagination. Major parents toreadprimarythencompare unexaminedforms: firewoodd5440(21searchrows),spiderd2195,d2562,d2809,d1800;broomd2599,d1088,d4415,d4794;garlicd705 (watchwhite+onioncompounds). Existing direct rules include some ofthese?checkbeforeedit. Current unexaminedlargestlangs:Gondi6767,Irula3526,Kurux3011,Kannada2553,Muthuvan1541,Gadaba1445,Mannan1377. Commonremainingglosses man205,many180,far167,twenty164,woman162,white162,year158,sun157,needle156,near152,banana150,father148,twelve148,hair146,gold146,door143,cloud143. Candidatefetchqueriesalready146+28 pooled;reuseforlexicaldiscovery.

## Checkpoint 019 continuation notes

Latest019dryrunpasses5,575rows/5,538records,secondapplyzero;manifestagreementpassed. Clock11:30UTC,deadline17:50UTC. Thisturnadded121records afterfullprimarychecks inbroadfamilies: d5440firewood (65currentrecords,8Gondi),d2195spider4,d2562spider19includingprior1,d4876lightning34new. Exactlanguage/form/glossallowlists infamilies.json. Broadcomparativequalification statesinheritance/contactunresolved forminorlects absentDEDR. Gondi firewoodva/ve variants sourceTagscheckedMaria/Muria/Koya.

Freshactualhomepagequeries component-discovery-requests/results.json: mirror21,white372,lightning37rows. Whiteallmainmatchesd5496stillmislabelledbutterparent;mirrorāṭi componentnotindependentlyfound (mostlyd1182whole,d147Teluguloanaddam). Noaddedcompoundfromthese. Garlicremainingresponses mostlywhite+onioncompounds orIAlasun;don’tshortento ulli. Broomprimaryd2599,d1088,d4415,d4794read;mostremainingresponses longer/uncertain,mārūfamilyd4794explicitTamilquestion. No newbroomproposals. Lightningd4876,d5074,d2759,d4150read;simpleminnal+miñcufamiliesselected,finiteGondimircanta,etc.andMannanminnalarenaexcludedpendingmorphology. Broadfamilycheckpoint019counts retained.

Potentialnextpriority: examine published cognateset annotations in survey-inventory (especially dravlex) as primary source-supported discovery alongsideexistinghomepagesearch; map onlydiscoverygroups toknownparentcandidates, manuallyverifyformsandfullDEDR before proposing. This mayidentify nonexactfold matches without blindsoundmatching. Remaininglargequeue31krecordsneedsfamiliesbeyondbasicsandborrowedterms. Keep surveysourceboundaries and compound/finiteverbforms intact.

## Checkpoint 022 — 11:42 UTC

5,650 records / 5,687 assignment rows, 396 parents, 623 holds, 31,031 unexamined, 433 batches. Shared dry-run and current manifest/helper agreement passed; second temporary application changed zero. No rebuild or accepted writes.

New source-specific path: `cognacy-discovery.json` joins 1,537 still-unexamined Dravlex records to preserved upstream form/cognacy annotations (614 numeric leads). Numbers are heuristic discovery only: they include page numbers, main-versus-appendix ambiguity, and unrelated cross-references. All new proposals use exact record IDs in `relation-rules.json`, with kind reflex and a primary evidence_parent, plus the exact source locator. Broad look-alikes remain explicitly comparative, pending review. `scoped-holds.json` and prepare.py now support exact-ID research holds.

This checkpoint added112 proposals since019 and37holds: first-person pronouns, interrogatives, proximal demonstratives, fish/horn/seed/drink, skin/bone/blood/come/stand. Notable exclusions: conflicting Malayalam/Toda inclusive/exclusive annotations; questionable Toda inno; seed bīj- forms with possible Indo-Aryan origin; Toda road wrongly annotated as black; numeric57 tail refers to Appendix57, with an empty existing parent and Indo-Aryan comparison, not main57 cloth edge. Full appendix saved in inspected-da57.json. Never mass-map upstream cognateset IDs: even Doubt=false and Loan=false accompany questionable or erroneous comments.

New homepage query logs: pronoun-discovery-* (8 queries,1201 rows), demonstrative-discovery-* (this,597 rows). Full primary articles inspected d5160,d5154,d3647,d5151,d410,d57,d5401,d4885,d600,d2115,d3559,d1395,d3748,d839,d3675,d5270 and da57. Reused existing semantic searches for ordinary nouns/verbs.

`verify-checkpoint.py NNN` now verifies a completed check-pass log against current manifests and saves checkpoint-NNN-validation.json. Run it only when current decisions match that checkpoint. Next dry-run checkpoint023. The eight-hour deadline remains17:50UTC; continue research autonomously.

## Checkpoint 024 — 11:44 UTC

5,690 records / 5,728 assignment rows, 396 parents, 626 holds, 30,988 unexamined, 440 batches. Shared dry-run024 and manifest/helper agreement passed; second temporary application zero. No rebuild or accepted writes.

Added39 source-specific reflex proposals after full-primary verification of d474 two, d561 nail/claw, d5024 nose, d3986(a) tooth, d4395(a) woman. Exact source IDs in relation-rules.json; minority-lect extensions explicitly comparative. Three new outstanding holds concern Brahui hor ad claw, Betta Kurumba pəṇakkən woman, and Kodagu ponnāḷə woman, whose extra constituents require analysis. Kota pemagu woman was initially held, then resolved after reading d4616: both d4395 and d4616 explicitly give pe-mog. It is now an ordered compound proposal d4395 female + d4616 child/person. Total compounds38. Check023 before this compound also passed and is retained. Next checkpoint025.

Numeric cognacy leads still need filtering by actual meaning: d474 in Ravula belly comments is a page number belonging to article5259, not two. No erroneous numeric link was proposed. Continue with remaining cognacy leads (d1977 parent mismatch, d5517, d4935, d1931, d1207, d1458, d1514, d3843, d4682, d5516, d4990, d4449, d2559) or broader unexamined survey material. Review each full article, source wording, and current parent metadata. Deadline remains17:50UTC.

## Checkpoint 025

5,706 records / 5,744 assignment rows, 399 parents, 635 holds, 30,963 unexamined, 443 batches. Shared dry-run025 and manifest/helper agreement passed; second temporary application zero. No rebuild or accepted writes.

Added16 source-specific proposals after full-primary reading d5517 heat/fire/sun, d4935 knee, d1931 red/blood, d1207 burning, d1458 heat. Nine additional holds preserve multiword verbs, uncertain allomorphs and semantic mismatches. Fourth explicit immediate-donor proposal: Kodagu cōpə red → existing Malayalam cōppu f_fju3engnmxfee, directly identified as a Malayalam loan by DEDR1931. Donor discovered in existing homepage red results, then full article read. Ravula cōpə is held because the Kodagu loan statement does not establish Ravula’s route.

Next checkpoint026. Remaining high-frequency numeric source leads include d1977 (known ear parent/subsection mismatch), d1514, d3843, d4682, d5516, d4990, d4449, d2559. Numeric leads include irrelevant cross-references; inspect actual semantics before proposing. Eight-hour deadline remains17:50UTC; continue autonomously.

## Checkpoint 026

5,734 records / 5,775 assignment rows,401parents,642holds,30,928unexamined,449batches. Shared dry-run026 and manifest/helper agreement passed, second temporary applicationzero. No rebuild or accepted writes.

Added25 exact-ID reflex proposals after full-primary reading d1514fire,d4682head,d5516hear,d4990knee,d4449louse,d2559negative existential. Added3 ordered knee compounds d4990joint+d1479leg for Kannada muṇakālo,Kota mōgāl,Malayalam mul̪aŋkāl; total41compounds. Seven holds include five pāda/pādam foot responses miscompared to d3843 paṭam instep (long vowel/dental versus short-vowel/retroflex distinction and possible IA loan), Toda maṭṭə head, Toda irs louse (full2625nit alternative also checked but unresolved). Source Loan=false/Doubt=false is never treated as decisive. Inspected full d2625,d1479 as supporting articles. Nextcheckpoint027; continue remaining numeric candidates or broader survey research. Deadline17:50UTC unchanged.

## Checkpoint 027

5,753 records / 5,794 assignment rows,403parents,647holds,30,904unexamined,453batches. Shared dry-run027 and manifest/helper agreement passed after refreshing concurrently changed inputs; second temporary applicationzero. The first concurrent-input-change failure log is preserved. No rebuild or accepted writes.

Added19 exact-ID source-cognacy proposals after full-primary reading d3876bark,d5259belly,d2654burn/warm,d2146fat,d3682full. Full d4316 ashes read; four ashes responses held for the article’s explicit probable Sanskrit-borrowing caveat and unresolved multiword structures. Kota koḷupu fat held because the article expressly questions Kota’s comparison. Ravula belly reference474 correctly treated as page474 of5259, not numeral two. Comparative minority-lect extensions and survey morphology remain explicit in evidence. Nextcheckpoint028; continue source-specific candidates and wider survey scope. Deadline17:50UTC unchanged.

## Checkpoint 029

5,768records/5,811assignmentrows,404parents,662holds,30,874unexamined,454batches. Dry-runs028and029 and current manifest/helper agreement passed; second temporary applicationszero. No rebuild or accepted writes.

Added12source-specific reflex proposals from full d2598give,d497leaf,d399man,d1695stone, plus2knee compounds Tamil/Telugu d4990joint+d1479leg. Upstream knee annotation4490 was wrong (burst); full4990 provides the correct forms. Gondi/Ravula related knee responses held for contact/component analysis. Road look-alikes d3170 and northern round candidates d1695 held; Kodagu ānaaḷə requires handling two semantically distinct components grouped under399. Total43compounds.

Ear correction: full1977explicitly separates subsection(a)cevi/kev from(b)kir. Existing parent d1977 is *kir-V-. Therefore one exact-ID Kuwi kiriyu response is now proposed to1977(b), while five subsection(a)/uncertain Toda responses are held. Do not blanket-exclude all1977: the stored parent is suitable specifically for(b). Existing generic familyholds remain appropriate for(a).

Nextcheckpoint030. Remaining high numeric leads: d1479,d3821,d3103,d5535,d3633,d3690,d5496(4each), then d3582,d990,d3541,d4494,d5276,d4020,d4641,d3650,d1159,d3266,d3098,d4707,d4720. Several have known parentheadword mismatches; select by actual source semantics, not annotation numbers alone. Deadline17:50UTC unchanged.

## Checkpoint 030 — 11:55 UTC

5,780records/5,823assignmentrows,404parents,678holds,30,846unexamined,456batches. Dry-run030 and manifest/helper agreement passed; second temporary applicationzero. No rebuild or accepted writes.

Added12 exact-ID proposals: foot1479, Kuwi head3103, root5535, tongue3633. Added16 holds: green/yellow3821 existinggreen-gramparent, water3690 existingtearcompound, white5496 existingbuttercompound; questioned Kodagu/Ravula talami hair in3103; Gondi talakāy head compound/contact; Kolami vērkam unestablished extra morphology. Full3821,3103,5535,3633,5496 reread, existing parentmetadata checked. Parent mismatches are concrete data obstacles, not approval requests. Do not repair parentmetadata or create new donor/parentnodes during this review-only pass. Nextcheckpoint031; continue remaining cognacy leads and larger survey scope until17:50UTC deadline.

## Checkpoint 031

5,788records/5,831assignmentrows,405parents,686holds,30,830unexamined,457batches. Dry-run031 passed on fresh retry after concurrent input mutation; first failure log retained. Manifest/helper agreement passed, second temporary applicationzero. No rebuild or accepted writes.

Added8proposals and8holds after reading full3582,3541,4494,5276,4020,4641. Proposed belly4494,big5276,Badaga cloud4641,Kota walk3582. Ashes annotation3541 means OLD DED3541=currentDEDR4316(probable Sanskrit loans), not current3541plunder; held. Toda paravai bird flagged possibleTamil loan; Tulu flycomparisonquestioned byprimary, Kota flymorphologyunresolved. Kolami naḍus differs from primaryaḍg-. Toda allannotation3582ismerelycross-referenceinside990, not relevantwalkingetymology.

Remaining numeric leads3each:3650,1159,3266,3098,4707,4720,4677,93,4652,3692,621,865,2735,725,4240,5131,4559,3433. Nextcheckpoint032. Consider batching several primary-review groups before the next dry-run, since shared inputs are being updated by another task and frequent validation repeats the same graph checks. Keep proposals pending and clearly record any rows awaiting validation. Deadline17:50UTC unchanged.

## Checkpoint 032 — 12:01 UTC

5,810records/5,854assignmentrows,411parents,697holds,30,797unexamined,464batches. Dry-run032 and manifest/helper agreement passed; second temporary applicationzero. No rebuild or accepted writes.

Two primary-review groups batched: full3650,1159,3266,3098,4707,4720 then4677,93,4652,3692,621,865. Added21reflexproposals plus1Kota head+hair compound d4682+d4707 (total44compounds). Elevennewholds preserve uncertain dog/eye/fire/give responses, extra material in long constructions, doubtfulToda nerti, Gondi errag/yerrag red where the article’s explicit S/Ko Telugu-loan statement needs lect/morphology reconciliation, and Kolami labāḍ muḍsati lie where untruth versus reclining may be confused.

New discovery-false-positives.json records3 lie responses whose number93 comes from /cognate/93/ upstream group IDs, not DEDR93attic. They remain unexamined for actual etymology: Malayalam kiḍakkuga,Tamil keḍə,Telugu paḍukonu. Do not keep re-reading attic on their behalf. Actual homepage lie search could supply proper candidates. Nextcheckpoint033; deadline17:50UTC unchanged.

## Post032 research — three proposals await next batched validation

Current5,813records/5,857rows,413parents,698holds,30,793unexamined,464batches. Last validated032 covers5,854rows; latest3 await checkpoint033. No rebuild or accepted writes.

Actual new homepage discovery: lie-discovery-* (461rows), lie-form-discovery-* (2queries41rows). Full1524 and3852 inspected. Resolved prior spurious /cognate/93/ for Malayalam kiḍakkuga and Tamil keḍə to1524; its existing parent *kōḍ- (?) is uncertain, explicitly retained in proposal evidence. Primary Ta/Ma comparisons are unqualified, unlike the article’s questioned Malto comparison, which is not proposed. This is reconstruction uncertainty, not a known wrong-semantic/compound parent like water/tears. Added Tamil paḍə to3852. Telugu paḍukonu now explicitly held for full auxiliary/derivational analysis, not shortened to paḍu. discovery-false-positives.json describes the spurious numeric leads historically; two now have real proposals and the third a hold.

Unexamined Irula kaḍa/keḍaṇḍuko etc were observed in lie-down inventory but not linked yet; their full constructions need analysis. Continue with remaining annotated candidates or expanded actual semantic searches. Batch several substantive groups before next focusedcheck. Deadline17:50UTC unchanged.

## Checkpoint 033 — 12:06 UTC

5,827records/5,871assignmentrows,416parents,706holds,30,771unexamined,466batches. Dry-run033 and manifest/helper agreement passed, second temporary applicationzero. This includes all three previously unvalidated lie-down proposals. No rebuild or accepted writes.

Added14proposals this turn after full2735see,725sit,4240smoke,5131northernsmoke,3433white,3160grandfather,4417bark,4559sun/time. Corrected Kota ān I source3160misreference to fully inspected5160. Kuwi mu I remainsheld because neither grandfather3160 nor documentedKuwi nānu5160 supports it. Eightnewholds concern shortOllari cu see, Telugu-likeGondi white/bark, and simple sun forms whose4559parentis thecompound *ap-pōẓ. Source-specific variants remainpendingreview. Nextcheckpoint034; continue until17:50UTC.

## Checkpoint 034 — 12:10 UTC

5,918records/6,010assignmentrows,417parents,706holds,30,680unexamined,475batches. Dry-run034 and manifest/helper agreement passed; second temporary applicationzero. No rebuild or accepted writes.

Expanded beyond Dravlex annotations:91new proposals.48banana compounds d5373banana-plant+d4004ripefruit across explicit reviewed form/language lists (92totalcompounds now),28Gondi many val/vel/vol forms under5276 (primaryTr wallē much/very),11niraya-type many forms under3682,4tumba/dumba many forms under3331. All IDs captured in broad-banana-quantity-candidates.json before rule addition. Compound evidence explicitly marks comparative extensions/contact uncertainty; final vowels, source spacing, parenthetical suffixes, consonant differences preserved. Highly contracted banana forms were excluded; familiar-looking constituent hypotheses are still pending review, not accepted.

Fullprimary5373,3987,4004,3331 read this turn;5276/3682 full articles already read in earlier passes. Actual searchreuse banana/fruit/many pools; quantity-discovery-* adds3queries25rows (tumba,wallē,niṟaiya). wallē literalquery returned0, but fullarticle5276 explicitly documents it. Kannada tuṁba actualsearch anchors3331. Parentmetadata checked for all5relevant nodes.

Current scope audit: about1128unexamined Dravlex cognacy rows at start ofturn,206numeric leads; largest overall unexamined languages Gondi6721,Irula3489,Kurux2988,Kannada2534. Many unexamined needle/year responses are likely loans, so no automaticremote-root links. Needleūci/sūji and yearvarṣam/etc remain for donor analysis. Nextcheckpoint035; continue broader comparative families and remaining numeric source leads until17:50UTC.

## Checkpoint 035 — 12:13 UTC

6,010records/6,113assignmentrows,419parents,708holds,30,586unexamined,484batches. Dry-run035 and manifest/helper agreement passed; second temporary applicationzero. No rebuild or accepted writes.

92new proposals:30mañj-cloud4641,29mōḍa-cloud5033,4mukil-cloud4892,1Telugu mabbu4728,15ār-river5159,11Gondi big-river compounds4411+5159,2Koya mabbu explicit Telugu loans. Compoundtotal103, borrowedrecords6(5existingdonors). Exact cloud/rivervariants reviewed in broad-cloud-river-candidates.json; one candidate Kannadaāttu excluded tohold. Otherhold untaggedGondi mabbu becauseprimaryloanlabelspecificallyKoya. Koya-taggedborrowrecords f_zsvcans3lmy5m (MalakanagiriKoya),f_ieyc6g7evdlcw (DravlexKoya); donor f_3lv7ov4u3m7yg Telugu mabbu. Untagged f_gjesg4sots6sq remainsheld.

Actual cloud-discovery-* search120rows; full5033,4892,4728,5159,4411 read,4641previousfullretained. Parentmetadata checked. Further unexamined clouds include IA-lookingmegha/badal/abar, Paliyamaḷamoḍam rain+cloud compound, heavilycontractedforms. Mannanātto ram river likelyriver-bank compound ratherthansimpleāṟu; notassigned. Nextcheckpoint036,deadline17:50UTC unchanged.

## Checkpoint 036 — 12:24 UTC

6,083 proposed records / 6,186 rows, 419 parents, 709 holds, 30,512 unexamined, 488 batches. Shared dry-run and manifest/helper agreement passed; second application changed zero. Added 73 same-language survey variants after full primary checking; Tamil census ṇān held for initial retroflex transcription. No rebuild or accepted writes.

## Checkpoint 037 — 12:26 UTC

6,113 proposed records / 6,216 rows, 431 actual parents, 725 holds, 30,466 unexamined, 495 batches. Shared dry-run and manifest/helper agreement passed; second application zero. Added 29 reflex proposals and one derived proposal since036. Kota tāvir kill is derived from existing Kota tav- death verb f_qwurp4ggeo6x6, with explicit DEDR3068 support. Six borrowed records, 103 compounds unchanged. Source wrong references and look-alikes held. No rebuild or accepted writes.

Post037 research: two additional relation rules (Tamil tā give→3098, Malto hēḍu good→848) correct wrong/current-versus-old DED references; full primary and actual parent verified. They have NOT yet been rendered or dry-run validated. Next: prepare/render and include in checkpoint038 after further research.

## Checkpoint038 — 12:31 UTC

6,128 proposed records /6,231 rows,437 actual parents,735 holds,30,441 unexamined,497 batches. Shared dry-run and manifest/helper agreement passed,second application zero. Two derived proposals now: Kota kill and Gondi nirus burn (causative), latter→existing Gondi nirvānā f_2hlivtgfwnqpk via2929, correcting3693. Six borrowed records,103compounds. No rebuild/accepted writes.

Next research:132 remaining numeric cognacy leads at this checkpoint. Full primary articles just read:5249,17,3419,141,546,3120,1173,2030,529,5345,4588,728,3373,4975. No rules yet for this group. Clear candidates Kurux know/leaf,Brahui know,Kannada ganḍasu,Malayalam/Tamil meat,Kannada bāḍu,Kota/Toda meat,Gondi/Ollari liver. DEDR5249 explicitly questions Ollari knee;3373 questions Gondi meat;4975 questions Kolami muriya but explicitly suggests Telugu loan. Resolve donor before proposal. Kota ganmagu likely1173male+4616child compound.

## Checkpoint039 — 12:32 UTC (approximate)

6,139 records/6,242 rows,441parents,741holds,30,424unexamined,500batches. Shared dry-run and manifest/helper agreement passed,zero second application. Added11supported meat/liver/know/leaf/man forms. New parent mismatch:DEDR17 Kurukh know currently headed*ampu arrow,held. No rebuild or accepted writes.

Next: fullprimary just read3754,4422,3338,2287,4742,5058,5352,1366,1662,1996,114,4622,4275,2552,4781,3584,3621. No selections from this group yet. Source3584Gondinarkaḍnight likely wrongcompoundmiddle reference; source114Kannadakutʰigeneckwrong; Maltoasardunecklikely1996(old1662),needsmatch. Primary3338Brahuitūbēmoonexplicitlyquestioned. Actualremainingnumericcognacylistcanbefilteredagainstunexamined.

## Checkpoint040

6,147records/6,250rows,445parents,748holds,30,409unexamined,502batches. Shared dry-run/manifest agreement passed,zero second application. Added8moon/mountain/mouth/neck/night proposals;7holds for primary doubt/transcription. No rebuild/accepted writes.

Next fullprimary already read333,4743,4753,761,5381,4199,5297,4108,2417,664,575,868,4430,3887. No rules yet for these latest candidates. DEDR575 text starts with Kannada forms and id.; inspect preceding574 or source page for missing gloss before claiming complete evidence. Candidate lists usecognacy-discovery filtered unexamined. Deadline17:50UTC unchanged.

## Checkpoint041

6,161records/6,264rows,450parents,751holds,30,392unexamined,504batches. Shared dry-run/manifest agreement passed,zero second application. Added14rain/road/negative/speech proposals,3holds. No rebuild or accepted writes.

Primary text issue investigated: d575 sand article has no defining gloss, only id., even in original cached page56 HTML (saved dedr-page56-cached.txt); preceding574 is gooseberry, not sand. Live DSAL opens failed twice. Do not infer meaning from id. alone. Need independent primary lexical support or hold Kolami usuka. OriginalcachedHTML is4.1MBpickle list514strings, indexpage-1; safe read-only inspection withoutrunningparser.

## Checkpoint042

6,169records/6,272rows,455parents,758holds,30,377unexamined,505batches. Shared dry-run/manifest agreement passed,zero second application. Added8sit/sleep/see/small proposals and7holds. Corrected Kannada small4295→2495,Kurukh see786→903,Tamil sleep3276→3376(a). Full primary checked. No earlier pending575sandproposals existed; single Kolamiusuka nowheldforincompleteprimary. No rebuild/accepted writes.68numericcognacyleadsremain. Nextstar/stone/sun/swim/tail/demonstratives/tongue group.

## Checkpoint043

6,177records/6,280rows,456parents,763holds,30,364unexamined,506batches. Shared dry-run/manifest agreement passed,zero second application. Added8star/stone/swim/tail proposals,5holds. New parent mismatch5285headedmesi-valwhetstonepreventsKuwivalistone. No rebuild/accepted writes.

Next fullprimary alreadyread3684,3064,5470,4711,871,3151,5503,5159,3647,4635,4102,2608 and correspondingparentmetadata checked (except4635notyet). No rules yet for these candidates. Clearcorrected Kuwieyuwater761→5159,Malayalamnāmwe3674→3647. Kuruxwalkēkʰna871direct; Maltohēkponeedsmorphology. Demonstrative sourceannotationsquoteDED1 but candidate-numberextractor picked embeddedcrossrefs; inspect1 forKodaguadə/Parjiā.

## Checkpoints044–045

044passed6,189records/6,292rows after12proposals,4holds. Then parentaudit flaggedd1:*aṯṯ- label versus broadprimarya- paradigm.19earlierproposals movedtoholds plus2newKodagu/Parji demonstrativeholds. This is metadata review,notrejectionofcognacy. Fullarticle1/764read,parentmetadata inspected.045passed6,170records/6,273rows,459parents,788holds,30,346unexamined,508batches;secondapplicationzero,manifestagreementpassed. No rebuild/acceptedwrites.

Next fullprimary justread2164,3613,5320,4792,2655,1693,2983,4203,833. No rulesyet: Gondi korakabitepossible2164/Gondikork; tongraknee2983direct; narkaḍnightcorrect3613not3584. DEDR4203explicitlyquestionsGondilousecomparison;833ratnotBadagalouse;2655hairmayincludeIAloans. Gondiartāddry5320notconvincing; maṭṭiearth4792wrongentry. Continue remaining37numericcognacyleads,includingcompoundsanddemonstratives.

## Checkpoint046

6,175records/6,278rows,461parents,797holds,30,332unexamined,510batches. Shared dry-run/manifest agreement passed,zero second application. Added5Gondibite/knee/nightandBrahui/Kuwiē-demonstrativeproposals,9holds. No rebuild/accepted writes.23numericcognacyleadsremain. Fullprimaryjustread4616,1524,4459,1121,1195,60,820. No newrulesforthese: Malto1524explicitquestion; Todapokkumlieprobablyuntruthnotrecline; round1121/1195wrong; seed60wrong; sleep820wrong. Kota ganmagu likelycomponents1173male+4616child; Telugu magavāḍu4616explicit butpotentiallycompoundmaga+vāḍu(d1parentcurrentlyheld). Continuecorrecteddonor/compoundresearch.

## Checkpoint047

6,178records/6,282rows,462parents,805holds,30,321unexamined,510batches. Shared dry-run/manifest agreement passed,zero second application.104compounds now:Kotaganmaguman1173male+4616child,comparativeanalysisqualified. CorrectedKannadakutʰigeneck→1718,Gondiguṇḍralround→1695;1121/1195werevocabularylocators.8newholds including false rat/louse/falsehood andwrongnumber routes. No rebuild/acceptedwrites.

Remaining12numericcognacyleads. Full4411readagain:Kolamiperetmanysupportsperemanypossible,Go peddabigneedsroute(noexplicitplainadjectivalpeddaGo). Gondi nāliketongueannotation3663wrong(truth);usehomepagecorrectedsearch. Remaining-correction logs42rows. nidrsearchDEDRempty(needsIA/sourcelexemeref),maṭṭiearthsearchnoappropriateDEDRmatch. Full1718/1695verified;primary3663readunrelated.

## Checkpoint048

6,181records/6,285rows,464parents,814holds,30,309unexamined,511batches. Shared dry-run/manifest agreement passed,zero second application.104compounds/2derived/6borrowed unchanged. All numeric candidate_articles leads in cognacy-discovery are now accounted for (zero remain unexamined); this does NOT mean all source annotations or survey records are examined. AddedKolamiperemany andBetta pərimoon/pə̄ḷsay comparativeproposals,9holds. CorrectedGondinālike3663→3633butheldtransmission. No rebuild/acceptedwrites.

Next broader unexamined scope: near152/far167records. Existing additional-discovery-results.json includes actualhomepagenear/far searches (reuse). Nearforms includeIrulapakka12,Muthuvankiṭṭa8,Mannankiṭṭa6,Kuruxheddevariants,Gondihere/kacum/dagire,Parjilegge,Ka hattara/voṭṭurā,Kadaraḍuttɨ. FarGondilak/lakh21+variants,lang/lvng;Kuruxgetcha;Parjikōmāḍ;manyIA-lookingdūra/tūrarequireimmediatedonorevidence. Top otherglossesman197,twenty164,year158,needle156,white155,woman154,sun150,roof139,potato138,ant137. Deadline17:50UTC unchanged.

## Checkpoint049

6,208records/6,312rows,467parents,824holds,30,272unexamined,514batches. Shared dry-run/manifest agreement passed,zero second application. Added27nearcomparativeforms:17kiṭṭa/kiṭṭe1538,6aḍuttu/aḍukka79,2arike222,2Badagasāre2460. Fullprimary1538/79/222/2807/2460/1293andparentsread.10Kuruxfarformsheldbecause2807explicitlyquestionscomparison. DEDR1293alsoquestionsGondikarumnear(notnewlyassigned). No rebuild/acceptedwrites.

Additionalactualhomepagesearchinprogress near-far-additional requests/forms pakka,hedd,hattira,lak; shellsession67460 lastconfirmedrunning. Waitsamehandle orinspectresultbeforecontinuing,donotrestartfromtimeout. Broadlakmayreach1000rowcap.

Session67460 subsequentlycompletedexit0,4queries471rows; near-far-additional-results.json ready. No processremainsrunning.

## Continuation at 16:21 UTC (deadline still17:50 UTC)

Checkpoint111 passed, exact helper/manifest agreement verified, zero second-application changes. Current rules through111 all rendered and validated. No jobs pending. Latest protected hashes110 unchanged. New `kinship-remnants-inventory.json` and `kinship-remnants-results.json` ready (6 actual homepage queries,3179rows; broad man/woman queries may be truncated). No112 rules authored. Investigate remaining child/simple kin terms and male/female+child compounds; retain IA-contact and parent-metadata holds. Recent complete primary inspections:1120 gram,1578 tuber,705 onion,1462 pungent seasoning,4867 pepper/chilli,4089 divide/clove,664 round,5535 root,1459 fruit. Vegetable111 proposals110 include59 ordered compounds;35 white-garlic responses held under wrong5496butter parent. No rebuild, accepted writes, node creation, or publication.

## Checkpoints112–113 — 16:40UTC

11,042 proposed records /11,804 rows,615parents,2,044holds,24,218unexamined,857batches. Check113 retry passed after concurrent input mutation; exact helper/manifest agreement and zero second application verified. Added child and kinship families,5 child compounds, and explicit Konda mogavaṇḍru borrowing from existing Telugu dictionary donor. No accepted writes or rebuild. Printed Tamil11 pp112/115 visually verified for contracted person compounds;114 rules now drafted, not yet validated.

## Checkpoint114 validated;115 in progress —16:39UTC

114 passed at11,073records/11,866rows,615parents,2,044holds,24,187unexamined,862batches.31person compounds supported by visually checked Tamil11 pp112/115.115 now rendered at11,102records/11,895rows,617parents,2,094holds,24,108unexamined,866batches; dry-run running. Added29night/gold proposals and50holds for questioned taṅkam origin, morning/night semantic mismatches and unsuffixed Gondi nela contact ambiguity. Earlier16:40timestamp was premature; current actual clock16:39UTC.

## Checkpoint120 —16:55UTC (deadline17:50UTC)

11,743 proposed records /12,543 assignment rows,624parents,2,161holds,23,400unexamined,39languages,902batches. Shared check120 passed, exact manifest/helper agreement and zero second application verified.119 first check hit concurrent mutation and is validated together with120. Protected hashes120 unchanged. No rebuild or accepted writes.

114 added31contracted male/female+child compounds, verified in Sri Lanka Tamil11 pp112/115.115 added29night/gold proposals and50holds.116 added90millet/fat/heart proposals and4holds.117 added63contact/source-parent holds only; assignment agreement remained exact to116.118 added17verb proposals(7compounds), validated.119 added300paired-prompt verb records;120 added234speech/sleep/sit/lie/fly records. Paired prompts retain individual bare/finite source forms; no absent clause is reconstructed. DEDR1524 existing *kōḍ-(?) parent uncertainty is explicit in all new evidence; questioned Malto is not used.

Next independent research: paired-prompt transparent movement/modal compounds. No121rules authored. Full primary4681maṇṭu move swiftly/Malayalammaṇṭuka run,5528vēṇṭu want/negativevēṇṭa and Kannada bēḍa must not,1882kūṭu be possible,2426cādie,2132kolkill,2151koḷreflexiveauxiliary,4572pōgo,4020paṟafly,1041ōṭurun,3582naṭawalk,2017(a)kēḷhear,1524kiṭalie have all been read and actual parent metadata checked. Candidate sources: mand-movement-results.json, negative-aux-results.json, paired-posture-results.json. Clear remaining fly+go: Muthuvan pari po/paripokaṇay, Kadar parɨpɔ, Irula parandɨpɔ, MalaPulaya parannapo; run+go Muthuvanōḍipoyintɛ; walk+go MalaPulayanaḍandupo; walk+reflexive Irula naḍandɨkɔ/nadanduko; listen+reflexiveIrula kēṭukɔ; lie+reflexiveIrula kaḍanduko. Mannanmaṇḍ movement/go and Uralimaṇḍirun compare4681; compounds fly/run/walk+maṇḍ also present. Negative cāka/kolla+vēṇḍa/bēḍa and cāka+kūḍātu can be ordered components, preserving grammatical negation; longer ambiguous -ṭṭattu/-acci perfects remain excluded. No jobs pending.


## Checkpoint 122 — article-scope clarification and unlimited continuation

User explicitly says DEDR IDs are whole articles; leave headings unchanged. User also updated the goal to unlimited time. ARTICLE-SCOPE-POLICY.md records this policy, superseding previous label-only holds and the original deadline.

382 exact records moved from held to pending proposals through article-scope-reviewed.json (345 checkpoint121 +37 checkpoint122). Totals: 12,125 proposed records, 12,925 assignment rows, 639 distinct parent IDs, 1,779 holds, 23,400 unexamined in the historical survey inventory; 931 owned batches across39 languages. Both dry runs passed; checkpoint122 manifest agreement and zero second-application changes verified. No accepted writes or rebuilds by this task.

IMPORTANT: protected-input-check-122.json detects changed compiled forms, edges, identities and browser DB versus the original baseline. The files have Sept21 afternoon local mtimes; this turn did not mutate them. Do not claim the original inputs remain unchanged. Reconcile the original survey inventory and cached discovery against the newer shared data before broad expansion; preserve other tasks' changes. prepare.py already reconciles current accepted sidecars; it may not refresh the original record inventory.

Next research: full primary d864 (unqueried southern ant family) and d410 (today subsection b) were just re-read. Current held candidates exactly equal to these IDs show only one each: Tamil erumpu f_j57t76wijg3ne and Kannada īndu f_hv66dnb5iz7um. Earlier quick-adjudication larger counts may be encoded differently or superseded; inspect exact held structures, do not blindly release. Uncertain d4169 ant/d3373 Gondi meat/d4794 broom remain held. Other head-only holds still need review. Pending movement/modal compound research is described in earlier checkpoints and research logs.


## Checkpoint123 — source reconciliation and expanded user objective

New objective supersedes surveys-only scope: etymologise remaining Dravidian sources, prioritizing smaller and underetymologised sources for diversity. Unlimited time; no rebuild. Preserve existing survey research and broaden to dictionary and other attestation sources. Do not blindly expand old family allowlists over new source records.

All37,304 historical survey records match current compiled forms exactly, and none has a compiled rank1 edge (inventory-reconciliation-123.json, compiled-survey-links-123.json). New data additions account for changed input hashes; source IDs and survey values remain usable. Current sidecars still reconciled by prepare.

Checkpoint123:12,286 pending records /13,086 assignment rows /639 parents /1,618 holds /23,400 unexamined historical survey records,942batches,39languages. Shared check passed with exact manifest agreement and zero second application.

164 additional pending proposals:73 southern ant DEDR864 (unqueried; distinct from questioned Gondi4169),91todayDEDR410(b), per whole-article user clarification. Three Kurux xocol bone proposals withdrawn: cached duplicate-excluded website1288/page197 is NOT canonical sugarcane1288/page119. DEDR1296 crossreference and Kobayashi Malto basic vocabulary (https://www.gengo.l.u-tokyo.ac.jp/masatok/BT_2006.pdf) point to2188, absent from current compiled nodes. Keep held; no node creation authorized. prepare.py and inspect-primary.py now exclude duplicate-excluded rows. Onlyd1288 among used primary records had this issue. Search bone-reconciliation-results.json confirms current discovery itself stillmislinks; don'tuse as proof.

New all-source inventory:all-source-priorities-123.json and all-source-unexamined-123.json, generated by inventory-all-sources.py.57,779 unexamined Dravidian records across103 source/language groups; counts exclude current accepted/compiledrank1 and this pass's pending/heldIDs. Ranking is by source/language slice so aggregate full source sizes before final ordering. Includes new zoller2023, das1987yerava, lindgren2023dravidian(Pattapu101,Belari108) absent from old surveyinventory.

NEXT: Das1987Yerava90responses(Ravula48,Paniya42), allunlinked/unreviewed; no research proposals authored for them yet. Bibliographydata/cldf/sources.bib around5761. Local primary evidence data/data/other/forms/raw_data/das_yerava_1987/README.md, reviewed.tsv, evidence/pdf-*; fullcomparativewordlist pp65–66. Preservepopularspellings, no inventedphonemics. Inventoryall-source-unexaminedcontains exactrecords. Use existing homepage discovery and fullDEDR verification, write separate source-scoped pending pass (or deliberateexactrecord extension), then canonicaldryrun. This is etymologyreview of installedsource, notnewingestion; do not rebuild.


Active next source pass: ../das-yerava-20260922/STATE.md. Checkpoint002 verified:38pending,7holds,45unexamined. Continue its remaining45before selecting another small source. Historicalsurveyproposals12,286 remain unchanged. Goalunlimited and expandedallDravidiansources.

LatestDasYerava004:58pending,10holds,22unexamined; sharedcheckverified. See siblingdas-yerava-20260922/STATE.md fornext22. Main12,286surveyproposalsunchanged.

LatestDasYerava005+holdrevision006:61pendingrecords64rows,13holds,16unexamined; verified005. Continue siblingSTATE.md.

Dasinitialassessment007covers90:62pending65rows,28holds,0unexamined,checkpassed. Continue newPattapusmallsourcepass inventory; seey eravaSTATE(nextsourcepaths). Broadergoalnotcomplete.

Active Pattapu pass: ../pattapu-20260922/STATE.md. Checkpoint002 passed:23pending records,78unexamined. Continue Pattapu core vocabulary and pronouns.
