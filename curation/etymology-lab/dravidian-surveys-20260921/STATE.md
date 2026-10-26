# Dravidian survey etymology pass — 2026-09-21

User scope: “etymologise remaining Dravidian surveys. do not rebuild. work autonomously and do what you can, you have 8 hours, i will review once your time is done.”

In progress. New research is being collected for deferred user review. No rebuild, importer, publication, or donor-node creation is authorized by this pass.

Initial inventory: 37,304 survey records without rank-1 compiled links or accepted rank-1 sidecar assignments. All sidecars were read with `etymology_assignments.read_assignments()`. The broader inventory includes dictionaries, which are excluded from this survey pass. Survey scope includes SIL/local elicitation surveys, census wordlists, LSI and the Kolipakam comparative basic-vocabulary collection. Existing Gondi decisions are preserved.

Candidate discovery invokes the actual `fetchLemmaList` implementation in `jambu-static/src/lib/query.ts` against the existing read-only `.dbwork/jambu.db`; request/result logs are kept here. Search scores or normalized matches are not evidence. Shortlisted dictionary articles are checked in the cached primary DEDR article audit.

Artifacts:
- `inventory-counts.json`: complete language/source inventory, linked and unlinked.
- `survey-inventory.json`: persistent IDs and exact source records in scope.
- `discovery-requests.json`, `discovery-results.json`: primary entry searches.
- `reflex-discovery-requests.json`, `reflex-discovery-results.json`: attestation searches.

## Checkpoint at 2026-09-21 03:20 PDT (approximate)

3,529 proposals under 115 distinct parent IDs; 292 explicit holds; 33,483 unexamined. The shared dry-run validator passed; no accepted sidecars were written. See `REVIEW.md` and `HANDOFF.md`. Continuous work now uses the persistent goal. The redundant heartbeat `dravidian-survey-etymology-research` is paused; research deadline remains 17:50 UTC.

## Checkpoint 003 in progress

4,492 proposals, 145 parent IDs, 39 languages, 393 holds, 32,419 unexamined; 307 current review batches. Parent metadata auditing moved 94 previously proposed records to explicit holds. Current-sidecar reconciliation and superseded-batch handling are automatic in the research scripts. No saved assignments or rebuild.

## Checkpoint 004

4,588 proposals, 162 parents, 395 holds, 32,321 unexamined. Shared dry-run passed. Additional semantic search discovery yielded 83 new candidates pending primary checking; none automatically approved. No rebuild or accepted writes.

Current post-validation research: 4,629 proposals, 169 parents, 399 holds, 32,276 unexamined; 314 current review batches. The last 41 proposals await the next batched dry run.

## Checkpoint 005

4,834 proposals, 189 parents, 401 holds, 32,069 unexamined, 332 batches. Shared dry-run passed with zero second-application changes. No rebuild or accepted writes.

## Checkpoint 006

Temporary-graph validation passed for 5,085 after retrying concurrent input changes. Current post-check research: 5,100 proposals, 215 parents, 415 holds, 31,789 unexamined, 353 batches. Latest 15 proposals await next validation. No rebuild or accepted writes.

## Checkpoint 007

5,131 records / 5,139 assignment rows, 224 parents, 411 holds, 31,762 unexamined, 356 batches. Eight compounds have ordered components. Shared dry-run and manifest/helper row agreement passed. No rebuild or accepted writes.

## Checkpoint 008

5,307 records / 5,344 rows, 233 parents, 411 holds, 31,586 unexamined, 373 batches. 37 compounds. Shared dry-run and manifest/helper agreement passed. No rebuild or accepted writes.

## Checkpoint 010 — 11:07 UTC

5,277 records / 5,314 rows, 270 parents, 546 holds, 31,481 unexamined, 378 batches. Shared dry-run and manifest/helper agreement passed; second temporary application changed zero. Numeral d990 audit moved non-okk formations to holds (the stored parent is *okk-); source-specific Muduga evidence added. No rebuild or accepted writes.

## Checkpoint 013 — 11:14 UTC

5,325 records / 5,362 rows, 302 parents (including two immediate attested donors), 561 holds, 31,418 unexamined, 388 batches. Shared dry-run and manifest/helper agreement passed; second temporary application changed zero. Two explicit borrowed proposals now use existing Telugu donors for Konda cloth and Gondi pestle. No rebuild or accepted writes.

## Checkpoint 015 — 11:19 UTC

Dry-run passed for5,353 records/5,390rows; second application zero. Current research is5,362records/5,399rows,338parents,571holds,31,371unexamined,395batches. Latest9proposals await next batched graph validation; current manifest/helper agreement passed. Corrected false run-as-ink match to running verb DEDR1041, rejected nose-ornament substring match. No rebuild or accepted writes.

## Checkpoint 018 — 11:26 UTC

5,417records/5,454rows,391parents,586holds,31,301unexamined,407batches. Shared dry-run and manifest agreement passed, second applicationzero. All current direct-candidate records now either proposed or held. Three explicit Telugu donor proposals. Next research expands beyond exact-fold matches using existing homepage search results. No rebuild or accepted writes.

## Checkpoint 019 — 11:30 UTC

5,538records/5,575rows,393parents,586holds,31,180unexamined,412batches. Added121broad comparative proposals in firewood,spider,lightning afterfullprimaryandparentchecks. Shared dry-run andmanifestagreementpassed, secondapplyzero. No rebuild oracceptedwrites.

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

## Checkpoint 050 — 12:56 UTC

6,235 records / 6,339 rows, 468 parents, 857 holds, 30,212 unexamined. Shared dry-run and manifest/helper agreement passed; second application zero. Added simple roof/door proposals and held questioned broom comparisons. No rebuild or accepted writes.

## Checkpoint 051 — 13:03 UTC

6,247 records / 6,353 rows, 468 parents, 858 holds, 30,199 unexamined, 516 batches. Shared dry-run and manifest agreement passed; second application zero. Added two ordered Muduga roof compounds and ten simple roof/year forms. 106 compounds total. No rebuild or accepted writes.

Post-checkpoint needle research: DBIA171 full cached OCR inspected after homepage discovery. Selected needle forms are held for immediate-donor evidence and a malformed DBIA comparison node (Parji cūcub erroneously used as IA head). No new graph proposals from these holds. Next checkpoint052.

Needle holds rendered: 86 new, giving944 holds and30,113 unexamined. Proposed graph unchanged at6,247 records/6,353 rows; checkpoint051 manifest/helper agreement rechecked successfully. No running jobs at this checkpoint. Continue broader unexamined vocabulary or source-local loan research until17:50UTC.

## Checkpoint 052 — 13:01 UTC (tool clock)

6,254 records / 6,367 rows,470 parents,979 holds,30,071 unexamined,519 batches. Added7 ordered potato compounds (round d664(a) + tuber d1578),113 total compounds; held35 Gondi ant forms because primary d4169 explicitly questions the comparison. Full articles d664,d1578,d4169 inspected; actual parents checked. Shared dry-run and manifest/helper agreement passed, second applicationzero. No rebuild or accepted writes.

Earlier051 handwritten13:03 timestamp was approximate and incorrect; tool clock now13:01UTC. Deadline17:50UTC remains unchanged. Next dry-run053. New discovery logs tuber-requests/results. Remaining likely useful areas: month tiŋgal variants, palm compounds uḷḷam-kai and aŋgai, rainbow bow compounds (analyse first member independently). Most month/year forms look IA and need immediate-donor evidence. No running jobs.

## Checkpoint 053

6,316 records / 6,481 rows,472 parents,980 holds,30,008 unexamined,522 batches. Added52 palm compounds (d7 akam + d2023 hand; d698 uḷḷam + d2023 hand) and10 month forms d3213. The d698 component deliberately matches its current *uḷḷ-am parent and does not lift the house-word hold. One Kannada ulḷāŋkāl palm response held for apparent foot constituent. 165 compounds total. Shared dry-run and manifest agreement passed, second applicationzero. No rebuild or accepted writes. Nextcheckpoint054.

Next research: rainbow. Full d2124 read from existing homepage rainbow discovery; Konda kuruveli and Kannada karuḍu candidates need parent check. New rainbow-requests/results discovers vil bow and koḍincil. Continue primary checks.

## Checkpoint 054

6,374 records /6,539 rows,474 parents,981 holds,29,949 unexamined,527 batches. Added7 rainbow proposals (d2124 Konda kuruveli/Kannada karuḍu; five simple bow→rainbow extensions d5422) and51 father proposals d156/d183/d3067 with nursery-family limitations explicit. Konda koler rainbow held for unexplained divergence. Shared dry-run and manifest agreement passed, second applicationzero. No rebuild or accepted writes. No running jobs. Nextcheckpoint055.

Next work: remaining man/woman forms and compounds. Full primary d399,d4395,d4198,d4616 just inspected. Simple āṇɨ and peṇṇɨ variants remain, plus āṇ/peṇ + piḷḷai or makkaḷ compounds; compare exact constituents before adding. Konda magavaṇḍu explicitly Telugu loan, existing donor required; Gadaba maginḍ sinḍ is a two-member expression (sinḍ son), not simple maga. Kota census pemug glossed man conflicts pe-mog woman and needs a hold.

## Checkpoint 055

6,420 records /6,605 rows,474 parents,982 holds,29,902 unexamined,530 batches. Added26 simple gender/person terms and20 ordered male/female + child/people compounds (185 compounds total). Kota census pemug man held against explicitly female pe-mog in primary. Shared dry-run and manifest agreement passed, second applicationzero. No rebuild or accepted writes. Nextcheckpoint056.

Next research livestock: many southern maḍɨ cow and kōyi/kośi/kojī chicken still unexamined; initial exact-gloss search missed generic cattle/fowl comparanda. livestock-requests.json prepared for actual homepage form searches. Buffalo d816 found through existing searches, full primary needed.

## Checkpoint 056

6,544 records /6,733 rows,477 parents,982 holds,29,778 unexamined,535 batches. Added120 simple livestock forms (two fowl families d2248/d2160, cow māṭu d4798, buffalo d816) and4 ordered buffalo+cattle compounds. 189 compounds total. d4798 full primary glosses Tamil/Malayalam ox and Kota beast including cattle/buffalo; survey cow semantic narrowing is explicitly qualified. Shared dry-run and manifest agreement passed, second applicationzero. No rebuild or accepted writes. No running jobs. Nextcheckpoint057.

Next research eggs: reused homepage egg results and read full d4939 muṭṭai, d5088 Gondi mēnj/pl.mēsk (ordinary plural), d1279 Parji kerba/Gondi garba egg. Parent labels not yet checked for this next group. Remaining variants include12 or so moṭṭu/mɔṭṭɨ/matte and Parji kɛrbā4, Gondi mesku/mɛ̄sk/mīnj3. Gondi garbham-type forms may require loan-route caution despite primary comparison.

## Checkpoint 057

6,590 records /6,779 rows,478 parents,985 holds,29,729 unexamined,539 batches. Added18 eggs (d4939,d5088,d1279) and28 simple tuṇi cloth forms d3307, preserving source spaces. Held3 Gondi garbham-type eggs for contact-route evidence. Ordinary Gondi egg plurals explicitly documented, not derived links. Shared dry-run and manifest agreement passed, second applicationzero. No rebuild or accepted writes. No running jobs. Nextcheckpoint058.

cloth-additional-requests/results now completed (17rows) for cīle,baṭṭe,araba. Inspect those linked candidates and primary source before adding; remaining many cloth forms may be IA contact.

## Checkpoint 058

6,593 records /6,782 rows,478 parents,1,020 holds,29,691 unexamined,539 batches. Added3 Kannada āraba/ārbā cloth proposals d318; held13 baṭṭe-family forms after fullDBIA318 and22 cīla/sīre-family forms after fullDEDR2629. DBIA318 explicitly derives the former ultimately from paṭṭa, not vastra; DEDR2629 discusses cīra/cela contamination. Immediate donors remain unresolved. Shared dry-run and manifest agreement passed, second applicationzero. No rebuild or accepted writes. No running jobs. Nextcheckpoint059.

Next promising area: bodypart variants. Remaining fingernail forms Kannada vuguru6/vugru1/uyɨrɨ1, AluKurumba vuguru, Irula ugarɨ2,Kadar ugarɨ,Muduga/Kurumba oyɨrɨ4; existing homepage nail results give d561. Full primary and current parent still need recheck. Kurux orox9 may also be in d561. Many naga/nakam forms need IA-route evidence. Lip/chin variants also available but require new semantic queries because earlier logs lack them.

## Checkpoint 059

6,626 records /6,815 rows,481 parents,1,023 holds,29,655 unexamined,543 batches. Added33 nail/lip/cheek proposals; held3 Kurux oṛox because d561 explicitly corrects Grignard’s retroflex-r spelling. Audited earlier pending Kurux d561 proposals: none has that rejected spelling (one orokʰ claw remains, plus5neworox). Primary d561,d3296,d1989,d1591,d606,d3596,d1156 fullyread. Shared dry-run and manifest agreement passed, second applicationzero. No rebuild or accepted writes. No running jobs. Nextcheckpoint060.

Potential useful focus:20 remaining OllariGadaba records (same source-specific Kolipakam set). Form/gloss list in unexamined-inventory: mure big,palgil breast,kaṣvanḍe claw,mēn fat,puneṭ full,cendi/talcenṭi hair,jibe heart,sun know,ṭunḍe leaf,kappe mountain,cālu mouth,kanḍ neck,nia new,rasta road,kure root,puṭ star,kindir tail,adə that,iruḍu two. These lack numeric source leads. Actual homepage form/gloss discovery should narrow; compare Gadaba versus OllariGadaba language IDs carefully. d1parent hold affectsadə; potentialIAformsrequireimmediate donor evidence.

## Checkpoint 060

6,630 records /6,820 rows,483 parents,1,024 holds,29,650 unexamined,544 batches. Added3 Ollari Gadaba simple proposals (kure root d1799,cendi hair d2002,iruḍu two d474),1 ordered talcenṭi head+hair d3103+d2002; all cross-variety comparisons explicit.190compounds total. Heldadə that under existingd1parentmetadataissue. Shared dry-run and manifest agreement passed, second applicationzero. No rebuild or accepted writes. No running jobs. Nextcheckpoint061.

Ollari discovery complete:20 form+gloss queries76rows;15 semantic Gadaba-filtered queries35rows. Remaining15 have no straightforward printed matches (mouth cālu vs sol possible but vowel/consonant comparison unresolved). Sourcepalgil breast has emptyComment,Loan=false notproof. Do not force broadGadaba identity: primaryS./S.2 maydifferfromOllari. More source-specific references needed forremaining. Other useful nextscopeKuwi27unexamined/Kolami87 or broaderhousehold/verbs. Deadline17:50UTCunchanged.

## Checkpoint 061

6,631 records/6,822 rows,484 parents,1,027 holds,29,646 unexamined,544 batches. Added Kuwi sipadaki breast as explicit ordered d2620+d2976 compound;191compounds total. Held Kuwi bhumi earth,patra leaf,bali sand for unresolvedimmediateIA donor (homepage leads only, notprimaryproof). All27Kuwi form+gloss queries completed97rows; most remaining forms lackdirectsupport. Fullprimaryd2620,d2976,d3735 inspected,currentparentschecked. Shared dry-run andmanifestagreementpassed,secondapplicationzero. No rebuild or acceptedwrites. No runningjobs.Nextcheckpoint062.

Promisingnext: southern compound numerals twenty/twelve. Unexaminedtwenty148 includes Muthuvan iruvadɨ5,Irula iru vatɨ5+iruvatɨ3+īruvaḍɨ3+īruvaḍu3,Mannanirupatɨ4/iruvadɨ2,etc; Kannadaippaṭṭuvariants. DEDR474fullread explicitlylistsiru-patu/iru-vatu,Kaippattu; need actualhomepage discoveryfortwenty andfulltenentry(currentparentcheck). Twelve148 hasmanypannaṇḍɨ/pandraṇḍɨ southernforms; d474panneṇḍuprint. Eleven130 facesd990parent-subsectionholdone. PreservehighernumeralcontactcautionGondi/Gadaba/Konda;donotmasslinkIAvis/bara/gyaravariants.

## Checkpoint 062

6,777 records/7,114 rows,484 parents,1,027 holds,29,500 unexamined,553 batches. Added146 ordered numeral compounds:70twenty=d474two+d3918ten;76twelve=d3918ten+d474two.337compounds total. Explicitly documented formations in bothprimaryentries; d3918current*paH-ten checked. Cross-lect variantsqualified, Gondi/Gadaba/Konda/contact-sensitive groups excluded. Actualhomepage numeral-compound/ten-compound discoverylogs saved. Shared dry-run andmanifestagreementpassed,secondapplicationzero. No rebuild or acceptedwrites.No runningjobs.Nextcheckpoint063.

Remainingnumeralsinclude11sixteen and11forty mostlycensusTamil/Badaga/Kannada/Irula/Kota/Maliyad/Paniya/Sholaga/Toda/Urali; Kuruxdoysoī sixteen likelymismatchandnāxhdoyforty compoundrequiresanalysis. Noother13-19/30-90 glosses in unexamined exceptthese. Nextreadactualfour/sixparentsviaexistingdiscovery,fullprimaryandd3918componentnotation. Irula pətənālɯ sixteen conflictsapparentfourcomponent; holdforalignment. Otherhighfrequencyremaining:elbow117,olderbrother118,turmeric113,night112,mud110,etc.

## Checkpoint 063 IN PROGRESS — validation pending

6,794 records/7,148 rows,484parents,1,029holds,29,481unexamined,555batches. Added17 sixteen/forty compounds (354total),heldIrula pətənālɯ sixteen(apparentfour) andKannadagətənāru sixteen(unexplainedinitialg). Fullprimaryd3655,d2485 read,parentchecked. TWO dry-runs terminated becauseinputs changed duringvalidation; logs preserved check-pass-063-concurrent-inputs.log and-2.log. Freshprepare/renderbetweenattemptskeptsamecounts. No bypass, noacceptedwrites,no rebuild. Lastfullypassed062. No runningjobs. Retry063aftermoreusefulresearch; do notmarkpassedwithoutsuccessfulnewrun.

Kuruxcensusnumberresearch:actualhomepageKurux(correct name;Kurukh returnedonlyprotoentry) searchesandrawaudit give sixsoi,tendoy,sixteendoysoi,fortynaxhdoy,twentyenddoy. Savedkurux-census-number-audit.json completeexactrecords. Earlier inference thatdoysoimightbeglossconflictwasunfounded; leaveunexaminedforcompoundanalysis. PotentialcomponentstoEXISTINGsame-languageattestedsoi/doy canavoidguessingdeeperIAorigins;resolveIDsandcitations. prepare.pycompoundrulecurrentlyassumesDEDRcomponents for citation; render.pyline31likewiseexpectsprimaryarticleforfirstcomponent. Need scopedgeneral-source support ifusingattestedcomponents (do notmodifysharedhelper). Existingborrow/derivedrulesalreadyuseevidence_parent. Nextworkmayimplementexplicitcitation/evidence_urloptionandfullgraphdryrun.

## Checkpoint 063 PASSED

6,794 records/7,148 rows,484parents,1,031holds,29,479unexamined,555batches. Fresh dry-run andmanifestagreementpassed,secondapplicationzero. Noacceptedwrites/rebuild. All17sixteen/forty proposals nowvalidated.

Kuruxdoysoī/nāxhdoy source compositions verified from samecensusstandalonewords andrawaudit. Attemptedattestedcomponentlinks were REJECTED by sharedhelper because parentdoy f_7urh6hlfrgiju has Status=unlinked (also soī f_yym77isj7lxbs unresolved). Bothanalyses movedtoexplicitresearchholds. Preserve kurux-rejected-component-proposals.json andcheck-pass-063-unlinked-parent.log. Do not bypass sharedparentguard. GenericnonDEDRcomponentcitation/render support was tried thenfullyreverted becauseunused. Currentresearchscriptsretainpriorbehavior. Parentnāxh f_pnhr2rhnluc4u fouralsoresolvedasexistingrecord. No newnodes. No runningjobs.Nextcheckpoint064.

Continuebroaderremainingvocab:turmeric113 includesmany southernmañjalvariants andGondikamka~40. Actualhomepageglossturmericsearchshouldexist; inspectwholeprimaryandmetadata beforeproposing. Next deadline17:50UTC unchanged (lastclock13:31UTC,thenworkseveralminutes).

## Checkpoint 064

6,845 records/7,199 rows,484parents,1,039holds,29,420unexamined,557batches. Added51 southernmañjal/mañjaturmericforms d4635. Held7Gadabaparep/pareb expresslyquestionedind3821and1Kondapasiŋbecauseexistingparentgreen-gramlabelmismatch. Fullprimaryd4635,d3821,d220inspected. Gondikamka~40notmapped:actualhomeformsearchfoundonlysurveyrecords, nounrelatedkaṇka/heronlink. Logs turmeric-requests/results,turmeric-form-requests/results. Shared dry-run andmanifestagreementpassed,secondapplicationzero. Noacceptedwrites/rebuild. No runningjobs.Nextcheckpoint065.

Nextqueriesalreadycompleted:elbow-mud-night-requests/results3queries508rows. UnexaminedcommonelbowformsMannankaimoḷi/kaimuḷi,Muthuvanmū̃kay,Irulamonaŋgai/monakkai,Malasarmuḷamkai; mudMannanceḷi14,Irulasērɨ7,Muthuvantoḷi5,Kadarcaḷi2; nightMannanravɨ14,Muthuvanravɨ2,Uraliirā2,Irulairuṭṭɨ/riṭṭɨ. Inspectfullprimariesandparentlabels before adding. IA-lookingcikla/rātriformsneedroutechecks. Deadline17:50UTCunchanged.

## Checkpoint065

6,967 proposed records,7,361 rows,486 parents,1,054 holds,29,283 unexamined;567 owned batches. Added82 simple mud/night/elbow matches and40 ordered elbow compounds;15 parent-scope holds. Shared dry-run and manifest agreement passed,zero second-application changes. No accepted writes or rebuild.

## Checkpoints066–067 pending validation

066 added134 thorn/tail/son proposals and74 tail-parent/route holds. Two shared dry-runs rejected concurrent input changes; logs preserved. 067 adds181 mother/daughter/elder-sibling proposals. Current7,282 records,7,676 rows,487 parents,1,128 holds,28,894 unexamined;587 batches. Checkpoint065 is last passed until067 completes. Protected forms/edges/identity/browser DB hashes all unchanged in protected-input-check-066.json.

Checkpoint067 PASSED shared dry-run and exact manifest/helper agreement;zero second-application changes. Both066 input-race rejections are superseded by this successful fresh validation.

## Checkpoint068 pending

114 dust/below/morning proposals,27 explicit dust-loan/route holds. Current7,396 records,7,790 rows,491 parents,1,155 holds,28,753 unexamined;596 batches. Fullarticles3283 and4316 distinguish loan-contaminated branches; simple powder/dust and below forms selected separately. Checkpoint067 passed. Next motion discovery is running; no069 rules yet.

Checkpoint068 PASSED shared dry-run and exact manifest/helper agreement;zero second-application changes. Motion homepage discovery completed5queries812rows in motion-results.json; next action inspect relevant fullarticles and author069. No running jobs. Deadline17:50UTC remains unchanged; goal active.

## Checkpoint069 pending

Added53 run/walk/sit/stand proposals after fullprimary articles1041,3582,1835,3675,4850(a),2950,3151. Gondi tāk-s causatives and phrases excluded. Current7,449 records,7,843 rows,493 parents,1,155 holds,28,700 unexamined;602 batches. Next monkey/fruit/day discovery underway.

Checkpoint069 PASSED shared dry-run and exact manifest/helper agreement;zero second-application changes. No accepted writes/rebuild. Monkey/fruit/day homepage discovery completed3queries1824rows.

## Checkpoint070

Added248 monkey/fruit/day/tomorrow proposals after fullprimary inspection. Current7,697 records,8,091 rows,496 parents,1,155 holds,28,452 unexamined;617 batches. Shared dry-run and exact manifest/helper agreement PASSED,zero second-application changes. No accepted writes/rebuild. Next071rain/smoke/meat discovery started.

At14:01UTC:070 authoritative and passed. Rain/smoke/meat homepage discovery complete3queries1268rows in rain-smoke-meat-results.json;071not yet authored. Inspect full shortlisted articles and exact parent metadata next. Also hold Maliyad rain form consisting only of combining tilde after confirming exact record. No running jobs. Goal deadline17:50UTC; continue research.

## Checkpoint071

Added154 rain/smoke/meat proposals and38 holds, including questioned Gondi/Parji comparisons, meat-loan route and7 standalone nasalization marks in Maliyad. Current7,851 records,8,245 rows,498 parents,1,193 holds,28,260 unexamined;627 batches. Shared dry-run and exact manifest/helper agreement PASSED,zero second-application changes. No accepted writes/rebuild.

## Checkpoint072 pending

Added157 path/rope/left/right proposals including9 ordered hand compounds,34holds for left-loan routes and2left/right mismatches. Current8,008 records,8,411 rows,501 parents,1,227 holds,28,069 unexamined;638 batches.071last validated until072completes. Next belly/hair/heart discovery underway.

Checkpoint072 PASSED shared dry-run and exact manifest/helper agreement;zero second-application changes.

## Checkpoint073 pending

Added172 belly/hair/heart records including42 ordered compounds,12holds(6competing vaṭṭi/poṭṭe belly candidates,5possible IA hair loans,1Kurux belly origin). Current8,180records,8,625rows,501parents,1,239holds,27,885unexamined;649batches. Consistency audit confirms no earlier pending reflexes remain under2244/3373/2655/1175. No accepted writes/rebuild.

Checkpoint073 PASSED shared dry-run and exact manifest/helper agreement;zero second-application changes. Authoritative8,180records/8,625rows,445compounds. No running jobs. Next checkpoint074: continue remaining basic vocabulary; potential groups bone/blood/skin or adjectival simple forms. Deadline17:50UTC remains active; currentabout14:12UTC.

## Checkpoint074 pending

Added55 bone/blood/skin proposals and60holds, mainly rakta/rattam immediate-donor questions. Read fullCDIAL10539 cachedprimary, saved inspected-cdial10539.json after actualhomepage discovery. Reconfirmed1288mislabel *kar-umpu sugarcane; holds also Toda pōx and cakkili-parent skin forms. Current8,235records/8,680rows,501parents,1,299holds,27,770unexamined;655batches. Size-adjective homepage queries running for075; no075 rulesyet.

Checkpoint 074 verified: 8,235 records, 8,680 rows; manifest agreement and zero second-application changes.

## Checkpoints 075–078, 2026-09-21 ~14:30 UTC
- 075 passed: +144 size-adjective records; 8,379 records / 8,824 rows. Families d5276 big, d3491 doḍḍa, d1646 small, d2594 short, d1851/d1839 short, d3738/d621 long. Full articles and exact allowlists preserved.
- 076 passed: +110 crop-name records, including 73 ordered compounds and one explicit Gondi arati borrowing from existing Telugu araṭi f_oo47a4ebasnoi (DEDR205). Totals 8,489 records / 9,007 rows, 512 parents.
- 077 passed: 100 explicit crop holds added (total 1,399), no new assignments. Full CDIAL2712 read for northern banana contact family; DBIA72 kattiri scissors homonym rejected for eggplant; white+onion garlic held for d5496 parent scope.
- 078 validation passed: +86 insect/flight records, including one eight+leg compound, and 5 holds. Current 8,575 records / 9,094 assignment rows, 516 parents, 1,404 holds, 27,325 unexamined; 679 owned batches /39 languages. d4353 insect parent is wrongly labelled *nakk- to lick; four eight+leg+insect spider compounds held. No earlier pending proposals use d4353. Toda spider d2562 explicitly questioned and held.
- protected-input-check-078.json confirms forms, edges, identities and browser DB unchanged. No accepted writes/rebuild.

Checkpoint078 verified manifest agreement, zero second-application changes. Next checkpoint079; posture/bite discovery initiated.

## Checkpoint079
Validation passed for 8,636 records / 9,155 rows / 518 parents; 1,404 holds / 27,264 unexamined; 684 owned batches. Added 61 sleep/lie-down/bite proposals, with explicit full-article citations. Opaque Gondi minmudu f_47z72qpysroqg was excluded after inspection and remains unexamined; its initial tentative candidate was removed before prepare/render. Full pronominal clauses and uncertain causatives remain untouched. Next prospective checkpoint080; motion-tail homepage requests narum/narm/bettu/kōsi/taro launched for discovery.

Checkpoint079 verified: exact manifest agreement and zero second-application changes. Current exception totals: 519 compound records, 7 borrowings across 6 immediate donors, 2 derived records. All still pending review.

## Checkpoints080–083, 2026-09-21 ~14:46 UTC
- 080 passed: +119 door/sky/cut records, +59 holds; 8,755 records /9,274 rows. Door d5354 variants50, Gadaba door5277, moon3213/3754/2287, stars2646/4876. White/silver stars and white-moon compounds held for d5496 parent mismatch. Gondi/Gadaba/Maliyad talupu door forms held for immediate-route evidence. Urali kagagɨ f_kqgrqdvb4wure removed from provisional d1187 comparison and held for source/consonant analysis.
- 081 passed: +47 broom/roof records, including3 cīva-keṭṭu sweep/broom + bundle compounds2599+1147. Totals8,802/9,324,525parents. Full DEDR2599,4415,4794,2375,1042,4796,1904,3922,5532 read; source roof gloss retained where elicited form is the thatch verb.
- 082 passed: +127 colour/new/old records,8,929/9,451,525parents. Full1278/1395 black,1931red,3999old,4275new reread. Ordinary person/thing forms included with explicit evidence; Kannada c-initial red contact forms remain unassigned.
- 083 passed: +59 temperature records and -1 overlapping Kannada kare black proposal =>8,987records/9,509rows,529parents;1,475holds/26,842unexamined,709ownedbatches. kare held because1278and1395both support differently rhotic blackness stems. 9taṇ-cold forms held for d3045parentwronglabel*caḷ-,2Toda cold forms held for source/contact reading. Temperature proposals5517,2654,1458,588,1834,404,1568,1601,2168,2056.
- All work remains pending review, no accepted writes/rebuild. Next checkpoint084. Checkpoint083 verified exact manifest agreement and zero second-application changes.

## Checkpoints084–085

084 passed after a concurrent-input retry (first failure log preserved):9,071 records/9,593 rows,536parents,1,506holds. Added84 dry/wet proposals,31holds for unresolved wet parent and dry-gloss nan- responses. 085 passed:9,135records/9,657rows,540parents,1,534holds,26,635unexamined,724batches. Added64 few/many/heavy proposals; 28 small-quantity forms held for demonstrative versus whole-stem parent analysis. Full primary d410/d557 inspected. Sanskrit influence in d1404 heaviness explicitly retained. Both shared dry-runs and manifest agreement passed, second application zero. No rebuild or accepted writes. Next086 people/kinship; deadline17:50UTC.

## Checkpoint086

Shared dry-run and manifest agreement passed:9,390records/9,912rows,541parents,1,538holds,26,376unexamined,735batches;second application zero. Added255 people/kin proposals after full primary399/1173/4395/4616/4791/4944/364/400/4198/1646/4756/4954/2030/2813/3939/4259/1873. Four tentative whole-stem proposals were removed before validation:gaṇḍāḷu-type man compounds and contracted Gondi per girl; explicitholds. No rebuild/acceptedwrites.

Post086:29 clear ordered people/child compounds added (Gadaba male+son, southern male/female+child). Current9,419records/9,970rows,541parents,1,538holds,26,347unexamined,738batches;checkpoint087 dry-run running. Full2594/2495 reread. Search for contraction evidence retrieved Madras Lexicon peṇ-piḷḷai and āṇ-piḷḷai (latter says pronounced āmpḷe); only peṇ+piḷḷai used for clear compounds so far. More contracted and three-part responses remain unexamined. No new sources/nodes created.

## Checkpoints087–088

087 passed9,419records/9,970rows;088 passed9,429records/9,986rows,542parents,1,587holds,26,288unexamined,739batches. Both shared dry-runs, manifest agreement and zero second-application checks passed.088 added4simple kin forms and6female+person compounds;49holds concern Gondi tula/tudi boy/girl/child possible IA contact, Muthuvan keṭṭiyōn/keṭṭiyōḷ nominal derivations and Kadar maṇavan husband morphology. Actual kin-additional discovery complete (4queries500rows), primary1147/4667/1655 checked. No rebuild or accepted writes.

Post088 sibling work:127proposals,12holds;current9,556records/10,113rows,544parents,1,599holds,26,149unexamined,744batches. Check089 running. Sibling discovery complete4queries393rows;full3085/3015/2783/3563/196/513/23/273 checked, parents verified. Correct sibling article3085 (tampi), not3485 enclosure: older handoff shorthand about a3485parent mismatch should NOT block proper3085. No current rules reference3485. Holds preserve older-brother responses resembling younger-brother stems and possible Gondi/Te transmission.

Next090:temporal-remaining-results.json (6queries425rows) complete. Yesterday3758 looks promising for many southern forms and directly documents Irula nagaṯṯu/næṯṯu;989 Parji ori/Gadaba orgun. Today410 is proximal pronoun article: must resolve whole temporal form/components rather than indiscriminately attach fulltodayword to pronoun (cf quantity holds).2920nēnḍ/nēnṟu; month3754/3213; year5153/2203; morning3805/4305/5554. No temporal rules written yet. Mainremaininginventory shown in terminal, regenerate bygloss+fold asneeded. Deadline17:50UTC remains.

089 completed and verified at15:05UTC:9,556records/10,113rows,544parents,1,599holds,26,149unexamined,744batches. Shared dry-run/manifest agreement passed; second applicationzero. No running jobs remain. Continue temporal research090 until17:50UTC. No rebuild or accepted writes.

## Checkpoint090

Shared dry-run and manifest/helper agreement passed for9,662records/10,219rows,548parents,1,599holds,26,043unexamined,754batches. Second applicationzero. Added106 temporal proposals:62yesterday3758/989,7today2920,8month3213/3754,3year5153/2203,26morning/noon3805/4305/5554. Fullprimary3758/989/2920/3213/3754/5153/2203/3805/4305/5554/656 read, exactparentsverified. No rebuild or acceptedwrites. No running jobs.

Next091:temporal-extra-results.json complete(3queries921rows dawn/night/time); full410 read again: today is separate(b), parent*iH proximal(a), no explicit constituent etymology forwhole todayword in article. Do not casually linkallinnu/indu/innatopronoun; exactsubsection/wholeword analysisneeded akinquantity410(d)holds. Full5475,3613,2552 read.5475dawnviṭi fitsPaliyaviḍiñjiric potentially; Muthuvanviyāṯu morning could5475vs5554,notassigned.3613narka primarynight versus Gondinarki/narkommorning needssemantic/source-localchecking.2552rāvu night withIrularāvepodu night; morningrāvē/rāvilē likelylocativetimeuse but requireprimarymorningcorroboration. Webfound Nair Malayalam-English dictionary p431/PDFpage446 hasrāvile morningandrāvnight: https://www.wisdomlib.org/history/compilation/dictionaries-of-indian-languages-kosha/d/doc1622340.html (OCRnotyetverifiedagainstscan). Do not treatblogs/Wiktionaryasprimary. Nothingfromthisextra groupwrittenyet.

## Research091 / Checkpoint092

091 was holds-only after withdrawing tentative Paliya viḍiñjiric single-root proposal for possible auxiliary -iric.110newholds:today410(b) vs bareproximalparent, Muthuvanviyaṯudawn5475vs5554, Gondi nark-morningvsnight3613, Paliyacompletive. No091graphcheckwasneeded before the batched092. AttempttofetchNairprimarymorningpagefailed; do notclaimscanverified.

092 passed9,864records/10,421rows,549parents,1,709holds,25,731unexamined,769batches;shared dry-run,manifestagreement,zero secondapply. Added202eat/drink ordinary stem/inflection proposals after full600/1654/3174/3263/5127 read andparentschecked. Brahui kun Sindhicontaminationexplicit; Kurux mōx bread vsōn rice distinctionexplicit. LongerGondipronoun/auxclauses, roundedMannan tuneyforms, causative-lookingkuḍisu andopaquecompletivesnotincluded. Actualconsumption queryeat wastruncated1000 due substringgreatetc; corrected exact phrase to eat query produced130rows, drink205. No relianceontruncatedpoolabsence.

Next093:state-adjectives-inventory.json captures broken/different/same/all/whole remainingrecords. Requestsfileexists4queriesbreak,different,same,entire. Discoveryrunning session22535 after initial commandpath typo (failedbeforeexecution; corrected once). Pollsamehandle. No newadjectiverulesorfullprimaryinspectionyet. 092checkterminalexit0; no otherjobs. No rebuild/acceptedwrites. Deadline17:50UTC remains.

Discovery22535 completedexit0:state-adjectives-results.json has4queries630rows. No runningjobs. Time15:15UTC; continue093 fullprimaryreview.

## Checkpoints093–094

093 passed9,966records/10,523rows,554parents,1,724holds,25,614unexamined,772batches. Added102broken/different/whole proposals and15holds afterfull946/520/5008/5548/4992/5017/4206. Gondiuruŋbreak explicitlycompetes946vs662; full662subsequentlyread confirms reciprocalcross-reference, held. Gadabapunicende brokenheldextra-morphology. Southernvēre/bēre reduplicationonegrammaticalstem; nottwoetymonrows.

094 passed10,019records/10,576rows,555parents,1,731holds,25,554unexamined,776batches. Added53total/all/same proposals via5119/958/844/990(b),7holdsforKuruxhorm/hurm990(a)parentmismatch,Gondi/Gadabamottamtransmission,Mannanmottamasamegloss. Full5119/958/844/990read,parentschecked. Bothshared dry-runs, manifestagreement,zero secondapply passed. No rebuild/acceptedwrites.

Next095:equal-gloss-results112rows identifies924 forIrulaoppasame; full924readandvaliddirectcomparisonbutparentmetadata/ruleNOTyetdone. Earlieroppa/oppu formsearchesdidnotexposematchinggloss, actualequalglossquerydid. Otherone+like compounds remainblockedby990or-/on-branchparentdifference; noassignmentto*okk.

Quality-adjectives-inventory.json capturesbad/light/good/weak remaining; discovery requestsbad/light/strong/weak running session81989, pollsamehandle. Manylightformsarekanam-illa weight+absence (1404+2559) orweight+less(1851); couldanalyseorderedcomponentsafterfullprimary/currentparentchecks. Stronghasnoremainingexactgloss. Goodoli-/olli-formsneedhomepagegoodlead; good3610nalparentknownwetlandmismatchmustnotoverride. No qualityadjective ruleswrittenyet. Timeabout15:20UTC; deadline17:50UTC unchanged.

81989completedexit0:quality-adjectives-results.json4queries1343rows. No runningjobs. Continue095fromprimaryevidence;15:20UTC.

## Checkpoint095

Passed10,065records/10,641rows,559parents,1,739holds,25,500unexamined,783batches. Added46quality/light proposals including19ordered compounds;8holdsforKondalight Sanskritsulabha, GonditelikaTelugutransmission, puholightversusheavy4127andardo/Halbicomparison. Full1017/5086/5078read,allcurrentparentschecked. Shared dry-run/manifestagreement/zero secondapply passed. No rebuild oracceptedwrites.

Next096:good-gloss-results3queries428rows completed. Full3735read:niya/niyāṭGadaba,negiKonda,nehĩKuwi, explicitIAconvergenceandParjiniko/Go nikkaborrowed. GoodGadabaniyate7andKondanegika1potential;no096rulesyet. Full848,2616,4547,5029read;Muthuvantsatta/cattabad vs cītta vowelissue, mosambad mayexternaletymology;do notforce. Nature-extra-inventory/requests written(wind,cloud,lightning,rainbow,mud,firewood,river),actualhomepage discoveryrunning49433. Nootherjobs. Timeabout15:32UTC,deadline17:50UTC.

## Checkpoint096

Passed10,165records/10,741rows,562parents,1,739holds,25,400unexamined,792batches. Added100proposals:8good3735,65wind1481/1499/3149/5312,23lightning4876/5074,4firewood5440/2586/4328. Fullarticles/currentparentschecked; IAconvergencecaveat3735retained. Shared dry-run/manifestagreement/zero secondapply passed. No rebuild/acceptedwrites.

097currently38newlandscape proposals and16holds drafted, notyetvalidated; withdrewtentativePaliyacatayi-mud2266forunverifiedmedialreduction. Historicallandscape-reviewed.jsonincludes39tentatives;canonicalrulesare38. Full3370/4318/5192/3543/1478/4016/4291/2020/2266/5245/3555/1278/1395/1619/1765/4641/5033/1041/1165/1370read. Parji/Gadabafirewood1165vs1370explicitlyalternative12held;Kotakiṛmcloud1619derivationheld;Kadarkarucloud1278vs1395held;Gondikuhra1765contactheld. Actualnature-form-results9queries1599rowscompleted. Nature-contactdiscoveryrunning18418(stream,cikala,kāṣṭha,mēgha);prepare097running. Deadline17:50UTC.

097completed:10,203records/10,779rows,570parents,1,755holds,25,346unexamined,797batches. Sharedcheckandmanifestagreementpassed,zero secondapply. Protected-input-check-097.jsonconfirmscompiledforms/edges,identities,browserDBallunchanged. No rebuild oracceptedwrites.

## Checkpoint098

Passed10,209records/10,791rows,571parents,1,842holds,25,253unexamined,797batches. Added6Mannanāttōramriverbankcomponents5159+1060,87contactholds:Gondicikalmud4784/4780,katiafirewood3120,Dravidianmeghacloud10302,1big-riverbankextraunresolvedcomponent. FullCDIAL4780/4782/4784/3120/10302read. 10304wasalsoinspectedbutmeansmēghyanotmēgha;noassignmentmadetoit. DEDR2488read:gumfamilydoesnotdocumentGondicikalmud. Sharedcheck/manifestagreement/zero secondapplypassed.

099drafted10compounds afterfull5276/178/5422/5381andparentchecks:5Muthuvanarrow+bowrainbows,1Irulasky+bowrainbow,3big-watercourse,1big-river-margin(the1holdfrom098nowresolvedbecause5276explicitvaliyabig). Theseawaitvalidation;prepare099running. Motion-verbs-inventory/requestsready,11actualhomepagequeriesrunning7754. Mostlyremainingverbphraseshavepronouns/auxiliaries;do notassignwholephraseasoneunanalysedstem. Deadline17:50UTCstillactive.

## Checkpoint100

099failedbecauseinputs changed duringvalidation(concurrent task);preservedlog. Freshprepare/render100andsharedcheckpassed:10,351records/10,944rows,577parents,1,845holds,25,108unexamined,805batches,zero secondapply andmanifestagreement. Includes10compounds099+132simpleverbproposals100.4Kuruxkeraspastgoheld1419vsquestioned2814suppletion;full2814nowconfirmsquestionmark. Full5270/2781/4572/1419/2598/4473/713/3151/3582/871/2132/2831readfor100. Additionalfull3098/2053/984/480/930/1728/2602/2343/3675/4778readfor101(notyetanyrules).

Motion-verbs-results11queries2061rows,verb-stems-results6queries3068rows(ta/da1000substringtruncation;donotassumeabsence),gondi-verbs-results4queries153rows,gondi-aux-results3queries262rowsallcomplete. Gondi ji-kill2602 nowexplicitprimary;runvitsearchreturns5401SOWonly,norunleadconfirmed. Gondi man-exist4778validlead forauxphrases. WebprimarySouthernGondiAheriGrammarbyBennyKurian/MaryB.Kurian/JoPennyfound at https://www.gondwana.in/sites/www.gondwana.in/files/pdf-files/SGondi%20%28Aheri%29%20Grammar%20final%202000%20%28Unicode%29.pdf . Pages24–25(PDF30–31)documentperfectwithjī/cī/sī+manandcontinuousjeke/ceke/seke+man. PDFskillloaded;webscreenshotfailedcachemiss;defaultcurlDNSblocked,escalatedpublicdownloadattemptpending. Do notclaimscanverifiedyet. No101ruleswritten. Deadline17:50UTC.

## Checkpoints101–102

101added21simplegive/take/sit/stand/killrecords,batchedwith102forvalidation.102added52Gondimorphology proposals:30orderedlexicalverb+manauxiliarycompounds,22gerundreflexes. PDFpublicdownloadescalationallowedandsucceeded;local/tmp/gondi-aheri-grammar.pdf. Completeprintedpp24–25,27–28visuallyverifiedfromrenders31–32,34–35. Thegrammarseparatesparticipialjī/cī/sī+manauxfrom-maḍ/-uḍuverbalnounendings. Sourceauditgondi-grammar-evidence.mdcontainsURL/hashandlimitedsummary;nonewingestion/nodes/referencewrites. Shared102check/manifestagreementpassed10,424records/11,047rows,583parents,1,845holds,25,035unexamined,810batches,zero secondapply.622compoundrecords(nowone3componentrecord),7borrowings,2derived.

103drafted18newburn/hear/restproposalsand4Kuruxburnholds;prepare103running. Full5517/4801/5178/4537/4517/1207/656/1064/1514readandparentschecked. Kuruxoddas/olnaburnheld1064rhoticrootnotcorroboratedpast. Gondi(kisu)masaburn1compound1514+4801,2kenjmadhear2017,2uai(madi)burn5517. Theseawaitvalidation. Remainingburncausatives/auxphrasesnotforced. Nootherjobs. Deadline17:50UTCunchanged.

## Checkpoints 103–104

10,528 proposed records / 11,152 assignment rows / 589 distinct parents; 1,849 holds; 24,927 unexamined; 816 review batches. Both shared dry runs and exact manifest agreement passed, with zero second-application changes. Added burning/hearing forms (18), then above/below/behind/front forms (86). Four Kurux burning forms held for unresolved consonants and inflection. No rebuild or accepted writes.

## Checkpoints 105–107

10,781 proposed records / 11,470 assignment rows / 602 distinct parents; 1,968 holds; 24,555 unexamined; 831 review batches. All shared dry runs and exact manifest agreement passed with zero second-application changes. Added 27 distance/proximity forms, 17 insect/louse/bee records (8 compounds), then209 tool/thread/oil records (57 compounds). Held contact-route cases, explicitly questioned comparisons, opposed spatial glosses, ant parent d864 (currently hill) and insect d4353 (currently lick). No rebuild or accepted writes.

## Checkpoints 108–109

10,837 proposed records / 11,526 assignment rows / 609 distinct parents; 1,994 holds; 24,473 unexamined; 837 review batches. Both dry runs and exact manifest agreement passed, zero second-application changes. Added28 sand/rope forms and28 body/dimension forms. Held14 sand575 forms because primary text lacks its antecedent head/gloss,8 Muthuvan rope forms with unresolved medial consonant and4 Appendix2 rope contact forms. Checkpoint110 adds11 records (8 compounds) and6 holds but first validation stopped on concurrent-input change; fresh retry in progress. Protected hashes110 all unchanged.

## Checkpoint 110 refreshed validation

10,848 proposed records /11,545 assignment rows /609 parents;2,000 holds;24,456 unexamined;841 batches. First110 attempt stopped on concurrent input change. Fresh110-retry passed and exact manifest agreement/zero second-application changes verified. Added11 records (8 compounds);6 Muthuvan mēvīḍu roof forms held for upper-house versus thatch-house ambiguity. No rebuild or accepted writes. Checkpoint111 draft adds110 food-plant records and35 white-garlic compound holds; awaiting validation.

## Checkpoint111

10,958 proposed records /11,714 assignment rows /612 parents;2,035 holds;24,311 unexamined;851 batches. Shared dry-run and exact manifest agreement passed, zero second-application changes. Added110 food-plant records (59 compounds), with explicit modern crop extensions and source botanical meanings retained;35 garlic compounds held for the white/butter parent mismatch and exact second-component analysis. No rebuild or accepted writes. Next research:remaining kinship/person terms; homepage discovery complete, primary checking pending. Deadline17:50 UTC unchanged.

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
