# Das (1987) Yerava etymology research

Active objective: etymologise remaining Dravidian sources, prioritizing smaller and underetymologised sources for diversity. Unlimited time; no rebuild. This pass reviews already installed lexical records and does not re-ingest the source.

90 records verified in current CLDF, all initially unlinked: Ravula48, Paniya42. Source preserves popular orthography, with no phonemic transcription. Raw OCR and reviewed.tsv read; existing high-resolution kinship scan visually inspected. Grandparent brace and Avva co-equivalents confirmed. Source p66 describes Kannada/Malayalam/Kodagu multilingual contact: comparison with a nearby language does not establish a unique inherited or borrowed route.

First checkpoint:18 pending records under8 DEDR articles;6 explicit kinship holds;66 unexamined. Four language-owned review batches in REVIEW.md. Full DEDR articles183,156,273,131,23,3085,3015,4813,513 read. Homepage discovery-results.json records10 actual fetchLemmaList queries. DEDR IDs are whole articles per user clarification; do not modify headings. Exclude duplicate-excluded primary audit rows (bone1288 problem documented in sibling survey pass).

prepare-kinship.py reproduces ONLY the first checkpoint and will overwrite later decisions. Do not rerun blindly after extending this pass. It reads current accepted sidecars. Future proposals should use exact persistent IDs and preserve accepted assignments by others. render.py and verify-checkpoint.py produce and check owned pending manifests. Never use save-pass without user review approval.

Next: basic-results.json (10 homepage gloss/language queries) supplies discovery for fire/water/moon/hill/forest/milk/sleep/drink/eat/stand. Read complete shortlisted DEDR articles before proposing. The remaining source spellings include Thee/Theeyu, Neeru, Nilavu, Betta/Male, Kadu, Palu, Urang/Urangu, Kudi/Kuti, Thinnu, Nil/Ninja. Preserve popular spelling; do not infer phonemes. Compound hot water, Bolle Kallu and possible derivations need full analysis. Sanskrit-looking sun/moon/earth/god terms need immediate donor research; do not jump to Sanskrit ancestors.

Broader source ranking and historical12,286survey proposals remain at ../dravidian-surveys-20260921/. These are distinct owned research passes. No accepted sidecar writes, rebuild, node creation, commit or publication.


## Checkpoint002

38 proposed records /38 assignment rows /19DEDRarticles;7holds;45unexamined,8ownedlanguagebatches. Both001and002 shared dryruns passed with exactmanifest agreement and zero secondapplication. No accepted writes or rebuild.

Added20 basicvocabulary proposals after full primary3266fire,3690water,3754moon,4096milk,1438forest,5474Betta hill,4742Male hill,707sleep,1654drink,3263eat,3675Nilstand. PaniyaNinja stand held for unexplainedformation. add-basic-002.py extends originalkinshipcheckpointexactly; do not rerunprepare-kinship.py overcurrentdecisions. It would reset this pass.

Next45sourcewords inunexamined-inventory.json: houseKullu/Pire, hutPalli/Pire; firewoodPure/Kodu; hotwaterBengeNeeru/KanjaNeeru; sunNera/Chooriya; moonChandra; earthBoomi; godDevaru/Pagava/Thayya; cookedriceCurry; meatRechi/Irechi; arrackCharaya/Kallu; toddyKallu/BolleKallu; marriageKanjala; deathSavu/Chavu; birthPuranda/Pechazhainju; lightBulaku/Boli; dayPagelu; nightAndi/Iravu; clothSinde/Battenje; comeBa,goPo;sitUlesu/Ilecha. Use actualhomepage queries; fullprimary evidence for everychosenarticle; compounds and contact donor routes need separateanalysis. Keep reviewedpopularspellings. Sourcepage97and98PNGs available under workspace tmp/pdfs/yerava-census-1981/; kinshipcroppreviouslyviewed.


## Checkpoints003–004

58proposed records/58rows/30DEDRarticles,10holds,22unexamined,12ownedlanguagebatches. Both shared checks passed with exactmanifest agreement and zero secondapplication. No accepted writes/rebuild.

003added16: Pire house/hut4294; Palli hut4018; Kallu arrack/toddy1374; Pagelu daytime3805; Iravu night2552; Savu/Chavu death2426; Ba come5270; Po go4572; Bulaku/Boli light5496. All fullarticlesread. 004added4: Rechi/Irechi meat529 and Nera sun3774, with complete primary. Three newholds: Curry cookedrice(2) cannot choose kūḻ rice1911 vskaṟi curry1391 frompopularspelling; Pure firewood(1) uncertainpuḷḷe4328 vsloadpore4565 definition-substring falsematch. Source page98PNG visually checked this turn; transcription exact.

Research logs remaining-results.json (10formqueries,1491rows) and harder-results.json (7glossqueries,258rows). Broadformshortsearchescanbe capped; neverassert exhaustive. add-remaining-003.py andadd-meat-sun-004.py reproduceextensionswithIDskip; do notrunprepare-kinshipreset.

Next22: KulluHouse,KoduFirewood,BengeNeeru/KanjaNeeruHotWater,ChooriyaSun,ChandraMoon,BoomiEarth2,Devaru/Pagava/ThayyaGod,CharayaArrack,BolleKalluToddy,KanjalaMarriage2,Puranda/PechazhainjuBirth,AndiNight,Sinde/BattenjeCloth,Ulesu/IlechaSit. Compoundfirstelement and donorroute evidence required. No hypothesesauthored for these22yet. Earlier held6kinship+Ninja also unresolved.

Global inventory script now reads explicitgoal-pass-registry.json including thispass and oldsurvey pass, so future sourcepriority refresh counts bothpasspendingrecords. Current priority files remain the123historicalsnapshot untilnextrefresh. Script stillwrites123namedoutputs; rename beforepreservingnewcheckpoint.


## Checkpoint005 and hold revision006

61proposed records /64assignment rows;13holds;16unexamined;12ownedlanguagebatches. Sharedcheck005 passed, manifest agreement and zero secondapplication verified. Revision006 adds3holds only; proposedrowsunchanged andverify005 rerun successfully. No check006needed/run. Noacceptedwrites/rebuild.

Three compounds added by add-compounds-005.py: BengeNeeru5517+3690 (Kannada beṅge heat/fire directly in primary), KanjaNeeru1458+3690 (Kodagu kāñj- secondstem of kāy be hot/boil; tentative Paniya formation), BolleKallu5496+1374 (tentativewhite+toddy; dictionaryTulu boḷḷena/boḷḷarè, sourceonlyToddy, keepqualification). Full5517and1458readthisturn;5496,1374,3690fullarticlesreadprior. Render includes orderedPos. Familyproposal doesnotproveactualborrowingroute.

DBIA13/219/318fullprimary read fromparams.csv andoriginalOCR219. DevaruGod held for immediate donor (DBIA219 IndoAryandeva, Kannada/Kodagu alternatives); AndiNight heldDBIA13(<saṃdhi twilight, Kodaguandi eveningjustafterdark; sourcebroaderNight); BattenjeCloth heldDBIA318(baṭṭe<paṭṭaNOTvastra, unexplained-nje). See dbia-hold-evidence-006.md.

compound-loan-results.json:7actualhomepagequeries91rows. residual-results.json:9queries36rows. KulluRavulaHouse andKoduPaniyaFirewood onlyreturnthesourceresponses; KanjalaMarriage globalexactsearchreturns unrelatedIAkañjalabird; PurandaBirth onlysource; SindeKannada noresult; Ulesu/Ilecha onlysource; Charaya globalhitsPrakritalkali/ashesnotarrack. These searches donotruleout plausiblevariants and theseentriesare stillunexaminedpendingexpandedsearch. Do not convertanexactsearchmiss intoetymologicalexclusion.

Next16: KulluHouse,KoduFirewood,ChooriyaSun,ChandraMoon,BoomiEarth2,Pagava/ThayyaGod,CharayaArrack,KanjalaMarriage2,Puranda/PechazhainjuBirth,SindeCloth,Ulesu/IlechaSit. LikelynextqueriescārāyamTamilforarrack, kannālam/kalyāṇammarriage, piṟantabirthfamily, cindecloth, Malayalam/Tamildivineborrowings, same-lectsitcomparisons. Fullprimaryrequired; immediate donors unresolved for transparentloanappearance. Keepunlimitedgoalactive.


## Checkpoint007: initial assessment covers all90records

62pendingrecords/65rows/28holds/0unexamined;13ownedlanguagebatches; shareddryrun007passed, exactmanifestagreement andzero secondapplication verified. Added qualifiedPurandaBirth→4422, fullprimarypiṟantai birth/piṟa(-nt-) read. Fifteen remaining records explicitly held after variantsearches, with evidence/reasons inresearch-holds.json. This completes initial coverage, NOT etymologyresolution or userapproval. The28holds remain researchwork; prioritize broader small-source diversity next as requested.

final-variant-results.json8queries112rows;last-results.json5queries6rows;CharayaTamilLexiconprimaryreproduction read (external-evidence-007.md). Avoid derivativewebsiteclaims as etymologicalevidence. KulluHouse,KoduFirewood,SindeCloth,Ulesu/IlechaSit need same-lectsourceevidence; KanjalaMarriage cannot silentlynormalize tokalyāṇa; PechazhainjuBirth cannottruncate; Sanskrit-looking solar/lunar/earth/godterms have unresolvedimmediatedonors. classify-residual-007.py records exactID-safechanges. REVIEW.md now saysallrecordsassessedratherthanunexaminedremain; rendererboilerplatewillneedthatsamefixifrerun.

NEXT source ../pattapu-20260922/inventory.json generatedfreshfromcurrentCLDF+allsidecars+rank1graph. Source Lindgren2023, upstreamPattapuIRA2013. Sourcepackage data/data/other/forms/raw_data/ia_dravidian_2026/lindgren.tsv, lindgren-thesis.pdf, metadataandlanguage-map. Read upstreamsource/phonemicnotes, thenhomepagefamilysearches andfullprimaryevidence. NoPattapuproposalsauthoredyet. Goalunlimited, no rebuild, noacceptedwrites.
