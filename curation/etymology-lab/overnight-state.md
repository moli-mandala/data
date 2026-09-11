# Latest checkpoint — all approved proposals saved

All 741 approved proposals are saved: 957 links / 899 records. The donor supplement installed 406 heads with real persistent IDs and used them in 416 approved analyses; two dependent Sauji numeral proposals are also saved. Final addition: 435 links / 432 records / 418 proposals. Approved outstanding: zero. Remaining unreviewed 2,683, held 78. The donor ledger and approved-save-ledger are the current state. Historical checkpoints below are superseded. Automation remains paused.

# Overnight etymology checkpoint

## User instruction and deadline

Research all supportable remaining etymologies in order **Sauji (Sv), Ushojo (Ush), Palula (Phal)**. Exhaust supportable analyses and explicitly record hard cases for each language before moving on. The user superseded the 40-proposal batch limit with overnight work. **Final deadline: 09:00 America/New_York on September 10, 2026 (13:00 UTC).** The user explicitly corrected an earlier September 11 interpretation. Do not use September 11.

Do not ask for intermediate batch approval overnight. Keep all new analyses pending for the morning review, grouped by language and difficulty: straightforward, qualified, unresolved. Use the jambu-etymology skill's four-column table. Do not force doubtful analyses just to exhaust the inventory. Distinguish unresolved from not yet researched. No subagents were requested. No commits, deployment, or shared compiled data replacement.

## Scheduling

One active thread heartbeat: `overnight-shinaic-etymology-research`, attached to task `01a087a0-28d7-7350-ae11-13dfaee21c1e`. It runs hourly on the hour, including 09:00. The same heartbeat handles research and the final review; the app allows only one per thread. **At or after the deadline, present what is ready honestly and pause it.** If research finishes early, update this same heartbeat to the one-time 9/10 09:00 handoff specified in its prompt. Do not create a workaround cron or a second heartbeat. Read its TOML through the app configuration if updating, preserving full fields.

## Accepted state

Sauji batches 001–009, proposals #1–320, are saved. Latest batch009 (#301–320): 27 rows across 27 records, graph validation and idempotence passed. No unapproved Sauji assignments have been saved. Accepted overlay: `data/data/etymology-assignments.csv`.

Starting remaining inventories (fresh overlay applied logically to working forms): Sv 281; Ush 529; Phal 2850. Full record snapshots are in `overnight-research/{Sv,Ush,Phal}-initial-inventory.json`. No records have been removed by overnight proposals yet. Keep one disposition per record, avoiding double counting variants and proposals.

## Current pending work

Active research has resumed. All proposals remain pending; the accepted overlay is unchanged.

- Sauji `Sv/batch-010.json`: 42 proposals #321–362 covering 54 records; 14 existing-ID candidate rows and 36 donor requirements. Last additions: brother’s wife under CDIAL9660; Pashto hammer and question particle; source-attributed Gawarbati rate on and rupai money. Some existing-ID component rows still depend on uninstalled donor ancestry, so full Sauji candidate validation remains deferred.
- Ushojo `Ush/batch-001.json`: 220 proposals #1–220 covering 296 records; 324 existing-ID candidate rows and 21 donor requirements. All 324 rows passed current-overlay validation, temp-graph inspection and repeat application with zero changes. Later ten donor-only proposals did not change these rows. The validation does not cover donor requirements.
- Palula `Phal/batch-001.json`: 479 proposals #1–479 covering 549 records; 190 existing-ID candidate rows and 359 donor requirements. All 190 rows passed current-overlay validation, temp-graph inspection and zero-change repeat application. Fifty-three Pashto donor comparisons independently checked in OPED (one has an existing ID); ten Urdu donor IDs independently resolved via full CDIAL prose. Most remaining donor requirements are source-attributed comparisons awaiting independent lookup. Urdu language ID is H (Hindi-Urdu), with Urdu variety explicit; not U.

Current total: 741 proposals covering 899 distinct records. Of 3660 starting records, 2761 have no pending proposal yet; 78 explicitly held cases are recorded separately. These are proposal coverage counts, not accepted saves. The accepted remaining count still includes all unsaved proposals. Accepted overlay SHA256 remains fed5fa9e627857ec47fdd7cf3bb7342a475979b33c3a7b03460254e6c7e2b051.

Recent builders: `/tmp/palula-first-loans.py`, `/tmp/palula-native-first.py`, `/tmp/palula-inflections.py`, `/tmp/palula-urdu-reviewed.py`, `/tmp/palula-core-duplicates.py`. Do not rerun builders blindly: they append or assert no duplicates. `/tmp/validate-palula-pending.py` and `/tmp/validate-ushojo-pending.py` are safe validation helpers. `/tmp/refresh-overnight-review.py` regenerates review and dispositions; it includes unresolved-notes.json. Source IDs can repeat headwords: resolve by NFC Original AND matching Origin, not a single-headword dict.

Next priorities: independently verify remaining proposed donor entries; expand still-unreviewed Palula native inflections and transparent compounds with source morphology; finish the Sauji and Ushojo held/research queues. Do not treat past-tense crossreferences as proof of identical etymons: suppletion is common. Palula kʰāṇ was proposed under khaṇḍa 3792 (qualified) because Turner 13627 explicitly prefers it to skandha; gāvaṇḍí is a Lahnda loan per addendum 14461; kʰēci bad uses 2613 per addendum 14343 rather than the superseded mud comparison. Survey nū̃ eight, hall/condition merged senses and English response contamination need source checking.


`overnight-review.md` is generated by `/tmp/refresh-overnight-review.py`; counts and record dispositions are under `overnight-research/`. Not yet researched remains distinct from unresolved. Continue extending these open manifests, not creating duplicate proposals.

Primary Palula source CSV: `/tmp/palula-entries-v1.2.csv` (2700 entries), fetched from https://raw.githubusercontent.com/dictionaria/palula/v1.2/cldf/entries.csv . Exact headwords match Jambu Original after NFC normalization. Liljegren says Origin gives the most likely donor, with transmission and inheritance/borrowing sometimes uncertain; do not treat Origin as proof of a certain borrowing. Conjunct verbs need local components, and compound/derived/multidonor/semantic mismatch entries require individual treatment.

OPED complete published XML `/tmp/oped-archive-20251030.xml`; searchable index `/tmp/oped-full-index.json`. Use text rather than the effectively empty trans field. Archive DOI https://doi.org/10.5281/zenodo.17487678 . Entry links https://oped.univie.ac.at/oped.php?entry=N .

The accepted database cannot point to a parent whose Status is `unlinked` unless its ancestry resolves through other accepted assignments. Do not bypass that validation. Some imported Persian/Arabic category nodes (e.g. Kalasha generic donor categories) are **not lexical donor forms**; never use them for new loan links. New curated donor heads require true identity registry resolution; non-CDIAL parameter strings are not public persistent IDs.

## Sources and working data

- Full working forms: `/tmp/kalkoti-b18-build/data/cldf/forms.csv`
- Base edges: `/tmp/kalkoti-b18-build/data/cldf/edges.csv` (Kalkoti curated; Sauji overlays need applying to TEMP graph)
- Language IDs confirmed from sibling languages.csv: Sv, Ush, Phal.
- Full CDIAL prose/addenda cache: `/tmp/kalkoti-cdial-entries.json` (numeric-key dict, `text` and `page`). Inspect full prose, not merely parsed candidates. Generated subsection labels can be wrong.
- Raw CDIAL: `data/data/cdial/cdial.pickle`, `cdial.csv`.
- Primary Knobloch Sauji text/PDF: `/tmp/palula-accent/knobloch2020.txt`, `knobloch2020.pdf`. Local extraction `data/data/other/forms/raw_data/20260825-knobloch-sauji-extract.psv`. Printed page = PDF page minus 4.
- Knobloch URL: https://su.diva-portal.org/smash/get/diva2:1440556/FULLTEXT01.pdf
- Save helper latest `/tmp/save-sauji-b9.py`; backup `/tmp/sauji-before-b9-assignments.csv`; verified temp graph `/tmp/sauji-b9-verification-edges.csv`.
- Pending batch009 was prepared by `/tmp/prepare-sauji-b9.py` and is now saved.

## Current loan research

Primary OPED results preserved in `overnight-research/sauji-oped-searches.json` and `sauji-oped-extra.json`. These are research evidence, not lexical import files. Entry text is between `Pronunciation guide` and `Give feedback` in `text`. Direct entry URL: https://oped.univie.ac.at/oped.php?entry=NUMBER . The search sometimes lands on the **wrong homonym**; inspect definition before using.

Verified and seeded:
- kumāk help → Psht kumák/komák 31579. Knobloch p39 explicitly calls it a Pashto loan.
- mana apple → Psht maṇá 38762. Knobloch p39 explicit.
- śpank-ay shepherd → Psht špankáy 27888 herd boy/young shepherd.
- māśum child → Psht māšúm 37807 child/innocent, probable route.
- musāferi/musafari/musapʰari journey → Psht mosāferí 38424, whole-noun borrowing preferable to assuming local derivation.
- rakam different → Psht raqám 23838 kind/type; semantic extension qualified.
- bat after → Psht baʿd/bād 9343 after/afterward; route qualified.
- qazī judge → Psht qāzí 30233, regional alternatives.
- gap word → Psht gap 32333 talk/conversation, Persian possible.
- zindagi life → Psht zindagí 24972, Persian possible.
- ban close → Psht band 9593 closed/blocked; Palula ban also exists, route unresolved.
- sagardan nervous → Psht sargardā́n 26630 worried/anxious; first r lost.

Promising next OPED work:
- aw/av and: او has homonyms 7496–7499; fetch each, choose conjunction.
- xo but: خو search gave 20834 **tinder**, WRONG. Use homonym navigation or another primary source; existing Psht xō record f_bgcd4yx7vhpak is unlinked and cites Grierson LSI1928 p230–231.
- sava hundred: سوه has 27604–27606; سل 26969–26971. English search hundred includes دوه سوه 22503/22504 (inspect link-text mapping), سل 3 26971. Check the inflected/plural hundred form, not a guessed head. Knobloch p25 explicitly says sawa hundred in sāt-sava is Pashto; once donor is verified, analyse dusava 200 and sāt-sava/sāʦava 700 as components with local numerals.
- yaŋg/yang battle: جنګ homonyms 16988/16989; verify battle versus rust.
- daro valley: دره search gave 21828 **lash/whip**, WRONG. English valley results are in extra JSON; identify correct homonym.
- qaśuq spoon: English results include قاشوغه (30230) and کاچوغه (30632), inspect before assigning.
- māxom afternoon: Psht māxustən/māxām prayer/evening family, but semantic time mismatch needs qualification. Extra evening results show ماخوستن and ماښام; inspect IDs via HTML mapping.
- xvax good: خوښ likely xwax/xwəx, extra good result IDs; verify.
- topos question: Knobloch p39 explicitly Pashto. OPED Arabic search تپوس no result; investigate spelling or use another primary source. Do not use generic question etymon from old Kalkoti.
- vosedal/vasedal live: very likely Psht osēdəl with prothetic v; OPED اوسېدل no result, try variants/different primary dictionary.
- xavar flat: Knobloch p39 explicitly Pashto. Search هوار, not خوار poor.
- tayar, taqriban, śair, tāndor-e: obvious regional loan families but OPED head searches failed; alternate orthography or primary Persian/Urdu dictionary needed.
- navāsu grandchildren: OPED nawāsá 34747 exists. CDIAL6954 explicitly calls Dardic nawāsa forms Persian loans. Immediate Gawarbati versus Pashto route needs adjudication. Gawarbati navāsa/navāsi kinship attestations exist but are unlinked; do not simply call these inherited nápāt. Feminine navāsi may be independently borrowed; do not default to local derivation.

Public OPED research scripts `/tmp/sauji-oped-research.py` and `/tmp/sauji-oped-research-extra.py` POST only searchQuery to oped.php. They used read-only network sandbox escalation successfully. Raw HTML is `/tmp/sauji-oped-N.html` and `...-extra-N.html`. Search kwargs alph0,nons1,dir0,how0/1,what0/1,dev1. English partial queries have many irrelevant substring hits; match headwords to entry IDs before fetching. Do not blindly trust first result or IDs printed without head mapping.

## Remaining native leads and cautions

- am mango, kelo banana, ālū potato, bāźāro millet likely regional loans. Existing Hindi parents available; exact immediate route still needs evidence. CDIAL1268,2712,1388,9201 primary prose verified.
- botīŋgaṛeā tomato matches Shina batīŋgaṛe under vātigaṇa11503 (older eggplant name); semantic extension/borrowing route needs assessment.
- torc- thirsty: tr̥ṣyati5942 (Gy truš-, Lahnda tarsaṇ); tr̥ṣā5936 has Palula triṣel-. Need account for Sauji c.
- nilbel- grow resembles Shina niliž- under *nirlīyate7389, but medial b unexplained; hold unless source morphology resolves.
- gerān-al-ē walk around: compare regional gir-/ger- under *ghir/*ghēr4474; transitivity and stem subsection need care.
- lo red may shorten native Sauji lohĩló (f_3jpsht5sirqqg; *lohila11168); do not equate lo dust or loy black automatically.
- ūpʰo light resembles ūbo under udbhūta2046 but pʰ needs explanation.
- dʰṓ thread resembles Shina dōm under dāman6283 but h and lost m unresolved.
- pēkibo cooked: pakva7621/*pikva alternative; additional morphology unresolved.
- śan roof, piśo flour, dēś perfective see already saved; do not repeat. Flour explicitly belongs to *peṣita8386.2, not piṣṭa8218.
- 'soil' sum linked by old source to sumahānt13493 is **very doubtful** in Turner; do not promote without new evidence.
- 'broom' bāborī: Turner11378 says regional comparison unclear; do not promote by similarity alone.
- kirmi worm, beautiful śubāṇu, say man-/men- were saved with alternatives; don't remove qualifications.
- Pronoun la/li/le family unresolved; no automatic t→l assumption. Past be al-/bil- unresolved. Shina kārē when →2918 is parser spillover, NOT evidence.
- Knobloch pp20–22 treats maṭē,tuṭē,asonṭē,tusoṇṭē and asondiyo as postpositional constructions with ṭē/diyo. Need establish those postpositions before complete component analyses. Remote taseṭe/taṭe/tenoṭe same issue.
- Knobloch p26 lists postpositions and suggests ratē on may be Gawarbati loan. It is not an explicit certainty. Verify donor lexical entry.
- Mixed Decker elicitation responses and apparent wrong number glosses (sātʰāṣ seven looks seventeen, ṇū̃ eight, bīś pā̃y twenty, ī neheā water, etc.) stay separate and unresolved; don't silently correct source content.
- Kinship bovṛi paternal uncle and compound family, sarāṇi wife's sister, avxey brother-in-law, ʣ̣āmilī husband's sister, barkaṭey stepchild family need further source research. Preserve gender/ego distinctions.

## Most recent heartbeat research (September 10, before 06:00 UTC)

Added 29 proposals covering 39 additional records (five Sauji, 24 Ushojo), verified 17 further Palula Pashto donor entries, resolved ten existing Hindi-Urdu donor nodes, and expanded source/onomastic triage. New helpers `/tmp/sauji-next-verified.py`, `/tmp/ushojo-extra-native.py`, `/tmp/ushojo-extra-loans.py`, `/tmp/verify-palula-pashto-next.py`, `/tmp/palula-resolve-urdu-ids.py`. Do not rerun append builders. `/tmp/oped-candidate-matcher.py` creates normalization-based discovery candidates only; it intentionally does not accept matches. OPED XML has usable direct `<trans>` and `<alts><alt-t>` fields even though the original index trans field was empty. Accent-stripped matching must be checked against the sense.

Key donor homonyms: OPED nwasáy grandson 34798 versus núsay tweezers 34799; nwasəy granddaughter34800; alú potato6953 versus ālú plum6951; ləka similative33342 versus laká stain33344; dáma rest22197 versus damá snowstorm22198. Palula kāti pack-saddle matches OPED plural káti of káta30889; qualify plural borrowing versus local adaptation. Palula čāṛā inability to speak is stronger than OPED stammering meaning17424. Sauji aya likely Dari via Pashto per Knobloch; preserve route uncertainty.

Ushojo aẓo wet/rain/cloud is supported by Shina áẓu and Prakrit adda's complete meaning range under ārdrá1340. Bone and louse words held because Turner explicitly marks comparable Shina forms as borrowed; do not automatically label inherited. Ushojo pīval ant provisionally uses pipīla8201 with Prakrit pivīliā, not the *pilīla branch, because v needs accounting for. Review is qualified.

Existing Urdu donor candidate matching produced many false friends (ām mango versus common, agar aloe versus if, har green versus every); these were rejected. Only ten semantically verified existing nodes were used, keeping Liljegren's immediate Urdu attribution and any contact uncertainty. Relevant CDIAL11745,5543,6582,10094,10331,7761,13577,5466,4116. No accepted overlay edits, commits, deployment or shared compiled-data replacement.

## Research checkpoint September 10, 06:15 UTC

Added eight Ushojo proposals #208–215 covering nine records: separate short (3895) and lame/dwarf (3941-4) branches, elder sister *diddā6327, father/grandfather *dādda6261, fox *lōpi11142, grape drākṣā6628, sieve paripavana7843, and past grind pēṣayati8386. All except short remain qualified, including nursery/contact ambiguity, exact vowel outcomes and final-nasal loss. Do not add deyādī grandmother until the medial expansion is explained. All319 candidate rows passed current-overlay validation and temporary-graph idempotence; accepted overlay unchanged.

Independently verified15 more Palula Pashto donor entries (43 total now), using full published OPED XML. Helpers /tmp/ushojo-next-native.py and /tmp/palula-oped-third.py append/update and must not be rerun blindly. No new donor nodes installed. Latest review regenerated with700 proposals/844 records. Remaining2816 records include65 explicit held cases and other unresearched entries.

Research leads from this run: widow ṛiṇḍī needs comparison with *rēṇḍa10815-2, not an automatic raṇḍa10593-5 assignment; knot goṇ matches Shina gŭṇ under4354 but Turner marks that Shina form a possible Indian loan; navel tʰūnī has several competing branches in5860 and aspiration unresolved. Husband kaman/kʰaman, full fūrālo/purhālu, and small śinoṭo/śīnoṭo remain unassigned. Remaining first Palula Pashto donor lookups include ālúg, axpúl, caukāṭ, cíɣi, darák (21675 sign/comprehension versus21676 aqueduct lock), ɣáli, ɣōjúli, ɣoṛ, hum, kuhí, laxkár and mēx. Do not use broad unbounded substring searches: print candidate IDs first and then inspect exact full entries.

## Research checkpoint September 10, 07:15 UTC

Added Ushojo #216–217 mustache *phuṅga9083 and turmeric haridrā13992 (qualified). Added six Palula proposals #444–449, covering10 records: article ā̂k2462-2, demonstrative anú283, other dúi6402, remote nominative so12815, singular tū5889, polite/plural tūs10511. Full CDIAL prose checked and primary Palula entries inspected, including duplicate headword records. Ordinary grammatical functions do not imply derivation.

Verified nine more Palula Pashto donors in the published OPED archive: relative20186, doorframe18306, cry17911, silent29570, cowshed29662, greasy29707, also37569, well32144, army33284. Total52 independently verified OPED comparisons. The well entry explicitly supplies inflected kuhí; plural/citation-form transmission remains qualified. Six source-field mismatches were added to held cases (híṛu, kō̌, mī̂, ṣō̂, pʰalū̂ṛu, tʰī̂); these fields are not safe ancestry evidence.71 held records total.

Validation: Ushojo321 and Palula150 existing-ID rows pass against the CURRENT accepted overlay; temporary graph edges inspected and second application changes zero. The accepted overlay changed externally since the previous run: current hash e77c151ce556143fb649cc194c01a2767f276fe22d4f6f0cff7118be116dbd13. This run made no accepted-overlay writes and validation confirmed byte-for-byte preservation of the current file. Pending target IDs do not conflict with its accepted assignments. Historical hash above refers to earlier checkpoints. Coverage remains relative to the3660-record starting snapshot, not a fresh accepted remaining count.

Helpers: /tmp/palula-oped-fourth.py, /tmp/ushojo-native-seventh.py, /tmp/palula-pronouns-review.py. Do not rerun append scripts. Review/dispositions regenerated. Current708 proposals cover856 records;2804 snapshot records lack a pending proposal, including71 explicit held cases.

## Research checkpoint September 10, 08:05 UTC

Added11 proposals covering16 records: Palula #450–459 (historical past participles/adjectives pā̂ku/pē̂ki7621, píṣṭu8218, sútu13479, dā̌du/dē̌di6121, múṛu10278, lā̂du/lē̂di10946, dítu6140-4; oblique ten daśúm6227 and two dʰuím6648; kʰūr leg3906 across3 attestations). Ushojo #218 aśpī(f)/aśpo(m)horse920 is a single elicitation record explicitly giving gender variants. FullCDIALprose and Palula Main_Entry crossreferences checked. Past stems are linked to historical participles, not mechanically to present-stem etyma. The feminine pē̂ki belongs with pakva7621 via the source paradigm, not an independent *pikva claim. Dítu’s merged giving/coming-upon senses remain qualified.

Validated322 Ushojo and165 Palula existing-ID rows against current accepted overlay, inspected temporary graph edges, and confirmed zero changes on repeat application. Accepted overlay hash e77c151ce556143fb649cc194c01a2767f276fe22d4f6f0cff7118be116dbd13 preserved. Current719 proposals cover872 records, leaving2788 starting-snapshot records without proposals (71 held). No accepted saves. Helpers /tmp/palula-past-stems.py and /tmp/shinaic-extra-core.py must not be rerun blindly. Review and dispositions regenerated.

## Research checkpoint September 10, 09:05 UTC (05:05 New York)

Added Palula #460 dʰríṣṭu saw6518 and #461 śúku dry/dried12548, each explicitly cited by Turner; added Ushojo #219 nēro near7136, qualified for rhotic/contact history. Independently checked Palula darák donor OPED21675 sign/attribute, with semantic qualification for trace and excluding the aqueduct-lock homonym21676.53 OPED donor comparisons now verified. Potato proposal1 has additional stem-only verification from OPED6953; this does NOT verify the aluugaan plural needed to explain final g. Do not count it as full donor verification.

Added five explicit Ushojo holds: knot goṇ (possible borrowed Shina comparator4354), navel tʰūnī (multiple5860 branches), widow ṛiṇḍī (Turner prefers*rēṇḍa10815-2 over raṇḍa10593-5), garlic līśīm (unexplained m), sickle ūŋgī (aṅka100 comparator but vowel/ending unresolved).76 held records total. Review/dispositions regenerated. Current722 proposals cover875 records;2785 starting-snapshot records have no proposal.

Helpers /tmp/palula-seen-dry.py and /tmp/ushojo-near-review.py append/update manifests; do not rerun blindly. Palula167 existing-ID rows validated with current overlay and temporary graph; second application changed zero. Ushojo323 rows likewise passed validation, temporary graph inspection and zero-change repeat application. Accepted overlay remains untouched by this research.

## Research checkpoint September 10, 10:05 UTC (06:05 New York)

Added11 proposals covering14 records. Palula #462–471: forgot1265-2, understood9276, tired3884-2 (*khidna cluster explanation), came out7114-2 (alternative *niṣkasta qualified), came up2263-2, feared9511 (strengthened *bhītta qualified), sold3594, came10452 (source-defined L-class past), entered227 and sowed11525 (source-defined productive past endings). Ushojo #220 kirnīno sell future3594 retains vowel-metathesis qualification. FullTurnerprose and explicit Palula source stem/inflection paradigms checked. Do not force all pasts either into independent ancestral participles or directly into present-stem ancestry; distinguish the evidence case by case.

Two additional holds: Palula dʰā̂tu/dʰē̂ti satisfied, since Turner6890 expressly doubts the r-loss in the *dhrāta comparison and mentions dhvasta.78 held records total. Helpers /tmp/palula-past-next.py and /tmp/palula-regular-pasts.py append proposals and must not be rerun blindly. Current733 proposals cover889 records, leaving2771 starting-snapshot records without a pending proposal. Review/dispositions regenerated. Palula180 and Ushojo324 existing-ID rows both passed current-overlay validation, temporary graph edge inspection and zero-change repeat application. Accepted overlay hash e77c151ce556143fb649cc194c01a2767f276fe22d4f6f0cff7118be116dbd13 preserved. No accepted-overlay edits.

## Research checkpoint September 10, 11:05 UTC (07:05 New York)

Added Palula #472–475 covering6 records: gūm/gī/gīa past go4008 (explicit suppletive paradigm); future bēm12225; stood utʰī̂tu1900-2 (dental reformulation *utthāti, not retroflex branch); reaped lūntu11082 (local T-class ending, qualified). Helper /tmp/palula-motion-reap.py must not be rerun blindly. Palula186 candidate rows passed current-overlay validation, temporary graph edge checks and zero-change repeat application; accepted hash e77c151ce556143fb649cc194c01a2767f276fe22d4f6f0cff7118be116dbd13 unchanged.

Fixed review-generator bookkeeping: /tmp/refresh-overnight-review.py now applies every explicit held case to dispositions, preserving pending-review precedence, and labels qualified reflexes Proposed inheritance from. Asserted all78 held statuses match. Current disposition counts:895pending-review;2687not-yet-reviewed;46needs-further-research;21source-check-required;11deferred-onomastic-research. These sum to3660; the last three categories are the78 held records. Total737 proposals cover895records;2765 lack proposals. Final handoff must keep not-yet-reviewed distinct from researched-but-unresolved.

## Research checkpoint September 10, 12:05 UTC (08:05 New York)

Added Palula #476–479: manī̂tu said9837 (abnormal sound change/alternative manutē retained), palī̂tu hid447 (direct sun-eclipsed comparator), urbʰī̂tu flew2038 (metathesis qualified), uṛī̂tu poured/let loose1697 printed subsection4. IMPORTANT: the existing persistent etymon ID for uḍḍāpayati is1697-3 despite its printed subsection4; candidate uses1697-3 with correct CDIAL[1697.4] citation. Initial builder assertion prevented any partial write; corrected after inspecting exact existing-node labels. /tmp/palula-final-past-research.py must not be rerun blindly.

Prepared overnight-research/handoff-counts.json for final review. Current741 proposals cover899 records: Sv42/54; Ush220/296; Phal479/549.252straightforward and489qualified proposals. Remaining2761 snapshot records consist of2683not-yet-reviewed and78held. By language unreviewed/held: Sv206/21, Ush222/11, Phal2255/46. Existing-ID candidates14+324+190=528 rows; only Ushojo and Palula have complete existing-ID candidate validation. Sauji components still depend on uninstalled donors.416proposals require donor nodes (36+21+359). Most Palula donor comparisons remain source-attributed;53 OPED comparisons and10 existing Urdu donors have independent checks; potato stem-only check is not full donor verification.

Checked current accepted overlay against every starting-inventory ID: zero have subsequently been accepted through the overlay in all three languages, so all3660 original remaining records still await accepted etymologies. The899 pending records are not saves. No accepted overlay or shared graph writes this run. The 09:00 New York final review is due at13:00UTC; next scheduled run should deliver an honest incomplete handoff and pause the existing heartbeat. Do not call remaining work exhausted.

## Final handoff September10,13:02UTC

Existing heartbeat successfully PAUSED via automation_update. All three overnight manifests marked pending-review; inventoryExhausted=false. Final review banner and counts added, unresolved tables grouped by language.741 proposals/899 records,252straightforward/489qualified;2683unreviewed+78held=2761withoutproposal. No overnight accepted saves. All3660 starting records still await accepted etymologies. Donor requirements416; Ushojo324+Palula190=514 existing-ID rows validated; Sauji14 have deferred donor dependencies. Do not rerun the temporary review generator blindly: it would replace this final handoff banner and grouped unresolved tables.

## User approved all741 proposals; partial save completed

All approvals persist. Canonical overlay has522 new rows covering467 records from323 proposals. By language: Sv4proposals/6records/8rows; Ush199/271/324; Phal120/190/190.418approved proposals remain:416 missing donor-entry preparations plus Sauji336–337 depending on sava. Approved outstanding records432. Total still unetymologised3193=432approved outstanding+2683unreviewed+78held. Save proof in approved-overnight-save.json; exact pending queue in approved-donor-queue.json; human ledger approved-save-ledger.md. All existing overlay rows preserved. Before hash e77c151ce556143fb649cc194c01a2767f276fe22d4f6f0cff7118be116dbd13; after16b8e77bef9d86fe089135db3ea4d2881c12b7451eebeb31192e03528f057ff9. Backup/tmp/overnight-before-approved-save.csv. Tempgraph/tmp/overnight-approved-verification-edges.csv.

Next donor preparation must follow SOURCE_INGESTION_CHECKLIST (read fully during approval turn), dictionary/glossary, website/CLDF, comparative-source addenda. Existing supported precedent: data/other/params/raw_data/kalkoti_donors.py and source_checklists/20260909-kalkoti-donors.md use curated selected donor heads in five-column parameter format with NFC preservation, audit and real build-issued IDs. Do not treat all416 donor requirements as independently verified dictionary entries; source-attributed comparisons retain that qualification. No new donor source was installed in this save. No identity edits, shared compiled-data replacement, commits or deployment. Approval already exists for all741; do not ask the user to approve again.
