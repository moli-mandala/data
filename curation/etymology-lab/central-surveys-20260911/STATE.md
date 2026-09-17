# Central surveys overnight research — September 11, 2026

User requests all supportable Malvi, Nimadi and Bagheli proposals for morning triage, with eight hours available. Start 07:30 UTC; deadline **2026-09-11 15:30 UTC / 11:30 America/New_York**. No intermediate approval requests. All new analyses remain pending; no accepted overlay writes. Do sustained research, not just scheduling/status. Prioritize Malvi, then Nimadi, then Bagheli; ensure a substantive first pass for each before deep hard-case work. No subagents requested.

Heartbeat `malvi-nimadi-bagheli-overnight-etymologies` is ACTIVE every half hour in this thread. At the deadline deliver the cumulative REVIEW.md and accurate counts, then pause this heartbeat. Preserve existing unrelated research, compiled CLDF and identities; no commits, deployment or DB rebuild.

## Scope and inventory

- Malvi: source `varghese-john-samuel2009malvi`, canonical language `mewari_basad`.
- Nimadi: source `vunnamatla-john-samuvel2012nimadi`, canonical language `Nimadi`.
- Bagheli: source `koshy2022bagheli`, canonical language `bagheli_lakshman`.
- Initial conversation counted 7,518 / 2,826 / 5,752 unlinked records respectively. Shared forms.csv changed during setup (mtime 03:32:40 local); a fresh snapshot excluding redirects contains 2,128 / 2,826 / 1,628. This is a concurrent rebuild/merging change, NOT research progress. Reconcile final persistent IDs and redirects before proposing assignments. Do not claim the reduction as etymologisation.
- `*-inventory.json` freezes this fresh snapshot, retaining all fields and locality tags. `*-gloss-inventory.txt` groups forms for inspection. `inventory-summary.json` records scope/counts.
- Malvi/Bagheli merged records can contain incompatible glosses (e.g. elbow + who, arm + seven); do not assign a single etymology to these without resolving source-defined homonyms. Hold ambiguous merged nodes. Raw rich-source files and identity registry can recover source record distinctions; pure review does not authorize reparsing or overwriting compiled data.

## Discovery and evidence

Read jambu-etymology skill. Homepage search is `fetchLemmaList` in `jambu-static/src/lib/query.ts`, with `params.form`, `params.gloss`, `params.word`, `params.relaxed=true`, `withOrigin=true`. Exact current implementation bundled to `/tmp/central-surveys-search.mjs` via `/tmp/central-surveys-search-build.mjs` with a read-only better-sqlite3 adapter; imports can call its exported fetchLemmaList. It uses current `.dbwork/jambu.db` and the actual query implementation, not an approximate search. Rebuild this tiny code bundle if code changes; never rebuild the DB merely for suggestions. The dev server on 127.0.0.1:5173 is in jambu-static; candidates API is complementary, accepts mode=candidates and forms=comma-separated persistent IDs (max100).

Verify shortlisted parents from full existing CLDF Etymology prose and CDIAL addenda, using cached CDIAL HTML if needed. Similarity is discovery, not evidence. Resolve exact persistent subsection IDs. Preserve immediate borrowing donors and morphology. Missing donor nodes remain explicit proposal dependencies; no installation required for triage. Pending manifests must carry exact record IDs, forms/glosses/tags/citations, evidence, relation kind, parents and assignment rows, with no unsupported blanket propagation.

## Deliverable

Create separate pending manifests by survey, with independent numbering, and cumulative REVIEW.md in the skill's four-column format (#, bold form with meaning, proposed etymology, substantive evidence). Group straightforward and qualified proposals; distinguish held/unresolved from unexamined records. Count etyma/proposals, records and assignment rows separately. Validate existing IDs and conflicts against fresh overlay in a temporary graph. At final deadline, honestly report any unexamined inventory.

## Checkpoint — initial research pass complete

All three surveys now have a substantive first pass. Pending batch-001 manifests are in canonical directories `mewari_basad`, `Nimadi`, and `bagheli_lakshman`; proposal numbering currently ends at **Malvi 20, Nimadi 19, Bagheli 17**. Next proposals start at 21/20/18 respectively; use batch-002 and later to preserve the completed batch.

**56 pending proposals / 319 records / 319 planned assignment rows**: Malvi 20/79; Nimadi 19/185; Bagheli 17/55. Difficulty totals: 37 straightforward, 19 qualified. Held 24 records; unexamined 6,232. `counts.json` and `dispositions.json` are the current exact record accounting. None accepted or saved to the overlay.

- Body-part and village families were discovered via the actual homepage query implementation, then checked against full CDIAL main entries and addenda. Selected primary articles and page numbers are in `cdial-articles.json`; exact DB parent nodes in `parents.json`.
- `REVIEW.md` and each `batch-001-review.md` contain the four-column tables. Primary citations now link to the exact DSAL dictionary page observed in the existing cached page HTML. A live DSAL page request was blocked by the web reader; the full existing cache supplied the evidence.
- Stable source snapshots were refreshed after persistent IDs reappeared. Frozen remaining inventories are Malvi **2,128**, Nimadi **2,826**, Bagheli **1,621**. `snapshot.py` guards against legacy-ID intermediate builds, but **do not rerun it blindly once dispositions exist**: preserve this baseline and reconcile new or changed IDs separately.
- The old app DB is differently merged, especially Nimadi (1,236 survey rows there). Its `*-db-inventory.json` files are discovery snapshots, not the authoritative working inventory. Do not replace the working 2,826 Nimadi inventory with that old DB inventory.
- Read-only app database backup: `/tmp/central-surveys-frozen.db`. `/tmp/central-surveys-search.mjs` now uses that frozen DB, which avoids changing underneath a research pass. Search still uses the actual current `query.ts`. `/tmp/central-surveys-discover.mjs` takes `form:gloss` arguments; it returns the first page, so broaden via pagination or a language filter if a sought hit is missing. Discovery logs are `/tmp/central-surveys-discovery{1,2,3}.jsonl`.
- `validate.py` passed on the current stable corpus: all **319 target IDs and all parent IDs exist**, every target is in the identity registry, no accepted overlay conflicts, temporary graph application **638 changes** (319 edges plus 319 status changes), repeat application **0**. No shared graph/CLDF/identity/overlay writes. Validation proof in `validation.json`. Evidence wording was subsequently refined without changing any IDs or assignments.
- `prepare-initial.py` is a one-shot materialisation script and intentionally refuses to overwrite existing batch-001 manifests. **Do not rerun it.** `render.py` is safe to rerun after new compatible manifests, and `validate.py` can validate the combined pending proposals. Both currently assume one reflex/loan assignment per record; extend explicitly if components are proposed.

## Next substantive work

Continue Malvi remaining gloss groups beyond the initial body parts: dwellings, tools, animals, plants, basic adjectives and verbs, then the same in Nimadi and Bagheli. A broad first pass over these shared basic-vocabulary sets should precede spending most of the remaining time on individual hard cases. Read `*-gloss-inventory.txt` and exclude IDs in pending proposals/holds; do not treat every leftover body-part response as researched. Full prose checks remain essential even for familiar words.

Known leads needing their own research: head `sir/ser` (find exact śiras entry); head/mouth `muṇḍ-/muḍ-` (CDIAL 10247 explicitly retains possible contamination from 10191 muṇḍa ‘shaven’); Persian/Hindi loan families `badan`, `dil`, `khun`, `jabān`, `nakhun` need verified immediate donors. Historical stem extensions can be proposed with a qualification, but do not invent a synchronic donor/base ID or silently assert inheritance where contact is plausible. Avoid classifying the survey’s lexical-similarity numbers as historical cognacy.

Continue substantial research every heartbeat until 15:30 UTC. At the deadline make the final review banner and counts honest, revalidate/reconcile against the latest stable corpus, and pause the existing heartbeat. The deadline is today, September 11, not tomorrow.

## Checkpoint — 08:00 heartbeat research

Completed four further substantive passes across all three surveys: household/tools (batch002), nature (batch003), food and common animal products (batch004), and animals/name (batch005). **Current pending total: 204 proposals / 1,264 records / 1,264 assignment rows.** By survey: Malvi **71 proposals / 337 records**; Nimadi **67 / 683**; Bagheli **66 / 244**. Difficulty: 136 straightforward, 68 qualified. **Held 64; unexamined 5,247.** All remain pending; no overlay changes.

Next proposal numbers: **Malvi72, Nimadi68, Bagheli67**; use batch006 next. `research_helpers.py` provides `Batch(number).add(...)` and `.save()` for explicitly researched decisions, refusing reused records or existing batch files. It now resolves requested parent IDs to the canonical `getLemma` ID. Do not rerun one-shot `prepare-household.py`, `prepare-nature.py`, `prepare-food.py`, or `prepare-animals.py`.

### Evidence and corrections

- Discovery used current homepage `fetchLemmaList` against the frozen app DB. Logs004–007 are durable here. Full shortlisted CDIAL prose/addenda appended to `cdial-articles.json`; selected exact parent nodes appended to `parents.json`.
- An explicit subsection audit found existing editorial extension nodes. **Corrected pending batch001 skin proposals to `4701-2` *carmaḍa-*.** Plain Bagheli cam was split out as batch005 #53 to root4701. This was a research correction, not a change to approved data.
- Fish l/r extensions use `9758-3` *matsyal-*; -ḍ- tail extensions use `8249-2` *pucchaḍa-*. These editorial subsection IDs differ from printed subsection numbering: cite the actual prose extension, not a fabricated printed subsection.
- **4147-2 is a legacy alias**, not an extant canonical node. Its real *gāvā* ID is `f_wb26lvatz7tcg`, confirmed by current form-id-aliases.csv and the returned getLemma object's ID. Corrected the two pending gau/geu proposals. The unambiguously y/i-bearing cows use4147-3 gāvī; gau/geu explicitly retain that alternative too.
- Gold proposals retain the source's expressly unresolved *suvarṇa / sauvarṇa* ancestry. Parent13519 is the shared dictionary-family reference; alternative13519-2 is recorded, with no claim that branch1 is proven.
- Name uses7067 nāman, **not7064 nāma ‘indeed’**, which appeared in broad substring discovery.
- New holds document unresolved immediate donors or source issues for rassi-type rope, Bagheli kapṛā cloth, suji needle, retained-d nadi river, suraj sun and gelū wheat. A held case is eligible for renewed research; it is not a claim that the etymology is impossible.
- The Bagheli package README confirms manual image transcription was authoritative and OCR only a scaffold; the normalization profile distinguishes t/d from ṭ/ḍ. Thus systematically retroflex-looking forms are retained with qualification rather than silently respelled as Hindi.

### Next priorities

A broad pass is still needed for **people/kinship, adjectives, numerals, pronouns and verbs** in all three surveys. Then revisit the remaining nouns and held donor families. Numerous ordinary body-part, food and animal variants remain unexamined: do not equate the number of gloss groups touched with exhaustion. Read the full inventories and per-record dispositions; select compatible records explicitly. For new derivation/compound proposals, extend helper/render/validation schemas deliberately and preserve immediate parent/component relations.

Further concrete noun leads: mango am/amba vs keri; rice cāval vs cokkā; chilli; ants; onion pyāj/goṇḍeli; paths, roofs, doors, firewood alternatives; h-initial Malvi outcomes. Existing donor heads may make the Persian/Hindi loan tranche productive after the broad native-vocabulary pass.

Current validation details are in validation.json. A check first flagged alias4147-2; this was resolved as above, and full temporary-graph validation was rerun. Always read the latest report rather than citing the earlier deferred result.

Validation confirmed after alias correction: all1,264 target IDs and selected parents present; no missing registry targets or current accepted-overlay conflicts; temporary graph first application2,528 changes (edges plus status updates), second application0. Accepted-overlay SHA256 remains405bb822c7298b600abd376f9d2e087588400369e288111f55ad108d05b8806b. Research remains active for the later heartbeats through15:30UTC.

## Checkpoint — 08:30 heartbeat research (08:31–08:55 UTC)

Completed batches006–010: **kinship/time, adjectives, numerals, selected simple verbs, and interrogatives/quantifiers/pronouns**. Current cumulative **400 pending proposals / 2,363 records / 2,363 planned assignment rows**. Malvi153/622, Nimadi124/1,301, Bagheli123/440. Difficulty after the new source correspondence audit: **237 straightforward, 163 qualified**. **Held74; unexamined4,138.** All remain pending; no accepted overlay edits.

**Next batch011; next proposal numbers Malvi154, Nimadi125, Bagheli124.** One-shot prepare-kinship.py, prepare-adjectives.py, prepare-numerals.py, prepare-verbs.py, prepare-pronouns.py must not be rerun. These completed builders document explicit word selections, full sources and alternatives. research_helpers.Batch.add now accepts optional `source_glosses=[exact full source glosses]`, allowing deliberately compatible merged senses without losing source records. The small `family` wrapper in prepare-pronouns.py selects only glosses whose semicolon-separated senses belong to an explicitly supplied allowed set; it does not infer semantic compatibility.

### New review/navigation and validation

- **TRIAGE.md** is now the compact morning entry point: by-survey batch links, proposal ranges, record counts and straightforward/qualified counts. REVIEW.md still contains the complete four-column tables and held cases. render.py regenerates both, with topic names for batches001–010. Extend the topic map for future batches.
- render.py now recovers a DSAL page from the CDIAL citation number when a parent has canonical f_ ID, fixing the previously generic gāvā primary link.
- validate.py now additionally checks frozen versus current target Language_ID/Form/Gloss, file-stat stability across the check, timestamp, and preserves **validation-last-passed.json** on success. Console output prints counts rather than thousands of missing IDs. Full details remain in validation.json.
- Current full successful check: **08:49:01 UTC**, all400proposals/2,363records, no missing current IDs, no changed target records, no missing registry IDs, no accepted overlay conflicts; temporary graph4,726changes, second0, inputs unchanged. Overlay SHA remains405bb822c7298b600abd376f9d2e087588400369e288111f55ad108d05b8806b. A preceding check encountered another temporary legacy-ID rebuild, but stable IDs returned and the full check passed. Later changes only strengthened evidence/difficulty for four Malvi proposals, not IDs/edges.

### Scholarly distinctions preserved

- Kinship: father9209.1 *bāppa, mother10016 mātṛ, sister9349 bhaginī, brother9661 bhrātṛ. Child chor- is5070.1 *chōkara, **not5072 ‘orphan’**. Son/daughter beṭ- is9238-2 *bēṭṭa. bālak has explicit learned-versus-extended uncertainty in9216; laṛkā family10924 retains10924-2 *laḍḍikka alternative.
- Adjectives: big moṭ- **10187-11 *mōṭṭa**, light halk- **10896-5 *laghukk- with metathesis**; cold13676.2 retains Dravidian influence and immediate contact uncertainty. New nav-6983 versus nay-**7025-2 naviya**; old jūn5260 versus purān8283. Good acch-142 explicitly notes Punjabi→Hindi transmission, so local proposals qualified.
- Numerals: one **2462-2 *ēkka** (expressive vs early MIA learned origin unresolved), three **5994-3 trīṇi**, four **4655-2 catvāri**, six **12803-3 *kṣaṭ / *kṣvaṭ** (explicit alternatives), nine6984 (not ‘new’6983), ten6227. Full articles/addenda checked. Some broad searches found wrong compounds/heads, so actual search was narrowed or cross-references followed.
- Verbs: eat present3865 vs past3865-2; sit **14327 upaviṣṭa** (explicit node; cite full2245 plus14327); give bare de6141 versus past *dita6140.3 (not proposed yet); fly1697-2 has *uddayati alternative explicitly allowed in addenda; walkshorta4715 versuslongā4721; runretroflex6624-2 extension; bhāg9361-2 participial origin; come āv **1200 āpayati**, NOT generic āyāti1288/āvrajati1451; go jā10452 versus suppletive gayo not yet proposed. Selected Nimadi comma-pairs contain only compatible simple inflections, with each component considered; auxiliary-bearing and negative expressions remain mostly unexamined.
- Pronouns: hũ/hāũ ‘I’992 aham, mẽ/mey9691 oblique ma- nominativization; tu5889, tum10511 remodeled yuṣmad, ham986 asmad. Who nasalforms2575 kaḥpunar; what kāy3196-2 *kādṛk; howmany3167 includes explicit -r-/-n- extensions without separate node; whatkind3197 analogicalformations. Quantifier sab13276 qualified because source expressly identifies Hindi contact in OMarwari; bhot9190 retains prabhūta influence.

### Local phonology breakthrough

**MALVI-S-H.md** and `malvi-h-correspondence-records.json` compare existing diplomatic SIL snapshot rows, not new ingestion. In Jesingpura, Bhunyakhedi, Jamli and Chandukhedi, the four independent families ‘dry’, ‘seven’, ‘hundred’, ‘hear’ consistently start h; Harsodan/Jhadmu have s. Exact source pages124/136/140/167 and source IPA are documented. This supplies local comparative evidence beyond distant CDIAL analogues.

Strengthened **Malvi #90 (h-dry), #114(hat seven), #115(ho hundred), #135(huṇ hear)** to straightforward with local-source evidence. Merged hāt ‘arm/palm/seven’ still held. Broad claim that every sibilant always changes to h is explicitly disavowed. Can use this local pattern to research other compatible h-forms, but verify each etymon and morphological branch independently.

A web search for general Malvi phonology found a primary Census PDF at https://censusindia.gov.in/nada/index.php/catalog/34830/download/38518/LSI_RAJASTHAN_PART-I.pdf but live web fetch timed out. No claims were based on the inaccessible PDF or Wikipedia. Actual correspondence evidence is the existing source snapshot.

### Next priorities (more than six hours remain)

1. Revisit **remaining unexamined nouns and adjectives**: many variants remain after each grouped pass. Do not equate touching a gloss with exhaustion. Especially full inventories’ kinship alternatives, mango, rice, chilli, ants, onion, paths/doors/roof/firewood, wet, short/nāno, big/baḍo, red, white/ujer, good/nīk, heavy/garu.
2. **Immediate donor families**: Persian/Hindi badan, dil, khun, jabān, nakhun, ādmī, aurat, mahīnā, sāl, garam, safed, zyādā, etc. Discover actual existing donor nodes first; verify primary dictionary entries. No new lexical installation required merely to state a missing-donor dependency, but accepted borrowed edges must not skip immediate donors. ExistingPlatts orHindi nodes may make this productive.
3. **Complex verbs and kinship compounds** now dominate many remaining rows. E.g. choṭo/moṭo/nāno+bhāi, adjective+ben, gharvālo/vāli, khā-l-/sun-l-/de-de, jā/gayo suppletion. Extend helper/render/validation deliberately for component links with ordered Pos; do not claim whole utterances are inherited monomorphemes. Consider whether existing same-language component nodes can be used before pointing to historical ancestors. Preserve explicit relation semantics and any dependent pending parent proposals.
4. **Demonstratives and remaining pronouns**, where forms often merge compatible number/gender uses but occasionally incompatible first-person senses. The current working inventories preserve full gloss lists; inspect those before grouping.
5. Known currently unresolved leads: sakala13066 primary prose has sayala forms but does NOT by itself explain sagḷ-/hagḷ- retained g+l; research the actual extended formation. ‘where’ kahā̃ discovery found existing Hindi donor f_dbddwpqujdwzu but no securely checked remote head yet. ‘husband’ dhaṇī6722 explicitly allows dhanika under dhanikā; inspect that before choosing. Do not automatically reuse initial broad-search root.

Discovery logs008–017 and full selected article texts are durable. Frozen DB remains `/tmp/central-surveys-frozen.db`; actual homepage query implementation `/tmp/central-surveys-search.mjs`, parents/evidence/sections scripts under /tmp as before. No rebuild needed for research. Continue substantive work at the next heartbeat; finish and pause the existing heartbeat only at15:30UTC.

## Checkpoint — 09:00 heartbeat (through 09:15 UTC)

Completed batch011 **Hindustani loans** (31 proposals/92 records) and batch012 **further nouns/adjectives** (23/121). Cumulative **454 pending proposals / 2,576 records**, 243 straightforward/211 qualified. Malvi172/682, Nimadi142/1410, Bagheli140/484. Held72, unexamined3927. Next batch013, numbers **Malvi173, Nimadi143, Bagheli141**. All pending, no accepted overlay edits. One-shot prepare-loans.py and prepare-extra-nouns-adjectives.py must not be rerun. Continue overnight through15:30UTC; heartbeat remains active.

### Loan scholarship and dependency

Actual query discovery018/019 preceded full primary Platts entries, read through Rekhta reproductions. Existing donor IDs cached parents.json: badan f_7tuydfvqt7ici; ādmī f_gsynajbspnc2y; ʻaurat f_vhi3eao4uvyjm; nāxun f_bgpkwqir4pwla; darwaaza f_prmeiidvi4gto; zabaan f_zqbwnodbqwsho; s̤abūt f_dbuiao2qkdgv4 (whole adjective, not proof noun). Hindi survey controls: dil f_snjuyk7aztjyu; khun f_nomhmgwrljcl4; mahina f_n4lkgn2ws3cvw; garam f_cdko5w5xxg5qg; safed f_lwa4hsrbk5gee. These control records anchor immediate Hindustani forms, not evidence that those control villages were historical donors. Transmission is qualified throughout.

PrimaryURLs are recorded per proposal. Rekhta keyword aadmii, aurat, naakhun, zabaan, darvaaza, saabit, mahiina, dil, khuun, safed, garm yielded full relevant Platts heads. badan Latin query failed; encoded Urdu بدن yielded the full Arabic-body head, distinct from Sanskrit-mouth homonym. saal returned other homonyms without the year head, so sal/year remains unproposed. Existing Hindi sal f_thgcjhj7vr5i2 and foreign sāl f_7a3ccguwm7lam are research leads only. Avoid citing the wrong homonym. Two nakhun holds reopened with old reasons preserved in reopenedHolds.

**Validation caught safed donor f_lwa4hsrbk5gee exists but Status=unlinked.** Malvi163 and Bagheli134 now explicitly have acceptanceDependency and evidence warnings: donor ancestry must be established before borrowed edges can apply. Do not bypass with a remote Persian parent. Could research a separate pending ancestry proposal linking this Hindi donor to existing Persian safēd f_tfolnhjem4qz6, but inspect actual primary evidence and schema first. All other selected donor parents are currently linkable.

validate.py now reports unlinked_parent_dependencies and dependency_blocked_rows, applies eligible rows to a temporary graph, and clearly labels a subset success rather than pretending complete validation. Latest validation was launched session37183; poll if necessary, inspect validation.json. Last fully successful checkpoint remains08:49,400/2363, until the safed dependency is resolved. No production graph changed.

### New noun/adjective scholarship

Discovery020 and full CDIAL prose cached for1268,4749,9875,3193,4386,1340,11225,12732,10539,4209,7150 (plus rejected false leads3475/4822/10882). Specific subsection nodes checked where applicable.

- Mango1268 amba/ām/ā̃b; keri NOT explained by3475kēsarin (only Marathi fibrous-mango adjective); investigate further.
- Rice4749 expressly *cāmala OR *cāvala, remote origin uncertain. Prakritcavala, Bhojcaur, Hcāwal/cā̃war, Gcāvaḷ. cokka/cokha NOT covered, still unexamined.
- Chilli c-forms9875-2 *maricca (explicit Bhojmaricā chillies, Gmarcī red pepper); Nimadi miri/mirin9875marīca branch, later semantic extension. No claim chilli was ancient referent.
- Ant kiḍi3193kīṭa with Prakritkīḍī/kīḍiyā insect/ant; citi forms NOT supported by4822cimb pinch; research them separately.
- Wet gilo4386*grilla; deeper*gr̥dla explicitly very doubtful. Alo1340-2*ālla<*ārdla. l-initial lilo and bhij- forms not covered.
- Big baḍo11225vaḍra: addenda extraction from evaḍa/kevaḍa with-vant+-ḍa, not straightforwardvṛddha. Red rāto10539rakta; retroflexrāṭo separately qualified. Bagheli garu/geru4209guru qualified.
- Bagheli mergedwhole/good nikaha/nikeha7150nikta has explicit MIA*nikka and Mthnikāh good; whole meaning/endings uncertain; Hnekā Persian influence acknowledged. No distinct *nikka node in section query.
- Full12732ślakṣṇa has Pklaṇha, nannha/nānũ etc relevant short/small nānā/nāno, but not yet selected/proposed. 10882lakṣaṇa was false good-search hit.

### Presentation changes

research_helpers accepts custom citation/source_url for loans and records parentLanguage/primarySourceURL. It now preserves sourceSenses; if broad grouping label only selected one actual source sense, display the precise sense and preserve groupingLabel. Retroactively narrowed9 labels in pending manifests, no IDs/edges changed. render.py now computes relation kinds dynamically, supports topics011/012 and flags validation dependencies in footer; all reviews regenerated.

### Next research priorities

Continue remaining inventories; many records remain even within touched glosses. Strong next targets: ants citi/cihuṭi; rice cokha; mango keri; onion pyaj; white ujer; small/short nana; kinship manak/maney, lugai, dhani; body deh/sarir/tan; bloodlohu; heartkaljo/hirday; roof and doors; merged compatible adjective+kin compounds and complex verbs. Need exact parent/component relations, no whole-utterance monomorpheme shortcuts. Existing donor audit data/data/other/params/raw_data/20260911-mewari-donors-audit.json contains wazn/weight, wazandār, hafta, kam, mard, kamzor etc additional useful installed heads. Full remaining forms are in per-language inventories/dispositions; consult those rather than guessing spellings.

## Checkpoint — 09:30 heartbeat

Completed batch013 **body/adjective remainders** (25 proposals/94 records) and batch014 **ant family plus tan** (4/15). Cumulative **483 pending proposals / 2,685 records**, 247 straightforward/236 qualified. Malvi183/716, Nimadi151/1454, Bagheli149/515. Held72, unexamined3818. **Next batch015; next numbers Malvi184, Nimadi152, Bagheli150.** One-shot prepare-body-adjective-remainders.py and prepare-ant-family.py saved; do not rerun. Reviews/counts/dispositions regenerated. Validation launched session75504; latest report should cover483/2685. Same safed donor dependency affects6rows, no accepted overlay mutation.

### Further primary-source findings

Discovery021–025 actual homepage queries persisted. Full cached CDIAL entries checked and parents resolved:4918,6557,12335,11165,3103,14152,1670,1671,1672,12732,5071,6767,4822,5656; leads2804,2805,6722,9828 also read.

- **cokha rice4918 cōkṣa**, Prakritcokkha clean, Sindhicokho cleanedgrain, Gujarati pluralcokhā rice. Not4749cāvala. DeaspiratedandMalvisibilantvariantqualified.
- **lohu11165lohita**, with contractedloi/liquidvariantsactualcomparanda; Malvi loi/luiqualifiedbecause distant similar contractions do not establish local soundlaw.
- **kaljo3103kāleyaka** explicitly discusses heart/liver confusion, Hkarejā, Gkāḷjũ. Preserve elicitedheartgloss. **deh6557** Bagheliinitialḍqualified. **sarir12335** learned/reborrowed possibilityexplicit; notpresentedasproveninheritedwholechain.
- **hiyo14152hr̥daya** NimadicomparesMarwhīyo directly. Retainedd/ronraday/hirdeyvariants separatelyqualifiedlearned/restored/contactfamily; ordinaryhiyāevolutiondoesnotexplainthosebyitself.
- **whiteujer1670ujjvala** explicitlyAwadhiujar/Mthujjarwhite; **ujiar1672*ujjvāra** qualifiedwith1670alternativeandandhakārainfluence. Not1671verbmerelybecauseexistingattestationlinkedthere.
- **smallnāno12732** selectedordinaryNimadi/Malviformsplusseparatequalifiedmergednano/nānochild/son/youngerbrother/short/smallsenses, withMthnanuāyoungchildcomparandum. Mergedcompatiblechoṭshort/small5071added; phoro withlightleftout. deaspiratedwhite6767qualified; merged doro thread/white excluded.
- **Ant breakthrough:** CDIAL4822aloneonlypinch/pincers, but fullPlattsreproductions supply bridge: https://www.rekhta.org/urdudictionary?keyword=chimtii gives cimṭī pincers/pinch/ANT and crossrefersto cīṅṭī/cyūṅṭī; https://www.rekhta.org/urdudictionary?keyword=chintaa has cīṅṭīsmallant andcīṅṭālargeantcrossreferencedtocimṭā/cimaṭnā; https://www.rekhta.org/urdudictionary?keyword=chyuu.nte gives cyūṅṭī/ciʼuṅṭīdimofcyūṅṭā. These support **qualified** citi/ciuti family proposals4822, not a uniquely reconstructed formation. Bagheliinternalh/extensionsandnasallossopen. CrossrefchūṅṭīhasalternativePlattsPrakritcuṇṭiā/Sanskritcuṇṭa, so deeperconvergencepossible. ThreeantproposalscitebothCDIALandPlattsandlinkprimaryPlatts.
- Mango keri still unresolved: Platts https://www.rekhta.org/urdudictionary?keyword=karii giveskairīunripemango withkarīra+ikā, butCDIAL2804karīrashoot and2805karīraCapparishavenomango. CDIAL3475kēsarinfibrousmangoadjectivestillinsufficient. Do notsimplycopyexisting3475links. SeveralRekhtadirectcase-sensitiveURLsfailed; lowercaseindexedURLsworked.

### Fresh next leads

1. **lugai** actualqueryfinds existingfamilyhead **f_k66mptwk5okq6 *lugāī ‘wife’**, cachedparents. Source `arora` personalcommunication2021, nofullprose, 66reflexes14languages. Canpotentiallyproposefamilymembershipifverifyactualattestationsandsource provenance; do NOT inventSanskritancestry. Sourcefilesdata/other/forms/20220913-thari.csvand20230403-arora.csv. Thisisexistingfamilyheadnotnewdonorinstallation.
2. **manak/manakh/maney**: manukhquery9828manuṣya, fullarticlecheckedBUTmainprosehasmaṇussa/manus/muns etc, notmanakh. Needreadadjacent9827manuṣa and10049mānuṣa andtheiraddenda to findkhaforms. Don'tuse9847mandākṣafalsehit.
3. **dhaṇi husband6722** fullarticleexplicitlysaysH/Ghusbandpossiblydhanika s.v.dhanikā. Querysection6722showsnoown-dhanikanode; researchdhanikābeforechoosing. Fullprosealreadycached.
4. Remainingnouns,verbs,compoundkinshipandpronounsasprevious. Six hoursremainuntil15:30UTC; ensure substantive continuedwork next heartbeat.

## Checkpoint — 10:00 heartbeat

Completed batch015 **household/kinship remainders** (19 proposals/74 records) and batch016 **nature/time remainders** (26/124). Cumulative **528 pending proposals / 2,883 records**, 249 straightforward/279 qualified. Malvi202/786, Nimadi165/1544, Bagheli161/553. Held72, unexamined3620. **Next batch017; next proposal numbers Malvi203, Nimadi166, Bagheli162.** One-shot prepare-household-kinship-remainders.py and prepare-nature-time-remainders.py saved, do not rerun. Reviews/dispositions/counts regenerated. Validation current run session63294; previous502/2759 fully ID-valid,2753eligible graph rows passed and six safed-dependent rows excluded. No accepted overlay mutation. Continue through15:30UTC.

### Specific research decisions

Actual homepage discovery026–030 persisted; full selected CDIAL prose and addenda read/cached, exact nodes resolved.

- Existing *lugāī family f_k66mptwk5okq6 is `arora`, not CDIAL. Primary local file data/data/other/forms/20230403-arora.csv line7 H,e54,lugāī,woman/wife,sometimespejorative. Proposed qualified Malvi/Nimadi modern comparative-family membership with explicit statement this is not a Sanskrit reconstruction. Source link points to actual absolute local source file. No new root installed.
- dhaṇi husband6722 explicitly contrasts owner/master development with **6721dhanikā/dhanika**, Sanskritization of Prakritdhaṇia praiseworthy <dhanya. Both preserved, not falsely unanimous.
- Roof chat4971*chatti explicitly Punjabi→Hindi contact; chanifamily4992*channi, Bihchānh/chānhī/chānhiyā. Baghelicaṇḍi/chaṇḍhistop-bearingformsstillunexamined. Taproof **5725-3*tarpar-**, entry -r extensionwithPṭapparā/ṭapparī andHṭāprāthatch. Jopəḍi5403-3*jhōppa, qualifiedvsnasalbranch5403-4/roof-hutmetonymy.
- Malvi/Nimadibārṇu/bāiṇudoors **6663dvāra** addendaG bārṇũ, Kachchibāyṇo. **Not11553vāraṇa**, whichonlysupportsdoorstep. Bagheliḍuar/ḍuari **6459*duvāra**, not6651*dvaraorbare6663; exactexpandedbranchread.
- Bodyaŋg114.1; narī7078; istrī13734/pati7727familyqualifiedlearned/contactbecause ordinaryMIAloseclusters/stops. Baghelimaney/menayetc **10048mānava, specific*mānavikaformation** withBhoj/Aw/Hmanaī (noindependentchildnodeexists). Not9828manuṣya.
- Manak/manakhremainunresolved. Full9827manuṣa,10049mānuṣa and9828manuṣyareadbutnoneexplicitlyexplainkh; don'tcopyexistinglinksblindly.
- Morning saver13291*savēla (explicitb/vdiscussion); Malvihaverqualifiedusingestablishedlocalh/scorrespondencebutnotyetcheckedmorningbylocality. Bhinsar **11813a*vibhāniḥsāra** exactaddendum, not11813. Bhor9634*bhōrā.
- Mudkicaḍ3153.1*kicca, expressivefamily/Dravidiancomparisons; gara4137*gāra mud/mortar. Rootjaḍ5086.1jaṭāfibrousroot; treejhāḍ5362.1jhāṭa, bothpossibleDravidian/Mundaremoteroutespreserved. Don'tcollapse rootandtreebecauseformslookalike.
- Pathvāṭ/bāṭ11366vartman with **11363vartis explicit alternative**; windpon/pavan7978.2 withlearned/contactquestionforconservativeforms. Rainbarsat11398varṣārātri **alternativeorcontaminationvarṣartu** explicitlypreserved.
- Full11392varṣaand11396varṣāread; barkha/varṣarainstillnotproposedbecausekhcorrespondencenotresolved. Yearbarascanbeproductive11392.2 withdirectOAw/Marw/Gcomparanda, butnotselectedthisrun.
- **gail/path** actualqueryfound **4009-2**, but stored parent word `*ga{l}lati` appears incorrectly parsed. Full4009gati entry clearlycallsHgayal/gail/galī,OMarwgailo/galo the **-(l)la extension**. Do notpresentbogusreconstructionasifprimary. Couldpendingproposalexplicitlynamecorrectextensionwithparent-labelissueflag, orleaveuntilsourcecorrectionreview. No sharednodeschanged.

### Remaining productive inventory

A counts scan shows many entire simple families still unexamined; do not focus only hardcases. Discovery030startedgoodnewpass. Examples:
- Skyakāś/āsmān/vādal; rainbarkha/comāsa/bāriś; windvāyiro/beyar/havā; rootBagheliḍar/sor; treepeḍ/birba/rukheba; broomuvāri/bāyri/jhāḍu/kūca; ropedoḍo/dori/rāśi/lejurī; pathrasto/gail/mārg/pagḍanḍi; morningearlycompoundssunrise.
- Malviunexaminedhighcounts: thirsty36,sit34,burn34,lie33,bite28,run25,same22,eat22,give22,come21,broom20,rope20,different20,broken20. Manyarecomplexfinitepairs, requirecomponent-levelanalysisnotbarestemshortcut.
- Nimadi: rainbow19,noon18,mud16,same16,few16,sky15,rain15,eggplant15,morning15,when15,knife14,rope14,ring14,lightning14,wind14,path14,sand14,tree14,root14,cabbage14,oldersister14,day14. Somecountsreducedby016.
- Bagheli: lie25,millet24,youplural19,lightning18,broom17,path17,morning17,same17,bite17,burn17,few15,sleep15,ring14,broken14,thirsty14,many13,hungry13. Wholemeaninggroupsnotexhausted.

Resume inventories/dispositions eachrun. Needeventualcomponent/derivedschemaextensionforcomplexverbsandkinshipcompounds; donotquietlydiscardthem. Donordependency6rowsremainsapproved-reviewdependencyonly, userhasnotacceptedanything.

## Checkpoint — 10:30 heartbeat

Completed batch017 weather/rope/tree (22 proposals/76records) and batch018 ordered kinship compounds (16proposals/46records/**92component links**). Cumulative **566 pending proposals / 3,005 records / 3,051 links**,253straightforward/313qualified. Malvi217/824records/835links; Nimadi174/1585/1609; Bagheli175/596/607. Held72,unexamined3498. **Nextbatch019; numbers Malvi218,Nimadi175,Bagheli176.** One-shotprepare-weather-rope.py,prepare-kinship-components.py; donotrerun. Reviews/counts/dispositionsupdated.

Validation **10:35:23UTC**:566/3005/3051,missingIDs0,changedrecords0,registrymissing0,overlayconflicts0,stableinputs. **3045eligiblelinks passed temporarygraph**,6044changesfirstapplication,0second. SixrowsstilldependonunlinkedHindi safed donor; noacceptedoverlaymutations. SHAunchanged405bb822c7298b600abd376f9d2e087588400369e288111f55ad108d05b8806b.

### New scholarship

Discovery031–032 actualhomepagequerylogs. FullCDIAL1008,11567,11497,11544,6225,10648,12060,3408,5328,10582read/cached. Exactparentsectionsresolved.
- Skyākāś1008qualifiedlearned/contact; vādal11567cloudfamilyskysemanticextensionqualified.
- WindMalvivāyiro11497.1vātara with11497.3alternative; Baghelibeyar11497-3*vātāra specifically Pkvāyāra,Hbayār; h/eoutcomesqualified. Malvibāvo11544.1vāyuwithvātaambiguityexplicit.
- Ropeḍori6225davara, l/retroflexoutcomesdirectlyattested. Ras/rāśṛī10648raśmi, explicitPunjabi/HindicontactandMalvi-s/hqualification. Lej/lejurī10582rajju/PklajjuwithBhojlajurīwellrope.
- Bagheli kūca/kuche broom3408kūrca, Bihkū̃cābroom; possibleDravsourceandaspirationvariantsqualified.
- peḍtreeexistingfamilyf_rr7dv53h3a5pm *pēḍa citedarora+bundeli. Sourceextensions_ia.csv e53; actualBundelipeɖtree rows4911ffdata/other/forms/20230522-bundeli.csv. Qualifiedmodernfamilymembership, notSanskritreconstruction. Baghelibirba12060vīrudha, Hbirwāsmallplant; planttotreequalified. bircaexcluded.
- Yearbaras11392.2withdirectOAw/Marw/Gforms; conservativevarśandcontractionsqualified.
- jhāḍubroomresearchnotyetproposal:5328fullarticleverbfalls,branch5328-2*j hāṭayati shakesdown/sweeps; needsnominalformationevidence, don'tlinknounstraighttobareintransitiveverb. Similarityhit625arkaunrelated.

### Component schema now operational

`component-anchors.json` records exact existing same-survey lexical IDs, forms, pending proposal numbers and citations. Anchors fetchedthroughgetLemma, fullprimaryancestryalreadyresearched. `prepare-kinship-components.py` builds explicit manually selected compound groups only; eachproposalhas components[{id,form,position,proposal,citation,...}], acceptanceDependencies, twoorderedcomponentassignmentsperrecord. Preservesordinaryfeminine/numbervariantsandexplainssemanticissues. E.gNimadibeiṇanchor elicitedoldersisterusedinsideyoungersistercompound: explicitqualificationthatageissuppliedbymodifier. Noassertionthatanchorvillageishistoricaldonor. Malvinanoanchorhasmergedchild/short/smallsensesandqualification. AmbiguousBaghelibheybothsexesexcluded.

`render.py` renders **both component IDs, Pos1/2 and separate primary citations**; TRIAGEoverallnowreportslinksseparatelyfromrecords. `validate.py` allowsmultiple rowsforoneForm_IDonlywhenallcomponent,contiguousuniquepositionsanddistinctparents; checksPosinthetemporarygraph. Existingvalidate_assignmentsresolvespendingancestrychains, soorderedcomponentscanbereviewedalongsidependingcomponentanalyses. No root/helperbroadautoanalysisintroduced.

Next work: expandcompoundbrotherswhereNimadiunmarkedbrotheranchorstillmissing (don'tborrowMalvianchor); remainingkinshipcouldfindexistingNimadilinkednodeoutsidefrozenunlinkedinventory. Complexverbresponsesneedcarefulpercomponent/inflectionanalysis—currentcompoundbuilderpatterncanbeadaptedbutdoNOTtreatcommaseparatedalternativeformsasconcatenatedcomponents. Manysimplenouns/loansandadjectivesstillunexamined. Continue substantivework until15:30UTC.

## Checkpoint — 11:00 heartbeat

Completed batch019 adjective/simple-burn remainders (19proposals/74records) and020 simplepasts(7/14). Cumulative **592 pending proposals / 3,093 records / 3,139 links**,258straightforward/334qualified. Malvi228/850records/861links;Nimadi183/1620/1644;Bagheli181/623/634. **Nextbatch021; nextnumbersMalvi229,Nimadi184,Bagheli182.** One-shotprepare-adjective-verb-remainders.py,prepare-simple-pasts.py. Allpending,noacceptedoverlayedits.

**Researched-unresolved separation:** record-researched-holds.py moved32investigatedcasesintoholdswithspecificreasons:keri/kairimango(contradictoryetymologicallead),manak/manakh(khnotexplained),gail/path4009-2malformedparentlabel,jhāḍubroomnominalformation,barkharainkh/contactroute. Theyarenotruledout;canreopenwithbetterprimaryevidence. **Held104 total, unexamined3378** (Malvi41/1237,Nimadi31/1175,Bagheli32/966). Donotre-runholdscriptblindly; pendingrecordsareexcludedbutoldreasonhistoryshouldbepreservedwhenreopening.

Validation **11:05:07UTC**592/3093/3139:missingIDs0,targetchanges0,registrymissing0,overlayconflicts0,stableinputs.3133eligiblelinksgraphpassed,6220firstchanges,0second;6safeddepsunchanged. Reportvalidation.jsoncurrent;allreviewscountsdispositionsregeneratedafterholds.

### Primary-source decisions

Discovery033–035 homepagequeriespersisted. FullCDIAL13720,6098,700,404,13119,6065,13841,13845,6654,5306,5308,13211,4711,4008,6140,1045,1200read/cached, specificnodeschecked.
- Fewthoḍ **13720-2 -ḍ extensionofstoka**, not6098thuḍatreetrunk. -kaextensionsqualified, -soexpressionsleftforcomponentanalysis.
- DifferentNimadialag/Baghelieleg700alagna; nyare404*anyākāra. Reduplicatedalagalag/nyāranyārāstillunexamined; shouldbesynchronicderivedreduplicationwithimmediatebase, nottworepeatedcomponentedgeswhichoverlaypairkeymaynotpreserve.
- Sameharika/sāriko13119sadr̥kṣa, Malvih/scorrespondencequalified; samān13211samāna conservative/learnedquestion. ek+adjectivecompoundsnotyetproposed.
- Brokenṭuṭ6065 explicitlyincludesPrakrittuṭṭaparticiple; phuṭ**13845sphuṭyati**, entryincludesphuṭṭa‘burst’. **Not13841sphuṭa‘clear/open’**. Noownparticiplechildnodefortheseinquery;citationsnameparticipialpassage. Regional-l/-ehaadjectivalendingsqualified.
- SimpleMalvibaḷiyo/baḷioburn6654*dvalati; Baghelijeleṭh/jeleṭhe/jereṭa5306jvalatiwithtentative-tparticiple/habitualmorphology. Auxiliary-bearingstringsnotincluded.
- Malvigyo/gayo4008gata, actualgo-preterite, notyātipresent. āyo/ayā1045āgata, not1200present. Retained-vāvi/aviya/āviyāqualified1200āpayati regularizedforms; exactgender/tenseopen.
- Malvidiyā**6140-3*dita**; didoqualifiedsamebranchwith-ddhaparticipialreplacement(OHdīdhau,Gdīdhũ), alternative*dittaexplicit. Repeatedde-deandday-diyostringsexcluded.
- Nimadi`sav, soyo` inbothsleep/lie-downmeaninggroupsproposed13902: provisionallytwoalternativeordinaryinflections, notconcatenatedcomponents. Compatiblelexicalfamily,localvowelsandtenseopen.

### Next leads and schema cautions

Bitecabfamily4711primaryexplicitlychew/bitewith-vv-/-bb-/-bbh-/-mb-alternants. Specificnode4711-3islabelled`*carvabbati`, whichlooksparser-syntheticratherthanprintedreconstruction(similar4009-2`*ga{l}lati`). Avoidpresentingthisasprimary;couldrecordlabelcorrectiondependencybeforeassigningfamily.

Many finite compoundresponsesremain; specimensareinexactinventories and discovery. Go`jā,gayo`combinesordinarypresent+suppletivepastascomma-separatedelicitationanswers, **notconcatenatedcomponents**. Existinggraphmayrequirekeepingtheseasreview-onlyunresolvedratherthanfalsecomponentlinks. In contrastgoauxin`caligayo`hasgenuineorderedpartsifactualsame-languagebase/auxnodescanberesolved.

RemainingproductivewholefamiliesstillincludePersian/Hindidonors(asmān,havā,barābar,kam,sāl,subah,rāsta,pyāj etc), instruments/foods,many/fewexpressions,numeralsnotyetallcovered,kinshipnāri/bāi/dādā/jījīvariants,sametypes,andwhen/where. Prioritizeprimaryresearchratherthanautomatingapproximateassignments. No needtoaskuserorinstallnewnodesforpendingdependencies. Continue through15:30UTC.

## Checkpoint — 11:30 heartbeat

Completed batch021 **further Hindustani loans**,17proposals/60records. Cumulative **609 pending proposals / 3,153 records / 3,199 links**,258straightforward/351qualified. Malvi233/863records/874links;Nimadi189/1659/1683;Bagheli187/631/642. **Nextbatch022; nextnumbersMalvi234,Nimadi190,Bagheli188.** One-shotprepare-more-loans.py;donotrerun. Currentheld108,unexamined3314:Malvi42/1223,Nimadi32/1135,Bagheli34/956. Addedfourheldsubahmorningrecordswithmissingimmediatedonorreason. Noacceptedoverlayedits.

Validation11:33:06UTC609/3153/3199:missingIDs0,changedrecords0,registrymissing0,overlayconflicts0,stableinputs.3193eligiblegraphlinkspassed;6340firstchanges,0second. Same6safeddependencyrowsremain. Reviews/counts/dispositionsregeneratedafterholds. Continue through15:30UTC.

### New donors and verified primary entries

Discovery036–039 useactualhomepagequery. New`/tmp/central-hindi-donors.mjs` runsactualfetchLemmaList modelexicon languageIdH andparamsrelaxedtrue. **Spelling variants matter:** barabarqueryreturnsnoHbecauseinstalledheadspelledbaraabar; followingPalulacomparisonfoundcorrectHhead. Donotmistakesearch0forproofabsencewithoutfollowup.

SelectedexistingHdonorIDs(incachedparents):
- **f_alir5ifyzfzvq asman sky**,Kannaujicontrolp64; **f_ra6uvjvpsrlbe hava wind**,p66; **f_44kuz3f4pxo5e rasta path**,p67; **f_22utxbhjyd5lu pyaj onion**,p73; **f_thgcjhj7vr5i2 sal year**,p85. AllareactualH-languagecontrolrecordsandcurrentlylinkable.
- **f_xitegx7mbquyi baraabar equal/healthy/justright**,HdonorheadfromLiljegrenentryLX000263; **f_ymjgts524mota kam less/inferior**,HdonorLiljegrenentrykam. Bothcurrentlylinkable. f_clan4z6d5wyky-stylefewcontrolnotused;donotguessID,lookupifneeded.

FullPlattsprimaryreproductionschecked:
- āsmān/asmān https://www.rekhta.org/urdudictionary?keyword=%D8%A2%D8%B3%D9%85%D8%A7%D9%86 (fullskyheadandidioms),Persianorigin.
- hawā https://www.rekhta.org/urdudictionary?keyword=havaa andindexed.uhva page,air/wind,Persian-mediatedArabic. SourceArabicdesirehomonymnotconfusedwithwind.
- rāstā https://www.rekhta.org/urdudictionary?keyword=raasta **explicitHindustani rāstā forPersianrāsta**;don'tuseonlyadjectiverāstdexterous.
- piyāz https://www.rekhta.org/urdudictionary?keyword=pyaaz onion/leek; actualHindi pyaj suppliesaffricateadaptation, Baghelipiyaj/uformsqualified.
- **Sāl year finally resolved**: RekhtaLatin/Urduquerieskeeppresentingthorn/house/treehomonyms. FullcorrectPlattsentryat **https://urdu.hawramani.com/%D8%B3%D8%A7%D9%84-4/**, followedobservedentry30357linkfromhomonymindex; Persianyearwithyearphrases. Notusingforumquotationasprimary. Allthree sāl groupsnowproposedagainstHsal, notPersianancestor.
- barābar https://www.rekhta.org/urdudictionary?keyword=baraabar explicitlysame/equal. Hwholewordborrowproposal, notinventedlocalsynchronicbar+barcompound.
- kam https://www.rekhta.org/urdudictionary?keyword=kam little/less; distinctfromkāmwork. Nimadikamfewproposed.

**Subah morning held4records**: Plattsṣubḥvulgṣubaḥprimaryreproduction https://urdu.hawramani.com/%D8%B5%D8%A8%D8%AD/ confirmsdawn/morning (sitehasbrokenromantransliterationdiacritics). ActualHqueries subah/subh/subhaa/subaa/subha foundonly unrelated sũbhāironinstrument andsubhānāmakebeautiful. NeedimmediateH/Urdu donor record;noArabic-bypassingedgeandnonewnodeingestion. Theseareinvestigatedunresolved, notunexamined.

### Next productive steps

Stillremainingphysicalnouns/foods/kinship/locativesandfinitecompounds. Repeatedadjectivescanuse`derived`withanexistingimmediatesame-languagebase(e.gNimadialag→ālagalag,Baghelieleg→elegeleg), keepingreduplicationinferred/qualifiedandbaseproposaldependency. **Do not use two identicalparentcomponentrows**:overlaykeyForm_ID/Etymon_IDmaycollapsepositions. Transparentek+sameadjectivecompoundsalsohavepossibleexistinganchors, butinspectmergedsenseconstraints. MissingNimadibarebrotheranchorcanremainexplicitdependency.

Needlatercomprehensiveauditofproposalquality/numbering/counts,notjustadditions. Fullreviewhas600+proposals;usercantriagebyseparatesurveyandnumbers. Keepallpending,noapprovalquestionswhileasleep.

## Checkpoint — 12:00 heartbeat

Completed batches022–023: **30 additional proposals / 171 records**. Cumulative **639 pending proposals / 3,324 records / 3,370 links**,258straightforward/381qualified. Malvi244/910records/921links;Nimadi196/1736/1760;Bagheli199/678/689. **Nextbatch024; nextnumbersMalvi245,Nimadi197,Bagheli200.** One-shotprepare-food-instruments.py and prepare-knife-noon-ring.py; do not rerun. Currentheld108,unexamined3143:Malvi42/1176,Nimadi32/1058,Bagheli34/909. Noacceptedoverlayedits.

Validation12:08:02UTC639/3324/3370:missingIDs0,changedrecords0,registrymissing0,overlayconflicts0,stableinputs.3364eligiblegraphlinkspassed;6682firstchanges,0second. Same6safeddependencyrowsremain. Reviews/counts/dispositionsregenerated. Continue through15:30UTC.

### Resolved food and instrument families

Discovery040–046 fromactualhomepagequerypersisted. FullprimaryCDIALprose/addendaandparentnodeschecked.
- Eggplantbhaṭṭāfamily9369.1bhaṇṭākī (Biharibhaṇṭā,Maithili/Awadhibhā̃ṭā),qualifiednasalloss/gemination. Baghelibeygen11503.1vātiṅgaṇa (Pk vāiṁgaṇa,Hbaigan). **riŋgaṇfamily still unresolved**,11503fullprosedoesnotexplainr-initial.
- Malvijuār/juvārmillet10437yavākāra (Prakritjuāri/jōvārī,OldMarwarijuvāri). Baghelibajer-family9201*bājjara;addendarejectsunsupporteddeepvarjaraderivation. Baghelikoḍou/koḍoua3515kodrava. Preservebroadsource‘millet’,noinferredprecisebotany.
- curiknife3727,ch-branch;Turnerexplicitlydiscussesdialectdiffusion. mundaḍi/mundiring10203mudrā,possibledeepIraniandonorandlocalextensionsqualified.
- Sandret10816retra,limitedtrevidenceexplicit; bālu/bāru/vāḷu11580vālukā(noun,not11579adjective).
- Lightningbijli/bijurī11745vidyullatācompound. Unusualbijkliandijəḷiqualified. **Malvivijəḷāvnotselectedbecausebuilderhadbijəḷāv**,remainingcandidate;don'tsilentlyassumealreadycovered. Barebij/vijandgajnotincluded.
- Malvimortaruŋkaro/ũkhrā/uŋkiḷi2360-4*udukkhala, exactukkhala/okkhala/okharbranch. Nimadikhāyṇo unresolved; don'tguessfromkhādati.

### Knife, noon, finger-ring

ExistingHdonor **f_xqxbqhxafwpsu caku ‘knife’**,KannaujisurveyHindiSarhaticontrolp62;Plattsćāqūprimaryathttps://www.rekhta.org/urdudictionary?keyword=chaaquu explicitlyPersianclaspknife/penknife. Allthreecakku-familyborrowedproposalsqualifiedgemination/localvowelsandMalvis;nomediationbypasstoPersianf_f5yhi6kl6jazu.

Noonexisting**f_jg7fkl55nyhjm *dva-prahara ‘secondwatch;noon’**,Aroraentry. FullPlattsdoentryathttps://www.rekhta.org/urdudictionary?keyword=ro includesdo-pahar‘noon’with**dvi-prahara**explicit,do-paharī/do-pahrīanddu-pahriyā. Threequalifiedfamiliesproposed; **dva-vs-dvireconstructiondifferenceisexplicitreviewissue**,alongwithcontraction/aspiration/Bagheliretroflexinitialsandinheritancevsdiffusion. Notfalselyclaimedexactprintedreconstructionmatch.

Homepagequery`anguthi:ring`returned0while`aŋgu:ring`found138and**138-2x**. FullCDIAL138/14203read,**138.2*aṅguṣṭhiya**specificallycontainsBihari ãguṭhī,Awadhi/Hindi ãgūṭhī,Gujaratiãguṭhī. MalviandBagheliangutiringproposedagainst138-2x,notbare137thumb;qualifieddeaspiration/initiale.

### Active next leads

- Malvibiṭi/viṭi/beṭringandbinḍi/vinḍi: discovery044leads12045vīṭā. Fullverylongarticlealreadyread/cached;*vīṭṭasubfamilygivesGvīṭīring,*vĭ̄ṇṭagivesGvĩṭīring. Broadround/rolledfamilywithb/v,geminate/nasalalternationsandpossibleAustroasiaticconnections. Noindividualsubnodesfoundbysectionshelper (only11713viṭapa/12077vṛntaderivatives). Needdecidewhetherqualifiedspecificlocatoron12045adequateorholdforparentgranularity;donotpresentbaretipcatheadasdirectringmeaning.
- Baghelijoneri/joṇṇerimilletstillunresolved. Couldqueryyavanālafamilyhomepagebeforeprimary10438;noguessing.
- Cabbageband/patta-gobicompounds: existingHbandgobhi f_ody2xkuwe2bqg seenindonorauditbutnotyetprimaryresearched.
- Bagheliringnegina/celli/celle/chellaalsoavailableunexamined.
- RemainingNimadinoonjuaro/madyanə/dindhaḷe;otherhouseholdandkinship/locatives/finitecompoundsstillproductive.

Laterqualityauditsstillneededinadditiontoresearch. Sameheartbeatactiveeveryhalf-hour;don'tduplicateoraskforapprovalwhileuserasleep.

## Checkpoint — 12:30 heartbeat

Batches024–025 add **16 proposals / 58 records**. Cumulative **655 pending proposals / 3,382 records / 3,428 links**,258straightforward/397qualified. Malvi250/920records/931links;Nimadi201/1764/1788;Bagheli204/698/709. **Nextbatch026; nextnumbersMalvi251,Nimadi202,Bagheli205.** One-shotprepare-grain-mortar-cabbage.py and prepare-time-kinship-remainders.py; do not rerun. Added12investigatedholds(record-ring-grain-holds.py): total120held,3073unexamined(Malvi47/1161,Nimadi33/1029,Bagheli40/883). Noacceptedoverlayedits. render.pyupdatedthrough25.

### New primary research

Discovery047–053 actualhomepagequeriescached. FullCDIAL10434,3796,9818,10039,10431,3104,2998,997,5232,6261read/cachedandparentnodesresolved.
- Baghelijinhora/joṇṇeri/joneri/junheri/juneri/juneṛi/joṇḍari **10434yavanāla**, Bihari janer/jonhrī/jõdhrī/jinorā andHindi junhār/jundrī/jõḍrī. Broadmilletglosspreserved. **Not10438** guessednumberinoldlead. dhuneri excludedpendinginitiald evidence.
- Nimadikhāyṇomortar **3796khaṇḍana**, fullarticleexplicitG khā̃yṇī/khā̃yṇiyɔ mortar besidepoundingverb. Qualifiedlocalnasalization/-o/contact. Noindividualinstrumentsubsectionexistsinprintedentry.
- Cabbagebandgobi-familyallthreeborrowedfromexistingH **f_ody2xkuwe2bqg bandgobʰī**. Existingdonorauditisdata/data/other/params/raw_data/20260911-mewari-donors-audit.json. PrimaryCSTTtableindependentlyreturnedbywebsearch`site.cstt.education.gov.in "Cabbage" "बंदगोभी"`: Englishcabbage/Hindiबंदगोभी/Dogribंदगोभी. FullPDFopenrepeatedlytimedout; nofalseclaimfullPDFvisualinspection. Citationhttps://cstt.education.gov.in/sites/default/files/fundamental-glossary-agriculture-eng-hin-dogri.pdf . OnlyHdonorselected. **pattagobicompoundsremainunexamined**: anotherprimaryCSTTbotanyglossarytableexplicitlyhasपत्तागोभी,butnoexistingwholeHdonorfoundyet. URLhttps://cstt.education.gov.in/sites/default/files/glossary-botany-eng-hindi-bodo.pdf searchresultprintedp43. Canrecordmissingdonorratherthaningest.
- Baghelicelli/celle/chella ring qualifiedagainstexisting **f_grji7azl5s42k *chhallā** familyhead(Arora). Plattsfullringentryat https://www.rekhta.org/urdudictionary?keyword=%E0%A4%9B%E0%A4%B2%E0%A5%8D%E0%A4%B2%E0%A4%BE explicitlyplainfinger/toeringandcompetingcakkala/chakra+la+ka formations. EvidencemakesprovisionalfamilyheadstatusandpossHmediationexplicit;doesnotclaimsecuredeepreconstruction. Hindiattestation f_j7zecsosiaviq alsoexists(McGregor/Arora).

### New investigated holds

Malvi5ringrecordsbiṭi/binḍi/vinḍi/viṭi/beṭ:12045primarygoodringcomparandabutnospecificvīṭṭa/vĭ̄ṇṭaparentnode,heldforgranularityratherthanbaretipcathead. Bagheli6grainrecordsjeba/jabe/geba/jeua/jaua/jo:10431yavafullprosemeansbarley,butsurvey‘millet’;needssemantic/sourceclarification. Nimadimadyanənoon:9818madhyāhnafullprose,learnedretaineddyvsmajjhaṇhacandidate;immediatetransmissionunresolved. All12nowinvestigated-unresolved,notunexamined.

### Temporal and kinship remainders

- Simplekal/kāl/kale yesterday/tomorrowselectedacrossallsurveys,includingcompatiblemergedglosses. **3104-2kalya** exactspecificbranchchosen;homepageoftenmisplacesattestationsunder3104kālyabroadhead,butfullprimary3104.2hasPk kallaṃ/kalhiṃ,Hkal,Aw/G/B/Mlongvowelformsandbothtemporalsenses. Qualifiedlengthandfinal-e.
- Malvikākāfather2998*kākka,seniorrelativefamilyinclKashmirifather;TurnerproposesDravidiandonor,qualifiedfamilynotsecuredeepinheritance.
- Malviāimother997*āī probablynurseryword;G/Marathimother. Do notautomaticallyderivefrom1351āryikā(Dardicpossibility).
- Malvidadā/dādo(mergedfather/olderbrother),Nimadidādo6261*dādda;primaryexplicitfather/elderrelativevariation,qualifiednursery/contact.
- Malvijiji(mergedmother/oldersister)andjijā,Nimadijiji,Baghelijiji/jidyi/jiyyi5232*jījja. FullprimaryGmother,H/Marathioldersister,Sindhijījā/jījīaffectionatemother/aunt. Qualifiedgenderending/palatal/contactdifferences. Noheldrecordsactuallyreopened(batch-025-reopened-holds.jsonempty);thesewereunexamined.

### Quality checks and next leads

Scannedallpendingnoncomponentformsforspaces/commas/slashes: onlyNimadialternativeinflectionresponses106/107/109/110/111/182/183plusBaghelibeṇḍhegobiwholeloan. Nosilentcompoundcollapseidentified. Needmorebroadqualityauditlater.

MissingHdonorsearchjyada/zyada/ziyaada/zyadaaall0; **notproofofabsence**,tryotherorthographiesorfollowcomparandafirst. ExistingHkhūb **f_j3jiehdg5zrla** confirmed;primaryPlattsmeaningquantitativebridgecouldsupportBaghelikhub/khib. Newlearnedpitā/mātā/patnīandwife/husbandghar-vālāfamilyremain. Otherunsurveyedlocatives/morphologicalresponsesstillmany. Continue through15:30UTC;don'tduplicateheartbeat.

Latestvalidation12:35:54UTC:655proposals/3382records/3428links,missingIDs0,changedrecords0,registrymissing0,overlayconflicts0,stableinputs.3422eligiblelinkspassed;6798firstapplicationchanges,0second. Same6safeddependencyrowsremain. OverlaySHAunchanged405bb822c7298b600abd376f9d2e087588400369e288111f55ad108d05b8806b.

## Checkpoint — 13:00 heartbeat

Batches026–027 add **9 proposals / 28 records**. Cumulative **664 pending proposals / 3,410 records / 3,456 links**,258straightforward/406qualified. Malvi252/929records/940links;Nimadi203/1770/1794;Bagheli209/711/722. **Nextbatch028; nextnumbersMalvi253,Nimadi204,Bagheli210.** One-shotprepare-spouse-families.py and prepare-bride-remainder.py; do not rerun. Added21investigatedholds(record-spouse-holds.py):total141held,3024unexamined(Malvi51/1148,Nimadi48/1008,Bagheli42/868). Noacceptedoverlayedits.

Validation13:05:31UTC664/3410/3456:missingIDs0,changedrecords0,registrymissing0,overlayconflicts0,stableinputs.3450eligiblegraphlinkspassed;6854firstchanges,0second. Same6safeddependencyrowsremain. OverlaySHAunchanged405bb822c7298b600abd376f9d2e087588400369e288111f55ad108d05b8806b.

### New spouse analyses

Discovery054–060 homepagequeriespersisted. FullCDIAL4435,7742,9962,9963,8179,10016,11030,6446,9198,9471readandcached.
- Sixhusband/wifeproposalsacrossallthree **4435*gharapāla**: primaryexplicitSindhigharavārohusband,Hgharwālāhouseholder/husbandandfemgharwālīwife. Genderformskeptseparatereviewrows;qualifiedlocalv/l/ḷ/r,contractionandpossibleproductive-vālā/contactreinforcement. Notinventednewcompoundnodes.
- Baghelimeheṛiya/meheriye/meheṛiye/meheriya **9962mahilā**: Mthmehar,Hmihariyāwoman/wife. Deepmahiḍā/mahilā/mahī/mahiṣīdebateexplicit;noguessofsingleorigin.
- Baghelimeheraru/meheṛaru/meheṛaṛu/mehəṛaru **9963*mahilārūpa**,primaryBi/Mth/Bhoj/Awmehrārū/meharārū,matchesfullrārucompoundextension. Exactparentchoseninsteadofbare9962.
- Bagheliḍulhin/ḍuləhinwife **6446durlabha, feminine derivatives**: fullentryMth/OAwdulahini,Hdūlhin/dulhanbride;qualifiedbride/wifesenseandinitialretroflex. Noindependentsubsectionheadinprimary.

### Investigated holds and unresolved leads

21newholds:Malvipitāfather/patniwife,Baghelipiṭafather/peṭniwife: learned/conservativeretainedt/tnplusregionalretroflexion,immediatetransmissionnotresolved. ActualHqueriespita/mata/patnifoundonlypitākāpaternal,mātādrunk,and**paṭnīferryman**wronghomonym. FullprimaryCDIAL8179,10016,7742checked. AvoidclaimingallretainedtformsnecessarilydirectSanskritloans.
Nimadibāi/baywoman(10records)held:9198.2explicitmothernotwoman;homepageNihalibāiwoman/sisterunderbhaginīsuggestsalternativefamily. Plattsbhavatīderivationseenonlyinforumquotation—notprimaryverified;notused. Malviwife bhera/beirā andNimadibairu/bairo/bāiruheld:9471bhāryā/*bhāriyāfullprosedoesn'testablishwesternbāirufamily. ExistingG/Marwcomparandaunlinked;Plattsbairīenemy/falconirrelevant.

Lugāī:existingrootf_k66mptwk5okq6 *lugāīandHdonorf_etpo5b22m7jvi. Plattsprimaryhttps://www.rekhta.org/urdudictionary?keyword=.lgu (alsologo)explicitlugāʼīwoman/wife,contractionoflogāʼī. **Crossreferencelogāʼīetymologyunresolved**,didnotaddnewproposal. Existingearlierpendinglugai mayalreadycovermergedrecords;inspectbeforecontinuing.
Lāḍa/lāḍiwife/husband:homepagequeriesbride/bridgroom/petnoparent. Platts https://www.rekhta.org/urdudictionary?keyword=lade giveslāḍobelovedwife/daughter,redirectlāḍā;**notfulllāḍāformation**. 11030lālyaprimaryonlylāldarling/lālā,notḍforms;donotforcebarelālya. NeedfullPlattslāḍāorregionaldictionary.

### Structural evidence audit

Durableaudit-evidence-structure.py checksall664proposalnumberscontiguousperlanguage,nonemptyevidence/citations,allcitedCDIALarticlescached,andborrowedrowsnamedonorlanguage;noissues13:06:28. **Notfreshscholarlyre-reviewofeveryclaim.** Initialdigit-onlycitationregexfalselyreported11813missingfor11813a;fixedtopreservealphabeticsuffixes. Actual11813aprimarywasalreadycachedandsupportsNimadi158/Bagheli156bhinsārproposals. Read11813fullproseaswell(alternativevibhāyanaderivation)butnocorrectiontothoseproposalsneeded. Renderer'scitationfallbackregexalsofixedtopreservealphabeticsuffixes;reviewsregenerated.

Continue through15:30UTC. Remainingproductiveadjectivesquantifiers/verbs,readsourceglossesfully. Preferqualityofevidencetoexpandingweakproposalfamilies;standaloneformsandtransparentcomponentsstillavailable. Sameheartbeatactive,noduplicateorapprovalquestion.

## Checkpoint — 13:30 heartbeat

Batch028 adds **10 proposals / 27 records**. Cumulative **674 pending proposals / 3,437 records / 3,483 links**,258straightforward/416qualified. Malvi256/940records/951links;Nimadi206/1781/1805;Bagheli212/716/727. **Nextbatch029; nextnumbersMalvi257,Nimadi207,Bagheli213.** One-shotprepare-quality-quantity.py; do not rerun. Added3Bagheliinvestigatedholds(khub/khib/eḍhik):total144held,2994unexamined(Malvi51/1137,Nimadi48/997,Bagheli45/860). Noacceptedoverlayedits.

Validation13:33:46UTC674/3437/3483:missingIDs0,changedrecords0,registrymissing0,overlayconflicts0,stableinputs.3477eligiblegraphlinkspassed;6908firstchanges,0second. Same6safeddependencyrowsremain. Structural evidence audit13:34:13zeroissues(all674numbering/evidence/cachedarticles/donorlanguages). OverlaySHAunchanged405bb822c7298b600abd376f9d2e087588400369e288111f55ad108d05b8806b.

### Quality and quantity proposals

Discovery061–063 actualhomepagequeriespersisted. FullCDIAL13066,9289,3503,250read/cachedandparentsresolved.
- AllthreekharābbadfamilyborrowedfromHindi **f_dbu2w7q6m7sh2 xaraab bad/spoiled**,LiljegrenLX002408. FullPlattsprimaryhttps://www.rekhta.org/urdudictionary?keyword=kharaab includesbad/spoiled/worthless. Qualifiedx→kh,vowels,Baghelifinalh(kherah). NoArabicdonorbypass.
- Malvi/BaghelibekarbadborrowedfromexistingH **f_nc5i5pyxldlxg बेकार bekaar** (nihali-provisional2026). **Donorprovisionalprovenanceexplicitinevidence**,independentPlattsprimarybe-kāruseless/worthlessverifiedathttps://www.rekhta.org/urdudictionary?keyword=be lines114–115. Wholewordloannotsegmentedbe+kār. **Rekhtabekaarqueryreturnsoppositeba-kār‘useful’**,wronghead;citationbeentryisintentional. Firstsearch.eba&reftype=rwebreturnedrelevantfullcompoundpassagebutdirectopenfailed;beURLworks.
- Malvibura/buro/burā,Nimadiburo **9289.1*bura**,fullprimaryS/P/H/G/Mbad/wicked. Specificsection1not9289-2*bōra; retainedburojemphaticformnotincluded. Malviboro/bodastillunexamined—notautomaticallyassignedbecausevowelalternativessubsectionsdifferent.
- Baghelikori‘twenty’ **3503*kōḍi‘score’**,Hkoṛī,Nkori,easternforms. TurnerAustroasiatic-origin/diffusionhypothesisexplicit,notviṃśati. Malvikhoḍinotselectedyet(aspirationrequirescheck).
- Malvisagḷo/hagaḷa/sagaḷa/hagaḷe/sagḷā,Nimadisagəḷo‘all’ **13066sakala**. CDIALshortarticleonlyPk sayalaandrelatedlost-kforms; supplementedwith**Molesworth MD[s.v. sagaḷā]** existingreferencekey. Fullprimaryreproductionhttps://www.wisdomlib.org/definition/sagala (Marathi-Englishsection)explicitMarathisagaḷāall/entirederivedsakala. Qualifiedretainedg/ḷ,Malvis/h,possiblelearned/regionalreinforcement. Malvihagra/hagri/haŋgraandBagheliselaganotincluded. AdditionalprimaryGrayIndo-IranianPhonologysearchresultp50§116givesH/Psagrā,Gsaglō,Marsaglāas< sakala;notcitedorassignedfurtheryet. Plattssagal=saglā crossrefsagrāfoundathttps://urdu.hawramani.com/%D8%B3%DA%AF%D9%84-2/ butfullsagrānotresolved. Molesworthprimarysufficesselectedlforms;don'tclaimrformsalreadyresearched.

### New holds and donor caution

Baghelikhub/khib‘many’:existingH **f_j3jiehdg5zrla khūb much** isnihali-provisional2026 editorialhypothesis,notprimaryattestation. FullPlattshttps://www.rekhta.org/urdudictionary?keyword=khuub confirmsPersiangood/wellnotquantity. Thereforeheldforindependentprimarymany-senseevidence; don'tautomaticallyuseprovisionaldonorglossasproof. NoteearlierMewaridonorauditalsoexplicitlydescribesquantitativeextensionaseditorial. Baghelieḍhikmany:250adhikafullprimaryadditional/superior,learned/retroflextransmissionunresolved;heldnotrejected.
Stillunresolvedjyadādonor:queryzyad:more,ziad:more0,priororthographicvariants0. Couldfollowotherlanguagecomparandawithdifferentglosssuchasmuch/excessively;donotassumeabsent. GañjmanypotentialPersianganjheap/treasurebutcurrentqueriesonlyunlinkedlocalfamilies;needsprimarymeaningandimmediatedonor.

### Next productive inventory

Remainingmany/fewandqualityformsprinted13:30run; exactdispositionsfileauthoritative. Heavy:Malvibajandār/vajandār/bajini/vajani;Nimadibajanda/vajandār/bajni;Baghelibeḍženi/veḍženi. ExistingHdonorswazanīf_7u6j2wcwpjlti,wazn-dārf_47fi53dmujrya,waznf_vorncsozzigxk primarydonorauditavailable;couldproposeadjectiveswithactualfullprimaryverification. Don'ttreatbarenounbajanweightasadjheavywithoutsemanticbridge. Lightkambajaniscompound,notbarekam.
Repeateddifferentformsandek+samecompoundsremain; useimmediatesame-languageanchors/orderedcomponentswithdependencynotes. Needfinalscholarlysampleauditandhonestcoveragecountsby15:30UTC. Sameheartbeatactive; don'tduplicateoraskuserwhileasleep.

## Checkpoint — 14:00 heartbeat

Batch029 adds **5 proposals / 9 records**. Cumulative **679 pending proposals / 3,446 records / 3,492 links**,**257straightforward/422qualified** (Bagheli27movedfromstraightforwardtoqualifiedafteraudit). Malvi258/944records/955links;Nimadi208/1784/1808;Bagheli213/718/729. **Nextbatch030; nextnumbersMalvi259,Nimadi209,Bagheli214.** One-shotprepare-weight-adjectives.py;don'trerun. Held144unchanged;unexamined2985(Malvi51held/1133unexamined,Nimadi48/994,Bagheli45/858). Noacceptedoverlayedits.

Validation14:04:15UTC679/3446/3492:missingIDs0,changedrecords0,registrymissing0,overlayconflicts0,stableinputs.3486eligiblegraphlinkspassed;6926firstchanges,0second. Same6safeddependencyrowsremain. Structural evidence audit14:04:29zeroissues. OverlaySHAunchanged405bb822c7298b600abd376f9d2e087588400369e288111f55ad108d05b8806b.

### Weight adjective loans

Discovery064–065 actualhomepagequeriescached. ExistingHdonorsf_7u6j2wcwpjltiwazanīandf_47fi53dmujryawazn-dārfetched. FullPlattsprimaryhttps://www.rekhta.org/urdudictionary?keyword=vaznii nowdirectlyworksandexplicitwaznī/vulgwazanīweighty/heavy. **Alsoindependentlyviewed /tmp/mewari-wazni.png**,scanofPlattspagewithwazanītoprightandwazn/wazn-dārlowerleft;confirmsprintedmacronsandmeanings. https://www.rekhta.org/urdudictionary?keyword=vazn fullwaznentryexplicitwazn-dārweighty.
- Malvibajini/vajani,Nimadibajni,Baghelibeḍženi/veḍženi→H wazanī,qualifiedw/v/b,z/j/ḍžandvoweladaptation.
- Malvibajandār/vajandār,Nimadibajanda/vajandār→H wazn-dārwholeadjective;qualifiedfinalrlossinbajanda.
- Barenouns b aj an/vajan/beḍžen/voḍženheavyremainunexamined;don'tsubstitutenounweightforadjectivewithoutsourceevidence. Existingwaznf_vorncsozzigxkmeansweightonly.
- Bhāriheavyfamily9465fullprimarycheckedagain; Nimadibhahrihasunusualaspirationsequence,Malvibharoendingdiffers;notadded. Ghana/jabroheavyhomepageyieldsonlyunlinkedforms,noetymologyyet.

### Scholarly sample audit

**scholarly-sample-audit.json** records16purposiveselectedearlierproposals:
Malvi1,24,37,72,99,112,141,170;Nimadi15,29,44,61;Bagheli10,27,49,70.
Rereadfullprimaryarticles/addenda9926,10875,10555,9209,3083,4655,3167,11225,6582,4661,4287,9153,13952,4147,9349; sample-audit-primary.txt preservesmostre-readprose(allareincdial-articles.json). Exactsectionnodeschecked9926,9209,4661. **4661moon isnumberedmeaning2underthesamehead; thereisnopromoted4661-2**,socurrentparent4661correct. 10875-2woodretainedvelarcorrect;9926.1headand9209.1bāppacorrect. 11225vaḍraaddendumretainednotvṛddha;4147cowbranchuncertaintyalreadyexplicit.

Onlysampleeditorialchange: **Bagheli27moon nowqualified**,evidenceexplicitPkcaṃda,Hcandā,Sindhicaṇḍrucomparisonandlocalceṇḍa/caṇḍ/caṇḍevowel/clusterreview.Sindhicomparisonnotdonor. UpdatedpendingmanifestassignmentNotesandregeneratedreviews;noedge/parentchanges.Thisis**sampleaudit,notcompleteindependentre-reviewofall679claims**.

Needlastresearchpasses14:30/15:00,then15:30finalvalidation/triageandpausesameheartbeat. Remainingtransparentadjectiveredu plicationsandek-compoundspossiblewithsame-languageanchors;allsourceglossrecordsretained. Userhasmanyqualifiedproposals;prioritizehelpfultriageandclearevidencelimits. Noapprovalquestionsorsubagents.


## Checkpoint — 14:30 heartbeat

Batch030 adds **2 proposals / 4 records**: Nimadi209 and Bagheli214 reduplicated ‘different’. Cumulative **681 pending proposals / 3,450 records / 3,496 links**,257 straightforward/424 qualified. Malvi258 proposals/944 records/955 links; Nimadi209/1787/1811; Bagheli214/719/730. **Next batch031; next numbers Malvi259, Nimadi210, Bagheli215.** Held144, unexamined2981 (Malvi51/1133; Nimadi48/991; Bagheli45/857). No accepted overlay edits.

Discovery066 and full CDIAL700,404,9338 re-read. Nimadi ālagalag (3 records) derives from same-survey alag anchor f_xmyt2e5ltrhie, pending176. Anchor Sonipura-Balai is explicitly a survey-level proxy, not a donor village for the three target localities. Bagheli elegeleg derives from eleg f_swnisrqkeocbe, pending177; merged source lists overlap at P,b despite different first locality tags. Both qualified, kind derived, with acceptanceDependencies. One-shot prepare-reduplication.py must not be rerun. No duplicate component edges for reduplication. Malvi alagalag/nyārānyārā lacks a verified bare base in this exact survey; canonical-language nyārā may belong to another source. bhatbhat not proposed: 9338 supplies manner/sort but the local formation still needs support.

Validation caught a discovery alias mismatch for the Nimadi base: frozen app resolved f_xmyt2e5ltrhie to older merged f_wk4w3je5oatky. Corrected pending batch030 parent and assignment IDs to the frozen/current survey ID f_xmyt2e5ltrhie, matching pending176. parents.json retains discoveryResolvedId and an identityResolution note; reduplication-identity-reconciliation.json documents the correction. Do not overwrite this correction by blindly rerunning the parent helper. No accepted data changed.

New **DEPENDENCIES.md**, linked from TRIAGE.md, lists all compound/reduplication base dependencies and unresolved safed donor dependencies. render-dependencies.py regenerates it. render.py supports derived displays and topics through030.

Validation14:33:37UTC: 681/3450/3496; missing IDs0, changed target records0, missing registry targets0, overlay conflicts0, stable inputs. **3,490 eligible links passed**,6 safed rows remain blocked by f_lwa4hsrbk5gee; first application6934 changes, second0. Structural audit14:34:12 reports no issues across681 proposals; this is not an independent scholarly re-review of every proposal. Overlay SHA unchanged405bb822c7298b600abd376f9d2e087588400369e288111f55ad108d05b8806b.

Remaining:15:00 research/triage polish,15:30 final validation and honest handoff, then pause the existing heartbeat. Deadline15:30UTC/11:30a.m. ET. Keep all proposals pending; no approval questions, ingestion, subagents, or accepted-overlay changes.


## Checkpoint — 15:00 heartbeat (research through 15:19 UTC)

**757 pending proposals / 3,731 records / 3,777 links**: 266 straightforward and 491 qualified. This run added batches031–038: **76 proposals / 281 records**, and 20 investigated holds. Malvi291 proposals/1043 records/1054 links (115 straightforward,176 qualified); Nimadi230/1908/1932 (100/130); Bagheli236/780/791 (51/185). Held164; unexamined2680 (Malvi68/1017; Nimadi50/868; Bagheli46/795). Next batch039; next numbers Malvi292, Nimadi231, Bagheli237. One-shot prepare scripts for031–038 must not be rerun.

### New research

Discovery067–079 saved;072,076,079 use the actual homepage lexicon implementation filtered to Hindi, the other files its reflex search. Full primary CDIAL articles and addenda read/cached; existing parent/section IDs resolved.

- **031 (25 proposals/82 records):** head sir across all three →12452 śiras, Bagheli ser qualified. Above uppar/upar family→2333 *uppari (not upper *uppara2330); below nice/nico etc→7540, whose full prose explicitly explains cc after opposite ucca. Aspirated ch forms separate qualified rows. Malvi left b-forms→5539.3 *ḍābba, v-forms and Nimadi ḍāõ→5539.2 *ḍāva; full primary/addendum Kacchiḍābo supports exact branch, non-Aryan/Dravidian-origin hypothesis retained. Hot Malvi unːo etc/Nimadiuno→2389 uṣṇa (westernūnũ/ūn; gemination/vowel length qualified). Malvi ghaṇ hammer→4423. Malvi latta/letta/letto cloth→10930.1; latəra/lattra→10930.2 *lattara, with nominal/cloth semantic qualification. Malvi/Nimadi jēmana-type right-hand forms→5268, whose full prose gives Prakrit jēmaṇaya right/eating hand and OGjimaṇaüṁ righthand side. High/above uñco/uce/ucho→1634, nasal origin uncertain per Turner and aspiration not claimed to follow an established local rule.
- **032 (12/46):** all three dahina/dāyā families→6251 dākṣiṇa, including Prakrit retroflexḍāhiṇa and explicit crossing with vāma for contracted dā̃yā; not bare dakṣiṇa. Left bāyā/bāyo→11533vāma, missing nasalization qualified; retained-m bama and behiya excluded. Malvi/Nimadi uḷṭa→2368.2 *ullaṭyate (full addendum explicitly ulṭɔ left/reverse), heṭa/heṭṭā→248.2 *adhiṣṭāt; sido/sidā/sito right→13401.1 siddha plus Platts full sīdhā entry explicitly right hand at https://www.rekhta.org/urdudictionary?keyword=siidhaa . sudo/huda excluded pending proper-vs-right-hand evidence.
- **033 (5/12), Malvi:** bij/vij→11742vidyut (possible contraction of11745qualified); hono/honːo/honā/huṇa/hunːo→13519 (suvarṇa/sauvarṇa uncertainty explicit), hāp/hap→13271, ā̃p separate qualified13271, hui/ui→13551.1. Independently checked exact source localities against MALVI-S-H.md. **Corrected an overstatement before rendering:** Bhandia has ā̃p, not an attested hāp intermediary. Both manifest and builder now state regional hāp/hap occurs elsewhere. No accepted edits.
- **034 (10/29):** carbi fat all3→existing Hf_cdzupphnjmodm, primary Plattsćarbī https://www.rekhta.org/urdudictionary?keyword=charbii . All3makān house→Hf_bk76lqk32ood6, Griersonpp102–103 WESTERNHINDIHINDOSTANI-51_house-1, primaryPlatts https://www.rekhta.org/urdudictionary?keyword=makaan . **Existing Hmakān is unlinked**, explicit dependency for5links. Malvigos/gōs meat→Hf_qbahea6yfb6qm gost(Kannauji75), Platts https://www.rekhta.org/urdudictionary?keyword=gosht . Chicken murgi all3 (+Nimadimurəgi)→Hf_5azv73u3n6upa Kannauji75Rohili; not a claim of donor village. **Full Platts murġī verified in local DSAL PDF page2075 (zero-based2074), including murġ/Persian and Hindi feminine formation.** Rendered and viewed; durable platts-murghi.png linked from reviews. Rekhta murgii/murGii pages did not expose that exact Platts entry; do not claim they did. Nimadimurgo and Baghelimurgihin excluded.
- **035 (5/13):** Bagheli unaspirated mūḍ/muḍ/muḍi/muḍe/mur→10247 with explicit alternative/crossing10191muṇḍa per full primary; aspiratedmuḍh/muḍhi/muḍhə separate qualified10247. Baghelikepar→2744.4*kappāla, easternkapār and karpara/*kōppara interference noted. Malvi/Bagheligāj lightning→4048garjā, full primaryHthunderbolt; lightning/thunder distinction qualified.
- **036 (6/40):** potato alu forms all3→1388.1ālu/āluka, full primary explicitly modern potato alongside older aroid roots; semantic transfer and contact path qualified, not ancient potato or demonstrated inherited transmission. Tomato all3→Hf_ioja36d77utxa(Kannauji74), full HindiŚabdasāgara entry read in explicitly attributed reproduction https://hi.wiktionary.org/w/index.php?title=टमाटर&oldid=486401#शब्दसागर . Entry gives English source; immediate Hindi retained. Collins independent Hindi entry also checked. Platts PDF has vilāyatī-baiṅgan tomato, no verified ṭamāṭar head; do not cite Platts for tomato.
- **037 (10/53):** eleven all3→2485, full primaryOGigyāra/Kotgarhigyāra; twelve all3→6658.1, b-initial selects dvādaśa over duvādaśa branch2. Malvi/Nimadi straightforward, Baghelivowels/ṛqualified. Malvi/Bagheli evening/afternoon forms→12918saṃdhyā; c/s/h and stop sequences separately qualified, exact combined source gloss preserved. Bagheliseṇḍiya/seṇche/sāḍž are explicitly weaker family candidates needing dialect review, not normalized forms. śām excluded from inherited evening family.
- **038 (3/6):** whole phulgobi cauliflower all3 (+Malvip hulgobhi, Nimadiɸulkobi)→Hf_wl2o55yco7mxk(Kannauji73). Existing Hdonor unlinked, explicit dependency6links. Primary Hindi–English table in Central Bank of India Cent Saral Bhasha PDFp11 gives फूलगोबी/Cauliflower (https://www.centralbankofindia.co.in/sites/default/files/%E0%A4%95%E0%A4%A8%E0%A5%8D%E0%A4%A8%E0%A4%A1.pdf#page=11). Full PDF opened and row read. Reversed gobiphul and baregobi excluded.

### Investigated holds / remaining leads

20 new holds: Malvi9+Nimadi2clothforms (CDIAL4802 *citth verbal head and nominal -ḍ-/-r- extensions not promoted; suffix/aspiration/nasalization unresolved); Malvisudo/huda right (12520 clean/proper ≠ demonstrated right-hand); Malvidumtail (6419 explicitly leaves Iranian vs IA origin unresolved; Hindi borrowing not established just by identical form); Malvikabal head(2744competing p/bh branches); Malvikopəḍa head(full3519,3936,2876 competing r/ḍ and aspiration); Malvisām/śām/śamki plusBagheliśam evening (fullPlattsshām https://www.rekhta.org/urdudictionary?keyword=shaam verified, immediateHindi donor not found in current homepage inventory; ferrulesām is unrelated). These are held, not rejected. Some later hold timestamps were manually labelled15:18 ahead of actual clock15:17; treat as approximate pass metadata, not precise wall-clock evidence.

### Validation and review presentation

Validation **15:17:18UTC**: all757 proposals,3731targetrecords,3777links. MissingcurrentIDs0,changedrecords0,missingregistrytargets0,overlayconflicts0,stableinputs. **3,760 eligible links pass**,7,474first-application changes,0second. **17 blocked links across3donors:** safedf_lwa4hsrbk5gee6, makānf_bk76lqk32ood6 5, phulgobif_wl2o55yco7mxk6. All explicit dependencies, no donor ingestion. OverlaySHA unchanged405bb822c7298b600abd376f9d2e087588400369e288111f55ad108d05b8806b. Structural audit15:17:43 noissues; not universal scholarly re-review.

render.py now knows topics through038. Fixed misleading labels: a supplemental primarySourceURL is now a separate **Primary evidence** link beside the exact citation text; a Kannauji locator is no longer made a hyperlink to Platts/another primary dictionary. CDIAL-only links retain exact page links. Corrected singular proposal grammar. Reviews regenerated after the change. render-dependencies.py now computes blocked-link counts by donor dynamically; DEPENDENCIES.md lists17links and all base dependencies.

**Next scheduled run is the15:30 deadline handoff. First check time; at/after15:30 stop new research, rerun fresh validate.py and structural checks, render final counts, mark TRIAGE/REVIEW research finished (all still pending), and pause the SAME heartbeat malvi-nimadi-bagheli-overnight-etymologies. Inspect existing automation TOML and preserve fields. Do not create a duplicate or ask user approvals.** Final answer should give review link,757proposal/3731record/3777link counts if unchanged,164held/2680unexamined, and17dependencylinks. No accepted overlay edits, no build/commit/push/deploy, no subagents. User’s eight-hour research deadline15:30UTC/11:30a.m. ET remains active.


## Final handoff — deadline reached, 15:31 UTC

Research stopped at the eight-hour deadline. **757 pending proposals, 3,731 records, 3,777 proposed links**; 266 straightforward and 491 qualified. Malvi291/1043/1054; Nimadi230/1908/1932; Bagheli236/780/791 (proposals/records/links). **164 held unresolved and 2,680 unexamined**, together accounting with proposed records for all6,575 frozen records. These are research coverage counts, not accepted etymologies.

Fresh validation15:31:10UTC found no missing current IDs, changed target records, missing registry targets or overlay conflicts; inputs stable. Temporary graph passed **3,760 eligible links**, first application7,474 changes, second0. **17 links excluded for unresolved donor ancestry:** safed6, makān5, phulgobi6. The accepted overlay SHA is unchanged405bb822c7298b600abd376f9d2e087588400369e288111f55ad108d05b8806b. Structural evidence audit15:31:44 reports no issues across757 proposals. This is not a fresh independent scholarly review of the entire set; the earlier16-proposal sample audit remains explicitly limited.

TRIAGE.md, REVIEW.md, counts.json, dispositions.json and DEPENDENCIES.md regenerated; overview marked research finished and all analyses pending. The triage entrypoint now puts per-survey totals and validation/dependency limits before the batch index. HANDOFF.json freezes this handoff metadata. No new research or new proposal rows after deadline. No accepted-overlay changes, corpus rebuild, commit, push or deployment.

**The SAME heartbeat malvi-nimadi-bagheli-overnight-etymologies was successfully set to PAUSED via the app tool**, preserving its name, prompt, schedule and target thread. Do not resume research automatically. Await the user’s triage; approval must identify survey/proposal numbers, and any accepted save requires fresh ID and overlay reconciliation. Main user entrypoint: TRIAGE.md.


## User approval applied — September 11, 2026

User approved all 757 proposals and explicitly requested Persian/Arabic nesting instead of Hindi-Urdu donor claims. All 3,777 rows over 3,731 survey records are saved to the accepted overlay. 223 Perso-Arabic links were reparented; 6 cauliflower links gained an independent Hindi lexical head. Nine dictionary heads installed and registered without changing existing Hindi heads. All 164 held and 2,680 unexamined records remain outside this acceptance.

Do not replay `prepare_acceptance.py`, `correct_approved.py`, or `save_approved.py`: those are one-shot acceptance provenance scripts, with guards. The reusable source emitter is `data/other/params/raw_data/central_surveys_donors.py`. All batch manifests and review tables are marked saved, and `HANDOFF.json` points to fresh acceptance validation. Original proposals are preserved in `before-approval-manifests.json`.

Fresh temporary graph validation passed for all rows, second application zero, preserving all unrelated edges and 22,644 old overlay rows. Isolated full compilation added exactly nine IDs, lost none, changed zero unrelated compiled forms and preserved every approved target form, meaning, source and dialect tag. Shared generated CLDF/browser database were not replaced. `make all` failed only its final two manual-survey tests; both reproduce against the shared checkout. Combined focused checks: 36 passed, 9 skipped, one existing self-reference count failure, independently reproduced. Broader suite final outcome will be recorded in ACCEPTED.md and acceptance-logs.

Automation remains PAUSED. Do not resume overnight proposal generation without a new user request. No commits or deployments.

Full suite finished: 1,733 passed, 21 skipped, 52 failed (437.70s). No new donor-test failures. Missing temporary PDF/sibling-frontend fixtures and unrelated corpus assertions remain; logs and complete failure list saved in acceptance-logs/full-suite.log and test-results.json. All intended accepted rows are saved; do not repeat the save.


## Subsequent user-requested database rebuild

Shared CLDF and local browser database refreshed (cache version 32). All 3,777 links and nine donor heads verified; browser QA passed for Arabic borrowing and ordered compound components. Frontend check: zero errors, seven warnings. Same two existing manual-survey build-gate failures remain. No deployment or commit. See DB-REBUILD.md. Earlier refresh exclusions describe previous stages.
