import csv,hashlib,json
from pathlib import Path
O=Path(__file__).resolve().parent;P=O.parent;I=P/'integration'
W=next(p for p in P.parents if (p/'jambu-static').is_dir())
s=json.loads((O/'independent-selection-20260927317.json').read_text())
for n,h in s['frozen_hashes'].items():assert hashlib.sha256((I/n).read_bytes()).hexdigest()==h,n
rows={r[10]:r for r in csv.reader((I/'combined-candidate.csv').open())}
observations=[
 ('kui-alternate-leaf685-bottom.png','kui-physical-685.png','Gule negi lokurki reads clearly, including plain e/i; English126 To good men confirms dative plural expression.'),
 ('kui-alternate-leaf689-top.png','kui-physical-689.png','Poṛā goṛāṅgā matches the source row and English140 Horses. Plural and animal-sex note agree; no grammatical masculine inference.'),
 ('kui-alternate-leaf693-bottom.png','kui-physical-693.png','Sāhāmu is the first of three imperative alternatives for175 Beat. Its glyphs are clear; uncertainty remains only on third kommu and is correctly absent here.'),
 ('kui-alternate-leaf693-top.png','kui-physical-693.png','Āmu māsēmu and English165 We were agree; first-person plural/past scope is correct.'),
 ('kui-alternate-leaf681-bottom.png','kui-physical-681.png','Ābā gule matches last row and English106 Fathers; source quantity on Ā/ā and plain final e retained.'),
 ('kui-grammar-original-leaf483.jpg',None,'Noun prose explicitly tināri, she who eats, contrasted with plural tināwi; feminine singular noun agrees.'),
 ('kui-grammar-original-leaf484.jpg',None,'Muĩ, I can, appears in the ability construction prose. Nasal i, capitalization, first-person and modal analysis agree.'),
 ('kui-grammar-original-leaf484.jpg',None,'Personal terminations table positive imperative plural2 prints pāgātu. Selected column/person tags agree, separate negative cell remains separate.'),
 ('kui-grammar-original-leaf483.jpg',None,'Pronoun margin explicitly says genitive often trā instead of tānā. Reflexive/genitive gloss and source note agree.'),
 ('kui-grammar-original-leaf484.jpg',None,'Compound-tense prose parenthesis in Gumsur gipkissē I was doing matches exact literal and source-specific standard route.'),
 ('kui-specimens/alternate-n488.jpg',None,'p467 lower interlinear line prints pānpā-sāsēñju; aligned was-found; quantity, ñ and final semicolon preserved.'),
 ('kui-specimens/alternate-n492.jpg',None,'p471 line15 māṅgē / me-to matches visible upper n dot and quantity; Kalahandi heading preserved with approved historical identity uncertainty.'),
 ('kui-specimens/alternate-n489.jpg',None,'p468 line4 pājā / having-killed matches literal/alignment and District Khondmals heading.'),
 ('kui-specimens/alternate-n488.jpg',None,'p467 line7 ēsēkābē / ever matches all marked vowels and atom boundary.'),
 ('kui-specimens/alternate-n485.jpg',None,'p464 line11 iñji / saying matches printed atom; Smith upper-nasal typography caveat remains in Notes/uncertainty.'),
 ('kui-outside-291-detail.png',None,'Explicit Kanarese plural list begins gaḷu; Kannada route, plural suffix and blank lexical gloss agree.'),
 ('kui-outside-293-comparands-table.png',None,'Kurukh column first row We exclusive prints ēm; Kurux route, 1pl and exclusive tags agree.'),
 ('kui-outside-296-comparands-tense.png',None,'Past-tense conjunctive suffix list explicitly prints Telugu t on the next line after Gondi t. The sampled Telugu row is correct; blank lexical gloss and suffix/past analysis agree, with no comparative ancestry edge.'),
 ('kui-outside-291-detail.png',None,'Tamil colloquial plural list explicitly ends āṅg; quantity and dotted nasal agree with colloquial plural suffix tags.'),
 ('kui-outside-296-comparands-tense.png',None,'Tamil past/conjunctive suffix prose explicitly says vulgarly pronounced chchu. Literal, vulgar register and morphology retained; conjectural ku derivation stays prose.')]
result=[]
for idx,(item,obs) in enumerate(zip(s['sample'],observations),1):
 assert rows[item['entry_key']]==item['row']
 images=[]
 for rel in obs[:2]:
  if rel:
   p=W/'tmp'/rel;images.append({'path':str(p.relative_to(W)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 result.append({'sample_number':idx,'entry_key':item['entry_key'],'row':item['row'],'status':'pass','original_evidence':images,'observation':obs[2]})
assert len(result)==20
report={'reviewer':'/root/asur_finish','seed':s['seed'],'status':'passed','sample_size':20,'material_error_count':0,
 'design':s['design'],'frozen_hashes':s['frozen_hashes'],'sample_rows_verified_against_actual_csv':True,
 'checks':'All twenty compared visually against retained original page/crop images, including literal marks, aligned gloss, language/lect attribution, explicit grammar/register tags, Notes and source locator. Four English prompt pages checked separately for five table entries.',
 'results':result,'limitations':['Fresh twenty-entry review supplements prior complete source readings; it is not a replacement for full installation validation, database build or browser QA.'],'canonical_changes':False}
(O/'independent-original-review-20260927317.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
report.update(csv_sha256=s['frozen_hashes']['combined-candidate.csv'],audit_sha256=s['frozen_hashes']['combined-audit.jsonl'],sampled_count=20,material_errors=[])
(O/'fresh-original-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
(O/'fresh-original-observations.md').write_text('# Fresh original audit observations\n\nSeed 20260927317; 20 of 20 pass. Sample 18 observation corrected from Gondi to Telugu; both labels have t in the original, and the sampled CSV row was always Telugu. No candidate bytes changed.\n\n'+'\n\n'.join(str(r['sample_number'])+'. `'+r['entry_key']+'` — '+r['observation'] for r in result)+'\n')
(O/'reviewer-findings.md').write_text('''# Final independent source-stage review

Reviewer: `/root/asur_finish`, 2026-09-27.

Final CSV: `b3869cae1a515ae745f00e99edb9344c73939f54a941fc7a9dc241f4c4b8a133`.
Final audit: `9b76c68718d5258ac85cd635e6aa4027b554e7260499d53b7b9c45e5fa34782e`.

- **Complete transfer checks pass.** `field-audit.json` accounts for all 1,832 rows and 1,859 audit units. Literal forms, glosses, source tags/lects, six approved component edges, alternative expansions, two physical joins, expression links, unique keys and NFC/NFD profile coverage pass. All nonemission and existing-occurrence reuse dispositions remain explicit.
- **Fresh original audit passes.** `fresh-original-audit.json`, seed 20260927317, records 20/20 passes with zero material errors: five entries each from table, grammar, specimens and outside comparisons. Retained originals were visually checked, including four separate English prompt pages. Actual sampled CSV rows and frozen hashes were reverified.
- **Repair continuity passes.** `postrepair-delta-review.json` proves exactly forty Notes changes and five uncertainty-tag removals; all other fields are identical to the immutable d9c2bd20 baseline. Uncertainty is correctly scoped to second alternatives at47/51, first at99 and third at175. Causal remains causal, not causative. Every Notes replacement retains actual source claims and qualifications; all three genuine Letchmajee disagreement statements remain.
- **Routes pass.** Historical Kalahandi stays source-qualified Kui with the modern identity conflict explicit. Kōi is held, never mapped to Kui. Generic Korava, Burgandi and Bhili of Basim remain unresolved; the ambiguous p570 Kolami expansions remain documentary. Exact Berar Kaikadi routing is used. Literary Telugu and unspecified Tamil dialects have no invented geographical identities.
- **Configuration passes.** `install-metadata-review.json` validates actual proposed settings with global SourceMeta, guarded importer command, all five reference blocks and source-local profile isolation. Four historical Kui dialect proposals retain blank coordinates. Actual generic-parser fixture evidence is separately owned by the integration reviewer.
- **Bibliography passes.** `bibliography-review.json` pins d8c4e7ce91fba54021f338ffa193bb71837bce8e71157a89e192ce57242d73a1. Five references parse, have no key collisions, and render using the production Pybtex formatting call. Reported authority fields match original pp458–459. Three records use @misc to avoid invented publishers; auxiliary works remain explicitly quoted through Grierson rather than independently collated. Main provenance correctly names the planned canonical CSV, audit, receipt and scope.

No independent review blocker remains. This is source-stage readiness, not a claim that canonical installation, full build, database validation or browser QA has occurred. No such operations were performed by this reviewer.
''')
print('PASS 20/20; exact frozen hashes and sampled actual rows verified')
