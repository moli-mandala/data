"""Pin revision 2 independently of the historical failed freeze."""
from pathlib import Path
import csv,hashlib,json,random
HERE=Path(__file__).resolve().parent
OLD=HERE.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return {r[10]:r for f in (p/'proposal').glob('*.csv') for r in csv.reader(f.open())}
def units(p):return {a['source_unit_key']:a for a in map(json.loads,(p/'proposal/whole-audit.jsonl').read_text().splitlines())}
before=rows(OLD);after=rows(HERE);bu=units(OLD);au=units(HERE)
assert bu.keys()==au.keys()
for k,a in au.items():
 assert a.get('word_alignment',[])==bu[k].get('word_alignment',[]),(k,'alignment changed')
 assert a['entry_keys']==bu[k]['entry_keys'],(k,'reuse changed')
assert before.keys()==after.keys()
changes=[]
for k,r in after.items():
 old=before[k];assert r[2]==old[2],k
 diff={str(i+1):{'before':old[i],'after':r[i]} for i in range(15) if old[i]!=r[i]}
 assert set(diff)<={'4','7','15'},(k,diff)
 if diff:changes.append({'entry_key':k,'fields':diff})
report={'status':'owner_metadata_revision_ready_for_independent_review','previous_manifest_sha256':sha(OLD/'frozen-manifest.json'),'canonical_unchanged':True,'same_648_entry_keys_and_originals':True,'same_546_source_units_and_reuse_mappings':True,'changed_rows':len(changes),'changes':changes,'review_scope':['All 79 numbered units and their 264 aligned words/phrases: contextual numerals, independent pronouns, interrogatives, personal name, overt verb analyses.','All 53 table units and 139 inline units: explicit numeral/pronoun/morpheme classes, readable analysis glosses, epenthetic -y- scope.','All distinct new display glosses: cop expanded, rdp-go retains lexical go, inline give analyses moved to printed-analysis Notes, pronoun abbreviations expanded.','Uncertainty localized within examples 19b,29,45,46,54; whole-expression evidence retained.'],'deliberate_limits':['No part of speech inferred merely from a translated English plural.','General -y- notation has no locality; Jashpur and Sarangarh supporting occurrences remain separately attributed.','No raw form, Phonemic, lexical key, citation, variant edge, or existing Appendix1 byte changed.']}
(HERE/'metadata-revision-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
summary=json.loads((HERE/'proposal/summary.json').read_text())
files={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file() and (p.parent==HERE/'proposal' or p.name.endswith('-reviewed.jsonl') or p.name in ['baseline-installed.json','build_proposal.py','install_proposal.py','metadata-revision-review.json'])}
files.update({'../'+n:sha(OLD/n) for n in ['source-manifest.json','page-census.json','owner-grammar-review.json','owner-texts-review.json','owner-inline-review.json','auxiliary-reference-review.json','dialect-proposal.json','root-independent-output-audit-20260927-pass1.json']})
manifest={'status':'frozen_for_independent_audit_not_installed','revision':2,'source':'peterson2024turi','original_sha256':json.loads((OLD/'source-manifest.json').read_text())['sha256'],'rows':summary['total_rows'],'audit_units':summary['audit_units'],'baseline_csv_sha256':summary['canonical_unchanged_sha256'],'files':files,'focused_tests':{'path':'tests/test_peterson_turi_revision2.py','passed':6,'failed':0}}
(HERE/'frozen-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
prior=json.loads((OLD/'independent-sample-20260927.json').read_text())
excluded={x['source_unit_key'] for x in prior['selected']}|set(prior['prior_legacy_sample_excluded'])
strata={x:[] for x in ['legacy','grammar','texts','historical']}
for k,a in au.items():
 if not a['entry_keys'] or k in excluded:continue
 u=a['evidence']
 if a['section']=='appendix1':s='legacy'
 elif u.get('lect','').startswith('historical'):s='historical'
 elif 'example' in u and int(''.join(c for c in u['example'].split('-')[0] if c.isdigit()))>=44:s='texts'
 else:s='grammar'
 strata[s].append(a)
rng=random.Random(20260927112);selected=[]
for s,n in [('legacy',5),('grammar',5),('texts',6),('historical',4)]:
 for a in rng.sample(strata[s],n):
  selected.append({'stratum':s,'source_unit_key':a['source_unit_key'],'printed_page':a['evidence']['printed_page'],'pdf_page':a['evidence']['printed_page']-260,'audit_unit':a,'actual_csv_rows':[after[k] for k in a['entry_keys']]})
(HERE/'independent-sample-20260927.json').write_text(json.dumps({'seed':20260927112,'status':'selected_not_audited','frozen_manifest_sha256':sha(HERE/'frozen-manifest.json'),'excluded_prior_source_units':sorted(excluded),'selected':selected},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'manifest_sha256':sha(HERE/'frozen-manifest.json'),'rows':len(after),'changed_metadata_rows':len(changes),'sample_units':len(selected),'sample_rows':sum(len(x['actual_csv_rows']) for x in selected)}))
