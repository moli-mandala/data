"""Pin source-scoped metadata revision and a disjoint twenty-unit sample."""
from pathlib import Path
import csv,json,hashlib,random,re
H=Path(__file__).resolve().parent;ROOT=H.parent;OLD=ROOT/'revision3'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def rows(h):return {r[10]:r for p in (h/'proposal').glob('*.csv') for r in csv.reader(p.open())}
before=rows(OLD);after=rows(H);assert before.keys()==after.keys()
changes=[]
for key,r in after.items():
 old=before[key];diff={str(i+1):{'before':old[i],'after':r[i]} for i in range(15) if old[i]!=r[i]}
 assert set(diff)<= {'7','15'},(key,diff)
 if diff:changes.append({'entry_key':key,'fields':diff})
audit=list(map(json.loads,(H/'proposal/whole-audit.jsonl').read_text().splitlines()))
prior={u['source_unit_key']:u for u in map(json.loads,(OLD/'proposal/whole-audit.jsonl').read_text().splitlines())}
for u in audit:
 assert u['entry_keys']==prior[u['source_unit_key']]['entry_keys']
 assert u.get('word_alignment')==prior[u['source_unit_key']].get('word_alignment')
review={'status':'owner_review_ready_for_independent_audit','rows':648,'source_units':546,'previous_manifest_sha256':sha(OLD/'frozen-manifest.json'),'unchanged':['all literals','all IPA','all keys','all glosses','all citations','all graph edges','all69 exact-reuse mappings'],'changes':changes,'changed_rows':len(changes),'review_scope':'All139 inline audit units, including101 emitted target/marker entries, individually scoped against source pp267–279/281/290. No English gloss/Notes token mining remains in inline tags. Table5 argument indices likewise carry explicit subject/object roles. Full copula stems and suppletive forms distinguished from enclitic identity copula and inflectional markers.'}
(H/'metadata-revision-review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
files={str(p.relative_to(H)):sha(p) for p in H.rglob('*') if p.is_file() and (p.parent==H/'proposal' or p.name.endswith('-reviewed.jsonl') or p.name in ['baseline-installed.json','build_proposal.py','install_proposal.py','review_legacy_metadata.py','legacy-metadata-review.json','review_inline_metadata.py','inline-metadata-review.json','metadata-revision-review.json'])}
for name in ['source-manifest.json','page-census.json','owner-grammar-review.json','owner-texts-review.json','owner-inline-review.json','auxiliary-reference-review.json','dialect-proposal.json','root-independent-output-audit-20260927-pass1.json','revision2/root-independent-output-audit-20260927.json','revision3/root-independent-output-audit-20260927.json']:
 files['../'+name]=sha(ROOT/name)
summary=json.loads((H/'proposal/summary.json').read_text())
manifest={'status':'frozen_for_independent_audit_not_installed','revision':4,'source':'peterson2024turi','original_sha256':json.loads((ROOT/'source-manifest.json').read_text())['sha256'],'rows':648,'audit_units':546,'baseline_csv_sha256':summary['canonical_unchanged_sha256'],'files':files,'focused_tests':{'path':'tests/test_peterson_turi_revision4.py','passed':6,'failed':0}}
(H/'frozen-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
samples=[json.loads((p/'independent-sample-20260927.json').read_text()) for p in [ROOT,ROOT/'revision2',ROOT/'revision3']]
exclude=set(samples[0]['prior_legacy_sample_excluded'])|{u['source_unit_key'] for s in samples for u in s['selected']}
strata={s:[] for s in ['legacy','grammar','texts','historical']}
for a in audit:
 if not a['entry_keys'] or a['source_unit_key'] in exclude:continue
 u=a['evidence']
 if a['section']=='appendix1':s='legacy'
 elif u.get('lect','').startswith('historical'):s='historical'
 elif 'example' in u and int(re.match(r'\d+',u['example'])[0])>=44:s='texts'
 else:s='grammar'
 strata[s].append(a)
print('remaining strata', {s:len(v) for s,v in strata.items()})
text_count=min(6,len(strata['texts']))
rng=random.Random(20260927114);selected=[]
for s,n in [('legacy',5),('grammar',11-text_count),('texts',text_count),('historical',4)]:
 for a in rng.sample(strata[s],n):selected.append({'stratum':s,'source_unit_key':a['source_unit_key'],'printed_page':a['evidence']['printed_page'],'pdf_page':a['evidence']['printed_page']-260,'audit_unit':a,'actual_csv_rows':[after[k] for k in a['entry_keys']]})
(H/'independent-sample-20260927.json').write_text(json.dumps({'seed':20260927114,'status':'selected_not_audited','frozen_manifest_sha256':sha(H/'frozen-manifest.json'),'excluded_prior_source_units':sorted(exclude),'stratum_adjustment':f'Only {len(strata["texts"])} unselected emitted text unit(s) remain after prior passes; select {text_count} and move remaining text slots to grammar/inline rather than recycle prior units.','selected':selected},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'manifest_sha256':sha(H/'frozen-manifest.json'),'rows':648,'changed_metadata_rows':len(changes),'sample_units':20,'sample_rows':sum(len(u['actual_csv_rows']) for u in selected)}))
