"""Freeze metadata revision 3; preserve both failed historical versions."""
from pathlib import Path
import csv,json,hashlib,random
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OLD=ROOT/'revision2'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def rows(h):return {r[10]:r for p in (h/'proposal').glob('*.csv') for r in csv.reader(p.open())}
before=rows(OLD);after=rows(HERE);assert before.keys()==after.keys()
changes=[]
for k,r in after.items():
 old=before[k]
 assert r[:14]==old[:14],k
 if r[14]!=old[14]:changes.append({'entry_key':k,'old_tags':old[14],'tags':r[14]})
audit=list(map(json.loads,(HERE/'proposal/whole-audit.jsonl').read_text().splitlines()))
prior_units={a['source_unit_key']:a for a in map(json.loads,(OLD/'proposal/whole-audit.jsonl').read_text().splitlines())}
for a in audit:
 old=prior_units[a['source_unit_key']]
 assert a['entry_keys']==old['entry_keys'] and a.get('word_alignment')==old.get('word_alignment')
review={'status':'owner_review_ready_for_independent_audit','previous_manifest_sha256':sha(OLD/'frozen-manifest.json'),'rows':648,'source_units':546,'all_originals_ipa_keys_glosses_citations_and_links_unchanged':True,'changed_tag_rows':len(changes),'changes':changes,'classes_reviewed':['All numbered whole-expression boundaries: ellipsis excerpts4/5/25 are multiword-expression only, not sentential; title48 and noun paradigms1a–c already nonsentential.','All documented encoded-versus-context analysis discrepancies: ex45 raw ipl.incl retained with uncertainty, but source-explicit1pl.incl structured; ex15b/19b/46/54 conflicting readings remain uncertain rather than silently corrected.','All275 Appendix1 prompts and246 emitted rows, with47 additive metadata changes; exact per-key evidence in legacy-metadata-review.json.'],'pending':['Fresh independent audit','Guarded installation plus original importer/YAML route reconciliation','Full CLDF/build/browser deferred by user']}
(HERE/'metadata-revision-review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
files={str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*') if p.is_file() and (p.parent==HERE/'proposal' or p.name.endswith('-reviewed.jsonl') or p.name in ['baseline-installed.json','build_proposal.py','install_proposal.py','review_legacy_metadata.py','legacy-metadata-review.json','metadata-revision-review.json'])}
for name in ['source-manifest.json','page-census.json','owner-grammar-review.json','owner-texts-review.json','owner-inline-review.json','auxiliary-reference-review.json','dialect-proposal.json','root-independent-output-audit-20260927-pass1.json','revision2/root-independent-output-audit-20260927.json']:
 files['../'+name]=sha(ROOT/name)
summary=json.loads((HERE/'proposal/summary.json').read_text())
manifest={'status':'frozen_for_independent_audit_not_installed','revision':3,'source':'peterson2024turi','original_sha256':json.loads((ROOT/'source-manifest.json').read_text())['sha256'],'rows':648,'audit_units':546,'baseline_csv_sha256':summary['canonical_unchanged_sha256'],'files':files,'focused_tests':{'path':'tests/test_peterson_turi_revision3.py','passed':5,'failed':0}}
(HERE/'frozen-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
first=json.loads((ROOT/'independent-sample-20260927.json').read_text());second=json.loads((OLD/'independent-sample-20260927.json').read_text())
exclude=set(first['prior_legacy_sample_excluded'])|{u['source_unit_key'] for sample in [first,second] for u in sample['selected']}
strata={s:[] for s in ['legacy','grammar','texts','historical']}
for a in audit:
 if not a['entry_keys'] or a['source_unit_key'] in exclude:continue
 u=a['evidence']
 if a['section']=='appendix1':s='legacy'
 elif u.get('lect','').startswith('historical'):s='historical'
 elif 'example' in u and int(''.join(c for c in u['example'].split('-')[0] if c.isdigit()))>=44:s='texts'
 else:s='grammar'
 strata[s].append(a)
rng=random.Random(20260927113);selected=[]
for s,n in [('legacy',5),('grammar',5),('texts',6),('historical',4)]:
 for a in rng.sample(strata[s],n):selected.append({'stratum':s,'source_unit_key':a['source_unit_key'],'printed_page':a['evidence']['printed_page'],'pdf_page':a['evidence']['printed_page']-260,'audit_unit':a,'actual_csv_rows':[after[k] for k in a['entry_keys']]})
(HERE/'independent-sample-20260927.json').write_text(json.dumps({'seed':20260927113,'status':'selected_not_audited','frozen_manifest_sha256':sha(HERE/'frozen-manifest.json'),'excluded_prior_source_units':sorted(exclude),'selected':selected},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'manifest_sha256':sha(HERE/'frozen-manifest.json'),'rows':648,'changed_tags':len(changes),'sample_units':20,'sample_rows':sum(len(x['actual_csv_rows']) for x in selected)}))
