import hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;O=H.parent/'interlinear-review'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def readrows(p):return [json.loads(s) for s in p.read_text().splitlines()]
s=json.loads((H/'pass2-selection.json').read_text())
for name,h in s['input_hashes'].items():assert sha(O/name)==h,name
old=readrows(O/'before-grammar-repair/continuous-semantic-proposal.jsonl')
new=readrows(O/'continuous-semantic-proposal.jsonl')
assert len(old)==len(new)==790
changes=[]
for a,b in zip(old,new):
    assert a['source_unit_key']==b['source_unit_key']
    fields=[f for f in set(a)|set(b) if a.get(f)!=b.get(f)]
    assert set(fields)<={'tags','grammar_review'}
    assert set(a['tags'])<=set(b['tags'])
    if fields:changes.append(b['source_unit_key'])
assert len(changes)==90
assert all(c['all_recommended_present'] and c['literal_continuity'] for c in s['class_checks'])
inputs=json.loads((O/'continuous-review-input.json').read_text())
for x in inputs:
    rr=[r for r in new if r['physical_p_key']==x['key']]
    assert ' '.join(r['form'] for r in rr)==' '.join(x['form_tokens'])
    assert ' '.join(r['raw_interlinear_gloss'] for r in rr)==x['gloss']
oldblocks=set(s['excluded_physical_blocks'])
assert not oldblocks & {r['physical_p_key'] for r in s['selected_rows']}
assert len({r['physical_p_key'] for r in s['selected_rows']})==20
comments={1:'Original will study is retained literally; no automatic target future paradigm is inferred from an English auxiliary.',5:'Printed digging retained without automatic present-participle analysis.',6:'Printed them supports pron/3pl/pl.',13:'Two printed target tokens share force; blank lexical gloss and uncertain internal alignment retained.',16:'Printed my supports first-person singular possessive analysis.',17:'Two printed target tokens share small; joint evidence retained with blank lexical gloss and uncertainty.',18:'Printed our supports first-person plural possessive analysis; no inclusive/exclusive inference.'}
report=dict(status='passed',reviewer='rohru_finish',seed=s['seed'],input_hashes=s['input_hashes'],selection_sha256=sha(H/'pass2-selection.json'),prior_failed_report_sha256=s['previous_report_sha256'],counts=dict(rows=790,partitions=85,grammar_recommendations_verified=90,fresh_original_sample=20,material_errors=0),source_original=dict(path='tmp/pdfs/korwa-kodaku-2022/source.pdf',sha256='a8efbe88405e27024a7a6ec786cd6fde3e382f0eaf0d0081197d3880ed97eb0c',pdf_pages=list(range(94,103)),printed_pages=list(range(84,93)),method='Fresh visual comparison of selected p/g blocks on each original page; one overwritten render. Selection excludes all20 previously sampled physical blocks.'),class_verification=dict(baseline_sha256=sha(O/'before-grammar-repair/continuous-semantic-proposal.jsonl'),changed_rows=90,only_changed_fields=['tags','grammar_review'],prior_tags_and_all_literal_fields_preserved=True,partition_conservation='85/85 pass',checks=s['class_checks']),sample=[dict(sample_number=i,actual_proposal=r,original_pages=r['source_pages'],literal_gloss='pass',semantic_boundary='pass',metadata_scope='pass',finding=comments.get(i,'Original form, ordered source gloss, scope and current metadata agree.')) for i,r in enumerate(s['selected_rows'],1)],limits=['Acceptance concerns these continuous-span proposals and corrected grammar class, not whole-source installation.','64 unresolved internal-alignment spans remain explicitly typed; no invented word meanings.','Printed source anomalies retained as attributed evidence; no target morphology inferred solely from English inflection.','Canonical files unchanged; earlier failed sample preserved.'])
(H/'independent-continuous-review-pass2.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('PASS',report['counts']);print(sha(H/'independent-continuous-review-pass2.json'))
