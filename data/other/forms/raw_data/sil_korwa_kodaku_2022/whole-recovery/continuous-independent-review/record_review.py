"""Record independent original-page decisions; never edits the reviewed proposal."""
import collections
import hashlib
import json
import re
from pathlib import Path

H = Path(__file__).resolve().parent
O = H.parent / 'interlinear-review'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
selection = json.loads((H/'selection.json').read_text())
resolved_inputs={}
for name, digest in selection['input_hashes'].items():
    matches=[p for p in [O/'revision1'/name,O/name] if p.exists() and sha(p)==digest]
    assert matches, name
    resolved_inputs[name]=matches[0]
rows = [json.loads(s) for s in resolved_inputs['continuous-semantic-proposal.jsonl'].read_text().splitlines()]
coverage = json.loads((H/'partition-conservation.json').read_text())
assert len(rows) == 790 and len(coverage['blocks']) == 85
pronouns = {
    'I':['pron','1sg','sg'], 'i':['pron','1sg','sg'],
    'I/my':['pron','1sg','sg'], 'my':['pron','1sg','sg','poss'],
    'we':['pron','1pl','pl'], 'our':['pron','1pl','pl','poss'],
    'he':['pron','3sg','sg'], 'him':['pron','3sg','sg'],
    'they':['pron','3pl','pl'], 'them':['pron','3pl','pl'],
    'you':['pron'], 'your':['pron','poss'],
}
cardinals = {'one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','fifteen'}
ordinals = {'first','second','third','fourth','fifth','sixth','seventh','eighth','ninth','tenth','eleventh'}
recommendations=[]
for r in rows:
    if r['target_token_range'][0] != r['target_token_range'][1] or not r['gloss']:
        continue
    g=r['gloss']; add=[]; category=None
    if g in pronouns:
        add=pronouns[g]; category='explicit-pronominal-gloss'
    elif g in cardinals or re.fullmatch(r'\d+',g):
        add=['num']; category='explicit-cardinal-gloss'
    elif g in ordinals or re.fullmatch(r'\d+(?:st|nd|rd|th)(?:-also)?',g):
        add=['num','ord']; category='explicit-ordinal-gloss'
    if add:
        new=list(dict.fromkeys(r['tags']+add))
        if new != r['tags']:
            recommendations.append(dict(source_unit_key=r['source_unit_key'],physical_p_key=r['physical_p_key'],source_pages=r['source_pages'],form=r['form'],raw_interlinear_gloss=r['raw_interlinear_gloss'],field='tags',old=r['tags'],proposed=new,category=category,evidence='Printed g-line meaning in the reviewed original block; classification describes the attributed source analysis, not independently inferred target morphology.',qualification='No case inferred; no inclusive/exclusive distinction; English you does not establish singular/plural. I/my remains case/possessive-ambiguous.' if category=='explicit-pronominal-gloss' else 'Numeral/ordinal function explicit in this printed gloss; no target inflection inferred.'))
bykey={r['source_unit_key']:r for r in recommendations}
sample=[]
for i,r in enumerate(selection['selected_rows'],1):
    issue=bykey.get(r['source_unit_key'])
    sample.append(dict(sample_number=i,actual_proposal=r,original_pdf_pages=r['source_pages'],literal_gloss_alignment='pass',scope_uncertainty='pass',metadata='fail' if issue else 'pass',material_error=issue))
assert [s['sample_number'] for s in sample if s['metadata']=='fail']==[8,20]
blocks=[]
for b in coverage['blocks']:
    rr=[r for r in rows if r['physical_p_key']==b['key']]
    blocks.append(dict(source_unit_key=b['key'],index=b['index'],source_pages=rr[0]['source_pages'],emitted_proposal_count=len(rr),conservation=b['conservation'],semantic_boundary_review='Original p/g block compared with every proposed partition; no definite boundary correction identified. Unresolved multi-token spans retain blank gloss and original joint analysis.',metadata_correction_keys=[r['source_unit_key'] for r in recommendations if r['physical_p_key']==b['key']]))
anomalies=[
    dict(block_index=25,description='C.2 sir/study(past)/I is printed against saheb/pɐɖʰau/kɪjĕ. Preserve attributed I; do not silently repair using resemblance to another language.'),
    dict(block_index=47,description='C.4 printed were alignment includes t̪o. English were alone establishes neither target plural nor a tense paradigm; no new such tag proposed.'),
    dict(block_index=54,description='C.4 final uʈain is glossed too in the g-line. Preserve that reading rather than silently replacing it with took from the free translation.'),
    dict(block_index=57,description='C.4 sasɐn bʰedʒis shares the remaining government gloss. Joint source evidence is retained, internal alignment remains unresolved.'),
    dict(block_index=82,description='C.4 initial ɐb bɐtʃtʃamɔnla is retained as a joint unresolved span for children of; no false independent ab=children row accepted.'),
]
report=dict(status='failed-metadata-class-repair-required',reviewer='rohru_finish',independent_of_proposal_author=True,input_hashes=selection['input_hashes'],selection_sha256=sha(H/'selection.json'),seed=selection['seed'],original=dict(path='tmp/pdfs/korwa-kodaku-2022/source.pdf',sha256='a8efbe88405e27024a7a6ec786cd6fde3e382f0eaf0d0081197d3880ed97eb0c',pdf_pages=list(range(93,106)),printed_pages=list(range(83,96)),method='All 85 source p/g blocks and all proposed partitions read; originals PDF93–105 visually compared. Fresh20 actual rows inspected separately. This is not a claim of a new exhaustive glyph-by-glyph transcription of all790 rows.'),counts=dict(partitions=85,proposal_rows=790,multi_token_unresolved=sum(r['target_token_range'][0]!=r['target_token_range'][1] for r in rows),fresh_sample=20,sample_material_errors=2,literal_or_boundary_errors_found=0,tag_recommendations=len(recommendations),recommendation_classes=dict(collections.Counter(r['category'] for r in recommendations))),sample=sample,partition_reviews=blocks,keyed_recommendations=recommendations,source_anomalies_preserved=anomalies,policy=['Explicit parenthetical past/present/future labels retained as source labels.','No automatic present participle from English -ing, tense from went/were, ergative from by, or singular/plural from you.','Ambiguous multi-token groups are not accepted as individually glossed word equivalents merely because counts can be made equal.','Question marks/cm remain unresolved source analyses.','No original or owner file changed. Root reconciliation and a new frozen output audit remain required.'])
report['exact_reviewed_input_locations']={k:str(v) for k,v in resolved_inputs.items()}
report['revision_handling']='Owner archived revision1 during final report preparation. Every reviewed byte is resolved by the original selection hash; current repaired outputs are not silently substituted.'
(H/'independent-continuous-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report['counts'],indent=2))
print('report_sha256',sha(H/'independent-continuous-review.json'))
