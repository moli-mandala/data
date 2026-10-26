"""Nine exact root-approved grammar tag removals, preserving frozen r4 and sample."""
import argparse,copy,csv,hashlib,json,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def write(p,rr):p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rr))
def main(approval):
    assert sha(approval)=='7c94146db5e30d1c972e937b1bd8993326ce9da24a9d3cd760174e6facd8754b'
    source=HERE/'revision4';out=HERE/'revision5'
    a=json.loads(approval.read_text());assert a['status']=='accepted_source_local_metadata_revision'
    assert a['candidate_sha256']==sha(source/'combined-review-candidate.csv')=='944c6b0cf1edec453a0531e130af95d9832b29df54b40c23e214f3ba996fb87d'
    decisions=a['decisions'];assert len(decisions)==len({d['source_unit_key'] for d in decisions})==9
    with (source/'combined-review-candidate.csv').open() as f:old=list(csv.reader(f))
    rr=copy.deepcopy(old);by={r[10]:r for r in rr}
    objkeys={r[10] for r in rr if 'obj' in r[14].split()}
    assert objkeys=={d['source_unit_key'] for d in decisions}
    for d in decisions:
        r=by[d['source_unit_key']]
        assert r[2]==d['exact_form_guard'] and r[3]==d['exact_raw_gloss_guard']
        assert r[14].split()==d['old_tags']
        assert d['new_tags']==[t for t in r[14].split() if t!='obj']
        r[14]=' '.join(d['new_tags'])
    assert sum(x!=y for x,y in zip(old,rr))==9 and all(x[:14]==y[:14] for x,y in zip(old,rr))
    audit=rows(source/'combined-review-audit.jsonl')
    for r in audit:
        if r['source_unit_key'] in objkeys:
            r['actual_row']=by[r['source_unit_key']]
            r['whole_source_pronoun_consistency']={'approval_path':str(approval),'approval_sha256':sha(approval),'reason':'English oblique pronoun gloss is not itself an explicit target object-function label.'}
    out.mkdir(exist_ok=True)
    for n in ['grierson-kolami-basim-1906.txt','unicode-inventory.json','combined-review-candidate.yaml','specimen-metadata-proposal.jsonl','references-review-candidate.bib']:
        shutil.copyfile(source/n,out/n)
    with (out/'combined-review-candidate.csv').open('w',newline='') as f:csv.writer(f).writerows(rr)
    write(out/'combined-review-audit.jsonl',audit)
    grammar_source=PACKAGE/'grammar-review/sharp-reviewed-revision3';grammar_out=PACKAGE/'grammar-review/whole-metadata-consistency-revision';grammar_out.mkdir(exist_ok=True)
    tags=rows(grammar_source/'grammar-tag-proposal.jsonl');graphs=rows(grammar_source/'grammar-graph-proposal.jsonl')
    for t in tags:
        if t['witness_key'] in objkeys:
            assert 'obj' in t['proposed_tags']
            t['historical_pre_consistency_metadata']=copy.deepcopy(t)
            t['proposed_tags']=[x for x in t['proposed_tags'] if x!='obj']
            t['whole_source_scope_rationale']='Printed personal-pronoun list and English gloss support person/number; English me/him/thee alone does not establish target syntactic object function.'
    for g in graphs:
        if g['witness_key'] in objkeys:
            assert 'obj' in g['proposed_tags_guard']
            g['historical_pre_consistency_tag_guard']=g['proposed_tags_guard'][:]
            g['proposed_tags_guard']=[x for x in g['proposed_tags_guard'] if x!='obj']
    write(grammar_out/'grammar-tag-proposal.jsonl',tags);write(grammar_out/'grammar-graph-proposal.jsonl',graphs)
    shutil.copyfile(grammar_source/'reviewed-child-witnesses.jsonl',grammar_out/'reviewed-child-witnesses.jsonl')
    continuity={'status':'Exactly9 object-tag removals align grammar with accepted specimen source policy',
      'root_approval_sha256':sha(approval),'old_candidate_sha256':sha(source/'combined-review-candidate.csv'),
      'counts':{'rows':827,'changed_tags':9,'changed_other_fields':0},'decisions':decisions,'canonical_changes':False}
    (out/'pronoun-consistency.json').write_text(json.dumps(continuity,ensure_ascii=False,indent=2)+'\n')
    status=json.loads((source/'status.json').read_text());status['input_pins']['whole_pronoun_consistency_root_approval']=sha(approval)
    status['outputs']={n:sha(out/n) for n in ['combined-review-candidate.csv','combined-review-audit.jsonl','specimen-metadata-proposal.jsonl','grierson-kolami-basim-1906.txt','unicode-inventory.json','combined-review-candidate.yaml']}
    (out/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    # This reuses the exact sampled source units, never draws another sample.
    sample=json.loads((source/'fresh-output-selection-20260927.json').read_text())
    sample['historical_r4_selection_sha256']=sha(source/'fresh-output-selection-20260927.json')
    sample['candidate_sha256']=sha(out/'combined-review-candidate.csv');sample['audit_sha256']=sha(out/'combined-review-audit.jsonl')
    for field in ['sample','targeted_controls']:
        for item in sample[field]:item['actual_row']=by[item['source_unit_key']]
    for item in sample['literal_repair_regression_controls']:item['actual_emitted_row']=by.get(item['source_unit_key'])
    sample['revision_note']='Same20 selection re-pinned after9Tags-only grammar consistency repair; no independent final20 acceptance existed before this revision.'
    (out/'fresh-output-selection-20260927.json').write_text(json.dumps(sample,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'rows':827,'obj_tags_remaining':sum('obj' in r[14].split() for r in rr),'csv_sha256':sha(out/'combined-review-candidate.csv'),'audit_sha256':sha(out/'combined-review-audit.jsonl')}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--approval',required=True,type=Path);a=p.parse_args();main(a.approval)
