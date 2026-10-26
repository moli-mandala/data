"""Small, source-local integration census; no canonical or authored-input writes."""
from pathlib import Path
import collections
import hashlib
import json

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(name):return json.loads((SOURCE/name).read_text())
def lines(name):return [json.loads(x) for x in (SOURCE/name).read_text().splitlines()]

def main():
    paths=['table-metadata-candidate.json','table-prompt-inventory.jsonl','table157-recovery-review-20260927.json',
        'grammar-metadata-candidate.jsonl','grammar-scope-supplement-20260927.json',
        'specimens/interlinear-reviewed.jsonl','specimens/expression-emissions-reviewed.jsonl',
        'outsidechapter-independent-reviewed.jsonl','outside-comparands-reviewed-candidate-20260927.jsonl',
        'outside-comparands-supplement-reconciliation-20260927.json',
        'outside-comparands-routing-reuse-review-20260927.json','source-lect-mapping-approved-20260927.json',
        'metadata-profile-proposal/metadata-proposal.json','metadata-profile-proposal/literal-preservation-proposed.txt']
    atoms='specimens/atomic-emissions-reviewed.jsonl'
    pending=[]
    if (SOURCE/atoms).exists():paths.append(atoms)
    else:pending.append('Reviewed atomic emission derivative pending; do not emit all 890 physical atoms as separate lexical forms.')
    audit=[]
    def append(section,records,key='source_unit_key'):
        seen=set()
        for r in records:
            k=r[key]
            assert k not in seen,(section,k)
            seen.add(k)
            audit.append({'source_unit_key':k,'section':section,'input_record':r,'entry_keys':[],
                'integration_status':'awaiting explicit emission/disposition reconciliation'})
    table=lines('table-prompt-inventory.jsonl')
    append('table-physical-prompts',table,'key')
    readings=load('table-metadata-candidate.json')['readings']
    assert len(table)==241 and len(readings)==233
    assert {r['prompt_id'] for r in table}-{r['prompt_id'] for r in readings}=={133,136,195,196,197,198,199,200}
    grammar=lines('grammar-metadata-candidate.jsonl');append('grammar-physical',grammar)
    assert len(grammar)==432 and sum(not r['form'] for r in grammar)==16
    supplement=load('grammar-scope-supplement-20260927.json');append('grammar-interpretation-or-documentary',supplement['units'])
    append('specimen-physical-atoms',lines('specimens/interlinear-reviewed.jsonl'))
    append('specimen-expression-spans',lines('specimens/expression-emissions-reviewed.jsonl'))
    append('outside-target',lines('outsidechapter-independent-reviewed.jsonl'))
    append('outside-comparands',lines('outside-comparands-reviewed-candidate-20260927.jsonl'))
    assert len({r['source_unit_key'] for r in audit})==len(audit),'Cross-section duplicate source key requires explicit reconciliation'
    physical_atoms={r['source_unit_key'] for r in audit if r['section']=='specimen-physical-atoms'}
    for r in audit:
        if r['section']=='specimen-expression-spans':
            assert set(r['input_record']['atom_keys'])<=physical_atoms,r['source_unit_key']
    pending+=['Additional 38 comparator evidence units await root original acceptance; candidate is not frozen.',
        'Final language routing for source-specific identity holds and newly added comparator labels.',
        'Final combined literal profile coverage and actual-parser field/key/tag/citation verification.',
        'Final independent emitted-output review; no canonical installation authorized by this scaffold.']
    out=HERE/'source-unit-scaffold.jsonl'
    out.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in audit))
    report={'status':'integration scaffold, not lexical export','input_hashes':{p:sha(SOURCE/p) for p in paths},
        'source_unit_count':len(audit),'sections':dict(collections.Counter(r['section'] for r in audit)),
        'physical_token_count_policy':'Expression spans, derived comparative interpretations, and cross-reference expansions are not new printed tokens.',
        'pending':pending,'output_sha256':sha(out),'canonical_changes':False}
    (HERE/'scaffold-status.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','source_unit_count','sections']},indent=2))

if __name__=='__main__':main()
