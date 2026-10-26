"""Guarded owner reconciliation of every independently inventoried grammar child.

This produces review derivatives, never accepted/canonical exports.
"""
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
ROOT=next(p for p in HERE.parents if (p/'data/source_meta.py').exists())
PINS={
    'independent-grammar-review/independent-child-witnesses.jsonl':'16e4cf88d224a0528449d538f61f713dc13b93f51cbb0be9864468f2cb7b0050',
    'independent-grammar-review/independent-parent-coverage.jsonl':'4b08a534fc9ebaa8bbf0ca69e4f69536e8c12af393aa597ea4b860949881912f',
    'scope/physical-unit-census.jsonl':'4a4ba94662e25b66ead735bfbd8061332c530c359f194bcb2225bbc9ebc314cd',
}
FIX_KEY='grierson1906kolamibasim:p567:conjunctive-participle:1:independent:w5'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(path):
    return [json.loads(line) for line in path.read_text().splitlines()]

def write_jsonl(path,rows):
    path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))

def disposition(row):
    label=row['printed_label']
    if label=='Pronunciation' or 'Symbol witness' in row['qualification'] or label=='sometimes substituted for n' or label=='sometimes replaced by L' or label=='final before labial becomes m' or label=='final n before labial':
        return 'audit_only_notation_symbol', 'Quoted letter/typeform in pronunciation discussion; not an attested lexical head or identifiable morphological exponent.'
    if row['form']=='avnd' and row['language_label']=='Kolami':
        return 'audit_only_hypothesized_source_form', 'Grierson proposes avnd as origin of amd; retain literal hypothesis and attribution without treating it as independently attested lexical evidence.'
    return 'eligible_review_candidate', 'Printed lexical, whole-expression, explicitly functioned morphological or comparative witness; final metadata/profile/output acceptance pending.'

def main():
    for name,expected in PINS.items():
        assert sha(PACKAGE/name)==expected, f'Immutable input drift: {name}'
    inputs=load(PACKAGE/'independent-grammar-review/independent-child-witnesses.jsonl')
    parents=load(PACKAGE/'independent-grammar-review/independent-parent-coverage.jsonl')
    physical=load(PACKAGE/'scope/physical-unit-census.jsonl')
    structural={r['source_unit_key']:r for r in physical if r['unit_type'] not in {'physical-interlinear-line','documentary-heading'}}
    assert len(inputs)==224 and len({r['witness_key'] for r in inputs})==224
    assert len(parents)==103 and {r['physical_parent_key'] for r in parents}==set(structural)
    assert sum(r['unit_type']=='paradigm-cell' for r in parents)==58
    crop_evidence=json.loads((PACKAGE/'scope/grammar-control-locators.json').read_text())
    crop=next(r for r in crop_evidence if r['crop'].endswith('p567-conjunctive-last-owner-control.png'))
    reviewed=[];coverage=[];corrections=[]
    for original in inputs:
        row=copy.deepcopy(original)
        before=row['form']
        if row['witness_key']==FIX_KEY:
            assert before=='innekatānān'
            row['form']='innekādān'
            corrections.append(dict(witness_key=FIX_KEY,physical_parent_key=row['physical_parent_key'],field='form',
                old=before,new=row['form'],evidence='Owner isolated original shows k ā d ā n; independent reader Norton agrees. Earlier extra segments preserved only in historical input.',original_evidence=crop))
            # Historical uncertainty is retained as provenance, not silently erased.
            row['typed_uncertainties']=[]
        proposed_disposition,reason=disposition(row)
        row['source_unit_key']=row['witness_key']
        row['source_record']=copy.deepcopy(original)
        row['proposed_disposition']=proposed_disposition
        row['disposition_reason']=reason
        row['accepted_for_installation']=False
        row['owner_original_review']={
            'reviewer':'/root/rohru_finish',
            'status':'all original grammar pages read; child literal/gloss/scope reconciled; final output audit pending',
            'evidence':{'page':row['printed_page'],'original_image':row['original_image'],'original_image_sha256':row['original_image_sha256'],
                'original_control_crops':[r for r in crop_evidence if r['page']==row['printed_page']]},
            'optional_parentheses':'Literal; no generated optional-letter expansions',
            'source_labels':'Printed labels retained; p566 2.m. conflict is not silently repaired',
            'typed_uncertainty_policy':'Retain all unresolved quantities/shapes as source-local evidence; legible forms remain eligible rather than indefinite extraction holds.'}
        reviewed.append(row)
        coverage.append(dict(witness_key=row['witness_key'],physical_parent_key=row['physical_parent_key'],
            old_form=before,reviewed_form=row['form'],gloss=row['gloss'],printed_label=row['printed_label'],
            decision='literal correction independently confirmed' if before!=row['form'] else 'literal/gloss/context agree',
            proposed_disposition=proposed_disposition,reason=reason,
            unresolved=copy.deepcopy(row['typed_uncertainties']),accepted_for_installation=False))
    assert len(corrections)==1
    assert {r['source_unit_key'] for r in reviewed}=={r['witness_key'] for r in inputs}
    by_parent={key:[] for key in structural}
    for r in reviewed:by_parent[r['physical_parent_key']].append(r['source_unit_key'])
    parent_coverage=[]
    for r in parents:
        assert set(r['child_witness_keys'])==set(by_parent[r['physical_parent_key']])
        parent_coverage.append(dict(**r,owner_review='Original paragraph/cell/documentary context checked; no omitted parent',
            original_source_record=structural[r['physical_parent_key']],accepted_for_installation=False))
    write_jsonl(HERE/'reviewed-child-witnesses.jsonl',reviewed)
    write_jsonl(HERE/'owner-child-coverage.jsonl',coverage)
    write_jsonl(HERE/'owner-parent-coverage.jsonl',parent_coverage)
    status=dict(status='Complete owner original reconciliation; source-stage candidate only, no final acceptance',
        inputs=PINS,physical_parents=103,paradigm_cells=58,child_witnesses=224,
        disposition_counts=dict(Counter(r['proposed_disposition'] for r in reviewed)),
        per_page=dict(Counter(r['printed_page'] for r in reviewed)),corrections=corrections,
        source_caveats=['Grierson supplied interlinear translation; other mentioned materials are not printed here.',
            'Plural forms used in singular and vice versa; do not apply number globally from English.',
            'No feminine noun examples; qualified inferred gender system remains source claim.',
            'Optional letters literal; no unattested optional expansions.',
            'Basim copula printed2.m. conflicts with adjoining2/3n labels; retain typed conflict.',
            'Negative tense/agency interpretation explicitly tentative; no definitive subtype inferred.'],
        outputs={p.name:sha(p) for p in [HERE/'reviewed-child-witnesses.jsonl',HERE/'owner-child-coverage.jsonl',HERE/'owner-parent-coverage.jsonl']},
        gates_pending=['Root final reconciliation approval','Complete specimen glyph reconciliation/atomic and expression accounting',
            'Asur outside-chapter closure','Combined metadata/graph/profile/source-output independent audit','Canonical installation and all full-build/browser gates'],
        canonical_changes=False)
    (HERE/'owner-reconciliation-status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(status,ensure_ascii=False))

if __name__=='__main__':main()
