"""Source-local emission proposals; physical atoms and first readings stay immutable.

Whole quoted utterances remain together. Physical lines are never treated as
synthetic sentence boundaries. Every proposed expression needs original review.
"""
import copy
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
PINS={564:'9231c09556f8e658f823363bfbf408bc57651ec053677f8a02e75e19ce1974b7',
      565:'be018c7f7826a0f24fdcd7f2de67230a213665f3dc5bc0eaf3b8a30726ce0323',
      568:'7ac1c77dbb6f03e2f530ef13ce18eb0a2212fa173f97165418ec8d02ce54c2fd',
      569:'a831c8b55fca88ff9c3a1c44f937c3c844932e03a1a2343ab34b0fff15a71f36'}
JOIN_FIRST='grierson1906kolamibasim:p568:specimen76-line:11:word:8'
JOIN_TAIL='grierson1906kolamibasim:p568:specimen76-line:12:word:1'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,rows):path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))

def main():
    pages=[]
    for page,expected in PINS.items():
        path=HERE/f'p{page}-first-reading.jsonl'
        assert sha(path)==expected
        pages.extend(json.loads(line) for line in path.read_text().splitlines())
    physical=[a for page in pages for a in page['target_atoms']]
    assert len(physical)==561 and len({a['source_unit_key'] for a in physical})==561
    corrections=json.loads((HERE/'owner-literal-corrections-p564.json').read_text())
    atoms=copy.deepcopy(physical);by_key={a['source_unit_key']:a for a in atoms}
    for decision in corrections['corrections']:
        a=by_key[decision['source_unit_key']]
        assert a[decision['field']]==decision['old']
        a[decision['field']]=decision['new']
        a['literal_correction_proposal']=decision
    first=by_key[JOIN_FIRST];tail=by_key[JOIN_TAIL]
    assert first['raw_source_form']=='Bāp-' and tail['raw_source_form']=='phōkān'
    emissions=[];audit=[]
    for a in atoms:
        key=a['source_unit_key']
        raw=next(r for r in physical if r['source_unit_key']==key)
        if key==JOIN_TAIL:
            audit.append(dict(source_unit_key=key,raw_source_record=raw,proposed_entry_keys=[JOIN_FIRST],
                disposition='continuation tail mapped to joined lexical proposal',accepted_for_installation=False))
            continue
        e=copy.deepcopy(a)
        e['physical_source_keys']=[key]
        e['raw_source_record']=raw
        e['accepted_for_installation']=False
        if key==JOIN_FIRST:
            e['raw_source_form']='Bāp-phōkān'
            e['raw_interlinear_gloss']='Father-near'
            e['physical_source_keys']=[JOIN_FIRST,JOIN_TAIL]
            e['source_parts']=[raw,next(r for r in physical if r['source_unit_key']==JOIN_TAIL)]
            e['join_evidence']='Explicit terminal hyphen on Bāp- and next physical-line continuation phōkān. Joined Original is a documented source-layout reconstruction, not a literal single-cell witness.'
        emissions.append(e)
        audit.append(dict(source_unit_key=key,raw_source_record=raw,proposed_entry_keys=[key],
            disposition='atomic lexical emission proposal',accepted_for_installation=False))
    assert len(emissions)==560 and len(audit)==561
    assert {r['source_unit_key'] for r in audit}==set(by_key)
    expressions=[]
    for specimen in [75,76]:
        selected=[a for a in atoms if f':specimen{specimen}-line:' in a['source_unit_key']]
        pending=[];depth=0
        for i,a in enumerate(selected):
            pending.append(a)
            form=a['raw_source_form']
            opened=form.count('‘')+form.count('“')
            closed=form.count('’')+form.count('”')
            before=depth
            depth+=opened-closed
            assert depth>=0, a['source_unit_key']
            # A completed whole quotation is an explicit source boundary even
            # when the compositor ends it with comma+closing quote.
            quote_boundary=closed>0 and depth==0
            period_boundary=depth==0 and form.endswith(('.', '?', '!'))
            if quote_boundary or period_boundary or i==len(selected)-1:
                keys=[p['source_unit_key'] for p in pending]
                forms=[p['raw_source_form'] for p in pending]
                glosses=[p['raw_interlinear_gloss'] for p in pending]
                if JOIN_FIRST in keys:
                    j=keys.index(JOIN_FIRST)
                    assert keys[j+1]==JOIN_TAIL
                    forms[j:j+2]=['Bāp-phōkān']
                    glosses[j:j+2]=['Father-near']
                index=len([r for r in expressions if r['specimen']==specimen])+1
                expressions.append(dict(source_unit_key=keys[0]+':expression',
                    specimen=specimen,display_sequence=index,physical_source_keys=keys,printed_pages=sorted({p['printed_page'] for p in pending}),
                    first_source_key=keys[0],last_source_key=keys[-1],raw_source_form=' '.join(forms),
                    raw_aligned_interlinear_gloss=' '.join(glosses),
                    boundary_evidence='Whole completed printed quotation' if quote_boundary else 'Printed terminal punctuation' if period_boundary else 'Source specimen end',
                    proposed_scope='source-bounded complete-expression candidate; sentence/phrase classification requires review',
                    accepted_for_installation=False,raw_physical_atoms=copy.deepcopy(pending)))
                pending=[]
        assert depth==0 and not pending
    flat=[key for e in expressions for key in e['physical_source_keys']]
    assert flat==[a['source_unit_key'] for a in atoms]
    write(HERE/'atomic-emission-proposal.jsonl',emissions)
    write(HERE/'physical-atom-emission-audit.jsonl',audit)
    write(HERE/'expression-emission-proposal.jsonl',expressions)
    status=dict(status='Full specimen atomic/expression review proposals; no accepted final export',
        inputs={f'p{page}-first-reading.jsonl':pin for page,pin in PINS.items()},
        correction_proposal_sha256=sha(HERE/'owner-literal-corrections-p564.json'),
        physical_atoms=561,atomic_emissions=560,explicit_line_break_joins=1,
        expression_proposals=len(expressions),per_specimen_expressions={str(s):sum(e['specimen']==s for e in expressions) for s in [75,76]},
        conservation='Every physical atom occurs exactly once in the ordered expression partition and in physical atom audit; explicit join tail maps to first key.',
        outputs={p.name:sha(p) for p in [HERE/'atomic-emission-proposal.jsonl',HERE/'physical-atom-emission-audit.jsonl',HERE/'expression-emission-proposal.jsonl']},
        pending=['Original expression boundary/alignment and sentence/phrase metadata review','Root remaining specimen glyph controls',
            'Final combined-output independent audit and all later source-stage/full-build gates'],canonical_changes=False)
    (HERE/'emission-review-status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(status))

if __name__=='__main__':main()
