"""Guarded native-witness derivative; never overwrites historical proposals."""
import copy
import hashlib
import json
from pathlib import Path
from prepare_emission_review import PINS, JOIN_FIRST, JOIN_TAIL

HERE = Path(__file__).resolve().parent
OUT = HERE / 'sharp-reviewed-revision'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p): return json.loads(p.read_text())
def write(p,rows): p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))

def main():
    inputs = {}
    physical = []
    for page,pin in PINS.items():
        path=HERE/f'p{page}-first-reading.jsonl'
        assert sha(path)==pin
        inputs[str(path.relative_to(HERE.parent))]=pin
        physical.extend(a for line in path.read_text().splitlines() for a in json.loads(line)['target_atoms'])
    raw_by={a['source_unit_key']:a for a in physical}
    assert len(raw_by)==len(physical)==561
    overlay=[]
    for name in ['sharp-p564-review-decisions.json','sharp-p565-narrow-controls.json','sharp-p568-p569-review-decisions.json']:
        path=HERE/name; inputs[str(path.relative_to(HERE.parent))]=sha(path)
        report=load(path)
        if name=='sharp-p565-narrow-controls.json':
            for d in report['decisions']:
                if d['disposition']=='proposed_correction':
                    overlay.append(dict(d,new=d['reviewed_value']))
        else: overlay.extend(report['corrections'])
    old=HERE/'owner-literal-corrections-p564.json'
    inputs[str(old.relative_to(HERE.parent))]=sha(old)
    # Three earlier underdot corrections remain visually confirmed in the
    # native source; the old ectnā proposal is explicitly superseded by setnā.
    for d in load(old)['corrections']:
        if d['new'] != 'ectnā': overlay.append(d)
    punct=HERE.parent/'specimen-conservation-review/sharper-punctuation-correction-proposal.json'
    assert sha(punct)=='6365092d4cc395295aebb7d8238d85fa67661c9fd8ae3bb0ab4703fbc56de6c4'
    inputs[str(punct.relative_to(HERE.parent))]=sha(punct)
    overlay.extend(load(punct)['corrections'])
    assert len({(d['source_unit_key'],d['field']) for d in overlay})==len(overlay)
    atoms=copy.deepcopy(physical); by={a['source_unit_key']:a for a in atoms}
    for d in overlay:
        key=d['source_unit_key']; field=d['field']
        assert by[key][field]==d['old'], (key,by[key],d)
        if 'raw_interlinear_gloss' in d:
            assert by[key]['raw_interlinear_gloss']==d['raw_interlinear_gloss']
        by[key][field]=d['new']
        by[key].setdefault('native_review_changes',[]).append(d)
    uncertainty={
        'grierson1906kolamibasim:p564:specimen75-line:16:word:1':'Native source-bar tūltnā provisional; prior diaeresis hypothesis rejected after pixel-controlled bar comparison. No phonemic inference.',
        'grierson1906kolamibasim:p565:specimen75-line:7:word:6':'Retain vintēr.; medial cluster/lower-mark damage does not justify retroflex insertion.',
        'grierson1906kolamibasim:p569:specimen76-line:9:word:7':'Retain kasbirā; small ink adjoining r may be its shoulder or detached mark. No inserted punctuation.'}
    for key,reason in uncertainty.items():
        by[key]['typed_uncertainties']=[dict(type='source-glyph-identity',reason=reason)]
    emissions=[];audit=[]
    for a in atoms:
        key=a['source_unit_key']; raw=raw_by[key]
        audit.append(dict(source_unit_key=key,raw_source_record=raw,reviewed_source_record=a,
                          proposed_entry_keys=[JOIN_FIRST if key==JOIN_TAIL else key],
                          disposition='explicit joined continuation tail' if key==JOIN_TAIL else 'atomic emission proposal',
                          accepted_for_installation=False))
        if key==JOIN_TAIL: continue
        e=copy.deepcopy(a);e['physical_source_keys']=[key]
        e['raw_source_record']=raw;e['accepted_for_installation']=False
        if key==JOIN_FIRST:
            assert a['raw_source_form']=='Bāp-' and by[JOIN_TAIL]['raw_source_form']=='phōkān'
            e.update(raw_source_form='Bāp-phōkān',raw_interlinear_gloss='Father-near',
                     physical_source_keys=[JOIN_FIRST,JOIN_TAIL],source_parts=[raw,raw_by[JOIN_TAIL]],
                     join_evidence='Explicit printed line-break compound; joined form is source-layout reconstruction, not a single physical cell.')
        emissions.append(e)
    expressions=[]
    for specimen in [75,76]:
        selected=[a for a in atoms if f':specimen{specimen}-line:' in a['source_unit_key']]
        pending=[]; depth=0
        for i,a in enumerate(selected):
            pending.append(a);form=a['raw_source_form']
            closed=form.count('’')+form.count('”')
            depth+=form.count('‘')+form.count('“')-closed
            assert depth>=0
            quote_boundary=closed>0 and depth==0
            punct_boundary=depth==0 and form.endswith(('.', '?', '!'))
            if not (quote_boundary or punct_boundary or i==len(selected)-1): continue
            keys=[a['source_unit_key'] for a in pending]
            forms=[a['raw_source_form'] for a in pending]
            glosses=[a['raw_interlinear_gloss'] for a in pending]
            if JOIN_FIRST in keys:
                j=keys.index(JOIN_FIRST);assert keys[j+1]==JOIN_TAIL
                forms[j:j+2]=['Bāp-phōkān'];glosses[j:j+2]=['Father-near']
            expressions.append(dict(source_unit_key=keys[0]+':expression',specimen=specimen,
                physical_source_keys=keys,first_source_key=keys[0],last_source_key=keys[-1],
                printed_pages=sorted({a['printed_page'] for a in pending}),
                raw_source_form=' '.join(forms),raw_aligned_interlinear_gloss=' '.join(glosses),
                boundary_evidence='Whole printed quotation' if quote_boundary else 'Printed terminal punctuation' if punct_boundary else 'Source specimen end',
                accepted_for_installation=False,raw_physical_atoms=copy.deepcopy(pending)))
            pending=[]
        assert not pending and depth==0
    assert len(emissions)==560 and len(audit)==561
    assert [k for e in expressions for k in e['physical_source_keys']]==list(raw_by)
    historical=HERE/'expression-emission-proposal.jsonl'
    inputs[str(historical.relative_to(HERE.parent))]=sha(historical)
    old_expressions=[json.loads(s) for s in historical.read_text().splitlines()]
    changes=[]
    for e in expressions:
        intersect=[o for o in old_expressions if set(o['physical_source_keys']) & set(e['physical_source_keys'])]
        if len(intersect)!=1 or intersect[0]['physical_source_keys']!=e['physical_source_keys']:
            changes.append(dict(new_anchor=e['source_unit_key'],new_contributors=e['physical_source_keys'],
                                historical_anchors=[o['source_unit_key'] for o in intersect],
                                historical_contributors=[o['physical_source_keys'] for o in intersect]))
    OUT.mkdir(exist_ok=True)
    write(OUT/'reviewed-physical-atoms.jsonl',atoms)
    write(OUT/'atomic-emission-proposal.jsonl',emissions)
    write(OUT/'physical-atom-emission-audit.jsonl',audit)
    write(OUT/'expression-emission-proposal.jsonl',expressions)
    (OUT/'keyed-literal-overlay.json').write_text(json.dumps(overlay,ensure_ascii=False,indent=2)+'\n')
    (OUT/'expression-boundary-history.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2)+'\n')
    status=dict(status='Native reviewed derivative awaiting independent conservation and combined-output audit; no install',
                inputs=inputs,physical_atoms=561,atomic_emissions=560,expressions=len(expressions),
                field_changes=len(overlay),typed_glyph_uncertainties=uncertainty,
                outputs={p.name:sha(p) for p in OUT.iterdir() if p.name!='status.json'},canonical_changes=False)
    (OUT/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:status[k] for k in ['physical_atoms','atomic_emissions','expressions','field_changes']}))

if __name__=='__main__':main()
