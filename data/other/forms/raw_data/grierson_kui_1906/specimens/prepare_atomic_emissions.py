"""Keep physical audit atoms, but do not export printed line-wrap fragments."""
import collections
import copy
import json
from reconcile_specimen40_review import HERE, read, write, sha

def main():
    source=HERE/'interlinear-reviewed.jsonl'
    assert sha(source)=='d6960e5336320d539b83be14010b57fdf4b0183640c21e2936d64649843aee5d'
    atoms=read(source.name)
    bykey={r['source_unit_key']:r for r in atoms}
    join_path=HERE/'explicit-continuations-first-reading.json'
    joins=json.loads(join_path.read_text())
    starts={r['atom_keys'][0]:r for r in joins}
    tails={r['atom_keys'][1] for r in joins}
    output=[]
    audit=[]
    for atom in atoms:
        key=atom['source_unit_key']
        if key in tails:
            join=next(j for j in joins if key in j['atom_keys'])
            audit.append(dict(source_unit_key=key,emission_key=join['atom_keys'][0],
                              disposition='second physical part of explicitly continued word; no separate lexical head'))
            continue
        row=copy.deepcopy(atom)
        row['emission_key']=key
        row['physical_source_keys']=[key]
        row['physical_source_parts']=[dict(source_unit_key=key,form=atom['raw_source_form'],gloss=atom['raw_gloss'])]
        row['emission_form']=atom['raw_source_form']
        row['emission_gloss']=atom['raw_gloss']
        row['typed_uncertainties']=[dict(source_unit_key=key,evidence=atom['typography_uncertainty'])] if atom['typography_uncertainty'] else []
        if key in starts:
            join=starts[key]
            second=bykey[join['atom_keys'][1]]
            assert atom['parent_line']!=second['parent_line']
            assert [atom['raw_source_form'],second['raw_source_form']]==join['raw_form_parts']
            assert atom['raw_source_form']+second['raw_source_form']==join['combined_form']
            row['physical_source_keys']=join['atom_keys']
            row['physical_source_parts'].append(dict(source_unit_key=second['source_unit_key'],form=second['raw_source_form'],gloss=second['raw_gloss']))
            row['emission_form']=join['combined_form']
            row['emission_gloss']=join['combined_gloss']
            if second['typography_uncertainty']:
                row['typed_uncertainties'].append(dict(source_unit_key=second['source_unit_key'],evidence=second['typography_uncertainty']))
            row['boundary_decision']='Explicit printed word continuation; one lexical emission with both physical locators.'
        row['tags']=['uncertain'] if row['typed_uncertainties'] else []
        row['title_policy']='title atom retained; final title emission policy remains explicit' if atom['unit_type']=='title_atom' else None
        output.append(row)
        audit.append(dict(source_unit_key=key,emission_key=key,disposition='joined-word first part' if key in starts else 'one physical atom to one emission'))
    assert len(output)==888 and len(audit)==890
    assert len({r['emission_key'] for r in output})==888
    assert collections.Counter(k for r in output for k in r['physical_source_keys'])==collections.Counter(list(bykey))
    assert not tails & {r['emission_key'] for r in output}
    assert sum(r['unit_type']=='title_atom' for r in output)==6
    write('atomic-emissions-reviewed.jsonl',output)
    write('atomic-emission-accounting.jsonl',audit)
    result=dict(status='two explicit continuations joined; no separate fragment heads',
                input_sha256=sha(source),join_input_sha256=sha(join_path),physical_atoms=890,atomic_emissions=888,
                title_atoms=6,non_title_emissions=882,all_physical_keys_accounted_once=True,
                joined_units=[dict(emission_key=r['emission_key'],physical_keys=r['physical_source_keys'],form=r['emission_form']) for r in output if len(r['physical_source_keys'])>1],
                outputs={n:sha(HERE/n) for n in ['atomic-emissions-reviewed.jsonl','atomic-emission-accounting.jsonl']},
                parser_fixture_note='Earlier1225-row fixture tested890 physical atoms plus258table and77expression candidates; not final lexical inventory. Use888atomic emissions in integrated export.',canonical_changes=False)
    (HERE/'atomic-emission-validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
