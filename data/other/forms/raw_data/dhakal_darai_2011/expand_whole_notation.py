"""Apply individually recorded source-notation decisions to review metadata only."""
import copy
import hashlib
import json
import re
from pathlib import Path
from interlinear_grammar import parse
from whole_attribution import apply_attribution
from whole_translation_groups import reconcile as reconcile_translation_groups
P=Path(__file__).resolve().parent


def retain_agreement_roles(record):
    """Keep table arguments distinct; case headings describe agreement, not verb case."""
    claims=record['source_claims']; roles={}
    fields={'subject_label':'subject','source_subject':'subject',
            'object_label':'object','source_object':'object',
            'source_receptor':'receptor','source_agreement':'agreement',
            'source_person_number':'agreement'}
    for field,role in fields.items():
        if field not in claims:
            continue
        value=claims[field]
        assert role not in roles or roles[role]['source_label']==value
        alternatives=[]
        for label in value.split('/'):
            match=re.fullmatch(r'([123])(SG|PL)(\.H)?',label)
            assert match, (record['source_unit_key'],label)
            alternatives.append({'person':int(match[1]),
                'number':'singular' if match[2]=='SG' else 'plural',
                'honorific':bool(match[3]),'literal_label':label})
        roles[role]={'source_label':value,'alternatives':alternatives}
    if roles:
        record['source_agreement_roles']=roles
        record['source_role_notes']=['Source '+role+' label: '+value['source_label']+'.'
                                     for role,value in roles.items()]
        if claims.get('source_agreement_type'):
            record['source_role_notes'].append('Source agreement heading: '+claims['source_agreement_type']+'.')
    # Only explicit tense columns map here; root headings retain their own analysis.
    if claims.get('source_tense') and not record['disposition'].startswith('audit-only'):
        mapped=parse(claims['source_tense'])
        record['proposed_grammar']['tags']=list(dict.fromkeys(record['proposed_grammar']['tags']+mapped['tags']))
    if claims.get('source_mood') and record['kind']=='paradigm-inflection':
        tag={'Counterfactual':'counterfactual','Possibility':'possibility',
             'Inference1':'inferential','Inference2':'inferential','Optative':'opt'}[claims['source_mood']]
        record['proposed_grammar']['tags']=list(dict.fromkeys(record['proposed_grammar']['tags']+[tag,'verb']))
    if record['kind'] in {'bound-morpheme','bound-morpheme-repeat','bound-morphology','derivation-affix'}:
        form=record['form']
        marker='affix'
        if form.startswith('-') and not form.endswith('-'):marker='suffix'
        elif form.endswith('-') and not form.startswith('-'):marker='prefix'
        record['proposed_grammar']['tags']=list(dict.fromkeys(record['proposed_grammar']['tags']+[marker]))
    # These are reviewed source column/category fields, never words searched in
    # arbitrary definitions. Semantic headings without an exact tag stay notes.
    category_tags={'manner':['manner'],'Manner':['manner'],
        'location/direction':['spatial'],'location':['spatial'],'Location':['spatial'],
        'time':['temporal'],'Time':['temporal'],'interjection':['interj'],
        'onomatopoeia':['onomatopoeia'],'non-sound manner word':['manner']}
    role_tags={'interrogative':['interr'],'proximal':['prox'],'distal':['dist'],'relative':['relative']}
    extra=category_tags.get(claims.get('source_category'),[])+role_tags.get(claims.get('source_role'),[])
    if claims.get('source_function')=='particle':extra+=['part']
    if claims.get('source_function')=='completive':extra+=['completive']
    if claims.get('source_function')=='obligation, compulsion':extra+=['obligation']
    if record['kind']=='derivation-affix' and claims.get('source_function')=='adverb formation by suffixation':
        extra+=['adverbializer','suffix']
    record['proposed_grammar']['tags']=list(dict.fromkeys(record['proposed_grammar']['tags']+extra))
    return record


def split_statements(rows):
    """Use reviewed printed statement boundaries without inventing source IDs."""
    output=[]; parents={}
    for record in rows:
        groups=record.get('source_claims',{}).get('source_expression_groups',[])
        if not groups:
            output.append(record)
            continue
        base=record['source_unit_key']; locations={}
        for i,group in enumerate(groups,1):
            row=copy.deepcopy(record)
            key=base if i==1 else base+f':statement:{i}'
            row.update(source_unit_key=key,physical_source_unit_key=base,
                       raw_group_form=record['form'],form=group['form'],
                       source_gloss=group['gloss'],editorial_statement=i)
            row['printed_page']=group['source_surface_lines'][0]['printed_page']
            row['pdf_page']=row['printed_page']+22
            row['proposed_grammar']={'gloss':group['gloss'],'tags':[], 'labels':[]}
            row['open_review']=[x for x in row['open_review'] if x!='multiple translated statements require explicit output grouping']
            output.append(row)
            for line in group['source_surface_lines']:
                location=(line['printed_page'],line['line'])
                assert location not in locations
                locations[location]=key
        parents[base]=locations
    for row in output:
        parent=row.get('parent_expression_key')
        if parent in parents:
            match=re.search(r':word:p(\d+):l(\d+):w\d+$',row['source_unit_key'])
            assert match, row['source_unit_key']
            location=tuple(map(int,match.groups()))
            row['raw_parent_expression_key']=parent
            row['parent_expression_key']=parents[parent][location]
        if row['kind']=='whole-expression':
            match=re.match(r'^([AQ]):\s*',row['proposed_grammar']['gloss'])
            if match:
                row['source_claims']['speaker_turn_label']=match[1]
                row['proposed_grammar']['gloss']=row['proposed_grammar']['gloss'][match.end():]
    return output


def expand(record, decision):
    """One physical unit can yield several explicit forms; preserve its raw group."""
    if decision is None:
        return [record]
    forms=decision['expanded_forms']
    if forms is None:
        row=copy.deepcopy(record)
        row['source_notation_decision']=decision
        row['phonemic_policy']='blank; literal grouped source notation retained'
        row['proposed_grammar']['tags']=list(dict.fromkeys(row['proposed_grammar']['tags']+['uncertain']))
        row['open_review']=[x for x in row['open_review'] if x!='explicit source alternatives require expansion or attestation reuse']
        return [row]
    output=[]
    for i,form in enumerate(forms,1):
        row=copy.deepcopy(record)
        row['physical_source_unit_key']=record['source_unit_key']
        row['source_unit_key']=record['source_unit_key'] if i==1 else record['source_unit_key']+f':alternate:{i}'
        row['raw_group_form']=record['form']
        row['form']=form
        row['source_notation_decision']=decision
        row['open_review']=[x for x in row['open_review'] if x!='explicit source alternatives require expansion or attestation reuse']
        if decision.get('expanded_glosses'):
            row['source_gloss']=decision['expanded_glosses'][i-1]
            row['proposed_grammar']=parse(row['source_gloss'])
        # Co-listing is not evidence for a lexical variant relation.
        assert decision['variant_edge'] is False
        output.append(row)
    return output


def build():
    decisions=json.loads((P/'whole-notation-expansion-decisions.json').read_text())
    by_key={r['source_unit_key']:r for r in decisions['decisions']}
    rows=[];seen=set()
    for line in (P/'whole-metadata-review.jsonl').open():
        record=json.loads(line);key=record['source_unit_key']
        if key in by_key:seen.add(key)
        rows.extend(expand(record,by_key.get(key)))
    assert seen==by_key.keys()
    assert len({r['source_unit_key'] for r in rows})==len(rows)
    rows=[retain_agreement_roles(row) for row in split_statements(rows)]
    decisions=json.loads((P/'whole-conflicting-tag-decisions.json').read_text())['decisions']
    by_key={row['source_unit_key']:row for row in rows}
    for decision in decisions:
        row=by_key[decision['source_unit_key']]
        assert row['form']==decision['form'] and row['source_gloss']==decision['source_gloss']
        tags=row['proposed_grammar']['tags']
        assert set(decision['withhold_confident_tags'])<=set(tags)
        row['proposed_grammar']['tags']=[t for t in tags if t not in decision['withhold_confident_tags']]+['uncertain']
        row['conflicting_tag_decision']=decision
    editorial=json.loads((P/'whole-note-editorial-decisions.json').read_text())['decisions']
    applied=set()
    for decision in editorial:
        for key in decision['keys']:
            row=by_key[key]
            assert row['source_claims'].get('note')==decision['note'],key
            row['editorial_source_note']=decision['output_note']
            row['proposed_grammar']['tags']=list(dict.fromkeys(row['proposed_grammar']['tags']+decision['add_tags']))
            applied.add(key)
    assert {r['source_unit_key'] for r in rows if r['source_claims'].get('note') and not r['disposition'].startswith('audit-only')}<=applied
    for decision in json.loads((P/'whole-heading-scope-review-draft.json').read_text())['decisions']:
        row=by_key[decision['source_unit_key']]
        assert row['form']==decision['form']
        tags=[tag for tag in row['proposed_grammar']['tags'] if tag not in decision['remove_tags']]
        row['proposed_grammar']['tags']=list(dict.fromkeys(tags+decision['add_tags']))
        row['source_heading_decision']=decision
    for decision in json.loads((P/'whole-root-boundary-review-20260927.json').read_text())['corrections']:
        row=by_key[decision['source_unit_key']]
        assert row['form']==decision['before_form']
        assert row['kind']=='paradigm-root'
        row['form']=decision['after_form']
        row['root_boundary_decision']=decision
    for decision in json.loads((P/'whole-interlinear-scope-decisions-20260927.json').read_text())['decisions']:
        row=by_key[decision['source_unit_key']]
        assert row['form']==decision['form'] and row['source_gloss']==decision['source_gloss']
        action=decision['action']
        if action=='explicit-interlinear':
            previous=row['proposed_grammar']
            parsed=copy.deepcopy(decision['grammar'])
            parsed['tags']=list(dict.fromkeys(previous['tags']+parsed['tags']))
            if set(parsed['tags']) & {'part','suffix','prefix','affix','infix','clitic'}:
                parsed['tags']=[tag for tag in parsed['tags'] if tag!='verb']
            if 'existing_tags' in previous: parsed['existing_tags']=previous['existing_tags']
            row['proposed_grammar']=parsed
        elif action in {'scoped-translation-label','scoped-phrase-label','text-provenance'}:
            row['proposed_grammar']['gloss']=decision['gloss']
            row['proposed_grammar']['tags']=list(dict.fromkeys(row['proposed_grammar']['tags']+decision['add_tags']))
        elif action=='resolved-alignment':
            assert 'held_analysis' not in row['proposed_grammar'], row['source_unit_key']
            assert row['proposed_grammar'].get('alignment_adjudication'), row['source_unit_key']
        row['interlinear_scope_decision']=decision
    return apply_attribution(reconcile_translation_groups(rows))

if __name__=='__main__':
    rows=build();f=P/'whole-expanded-metadata-review.jsonl'
    f.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    report={'status':'Expanded review metadata, not installed or frozen lexical output',
            'records':len(rows),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),
            'retained_literal_groups':sum(r.get('phonemic_policy','').startswith('blank; literal') for r in rows),
            'automatic_variant_edges':0}
    (P/'whole-expanded-metadata-summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
