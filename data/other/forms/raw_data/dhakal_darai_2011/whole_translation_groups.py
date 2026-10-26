"""Owner-reviewed literal/idiomatic senses and the two source IDs in example 16b."""
import copy
import re

SENSES = {
    'dhakal2011:p190:ch6:surface:l54': ['I came, I went and I ate oil-cake.', 'I was indecisive.'],
    'dhakal2011:p327:ch12:surface:l28': ['I came, I went and I ate oil-cake.', 'I was indecisive.'],
    'dhakal2011:p328:ch12:surface:l6': ['When a deer wants to run, the dog wants to urinate.', 'To be engaged in some trivial things when it is time for urgent work.'],
}
TWO = 'dhakal2011:p343:ch12:surface:l58'


def reconcile(rows):
    output=[];seen=set()
    for original in rows:
        key=original['source_unit_key']
        if key in SENSES:
            seen.add(key)
            for index,gloss in enumerate(SENSES[key]):
                row=copy.deepcopy(original)
                row['physical_source_unit_key']=key
                row['source_unit_key']=key if index==0 else key+':sense:idiomatic'
                row['raw_source_gloss']=original['source_gloss']
                row['source_gloss']=gloss
                row['proposed_grammar']['gloss']=gloss
                row['source_translation_type']='literal' if index==0 else 'idiomatic'
                if index:
                    row['proposed_grammar']['tags']=list(dict.fromkeys(row['proposed_grammar']['tags']+['figurative']))
                output.append(row)
        elif key==TWO:
            assert original['form']=='pʰeri u bʰãṭakə dzəri dʰərikun boslə əse sãdzʰ əilə'
            for index,(form,gloss,identifier) in enumerate([
                ('pʰeri u bʰãṭakə dzəri dʰərikun boslə','He sat there by holding the root of the eggplant.','SQ.BLD.132'),
                ('əse sãdzʰ əilə','Then the evening came.','SQ.BLD.133'),
            ]):
                row=copy.deepcopy(original)
                row.update(source_unit_key=key if not index else key+':statement:2',
                           physical_source_unit_key=key,raw_group_form=original['form'],
                           raw_source_gloss=original['source_gloss'],form=form,source_gloss=gloss)
                row['proposed_grammar']['gloss']=gloss
                row['raw_source_text_ids']=original['source_claims']['text_ids']
                row['source_claims']['text_ids']=[identifier]
                row['source_statement_scope']='Individually translated/cited utterance within the combined italic line of example 16b'
                output.append(row)
        elif original.get('parent_expression_key')==TWO:
            row=copy.deepcopy(original)
            line=int(re.search(r':word:l(\d+):',key)[1])
            assert line in {59,60,61,62,69,70,71}
            second=line>=69
            row['raw_parent_expression_key']=TWO
            row['parent_expression_key']=TWO+':statement:2' if second else TWO
            row['raw_source_text_ids']=row['source_claims'].get('text_ids',[])
            row['source_claims']['text_ids']=['SQ.BLD.133' if second else 'SQ.BLD.132']
            output.append(row)
        else:
            output.append(original)
    assert seen==set(SENSES)
    return output
