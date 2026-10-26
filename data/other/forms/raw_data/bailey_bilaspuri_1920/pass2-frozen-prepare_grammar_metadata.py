"""Propose per-form grammatical analysis for human review; no installation."""
import hashlib
import json
from pathlib import Path

P=Path(__file__).resolve().parent
NOUNS={'horse':'m','house':'m','elephant':'m','girl':'f','sister':'f'}
PRONOUNS={'first-person','second-person','third-person','this','who','relative-who','what','indefinite'}
VERBS={'fall','be','come','go','remain','sit','beat','eat','drink','give','take','speak','do','know','bring','take-away','lie','strike','intro-do'}
CASES={'Nom.':'nom','Acc.':'acc','Gen.':'gen','Dat.':'dat','Abl.':'abl','Voc.':'voc','Agent':'erg'}


def interpret(unit,candidate):
    row=unit['source_review']['reconciled']
    tags=[];notes=[];gloss=row.get('gloss_or_scope',row.get('literal_translation',''))
    if ':sentences:' in unit['source_unit_key']:
        uncertainty=[fix['typed_uncertainty'] for fix in unit['source_review']['independent_corrections'] if fix.get('typed_uncertainty')]
        if unit['source_review']['status']!='independent_second_reading':
            uncertainty.append('review: original-page independent reading pending; not installable yet')
        return dict(gloss=gloss,tags=['sentential','multiword-expression']+(['uncertain'] if uncertainty else []),notes=[],typed_uncertainty=uncertainty,review='independent complete-expression grammar review completed; final output audit pending')
    group=row['group'];item=row['item'];label=row['source_label']
    field=candidate['source_field'];position=candidate['printed_position']
    if group in NOUNS:
        tags += ['noun',NOUNS[group]]
    elif group in PRONOUNS:
        tags+=['pron']
        if group in {'who','what'}:tags+=['interr']
        if group=='relative-who':tags+=['relative']
        if group=='indefinite':tags+=['indef']
        if group in {'this','third-person'}:tags+=['demonstrative']
    elif group in {'cardinal','intro-numerals','ordinal'}:
        tags+=['num']+(['ord'] if group=='ordinal' else [])
    elif group=='pronominal-adjectives':
        tags+=['pron','adj']
    elif group in {'time-adverb','place','other-adverb'}:
        tags+=['adv']
        if group=='time-adverb':tags+=['temporal']
        if group=='place':tags+=['spatial']
    elif group=='preposition':tags+=['postp']
    elif group=='preposition-example':tags+=['multiword-expression']
    elif group in {'substantive','intro-substantive'}:tags+=['verb','copula']
    elif group in VERBS or group=='intro-future':tags+=['verb']
    elif group=='comparison':
        tags+=['postp'] if item=='1' else ['adj']
        if item in {'3','4'}:tags+=['degree']
    elif group=='adjective-examples':tags+=['multiword-expression']
    elif group=='intro-ability' and item=='2':
        tags+=['sentential','multiword-expression']
        notes.append('The source gives this complete Bilaspuri example in contrast with a separately attributed Panjabi construction.')
    elif group in {'intro-nouns','intro-stative','intro-ability','adjective-declension'}:
        tags+=['suffix'];notes.append(gloss);gloss=''
    elif group=='ability':
        if item=='1':
            tags+=['verb','inf','auxiliary'];notes.append(gloss);gloss=''
        elif item=='7':
            tags+=['suffix'];notes.append(gloss);gloss=''
        elif item=='8':tags+=['verb','inf','auxiliary']
        else:tags+=['sentential','multiword-expression']
    elif group=='little':tags+=['adj'] if item=='1' else ['sentential','multiword-expression']
    elif group=='sentence-note':
        tags+=['adv'] if item=='1' else ['verb']
    else:raise ValueError(('Unclassified group',group))
    # Case scope is restricted to nominal/pronominal paradigms.
    if group in set(NOUNS)|PRONOUNS:
        tags += [tag for printed,tag in CASES.items() if printed in label]
        if row['plural']:
            tags+=['pl' if field=='plural' else 'sg']
        if group in {'first-person','second-person','third-person'}:tags.append(group)
    if group=='intro-nouns':
        tags+=['f' if item=='3' else 'm']
        if item in {'2','3'}:tags+=['pl' if field=='plural' else 'sg']
        if item=='2' and field=='plural':tags+=['erg']
        if item=='1':notes.append('The source identifies a noun class by final -ā; this is not a separately specified case ending.')
    if group=='intro-stative':
        tags+=['participle']
        notes.append('Source identifies the ending as stative, added to the past-tense root.')
    if group=='adjective-declension':
        tags+=['m' if position==1 else 'f']
        notes.append('Bailey says adjectives in other endings do not decline unless used as nouns, when they decline as nouns.')
    if group=='comparison':notes.append('Bailey says there are no special comparison forms; comparison uses te, from.')
    if group in VERBS|{'intro-future','substantive','intro-substantive'}:
        if 'verb heading' in label or (group=='beat' and item=='1') or group=='take-away':tags+=['inf']
        if 'Imperat.' in label:tags+=['impv']
        if label.startswith('Fut.') or label.startswith('another fut.') or group=='intro-future':tags+=['fut']
        if 'Pres. cond.' in label:tags+=['pres','conditional']
        if 'Pres. ind. or past cond.' in label:
            notes.append('Source assigns the same form to present indicative or past conditional; these are alternative functions.')
        elif 'Pres. ind.' in label:tags+=['pres','ind']
        if 'Imperf.' in label:tags+=['ipfv']
        if label.startswith('Past'):tags+=['pret']
        if 'Stat. part.' in label or 'stat. part.' in label:tags+=['participle']
        if 'Conj. part.' in label:tags+=['conjunctive-participle']
        if 'Pres. perf.' in label:tags+=['pres','perfect']
        if 'Plup.' in label:tags+=['pret','perfect']
        if 'Habitual' in label:notes.append(label)
        if 'Pass.' in label:tags+=['pass']
        if row['plural']:
            tags+=['pl' if field=='plural' else 'sg']
        if 'masculine then feminine' in label or 'singular m then f' in label:
            if group=='go' and field=='plural':tags+=['f']
            else:tags+=['m' if position==1 else 'f']
        for printed,tag in [('1st person','first-person'),('2nd person','second-person'),('3rd person','third-person')]:
            if printed in label:tags.append(tag)
        if 'Stat. part.' in label or 'stat. part.' in label:notes.append('Source labels this a stative participle.')
        if group=='intro-substantive':tags+=['pres' if item=='1' else 'pret']
        if group=='substantive':tags+=['pret' if item=='4' else 'pres']
        if group=='fall' and item in {'3','4','5'}:
            tags+=[{'3':'first-person','4':'second-person','5':'third-person'}[item]]
            notes.append('Person is interpreted from the printed three-row paradigm order; the row itself is not separately numbered by person.')
        if group=='fall' and item in {'11','12'}:
            tags+=['sentential','first-person','sg','pres' if item=='11' else 'pret']
        if group=='strike' and item in {'2','3'}:tags+=['sentential','first-person','sg','pret']
        if group=='beat' and item=='3':
            tags+=['pass','participle']
            notes.append("The source prints 'fut. plur.' for the final form within its gender/number account of the passive expression. The feminine-plural interpretation is editorial; the anomalous printed label is preserved here.")
        if group=='beat' and item=='1':notes.append('Bailey states that this verb is regular and agrees with its object in past tenses.')
        if group=='beat' and item=='2':notes.append('Bailey conjugates jāṇā as the ordinary verb go, while the beaten component agrees in gender and number.')
    if group=='sentence-note':
        if item=='2':
            tags+=['participle']
            notes.append('The source uses this stative form for the past: is-having-been-done, i.e. has been done.')
        if item=='3':notes.append('Bailey cites this and sentences15,16,18 as compound verbs, comparing their prevalence in Panjabi, Hindi and Urdu; this is a source comparison, not a borrowing claim.')
    if group=='pronominal-adjectives':
        if item in {'3','7','11'}:tags+=['interr']
        if item in {'4','8','12'}:tags+=['relative']
        if item in {'2','10'}:notes.append('Source labels this correlative.')
    if group=='ability':
        if item=='7':notes.append('In the first method the verb agrees with the logical object in gender and number; in the second, săkṇā agrees with the logical subject. The root plus -ī remains unchanged in both.')
    if group=='little' and item=='2':notes.append('Bailey describes this negative use of little as tautological; the complete literal and idiomatic translations are retained.')
    if group=='strike' and item=='1':notes.append('The source says a word for blow is understood with this verb.')
    if group=='lie' and item=='2':notes.append('The source states that the stative participle of fall/lie is used for ill.')
    if group=='fall' and item=='13':
        notes.append(gloss);gloss=''
        if position==1:tags+=['indecl']
    # Source qualifications must survive even when no additional lexical gloss
    # or inflection can be inferred from them.
    if 'indecl.' in label or 'indeclinable' in label:
        tags+=['indecl']
        if group=='intro-future':notes.append('Bailey explicitly describes this alternative future as indeclinable in both Bilaspuri and Sasi.')
    if 'no plural' in label:notes.append('The source specifies that this interrogative has no plural.')
    if 'regular except past' in label:notes.append('The source describes this verb as regular except in the past tense.')
    if 'declined as above' in label:notes.append('The source refers back to the preceding paradigm for agreement.')
    if group=='take-away':notes.append('The source conjugates this expression like jāṇā, go, given above.')
    if group=='intro-numerals':notes.append('Bailey cites this form as an example of cerebral '+('n and l' if item=='2' else 'n')+'; this preserves his terminology, not an additional phonemic claim.')
    if group=='sentence-note' and item=='1':notes.append('The source places this approximately marker after a number or a word expressing amount or size.')
    if group=='substantive':
        # Source English singular/plural labels correspond to table columns.
        gloss=gloss.split(';')[1 if field=='plural' else 0].strip()
    if ' ' in candidate['form']:tags+=['multiword-expression']
    if candidate['form'].startswith('-') and 'suffix' not in tags:tags+=['suffix']
    uncertainties=[]
    for fix in unit['source_review']['independent_corrections']:
        if fix.get('typed_uncertainty'):uncertainties.append(fix['typed_uncertainty'])
    if uncertainties:tags+=['uncertain']
    if unit['source_review']['status']!='independent_second_reading':
        uncertainties.append('review: original-page independent reading pending; not installable yet')
    variant_of=''
    alternate_cell=(group in PRONOUNS or (group in {'be','come','go','eat','give','take'} and ('Fut.' in label)) or (group=='sentence-note' and item=='1'))
    if alternate_cell and position>1:variant_of=candidate['candidate_key'].rsplit(':',1)[0]+':1'
    return dict(gloss=gloss,tags=list(dict.fromkeys(tags)),notes=notes,typed_uncertainty=uncertainties,source_label=label,variant_of_key=variant_of,review='independent metadata review completed; final output audit pending')


def main():
    path=P/'whole-expansion-draft-20260927.jsonl'
    units=[json.loads(s)for s in path.read_text().splitlines()]
    outputs=[]
    for unit in units:
        for candidate in unit['candidates']:
            outputs.append(dict(source_unit_key=unit['source_unit_key'],**candidate,**interpret(unit,candidate)))
    (P/'whole-grammar-metadata-proposal-20260927.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n'for r in outputs))
    summary=dict(status='per-form metadata proposal; human review and canonical installation pending',forms=len(outputs),explicit_uncertainties=sum(bool(r.get('typed_uncertainty'))for r in outputs),input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (P/'whole-grammar-metadata-summary-20260927.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
