"""Prepare grammatical analyses without installing lexical rows."""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def agreement(label):
    tags=[]
    for prefix,tag in [('III','third-person'),('II','second-person'),('I','first-person')]:
        if label.startswith(prefix+' '):
            tags.append(tag)
            break
    if 'singular' in label:tags.append('sg')
    if 'plural' in label:tags.append('pl')
    if 'feminine-neuter' in label:tags.append('fn')
    elif 'masculine-feminine' in label:tags.append('mf')
    elif 'masculine' in label:tags.append('m')
    elif 'feminine' in label:tags.append('f')
    elif 'neuter' in label:tags.append('n')
    return tags

def prepare():
    rows=[json.loads(x) for x in (ROOT/'lexical-candidates.jsonl').read_text().splitlines()]
    reviews={r['entry_key']:r for r in map(json.loads,(ROOT/'glyph-review.jsonl').read_text().splitlines())}
    output=[]
    for row in rows:
        section=row['section'];tags=[];notes=[];analyses=[]
        if section=='6':tags=['suffix']+agreement(row['source_grammar'])
        elif section=='7-root':tags=['verb','stem']
        elif section in ('7a','7b','7c'):
            mood={'7a':'non-past','7b':'pret','7c':'subjunctive'}[section]
            label=row['source_grammar'].split(' ',1)[1]
            tags=['verb',mood]+agreement(label)
            if section=='7b':notes.append('Source labels this paradigm past; section9 suggests a possible perfect origin without settling the analysis.')
        elif section in ('7d','7e'):
            tags=['verb','impv' if section=='7d' else 'prohibitive','second-person']+agreement(row['source_grammar'].split(' ',1)[1])
        elif section=='7f':tags=['verb','neg','non-past' if row['row_in_section_column']==1 else 'pret']
        elif section in ('7g','7h'):
            tags=['verb'];notes.append('Source category: '+('concessive' if section=='7g' else 'assertive')+'.')
        elif section=='7-nonfinite-a':tags=['verb','inf','non-past' if row['source_grammar']=='non-past' else 'pret']
        elif section=='7-nonfinite-b':
            tags=['verb','conjunctive-participle']
            if row['row_in_section_column']==2:tags.append('neg')
            notes.append('Source category: converb.')
        elif section=='7-nonfinite-c':
            tags=['verb']
            if row['row_in_section_column']==1:tags+=['conditional','non-past']
            else:tags+=['pret'];notes.append('Source groups this since-you-came form under conditional; temporal gloss preserved.')
        elif section=='10':tags=['verb','3sg','m','non-past' if row['column']==1 else 'pret']
        elif section=='11':tags=['verb','3sg','pret','n' if row['row_in_section_column']==2 else 'f']
        elif section=='13':tags=['suffix','loc']
        elif section=='14':tags=['suffix','pl'];notes.append('Source limits these plural suffixes to rational nouns, with lexical selection between the two suffixes.')
        elif row['source_gloss'].startswith('to '):tags=['verb']
        if section=='12':
            number=row['row_in_section_column']
            labels={1:('I',['pron','1sg']),2:('we',['pron','1pl']),3:('you',['pron','2sg','m']),
                    5:('he (remote)',['pron','3sg','m','dist']),6:('this man',['pron','3sg','m','prox']),
                    7:('that woman, thing',['pron','3sg','fn','dist']),8:('this woman, thing',['pron','3sg','fn','prox']),
                    9:('those persons',['pron','3pl','dist']),10:('these persons',['pron','3pl','prox'])}
            if number==4:
                analyses=[{'gloss':'you','tags':['pron','2sg','f']},{'gloss':'you','tags':['pron','2pl']}]
            elif number in labels:
                gloss,tags=labels[number];analyses=[{'gloss':gloss,'tags':tags}]
        if not analyses:analyses=[{'gloss':row['source_gloss'],'tags':tags}]
        issues=list(reviews.get(row['entry_key'],{}).get('issues',[]))
        if row['source_gloss']=='chilly':
            issues.append('gloss:printed chilly may denote chili pepper rather than cold; preserve source wording pending semantic review')
        for index,analysis in enumerate(analyses,1):
            analysis=dict(analysis, tags=list(analysis['tags']))
            if issues:analysis['tags'].append('uncertain')
            output.append({'entry_key':row['entry_key']+f':analysis:{index}','source_occurrence_key':row['entry_key'],
                           'form':reviews.get(row['entry_key'],{}).get('reviewed_form',row['candidate_form']),
                           **analysis,'notes':notes,'issues':issues,'source_grammar':row.get('source_grammar',''),
                           'glyph_reviewed':row['entry_key'] in reviews,'status':'analysis proposal; not installed'})
    return output

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    rows=prepare();args.output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    print(f'{len(rows)} proposed analyses; {sum(r["glyph_reviewed"] for r in rows)} glyph-reviewed; no installation')
