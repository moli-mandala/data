"""Build article-backed pending Toda proposals, specific holds, and explicit unexamined tail."""
import csv
import gzip
import json
from pathlib import Path

P=Path(__file__).resolve().parent
records=json.loads((P/'survey-inventory.json').read_text())
selection=json.loads((P/'selection.json').read_text())
selected={x['surveyIndex']:x for x in selection['selected']}
held={x['surveyIndex']:x for x in selection['held']}
assert not set(selected)&set(held)
assert len(records)==5605
accepted=[]
research_holds=[]
unexamined=[]
for index,record in enumerate(records,1):
    if index in selected:
        x=selected[index]
        assert x['record']['ID']==record['ID']
        number=x['parent'][1:]
        comparison=f"Toda {x['dedrForm']} ‘{x['dedrGloss']}’"
        accepted.append({'record':record,'parent':x['parent'],'kind':'reflex',
            'citation':record['Source']+f';dedr[entry {number}]',
            'evidence':f"The full DEDR {number} article includes {comparison}, corresponding to source {record['Form']} ‘{record['Gloss']}’. The proposed node is the whole article.",
            'status':'pending-review','surveyIndex':index})
    elif index in held:
        x=held[index]
        assert x['record']['ID']==record['ID']
        research_holds.append({'record':record,'reason':x['reason'],
            'candidateParent':x['candidateParent'],'status':'research-hold','surveyIndex':index})
    else:
        unexamined.append(record)
assert len(accepted)==153 and len(research_holds)==19 and len(unexamined)==5433
parents={x['parent'] for x in accepted}
articles={}
with gzip.open(P.parents[2]/'data/dedr/entry-texts-audit.csv.gz','rt') as fh:
    for row in csv.DictReader(fh):
        if row['Entry_ID'] in parents:
            assert row['Entry_ID'] not in articles
            assert 'To.' in row['Raw_Article']
            articles[row['Entry_ID']]={k:row[k] for k in ('Page','Citation','Raw_Article')}
            articles[row['Entry_ID']]['URL']='https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page='+row['Page']
assert set(articles)==parents
(P/'proposed-decisions.json').write_text(json.dumps({'status':'pending-review','accepted':accepted,'held':research_holds},ensure_ascii=False,indent=2)+'\n')
(P/'research-holds.json').write_text(json.dumps(research_holds,ensure_ascii=False,indent=2)+'\n')
(P/'unexamined-inventory.json').write_text(json.dumps(unexamined,ensure_ascii=False,indent=2)+'\n')
(P/'primary-evidence.json').write_text(json.dumps(articles,ensure_ascii=False,indent=2)+'\n')
print({'proposed':len(accepted),'holds':len(research_holds),'unexamined':len(unexamined),'articles':len(articles)})
