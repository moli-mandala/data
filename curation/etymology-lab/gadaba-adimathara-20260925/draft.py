"""Conservative whole-article leads for residual Adimathara Mudhili Gadaba."""
import csv,gzip,json,re
from pathlib import Path
P=Path(__file__).resolve().parent;DATA=P.parents[2]
records=json.loads((P/'survey-inventory.json').read_text());assert len(records)==903
search=[]
for start in range(0,903,35):
    end=min(start+35,903)
    search.extend(json.loads((P/f'discovery-{start+1:04d}-{end:04d}-results.json').read_text()))
assert len(search)==903
assert all(x['opts']['languageId']=='Gadaba' and x['opts']['params']['form']==r['Form'] for r,x in zip(records,search))
prior={x['index']:x for x in json.loads((P/'prior-family-suggestions.json').read_text())}
groups='''
1,2,3,4,5:3103;13,14,15:1159;16,17,18,19:1977;
20,21,22,23,24,25:5024;26,27:2664;57,58,59,60:1479;
72,73,74,75,76,77:4303;165,166,167,168,169:1514;
196,197,198:4711;206,207:5535;208,209,210:4345;
297,298,299,300,301,302,303,304,305,306,307,308:2674;
323,325:4096;326,327,328:2115;336,337,338:5087;
346,347,348:4085;443,444,445,446:3516;
755:1097;837:5516
'''
parents={}
for clause in groups.replace('\n','').split(';'):
    if not clause.strip():continue
    indices,number=clause.strip().split(':')
    for item in indices.split(','):
        i=int(item);assert i not in parents and prior[i]['priorParent']=='d'+number
        parents[i]='d'+number

direct={x['index']:x for x in json.loads((P/'direct-gadaba-article-candidates.json').read_text())}
direct_groups='''
28,29,30,31,32,34:3986;158,159,160,161,162,163,164:4108;
180,183:4291;201,203,204,205:775;241,244:4639;
576:1395;660,669,670,671,673,674:5151;
816,817,819,822:3151;823,824,826,829:5498;
830,831,833,835,836:2781;838,839,840,843,844:2735
'''
direct_extra=set()
for clause in direct_groups.replace('\n','').split(';'):
    if not clause.strip():continue
    indices,number=clause.strip().split(':')
    for item in indices.split(','):
        i=int(item);assert i not in parents
        parent='d'+number
        assert any(x['parent']==parent for x in direct[i]['hits'])
        parents[i]=parent;direct_extra.add(i)
articles={}
with gzip.open(DATA/'data/dedr/entry-texts-audit.csv.gz','rt') as fh:
    for article in csv.DictReader(fh):
        if article['Entry_ID'] in set(parents.values()):
            articles[article['Entry_ID']]={k:article[k] for k in ('Page','Citation','Raw_Article')}
            articles[article['Entry_ID']]['URL']='https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page='+article['Page']
assert set(articles)==set(parents.values())

def anchor(parent):
    text=articles[parent]['Raw_Article']
    return (re.findall(r'Ga\..{0,220}',text) or [text[:220]])[0].strip()
accepted=[]
for i,parent in sorted(parents.items()):
    record=records[i-1]
    if i in direct_extra:
        passage=next(x['segment'] for x in direct[i]['hits'] if x['parent']==parent)
        evidence=f'DEDR {parent[1:]} full article (p. {articles[parent]["Page"]}) gives {passage.strip()} The source {record["Form"]} “{record["Gloss"]}” fits this Gadaba whole-article family; the article passage was checked for sense as well as form. Dialect and inflection remain in the source record.'
    else:
        match=prior[i]
        evidence=f'DEDR {parent[1:]} full article (p. {articles[parent]["Page"]}): {anchor(parent)} The source {record["Form"]} “{record["Gloss"]}” is a same-item, same-sense dialectal form beside previously reviewed {match["priorForm"]}; the full article includes that Gadaba family. Its inflection and dialect label remain in the source record.'
    accepted.append({'record':record,'parent':parent,'kind':'reflex','citation':record['Source']+';dedr[entry '+parent[1:]+']','evidence':evidence,'comparison':'Homepage exact-form search, same-source item comparison, and full primary DEDR article checked.','status':'pending-review','surveyIndex':i})
special={
    220:'Maigebūlu “mango” contains a fruit stem but has additional unanalysed material; hold before single-family assignment.',
    222:'Maigebulu “mango” contains a fruit stem but has additional unanalysed material; hold before single-family assignment.',
    254:'Kakalkayu “eggplant” appears compound-like; the constituent sequence needs checking against DEDR 1422.',
    365:'Kokkavaje “spider” resembles the DEDR 2195 Gadaba reflex but its compound-like ending is unresolved.',
    368:'Kokavaḍe “spider” resembles the DEDR 2195 Gadaba reflex but its ending is unresolved.',
    727:'Venelvenne “different” resembles a repeated “other” stem, but its morphology needs verification.',
    729:'Venelvenel “different” resembles a repeated “other” stem, but its morphology needs verification.',
    731:'Ukkutpel “whole” resembles the DEDR 990 one/whole stem, but the ending is unanalysed.',
}
held=[]
for i,(record,result) in enumerate(zip(records,search),1):
    if i in parents:continue
    label=f'{record["Form"]} “{record["Gloss"]}”'
    if i in special:reason=special[i]
    elif ';' in record['Gloss'] or '?' in record['Gloss']:reason=f'{label} has merged or qualified source senses; no precise article family is selected.'
    elif ' ' in record['Form'] or len(record['Form'])>17:reason=f'{label} is multi-part or morphologically extended; ordered components remain unverified.'
    else:reason=f'{label} lacks a verified form-and-sense match in a full DEDR article at this checkpoint.'
    held.append({'record':record,'reason':reason,'status':'research-hold','surveyIndex':i,'searchResultCount':result['count']})
assert len(accepted)+len(held)==903
assert len({x['record']['ID'] for x in accepted+held})==903
def write(name,obj):(P/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
write('proposed-decisions.json',{'status':'pending-review','accepted':accepted,'held':held})
write('research-holds.json',held);write('unexamined-inventory.json',[]);write('primary-evidence.json',articles)
print(json.dumps({'proposed':len(accepted),'held':len(held),'articles':len(articles)}))
