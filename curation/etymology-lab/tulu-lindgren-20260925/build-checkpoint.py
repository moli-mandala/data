"""Bounded Tulu Lindgren primary-article checkpoint; pending review only."""
import csv,gzip,json,re
from pathlib import Path
P=Path(__file__).resolve().parent; R=P.parents[2]
records=json.loads((P/'survey-inventory.json').read_text()); assert len(records)==1730
target=json.loads((P/'target-candidates.json').read_text());assert len(target)==327
results=[]
for f in sorted(P.glob('discovery-*-results.json'),key=lambda x:int(x.name.split('-')[1])):
    results+=json.loads(f.read_text())
assert len(results)==1730
for rec,res in zip(records,results):assert rec['Form']==res['opts']['params']['form']

rejected={
    19:'DEDR 5515 includes Tulu baccirè “betel leaf” with a question mark and explicitly offers DEDR 3821 as an alternative family; the competing analysis needs review.',
    176:'The DEDR 4004 Tulu verb describes ripening or softening; the survey noun parndɯ “fruit” is not securely attested there.',
    297:'DEDR 2338 gives jabbe as “old man,” whereas the source gloss is the general adjective “old”; the part-of-speech mismatch needs review.',
    298:'DEDR 2338 gives jabbi as “old woman,” whereas the source gloss is the general adjective “old”; the part-of-speech mismatch needs review.',
    363:'DEDR 4666 gives Tulu maṇṇů “earth, clay, soil”; this record is glossed sand and needs a distinct semantic check.',
    500:'DEDR 612 gives udā̆lů “white ant” (termite), whereas the source gives generic “ant”; the biological sense cannot be equated without confirmation.',
    555:'DEDR 5450 gives Tulu bījuni/bīpuni “blow as the wind”; the survey specifies blowing with the mouth.',
    609:'DEDR 4797 gives Tulu māḍāvuni in a narrower sense; the survey aḍɯ “do” lacks a verified Tulu sound or donor path.',
    1055:'DEDR 156 gives Tulu appa as a respectful suffix or a mode of calling a mother, not unambiguously “father.”',
    1299:'DEDR 5450 gives Tulu bījuni/bīpuni “blow as the wind”; the survey specifies blowing with the mouth.',
    1353:'DEDR 4797 gives Tulu māḍāvuni in a narrower sense; the survey aḍɯ “do” lacks a verified Tulu sound or donor path.',
}
mapping={x['index']:x['consensus'] for x in target if x['index'] not in rejected}
# Lindgren groups generic eat under one cognate set, but DEDR distinguishes
# Tulu tin- from uṇ-; the latter is explicitly “eat rice / take a meal.”
for ix in (131,622,1366):mapping[ix]='d600'

# A second, bounded exact-form screen recovered direct Tulu reflexes that the
# other Lindgren lects did not supply. Each selected article is checked below.
supplement=(34,41,43,44,46,47,53,57,59,61,69,75,79,88,94,95,115,123,
            128,133,139,141,158,165,174,185,186,194,195,217,222,223,235,
            240,253,254,288,293,296,299,309,310,314,320,335,338,
            364,374,375,379,383,393,398,406,409,413,414,428,435,454,
            456,464,466,499)
ranked=json.loads((P/'homepage-candidates.json').read_text())
for ix in supplement:
    assert ix not in mapping and ix not in rejected
    cand=next((c for c in ranked[ix-1]['candidates'] if c['formScore']>=.85 and c['glossScore']>0),None)
    assert cand,(ix,ranked[ix-1])
    mapping[ix]=cand['parent']

parent_ids=set(mapping.values());articles={}
with gzip.open(R/'data/dedr/entry-texts-audit.csv.gz','rt') as fh:
    for row in csv.DictReader(fh):
        if row['Entry_ID'] in parent_ids:
            articles[row['Entry_ID']]={k:row[k] for k in ('Page','Citation','Raw_Article')}
            articles[row['Entry_ID']]['URL']='https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page='+row['Page']
assert set(articles)==parent_ids
def tu_excerpt(pid):
    s=articles[pid]['Raw_Article'];m=re.search(r'\bTu\.\s',s);assert m,pid
    s=s[m.end():];end=re.search(r'\s(?:Te|Kol|Nk|Pa|Ga|Go|Konḍa|Kui|Kuwi|Br|Ma|Ir|Ko|To|Ka|Koḍ)\.\s',s)
    if end:s=s[:end.start()]
    return s.strip()
def short(s):
    return s if len(s)<=280 else s[:280].rsplit(' ',1)[0]+'…'

accepted=[];held=[];unexamined=[]
for ix,rec in enumerate(records,1):
    pid=mapping.get(ix)
    if pid:
        article=articles[pid]
        homepage=any(row.get('etymon_id')==pid for row in results[ix-1]['rows'])
        evidence=f'The full DEDR {pid[1:]} article gives Tulu: {short(tu_excerpt(pid))}. Lindgren records {rec["Form"]} “{rec["Gloss"]}” in the same lexical family.'
        if ix in (131,622,1366):evidence+=' This uṇ- meal verb belongs to DEDR 600, distinct from DEDR 3263 tin-.'
        if ix in (441,):evidence+=' DEDR specifies an animal tooth; the survey gloss is broader.'
        if ix in (254,):evidence+=' The article specifies a cadjan leaf; the survey gloss is broader.'
        if ix in (379,):evidence+=' The article specifies a kind of harmless snake; the survey gloss is broader.'
        if ix in (503,1247):evidence+=' The article supports “arm” within the source’s broader “arm or hand” gloss.'
        if homepage:evidence+=' The same article is independently returned by Jambu’s exact-form homepage search.'
        accepted.append({'record':rec,'parent':pid,'kind':'reflex','citation':rec['Source']+f';dedr[entry {pid[1:]}]','evidence':evidence,'status':'pending-review','surveyIndex':ix})
    elif ix in rejected:
        held.append({'record':rec,'reason':rejected[ix],'status':'research-hold','surveyIndex':ix})
    else:
        unexamined.append({'record':rec,'reason':'Outside this bounded primary-article checkpoint; exact-form homepage results are retained for subsequent research.','status':'unexamined','surveyIndex':ix})
assert len(accepted)+len(held)+len(unexamined)==1730
owners={x['record']['ID']:'proposed' for x in accepted}
owners.update({x['record']['ID']:'held' for x in held});owners.update({x['record']['ID']:'unexamined' for x in unexamined});assert len(owners)==1730
(P/'proposed-decisions.json').write_text(json.dumps({'status':'pending-review','accepted':accepted,'held':held},ensure_ascii=False,indent=2)+'\n')
(P/'research-holds.json').write_text(json.dumps(held,ensure_ascii=False,indent=2)+'\n')
(P/'unexamined-inventory.json').write_text(json.dumps(unexamined,ensure_ascii=False,indent=2)+'\n')
(P/'primary-evidence.json').write_text(json.dumps(articles,ensure_ascii=False,indent=2)+'\n')
(P/'source-coverage-001.json').write_text(json.dumps({'source':'lindgren2023dravidian','language':'Tulu','compiledSourceRecords':1730,'proposedRecords':len(accepted),'heldRecords':len(held),'unexaminedRecords':len(unexamined),'uniqueOwnership':True,'recordOwnership':owners},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'proposed':len(accepted),'held':len(held),'unexamined':len(unexamined),'articles':len(articles),'homepageConfirmed':sum(any(y.get('etymon_id')==x['parent'] for y in results[x['surveyIndex']-1]['rows']) for x in accepted)}))
