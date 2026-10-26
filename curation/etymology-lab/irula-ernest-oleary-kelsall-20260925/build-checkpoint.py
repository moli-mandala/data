"""Bounded primary-article review of Nilgiri Irula SIL source residuals."""
import csv,gzip,json
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parents[2]
records=json.loads((P/'survey-inventory.json').read_text());assert len(records)==822
ranked=json.loads((P/'homepage-candidates.json').read_text());assert len(ranked)==822
cross=json.loads((P/'cross-pass-candidates.json').read_text())
excluded={'d5496','d2673','d4096','d4085','d5161','d4016'}
mapping={x['index']:x['parent'] for x in cross if x['parent'] not in excluded}
# The full DEDR 4671 article itself records Ir. maṇi (-t-) “speak.”
for ix in (724,726,727,728):
    assert ix not in mapping
    mapping[ix]='d4671'
holds={
    118:'DEDR 4016 gives Irula paḷḷa “river”; the source alla lacks the initial p-/paḷḷ- syllable, so this exact response needs source or sound-path review.',
    195:'soppɨ “leaf” resembles Kannada/Tulu soppu foliage in DEDR 2673, but the immediate donor into this Irula village variety is unresolved.',
    290:'alu “milk” resembles Kannada hālu in DEDR 4096; the immediate Nilgiri Irula donor path and missing h- need review.',
    291:'avu “snake” resembles Kannada hāvu; the source spelling and immediate donor are not secure enough for a DEDR 4085 proposal.',
}
assert not set(mapping)&set(holds)
parent_ids=set(mapping.values());articles={}
with gzip.open(R/'data/dedr/entry-texts-audit.csv.gz','rt') as fh:
    for row in csv.DictReader(fh):
        if row['Entry_ID'] in parent_ids:
            articles[row['Entry_ID']]={k:row[k] for k in ('Page','Citation','Raw_Article')}
            articles[row['Entry_ID']]['URL']='https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page='+row['Page']
assert parent_ids==set(articles),sorted(parent_ids-set(articles))
comparisons={
    'd5073':'Malayalam mai and Kota may “body” within the full mey/mai body family',
    'd1448':'the explicitly listed Irula kādu “ear”',
    'd3633':'Kannada nālige and Tulu nālayi “tongue,” matching the Irula nāḷaŋga stem',
    'd1904':'Tamil kūrai and Malayalam kūra “roof, hut,” matching Irula kurɛ/kura “house” by the hut-to-house sense',
    'd4559':'Tamil pōtu and Kannada pottu “time, sun,” matching Irula poḍu “sun”',
    'd3045':'Tamil/Malayalam taṇṇīr and regional taṇṇi “water”',
    'd497':'Tamil ilai and neighbouring Irula īlɛ “leaf”',
    'd4616':'Tamil makan and Kannada maga “son, child,” matching Irula maŋga',
    'd474':'the article’s Irula raṇḍu and regional raṇṭu “two”',
    'd3655':'Tamil nāl and Kannada nālku “four,” matching Irula nakku',
    'd5151':'the article’s Dravidian interrogative base e-/yā-, including Irula dāru “who” and Tamil forms for what, when, and how many',
    'd2426':'Tamil cāvu and Tulu saipini “die,” matching Irula savu/sai',
    'd2017':'Tamil kēḷ (past kēṭṭ-) and Malayalam kēḷkka “hear,” matching Irula keḷ-/keṭṭ-',
    'd4091':'Tamil pār and Malayalam pārkka “see,” matching Irula par-',
    'd4671':'the directly listed Irula maṇi (-t-) “talk, speak”',
}
assert set(comparisons)==parent_ids
accepted=[];held=[];unexamined=[]
for ix,rec in enumerate(records,1):
    if ix in mapping:
        pid=mapping[ix]
        evidence=f'The full DEDR {pid[1:]} article records {comparisons[pid]}. The village response {rec["Form"]} “{rec["Gloss"]}” is proposed within that article family.'
        if ix in (1,):evidence+=' The article lacks a separate Irula subsection for this form, so the proposed comparison is cross-dialectal.'
        if ix in (41,42,43,44,45,49,50):evidence+=' The source gloss is broader than the article’s “hut” wording.'
        if ix in (724,726,727,728):evidence+=' Jambu’s exact-form homepage search also returns this DEDR article.'
        accepted.append({'record':rec,'parent':pid,'kind':'reflex','citation':rec['Source']+f';dedr[entry {pid[1:]}]','evidence':evidence,'status':'pending-review','surveyIndex':ix})
    elif ix in holds:
        held.append({'record':rec,'reason':holds[ix],'status':'research-hold','surveyIndex':ix})
    else:
        unexamined.append({'record':rec,'reason':'Outside this bounded primary-article checkpoint; exact-form homepage results are retained for subsequent research.','status':'unexamined','surveyIndex':ix})
owners={x['record']['ID']:'proposed' for x in accepted};owners.update({x['record']['ID']:'held' for x in held});owners.update({x['record']['ID']:'unexamined' for x in unexamined})
assert len(owners)==len(records)==len(accepted)+len(held)+len(unexamined)
(P/'proposed-decisions.json').write_text(json.dumps({'status':'pending-review','accepted':accepted,'held':held},ensure_ascii=False,indent=2)+'\n')
(P/'research-holds.json').write_text(json.dumps(held,ensure_ascii=False,indent=2)+'\n')
(P/'unexamined-inventory.json').write_text(json.dumps(unexamined,ensure_ascii=False,indent=2)+'\n')
(P/'primary-evidence.json').write_text(json.dumps(articles,ensure_ascii=False,indent=2)+'\n')
(P/'source-coverage-001.json').write_text(json.dumps({'source':'ernest-oleary-kelsall2018irula','language':'Irula','compiledSourceRecords':2054,'priorProposedRecords':1139,'priorHeldRecords':93,'residualRecords':822,'proposedRecords':len(accepted),'heldRecords':len(held),'unexaminedRecords':len(unexamined),'uniqueOwnership':True,'recordOwnership':owners},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'proposed':len(accepted),'held':len(held),'unexamined':len(unexamined),'articles':len(articles)}))
