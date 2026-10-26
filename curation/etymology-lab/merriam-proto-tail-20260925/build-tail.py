"""Read-only source coverage and holds for six DEDR-0 proto-Dravidian tails."""
import collections,csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;D=P.parents[2]
languages={'PSTDr','PSD1','PSD2'}
forms=[r for r in csv.DictReader((D/'cldf/forms.csv').open())
       if r['Language_ID'] in languages and 'merriam2026dravidiandb' in r['Source']]
assert len(forms)==3812
ids={r['ID'] for r in forms};linked=set()
for edge in csv.DictReader((D/'cldf/edges.csv').open()):
    if edge['Child_ID'] in ids:linked.add(edge['Child_ID'])
assert len(linked)==3806
residual=[r for r in forms if r['ID'] not in linked]
assert len(residual)==6
assert {r['ID'] for r in residual}=={r['ID'] for r in json.loads((P/'survey-inventory.json').read_text())}
search=[]
if (P/'discovery-001-006-results.json').exists():
    search=json.loads((P/'discovery-001-006-results.json').read_text())
    assert len(search)==6
    assert all(result['opts']['languageId']==record['Language_ID'] and result['opts']['params']['form']==record['Form']
               for record,result in zip(residual,search))
notes={
    '5904':'Krishnamurti supplies a first-person singular oblique proto-stem *nā, but this Merriam record explicitly has DEDR 0. Assigning a DEDR pronoun article requires a separately justified paradigm-level family comparison.',
    '5907':'Krishnamurti supplies a second-person singular oblique proto-stem *nī, but this Merriam record explicitly has DEDR 0. A surface second-person article is not automatically a parent for this reconstruction.',
    '5908':'Krishnamurti reconstructs *nīr “you (2pl)” at the proto-stage, with DEDR 0. The plural and pronominal paradigm require direct article comparison before a link.',
    '5935':'The first-person plural oblique proto-stem *mā has DEDR 0 in Merriam; possible first-person plural articles have not been shown to include this exact paradigm reconstruction.',
    '5936':'The second-person plural oblique proto-stem *mī has DEDR 0 in Merriam; possible second-person plural articles have not been shown to include this exact paradigm reconstruction.',
    '5944':'The proto-stem *vay- “throw” has DEDR 0 in Merriam. No direct form-and-sense match to a complete DEDR article was verified; do not infer a family from the broad English gloss alone.',
}
held=[]
for index,r in enumerate(residual):
    number=r['Source'].split('record ',1)[1].split(',',1)[0]
    assert number in notes and 'DEDR 0' in r['Source']
    item={'record':r,'reason':notes[number],'status':'research-hold'}
    if search:item['searchResultCount']=search[index]['count']
    held.append(item)
assert len(held)==6
def write(name,value):(P/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
write('proposed-decisions.json',{'status':'pending-review','accepted':[],'held':held})
write('research-holds.json',held)
write('unexamined-inventory.json',[])
write('primary-evidence.json',{})
write('review-manifest-paths.json',[])
write('source-coverage-001.json',{'source':'merriam2026dravidiandb','languages':sorted(languages),
    'totalCompiledRecords':3812,'existingGraphLinkedRecords':3806,'currentHeldRecords':6,
    'newProposedRecords':0,'unexaminedRecords':0,'existingLinkedRecordIds':sorted(linked),
    'currentHoldIds':sorted(r['ID'] for r in residual),'exactPartition':True,
    'exactFormSearchesCompleted':len(search)})
print(json.dumps({'source':3812,'linked':3806,'holds':6,'proposals':0}))
