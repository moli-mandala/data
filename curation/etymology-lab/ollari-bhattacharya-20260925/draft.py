"""Research-only Ollari article decisions, preserving homograph and source holds."""
import csv,gzip,json,re
from pathlib import Path

P=Path(__file__).resolve().parent;D=P.parents[2]
records=json.loads((P/'survey-inventory.json').read_text());assert len(records)==645
candidates=json.loads((P/'primary-candidate-index.json').read_text());assert len(candidates)==645
search=[]
for start in range(0,645,35):
 end=min(start+35,645)
 search+=json.loads((P/f'discovery-{start+1:03}-{end:03}-results.json').read_text())
assert len(search)==645
assert all(q['count']==len(q['rows']) for q in search)

# The literal-match scan is only a finding aid. These exclusions were checked
# against each full DEDR article and the source gloss. In particular, an exact
# string in a different sense of a homograph is never enough for a proposal.
exclude={
 15,78,142,147,211,285,340,359,365,379,401,417,424,470,484,521,550,561,
 585,601,602,641,
}
overrides={
 42:'d474',43:'d410',72:'d725',130:'d1147',133:'d1298',140:'d1109',153:'d1422',
 154:'d1425',194:'d1957',203:'d2064',208:'d1731',358:'d4039',360:'d3884',
 370:'d3852',371:'d3853',380:'d4095',399:'d4275',400:'d4344',
 413:'d4299',433:'d4490',492:'d4722',493:'d4711',503:'d4616',
 504:'d4822',515:'d4915',516:'d4878',560:'d5422',567:'d5498',
 569:'d5516',578:'d2426',594:'d2510',595:'d1942',596:'d2594',
}
chosen={}
for item in candidates:
 index=item['index'];hits={x['parent'] for x in item['hits']}
 if index in exclude or not hits:continue
 if index in overrides:
  assert overrides[index] in hits,(index,hits)
  chosen[index]=overrides[index]
 elif len(hits)==1:chosen[index]=next(iter(hits))
 else:raise AssertionError(('unresolved homograph',index,hits))

ids=set(chosen.values());articles={}
with gzip.open(D/'data/dedr/entry-texts-audit.csv.gz','rt') as fh:
 for row in csv.DictReader(fh):
  if row['Entry_ID'] in ids:
   articles[row['Entry_ID']]={k:row[k] for k in ('Page','Citation','Raw_Article')}
   articles[row['Entry_ID']]['URL']='https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page='+row['Page']
assert set(articles)==ids

def excerpt(text,limit=210):
 s=' '.join(text.split())
 return s if len(s)<=limit else s[:limit].rsplit(' ',1)[0]+'…'

qualifications={
 11:'The article gives Ollari ayal “woman, wife”; it identifies the family, not the immediate route of every kin use.',
 42:'This inḍi is explicitly “two things” in DEDR 474; the homophonous “now” belongs to DEDR 410.',
 43:'This inḍi is explicitly “now” in DEDR 410; the homophonous “two” belongs to DEDR 474.',
 68:'DEDR 990 gives ukur “one man”; the source’s broader “alone; single” gloss extends the one stem.',
 78:'The article says “pound rice,” while the source says “husk rice”; the mechanical relationship needs review.',
 101:'The DEDR article identifies esel “when”; the source additionally glosses “then; ever,” which remain sense extensions.',
 132:'DEDR 1159 directly lists Ga. (Oll.) kaṇa “hole” beside kaṇ “eye”; both belong to the aperture family.',
 136:'The full article gives Ollari kat- “cut, sacrifice”; no separate sacrificial donor is proposed.',
 143:'The primary Ga. (Oll.) sense is “unripe mango”; the same source form also occurs as a month name and that record is held.',
 154:'DEDR 1425 is the crow article; the same Ollari spelling is printed for brinjal in DEDR 1422.',
 153:'DEDR 1422 is the brinjal article; the same Ollari spelling is printed for crow in DEDR 1425.',
 190:'The full DEDR article gives Ollari kusul “armpit”; the source also glosses “waist,” which is not established by this article.',
 208:'DEDR 1731 directly prints Ollari kope “hill, forest”; it is not the homophonous wild-dog compound in DEDR 3650.',
 287:'DEDR 3263 prints Ollari tin- “eat”; the source’s grazing use can be an extension of animal eating.',
 309:'DEDR 3563 gives Ollari toṛon “younger brother”; the source’s unqualified “brother” is narrowed to that sense.',
 345:'The article and source both connect the neliŋ lunar form to moon and month; no separate month etymon is needed.',
 358:'DEDR 4039 is the pig article; homophonous panḍ- “be able” belongs to DEDR 3884.',
 360:'DEDR 3884 gives Ollari panḍ- “be able”; panḍ “pig” is a different article, DEDR 4039.',
 363:'The full article gives Ollari pat- “take hold of, catch, buy”; all three source senses are included.',
 380:'DEDR 4095 directly gives Ollari pāp “young one, small one”; the calf compound in DEDR 3939 is narrower.',
 390:'DEDR 4205 explicitly gives Ollari piṭne “day after tomorrow”; this temporal sense is used.',
 399:'DEDR 4275 gives Ollari pun “new”; homophonous pun- “know” is DEDR 4344.',
 400:'DEDR 4344 gives Ollari pun- “know”; pun “new” is DEDR 4275.',
 416:'DEDR 4411 gives Ollari pernonḍ “big”; its separate “elder brother” source gloss is held.',
 433:'DEDR 4490 gives Ollari poṭ “snapping of fingers”; DEDR 4514 has a homophonous “back” form.',
 492:'DEDR 4722 gives Ollari mar- “learn”; mar “tree” belongs to DEDR 4711.',
 493:'DEDR 4711 gives Ollari mar “tree”; mar- “learn” belongs to DEDR 4722.',
 503:'DEDR 4616 gives Ollari māl “daughter”; māl “wine” belongs to DEDR 4822.',
 504:'DEDR 4822 gives Ollari māl “wine”; māl “daughter” belongs to DEDR 4616.',
 515:'DEDR 4915 gives Ollari muy- “cover oneself”; the bath verb is in DEDR 4878.',
 516:'DEDR 4878 gives Ollari (nīr) muy- “bathe”; the cover verb is in DEDR 4915.',
 542:'The article gives Ollari vaŋger “palate”; the source’s parenthetical tongue uncertainty is not endorsed.',
 567:'DEDR 5498 gives Ollari veṭ- “flee”; the homophonous past stem of “hear” is in DEDR 5516.',
 569:'DEDR 5516 gives Ollari ven- (veṭ-, vey-) “hear”; the homophonous veṭ- “flee” is DEDR 5498.',
 578:'DEDR 2426 gives Ollari say- “die”; the homophonous “stay” response is held.',
 594:'DEDR 2510 gives Ollari siṭ- “split, cut”; the homophonous extinguish verb is in DEDR 1942.',
 595:'DEDR 1942 gives Ollari siṭ- “fire goes out”; the homophonous split verb is in DEDR 2510.',
 596:'DEDR 2594 explicitly lists Ollari sinḍ “son”; its occurrence in a DEDR 4616 compound is not the source of this assignment.',
 609:'DEDR 2552 gives Ollari sirtal “evening” under the dark family; the primary article itself supplies the temporal sense.',
}
accepted=[]
for index,parent in sorted(chosen.items()):
 rec=records[index-1];item=candidates[index-1]
 hit=next(h for h in item['hits'] if h['parent']==parent)
 opening=excerpt(articles[parent]['Raw_Article'],145)
 local=excerpt(hit['snippet'],200)
 note=qualifications.get(index,'The source form and sense match this Ollari entry within the whole article.')
 evidence=f'DEDR {parent[1:]} full article opens “{opening}”; its Ga. (Oll.) passage gives “{local}”. {note}'
 accepted.append({'record':rec,'parent':parent,'kind':'reflex',
   'citation':rec['Source']+';dedr[entry '+parent[1:]+']',
   'evidence':evidence,
   'comparison':'The Ga. (Oll.) form in DEDR may derive from Bhattacharya itself; this is whole-article classification, not an independent second attestation.',
   'status':'pending-review','surveyIndex':index})

special_holds={
 15:'Both DEDR 319 and 367 cite the same Ollari arg- “bellow”; the duplicate article family needs adjudication.',
 78:'DEDR 583 says Ollari uyup- “pound rice”; the source gloss “husk rice” needs process/sense verification.',
 142:'Karke is a caitra month name here but “unripe mango” in DEDR 1281 and another source record; do not conflate them.',
 147:'DEDR 1165 and 1370 both list kaṛsid “firewood”; article ownership is unresolved.',
 211:'DEDR 2160 specifies Ollari kor “cock”; the source’s generic “fowl” may be sex broader.',
 285:'Tinḍ- “pull” is only a past form of “eat” in DEDR 3263; this separate source verb needs research.',
 340:'Nīr “water” appears incidentally in other Ollari article phrases, but no direct primary article for the noun was verified here.',
 359:'Panḍ- “become tired” is not established by the pig or ability homographs in DEDR 4039/3884.',
 365:'Pad “foot” only matched a head-pad compound in DEDR 2677; this is not direct family evidence.',
 379:'Pāṭe “beam” is not the homophonous “song” noun of DEDR 4065.',
 401:'Pun- “be broken” is distinct from DEDR 4275 “new” and DEDR 4344 “know.”',
 417:'Pernonḍ “elder brother” appears in DEDR 4411 only as “big”; the kin construction needs independent analysis.',
  424:'Ponḍ here names a calendar month; DEDR 4491 gives homophonous ponḍ “chaff.”',
  470:'DEDR 4411 gives Ollari pernonḍ/berit “big,” but its bare ber is printed for Parji in a “big river” compound; confirm the Ollari short form separately.',
 484:'Manḍ- “roll” matched unrelated occurrences in Ga. (Oll.) article phrases; no direct roll family was verified.',
 521:'Men “and; also; too” only matched incidental text in unrelated DEDR articles.',
 550:'Vande “finger” appears as part of “little finger” in DEDR 2594, but its own finger root needs verification.',
 561:'Vinḍ “highland for cultivation” is not the bow noun in DEDR 5422 or the white/light family of DEDR 5496.',
 585:'Say- “stay” is separate from the die/leave/say homographs in DEDR 2426/2458/4233.',
 601:'Sir “nerve” is not the buffalo noun in DEDR 2554.',
 602:'The source siriŋ “black” is close to Ollari siriŋg in DEDR 2552, but its article also lists Parji siriŋ “charcoal”; transcription/variety need checking.',
 641:'Soyup- “purchase” matched DEDR 2866 “send”; its buy family has not been established.',
}
held=[]
for index,(rec,result,item) in enumerate(zip(records,search,candidates),1):
 if index in chosen:continue
 if index in special_holds:reason=special_holds[index]
 elif ' ' in rec['Form'] or '-' in rec['Form'][:-1]:reason='The source response is multi-part; its ordered components have not been securely assigned.'
 elif item['hits']:reason='Literal form matches occur in DEDR, but no single article has been selected for this source sense or family.'
 else:reason='No direct Ga. (Oll.) form-and-sense match in a full DEDR article was verified in this pass.'
 held.append({'record':rec,'reason':reason,'status':'research-hold','surveyIndex':index,
              'searchResultCount':result['count'],'candidateArticleIds':[h['parent'] for h in item['hits']]})
assert len(accepted)+len(held)==645
assert len({x['record']['ID'] for x in accepted+held})==645

def write(name,obj):(P/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
write('proposed-decisions.json',{'status':'pending-review','accepted':accepted,'held':held})
write('research-holds.json',held);write('unexamined-inventory.json',[])
write('primary-evidence.json',articles)
print(json.dumps({'proposed':len(accepted),'rows':len(accepted),'held':len(held),'articles':len(articles)}))
