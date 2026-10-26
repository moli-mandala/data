import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text())
done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[
dict(parent='13551',gloss=['needle'],forms=['suji','sūjī','sūji','sujī','suzi','sījī','sūja','suza'],citation='CDIAL[13551,1];platts1884[s.v. sūjī, needle]',evidence='Platts explicitly distinguishes sūjī “needle, small awl”, from sūcikā, from the semolina and tailor homonyms. CDIAL 13551,1 includes sūcikā in its needle head as well as the contracted sūī family. The survey j/z forms therefore belong to this needle family, with the voiced medial consonant retained in the analysis. Regional learned or vernacular transfer remains unresolved; the link does not treat j as an arbitrary spelling of zero.'),
dict(parent='4715',gloss=['walk!','walk!; he walked','to walk','walk'],languages=['H','Kannauji-Central','bundeli_atarra','Buksa','Rana','Sunha','Dang','MagahiNepal','Or'],forms=['cal','calo','calaū','calnāi','calab','caliba'],citation='CDIAL[4715,1]',evidence='Full calati gives Hindi calnā, Mth calab, Bengali calā and Oriya caḷibā with the short-vowel cal stem. These central/eastern simple responses retain that stem with imperative or infinitive endings. Longer auxiliaries and the western cāl/call family, which Turner separately places under *calyati 4721, are not folded into this decision. The link identifies the family while retaining possible regional transfer, explicitly relevant to the eastern Hindi-type forms.'),
dict(parent='10702',gloss=['night'],forms=['rāṭ','rāṭi','raṭ','ṛati','ṛāt','ərāt'],citation='CDIAL[10702]',evidence='Full rātrī gives Kalasha rat, Oriya rāti and the feminine rāt/rāti family throughout central and western Indo-Aryan. These simple night responses preserve that family, including the source’s retroflex t/r and initial reduced vowel. The separate rātra 10700 is the masculine Kashmiri branch, not the comparator used here. Responses containing additional words are excluded.'),
dict(parent='10158',gloss=['mouth'],forms=['muk'],languages=['B','Hajong','Dotyali'],citation='CDIAL[10158];CDIAL[10174]',evidence='CDIAL 10158 supplies mukha “mouth/face”; in the full 10174 discussion Turner explicitly prefers mukha with expressive doubling to mukhya for retained-k(h) mouth/face forms. Following that primary preference and the preceding reviewed policy, these bare muk responses are attached to the mouth noun, not the adjective “chief”. This is a family-level assignment; expressive history versus learned/contact transmission in the individual survey languages remains open.'),
]
a=[]
for i,g in enumerate(groups):
 for r in rs:
  if r['ID'] not in done and r['Gloss'] in g['gloss'] and r['Form'] in g['forms'] and ('languages' not in g or r['Language_ID'] in g['languages']):
   a.append(dict(record=r,parent=g['parent'],kind='reflex',citation=g['citation'],evidence=g['evidence'],family=i))
day={}
langs={'Bhilali','Nimadi','Rana','Sunha','Dang','Kathoriya','bagheli_lakshman'}
for r in csv.DictReader(open('cldf/forms.csv')):
 if r['Language_ID'] in langs and r['Gloss']=='day' and r['Form'] in ['din','dīn','ḍin','ḍīn']:
  day.setdefault(r['Language_ID'],[]).append({k:r[k] for k in ['ID','Language_ID','Form','Gloss','Source']})
for r in rs:
 if r['ID'] not in done and r['Language_ID'] in day and r['Gloss'] in ['sun','sun; day'] and r['Form'] in ['din','dīn','ḍin','ḍīn']:
  candidates=day[r['Language_ID']]
  comparator=next((x for x in candidates if x['Form']==r['Form']),candidates[0])
  a.append(dict(record=r,parent='6328',kind='reflex',citation='CDIAL[6328];'+comparator['Source'],evidence='CDIAL dina supplies regional din “day, daytime”. The same surveyed language independently attests '+comparator['Form']+' “day” ('+comparator['ID']+'), supporting the local day/sun polysemy rather than choosing a remote sun root by sound alone. The source’s sun gloss, vowel length and dental/retroflex spelling are retained. This identifies the day-word family and leaves possible intra-Indo-Aryan transmission open.',family=4,comparator=comparator))
held=[]
for r in rs:
 if r['Gloss']=='belly' and r['Form'] in ['ṭɛḍ','ṭeḍ','ḍɛḍ / ḍʰɛḍ']:
  held.append(dict(record=r,candidates=['5589','5589-4'],reason='The expressive belly family is clear, but CDIAL separates i-vowel *ḍhiḍḍha from e-vowel *ḍhēḍḍha. Existing search links to 5589 do not by themselves establish whether the survey e/ɛ is a local vowel development or the fourth branch; retain the subsection ambiguity.'))
assert len(a)==len({x['record']['ID'] for x in a})
(P/'pass009-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
(P/'pass009-day-comparanda.json').write_text(json.dumps(day,ensure_ascii=False,indent=1)+'\n')
(P/'pass009-decisions.json').write_text(json.dumps({'accepted':a,'held':held},ensure_ascii=False,indent=1)+'\n')
print(len(a),[(i,sum(x['family']==i for x in a)) for i in range(5)],'held',len(held))
