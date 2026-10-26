import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[{'parent': '9661', 'gloss': ['older brother', 'younger brother'], 'forms': ['bʰaī yā', 'bʰai yyā', 'bʰai yā', 'pāī', 'pāū', 'bʰāyi', 'bʰayyā', 'bhaya', 'bʰæyā', 'bʰyyā', 'bʰæyyā', 'bʰirā', 'pʰirā', 'ẓo', 'źo', 'bʰyā', 'bʰāiāk', 'bhāyā', 'bʰevā'], 'citation': 'CDIAL[9661 and addenda]', 'evidence': 'Turner gives the full bhrātṛ brother family, including Lahnda bhirā, Punjabi bhā/bhāī, Pahari bhai/bhau, eastern bhāi and Kohistani žā/ẓā. These simple survey responses match the regional brother stems; bhaiyā and related affectionate/inflected endings are kept with the lexical base. The older/younger distinction belongs to the elicitation context, and responses containing only small/young adjectives or mixed alternatives are excluded.'}, {'parent': '9349', 'gloss': ['older sister', 'younger sister'], 'forms': ['bahanīyā', 'vahanīyā', 'pihin', 'pɛin', 'pin', 'bīhū̃', 'bihū̃', 'bʰīhāū', 'bāinik', 'boini', 'bʰouni', 'bohoni', 'bɔūnī', 'bʰāiṇyā', 'bʰahinya', 'bʰāⁱniya', 'bʰahini', 'bāhĩniyā'], 'citation': 'CDIAL[9349 and addenda]', 'evidence': 'Turner gives bhaginī sister through both peripheral bhaïṇī and central bahiṇī, with Punjabi bhaiṇ, Nepali baini, Oriya bhaüṇī and Kohistani bhẽ/bhī̃/bihā̃. The selected complete survey sister forms preserve these regional developments and ordinary affectionate endings. Bare bāī/bhayā forms are excluded where woman/sister or brother homonymy needs separate evidence.'}, {'parent': '6327', 'gloss': ['older sister'], 'forms': ['ḍiḍi', 'ḍīḍḍa', 'ḍīḍḍe', 'di'], 'citation': 'CDIAL[6327]', 'evidence': 'Turner explicitly groups western Pahari dēddī/daiddī, Kumaoni didī/dī and Nepali/Bengali/Oriya didi elder sister under *diddā. These survey didi/di variants match the nursery kinship family, with retroflex spellings retained as a qualification; this does not claim a uniquely inherited rather than diffusional history for the nursery term.'}]
a=[]
for i,g in enumerate(groups):
 for r in rs:
  if r['ID'] not in done and r['Gloss'] in g['gloss'] and r['Form'] in g['forms'] and ('languages' not in g or r['Language_ID'] in g['languages']):
   x=dict(record=r,citation=g['citation'],evidence=g['evidence'],family=i)
   if 'components' in g:x['components']=g['components']
   else:x.update(parent=g['parent'],kind=g.get('kind','reflex'))
   a.append(x)
assert len(a)==len({x['record']['ID'] for x in a})
h=[]
for r in rs:
 if r['ID'] not in done and r['Gloss']=='older brother' and r['Form'] in ['nikā','niko','nando','nanḍo','cotā','lakʰ','lak','lɛkū','ṇaṇḍa / niko','bʰāī / nīko','pirā / niko']:
  h.append(dict(record=r,reason='The older-brother row contains a small/younger adjective alone or joins older and younger answers. This appears to be SSNP prompt/column alignment, not a lexical elder-brother etymon. Source-page reconciliation is deferred; no spelling, gloss or identity repair in this overlay-only task.'))
(P/'pass052-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=1)+'\n')
(P/'pass052-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
print(len(a),'accepted',len(h),'held',[(g.get('parent',g.get('components')),sum(x['family']==i for x in a)) for i,g in enumerate(groups)])
