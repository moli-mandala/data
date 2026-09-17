import json,csv
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass86';assert not (P/(stem+'-decisions.json')).exists()
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())}
rules=[
 dict(parent='3674',citation='CDIAL[3674]',evidence='CDIAL 3674 kṣāra explicitly gives Gujarati khār/khārī salt and Konkani khāru salt, beside khāro saline/brackish forms. The western kharo/khāro/khāru/khara/kharõ survey nouns fit this salt/alkali family, with endings and nasalization retained; local Indo-Aryan contact remains open.'),
 dict(parent='14039',citation='CDIAL[14039]',evidence='CDIAL 14039 hastin explicitly gives Assamese, Bengali and Oriya hāti elephant, beside Nepali hāti. The eastern survey hati forms match these unaspirated regional descendants, with vowel length retained as transcribed and local Indo-Aryan transmission open.'),
 dict(parent='14028',citation='CDIAL[14028]',evidence='CDIAL 14028 *hastakūṭa hand hammer gives Bengali hātuṛi, Oriya hātuṛā, Bhojpuri hathaur and Marathi hatoḍā, plus Gujarati athɔṛɔ/athɔṛī with initial h loss. The selected simple hatur/haturi/hatuḍi/hatoḍu/athuḍi/athoḍi/athuḍu forms fit this whole hammer family. Final vowels and stop/flap or plain-r transcription remain qualified, without inferring a resolved local borrowing route.'),
 dict(parent='959',citation='CDIAL[959]',evidence='CDIAL 959 aṣṭhīvat explicitly gives Assamese/Bengali ā̃ṭhu and Bengali hā̃ṭhu knee. Hajong atʰu and Bishnupriya athu select that regional knee family, with the survey dental t and absent nasalization retained as transcription/phonological qualifications. This follows the same eastern knee comparison used for the reviewed hatu forms and does not impose a retroflex or nasal spelling on the source.')]
sets=[('salt',{'khāru','kharo','khāro','khara','kharõ'}),('elephant',{'hati'}),('hammer',{'hatur','hatuḍi','haturi','hatuɾi','hatoḍu','athuḍi','athoḍi','athuḍu'}),('knee',{'atʰu','athu'})]
acc=[];held=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 if r['Gloss']=='knee' and r['Form']=='atʰur':held.append(dict(record=r,families=[],reason='Hajong athur knee adds final r to the athu stem. CDIAL 959 has a differently arranged Lahnda aṭhrūā posture word, not evidence for this local suffix; establish the whole Hajong noun or case form.',passNumber=86))
 for i,(g,fs) in enumerate(sets):
  if r['Gloss']==g and r['Form'] in fs:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/'pass86_save.py').write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem))
print({'accepted':len(acc),'held':len(held)})
