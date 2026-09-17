import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass129';assert not (P/(stem+'-decisions.json')).exists()
rules=[
 dict(parent='13479',citation='CDIAL[13479]',evidence='CDIAL 13479 suptá asleep explicitly supplies both past forms (Lahnda suttā, Old Marwari sūto, Old Gujarati sūtaü) and new verbs Jaunsari sūtṇõ, Nepali sutnu, Bihari/Maithili sūtab, Bhojpuri sūtal and Hindi sūtnā. Thus the selected sut- responses can continue the old participle even when elicited as a verb. Regional vowel length, inflection and the dental/retroflex notation of Tharu suṭ/suʈʌt are retained, not corrected. Exact local transmission remains unresolved.'),
 dict(parent='13902',citation='CDIAL[13902]',evidence='CDIAL 13902 svápati explicitly treats the coalescence of svap- and sup- present stems and documents Prakrit su(v)aï/so(v)aï/sōi, Lahnda sễaṇ, Oriya soibā/suibā, Hindi sonā and Gujarati sūvũ. Selected simple so-/su-/huv- sleep responses belong to that present-stem family, separately from sut- continuations of suptá. Inflected responses and paired simple forms are preserved; western s/h, glide variation and regional l-extensions remain qualified rather than silently normalized. Local Indo-Aryan transmission is unresolved.'),
 dict(parent='4485',citation='CDIAL[4485]',evidence='CDIAL 4485 *ghummati explicitly includes Assamese ghumāiba to doze and Bengali ghumā, corrected in the addenda to ghumā̆na, as well as Assamese ghumaṭi sleep. This supplies the doze/sleep sense directly, despite the reconstructed headword meaning revolve. Bengali, Bishnupriya and Hajong simple ghum-/gum- sleep responses are linked here, retaining deaspiration and verbal endings; a particular inter-Indo-Aryan borrowing route is not asserted.')]
sets=[{'sūtɔ','sutɔ','sūtyɔ','sunʌnu','sutʌnu','suʈʌt','sut','sutʌi','sutʎa','sutilak','sutyo','sūtnāī','suṭ'},
 {'soo','su','soila','soi̩la','soi̯la','souk','soilani','soilañ','soyla','sui, suiyu','hulo','sulu','hulu','huḷo','sulo','sav, soyo','huvinu','huv','huveyo','huvio','suyo'},
 {'gʰuma','gʰumabo','gʰumani','gʰumau','gʰumuva','gum','ghumano','ghumani'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[];held=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or 'sleep' not in r['Gloss'].lower():continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
 else:
  if r['Form']=='se' and r['Language_ID'] in {'awan','poth'}:
   held.append(dict(record=r,families=[],reason='Short se sleep: CDIAL 13902 gives Lahnda sễaṇ and Awan saũ, while 12322a śayate gives Garhwali seṇu and 12605 śete gives Kumaoni seṇo/sīṇo. The survey monosyllable does not determine which present-stem history applies. This is a competing-etymon question, not merely uncertain IA transmission.',passNumber=129))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print(len(acc),len(held))
