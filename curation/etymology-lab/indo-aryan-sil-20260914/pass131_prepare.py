import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass131';assert not (P/(stem+'-decisions.json')).exists()
rules=[
 dict(parent='1200',citation='CDIAL[1200]',evidence='CDIAL 1200 āpayati explicitly gives Prakrit āvēi/āvaï comes, Punjabi āuṇā, Jaunsari āṇõ, Nepali āunu, Oriya āibā, Hindi ānā with stem āw-, and Gujarati āvvũ. These selected simple present/infinitive/imperative responses preserve regional vowel, glide and b/v realization and endings. Past stems and light-verb compounds are treated separately because the source explicitly says the preterite is usually from āgata. Local Indo-Aryan transmission is unresolved.'),
 dict(parent='1045',citation='CDIAL[1045]',evidence='CDIAL 1045 āgata arrived explicitly supplies Prakrit āya-/āa-, Nepali āyo and Hindi āyā, and the l-extension Bengali āila and Old Maithili ayalahũ. The selected āy-/āil- finite survey responses match this past-stem family even where the elicitation is glossed simply come. Regional l-endings, vowels and glides are retained without claiming an exact tense analysis for every spelling; no present-stem ancestry is substituted.'),
 dict(parent='1437',citation='CDIAL[1437]',evidence='CDIAL 1437 āviśati explicitly includes Bengali āisā to come. The eastern aśa/aʃa/asa and Hajong/Bishnupriya aha-/aha-ni responses are linked to that regional coming family, with contraction of the vowel sequence and s/h realization recorded as qualifications. This is a family link with inter-Indo-Aryan transmission unresolved, not a claim that the Sanskrit form directly predicts every inflection.')]
sets=[{'āo','āṇā','ānāī','āeb','abe','au','āu','aū','ao','awʌi','ave','avẽ','avinu'},
 {'āyīs','āīs','ailʌs','ail','aila','ailuk','ayla','aiyu','ayao'},
 {'aha','asa','ahani','aśa','aʃa','ās'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[];held=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or 'come' not in r['Gloss'].lower() or 'come down' in r['Gloss'].lower():continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
 else:
  if r['Language_ID']=='poth' and r['Form'] in {'aś','as','ac'}:
   held.append(dict(record=r,families=[],reason='Pothwari aś/as/ac come: full CDIAL 227 atyeti and 1044 āgacchati explicitly discuss competing derivations of northwestern ac-/āś- forms. The survey consonants alone do not choose between those roots, and the issue cannot be resolved as merely unspecified IA borrowing.',passNumber=131))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print(len(acc),len(held))
