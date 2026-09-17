import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass134';assert not (P/(stem+'-decisions.json')).exists()
rules=[
 dict(parent='3865',citation='CDIAL[3865]',evidence='CDIAL 3865 khādati gives Punjabi khāṇā, Nepali khānu, Bengali khāoyā, Oriya khāibā, Bhojpuri khāil and Gujarati khāvũ. The selected simple eating responses retain their infinitive/finite endings, regional deaspiration and vowels. l-extended eastern responses are analyzed with the Bhojpuri khāil family; western eat-plus-take compounds remain separate. Inter-Indo-Aryan transmission is unresolved.'),
 dict(parent='3865-2',citation='CDIAL[3865.2]',evidence='CDIAL 3865 subsection 2 khādita explicitly supplies Nepali khāyo and describes analogical past formations Sindhi khādho, Lahnda/Punjabi khādhā and Gujarati khādhũ, with Old Marathi khādilā. The selected khad-/khaḍ- and Dotyali khāⁱyo eating responses belong to that past-stem family. Survey dental/retroflex notation, aspiration, vowel quantity and regional l-extensions remain unchanged and are qualifications; no exact phonological derivation or direct borrowing route is asserted.'),
 dict(parent='5267-3',citation='CDIAL[5267.3]',evidence='CDIAL 5267 subsection 3 *jimyati/*jimmati explicitly gives Hindi jīmnā and Old Marwari jīmaï eats. Marwari jīmṇa is linked to that long-vowel branch, retaining the retroflex infinitive ending. The entry records a disputed Munda connection at the broader family level; this existing-node link does not settle that deeper history.'),
 dict(parent='5267',citation='CDIAL[5267]',evidence='CDIAL 5267 jemati explicitly includes Punjabi jeuṇā eat/dine, Bhojpuri jẽwal and Hindi jewnā. Hadothi jeo eat matches the jeu-/jew- regional family; its imperative ending and vowel sequence are retained. The deeper Munda proposal is disputed in the primary entry, and no direct Indo-Aryan borrowing route is asserted.')]
sets=[{'kʰāṇa','kʰailak','kʰelio','kʰenāī','kaila','kaiba','kʰauk','ka','kayla','kʰai̯la','kai̯la','khawa','khavana','kau','kaube','kai'},
 {'kʰāⁱyo','khadalo','khaḍulo','khadalu','khādo','khaḍu','khaḍo','kaḍo','keḍo','khaḍalo','khaḍõ','khaḍalu'}, {'jīmṇa'}, {'jeo'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss'].lower() not in {'eat','to eat','eat!, he ate','eat!','eat/he ate','(you) eat'}:continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print(len(acc))
