import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass133';assert not (P/(stem+'-decisions.json')).exists()
rules=[
 dict(parent='8209',citation='CDIAL[8209]',evidence='CDIAL 8209 pibati drinks explicitly supplies Bshk/Tor pū- from *piu-, Oriya piibā, Maithili piab, Bhojpuri piyal and Hindi pīnā. These simple drinking verbs preserve their inflection and regional vowel shapes. The Bshk pūg response retains its final verbal ending; no liquid-verb compound is stripped. Local Indo-Aryan transmission remains unresolved.'),
 dict(parent='3865',citation='CDIAL[3865];platts1884[868]',evidence='CDIAL 3865 khādati gives the eat/consume verb with Nepali khānu, Assamese khāiba, Bengali khāoyā, Oriya khāibā and Bhojpuri khāil. Platts page 868 explicitly lists drink among the meanings of khānā, supporting the consume-to-drink extension independently of the survey gloss. Selected kha-/ka- responses retain regional deaspiration, vowels and l-endings; direct borrowing from Hindi is not asserted. Water-plus-verb compounds and unclear kam-/duk- responses remain separate.'),
 dict(parent='3865-2',citation='CDIAL[3865.2];platts1884[868]',evidence='CDIAL 3865 subsection 2 khādita explicitly gives Nepali khāyo and Hindi khāyā ate. The Dotyali khāyo drinking responses, including a survey record glossed eat; drink, continue that past-stem branch. Platts 868 independently confirms drink as a meaning of the consume verb. The original survey meaning and transcription are preserved.')]
sets=[{'pū','pūg','pīyab','piuk','pina','pie','pilak','pilio','pilani'},
 {'kʰaila','kʰa','kʰava','kʰai','ka','kai̯li','kaila','kai̯la','kʰai̯la','kayla','kailani','kʰailani','kʰayla'}, {'kʰāyo'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or 'drink' not in r['Gloss'].lower():continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print(len(acc))
