import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass128';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='12962',citation='CDIAL[12962]',evidence='CDIAL 12962 *saṁbhalati attends to explicitly gives Prakrit saṁbhalaï hears/is attentive, Old Gujarati sāṁbhalaï hears and Gujarati sā̃bhaḷvũ attend/listen/hear/obey. Full entries 12961 and 13057 distinguish the similar support/take-care and remember families. These western survey hearing verbs are linked to 12962, with initial s/h, mb/m/b reduction, aspiration, vowel variation and l/ḷ retained as regional qualifications. Short homl-/homel- responses are interpreted alongside the fuller regional hombal-/homboḷ- hearing series, not by gloss alone. Inflected and paired simple stem forms remain unchanged; exact tense segmentation and local Indo-Aryan borrowing paths are unresolved.')]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
selected={'hobḷīyɔ','homboḷio','somoḷ, somoḷyu','somaḷ, soməḷiri','samaḷ, somiḷyu','sāmaḷ, sāmaḷyu','hombolːu','hombalːo','homle','homlyu','hombailo','hombalo','hombiliu','homelːo'}
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] in remaining and r['Form'] in selected and ('hear' in r['Gloss'].lower() or 'listen' in r['Gloss'].lower()):
  q=rules[0];acc.append(dict(record=r,parent=q['parent'],family=0,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print(len(acc))
